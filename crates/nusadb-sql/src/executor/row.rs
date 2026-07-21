//! Tuple codec: schema-aware encoding of a [`Row`] to/from the opaque byte
//! tuples the [`StorageEngine`](nusadb_core::StorageEngine) treaty stores.
//!
//! Per-column layout: one tag byte (`0` = NULL, `1` = present), then if
//! present, the value bytes (little-endian):
//!
//! | [`ColumnType`]      | Bytes after the present-tag                      |
//! |---------------------|---------------------------------------------------|
//! | `Bool`              | 1 byte (`0` / `1`)                                |
//! | `Int` / `Timestamp` | 8 bytes (`i64::to_le_bytes`)                      |
//! | `Float`             | 8 bytes (`f64::to_le_bytes`)                      |
//! | `Text` / `Bytes`    | 4-byte `u32` LE length, then that many raw bytes  |
//!
//! `Bytes` and `Timestamp` columns are catalog-creatable but have no SQL
//! literal syntax to populate them today — the parser only produces
//! `Bool/Int/Float/Text/Null` literals — so non-`NULL` values for those
//! types are rejected here as `Error::TypeMismatch`.

// The `i64 as f64` widening below is the analyzer-sanctioned coercion when
// an integer literal lands in a float column (`check_assignable`).
#![allow(
    clippy::cast_precision_loss,
    reason = "intentional Int->Float widening per analyzer's assignment rule"
)]

use nusadb_core::ColumnType;
use nusadb_core::engine::ArrayElem;

use crate::ast;
use crate::error::Error;

/// A row of values, one entry per column of its source table's schema.
pub type Row = Vec<ast::Value>;

const TAG_NULL: u8 = 0;
const TAG_PRESENT: u8 = 1;

/// Encode `row` into the opaque byte form the storage engine accepts.
///
/// `schema` is the source table's column types in declaration order; `row`
/// must have the same length.
pub(crate) fn encode(row: &[ast::Value], schema: &[ColumnType]) -> Result<Vec<u8>, Error> {
    if row.len() != schema.len() {
        return Err(Error::ArityMismatch {
            context: "tuple encode".to_owned(),
            expected: schema.len(),
            found: row.len(),
        });
    }
    let mut out = Vec::with_capacity(row.len() * 9);
    for (value, ty) in row.iter().zip(schema) {
        if matches!(value, ast::Value::Null) {
            out.push(TAG_NULL);
        } else {
            out.push(TAG_PRESENT);
            encode_value(value, *ty, &mut out)?;
        }
    }
    Ok(out)
}

/// Decode a stored tuple back into a [`Row`].
pub(crate) fn decode(bytes: &[u8], schema: &[ColumnType]) -> Result<Row, Error> {
    let mut row = Vec::with_capacity(schema.len());
    let mut pos = 0;
    for &ty in schema {
        let tag = *bytes
            .get(pos)
            .ok_or(Error::MalformedTuple { offset: pos })?;
        pos += 1;
        match tag {
            TAG_NULL => row.push(ast::Value::Null),
            TAG_PRESENT => {
                let (value, next) = decode_value(bytes, pos, ty)?;
                row.push(value);
                pos = next;
            },
            _ => return Err(Error::MalformedTuple { offset: pos - 1 }),
        }
    }
    Ok(row)
}

/// Decode a stored tuple into a *narrowed* [`Row`] holding only the columns in
/// `keep` — the projection-pushdown read path.
///
/// `schema` is the full table's column types in declaration order; `keep` lists
/// the source ordinals to materialize, **ascending**, and the returned row holds
/// exactly those values in that order. The tuple encoding is positional, so the
/// cursor is still advanced past every column, but a dropped column is *skipped*
/// (see [`skip_value`]) rather than decoded — so a dropped blob never allocates
/// the `String`/`Vec` a full decode would build and immediately discard.
pub(crate) fn decode_projected(
    bytes: &[u8],
    schema: &[ColumnType],
    keep: &[usize],
) -> Result<Row, Error> {
    let mut row = Vec::with_capacity(keep.len());
    let mut pos = 0;
    let mut want = keep.iter().peekable();
    for (idx, &ty) in schema.iter().enumerate() {
        let tag = *bytes
            .get(pos)
            .ok_or(Error::MalformedTuple { offset: pos })?;
        pos += 1;
        let wanted = want.peek().is_some_and(|&&w| w == idx);
        match tag {
            TAG_NULL => {
                if wanted {
                    row.push(ast::Value::Null);
                    want.next();
                }
            },
            // Materialize a kept column; for a dropped one, advance the cursor past it **without**
            // building the value (skipping the String/Vec allocation of a discarded blob), landing
            // at exactly the offset a full decode would.
            TAG_PRESENT => {
                if wanted {
                    let (value, next) = decode_value(bytes, pos, ty)?;
                    row.push(value);
                    pos = next;
                    want.next();
                } else {
                    pos = skip_value(bytes, pos, ty)?;
                }
            },
            _ => return Err(Error::MalformedTuple { offset: pos - 1 }),
        }
    }
    Ok(row)
}

/// Advance past one present (`TAG_PRESENT`) stored value of type `ty` at `pos`, returning the next
/// position, **without materializing the value** — the projected-read skip for a dropped column.
///
/// For the heap-allocating length-prefixed blob types (`TEXT`/`VARCHAR`/`CHAR`, `BYTEA`,
/// `JSON`/`JSONB`) it bounds- (and, for text, UTF-8-) validates and advances exactly as
/// [`decode_value`] does, but never owns the discarded `String`/`Vec`. Every other type — stack-only
/// scalars and the structured `ARRAY`/`VECTOR` — delegates to [`decode_value`] and discards the
/// value, which advances the cursor and raises any malformed-tuple error identically to a full
/// decode. So the returned offset (and any error) matches [`decode_value`] for every type.
pub(crate) fn skip_value(bytes: &[u8], pos: usize, ty: ColumnType) -> Result<usize, Error> {
    match ty {
        // TEXT/VARCHAR/CHAR and JSON/JSONB share the `[u32 len][len UTF-8 bytes]` layout;
        // `read_text_field` bounds- and UTF-8-validates and returns the end offset without owning
        // the string (the `.to_owned()` a full decode pays is exactly what a dropped column skips).
        ColumnType::Text
        | ColumnType::VarChar(_)
        | ColumnType::Char(_)
        | ColumnType::Json
        | ColumnType::Jsonb => Ok(read_text_field(bytes, pos)?.1),
        // BYTEA: `[u32 len][len raw bytes]`, no UTF-8 validation (matching `decode_value`).
        ColumnType::Bytes => {
            let len = u32::from_le_bytes(read_array::<4>(bytes, pos)?) as usize;
            let start = pos + 4;
            let end = start
                .checked_add(len)
                .ok_or(Error::MalformedTuple { offset: start })?;
            bytes
                .get(start..end)
                .ok_or(Error::MalformedTuple { offset: start })?;
            Ok(end)
        },
        // Stack-only scalars (Bool/Int/Float/temporal/Uuid/Numeric/Interval) allocate nothing, and
        // the structured Array/Vector layout is not a simple length prefix — decode and discard,
        // which advances the cursor identically.
        _ => Ok(decode_value(bytes, pos, ty)?.1),
    }
}

/// Canonical storage form of a `CHAR(n)`/bpchar text value: trailing U+0020 spaces are insignificant
/// and stripped (a trailing tab stays — only spaces pad). Returns `None` (encode the value as-is) for
/// any non-`CHAR` column or non-text value, so `VARCHAR`/`TEXT` keep their trailing blanks.
fn bpchar_canonical(value: &ast::Value, ty: ColumnType) -> Option<ast::Value> {
    match (ty, value) {
        (ColumnType::Char(_), ast::Value::Text(s)) => {
            // Only allocate when there are trailing blanks to strip — the common case (no padding)
            // returns `None` and the original value is encoded in place.
            let trimmed = s.trim_end_matches(' ');
            (trimmed.len() != s.len()).then(|| ast::Value::Text(trimmed.to_owned()))
        },
        _ => None,
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-(value, type) encoding dispatch; length tracks the type set"
)]
fn encode_value(value: &ast::Value, ty: ColumnType, out: &mut Vec<u8>) -> Result<(), Error> {
    // A `CHAR(n)`/bpchar value is stored in canonical form (trailing blanks stripped) so `length()`
    // and equality on a CHAR column follow the blank-padded semantics; `VARCHAR(n)` keeps its blanks.
    let trimmed_char = bpchar_canonical(value, ty);
    let value = trimmed_char.as_ref().unwrap_or(value);
    // Encode against the physical type: `VARCHAR(n)`/`CHAR(n)` values are written exactly like
    // `TEXT` (the declared length is enforced by the desugared `CHECK`, not the encoding).
    let ty = ty.physical();
    match (value, ty) {
        (ast::Value::Bool(b), ColumnType::Bool) => out.push(u8::from(*b)),
        (ast::Value::Int(i), ColumnType::Int) => out.extend_from_slice(&i.to_le_bytes()),
        (ast::Value::Int(i), ColumnType::Float) => {
            out.extend_from_slice(&(*i as f64).to_le_bytes());
        },
        (ast::Value::Float(f), ColumnType::Float) => out.extend_from_slice(&f.to_le_bytes()),
        // A NUMERIC value (e.g. a plain decimal literal) widens into a FLOAT column.
        (ast::Value::Numeric(d), ColumnType::Float) => {
            out.extend_from_slice(&d.to_f64().to_le_bytes());
        },
        (ast::Value::Text(s), ColumnType::Text) => {
            let len = u32::try_from(s.len())
                .map_err(|_| Error::Unsupported("text value larger than 4 GiB".to_owned()))?;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        },
        // Temporal + UUID: fixed-width integer / byte encodings.
        (ast::Value::Date(d), ColumnType::Date) => out.extend_from_slice(&d.to_le_bytes()),
        (ast::Value::Time(t), ColumnType::Time)
        | (ast::Value::TimeTz(t), ColumnType::TimeTz)
        | (ast::Value::Timestamp(t), ColumnType::Timestamp)
        | (ast::Value::TimestampTz(t), ColumnType::TimestampTz) => {
            out.extend_from_slice(&t.to_le_bytes());
        },
        (ast::Value::Uuid(u), ColumnType::Uuid) => out.extend_from_slice(u),
        // Implicit string coercion ("unknown literal" rule): a text value assigned to a
        // temporal/UUID column is parsed into the column's type at encode time.
        (ast::Value::Text(s), ColumnType::Date) => {
            let d = crate::temporal::parse_date(s).ok_or_else(|| invalid(ty, s))?;
            out.extend_from_slice(&d.to_le_bytes());
        },
        (ast::Value::Text(s), ColumnType::Time) => {
            let t = crate::temporal::parse_time(s).ok_or_else(|| invalid(ty, s))?;
            out.extend_from_slice(&t.to_le_bytes());
        },
        (ast::Value::Text(s), ColumnType::TimeTz) => {
            let t = crate::temporal::parse_timetz(s).ok_or_else(|| invalid(ty, s))?;
            out.extend_from_slice(&t.to_le_bytes());
        },
        (ast::Value::Text(s), ColumnType::Timestamp) => {
            let t = crate::temporal::parse_timestamp(s).ok_or_else(|| invalid(ty, s))?;
            out.extend_from_slice(&t.to_le_bytes());
        },
        (ast::Value::Text(s), ColumnType::TimestampTz) => {
            let t = crate::temporal::parse_timestamptz(s).ok_or_else(|| invalid(ty, s))?;
            out.extend_from_slice(&t.to_le_bytes());
        },
        (ast::Value::Text(s), ColumnType::Uuid) => {
            let u = crate::temporal::parse_uuid(s).ok_or_else(|| invalid(ty, s))?;
            out.extend_from_slice(&u);
        },
        // NUMERIC: the value (or a coercible Int/Float/Text) rescaled to the column's
        // declared scale, precision-checked, then mantissa(16) + scale(1).
        (ast::Value::Numeric(d), ColumnType::Numeric { precision, scale }) => {
            encode_numeric(*d, precision, scale, out)?;
        },
        (ast::Value::Int(i), ColumnType::Numeric { precision, scale }) => {
            encode_numeric(crate::numeric::Decimal::from_i64(*i), precision, scale, out)?;
        },
        (ast::Value::Float(f), ColumnType::Numeric { precision, scale }) => {
            let d = crate::numeric::from_f64_text(*f).ok_or_else(|| invalid(ty, &f.to_string()))?;
            encode_numeric(d, precision, scale, out)?;
        },
        (ast::Value::Text(s), ColumnType::Numeric { precision, scale }) => {
            let d = crate::numeric::Decimal::parse(s).ok_or_else(|| invalid(ty, s))?;
            encode_numeric(d, precision, scale, out)?;
        },
        // JSON: length-prefixed canonical text. A `Value::Json` is already canonical; a text
        // value is parsed + canonicalized (and rejected if it is not valid JSON).
        (ast::Value::Json(s), ColumnType::Json) => put_json_text(s, out)?,
        (ast::Value::Text(s), ColumnType::Json) => {
            let canon = crate::json::canonicalize(s).ok_or_else(|| invalid(ty, s))?;
            put_json_text(&canon, out)?;
        },
        // INTERVAL: months(4) + days(4) + micros(8). A text value is parsed.
        (ast::Value::Interval(iv), ColumnType::Interval) => put_interval(*iv, out),
        (ast::Value::Text(s), ColumnType::Interval) => {
            let iv = crate::interval::Interval::parse(s).ok_or_else(|| invalid(ty, s))?;
            put_interval(iv, out);
        },
        // ARRAY: an ARRAY[...] value, or a `{a,b,c}` array text literal.
        (ast::Value::Array(items), ColumnType::Array(elem)) => {
            encode_array(items, elem.column_type(), out)?;
        },
        (ast::Value::Text(s), ColumnType::Array(elem)) => {
            // Parse `{a,b,c}` + coerce each token to the element type via the shared helper.
            let elem_ty = elem.column_type();
            let items = super::eval::parse_text_array(s, elem_ty)?;
            encode_array(&items, elem_ty, out)?;
        },
        // VECTOR: `[count u32]` then each component as a little-endian f32. A `[..]` text
        // literal is parsed first. Both forms are checked against the column's declared dimension.
        (ast::Value::Vector(v), ColumnType::Vector(dim)) => encode_vector(v, dim, out)?,
        (ast::Value::Text(s), ColumnType::Vector(dim)) => {
            let v = crate::vector::parse(s).ok_or_else(|| invalid(ty, s))?;
            encode_vector(&v, dim, out)?;
        },
        // BYTEA: length-prefixed raw bytes. A text value is parsed as the `\x<hex>` form.
        (ast::Value::Bytes(b), ColumnType::Bytes) => put_bytes(b, out)?,
        (ast::Value::Text(s), ColumnType::Bytes) => {
            let b = crate::executor::eval::parse_bytea(s).ok_or_else(|| invalid(ty, s))?;
            put_bytes(&b, out)?;
        },
        _ => {
            return Err(Error::TypeMismatch {
                context: "tuple encode".to_owned(),
                expected: ty,
                found: runtime_type_of(value),
            });
        },
    }
    Ok(())
}

/// Encode an array: `[count u32]` then each element as `[tag(1)]` + (if present) its scalar
/// encoding. Each element is coerced to `elem_ty` via [`encode_value`].
fn encode_array(items: &[ast::Value], elem_ty: ColumnType, out: &mut Vec<u8>) -> Result<(), Error> {
    let count = u32::try_from(items.len())
        .map_err(|_| Error::Unsupported("array longer than 4 G elements".to_owned()))?;
    out.extend_from_slice(&count.to_le_bytes());
    for item in items {
        if matches!(item, ast::Value::Null) {
            out.push(TAG_NULL);
        } else {
            out.push(TAG_PRESENT);
            encode_value(item, elem_ty, out)?;
        }
    }
    Ok(())
}

/// Parse an array text literal `{a,b,c}` into text elements (the bare token `NULL` →
/// `Value::Null`). Each element is coerced to the column's element type at encode time. A
/// double-quoted element (`{"a,b","x\"y"}`) is taken literally with `\` escapes removed and is always
/// text (a quoted `"null"` is the string, not the null token) — the inverse of [`crate::display`]'s
/// array rendering, so the form round-trips. Nested/multidimensional forms are not supported here.
pub(crate) fn parse_array_text(s: &str) -> Option<Vec<ast::Value>> {
    let inner = s.trim().strip_prefix('{')?.strip_suffix('}')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    let mut elems = Vec::new();
    let mut chars = inner.chars().peekable();
    loop {
        skip_array_spaces(&mut chars);
        elems.push(parse_one_array_element(&mut chars)?);
        skip_array_spaces(&mut chars);
        match chars.next() {
            // A comma separates elements; the loop continues to the next one.
            Some(',') => {},
            None => break,
            // A stray character after an element (e.g. a closing quote mid-token) is malformed.
            Some(_) => return None,
        }
    }
    Some(elems)
}

/// Consume any run of whitespace between array elements and delimiters.
fn skip_array_spaces(chars: &mut std::iter::Peekable<std::str::Chars>) {
    while matches!(chars.peek(), Some(c) if c.is_whitespace()) {
        chars.next();
    }
}

/// Parse one array element, starting at its first non-space character. A `"`-quoted element reads to
/// the matching unescaped quote (`\` escapes the next character) and is always text; an unquoted run
/// reads up to the next `,`, trims, and maps the bare token `NULL` to [`ast::Value::Null`]. Returns
/// `None` on an unterminated quote.
fn parse_one_array_element(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<ast::Value> {
    if chars.peek() == Some(&'"') {
        chars.next();
        let mut buf = String::new();
        loop {
            match chars.next()? {
                '\\' => buf.push(chars.next()?),
                '"' => break,
                c => buf.push(c),
            }
        }
        Some(ast::Value::Text(buf))
    } else {
        let mut buf = String::new();
        while let Some(&c) = chars.peek() {
            if c == ',' {
                break;
            }
            buf.push(c);
            chars.next();
        }
        let token = buf.trim();
        if token.eq_ignore_ascii_case("null") {
            Some(ast::Value::Null)
        } else {
            Some(ast::Value::Text(token.to_owned()))
        }
    }
}

/// Encode a vector: `[count u32]` then each component as a little-endian `f32`. The number
/// of components must equal the column's declared dimension `dim` (a mismatch is a typed error).
fn encode_vector(v: &[f32], dim: u32, out: &mut Vec<u8>) -> Result<(), Error> {
    if v.len() as u64 != u64::from(dim) {
        return Err(Error::InvalidValue {
            ty: ColumnType::Vector(dim),
            value: format!("{}-dimensional vector", v.len()),
        });
    }
    out.extend_from_slice(&dim.to_le_bytes());
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    Ok(())
}

/// Write an interval as months(4) + days(4) + micros(8).
fn put_interval(iv: crate::interval::Interval, out: &mut Vec<u8>) {
    out.extend_from_slice(&iv.months.to_le_bytes());
    out.extend_from_slice(&iv.days.to_le_bytes());
    out.extend_from_slice(&iv.micros.to_le_bytes());
}

/// Write a JSON document as length-prefixed text.
fn put_json_text(s: &str, out: &mut Vec<u8>) -> Result<(), Error> {
    let len = u32::try_from(s.len())
        .map_err(|_| Error::Unsupported("JSON value larger than 4 GiB".to_owned()))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

/// Encode a `BYTEA` value: a `u32` length prefix then the raw bytes.
fn put_bytes(b: &[u8], out: &mut Vec<u8>) -> Result<(), Error> {
    let len = u32::try_from(b.len())
        .map_err(|_| Error::Unsupported("BYTEA value larger than 4 GiB".to_owned()))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(b);
    Ok(())
}

/// Build an `InvalidValue` error for a string literal that does not parse as `ty`.
fn invalid(ty: ColumnType, value: &str) -> Error {
    Error::InvalidValue {
        ty,
        value: value.to_owned(),
    }
}

/// Encode a `NUMERIC` value: rescale to the column's declared `scale` when constrained
/// (`precision > 0`), reject values exceeding the declared `precision`, then write the 16-byte
/// mantissa + 1-byte scale. An unconstrained column (`precision == 0`) keeps the value's own scale.
fn encode_numeric(
    value: crate::numeric::Decimal,
    precision: u8,
    scale: u8,
    out: &mut Vec<u8>,
) -> Result<(), Error> {
    let ty = ColumnType::Numeric { precision, scale };
    let d = if precision == 0 {
        value
    } else {
        let rescaled = value
            .rescale(scale)
            .ok_or_else(|| invalid(ty, &value.format()))?;
        if rescaled.required_precision() > u32::from(precision) {
            return Err(invalid(ty, &value.format()));
        }
        rescaled
    };
    out.extend_from_slice(&d.mantissa.to_le_bytes());
    out.push(d.scale);
    Ok(())
}

/// Read one field's NULL/PRESENT tag: `(present, payload position)` — the tag step the
/// vectorized typed builders walk tuples with (R2 stage 2). Same semantics as the inline tag
/// handling in [`decode`]/[`decode_projected`], including the error offset for an unknown tag.
#[inline]
pub(crate) fn field_tag(bytes: &[u8], pos: usize) -> Result<(bool, usize), Error> {
    let tag = *bytes
        .get(pos)
        .ok_or(Error::MalformedTuple { offset: pos })?;
    match tag {
        TAG_NULL => Ok((false, pos + 1)),
        TAG_PRESENT => Ok((true, pos + 1)),
        _ => Err(Error::MalformedTuple { offset: pos }),
    }
}

/// The shared leaf readers (R2 stage 2): [`decode_value`] and the vectorized typed builders both
/// parse fixed-width payloads through these, so the row and columnar paths read identical bytes
/// by construction. Each returns `(value, next position)`.
#[inline]
pub(crate) fn read_bool_field(bytes: &[u8], pos: usize) -> Result<(bool, usize), Error> {
    let b = *bytes
        .get(pos)
        .ok_or(Error::MalformedTuple { offset: pos })?;
    Ok((b != 0, pos + 1))
}

/// See [`read_bool_field`].
#[inline]
pub(crate) fn read_i64_field(bytes: &[u8], pos: usize) -> Result<(i64, usize), Error> {
    let arr = read_array::<8>(bytes, pos)?;
    Ok((i64::from_le_bytes(arr), pos + 8))
}

/// See [`read_bool_field`].
#[inline]
pub(crate) fn read_f64_field(bytes: &[u8], pos: usize) -> Result<(f64, usize), Error> {
    let arr = read_array::<8>(bytes, pos)?;
    Ok((f64::from_le_bytes(arr), pos + 8))
}

/// Text leaf reader (R2 stage 2b): a length-prefixed, UTF-8-validated **borrowed** `&str` and
/// the next position — the caller decides whether to own it ([`decode_value`] builds a `String`;
/// the vectorized text builder appends the bytes straight into its offsets+data buffers).
#[inline]
pub(crate) fn read_text_field(bytes: &[u8], pos: usize) -> Result<(&str, usize), Error> {
    let arr = read_array::<4>(bytes, pos)?;
    let len = u32::from_le_bytes(arr) as usize;
    let start = pos + 4;
    let end = start
        .checked_add(len)
        .ok_or(Error::MalformedTuple { offset: start })?;
    let slice = bytes
        .get(start..end)
        .ok_or(Error::MalformedTuple { offset: start })?;
    let text = std::str::from_utf8(slice).map_err(|_| Error::MalformedTuple { offset: start })?;
    Ok((text, end))
}

/// Decode one PRESENT field's payload at `pos` (the byte after its tag) — the entry the
/// vectorized builders' fallback uses for non-fixed-width types (R2 stage 2).
#[inline]
pub(crate) fn decode_present_value(
    bytes: &[u8],
    pos: usize,
    ty: ColumnType,
) -> Result<(ast::Value, usize), Error> {
    decode_value(bytes, pos, ty)
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per ColumnType; flatter than dispatching to per-type helpers"
)]
fn decode_value(bytes: &[u8], pos: usize, ty: ColumnType) -> Result<(ast::Value, usize), Error> {
    match ty {
        ColumnType::Bool => {
            let (b, next) = read_bool_field(bytes, pos)?;
            Ok((ast::Value::Bool(b), next))
        },
        // SMALLINT/BIGINT are stored as the same 64-bit integer as INT (encode_value normalized via
        // `physical()`), so they decode identically.
        ColumnType::Int | ColumnType::SmallInt | ColumnType::BigInt => {
            let (v, next) = read_i64_field(bytes, pos)?;
            Ok((ast::Value::Int(v), next))
        },
        // REAL decodes identically to FLOAT (stored as the same 64-bit double).
        ColumnType::Float | ColumnType::Real => {
            let (v, next) = read_f64_field(bytes, pos)?;
            Ok((ast::Value::Float(v), next))
        },
        // Temporal + UUID.
        ColumnType::Date => {
            let arr = read_array::<4>(bytes, pos)?;
            Ok((ast::Value::Date(i32::from_le_bytes(arr)), pos + 4))
        },
        ColumnType::Time => {
            let arr = read_array::<8>(bytes, pos)?;
            Ok((ast::Value::Time(i64::from_le_bytes(arr)), pos + 8))
        },
        ColumnType::Timestamp => {
            let arr = read_array::<8>(bytes, pos)?;
            Ok((ast::Value::Timestamp(i64::from_le_bytes(arr)), pos + 8))
        },
        ColumnType::TimestampTz => {
            let arr = read_array::<8>(bytes, pos)?;
            Ok((ast::Value::TimestampTz(i64::from_le_bytes(arr)), pos + 8))
        },
        ColumnType::TimeTz => {
            let arr = read_array::<8>(bytes, pos)?;
            Ok((ast::Value::TimeTz(i64::from_le_bytes(arr)), pos + 8))
        },
        ColumnType::Uuid => {
            let arr = read_array::<16>(bytes, pos)?;
            Ok((ast::Value::Uuid(arr), pos + 16))
        },
        // JSON: length-prefixed canonical text.
        // JSONB decodes identically to JSON (stored as the same canonical text).
        ColumnType::Json | ColumnType::Jsonb => {
            let arr = read_array::<4>(bytes, pos)?;
            let len = u32::from_le_bytes(arr) as usize;
            let start = pos + 4;
            let end = start
                .checked_add(len)
                .ok_or(Error::MalformedTuple { offset: start })?;
            let slice = bytes
                .get(start..end)
                .ok_or(Error::MalformedTuple { offset: start })?;
            let text = std::str::from_utf8(slice)
                .map_err(|_| Error::MalformedTuple { offset: start })?
                .to_owned();
            Ok((ast::Value::Json(text), end))
        },
        // NUMERIC: 16-byte i128 mantissa + 1-byte scale.
        ColumnType::Numeric { .. } => {
            let mant = i128::from_le_bytes(read_array::<16>(bytes, pos)?);
            let scale = *bytes
                .get(pos + 16)
                .ok_or(Error::MalformedTuple { offset: pos + 16 })?;
            // Reject a corrupt/out-of-range scale rather than building an unrepresentable Decimal
            // (the codec/arithmetic only carry up to MAX_SCALE fractional digits)
            if scale > crate::numeric::MAX_SCALE {
                return Err(Error::MalformedTuple { offset: pos + 16 });
            }
            Ok((
                ast::Value::Numeric(crate::numeric::Decimal {
                    mantissa: mant,
                    scale,
                }),
                pos + 17,
            ))
        },
        // INTERVAL: months(i32) + days(i32) + micros(i64) = 16 bytes.
        ColumnType::Interval => {
            let months = i32::from_le_bytes(read_array::<4>(bytes, pos)?);
            let days = i32::from_le_bytes(read_array::<4>(bytes, pos + 4)?);
            let micros = i64::from_le_bytes(read_array::<8>(bytes, pos + 8)?);
            Ok((
                ast::Value::Interval(crate::interval::Interval {
                    months,
                    days,
                    micros,
                }),
                pos + 16,
            ))
        },
        // ARRAY: [count u32] then each element as [tag(1)] + (if present) its scalar encoding.
        ColumnType::Array(elem) => {
            let count = u32::from_le_bytes(read_array::<4>(bytes, pos)?) as usize;
            let mut p = pos + 4;
            // Each element is at least its 1-byte present/NULL tag, so a `count` larger than the
            // bytes that remain is corrupt — reject up front rather than spinning the loop.
            if count > bytes.len().saturating_sub(p) {
                return Err(Error::MalformedTuple { offset: pos });
            }
            let mut items = Vec::with_capacity(count.min(1024));
            let elem_ty = elem.column_type();
            for _ in 0..count {
                let tag = *bytes.get(p).ok_or(Error::MalformedTuple { offset: p })?;
                p += 1;
                match tag {
                    TAG_NULL => items.push(ast::Value::Null),
                    TAG_PRESENT => {
                        let (v, next) = decode_value(bytes, p, elem_ty)?;
                        items.push(v);
                        p = next;
                    },
                    // An unknown element tag is corruption (matches the top-level decode).
                    _ => return Err(Error::MalformedTuple { offset: p - 1 }),
                }
            }
            Ok((ast::Value::Array(items), p))
        },
        // VECTOR: [count u32] then count little-endian f32 components. The stored count must
        // equal the column's declared dimension; anything else is corruption.
        ColumnType::Vector(dim) => {
            let count = u32::from_le_bytes(read_array::<4>(bytes, pos)?);
            if count != dim {
                return Err(Error::MalformedTuple { offset: pos });
            }
            let mut p = pos + 4;
            let mut v = Vec::with_capacity(count as usize);
            for _ in 0..count {
                v.push(f32::from_le_bytes(read_array::<4>(bytes, p)?));
                p += 4;
            }
            Ok((ast::Value::Vector(v), p))
        },
        // VARCHAR/CHAR values are encoded identically to TEXT (`ColumnType::physical`).
        ColumnType::Text | ColumnType::VarChar(_) | ColumnType::Char(_) => {
            let (text, end) = read_text_field(bytes, pos)?;
            Ok((ast::Value::Text(text.to_owned()), end))
        },
        // BYTEA: a u32 length prefix then the raw bytes (mirrors the TEXT layout).
        ColumnType::Bytes => {
            let arr = read_array::<4>(bytes, pos)?;
            let len = u32::from_le_bytes(arr) as usize;
            let start = pos + 4;
            let end = start
                .checked_add(len)
                .ok_or(Error::MalformedTuple { offset: start })?;
            let slice = bytes
                .get(start..end)
                .ok_or(Error::MalformedTuple { offset: start })?;
            Ok((ast::Value::Bytes(slice.to_vec()), end))
        },
    }
}

fn read_array<const N: usize>(bytes: &[u8], pos: usize) -> Result<[u8; N], Error> {
    let slice = bytes
        .get(pos..pos + N)
        .ok_or(Error::MalformedTuple { offset: pos })?;
    slice
        .try_into()
        .map_err(|_| Error::MalformedTuple { offset: pos })
}

pub(crate) fn runtime_type_of(value: &ast::Value) -> ColumnType {
    match value {
        ast::Value::Null | ast::Value::Bool(_) => ColumnType::Bool,
        ast::Value::Int(_) => ColumnType::Int,
        ast::Value::Float(_) => ColumnType::Float,
        ast::Value::Text(_) => ColumnType::Text,
        ast::Value::Date(_) => ColumnType::Date,
        ast::Value::Time(_) => ColumnType::Time,
        ast::Value::Timestamp(_) => ColumnType::Timestamp,
        ast::Value::TimestampTz(_) => ColumnType::TimestampTz,
        ast::Value::TimeTz(_) => ColumnType::TimeTz,
        ast::Value::Uuid(_) => ColumnType::Uuid,
        ast::Value::Numeric(_) => ColumnType::Numeric {
            precision: 0,
            scale: 0,
        },
        ast::Value::Json(_) => ColumnType::Json,
        ast::Value::Interval(_) => ColumnType::Interval,
        ast::Value::Array(items) => ColumnType::Array(array_elem_of(items)),
        #[allow(
            clippy::cast_possible_truncation,
            reason = "vector dim fits u32 by construction"
        )]
        ast::Value::Vector(v) => ColumnType::Vector(v.len() as u32),
        ast::Value::Bytes(_) => ColumnType::Bytes,
    }
}

/// Infer an array's element type from its first non-NULL element, defaulting to `Int` for an empty
/// or all-NULL array. Best-effort inference for already-validated/constructed arrays (and
/// error messages); the analyzer uses [`array_elem_checked`] to reject heterogeneous literals.
pub(crate) fn array_elem_of(items: &[ast::Value]) -> ArrayElem {
    items
        .iter()
        .find(|v| !matches!(v, ast::Value::Null))
        .and_then(|v| ArrayElem::from_column_type(runtime_type_of(v)))
        .unwrap_or(ArrayElem::Int)
}

/// Element type of an array literal, requiring every non-NULL element to share one runtime type:
/// a heterogeneous `{1, 'a'}` is rejected here at analysis time rather than silently
/// inferring the element type from the first element (and only failing later, per-element, at
/// encode). An empty or all-NULL array defaults to `Int`, matching [`array_elem_of`].
pub(crate) fn array_elem_checked(items: &[ast::Value]) -> Result<ArrayElem, Error> {
    let mut elem: Option<ColumnType> = None;
    for v in items {
        if matches!(v, ast::Value::Null) {
            continue;
        }
        let ty = runtime_type_of(v);
        match elem {
            None => elem = Some(ty),
            Some(prev) if prev != ty => {
                return Err(Error::TypeMismatch {
                    context: "array literal elements must share one type".to_owned(),
                    expected: prev,
                    found: ty,
                });
            },
            Some(_) => {},
        }
    }
    Ok(elem
        .and_then(ArrayElem::from_column_type)
        .unwrap_or(ArrayElem::Int))
}

#[cfg(test)]
mod tests {
    use super::{array_elem_checked, decode, encode};
    use crate::ast::Value;
    use crate::error::Error;
    use nusadb_core::ColumnType;
    use nusadb_core::engine::ArrayElem;

    #[test]
    fn array_elem_checked_requires_homogeneous_elements() {
        // Homogeneous arrays infer the element type; NULLs are skipped.
        assert_eq!(
            array_elem_checked(&[Value::Int(1), Value::Null, Value::Int(2)]).unwrap(),
            ArrayElem::Int
        );
        assert_eq!(
            array_elem_checked(&[Value::Text("a".to_owned())]).unwrap(),
            ArrayElem::Text
        );
        // Empty / all-NULL default to Int (matching array_elem_of).
        assert_eq!(array_elem_checked(&[]).unwrap(), ArrayElem::Int);
        assert_eq!(array_elem_checked(&[Value::Null]).unwrap(), ArrayElem::Int);
        // A heterogeneous array is rejected rather than inferring from the first element.
        assert!(matches!(
            array_elem_checked(&[Value::Int(1), Value::Text("a".to_owned())]),
            Err(Error::TypeMismatch { .. })
        ));
    }

    fn schema() -> Vec<ColumnType> {
        vec![
            ColumnType::Int,
            ColumnType::Text,
            ColumnType::Bool,
            ColumnType::Float,
        ]
    }

    #[test]
    fn roundtrip_full_row() {
        let row = vec![
            Value::Int(42),
            Value::Text("hello".to_owned()),
            Value::Bool(true),
            Value::Float(3.5),
        ];
        let bytes = encode(&row, &schema()).unwrap();
        let back = decode(&bytes, &schema()).unwrap();
        assert_eq!(back, row);
    }

    #[test]
    fn roundtrip_with_nulls() {
        let row = vec![Value::Int(1), Value::Null, Value::Null, Value::Float(0.0)];
        let bytes = encode(&row, &schema()).unwrap();
        let back = decode(&bytes, &schema()).unwrap();
        assert_eq!(back, row);
    }

    #[test]
    fn decode_rejects_numeric_scale_past_max() {
        // A corrupt scale byte beyond MAX_SCALE must error, not build an unrepresentable
        // Decimal from untrusted bytes.
        let ty = ColumnType::Numeric {
            precision: 0,
            scale: 0,
        };
        let mut bad = vec![1u8]; // present tag
        bad.extend_from_slice(&0i128.to_le_bytes()); // mantissa
        bad.push(crate::numeric::MAX_SCALE + 1); // scale out of range
        assert!(matches!(
            decode(&bad, &[ty]),
            Err(Error::MalformedTuple { .. })
        ));
        // A scale exactly at the limit still decodes.
        let mut ok = vec![1u8];
        ok.extend_from_slice(&123i128.to_le_bytes());
        ok.push(crate::numeric::MAX_SCALE);
        assert!(decode(&ok, &[ty]).is_ok());
    }

    #[test]
    fn int_widens_into_float_column() {
        let row = vec![Value::Int(7)];
        let bytes = encode(&row, &[ColumnType::Float]).unwrap();
        let back = decode(&bytes, &[ColumnType::Float]).unwrap();
        assert_eq!(back, vec![Value::Float(7.0)]);
    }

    #[test]
    fn arity_mismatch_is_rejected() {
        assert!(matches!(
            encode(&[Value::Int(1)], &schema()),
            Err(Error::ArityMismatch { .. }),
        ));
    }

    #[test]
    fn type_mismatch_is_rejected() {
        // Text into a Bool column.
        let result = encode(&[Value::Text("x".to_owned())], &[ColumnType::Bool]);
        assert!(matches!(result, Err(Error::TypeMismatch { .. })));
    }

    #[test]
    fn truncated_tuple_is_malformed() {
        let bytes = encode(&[Value::Int(99)], &[ColumnType::Int]).unwrap();
        // Drop the last byte of the i64 payload.
        let result = decode(&bytes[..bytes.len() - 1], &[ColumnType::Int]);
        assert!(matches!(result, Err(Error::MalformedTuple { .. })));
    }

    #[test]
    fn roundtrip_temporal_and_uuid() {
        let schema = vec![
            ColumnType::Date,
            ColumnType::Time,
            ColumnType::Timestamp,
            ColumnType::TimestampTz,
            ColumnType::Uuid,
        ];
        let row = vec![
            Value::Date(19737),
            Value::Time(34_200_000_000),
            Value::Timestamp(1_705_311_000_000_000),
            Value::TimestampTz(-1),
            Value::Uuid([
                0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44,
                0x00, 0x00,
            ]),
        ];
        let bytes = encode(&row, &schema).unwrap();
        assert_eq!(decode(&bytes, &schema).unwrap(), row);
    }

    #[test]
    fn text_coerces_into_a_temporal_column_on_encode() {
        // A text value assigned to a DATE column is parsed at encode time and decodes as a Date.
        let bytes = encode(&[Value::Text("2024-01-15".to_owned())], &[ColumnType::Date]).unwrap();
        assert_eq!(
            decode(&bytes, &[ColumnType::Date]).unwrap(),
            vec![Value::Date(19737)]
        );
    }

    #[test]
    fn invalid_temporal_string_is_rejected() {
        let result = encode(&[Value::Text("nope".to_owned())], &[ColumnType::Date]);
        assert!(matches!(result, Err(Error::InvalidValue { .. })));
    }

    #[test]
    fn roundtrip_numeric_rescales_to_declared_scale() {
        use crate::numeric::Decimal;
        let schema = vec![ColumnType::Numeric {
            precision: 10,
            scale: 2,
        }];
        // '19.9' rescales to scale 2 (19.90) on store.
        let bytes = encode(&[Value::Text("19.9".to_owned())], &schema).unwrap();
        assert_eq!(
            decode(&bytes, &schema).unwrap(),
            vec![Value::Numeric(Decimal::parse("19.90").unwrap())]
        );
        // A NUMERIC value round-trips exactly.
        let bytes = encode(
            &[Value::Numeric(Decimal::parse("123.45").unwrap())],
            &schema,
        )
        .unwrap();
        assert_eq!(
            decode(&bytes, &schema).unwrap(),
            vec![Value::Numeric(Decimal::parse("123.45").unwrap())]
        );
    }

    #[test]
    fn roundtrip_json_canonicalizes() {
        let schema = vec![ColumnType::Json];
        // Unordered keys + whitespace normalize to canonical form on store.
        let bytes = encode(&[Value::Text(r#"{ "b":2, "a":1 }"#.to_owned())], &schema).unwrap();
        assert_eq!(
            decode(&bytes, &schema).unwrap(),
            vec![Value::Json(r#"{"a":1,"b":2}"#.to_owned())]
        );
    }

    #[test]
    fn invalid_json_is_rejected() {
        let result = encode(&[Value::Text("not json".to_owned())], &[ColumnType::Json]);
        assert!(matches!(result, Err(Error::InvalidValue { .. })));
    }

    #[test]
    fn roundtrip_interval_from_text_literal() {
        use crate::interval::Interval;
        let schema = vec![ColumnType::Interval];
        let bytes = encode(&[Value::Text("1 day 02:00:00".to_owned())], &schema).unwrap();
        assert_eq!(
            decode(&bytes, &schema).unwrap(),
            vec![Value::Interval(Interval::parse("1 day 02:00:00").unwrap())]
        );
    }

    #[test]
    fn roundtrip_array_from_text_literal() {
        use nusadb_core::engine::ArrayElem;
        let schema = vec![ColumnType::Array(ArrayElem::Int)];
        let bytes = encode(&[Value::Text("{1,2,3}".to_owned())], &schema).unwrap();
        assert_eq!(
            decode(&bytes, &schema).unwrap(),
            vec![Value::Array(vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3)
            ])]
        );
        // Empty array + a NULL element round-trip.
        let bytes = encode(&[Value::Text("{}".to_owned())], &schema).unwrap();
        assert_eq!(decode(&bytes, &schema).unwrap(), vec![Value::Array(vec![])]);
    }

    #[test]
    fn numeric_precision_overflow_is_rejected() {
        let schema = vec![ColumnType::Numeric {
            precision: 4,
            scale: 2,
        }];
        // 123.45 has 5 significant digits > precision 4.
        let result = encode(&[Value::Text("123.45".to_owned())], &schema);
        assert!(matches!(result, Err(Error::InvalidValue { .. })));
    }

    #[test]
    fn bad_utf8_text_is_malformed() {
        // Manually construct a tuple with bad UTF-8 in the text payload.
        let mut bytes = vec![1u8]; // present
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&[0xff, 0xfe, 0xfd]); // invalid UTF-8
        let result = decode(&bytes, &[ColumnType::Text]);
        assert!(matches!(result, Err(Error::MalformedTuple { .. })));
    }

    #[test]
    fn decode_projected_skips_dropped_blobs_without_offset_drift() {
        use super::decode_projected;
        // Kept columns sit *after* dropped variable-length blobs (TEXT/BYTEA): if skipping a blob
        // advanced the cursor even one byte off, the kept scalars would decode garbage. Covers a
        // multi-byte-UTF-8 text, a raw byte string, a NULL blob, and an empty text.
        let schema = vec![
            ColumnType::Text,  // 0 dropped (multi-byte UTF-8)
            ColumnType::Int,   // 1 kept
            ColumnType::Bytes, // 2 dropped (raw bytes incl. 0x00/0xff)
            ColumnType::Text,  // 3 dropped (NULL)
            ColumnType::Bool,  // 4 kept
            ColumnType::Text,  // 5 dropped (empty string)
            ColumnType::Float, // 6 kept
        ];
        let full = vec![
            Value::Text("héllo→".to_owned()),
            Value::Int(42),
            Value::Bytes(vec![0x00, 0xff, 0x07, 0x41]),
            Value::Null,
            Value::Bool(true),
            Value::Text(String::new()),
            Value::Float(3.5),
        ];
        let bytes = encode(&full, &schema).unwrap();
        // Projecting every column must equal a full decode — proving the skip offsets are identical
        // to the decode offsets for every intervening blob.
        assert_eq!(
            decode_projected(&bytes, &schema, &[0, 1, 2, 3, 4, 5, 6]).unwrap(),
            full
        );
        assert_eq!(decode(&bytes, &schema).unwrap(), full);
        // Keep only the scalars that follow blobs: each blob must be skipped to the exact byte.
        assert_eq!(
            decode_projected(&bytes, &schema, &[1, 4, 6]).unwrap(),
            vec![Value::Int(42), Value::Bool(true), Value::Float(3.5)]
        );
        // A kept blob among dropped blobs still lands right.
        assert_eq!(
            decode_projected(&bytes, &schema, &[2, 6]).unwrap(),
            vec![
                Value::Bytes(vec![0x00, 0xff, 0x07, 0x41]),
                Value::Float(3.5)
            ]
        );
        // Dropping every column consumes the whole tuple cleanly.
        assert_eq!(
            decode_projected(&bytes, &schema, &[]).unwrap(),
            Vec::<Value>::new()
        );
    }

    #[test]
    fn decode_projected_skip_still_catches_a_malformed_dropped_blob() {
        use super::decode_projected;
        // A dropped TEXT column with invalid UTF-8 must still error (the skip validates identically
        // to a full decode), so a corrupt tuple is never silently read past.
        let mut bytes = vec![1u8]; // col 0 present
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&[0xff, 0xfe, 0xfd]); // invalid UTF-8
        bytes.push(1u8); // col 1 present
        bytes.extend_from_slice(&7i64.to_le_bytes());
        let schema = [ColumnType::Text, ColumnType::Int];
        // Even though column 0 is dropped, its bad UTF-8 is caught.
        assert!(matches!(
            decode_projected(&bytes, &schema, &[1]),
            Err(Error::MalformedTuple { .. })
        ));
    }
}
