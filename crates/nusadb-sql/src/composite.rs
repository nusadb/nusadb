//! Composite (row) types — the value a `CREATE TYPE name AS (a int, b text)` column holds.
//!
//! A composite value is a fixed sequence of fields, each either a value or `NULL`. Its canonical
//! text form is `(f1,f2,…)`: a `NULL` field is written as nothing at all, an empty string as `""`,
//! and any field containing a comma, parenthesis, quote, backslash, or whitespace is quoted with
//! `"` and `\` doubled inside.
//!
//! Note that a composite compares differently from the `ROW(…)` comparison the analyzer desugars:
//! `ROW(1,NULL) = ROW(1,NULL)` is `NULL` (SQL row-value comparison is three-valued), while two
//! composite *values* with `NULL` in the same field are equal. [`CompositeVal::compare`] implements
//! the composite rule; the three-valued form stays in the row-comparison desugaring.
//!
//! This module is groundwork: the value, its text form, and its ordering live here; wiring it into
//! the SQL surface (a column type, `CREATE TYPE`, `ROW(…)` as a value, and field access) is separate.

use nusadb_core::engine::ColumnType;

use crate::ast;
use crate::executor::eval::compare as compare_scalar;

/// A composite value: one entry per field of its type, in declaration order. A field that is SQL
/// `NULL` is [`ast::Value::Null`].
#[derive(Clone, Debug, PartialEq)]
pub struct CompositeVal {
    /// The field values, in the declared order of the composite type.
    pub fields: Vec<ast::Value>,
}

impl CompositeVal {
    /// Wrap field values as a composite.
    #[must_use]
    pub const fn new(fields: Vec<ast::Value>) -> Self {
        Self { fields }
    }

    /// Canonical text form: `(f1,f2,…)`, with a `NULL` field written as nothing.
    #[must_use]
    pub fn format(&self) -> String {
        let mut out = String::from("(");
        for (i, f) in self.fields.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            if !matches!(f, ast::Value::Null) {
                out.push_str(&quote_field(&crate::display::value_text(f)));
            }
        }
        out.push(')');
        out
    }

    /// Order two composites field by field.
    ///
    /// A `NULL` field sorts after every value and equal to another `NULL`, so this is a total order
    /// rather than the three-valued `ROW(…)` comparison. Two composites of different arity compare
    /// by their common prefix and then by field count — the caller is responsible for them being of
    /// the same type, exactly as the range predicates are responsible for a matching element kind.
    #[must_use]
    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        for (a, b) in self.fields.iter().zip(&other.fields) {
            let ord = match (matches!(a, ast::Value::Null), matches!(b, ast::Value::Null)) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater, // a NULL field sorts last
                (false, true) => Ordering::Less,
                (false, false) => compare_scalar(a, b),
            };
            if ord != Ordering::Equal {
                return ord;
            }
        }
        self.fields.len().cmp(&other.fields.len())
    }
}

/// Quote a rendered field if any of its characters would otherwise be read as composite
/// punctuation, doubling `"` and `\` inside. An empty field must be quoted — bare emptiness is how
/// the text form spells `NULL`.
fn quote_field(s: &str) -> String {
    let needs_quotes = s.is_empty()
        || s.chars()
            .any(|c| c.is_whitespace() || matches!(c, '"' | '\\' | '(' | ')' | ','));
    if !needs_quotes {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        if c == '"' || c == '\\' {
            out.push(c); // doubled, not backslash-escaped
        }
        out.push(c);
    }
    out.push('"');
    out
}

/// Parse the canonical text form of a composite whose fields have the given types.
///
/// Returns `None` if the text is malformed or a field does not parse as its type, and if the field
/// count does not match `field_types` — a composite is a fixed shape, so a short or long literal is
/// a real error rather than something to pad or truncate.
#[must_use]
pub fn parse(s: &str, field_types: &[ColumnType]) -> Option<CompositeVal> {
    let t = s.trim();
    let body = t.strip_prefix('(')?.strip_suffix(')')?;
    let raw = split_fields(body)?;
    if raw.len() != field_types.len() {
        return None;
    }
    let mut fields = Vec::with_capacity(raw.len());
    for (text, ty) in raw.iter().zip(field_types) {
        fields.push(match text {
            // An unquoted empty field is NULL; a quoted one is the empty string.
            None => ast::Value::Null,
            Some(v) => parse_field(v, *ty)?,
        });
    }
    Some(CompositeVal { fields })
}

/// Split a composite body into its fields, honouring quotes. Each field is `None` when it was
/// written bare-empty (SQL `NULL`) and `Some(text)` otherwise, with quoting already undone.
///
/// A backslash takes the next character literally, inside quotes or out; `""` inside quotes is a
/// literal quote; and a `"` otherwise opens or closes a quoted section, which may sit next to bare
/// text in the same field. Whitespace is never trimmed — a field of one space is that space, not
/// `NULL`.
fn split_fields(body: &str) -> Option<Vec<Option<String>>> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let (mut quoted, mut was_quoted) = (false, false);
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            // An escape takes the next character as itself; a trailing one escapes nothing.
            '\\' => cur.push(chars.next()?),
            '"' if quoted => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cur.push('"');
                } else {
                    quoted = false;
                }
            },
            '"' => {
                quoted = true;
                was_quoted = true;
            },
            ',' if !quoted => {
                out.push((was_quoted || !cur.is_empty()).then(|| std::mem::take(&mut cur)));
                cur.clear();
                was_quoted = false;
            },
            _ => cur.push(c),
        }
    }
    if quoted {
        return None; // an unterminated quoted field
    }
    out.push((was_quoted || !cur.is_empty()).then_some(cur));
    Some(out)
}

/// Parse one field's text as `ty`.
fn parse_field(s: &str, ty: ColumnType) -> Option<ast::Value> {
    Some(match ty.physical() {
        // The same boolean spellings a text cast accepts.
        ColumnType::Bool => ast::Value::Bool(match s.trim().to_ascii_lowercase().as_str() {
            "true" | "t" | "yes" | "y" | "on" | "1" => true,
            "false" | "f" | "no" | "n" | "off" | "0" => false,
            _ => return None,
        }),
        ColumnType::Int => ast::Value::Int(s.parse().ok()?),
        ColumnType::Float => ast::Value::Float(s.parse().ok()?),
        ColumnType::Text => ast::Value::Text(s.to_owned()),
        ColumnType::Numeric { .. } => ast::Value::Numeric(crate::numeric::Decimal::parse(s)?),
        ColumnType::Date => ast::Value::Date(crate::temporal::parse_date(s)?),
        ColumnType::Time => ast::Value::Time(crate::temporal::parse_time(s)?),
        ColumnType::TimeTz => ast::Value::TimeTz(crate::temporal::parse_timetz(s)?),
        ColumnType::Timestamp => ast::Value::Timestamp(crate::temporal::parse_timestamp(s)?),
        ColumnType::TimestampTz => ast::Value::TimestampTz(crate::temporal::parse_timestamptz(s)?),
        ColumnType::Uuid => ast::Value::Uuid(crate::temporal::parse_uuid(s)?),
        ColumnType::Macaddr => ast::Value::Macaddr(crate::macaddr::parse(s)?),
        ColumnType::Macaddr8 => ast::Value::Macaddr8(crate::macaddr8::parse(s)?),
        ColumnType::Inet => ast::Value::Inet(crate::inet::parse_inet(s)?),
        ColumnType::Cidr => ast::Value::Inet(crate::inet::parse_cidr(s)?),
        ColumnType::Json | ColumnType::Jsonb => ast::Value::Json(crate::json::canonicalize(s)?),
        ColumnType::Range(kind) => ast::Value::Range(Box::new(crate::range::parse(s, kind)?)),
        // A field type with no text form of its own is not something a literal can spell.
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> ast::Value {
        ast::Value::Int(n)
    }

    fn text(s: &str) -> ast::Value {
        ast::Value::Text(s.to_owned())
    }

    const T_INT_TEXT: &[ColumnType] = &[ColumnType::Int, ColumnType::Text];

    #[test]
    fn format_writes_null_as_nothing_and_quotes_what_needs_it() {
        let c = |fields: Vec<ast::Value>| CompositeVal::new(fields).format();
        assert_eq!(c(vec![int(1), text("abc")]), "(1,abc)");
        // A NULL field is bare; an empty string is a quoted empty field. The two must not collide.
        assert_eq!(c(vec![int(1), ast::Value::Null]), "(1,)");
        assert_eq!(c(vec![int(1), text("")]), "(1,\"\")");
        assert_eq!(c(vec![ast::Value::Null, ast::Value::Null]), "(,)");
        // Punctuation and whitespace force quoting; `"` and `\` double.
        assert_eq!(c(vec![int(1), text("a b")]), "(1,\"a b\")");
        assert_eq!(c(vec![int(1), text("a,b")]), "(1,\"a,b\")");
        assert_eq!(c(vec![int(1), text("a\"b")]), "(1,\"a\"\"b\")");
        assert_eq!(c(vec![int(1), text("a\\b")]), "(1,\"a\\\\b\")");
        assert_eq!(c(vec![int(1), text("(x)")]), "(1,\"(x)\")");
        assert_eq!(c(vec![int(1), text(" x")]), "(1,\" x\")");
    }

    #[test]
    fn parse_round_trips_the_format_including_null_versus_empty() {
        for fields in [
            vec![int(1), text("abc")],
            vec![int(1), ast::Value::Null],
            vec![int(1), text("")],
            vec![ast::Value::Null, ast::Value::Null],
            vec![int(1), text("a,b")],
            vec![int(1), text("a\"b")],
            vec![int(1), text("a\\b")],
            vec![int(1), text(" x")],
            vec![int(-5), text("(x)")],
        ] {
            let v = CompositeVal::new(fields);
            assert_eq!(
                parse(&v.format(), T_INT_TEXT),
                Some(v.clone()),
                "round-trip changed {}",
                v.format()
            );
        }
        // The distinction the text form turns on: bare-empty is NULL, quoted-empty is "".
        assert_eq!(
            parse("(1,)", T_INT_TEXT),
            Some(CompositeVal::new(vec![int(1), ast::Value::Null]))
        );
        assert_eq!(
            parse("(1,\"\")", T_INT_TEXT),
            Some(CompositeVal::new(vec![int(1), text("")]))
        );
    }

    #[test]
    fn parse_rejects_malformed_and_mis_shaped_literals() {
        for bad in [
            "1,abc",     // no parentheses
            "(1,abc",    // unterminated
            "(1,\"abc)", // unterminated quote
            "(1,abc\\)", // a trailing backslash escapes nothing
            "(1)",       // too few fields
            "(1,abc,2)", // too many fields
            "(x,abc)",   // field does not parse as its type
        ] {
            assert_eq!(parse(bad, T_INT_TEXT), None, "accepted {bad}");
        }
    }

    #[test]
    fn parse_follows_the_escaping_rules_of_the_reference_engine() {
        let b = |s: &str| parse(s, T_INT_TEXT).and_then(|c| c.fields.get(1).cloned());
        // A backslash takes the next character literally — inside quotes or out.
        assert_eq!(b("(1,a\\b)"), Some(text("ab")));
        assert_eq!(b("(1,\"a\\b\")"), Some(text("ab")));
        assert_eq!(b("(1,a\\,b)"), Some(text("a,b")));
        assert_eq!(b("(1,a\\)b)"), Some(text("a)b")));
        // A doubled backslash is one literal backslash — the form `format` writes.
        assert_eq!(b("(1,\"a\\\\b\")"), Some(text("a\\b")));
        // A doubled quote inside quotes is one literal quote, not a close-then-reopen.
        assert_eq!(b("(1,\"a\"\"b\")"), Some(text("a\"b")));
        // A quoted section may sit beside bare text in the same field.
        assert_eq!(b("(1,\"a\"b)"), Some(text("ab")));
        // Whitespace is never trimmed: a lone space is that space, not NULL.
        assert_eq!(b("(1, x)"), Some(text(" x")));
        assert_eq!(b("(1,x )"), Some(text("x ")));
        assert_eq!(b("(1, )"), Some(text(" ")));
    }

    #[test]
    fn compare_orders_field_by_field_with_nulls_last() {
        use std::cmp::Ordering;
        let v = |a: ast::Value, b: ast::Value| CompositeVal::new(vec![a, b]);
        // Earlier fields dominate.
        assert_eq!(
            v(int(1), text("a")).compare(&v(int(1), text("b"))),
            Ordering::Less
        );
        assert_eq!(
            v(int(1), text("a")).compare(&v(int(2), text("a"))),
            Ordering::Less
        );
        assert_eq!(
            v(int(2), text("a")).compare(&v(int(1), text("b"))),
            Ordering::Greater
        );
        // A NULL field sorts after every value, in any position.
        assert_eq!(
            v(int(1), text("a")).compare(&v(int(1), ast::Value::Null)),
            Ordering::Less
        );
        assert_eq!(
            v(int(1), ast::Value::Null).compare(&v(int(1), text("a"))),
            Ordering::Greater
        );
        assert_eq!(
            v(ast::Value::Null, int(1)).compare(&v(int(1), int(1))),
            Ordering::Greater
        );
        // Two NULLs in the same field are equal — unlike the three-valued `ROW(…)` comparison.
        assert_eq!(
            v(int(1), ast::Value::Null).compare(&v(int(1), ast::Value::Null)),
            Ordering::Equal
        );
    }
}
