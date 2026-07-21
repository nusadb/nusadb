//! DDL statement converters: ANALYZE, CREATE TABLE/INDEX/SEQUENCE/SCHEMA/VIEW, ALTER TABLE, DROP.
//!
//! Split verbatim out of `parser/mod.rs` (ADR 007); see that module for the
//! anti-corruption-layer contract. Cross-submodule converters resolve via `use super::*`.
#![allow(clippy::wildcard_imports)]

use super::*;

// === ANALYZE ==============================================================

/// Lower `ANALYZE [TABLE] name [(columns)]`. Dialect extras the engine has no
/// notion of — `PARTITION (...)`, `CACHE METADATA`, `NOSCAN` — are rejected so
/// the surface stays honest; `COMPUTE STATISTICS` is accepted and ignored
/// (it is the default behavior).
pub(super) fn convert_analyze(
    name: &sql::ObjectName,
    partitions: Option<&[sql::Expr]>,
    columns: Vec<sql::Ident>,
    cache_metadata: bool,
    noscan: bool,
) -> Result<ast::Statement, Error> {
    if partitions.is_some_and(|p| !p.is_empty()) {
        return unsupported("ANALYZE ... PARTITION");
    }
    if cache_metadata {
        return unsupported("ANALYZE ... CACHE METADATA");
    }
    if noscan {
        return unsupported("ANALYZE ... NOSCAN");
    }
    Ok(ast::Statement::Analyze(ast::Analyze {
        table: object_name(name)?,
        columns: columns.into_iter().map(|c| fold_ident(&c)).collect(),
    }))
}

// === CREATE TABLE =========================================================

pub(super) fn convert_create_table(ct: sql::CreateTable) -> Result<ast::Statement, Error> {
    if ct.temporary {
        // NusaDB has no temporary tables; silently creating a *persistent* table for
        // `CREATE TEMPORARY TABLE` would be a silent-wrong surface (class).
        return unsupported(
            "CREATE TEMPORARY TABLE is not supported (NusaDB has no temporary tables)",
        );
    }
    // `PARTITION BY` / `CLUSTERED BY` change where rows physically live. NusaDB has no table
    // partitioning yet, so silently dropping the clause and creating an ordinary unpartitioned heap
    // would mis-store rows and accept out-of-range inserts — the same silent-wrong class as the
    // TEMPORARY guard above. Reject loudly instead (QA).
    if ct.partition_by.is_some() {
        return unsupported(
            "CREATE TABLE ... PARTITION BY is not supported yet (NusaDB does not partition tables)",
        );
    }
    if ct.clustered_by.is_some() {
        return unsupported("CREATE TABLE ... CLUSTERED BY is not supported");
    }
    // `CREATE TABLE ... AS <select>`: the schema is derived entirely from the query. An
    // explicit column list (which `sqlparser` only accepts in typed form here) and inline constraints
    // are out of scope for v1 — use column aliases in the `SELECT` to name the output columns.
    if let Some(query) = ct.query {
        if !ct.columns.is_empty() {
            return unsupported(
                "CREATE TABLE ... AS SELECT with a column list (alias the SELECT columns instead)",
            );
        }
        if !ct.constraints.is_empty() {
            return unsupported("CREATE TABLE ... AS SELECT with table constraints");
        }
        return Ok(ast::Statement::CreateTableAs(ast::CreateTableAs {
            name: object_name(&ct.name)?,
            query: Box::new(convert_select(*query)?),
            if_not_exists: ct.if_not_exists,
        }));
    }
    if ct.columns.is_empty() {
        return unsupported("CREATE TABLE with no columns");
    }
    let (schema, name) = table_ref_name(&ct.name)?;
    let mut columns = Vec::with_capacity(ct.columns.len());
    let mut constraints = Vec::with_capacity(ct.constraints.len());
    for col in ct.columns {
        let (column, lifted) = convert_column_def(&col, true)?;
        columns.push(column);
        // Column-level CHECK / REFERENCES are lifted to table constraints.
        constraints.extend(lifted);
    }
    for constraint in ct.constraints {
        constraints.push(convert_table_constraint(constraint)?);
    }
    Ok(ast::Statement::CreateTable(ast::CreateTable {
        schema,
        name,
        columns,
        constraints,
        if_not_exists: ct.if_not_exists,
    }))
}

/// Convert a column definition, returning the [`ast::ColumnDef`] plus any column-level
/// `CHECK`/`REFERENCES` constraints lifted to table constraints. Column-local
/// options (`NOT NULL`, `PRIMARY KEY`, `UNIQUE`, `DEFAULT`, `GENERATED`) are folded into the
/// `ColumnDef` itself.
///
/// `synth_type_checks` controls whether the synthetic type-bound CHECKs (a `VARCHAR(n)` length limit
/// or a narrow-integer range) are lifted. `CREATE TABLE` passes `true`; `ALTER TABLE ADD COLUMN`
/// passes `false`, since that path cannot atomically add a table constraint — so an added narrow
/// column is stored without the desugared bound rather than rejecting the whole `ADD COLUMN`.
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-column-option dispatch; length tracks the option set"
)]
pub(super) fn convert_column_def(
    col: &sql::ColumnDef,
    synth_type_checks: bool,
) -> Result<(ast::ColumnDef, Vec<ast::TableConstraint>), Error> {
    // SERIAL/BIGSERIAL/SMALLSERIAL: an auto-increment INT. sqlparser does not model it as a
    // keyword, so it arrives as a custom type name; map it to INT here and flag the column serial. It
    // implies NOT NULL. The SQL-standard `GENERATED ... AS IDENTITY` form (a column option) also sets
    // `serial` below.
    let mut serial = serial_kind(&col.data_type);
    // An unresolved bare custom type name (e.g. a user-defined `ENUM`) is deferred: the executor
    // resolves it against the type catalog at CREATE TABLE time, with `TEXT` as the storage
    // placeholder (B-ENUM). Built-in / aliased / serial types resolve here as usual.
    let (ty, udt_name) = if serial {
        (ColumnType::Int, None)
    } else {
        match convert_data_type(&col.data_type) {
            Ok(ty) => (ty, None),
            Err(e) => match deferred_udt_name(&col.data_type) {
                Some(name) => (ColumnType::Text, Some(name)),
                None => return Err(e),
            },
        }
    };
    let name = fold_ident(&col.name);
    let mut nullable = !serial;
    let mut identity_always = false;
    let mut primary_key = false;
    let mut unique = false;
    let mut default = None;
    let mut default_sql = None;
    let mut generated = None;
    let mut lifted = Vec::new();
    for opt in &col.options {
        match &opt.option {
            sql::ColumnOption::NotNull => nullable = false,
            sql::ColumnOption::Null => {},
            // `PRIMARY KEY` implies `NOT NULL`; a bare `UNIQUE` does not. (sqlparser 0.62 models
            // these as dedicated constraint structs; index hints / characteristics on a
            // column-level key were ignored in 0.51 and still are.)
            sql::ColumnOption::PrimaryKey(_) => {
                primary_key = true;
                nullable = false;
            },
            sql::ColumnOption::Unique(_) => unique = true,
            // `DEFAULT <expr>`. Capture the canonical SQL text too (like CHECK's
            // `predicate_sql`) so the executor can persist + re-parse it per write.
            sql::ColumnOption::Default(expr) => {
                default_sql = Some(expr.to_string());
                default = Some(Box::new(convert_expr(expr.clone())?));
            },
            // Column-level `CHECK (<expr>)` → a table-level CHECK constraint. The 0.62
            // `[NOT] ENFORCED` suffix is not modelled — rejected rather than silently enforced.
            sql::ColumnOption::Check(check) => {
                if check.enforced.is_some() {
                    return unsupported("CHECK constraint with ENFORCED / NOT ENFORCED");
                }
                lifted.push(ast::TableConstraint::Check {
                    name: None,
                    predicate_sql: check.expr.to_string(),
                    expr: convert_expr((*check.expr).clone())?,
                });
            },
            // Column-level `REFERENCES t (cols)` → a single-column table FK.
            sql::ColumnOption::ForeignKey(fk) => {
                reject_characteristics(fk.characteristics.as_ref())?;
                if fk.match_kind.is_some() {
                    return unsupported("REFERENCES ... MATCH FULL/PARTIAL/SIMPLE");
                }
                lifted.push(ast::TableConstraint::ForeignKey {
                    name: None,
                    columns: vec![name.clone()],
                    foreign_table: object_name(&fk.foreign_table)?,
                    referred_columns: fk.referred_columns.iter().map(fold_ident).collect(),
                    on_delete: fk.on_delete.map(convert_referential_action),
                    on_update: fk.on_update.map(convert_referential_action),
                });
            },
            // `GENERATED ALWAYS AS (<expr>) [STORED|VIRTUAL]` — a computed column.
            sql::ColumnOption::Generated {
                generation_expr: Some(expr),
                generation_expr_mode,
                ..
            } => {
                let stored = matches!(
                    generation_expr_mode,
                    Some(sql::GeneratedExpressionMode::Stored)
                );
                generated = Some(ast::GeneratedColumn {
                    expr: Box::new(convert_expr(expr.clone())?),
                    sql: expr.to_string(),
                    stored,
                });
            },
            // `GENERATED { ALWAYS | BY DEFAULT } AS IDENTITY` — the SQL-standard auto-increment column
            // (no expression). Treated as a `SERIAL`: an INT backed by a per-column sequence,
            // implying NOT NULL. `ALWAYS` is stricter — it rejects an explicit value (#9a) — while
            // `BY DEFAULT` accepts one like `SERIAL`. `OVERRIDING` and identity sequence options are
            // ignored, so an `ALWAYS` column is always system-generated here.
            sql::ColumnOption::Generated {
                generation_expr: None,
                generated_as,
                ..
            } => {
                serial = true;
                nullable = false;
                identity_always = matches!(generated_as, sql::GeneratedAs::Always);
            },
            // Column-level `COLLATE`: accept the byte-order `C`/`POSIX` (a no-op, since NusaDB
            // already sorts text by byte value) and reject a locale collation loudly (D-COLLATE).
            sql::ColumnOption::Collation(name) => require_byte_order_collation(name)?,
            other => return unsupported(&format!("column constraint `{other}`")),
        }
    }
    if synth_type_checks {
        if let Some(check) = char_length_check(&name, &col.data_type)? {
            lifted.push(check);
        }
        if let Some(check) = int_range_check(&name, &col.data_type)? {
            lifted.push(check);
        }
    }
    let column = ast::ColumnDef {
        name,
        ty,
        udt_name,
        nullable,
        primary_key,
        unique,
        default,
        default_sql,
        generated,
        serial,
        identity_always,
    };
    Ok((column, lifted))
}

/// The bare user-defined type name to defer to the executor when `convert_data_type` rejects a
/// column type (B-ENUM): a single-identifier `Custom` type with no modifiers (e.g. an `ENUM` name).
/// `None` for anything else — a modifier'd or multi-part name is not a user-defined type reference,
/// so the original "unsupported type" error stands.
fn deferred_udt_name(ty: &sql::DataType) -> Option<String> {
    let sql::DataType::Custom(name, modifiers) = ty else {
        return None;
    };
    let [part] = name.0.as_slice() else {
        return None;
    };
    let ident = part.as_ident()?;
    modifiers.is_empty().then(|| fold_ident(ident))
}

/// Desugar a `VARCHAR(n)` / `CHAR(n)` length limit into a synthetic `CHECK (length("col") <= n)`.
/// The `ColumnType::VarChar`/`Char` variants carry `n` only so the declared type round-trips in DDL;
/// they are stored as `TEXT` and do *not* enforce length themselves, so the limit is enforced here by
/// reusing the existing CHECK machinery + the `length` scalar (char-count).
/// NULL passes (CHECK is not FALSE), so the limit applies only to present values.
/// Returns `None` for an unbounded character type. `CHAR(n)` is blank-padded (bpchar): the limit
/// applies to the value with trailing blanks stripped, so a value overflowing only by trailing
/// blanks is accepted; `VARCHAR(n)` counts every character.
fn char_length_check(
    name: &str,
    data_type: &sql::DataType,
) -> Result<Option<ast::TableConstraint>, Error> {
    let Some(limit) = char_length_limit(data_type) else {
        return Ok(None);
    };
    let col = name.replace('"', "\"\"");
    // CHAR(n)/CHARACTER(n) is blank-padded (bpchar): trailing blanks are insignificant, so the limit
    // applies to the value with trailing spaces stripped — a value that overflows only by trailing
    // blanks is accepted (e.g. `'abcde '::char(5)`). VARCHAR(n) counts every character.
    let blank_padded = matches!(
        data_type,
        sql::DataType::Char(_) | sql::DataType::Character(_)
    );
    let predicate_sql = if blank_padded {
        format!("length(rtrim(\"{col}\", ' ')) <= {limit}")
    } else {
        format!("length(\"{col}\") <= {limit}")
    };
    let expr = super::parse_expression(&predicate_sql)?;
    Ok(Some(ast::TableConstraint::Check {
        // A reserved synthetic name so introspection can hide this type-bound check (it is still
        // enforced); the column name keeps it unique within the table.
        name: Some(format!("{}{name}", crate::SYNTHETIC_TYPE_CHECK_PREFIX)),
        predicate_sql,
        expr,
    }))
}

/// Desugar a narrow integer type (`SMALLINT`/`INT`/`TINYINT`/`MEDIUMINT` and spelling variants) into
/// a synthetic range `CHECK ("col" BETWEEN lo AND hi)` so a value outside the declared width is
/// rejected at write time rather than silently stored (K4/K6). Every integer is stored as a 64-bit
/// [`ColumnType::Int`]; the declared width's bounds are enforced here by reusing the existing CHECK
/// machinery — exactly like the `VARCHAR(n)` length limit. `BIGINT`/`INT8` (the full `i64`) and any
/// non-integer type need no check. NULL passes (CHECK is not FALSE), so the bound applies only to
/// present values.
fn int_range_check(
    name: &str,
    data_type: &sql::DataType,
) -> Result<Option<ast::TableConstraint>, Error> {
    use sql::DataType as D;
    // MEDIUMINT is a 24-bit integer type: [-2^23, 2^23 - 1].
    const MEDIUMINT_MIN: i64 = -(1 << 23);
    const MEDIUMINT_MAX: i64 = (1 << 23) - 1;
    let (lo, hi): (i64, i64) = match data_type {
        D::TinyInt(_) => (i64::from(i8::MIN), i64::from(i8::MAX)),
        D::SmallInt(_) | D::Int2(_) => (i64::from(i16::MIN), i64::from(i16::MAX)),
        D::MediumInt(_) => (MEDIUMINT_MIN, MEDIUMINT_MAX),
        D::Int(_) | D::Int4(_) | D::Integer(_) => (i64::from(i32::MIN), i64::from(i32::MAX)),
        // BIGINT/INT8 span the full storage width, and non-integer types are out of scope.
        _ => return Ok(None),
    };
    let predicate_sql = format!("\"{}\" BETWEEN {lo} AND {hi}", name.replace('"', "\"\""));
    let expr = super::parse_expression(&predicate_sql)?;
    Ok(Some(ast::TableConstraint::Check {
        // A reserved synthetic name so introspection can hide this type-bound check (it is still
        // enforced); the column name keeps it unique within the table.
        name: Some(format!("{}{name}", crate::SYNTHETIC_TYPE_CHECK_PREFIX)),
        predicate_sql,
        expr,
    }))
}

/// Whether a column's declared type is `SERIAL`/`BIGSERIAL`/`SMALLSERIAL` (and the `serialN` spelling
/// variants). sqlparser has no keyword for these, so they parse as a single-word custom type name.
fn serial_kind(ty: &sql::DataType) -> bool {
    let sql::DataType::Custom(name, modifiers) = ty else {
        return false;
    };
    let [part] = name.0.as_slice() else {
        return false;
    };
    let Some(ident) = part.as_ident() else {
        return false;
    };
    modifiers.is_empty()
        && matches!(
            ident.value.to_ascii_lowercase().as_str(),
            "serial" | "serial2" | "serial4" | "serial8" | "smallserial" | "bigserial"
        )
}

/// The declared maximum character length of a `CHAR(n)` / `VARCHAR(n)` (and spelling variants), or
/// `None` for an unbounded character type (`TEXT`, `VARCHAR` without a length, or `VARCHAR(MAX)`).
/// Used to desugar the length modifier into a `CHECK` constraint.
fn char_length_limit(ty: &sql::DataType) -> Option<u64> {
    use sql::DataType as D;
    let length = match ty {
        D::Char(len)
        | D::Character(len)
        | D::Varchar(len)
        | D::CharVarying(len)
        | D::CharacterVarying(len)
        | D::Nvarchar(len) => len.as_ref()?,
        _ => return None,
    };
    match length {
        sql::CharacterLength::IntegerLength { length, .. } => Some(*length),
        sql::CharacterLength::Max => None,
    }
}

/// Narrow a declared character length to the `u32` the [`ColumnType::VarChar`]/[`ColumnType::Char`]
/// variants carry. Lengths beyond `u32::MAX` are absurd for a single column, so we saturate (the
/// authoritative enforcement is the desugared `CHECK`, which keeps the full `u64`).
const fn clamp_len(n: u64) -> u32 {
    if n > u32::MAX as u64 {
        u32::MAX
    } else {
        n as u32
    }
}

pub(super) fn convert_data_type(ty: &sql::DataType) -> Result<ColumnType, Error> {
    use sql::DataType as D;
    let mapped = match ty {
        D::Bool | D::Boolean => ColumnType::Bool,
        // SMALLINT/BIGINT keep their declared type so it round-trips in DDL / information_schema
        // (K4/K6 integer fidelity); the value range is still enforced by the synthetic CHECK from
        // `int_range_check`. TINYINT/MEDIUMINT have no standard catalog name, so they stay `Int`.
        D::SmallInt(_) | D::Int2(_) => ColumnType::SmallInt,
        D::BigInt(_) | D::Int8(_) => ColumnType::BigInt,
        D::TinyInt(_) | D::MediumInt(_) | D::Int(_) | D::Int4(_) | D::Integer(_) => ColumnType::Int,
        // REAL / FLOAT4 keep their declared type so it round-trips in DDL / information_schema (K4/K6),
        // stored identically to the 64-bit FLOAT; FLOAT8 / DOUBLE PRECISION and a bare FLOAT(p) are the
        // double type.
        D::Real | D::Float4 => ColumnType::Real,
        // `DOUBLE(p[, s])` (a non-standard precision form sqlparser 0.62 newly models) is out of surface — only the
        // bare form maps; a precision'd DOUBLE falls through to the loud reject below.
        D::Float(_) | D::Float8 | D::Double(sql::ExactNumberInfo::None) | D::DoublePrecision => {
            ColumnType::Float
        },
        // `TEXT`/`STRING` are unbounded; `VARCHAR(n)`/`CHAR(n)` keep their declared length so the
        // type round-trips in DDL (`SHOW COLUMNS`). A bare `VARCHAR`/`CHAR` (no length) and
        // `VARCHAR(MAX)` are unbounded, so they map to plain `TEXT`. The length is *enforced*
        // separately via the desugared `CHECK` constraint (see `char_length_limit`); these variants
        // are stored identically to `TEXT` (see `ColumnType::physical`).
        // Bit strings, full-text, and geometric types are also stored as their canonical text form
        // (no native operators yet) — the same mapping their 0.51 `Custom`-typed spellings got from
        // `aliased_type` below; sqlparser 0.62 models them as first-class data types.
        D::Text
        | D::String(_)
        | D::Bit(_)
        | D::BitVarying(_)
        | D::VarBit(_)
        | D::TsVector
        | D::TsQuery
        | D::GeometricType(_) => ColumnType::Text,
        D::Varchar(_) | D::CharVarying(_) | D::CharacterVarying(_) | D::Nvarchar(_) => {
            char_length_limit(ty).map_or(ColumnType::Text, |n| ColumnType::VarChar(clamp_len(n)))
        },
        D::Char(_) | D::Character(_) => {
            char_length_limit(ty).map_or(ColumnType::Text, |n| ColumnType::Char(clamp_len(n)))
        },
        D::Bytea | D::Blob(_) | D::Binary(_) | D::Varbinary(_) | D::Bytes(_) => ColumnType::Bytes,
        // Temporal + UUID. `TIMESTAMP WITH TIME ZONE` (and the `Tz` alias) is distinct from a
        // bare `TIMESTAMP`/`DATETIME`.
        D::Timestamp(_, tz) => match tz {
            sql::TimezoneInfo::Tz | sql::TimezoneInfo::WithTimeZone => ColumnType::TimestampTz,
            sql::TimezoneInfo::None | sql::TimezoneInfo::WithoutTimeZone => ColumnType::Timestamp,
        },
        D::Datetime(_) => ColumnType::Timestamp,
        D::Date => ColumnType::Date,
        D::Time(_, tz) => match tz {
            sql::TimezoneInfo::Tz | sql::TimezoneInfo::WithTimeZone => ColumnType::TimeTz,
            sql::TimezoneInfo::None | sql::TimezoneInfo::WithoutTimeZone => ColumnType::Time,
        },
        D::Uuid => ColumnType::Uuid,
        // Exact decimal: `NUMERIC`/`DECIMAL`/`DEC` with optional `(p[, s])`.
        D::Numeric(info) | D::Decimal(info) | D::Dec(info) => exact_numeric_type(info)?,
        // JSON and JSONB are both stored as canonical text, but keep their declared type so it
        // round-trips in DDL / information_schema (K4/K6).
        D::JSON => ColumnType::Json,
        D::JSONB => ColumnType::Jsonb,
        // Calendar duration. A field-qualified `INTERVAL YEAR TO MONTH` or a precision'd
        // `INTERVAL(6)` (newly parsed by sqlparser 0.62) is out of surface and falls through to
        // the loud reject below.
        D::Interval {
            fields: None,
            precision: None,
        } => ColumnType::Interval,
        // One-dimensional array of a scalar, e.g. `INT[]` / `TEXT[]`.
        D::Array(elem) => return array_type(elem),
        // `VECTOR(n)` parses as an unknown custom type with one numeric modifier.
        D::Custom(name, modifiers) if is_vector_name(name) => return vector_type(modifiers),
        // A custom (named) type: a standard SQL type NusaDB does not model natively (currency,
        // object-id, network, bit-string, geometric, range, full-text, XML) maps onto a base storage
        // type so schemas using it still load (B-types); a genuinely unknown name is still rejected.
        D::Custom(name, _) => {
            return aliased_type(name)
                .ok_or_else(|| Error::Unsupported(format!("column type `{name}`")));
        },
        other => return unsupported(&format!("column type `{other}`")),
    };
    Ok(mapped)
}

/// Map a standard SQL type NusaDB does not model natively onto the closest base storage type, so a
/// schema using it still loads (migration / compatibility, B-types). The value is stored in its base
/// form — native semantics (geometric / range / network / full-text operators, currency formatting)
/// are a follow-up. `None` for an unrecognized name (a genuine unknown type stays an error rather
/// than silently becoming text).
fn aliased_type(name: &sql::ObjectName) -> Option<ColumnType> {
    let [part] = name.0.as_slice() else {
        return None;
    };
    let ident = part.as_ident()?;
    Some(match ident.value.to_ascii_lowercase().as_str() {
        // Currency: an exact fixed-point decimal (the value, not the locale formatting).
        "money" => ColumnType::Numeric {
            precision: 19,
            scale: 2,
        },
        // Object identifier — an unsigned 32-bit integer.
        "oid" | "regclass" | "regtype" | "xid" | "cid" | "tid" => ColumnType::Int,
        // Stored as their canonical text form (no native operators yet): network addresses, bit
        // strings, geometric, range, full-text, and XML types.
        "xml" | "cidr" | "inet" | "macaddr" | "macaddr8" | "bit" | "varbit" | "tsvector"
        | "tsquery" | "point" | "line" | "lseg" | "box" | "path" | "polygon" | "circle"
        | "int4range" | "int8range" | "numrange" | "tsrange" | "tstzrange" | "daterange" => {
            ColumnType::Text
        },
        _ => return None,
    })
}

/// Map a sqlparser array type (`INT[]`, `TEXT[]`, …) to [`ColumnType::Array`]. The element
/// must be a supported scalar (no nested arrays / NUMERIC / JSON elements yet).
pub(super) fn array_type(elem: &sql::ArrayElemTypeDef) -> Result<ColumnType, Error> {
    use sql::ArrayElemTypeDef as A;
    let inner = match elem {
        A::AngleBracket(ty) | A::SquareBracket(ty, _) | A::Parenthesis(ty) => {
            convert_data_type(ty)?
        },
        A::None => return unsupported("array without an element type"),
    };
    nusadb_core::engine::ArrayElem::from_column_type(inner).map_or_else(
        || unsupported(&format!("array of `{inner:?}` (only scalar element types)")),
        |e| Ok(ColumnType::Array(e)),
    )
}

/// Whether a custom-type name is `VECTOR` (case-insensitive, unqualified).
fn is_vector_name(name: &sql::ObjectName) -> bool {
    matches!(
        name.0.as_slice(),
        [part] if part.as_ident().is_some_and(|i| i.value.eq_ignore_ascii_case("vector"))
    )
}

/// Map `VECTOR(n)` to [`ColumnType::Vector`]. Exactly one positive integer modifier — the
/// dimension — is required; `VECTOR` with no/zero/multiple modifiers is rejected.
fn vector_type(modifiers: &[String]) -> Result<ColumnType, Error> {
    let [dim] = modifiers else {
        return unsupported("VECTOR requires exactly one dimension, e.g. VECTOR(3)");
    };
    match dim.trim().parse::<u32>() {
        Ok(n) if n > 0 => Ok(ColumnType::Vector(n)),
        _ => unsupported(&format!(
            "VECTOR dimension must be a positive integer, got `{dim}`"
        )),
    }
}

/// Map sqlparser's `ExactNumberInfo` to a `ColumnType::Numeric`. No arguments => unconstrained
/// (`precision = 0`); a precision with no scale => scale 0. A precision/scale beyond the `u8` the
/// catalog stores is rejected with `Error::Unsupported` rather than clamped.
pub(super) fn exact_numeric_type(info: &sql::ExactNumberInfo) -> Result<ColumnType, Error> {
    // Reject a precision/scale beyond what the catalog stores (a `u8`) rather than silently clamping
    // it to 255 — a clamped declaration would round values to the wrong scale.
    let checked = |v: u64, what: &str| -> Result<u8, Error> {
        u8::try_from(v).map_err(|_| {
            Error::Unsupported(format!(
                "NUMERIC {what} {v} exceeds the maximum supported ({})",
                u8::MAX
            ))
        })
    };
    let (precision, scale) = match info {
        sql::ExactNumberInfo::None => (0, 0),
        sql::ExactNumberInfo::Precision(p) => (checked(*p, "precision")?, 0),
        sql::ExactNumberInfo::PrecisionAndScale(p, s) => {
            // sqlparser 0.62 models the scale as signed (a negative scale is a newer-standard extension);
            // the catalog stores an unsigned scale, so anything outside 0..=255 is rejected.
            let scale = u64::try_from(*s).map_err(|_| {
                Error::Unsupported(format!(
                    "NUMERIC scale {s} out of the supported range (0..=255)"
                ))
            })?;
            (checked(*p, "precision")?, checked(scale, "scale")?)
        },
    };
    Ok(ColumnType::Numeric { precision, scale })
}

// === CREATE INDEX =========================================================

pub(super) fn convert_create_index(ci: sql::CreateIndex) -> Result<ast::CreateIndex, Error> {
    if ci.concurrently {
        return unsupported("CREATE INDEX CONCURRENTLY");
    }
    // `USING <method>` is accepted only for the `hnsw` vector access method; fold the
    // method name and reject anything else (the default — no `USING` — is a B-tree).
    let using = match &ci.using {
        None => None,
        Some(sql::IndexType::Custom(method)) if method.value.eq_ignore_ascii_case("hnsw") => {
            Some("hnsw".to_owned())
        },
        Some(_) => return unsupported("CREATE INDEX USING <method> (only `hnsw` is supported)"),
    };
    if ci.nulls_distinct.is_some() {
        return unsupported("CREATE INDEX ... NULLS [NOT] DISTINCT");
    }
    if !ci.with.is_empty() {
        return unsupported("CREATE INDEX ... WITH (...)");
    }
    if ci.columns.is_empty() {
        return unsupported("CREATE INDEX with no key columns");
    }
    let name = match ci.name {
        Some(n) => object_name(&n)?,
        None => return unsupported("CREATE INDEX without an index name"),
    };
    let (table_schema, table) = table_ref_name(&ci.table_name)?;
    // Each key is either a plain column or an expression (functional index). If EVERY key is a
    // plain column, this stays a plain-column index (`columns`); if any key is an expression, the
    // whole key list is carried as expression SQL (`key_exprs`) — a plain column in that list
    // becomes its own identifier expression — so mixed `(a, lower(b))` indexes are modelled too.
    let mut columns = Vec::with_capacity(ci.columns.len());
    let mut any_expr = false;
    for key in &ci.columns {
        match convert_index_key(key) {
            Ok(col) => columns.push(col),
            Err(Error::Unsupported(_)) if is_expression_key(key) => {
                any_expr = true;
                break;
            },
            Err(e) => return Err(e),
        }
    }
    let (columns, key_exprs) = if any_expr {
        let mut exprs = Vec::with_capacity(ci.columns.len());
        for key in &ci.columns {
            exprs.push(convert_index_key_expr(key)?);
        }
        (Vec::new(), exprs)
    } else {
        (columns, Vec::new())
    };
    // Partial-index predicate: stored as SQL text, re-parsed + evaluated per row on the write path
    // (the same round-trip a CHECK constraint uses).
    let predicate = ci.predicate.as_ref().map(ToString::to_string);
    let include = ci.include.iter().map(fold_ident).collect();
    Ok(ast::CreateIndex {
        name,
        table_schema,
        table,
        columns,
        key_exprs,
        predicate,
        include,
        using,
        unique: ci.unique,
        if_not_exists: ci.if_not_exists,
    })
}

/// Whether an index-key entry is a (non-column) expression — a functional/expression index key,
/// as opposed to a plain (possibly `ASC`/`DESC`-annotated) column identifier.
const fn is_expression_key(key: &sql::IndexColumn) -> bool {
    !matches!(
        key.column.expr,
        sql::Expr::Identifier(_) | sql::Expr::CompoundIdentifier(_)
    )
}

/// Render an index-key entry as expression SQL for a functional/expression index (`key_exprs`). A
/// plain column identifier renders as itself, so a mixed `(a, lower(b))` key list is uniform. An
/// operator class, `WITH FILL`, or a per-column `ASC`/`DESC`/`NULLS` annotation is rejected —
/// see [`reject_index_key_modifiers`].
fn convert_index_key_expr(key: &sql::IndexColumn) -> Result<String, Error> {
    reject_index_key_modifiers(key)?;
    Ok(key.column.expr.to_string())
}

/// Reject index-key modifiers the engine cannot honor: an operator class, `WITH FILL`, and
/// per-column `ASC`/`DESC` / `NULLS FIRST`/`NULLS LAST`. Direction and null-placement only matter
/// for an *ordered* index scan (walking the index to satisfy an `ORDER BY`), which the engine does
/// not do — so accepting the annotation and silently building an ascending index would be a trap
/// (`(a DESC)` indistinguishable from `(a ASC)`, `ORDER BY a DESC` still not accelerated, the
/// catalog recording no direction). Rejecting loudly is honest; a plain
/// `(a)` index still serves equality and range lookups. `NULLS DISTINCT` is handled by the caller.
fn reject_index_key_modifiers(key: &sql::IndexColumn) -> Result<(), Error> {
    if key.operator_class.is_some() {
        return unsupported("CREATE INDEX with an operator class on a key column");
    }
    if key.column.options.asc.is_some() {
        return unsupported(
            "CREATE INDEX with ASC/DESC on a key column (only ascending indexes are built; \
             ordered index scans are not implemented)",
        );
    }
    if key.column.options.nulls_first.is_some() {
        return unsupported("CREATE INDEX with NULLS FIRST/LAST on a key column");
    }
    if key.column.with_fill.is_some() {
        return unsupported("CREATE INDEX with WITH FILL on a key column");
    }
    Ok(())
}

/// Extract a plain column name from an index-key entry. Per-column `ASC`/`DESC`/`NULLS` and other
/// key modifiers are rejected by [`reject_index_key_modifiers`].
pub(super) fn convert_index_key(key: &sql::IndexColumn) -> Result<String, Error> {
    reject_index_key_modifiers(key)?;
    match &key.column.expr {
        sql::Expr::Identifier(ident) => Ok(fold_ident(ident)),
        sql::Expr::CompoundIdentifier(_) => {
            unsupported("CREATE INDEX with a qualified column reference")
        },
        _ => unsupported("CREATE INDEX with an expression key (functional index)"),
    }
}

// === CREATE SEQUENCE ======================================================

/// Lower `CREATE SEQUENCE [IF NOT EXISTS] name [options...]`, keeping the
/// recognised options in declaration order. `TEMPORARY`, `AS <type>`, and
/// `OWNED BY` are rejected.
pub(super) fn convert_create_sequence(
    temporary: bool,
    if_not_exists: bool,
    name: &sql::ObjectName,
    data_type: Option<&sql::DataType>,
    options: Vec<sql::SequenceOptions>,
    owned_by: Option<&sql::ObjectName>,
) -> Result<ast::CreateSequence, Error> {
    if temporary {
        return unsupported("CREATE TEMPORARY SEQUENCE");
    }
    if data_type.is_some() {
        return unsupported("CREATE SEQUENCE ... AS <type>");
    }
    if owned_by.is_some() {
        return unsupported("CREATE SEQUENCE ... OWNED BY");
    }
    let mut converted = Vec::with_capacity(options.len());
    for opt in options {
        converted.push(convert_sequence_option(opt)?);
    }
    Ok(ast::CreateSequence {
        name: object_name(name)?,
        if_not_exists,
        options: converted,
    })
}

/// Convert a single sequence option. The trailing `BY`/`WITH` keyword flags
/// sqlparser records are cosmetic and dropped.
pub(super) fn convert_sequence_option(
    opt: sql::SequenceOptions,
) -> Result<ast::SequenceOption, Error> {
    use sql::SequenceOptions as O;
    Ok(match opt {
        O::IncrementBy(expr, _by) => ast::SequenceOption::Increment(convert_expr(expr)?),
        O::MinValue(value) => ast::SequenceOption::MinValue(value.map(convert_expr).transpose()?),
        O::MaxValue(value) => ast::SequenceOption::MaxValue(value.map(convert_expr).transpose()?),
        O::StartWith(expr, _with) => ast::SequenceOption::Start(convert_expr(expr)?),
        O::Cache(expr) => ast::SequenceOption::Cache(convert_expr(expr)?),
        // sqlparser 0.51 records the CYCLE flag inverted — its bool is `true`
        // for `NO CYCLE` and `false` for `CYCLE`. Negate it so our AST's
        // `Cycle(true)` faithfully means the cycling form.
        O::Cycle(no_cycle) => ast::SequenceOption::Cycle(!no_cycle),
    })
}

// === TRUNCATE =============================================================

/// Lower `TRUNCATE [TABLE] name [RESTART IDENTITY | CONTINUE IDENTITY]`.
/// Only a single-table, non-partitioned, non-cascading form is modelled.
pub(super) fn convert_truncate(
    table_names: &[sql::TruncateTableTarget],
    has_partitions: bool,
    only: bool,
    identity: Option<&sql::TruncateIdentityOption>,
    cascade: Option<&sql::CascadeOption>,
) -> Result<ast::TruncateTable, Error> {
    if only {
        return unsupported("TRUNCATE TABLE ONLY");
    }
    if has_partitions {
        return unsupported("TRUNCATE TABLE with PARTITION");
    }
    // RESTRICT is the default semantic (refuse when FK-referenced — enforced at execution);
    // CASCADE (truncate referencing tables too) is not modelled.
    if cascade == Some(&sql::CascadeOption::Cascade) {
        return unsupported("TRUNCATE TABLE ... CASCADE");
    }
    let restart_identity = identity == Some(&sql::TruncateIdentityOption::Restart);
    match table_names {
        [one] => {
            // `TRUNCATE schema.t` resolves like any other table reference.
            let (schema, name) = table_ref_name(&one.name)?;
            Ok(ast::TruncateTable {
                schema,
                name,
                restart_identity,
            })
        },
        _ => unsupported("TRUNCATE of multiple tables in one statement"),
    }
}

// === CREATE SCHEMA ========================================================

/// Lower `CREATE SCHEMA [IF NOT EXISTS] name`. Only the plain named form is
/// modelled; the `AUTHORIZATION <role>` variants are rejected.
pub(super) fn convert_create_schema(
    schema_name: sql::SchemaName,
    if_not_exists: bool,
) -> Result<ast::CreateSchema, Error> {
    let name = match schema_name {
        sql::SchemaName::Simple(name) => object_name(&name)?,
        sql::SchemaName::UnnamedAuthorization(_) | sql::SchemaName::NamedAuthorization(..) => {
            return unsupported("CREATE SCHEMA ... AUTHORIZATION");
        },
    };
    Ok(ast::CreateSchema {
        name,
        if_not_exists,
    })
}

/// Lower `CREATE DATABASE [IF NOT EXISTS] name` to the single-database compatibility no-op (NusaDB
/// is one database per data dir; this accepts the statement so ecosystem scripts run unchanged).
pub(super) fn convert_create_database(
    name: &sql::ObjectName,
    if_not_exists: bool,
) -> Result<ast::CreateDatabase, Error> {
    Ok(ast::CreateDatabase {
        name: object_name(name)?,
        if_not_exists,
    })
}

// === CREATE VIEW ==========================================================

/// Lower the plain `CREATE [OR REPLACE] VIEW name [(columns)] AS <select>`
/// form. The dialect-specific modifiers NusaDB does not model are each
/// rejected up front with a message naming exactly what was refused; the view
/// body is then converted through the shared `SELECT` lowering.
///
/// The wide parameter list (and its bools) is a faithful 1:1 mirror of
/// sqlparser's `CreateView` variant — bundling the fields would only obscure
/// which option a given rejection refers to.
#[allow(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    reason = "1:1 mirror of sqlparser's wide CreateView option set; per-field checks keep the rejection messages specific"
)]
pub(super) fn convert_create_view(
    name: &sql::ObjectName,
    or_replace: bool,
    materialized: bool,
    columns: &[sql::ViewColumnDef],
    query: sql::Query,
    if_not_exists: bool,
    temporary: bool,
    with_no_schema_binding: bool,
    cluster_by: &[sql::Ident],
    comment: Option<&String>,
    to: Option<&sql::ObjectName>,
    options: &sql::CreateTableOptions,
) -> Result<ast::CreateView, Error> {
    if if_not_exists && or_replace {
        // The two conflict: REPLACE overwrites, IF NOT EXISTS skips.
        return unsupported("CREATE OR REPLACE VIEW combined with IF NOT EXISTS");
    }
    if temporary {
        return unsupported("CREATE TEMPORARY VIEW");
    }
    if with_no_schema_binding {
        return unsupported("CREATE VIEW ... WITH NO SCHEMA BINDING");
    }
    if !cluster_by.is_empty() {
        return unsupported("CREATE VIEW ... CLUSTER BY");
    }
    if comment.is_some() {
        return unsupported("CREATE VIEW ... COMMENT");
    }
    if to.is_some() {
        return unsupported("CREATE VIEW ... TO <table>");
    }
    if !matches!(options, sql::CreateTableOptions::None) {
        return unsupported("CREATE VIEW ... WITH (...)");
    }
    // Render the body back to canonical SQL before consuming it, so REFRESH can re-execute it.
    let definition_sql = query.to_string();
    Ok(ast::CreateView {
        name: object_name(name)?,
        or_replace,
        if_not_exists,
        materialized,
        columns: convert_view_columns(columns)?,
        query: Box::new(convert_select(query)?),
        definition_sql,
    })
}

/// Extract the optional explicit output column names. A typed or optioned view
/// column (`name TYPE`, `name OPTIONS (...)`) is rejected — only bare aliases
/// are modelled.
pub(super) fn convert_view_columns(columns: &[sql::ViewColumnDef]) -> Result<Vec<String>, Error> {
    let mut names = Vec::with_capacity(columns.len());
    for col in columns {
        if col.data_type.is_some() {
            return unsupported("CREATE VIEW with a typed output column");
        }
        if col.options.is_some() {
            return unsupported("CREATE VIEW with output-column options");
        }
        names.push(fold_ident(&col.name));
    }
    Ok(names)
}

// === ALTER TABLE ==========================================================

/// Lower an `ALTER TABLE` statement. Exactly one column action per statement
/// is modelled; `ONLY`, Hive `SET LOCATION`, and comma-separated multi-action
/// forms are rejected with [`Error::Unsupported`].
pub(super) fn convert_alter_table(
    name: &sql::ObjectName,
    if_exists: bool,
    only: bool,
    operations: Vec<sql::AlterTableOperation>,
    has_location: bool,
) -> Result<ast::AlterTable, Error> {
    if only {
        return unsupported("ALTER TABLE ONLY");
    }
    if has_location {
        return unsupported("ALTER TABLE ... SET LOCATION");
    }
    let mut ops = operations.into_iter();
    let action = match (ops.next(), ops.next()) {
        (Some(op), None) => convert_alter_op(op)?,
        (None, _) => return unsupported("ALTER TABLE with no action"),
        (Some(_), Some(_)) => {
            return unsupported("ALTER TABLE with multiple actions in one statement");
        },
    };
    // `ALTER TABLE schema.t` resolves like any other table reference: an explicit qualifier is
    // honored, a bare name walks the session search path.
    let (schema, name) = table_ref_name(name)?;
    Ok(ast::AlterTable {
        schema,
        name,
        if_exists,
        action,
    })
}

/// Convert a single `ALTER TABLE` operation. Only ADD/DROP/RENAME COLUMN are
/// in scope; column-type/default modifiers are tracked as
pub(super) fn convert_alter_op(
    op: sql::AlterTableOperation,
) -> Result<ast::AlterTableAction, Error> {
    use sql::AlterTableOperation as Op;
    match op {
        Op::AddColumn {
            if_not_exists,
            column_def,
            column_position,
            ..
        } => {
            if column_position.is_some() {
                return unsupported("ALTER TABLE ADD COLUMN ... FIRST|AFTER");
            }
            // `synth_type_checks = false`: ADD COLUMN cannot atomically add a table constraint, so a
            // narrow-int / VARCHAR(n) column is added without its desugared bound rather than being
            // rejected outright. A *user-written* column CHECK/REFERENCES is still lifted and refused.
            let (column, lifted) = convert_column_def(&column_def, false)?;
            // ALTER ADD COLUMN cannot also introduce a table-level CHECK/FK in one action.
            if !lifted.is_empty() {
                return unsupported(
                    "ALTER TABLE ADD COLUMN with an inline CHECK / REFERENCES constraint",
                );
            }
            Ok(ast::AlterTableAction::AddColumn {
                column,
                if_not_exists,
            })
        },
        Op::DropColumn {
            has_column_keyword: _,
            column_names,
            if_exists,
            drop_behavior,
        } => {
            if drop_behavior.is_some() {
                return unsupported("ALTER TABLE DROP COLUMN ... CASCADE / RESTRICT");
            }
            // sqlparser 0.62 models a multi-column drop form; only the single-column
            // form is in surface.
            let [column_name] = column_names.as_slice() else {
                return unsupported("ALTER TABLE DROP COLUMN with multiple columns");
            };
            Ok(ast::AlterTableAction::DropColumn {
                name: fold_ident(column_name),
                if_exists,
            })
        },
        Op::RenameColumn {
            old_column_name,
            new_column_name,
        } => Ok(ast::AlterTableAction::RenameColumn {
            from: fold_ident(&old_column_name),
            to: fold_ident(&new_column_name),
        }),
        // `RENAME TO` and the synonymous `RENAME AS` both carry the new table name.
        Op::RenameTable { table_name } => {
            let (sql::RenameTableNameKind::As(new_name) | sql::RenameTableNameKind::To(new_name)) =
                table_name;
            Ok(ast::AlterTableAction::RenameTable {
                name: object_name(&new_name)?,
            })
        },
        Op::AddConstraint {
            constraint,
            not_valid,
        } => {
            if not_valid {
                return unsupported("ALTER TABLE ADD CONSTRAINT ... NOT VALID");
            }
            Ok(ast::AlterTableAction::AddConstraint(
                convert_table_constraint(constraint)?,
            ))
        },
        Op::DropConstraint {
            if_exists,
            name,
            drop_behavior,
        } => {
            if drop_behavior.is_some() {
                return unsupported("ALTER TABLE DROP CONSTRAINT ... CASCADE / RESTRICT");
            }
            Ok(ast::AlterTableAction::DropConstraint {
                name: fold_ident(&name),
                if_exists,
            })
        },
        Op::AlterColumn { column_name, op } => Ok(ast::AlterTableAction::AlterColumn {
            column: fold_ident(&column_name),
            change: convert_column_change(op)?,
        }),
        // Row-level security toggles. `FORCE`/`NO FORCE ROW LEVEL SECURITY` are not modelled by
        // sqlparser 0.51 and are a follow-up.
        Op::EnableRowLevelSecurity => Ok(ast::AlterTableAction::EnableRowLevelSecurity),
        Op::DisableRowLevelSecurity => Ok(ast::AlterTableAction::DisableRowLevelSecurity),
        // Trigger toggles: `{ENABLE|DISABLE} TRIGGER {name|ALL}` — sqlparser parses `ALL` as a
        // plain identifier, so it is folded into the `None` (= every trigger) form here. `USER`
        // (all non-system triggers) is rejected: NusaDB has no system triggers to exclude, so
        // accepting it would silently mean something different from the reference behavior.
        Op::EnableTrigger { name } => Ok(ast::AlterTableAction::EnableTrigger {
            name: trigger_toggle_target(&name)?,
        }),
        Op::DisableTrigger { name } => Ok(ast::AlterTableAction::DisableTrigger {
            name: trigger_toggle_target(&name)?,
        }),
        Op::EnableAlwaysTrigger { .. } | Op::EnableReplicaTrigger { .. } => unsupported(
            "ALTER TABLE ENABLE ALWAYS/REPLICA TRIGGER (session_replication_role modes are not \
             supported; use ENABLE TRIGGER)",
        ),
        other => unsupported(&format!("ALTER TABLE operation `{other}`")),
    }
}

/// The target of an `{ENABLE|DISABLE} TRIGGER` toggle: `None` for `ALL`, the folded trigger name
/// otherwise. `USER` is rejected — NusaDB has no system triggers, so "all user triggers" would
/// silently mean the same as `ALL` while reading as if it excluded something.
fn trigger_toggle_target(name: &sql::Ident) -> Result<Option<String>, Error> {
    let folded = fold_ident(name);
    match folded.as_str() {
        "all" => Ok(None),
        "user" => unsupported("ALTER TABLE {ENABLE|DISABLE} TRIGGER USER"),
        _ => Ok(Some(folded)),
    }
}

/// Lower an `ALTER [COLUMN]` modification. `SET DATA TYPE ... USING <expr>`
/// (a conversion expression) and `ADD GENERATED ... AS IDENTITY` are rejected.
pub(super) fn convert_column_change(
    op: sql::AlterColumnOperation,
) -> Result<ast::ColumnChange, Error> {
    use sql::AlterColumnOperation as Op;
    match op {
        Op::SetNotNull => Ok(ast::ColumnChange::SetNotNull),
        Op::DropNotNull => Ok(ast::ColumnChange::DropNotNull),
        Op::SetDefault { value } => Ok(ast::ColumnChange::SetDefault {
            sql: value.to_string(),
            expr: convert_expr(value)?,
        }),
        Op::DropDefault => Ok(ast::ColumnChange::DropDefault),
        Op::SetDataType {
            data_type,
            using,
            // Whether the statement spelled `SET DATA TYPE` or the bare `TYPE` — cosmetic.
            had_set: _,
        } => {
            if using.is_some() {
                return unsupported("ALTER COLUMN SET DATA TYPE ... USING <expr>");
            }
            Ok(ast::ColumnChange::SetType(convert_data_type(&data_type)?))
        },
        Op::AddGenerated { .. } => unsupported("ALTER COLUMN ADD GENERATED ... AS IDENTITY"),
    }
}

/// Lower a table-level constraint. Only PRIMARY KEY / UNIQUE / FOREIGN KEY /
/// CHECK are modelled; index hints (`USING`, `KEY`/`INDEX` display, index
/// options) and deferrable-constraint characteristics are rejected.
pub(super) fn convert_table_constraint(
    c: sql::TableConstraint,
) -> Result<ast::TableConstraint, Error> {
    use sql::TableConstraint as T;
    match c {
        T::PrimaryKey(pk) => {
            reject_index_hints(
                pk.index_name.as_ref(),
                pk.index_type.as_ref(),
                &pk.index_options,
            )?;
            reject_characteristics(pk.characteristics.as_ref())?;
            Ok(ast::TableConstraint::PrimaryKey {
                name: pk.name.as_ref().map(fold_ident),
                columns: constraint_column_names(&pk.columns)?,
            })
        },
        T::Unique(u) => {
            if u.index_type_display != sql::KeyOrIndexDisplay::None {
                return unsupported("UNIQUE constraint with KEY|INDEX display");
            }
            // `NULLS DISTINCT` is the SQL default (NULL keys never conflict — exactly the
            // engine's unique-index semantic), so the explicit spelling is accepted;
            // `NULLS NOT DISTINCT` changes the semantics and stays refused.
            if u.nulls_distinct == sql::NullsDistinctOption::NotDistinct {
                return unsupported("UNIQUE constraint with NULLS NOT DISTINCT");
            }
            reject_index_hints(
                u.index_name.as_ref(),
                u.index_type.as_ref(),
                &u.index_options,
            )?;
            reject_characteristics(u.characteristics.as_ref())?;
            Ok(ast::TableConstraint::Unique {
                name: u.name.as_ref().map(fold_ident),
                columns: constraint_column_names(&u.columns)?,
            })
        },
        T::ForeignKey(fk) => {
            if fk.index_name.is_some() {
                return unsupported("FOREIGN KEY constraint with a named backing index");
            }
            if fk.match_kind.is_some() {
                return unsupported("FOREIGN KEY ... MATCH FULL/PARTIAL/SIMPLE");
            }
            reject_characteristics(fk.characteristics.as_ref())?;
            Ok(ast::TableConstraint::ForeignKey {
                name: fk.name.as_ref().map(fold_ident),
                columns: fk.columns.iter().map(fold_ident).collect(),
                foreign_table: object_name(&fk.foreign_table)?,
                referred_columns: fk.referred_columns.iter().map(fold_ident).collect(),
                on_delete: fk.on_delete.map(convert_referential_action),
                on_update: fk.on_update.map(convert_referential_action),
            })
        },
        T::Check(check) => {
            if check.enforced.is_some() {
                return unsupported("CHECK constraint with ENFORCED / NOT ENFORCED");
            }
            Ok(ast::TableConstraint::Check {
                name: check.name.as_ref().map(fold_ident),
                predicate_sql: check.expr.to_string(),
                expr: convert_expr(*check.expr)?,
            })
        },
        other => unsupported(&format!("table constraint `{other}`")),
    }
}

/// Extract plain column names from a PRIMARY KEY / UNIQUE constraint key list. sqlparser 0.62
/// models constraint columns as full index columns; ordering / operator-class / expression keys
/// are not modelled by the catalog, so anything but a bare identifier is rejected loudly.
fn constraint_column_names(columns: &[sql::IndexColumn]) -> Result<Vec<String>, Error> {
    columns
        .iter()
        .map(|key| {
            if key.operator_class.is_some()
                || key.column.options.asc.is_some()
                || key.column.options.nulls_first.is_some()
                || key.column.with_fill.is_some()
            {
                return unsupported("constraint key column with ordering / operator-class");
            }
            match &key.column.expr {
                sql::Expr::Identifier(ident) => Ok(fold_ident(ident)),
                _ => unsupported("constraint with an expression key column"),
            }
        })
        .collect()
}

/// Reject index-method hints attached to a PRIMARY KEY / UNIQUE constraint
/// (`USING <method>`, a named backing index, or trailing index options) — the
/// catalog does not model them yet.
pub(super) fn reject_index_hints(
    index_name: Option<&sql::Ident>,
    index_type: Option<&sql::IndexType>,
    index_options: &[sql::IndexOption],
) -> Result<(), Error> {
    if index_name.is_some() {
        return unsupported("constraint with a named backing index");
    }
    if index_type.is_some() {
        return unsupported("constraint with USING <method>");
    }
    if !index_options.is_empty() {
        return unsupported("constraint with index options");
    }
    Ok(())
}

/// Reject deferrable-constraint characteristics (`DEFERRABLE`, `INITIALLY ...`).
pub(super) fn reject_characteristics(
    characteristics: Option<&sql::ConstraintCharacteristics>,
) -> Result<(), Error> {
    if characteristics.is_some() {
        return unsupported("constraint characteristics (DEFERRABLE / INITIALLY ...)");
    }
    Ok(())
}

/// Map a sqlparser referential action onto the AST's own enum.
pub(super) const fn convert_referential_action(
    action: sql::ReferentialAction,
) -> ast::ReferentialAction {
    use sql::ReferentialAction as R;
    match action {
        R::Restrict => ast::ReferentialAction::Restrict,
        R::Cascade => ast::ReferentialAction::Cascade,
        R::SetNull => ast::ReferentialAction::SetNull,
        R::NoAction => ast::ReferentialAction::NoAction,
        R::SetDefault => ast::ReferentialAction::SetDefault,
    }
}

// === DROP TABLE / DROP INDEX ==============================================

/// Lower a `DROP` statement. `DROP <kind> a, b, ...` desugars into an internal
/// [`ast::Statement::Batch`] of single-object drops, executed in order within the one
/// statement transaction (so `DROP TABLE a, b` is atomic, per the standard); unsupported
/// object kinds are rejected with [`Error::Unsupported`].
pub(super) fn convert_drop(
    object_type: sql::ObjectType,
    if_exists: bool,
    cascade: bool,
    names: &[sql::ObjectName],
) -> Result<ast::Statement, Error> {
    if names.is_empty() {
        return unsupported("DROP with no object names");
    }
    let one = |name: &sql::ObjectName| -> Result<ast::Statement, Error> {
        match object_type {
            sql::ObjectType::Table => {
                let (schema, name) = table_ref_name(name)?;
                Ok(ast::Statement::DropTable(ast::DropTable {
                    schema,
                    name,
                    if_exists,
                    cascade,
                }))
            },
            sql::ObjectType::Index => Ok(ast::Statement::DropIndex(ast::DropIndex {
                name: object_name(name)?,
                if_exists,
            })),
            sql::ObjectType::View => Ok(ast::Statement::DropView(ast::DropView {
                name: object_name(name)?,
                if_exists,
            })),
            sql::ObjectType::Schema => Ok(ast::Statement::DropSchema(ast::DropSchema {
                name: object_name(name)?,
                if_exists,
                cascade,
            })),
            sql::ObjectType::Sequence => Ok(ast::Statement::DropSequence(ast::DropSequence {
                name: object_name(name)?,
                if_exists,
            })),
            _ => unsupported("DROP of an unsupported object kind"),
        }
    };
    match names {
        [single] => one(single),
        several => Ok(ast::Statement::Batch(
            several.iter().map(one).collect::<Result<Vec<_>, _>>()?,
        )),
    }
}
