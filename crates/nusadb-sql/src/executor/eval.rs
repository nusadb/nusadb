//! Evaluator for [`TypedExpr`] over a single [`Row`].
//!
//! Evaluation follows SQL's three-valued logic: `NULL` propagates through
//! arithmetic and comparison; `AND`/`OR` use the standard truth tables
//! (`FALSE AND NULL = FALSE`, `TRUE OR NULL = TRUE`, everything else with a
//! `NULL` operand is `NULL`).
//!
//! All type-compatibility checks were performed by the analyzer, so a wrong
//! operand type at this layer indicates a planner/analyzer bug rather than a
//! user error — the defensive fallback is [`ast::Value::Null`] so an internal
//! mismatch never panics.

#![allow(
    clippy::cast_precision_loss,
    reason = "intentional Int->Float widening for mixed-numeric arithmetic/comparison"
)]
#![allow(
    clippy::float_cmp,
    reason = "exact `== 0.0` is the canonical division-by-zero guard"
)]

use std::cell::RefCell;
use std::cmp::Ordering;

use nusadb_core::ColumnType;

use crate::ast;
use crate::error::Error;
use crate::executor::row::Row;
use crate::planner::{TypedCaseBranch, TypedExpr, TypedExprKind};

thread_local! {
    /// Stack of enclosing-query rows bound while a correlated subquery runs. The executor
    /// pushes the current outer row before evaluating a correlated subquery; an
    /// [`TypedExprKind::OuterColumn`] with `level = n` reads the `n`-th frame from the top.
    static OUTER_ROWS: RefCell<Vec<Row>> = const { RefCell::new(Vec::new()) };
}

/// Bind `row` as the innermost enclosing-query row for the lifetime of the returned guard, so a
/// correlated subquery evaluated within it can read it through [`TypedExprKind::OuterColumn`].
/// Restored (popped) on drop, so nested correlations stay aligned.
#[must_use]
pub(crate) fn bind_outer_row(row: Row) -> OuterRowGuard {
    OUTER_ROWS.with(|stack| stack.borrow_mut().push(row));
    OuterRowGuard
}

/// Pops the outer row bound by [`bind_outer_row`] on drop.
#[derive(Debug)]
pub(crate) struct OuterRowGuard;

impl Drop for OuterRowGuard {
    fn drop(&mut self) {
        OUTER_ROWS.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

/// Read an `OuterColumn` from the bound outer-row stack: `level = 1` is the innermost enclosing
/// row. Returns `NULL` defensively if the stack is shorter than `level` or the ordinal is out of
/// range (an analyzer/executor bug rather than a user error).
fn outer_column(level: usize, ordinal: usize) -> ast::Value {
    OUTER_ROWS.with(|stack| {
        let stack = stack.borrow();
        level
            .checked_sub(1)
            .and_then(|back| stack.len().checked_sub(1)?.checked_sub(back))
            .and_then(|frame| stack.get(frame))
            .and_then(|row| row.get(ordinal))
            .cloned()
            .unwrap_or(ast::Value::Null)
    })
}

/// Evaluate `expr` against `row`. The row must have one entry per column of
/// the operator's source schema; the expression's `Column(i)` nodes index
/// directly into it.
#[allow(
    clippy::too_many_lines,
    reason = "flat one-arm-per-expression-kind dispatch; length tracks the expression grammar"
)]
pub(crate) fn eval(expr: &TypedExpr, row: &Row) -> Result<ast::Value, Error> {
    match &expr.kind {
        TypedExprKind::Literal(v) => Ok(v.clone()),
        TypedExprKind::Column(index) => row
            .get(*index)
            .cloned()
            .ok_or(Error::MalformedTuple { offset: *index }),
        TypedExprKind::OuterColumn { level, ordinal } => Ok(outer_column(*level, *ordinal)),
        // A set-returning function yields multiple rows and is expanded by the `ProjectSet` operator;
        // it must never reach the scalar evaluator.
        TypedExprKind::SetReturning { .. } => Err(Error::Unsupported(
            "set-returning function cannot be evaluated as a scalar value".to_owned(),
        )),
        TypedExprKind::Binary { left, op, right } => {
            let l = eval(left, row)?;
            let r = eval(right, row)?;
            apply_binary(*op, &l, &r, expr.ty)
        },
        TypedExprKind::Unary { op, expr: inner } => {
            let v = eval(inner, row)?;
            apply_unary(*op, &v)
        },
        TypedExprKind::IsNull {
            expr: inner,
            negated,
        } => {
            let v = eval(inner, row)?;
            let is_null = matches!(v, ast::Value::Null);
            Ok(ast::Value::Bool(if *negated { !is_null } else { is_null }))
        },
        TypedExprKind::IsDistinctFrom {
            left,
            right,
            negated,
        } => {
            let l = eval(left, row)?;
            let r = eval(right, row)?;
            Ok(apply_is_distinct_from(&l, &r, *negated))
        },
        TypedExprKind::IsBool {
            expr: inner,
            truth,
            negated,
        } => {
            let v = eval(inner, row)?;
            Ok(apply_is_bool(&v, *truth, *negated))
        },
        TypedExprKind::InList {
            expr: inner,
            list,
            negated,
        } => eval_in_list(inner, list, *negated, row),
        TypedExprKind::Between {
            expr: inner,
            low,
            high,
            negated,
        } => eval_between(inner, low, high, *negated, row),
        TypedExprKind::Like {
            expr: inner,
            pattern,
            negated,
            escape,
            case_insensitive,
        } => eval_like(inner, pattern, *negated, *escape, *case_insensitive, row),
        TypedExprKind::RegexMatch {
            expr: inner,
            pattern,
            case_sensitive,
            negated,
        } => eval_regex_match(inner, pattern, *case_sensitive, *negated, row),
        TypedExprKind::SimilarTo {
            expr: inner,
            pattern,
            negated,
        } => eval_similar_to(inner, pattern, *negated, row),
        TypedExprKind::Case {
            operand,
            branches,
            default,
        } => Ok(coerce_numeric_to(
            eval_case(operand.as_deref(), branches, default.as_deref(), row)?,
            expr.ty,
        )),
        TypedExprKind::Coalesce(args) => Ok(coerce_numeric_to(eval_coalesce(args, row)?, expr.ty)),
        TypedExprKind::ArrayLiteral(elems) => {
            let mut values = Vec::with_capacity(elems.len());
            for elem in elems {
                values.push(eval(elem, row)?);
            }
            Ok(ast::Value::Array(values))
        },
        TypedExprKind::Subscript { base, index } => eval_subscript(base, index, row),
        TypedExprKind::ArraySlice { base, lower, upper } => {
            eval_array_slice(base, lower.as_deref(), upper.as_deref(), row)
        },
        TypedExprKind::Cast(inner, try_cast) => eval_cast(inner, expr.ty, row, *try_cast),
        TypedExprKind::Crypto { op, value, key } => eval_crypto(*op, value, key, row),
        // NULLIF/GREATEST/LEAST return one argument's value verbatim, but their declared type is
        // the *unified* numeric type of all arguments — coerce like CASE/COALESCE so the value
        // variant matches the type. Scoped to exactly these three: the other scalar
        // functions construct their result in the declared type already.
        TypedExprKind::ScalarFunction { func, args }
            if matches!(
                func,
                ast::ScalarFunc::Nullif | ast::ScalarFunc::Greatest | ast::ScalarFunc::Least
            ) =>
        {
            Ok(coerce_numeric_to(
                eval_scalar_function(*func, args, row)?,
                expr.ty,
            ))
        },
        TypedExprKind::ScalarFunction { func, args } => eval_scalar_function(*func, args, row),
        // A registered scalar UDF: evaluate each argument, coerce it to the UDF's declared
        // parameter type, then invoke the Rust function. The analyzer only checks *assignability*
        // (e.g. an INT argument may reach a FLOAT parameter, or a NUMERIC literal a FLOAT one), so the
        // executor must coerce here to honour the `ScalarUdfFn` contract that values arrive already in
        // the declared types (deep-gate). The declared types were captured into the plan node at
        // analysis time, so no per-row registry read is needed. `cast_value` passes `NULL` through.
        TypedExprKind::ScalarUdf {
            name,
            args,
            arg_types,
        } => {
            let mut values = Vec::with_capacity(args.len());
            for (arg, want) in args.iter().zip(arg_types) {
                values.push(cast_value(eval(arg, row)?, *want)?);
            }
            crate::udf::call_scalar_udf(name, &values)
        },
        // `Project` runs over the single row produced by `ScalarAggregate`, whose column `slot`
        // holds the computed value for that aggregate call.
        TypedExprKind::AggregateRef(slot) => row
            .get(*slot)
            .cloned()
            .ok_or(Error::MalformedTuple { offset: *slot }),
        // `expr <op> ANY/ALL (array)`: fold `probe <op> element` over the array's elements — `ANY`
        // with OR (identity FALSE), `ALL` with AND (identity TRUE) — so an empty array yields the
        // identity and a NULL comparison propagates via 3-valued logic. A NULL / non-array right
        // operand yields NULL (as `x = ANY(NULL)` does in the reference engine).
        TypedExprKind::QuantifiedArray {
            expr,
            op,
            all,
            array,
        } => {
            let probe = eval(expr, row)?;
            let ast::Value::Array(elems) = eval(array, row)? else {
                return Ok(ast::Value::Null);
            };
            let mut acc = ast::Value::Bool(*all);
            for elem in &elems {
                let cmp = apply_comparison(*op, &probe, elem);
                acc = if *all {
                    apply_and(&acc, &cmp)
                } else {
                    apply_or(&acc, &cmp)
                };
            }
            Ok(acc)
        },
        // Subqueries are pre-resolved to literals by `resolve_subqueries` before any row is
        // evaluated. One surviving here sits in a position the executor does not
        // yet pre-resolve (e.g. GROUP BY) — reject honestly rather than mis-evaluate.
        TypedExprKind::ScalarSubquery(_)
        | TypedExprKind::Exists { .. }
        | TypedExprKind::InSubquery { .. }
        | TypedExprKind::QuantifiedSubquery { .. } => Err(Error::Unsupported(
            "a subquery in this position is not yet supported (only WHERE / SELECT / HAVING / \
             JOIN ON / ORDER BY)"
                .to_owned(),
        )),
    }
}

/// Evaluate `encrypt(value, key)` / `decrypt(value, key)`. A `NULL` value or key
/// yields `NULL` (SQL-style propagation); otherwise both are `Text` (the
/// analyzer guaranteed it) and the result is the hex ciphertext or recovered
/// plaintext.
fn eval_crypto(
    op: crate::planner::CryptoOp,
    value: &TypedExpr,
    key: &TypedExpr,
    row: &Row,
) -> Result<ast::Value, Error> {
    use crate::planner::CryptoOp;
    let (ast::Value::Text(value), ast::Value::Text(key)) = (eval(value, row)?, eval(key, row)?)
    else {
        // A NULL operand (or any non-text, which the analyzer rules out) → NULL.
        return Ok(ast::Value::Null);
    };
    let out = match op {
        CryptoOp::Encrypt => super::crypto::encrypt(&value, &key)?,
        CryptoOp::Decrypt => super::crypto::decrypt(&value, &key)?,
    };
    Ok(ast::Value::Text(out))
}

/// Which side(s) a [`pad`] / [`trim`] helper acts on.
#[derive(Clone, Copy)]
enum PadSide {
    Left,
    Right,
}

/// Which side(s) a [`trim`] helper strips.
#[derive(Clone, Copy)]
enum TrimSide {
    Left,
    Right,
    Both,
}

/// Evaluate a scalar built-in function. Most functions are NULL-strict: every
/// argument is evaluated and a `NULL` in any of them yields `NULL`. The variadic `CONCAT`/`CONCAT_WS`
/// are the exception — they skip `NULL` arguments and are dispatched first. The analyzer guarantees
/// arity and argument types, so an unmatched shape here signals an internal bug and falls back to
/// `NULL` rather than panicking (consistent with this module's defensive contract).
#[allow(
    clippy::too_many_lines,
    reason = "one arm per built-in function; a flat dispatch table is clearer than splitting it"
)]
fn eval_scalar_function(
    func: ast::ScalarFunc,
    args: &[TypedExpr],
    row: &Row,
) -> Result<ast::Value, Error> {
    use ast::ScalarFunc as F;
    use ast::Value::{Int, Text};
    // The niladic clock built-ins take no arguments — resolve them from the statement's
    // pinned wall clock before the argument-collection below (which expects at least one value).
    match func {
        // GROUPING(...) is rewritten by the analyzer against the query's grouping sets before
        // evaluation; reaching the per-row evaluator means it was used without GROUP BY.
        F::Grouping => {
            return Err(Error::Unsupported(
                "GROUPING is only allowed in an aggregated query with GROUP BY".to_owned(),
            ));
        },
        F::Now | F::CurrentTimestamp => {
            return Ok(ast::Value::TimestampTz(super::clock::statement_now_micros()));
        },
        F::CurrentDate => return Ok(ast::Value::Date(super::clock::statement_today())),
        F::CurrentTime => return Ok(ast::Value::Time(super::clock::statement_time_of_day())),
        // The niladic session-user built-ins read the statement's pinned session user.
        F::CurrentUser | F::SessionUser => {
            return Ok(Text(super::session_ctx::current_user()));
        },
        // Niladic system built-ins: read the pinned session context.
        F::Version => {
            return Ok(Text(format!("NusaDB {}", env!("CARGO_PKG_VERSION"))));
        },
        F::CurrentDatabase => {
            return Ok(Text(super::session_ctx::current_database()));
        },
        F::CurrentSchema => {
            return Ok(Text(super::session_ctx::current_schema()));
        },
        // UUID generator: a fresh random v4 UUID per call. Entropy comes from the OS
        // CSPRNG (`getrandom`), deliberately *not* the seedable `RANDOM()` PRNG (`super::rng`), so
        // generated UUIDs stay unpredictable and unique even after `SETSEED`. RFC 4122 §4.4: 16
        // random bytes with the version nibble pinned to 4 and the variant bits to `10`. An entropy
        // failure is surfaced as an honest error rather than a panic.
        F::UuidGenerateV4 => {
            let mut bytes = [0u8; 16];
            getrandom::fill(&mut bytes).map_err(|e| {
                Error::Core(nusadb_core::Error::Io(std::io::Error::other(format!(
                    "entropy failure generating UUID: {e}"
                ))))
            })?;
            bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
            bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10xx (RFC 4122)
            return Ok(ast::Value::Uuid(bytes));
        },
        // Sequence built-ins are side-effecting / session-stateful, so they are resolved against the
        // engine BEFORE the per-row loop (`resolve_sequence_calls`), where the executor can guarantee
        // exactly one evaluation. Reaching the pure per-row evaluator means the call sits in a context
        // that evaluates it per row (a scan projection, WHERE, UPDATE assignment) — which would
        // silently under-advance the sequence — so reject it loudly rather than return a wrong value.
        F::SequenceNext | F::SequenceCurrent | F::SequenceSet => {
            return Err(Error::Unsupported(format!(
                "{}() is only supported where it is evaluated once — a SELECT without FROM or a \
                 VALUES row; it cannot be used in a per-row query (scan, WHERE, or UPDATE)",
                func.name()
            )));
        },
        // CONCAT / CONCAT_WS skip NULL arguments instead of propagating NULL, so they cannot use
        // the NULL-strict collection below.
        F::Concat => return eval_concat(args, row),
        F::ConcatWs => return eval_concat_ws(args, row),
        // Conditional functions have their own NULL semantics: NULLIF is NULL-aware, and
        // GREATEST/LEAST skip NULL arguments rather than propagating.
        F::Nullif => return eval_nullif(args, row),
        F::Greatest => return eval_greatest_least(args, row, true),
        F::Least => return eval_greatest_least(args, row, false),
        // TO_JSON / JSON_BUILD_OBJECT serialize their arguments to JSON; a NULL argument becomes JSON
        // `null` rather than propagating NULL, so they skip the NULL-strict collection below.
        F::ToJson => return eval_to_json(args, row),
        F::RowToJson => return eval_row_to_json(args, row),
        F::JsonBuildObject => return eval_json_build_object(args, row),
        F::JsonBuildArray => return eval_json_build_array(args, row),
        // ARRAY_APPEND/PREPEND/CAT/POSITION/POSITIONS/REMOVE do not propagate NULL (a NULL element is
        // stored / searched, a NULL array operand is empty / NULL), so they skip the NULL-strict
        // collection. ARRAY_REPLACE likewise treats a NULL `from`/`to` as a real element.
        F::ArrayAppend
        | F::ArrayPrepend
        | F::ArrayCat
        | F::ArrayPosition
        | F::ArrayPositions
        | F::ArrayRemove => {
            return eval_array_mutate(func, args, row);
        },
        F::ArrayReplace => return eval_array_replace(args, row),
        // FORMAT substitutes its arguments into specifiers itself (a NULL renders per specifier, not
        // by propagating), so it skips the NULL-strict collection (B-fn).
        F::Format => return eval_format(args, row),
        // NUM_NONNULLS / NUM_NULLS count their arguments by NULL-ness — they inspect NULLs rather
        // than propagating them, so they skip the NULL-strict collection below.
        F::NumNonNulls => return eval_num_nulls(args, row, false),
        F::NumNulls => return eval_num_nulls(args, row, true),
        _ => {},
    }
    let mut vals = Vec::with_capacity(args.len());
    for arg in args {
        let v = eval(arg, row)?;
        if matches!(v, ast::Value::Null) {
            return Ok(ast::Value::Null);
        }
        vals.push(v);
    }
    Ok(match (func, vals.as_slice()) {
        (F::Length, [Text(s)]) => Int(char_len(s)),
        // Over BYTEA, LENGTH == OCTET_LENGTH == the byte count; BIT_LENGTH is 8x.
        (F::Length | F::OctetLength, [ast::Value::Bytes(b)]) => {
            Int(i64::try_from(b.len()).unwrap_or(i64::MAX))
        },
        (F::BitLength, [ast::Value::Bytes(b)]) => {
            Int(i64::try_from(b.len().saturating_mul(8)).unwrap_or(i64::MAX))
        },
        // OCTET_LENGTH = UTF-8 byte count; BIT_LENGTH = 8× that.
        (F::OctetLength, [Text(s)]) => Int(i64::try_from(s.len()).unwrap_or(i64::MAX)),
        (F::BitLength, [Text(s)]) => {
            Int(i64::try_from(s.len()).map_or(i64::MAX, |n| n.saturating_mul(8)))
        },
        (F::Upper, [Text(s)]) => Text(s.to_uppercase()),
        (F::Lower, [Text(s)]) => Text(s.to_lowercase()),
        (F::Sha256, [Text(s)]) => Text(super::crypto::sha256_hex(s)),
        (F::Sha512, [Text(s)]) => Text(super::crypto::sha512_hex(s)),
        (F::Md5, [Text(s)]) => Text(super::crypto::md5_hex(s)),
        (F::Substring, [Text(s), Int(start)]) => substring(s, *start, None)?,
        // The POSIX-regex form `substring(s FROM 'pattern')` — dispatched by
        // the analyzer when the second argument types TEXT.
        (F::Substring, [Text(s), Text(pattern)]) => substring_regex(s, pattern)?,
        (F::Substring, [Text(s), Int(start), Int(len)]) => substring(s, *start, Some(*len))?,
        (F::Overlay, [Text(s), Text(r), Int(start)]) => Text(overlay(s, r, *start, None)),
        (F::Overlay, [Text(s), Text(r), Int(start), Int(len)]) => {
            Text(overlay(s, r, *start, Some(*len)))
        },
        (F::Replace, [Text(s), Text(from), Text(to)]) => Text(replace(s, from, to)),
        (F::Position, [Text(sub), Text(hay)]) => Int(position(sub, hay)),
        (F::Lpad, [Text(s), Int(len)]) => Text(pad(s, *len, " ", PadSide::Left)),
        (F::Lpad, [Text(s), Int(len), Text(fill)]) => Text(pad(s, *len, fill, PadSide::Left)),
        (F::Rpad, [Text(s), Int(len)]) => Text(pad(s, *len, " ", PadSide::Right)),
        (F::Rpad, [Text(s), Int(len), Text(fill)]) => Text(pad(s, *len, fill, PadSide::Right)),
        (F::LTrim, [Text(s)]) => Text(trim(s, None, TrimSide::Left)),
        (F::LTrim, [Text(s), Text(set)]) => Text(trim(s, Some(set), TrimSide::Left)),
        (F::RTrim, [Text(s)]) => Text(trim(s, None, TrimSide::Right)),
        (F::RTrim, [Text(s), Text(set)]) => Text(trim(s, Some(set), TrimSide::Right)),
        (F::BTrim, [Text(s)]) => Text(trim(s, None, TrimSide::Both)),
        (F::BTrim, [Text(s), Text(set)]) => Text(trim(s, Some(set), TrimSide::Both)),
        (F::Left, [Text(s), Int(n)]) => Text(left(s, *n)),
        (F::Right, [Text(s), Int(n)]) => Text(right(s, *n)),
        (F::SplitPart, [Text(s), Text(delim), Int(n)]) => Text(split_part(s, delim, *n)),
        (F::Reverse, [Text(s)]) => Text(reverse(s)),
        (F::QuoteLiteral, [Text(s)]) => Text(quote_literal(s)),
        (F::QuoteIdent, [Text(s)]) => Text(quote_ident(s)),
        (F::StartsWith, [Text(s), Text(prefix)]) => {
            ast::Value::Bool(s.starts_with(prefix.as_str()))
        },
        (F::Ascii, [Text(s)]) => Int(s.chars().next().map_or(0, |c| i64::from(u32::from(c)))),
        (F::Chr, [Int(n)]) => char::from_u32(u32::try_from(*n).unwrap_or(u32::MAX))
            .map_or(ast::Value::Null, |c| Text(c.to_string())),
        (F::Initcap, [Text(s)]) => Text(initcap(s)),
        (F::Repeat, [Text(s), Int(n)]) => Text(s.repeat(usize::try_from(*n).unwrap_or(0))),
        // STRPOS(s, sub) is POSITION(sub IN s) with the haystack-first argument order.
        (F::Strpos, [Text(haystack), Text(needle)]) => Int(position(needle, haystack)),
        (F::Translate, [Text(s), Text(from), Text(to)]) => Text(translate(s, from, to)),
        (F::RegexpReplace, [Text(s), Text(pat), Text(repl)]) => {
            Text(regexp_replace(s, pat, repl, "")?)
        },
        (F::RegexpReplace, [Text(s), Text(pat), Text(repl), Text(flags)]) => {
            Text(regexp_replace(s, pat, repl, flags)?)
        },
        (F::RegexpMatch, [Text(s), Text(pat)]) => regexp_match(s, pat, "")?,
        (F::RegexpMatch, [Text(s), Text(pat), Text(flags)]) => regexp_match(s, pat, flags)?,
        (F::RegexpLike, [Text(s), Text(pat)]) => ast::Value::Bool(regexp_like(s, pat, "")?),
        (F::RegexpLike, [Text(s), Text(pat), Text(flags)]) => {
            ast::Value::Bool(regexp_like(s, pat, flags)?)
        },
        (F::RegexpCount, [Text(s), Text(pat)]) => Int(regexp_count(s, pat, "")?),
        (F::RegexpCount, [Text(s), Text(pat), Text(flags)]) => Int(regexp_count(s, pat, flags)?),
        (F::RegexpInstr, [Text(s), Text(pat)]) => Int(regexp_instr(s, pat, "")?),
        (F::RegexpInstr, [Text(s), Text(pat), Text(flags)]) => Int(regexp_instr(s, pat, flags)?),
        (F::RegexpSubstr, [Text(s), Text(pat)]) => regexp_substr(s, pat, "")?,
        (F::RegexpSubstr, [Text(s), Text(pat), Text(flags)]) => regexp_substr(s, pat, flags)?,
        (F::RegexpSplitToArray, [Text(s), Text(pat)]) => regexp_split_to_array(s, pat, "")?,
        (F::RegexpSplitToArray, [Text(s), Text(pat), Text(flags)]) => {
            regexp_split_to_array(s, pat, flags)?
        },
        // JSON inspection: a JSON value arrives as `Json`, or `Text` for a coerced literal.
        (F::JsonTypeof, [ast::Value::Json(s) | Text(s)]) => {
            crate::json::type_name(s).map_or(ast::Value::Null, |t| Text(t.to_owned()))
        },
        (F::JsonbStripNulls, [ast::Value::Json(s) | Text(s)]) => {
            crate::json::strip_nulls(s).map_or(ast::Value::Null, ast::Value::Json)
        },
        (F::JsonbPretty, [ast::Value::Json(s) | Text(s)]) => {
            crate::json::pretty(s).map_or(ast::Value::Null, Text)
        },
        // JSONB_EXISTS(json, key) → the `?` operator as a function (the tokenizer reserves `?` for
        // parameters): TRUE if `key` is a top-level object key / array string element / scalar string.
        (F::JsonbExists, [ast::Value::Json(s) | Text(s), Text(key)]) => {
            ast::Value::Bool(crate::json::has_key(s, key))
        },
        // Full-text search (F1): the two-argument forms name their configuration explicitly; the
        // one-argument forms use the default configuration, `english` — the same default as the reference engine's
        // stock `default_text_search_config`.
        (F::ToTsvector, [Text(config), Text(text)]) => Text(crate::fts::to_tsvector(config, text)?),
        (F::ToTsquery, [Text(config), Text(text)]) => Text(crate::fts::to_tsquery(config, text)?),
        (F::PlaintoTsquery, [Text(config), Text(text)]) => {
            Text(crate::fts::plainto_tsquery(config, text)?)
        },
        (F::ToTsvector, [Text(text)]) => Text(crate::fts::to_tsvector("english", text)?),
        (F::ToTsquery, [Text(text)]) => Text(crate::fts::to_tsquery("english", text)?),
        (F::PlaintoTsquery, [Text(text)]) => Text(crate::fts::plainto_tsquery("english", text)?),
        // Relevance ranking: the two-argument forms use the reference engine's default normalization (0); the
        // optional third argument is the normalization bit-mask. The score is a `real` (float4).
        (F::TsRank, [Text(v), Text(q)]) => real_value(crate::fts::ts_rank(v, q, 0)?),
        (F::TsRank, [Text(v), Text(q), Int(m)]) => {
            real_value(crate::fts::ts_rank(v, q, *m as i32)?)
        },
        (F::TsRankCd, [Text(v), Text(q)]) => real_value(crate::fts::ts_rank_cd(v, q, 0)?),
        (F::TsRankCd, [Text(v), Text(q), Int(m)]) => {
            real_value(crate::fts::ts_rank_cd(v, q, *m as i32)?)
        },
        // RRF_SCORE(rank [, k]) — the Reciprocal Rank Fusion contribution 1/(k + rank), with k
        // defaulting to 60, the standard constant. Each rank comes from a RANK() window
        // over one ranked list; summing the contributions fuses the lists (FTS + vector hybrid
        // search). Out-of-domain inputs are loud errors, not silent inf/NaN.
        (F::RrfScore, [Int(rank)]) => rrf_score(*rank, 60)?,
        (F::RrfScore, [Int(rank), Int(k)]) => rrf_score(*rank, *k)?,
        // JSONB_PATH_EXISTS(json, path) → TRUE if the jsonpath matches anywhere; an unsupported or
        // invalid path is a runtime error (mirroring JSONB_PATH_QUERY).
        (F::JsonbPathExists, [ast::Value::Json(s) | Text(s), Text(path)]) => {
            let matches = crate::json::path_query(s, path).ok_or_else(|| {
                Error::Unsupported(format!(
                    "jsonb_path_exists: unsupported or invalid jsonpath `{path}`"
                ))
            })?;
            ast::Value::Bool(!matches.is_empty())
        },
        // JSONB_PATH_QUERY_FIRST(json, path) → the first match as JSON, or NULL; an unsupported or
        // invalid path is a runtime error (mirroring JSONB_PATH_QUERY).
        (F::JsonbPathQueryFirst, [ast::Value::Json(s) | Text(s), Text(path)]) => {
            let matches = crate::json::path_query(s, path).ok_or_else(|| {
                Error::Unsupported(format!(
                    "jsonb_path_query_first: unsupported or invalid jsonpath `{path}`"
                ))
            })?;
            matches
                .into_iter()
                .next()
                .map_or(ast::Value::Null, ast::Value::Json)
        },
        (F::JsonArrayLength, [ast::Value::Json(s) | Text(s)]) => {
            crate::json::array_length(s).map_or(ast::Value::Null, Int)
        },
        // JSONB_SET(target, path, new_value [, create_missing]).
        (F::JsonbSet, [target, ast::Value::Array(path), new]) => json_set(target, path, new, true),
        (
            F::JsonbSet,
            [
                target,
                ast::Value::Array(path),
                new,
                ast::Value::Bool(create),
            ],
        ) => json_set(target, path, new, *create),
        // JSONB_INSERT(target, path, new_value [, insert_after]) — insert without overwriting.
        (F::JsonbInsert, [target, ast::Value::Array(path), new]) => {
            json_insert(target, path, new, false)
        },
        (
            F::JsonbInsert,
            [
                target,
                ast::Value::Array(path),
                new,
                ast::Value::Bool(after),
            ],
        ) => json_insert(target, path, new, *after),
        // `current_setting(name)` → the session setting's value, or NULL if it is unset.
        (F::CurrentSetting, [Text(name)]) => {
            super::session_ctx::setting(name).map_or(ast::Value::Null, Text)
        },
        // Date/time functions. The field is the leading Text literal the analyzer attached.
        (F::Extract, [Text(field), src]) => extract_value(field, src)?,
        (F::DateTrunc, [Text(field), src]) => date_trunc_value(field, src)?,
        (F::Age, [end]) => age_value(end, None)?,
        (F::Age, [end, start]) => age_value(end, Some(start))?,
        (F::AtTimeZone, [value, zone]) => at_time_zone_value(value, zone)?,
        // TO_CHAR formats a temporal value or a number; TO_DATE/TO_TIMESTAMP parse text.
        (
            F::ToChar,
            [
                src @ (Int(_) | ast::Value::Float(_) | ast::Value::Numeric(_)),
                Text(fmt),
            ],
        ) => to_char_numeric(src, fmt)?,
        (F::ToChar, [src, Text(fmt)]) => to_char_value(src, fmt),
        (F::ToDate, [Text(s), Text(fmt)]) => to_date_value(s, fmt)?,
        (F::ToTimestamp, [Text(s), Text(fmt)]) => to_timestamp_value(s, fmt)?,
        // `to_timestamp(epoch_seconds)` → TIMESTAMPTZ (QA category-D).
        (F::ToTimestamp, [epoch]) => to_timestamp_epoch(epoch)?,
        // `to_number(text, format)` → NUMERIC: read the digits/sign/decimal point (B-fn).
        (F::ToNumber, [Text(s), Text(_fmt)]) => to_number_value(s)?,
        // MAKE_DATE(year, month, day) → DATE, erroring on a non-existent calendar day.
        (F::MakeDate, [Int(y), Int(m), Int(d)]) => crate::temporal::make_date(*y, *m, *d)
            .map(ast::Value::Date)
            .ok_or_else(|| {
                Error::Unsupported(format!("make_date(): {y}-{m}-{d} is not a valid date"))
            })?,
        // MAKE_TIME(h, mi, sec) — seconds is double precision, so fractional seconds are kept as
        // microseconds. The seconds field, rounded to whole microseconds, must be in
        // `[0, 60s)`; the hour/minute fields are range-checked by `make_time(_, _, 0)`.
        (F::MakeTime, [Int(h), Int(mi), secs]) => {
            let s = to_f64(secs);
            let base = crate::temporal::make_time(*h, *mi, 0).ok_or_else(|| {
                Error::Unsupported(format!("make_time(): {h}:{mi}:{s} is not a valid time"))
            })?;
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the seconds field is bounds-checked to [0, 60_000_000) micros below"
            )]
            let sec_micros = (s * 1_000_000.0).round() as i64;
            if !(0..60_000_000).contains(&sec_micros) {
                return Err(Error::Unsupported(format!(
                    "make_time(): seconds {s} is out of range [0, 60)"
                )));
            }
            ast::Value::Time(base + sec_micros)
        },
        // MAKE_TIMESTAMP(y, mo, d, h, mi, s) → TIMESTAMP, erroring on invalid fields.
        (F::MakeTimestamp, [Int(y), Int(mo), Int(d), Int(h), Int(mi), Int(s)]) => {
            crate::temporal::make_timestamp(*y, *mo, *d, *h, *mi, *s)
                .map(ast::Value::Timestamp)
                .ok_or_else(|| {
                    Error::Unsupported(format!(
                        "make_timestamp(): {y}-{mo}-{d} {h}:{mi}:{s} is not valid"
                    ))
                })?
        },
        // MAKE_INTERVAL(years, months, weeks, days, hours, mins, secs) → INTERVAL. Every field is
        // optional and positional, defaulting to 0; the seconds field may arrive as FLOAT or INT.
        (F::MakeInterval, fields) => {
            let int_at = |i: usize| -> i64 {
                match fields.get(i) {
                    Some(Int(n)) => *n,
                    _ => 0,
                }
            };
            // The seconds field is FLOAT, but a coerced INT or a NUMERIC literal (e.g. `1.5`) can also
            // reach here since both widen to FLOAT at the call site without an inserted cast.
            let secs = fields.get(6).map_or(0.0, to_f64);
            ast::Value::Interval(crate::interval::Interval::make(
                int_at(0),
                int_at(1),
                int_at(2),
                int_at(3),
                int_at(4),
                int_at(5),
                secs,
            ))
        },
        // JUSTIFY_DAYS / JUSTIFY_HOURS / JUSTIFY_INTERVAL(interval) — roll 30 days → 1 month and/or
        // 24 hours → 1 day, keeping each lower field signed and in range (B-fn).
        (F::JustifyDays, [ast::Value::Interval(iv)]) => ast::Value::Interval(iv.justify_days()),
        (F::JustifyHours, [ast::Value::Interval(iv)]) => ast::Value::Interval(iv.justify_hours()),
        (F::JustifyInterval, [ast::Value::Interval(iv)]) => {
            ast::Value::Interval(iv.justify_interval())
        },
        // SCALE / MIN_SCALE(numeric) → int; TRIM_SCALE(numeric) → numeric (B-fn). An INT argument
        // coerces to a scale-0 decimal, so all three are zero-scale for it.
        (F::Scale, [ast::Value::Numeric(d)]) => ast::Value::Int(i64::from(d.scale)),
        (F::Scale | F::MinScale, [Int(_)]) => Int(0),
        (F::MinScale, [ast::Value::Numeric(d)]) => ast::Value::Int(i64::from(d.min_scale())),
        (F::TrimScale, [ast::Value::Numeric(d)]) => ast::Value::Numeric(d.trim_scale()),
        (F::TrimScale, [Int(i)]) => ast::Value::Numeric(crate::numeric::Decimal::from_i64(*i)),
        // ISFINITE(value) — always true for the (finite-only) NUMERIC / temporal values here; a NULL
        // argument already returned NULL via the strict collection above (B-fn).
        (F::IsFinite, [_]) => ast::Value::Bool(true),
        // ENCODE(bytea, format) → text (`hex` / `escape`); DECODE(text, format) → bytea (B-fn).
        (F::Encode, [ast::Value::Bytes(b), Text(fmt)]) => Text(encode_bytea(b, fmt)?),
        (F::Decode, [Text(s), Text(fmt)]) => ast::Value::Bytes(decode_bytea(s, fmt)?),
        // DATE_BIN(stride, source, origin) → snap `source` to its bin aligned to `origin`.
        (
            F::DateBin,
            [
                ast::Value::Interval(stride),
                ast::Value::Timestamp(source),
                ast::Value::Timestamp(origin),
            ],
        ) => {
            if stride.months != 0 {
                return Err(Error::Unsupported(
                    "date_bin: the stride interval must not contain months or years".to_owned(),
                ));
            }
            crate::temporal::date_bin(stride.days, stride.micros, *source, *origin)
                .map(ast::Value::Timestamp)
                .ok_or_else(|| {
                    Error::Unsupported(
                        "date_bin: the stride must be positive and the result in range".to_owned(),
                    )
                })?
        },
        // RANDOM() is volatile — a fresh value per row; SETSEED(x) pins the generator seed.
        (F::Random, []) => ast::Value::Float(super::rng::next_f64()),
        // The argument is a FLOAT, or an INT/NUMERIC that coerces to one (a plain decimal
        // literal like `SETSEED(0.5)` now types as NUMERIC).
        (
            F::Setseed,
            [v @ (ast::Value::Float(_) | ast::Value::Int(_) | ast::Value::Numeric(_))],
        ) => {
            super::rng::set_seed(to_f64(v));
            ast::Value::Bool(true)
        },
        // PI() → the constant π (niladic).
        (F::Pi, []) => ast::Value::Float(std::f64::consts::PI),
        // Math functions — numeric-polymorphic, dispatched on value type within.
        (
            F::Abs
            | F::Round
            | F::Ceil
            | F::Floor
            | F::Sign
            | F::Mod
            | F::Power
            | F::Sqrt
            | F::Ln
            | F::Log
            | F::Exp
            | F::Sin
            | F::Cos
            | F::Tan
            | F::Asin
            | F::Acos
            | F::Atan
            | F::Atan2
            | F::Cot
            | F::Cbrt
            | F::Sinh
            | F::Cosh
            | F::Tanh
            | F::Asinh
            | F::Acosh
            | F::Atanh
            | F::Gcd
            | F::Lcm
            | F::Div
            | F::Degrees
            | F::Radians
            | F::Trunc,
            _,
        ) => eval_math(func, vals.as_slice())?,
        (F::ToHex, [Int(n)]) => Text(format!("{:x}", u64::from_ne_bytes(n.to_ne_bytes()))),
        (F::Factorial, [Int(n)]) => factorial(*n)?,
        // BIT_COUNT(n) — set bits in the two's-complement 64-bit value; 0..=64, never panics.
        (F::BitCount, [Int(n)]) => Int(i64::from(n.count_ones())),
        // WIDTH_BUCKET(operand, low, high, count) — the count argument is INT after analysis.
        (F::WidthBucket, [op, low, high, Int(count)]) => {
            width_bucket(to_f64(op), to_f64(low), to_f64(high), *count)?
        },
        // CARDINALITY(arr) — element count. A NULL array already returned NULL above.
        (F::Cardinality, [ast::Value::Array(items)]) => {
            Int(i64::try_from(items.len()).unwrap_or(i64::MAX))
        },
        // ARRAY_DIMS(arr): `[1:n]` for a non-empty 1-D array, NULL for an empty array.
        (F::ArrayDims, [ast::Value::Array(items)]) => {
            if items.is_empty() {
                ast::Value::Null
            } else {
                Text(format!("[1:{}]", items.len()))
            }
        },
        // ARRAY_NDIMS(arr): dimension count — 1 for a non-empty (1-D) array, NULL for an empty array
        // (which has no dimensions); a NULL array already returned NULL above (B-fn).
        (F::ArrayNdims, [ast::Value::Array(items)]) => {
            if items.is_empty() {
                ast::Value::Null
            } else {
                Int(1)
            }
        },
        // ARRAY_LENGTH(arr, dim): length along dimension `dim`; only 1-D arrays, and an empty array
        // has no dimension (→ NULL) per the standard array semantics.
        // ARRAY_LENGTH(arr, dim) and ARRAY_UPPER(arr, dim) both give the length of a non-empty 1-D
        // array (the upper bound equals the length for 1-D); NULL otherwise.
        (F::ArrayLength | F::ArrayUpper, [ast::Value::Array(items), Int(dim)]) => {
            if *dim == 1 && !items.is_empty() {
                Int(i64::try_from(items.len()).unwrap_or(i64::MAX))
            } else {
                ast::Value::Null
            }
        },
        // ARRAY_LOWER(arr, dim): the lower bound — always 1 for a non-empty 1-D array.
        (F::ArrayLower, [ast::Value::Array(items), Int(dim)]) => {
            if *dim == 1 && !items.is_empty() {
                Int(1)
            } else {
                ast::Value::Null
            }
        },
        // ARRAY_TO_STRING(arr, sep): join the non-NULL elements' text with `sep`.
        (F::ArrayToString, [ast::Value::Array(items), Text(sep)]) => Text(
            items
                .iter()
                .filter(|v| !matches!(v, ast::Value::Null))
                .map(crate::display::value_text)
                .collect::<Vec<_>>()
                .join(sep),
        ),
        // STRING_TO_ARRAY(s, sep): split `s` on `sep` into a TEXT[]; an empty `sep` yields `{s}`.
        (F::StringToArray, [Text(s), Text(sep)]) => ast::Value::Array(
            split_on_literal(s, sep)
                .into_iter()
                .map(ast::Value::Text)
                .collect(),
        ),
        // Vector distance functions. NULL operands already returned NULL above.
        (F::L2Distance | F::CosineDistance | F::InnerProduct, args) => {
            eval_vector_distance(func, args)?
        },
        // Arity/type mismatch is impossible after analysis — fall back to NULL defensively.
        _ => ast::Value::Null,
    })
}

/// `EXTRACT(field FROM src)` — a temporal field of `src` as a `Float`. A `DATE`/timestamp
/// gets the full field set; a `TIME` only the intraday fields. An inapplicable field (e.g. `year`
/// from a `TIME`) is rejected honestly rather than returning a wrong number.
fn extract_value(field: &str, src: &ast::Value) -> Result<ast::Value, Error> {
    use ast::Value as V;
    let value = match src {
        V::Date(days) => super::clock::day_start_micros(*days)
            .and_then(|micros| crate::temporal::extract_from_micros(field, micros)),
        V::Timestamp(m) | V::TimestampTz(m) => crate::temporal::extract_from_micros(field, *m),
        V::Time(t) => crate::temporal::extract_time_field(field, *t),
        V::Interval(iv) => crate::temporal::extract_interval_field(
            field,
            i64::from(iv.months),
            i64::from(iv.days),
            iv.micros,
        ),
        _ => return Ok(V::Null),
    };
    value.map(V::Float).ok_or_else(|| {
        Error::Unsupported(format!(
            "EXTRACT field `{field}` is not valid for this value"
        ))
    })
}

/// `DATE_TRUNC(field, src)` — `src` floored to the named precision, preserving its temporal type.
fn date_trunc_value(field: &str, src: &ast::Value) -> Result<ast::Value, Error> {
    use ast::Value as V;
    let (micros, is_tz) = match src {
        V::Timestamp(m) => (*m, false),
        V::TimestampTz(m) => (*m, true),
        _ => return Ok(V::Null),
    };
    let truncated = crate::temporal::date_trunc_micros(field, micros).ok_or_else(|| {
        Error::Unsupported(format!("DATE_TRUNC field `{field}` is not supported"))
    })?;
    Ok(if is_tz {
        V::TimestampTz(truncated)
    } else {
        V::Timestamp(truncated)
    })
}

/// `AGE(end, start)` → `end - start`; `AGE(value)` → statement date (midnight) `- value`. The
/// result is a calendar `Interval`.
fn age_value(end: &ast::Value, start: Option<&ast::Value>) -> Result<ast::Value, Error> {
    let (end_micros, start_micros) = if let Some(start) = start {
        (temporal_to_micros(end)?, temporal_to_micros(start)?)
    } else {
        let today_midnight = super::clock::day_start_micros(super::clock::statement_today())
            .ok_or_else(|| Error::Unsupported("AGE(): statement date out of range".to_owned()))?;
        (today_midnight, temporal_to_micros(end)?)
    };
    let (months, days, micros) = crate::temporal::calendar_age(end_micros, start_micros);
    Ok(ast::Value::Interval(crate::interval::Interval {
        months,
        days,
        micros,
    }))
}

/// Evaluate `<value> AT TIME ZONE <zone>`. A `TIMESTAMP` (wall-clock) is read as being in `zone` and
/// becomes the equivalent UTC `TIMESTAMPTZ` (`utc = wall - offset`); a `TIMESTAMPTZ` (a UTC instant)
/// becomes the wall-clock `TIMESTAMP` in `zone` (`wall = utc + offset`). `zone` is `UTC`/`GMT`/`Z` or a
/// fixed `±HH[:MM]` offset; a named zone with DST is rejected.
fn at_time_zone_value(value: &ast::Value, zone: &ast::Value) -> Result<ast::Value, Error> {
    use ast::Value as V;
    if matches!(value, V::Null) || matches!(zone, V::Null) {
        return Ok(V::Null);
    }
    // The east-of-UTC offset is taken from a text zone name/offset or an INTERVAL fixed offset.
    let offset = match zone {
        V::Text(zone) => zone_offset_micros(zone).ok_or_else(|| {
            Error::Unsupported(format!(
                "AT TIME ZONE only supports UTC and ±HH[:MM] offsets, not the named zone `{zone}`"
            ))
        })?,
        V::Interval(iv) => interval_zone_offset_micros(iv)?,
        other => {
            return Err(Error::Unsupported(format!(
                "AT TIME ZONE requires a text or INTERVAL zone, got {:?}",
                runtime_type(other)
            )));
        },
    };
    match value {
        // A naive wall-clock time in `zone` → the UTC instant.
        V::Timestamp(t) => t
            .checked_sub(offset)
            .map(V::TimestampTz)
            .ok_or(Error::IntegerOutOfRange),
        // A UTC instant → the wall-clock time in `zone`.
        V::TimestampTz(t) => t
            .checked_add(offset)
            .map(V::Timestamp)
            .ok_or(Error::IntegerOutOfRange),
        other => Err(Error::Unsupported(format!(
            "AT TIME ZONE requires a TIMESTAMP or TIMESTAMPTZ value, got {:?}",
            runtime_type(other)
        ))),
    }
}

/// The east-of-UTC offset in microseconds for an `AT TIME ZONE` zone string: `0` for `UTC`/`GMT`/`Z`
/// (case-insensitive), else the offset of the fixed-offset zone the string names, or `None` for a
/// named DST zone (which would need a time-zone database).
///
/// The reference engine uses the **POSIX** sign convention for a fixed-offset *string* here, which is the **opposite**
/// of ISO 8601: `'+05:00'` names a zone five hours *west* of UTC (UTC−5), not east. So the ISO-parsed
/// offset is negated to give the zone's true east-of-UTC offset, matching the reference engine.
fn zone_offset_micros(zone: &str) -> Option<i64> {
    let z = zone.trim();
    if z.eq_ignore_ascii_case("utc") || z.eq_ignore_ascii_case("gmt") || z.eq_ignore_ascii_case("z")
    {
        return Some(0);
    }
    crate::temporal::parse_zone_offset(z).map(|iso_offset| -iso_offset)
}

/// The east-of-UTC offset in microseconds for an `AT TIME ZONE INTERVAL '...'` fixed offset. Unlike a
/// text offset, the INTERVAL form uses the **ISO** sign — `INTERVAL '5 hours'` is UTC+5 — so the
/// interval's day+time span is the offset directly (no negation). A day counts as 24 hours here. A
/// months component has no fixed length, so it is rejected rather than guessed.
fn interval_zone_offset_micros(iv: &crate::interval::Interval) -> Result<i64, Error> {
    if iv.months != 0 {
        return Err(Error::Unsupported(
            "AT TIME ZONE INTERVAL offset must not contain a months component (ambiguous length)"
                .to_owned(),
        ));
    }
    i64::from(iv.days)
        .checked_mul(super::clock::MICROS_PER_DAY)
        .and_then(|day_micros| day_micros.checked_add(iv.micros))
        .ok_or(Error::IntegerOutOfRange)
}

/// Microseconds since the epoch for a temporal value (a `DATE` is taken at midnight).
fn temporal_to_micros(v: &ast::Value) -> Result<i64, Error> {
    use ast::Value as V;
    match v {
        V::Date(days) => super::clock::day_start_micros(*days)
            .ok_or_else(|| Error::Unsupported("AGE(): date out of range".to_owned())),
        V::Timestamp(m) | V::TimestampTz(m) => Ok(*m),
        _ => Err(Error::Unsupported(
            "AGE() requires a date or timestamp argument".to_owned(),
        )),
    }
}

/// `TO_CHAR(value, fmt)` — render a temporal `value` per `fmt`. A `DATE` is rendered from
/// midnight; a `TIME` from the 1970-01-01 epoch date plus its time of day.
fn to_char_value(value: &ast::Value, fmt: &str) -> ast::Value {
    use ast::Value as V;
    let micros = match value {
        V::Date(days) => super::clock::day_start_micros(*days),
        V::Timestamp(m) | V::TimestampTz(m) => Some(*m),
        V::Time(t) => Some(*t),
        _ => None,
    };
    micros.map_or(V::Null, |m| {
        V::Text(crate::temporal::format_with_pattern(m, fmt))
    })
}

/// `TO_CHAR(numeric, fmt)` — render a number through a digit-picture format (B-fn).
///
/// v1 supports the picture characters `9` (a digit whose leading positions are suppressed to
/// spaces), `0` (a digit whose leading positions are forced to `0`), and `.` (the decimal point).
/// The value is rounded to the fractional picture width. The sign is a reserved leading column that
/// *floats* to just before the first shown digit — `-` for a negative, a space otherwise (so
/// `to_char(-0.1, '99.99')` is `'  -.10'`) — and an integer part too wide for its positions renders
/// as `#` fill. Any other format character (`,`, `FM`, `S`, `MI`, `PR`, `D`, `G`, `$`, ...) is
/// rejected rather than silently mis-formatted; those forms are a later increment.
fn to_char_numeric(value: &ast::Value, fmt: &str) -> Result<ast::Value, Error> {
    use ast::Value as V;
    let dec = match value {
        V::Int(i) => crate::numeric::Decimal::from_i64(*i),
        V::Numeric(d) => *d,
        V::Float(f) => crate::numeric::from_f64_text(*f).ok_or_else(|| {
            Error::Unsupported("to_char(): a non-finite number has no formatted form".to_owned())
        })?,
        // NULL (and any non-number, which the analyzer rejects) formats to NULL.
        _ => return Ok(V::Null),
    };
    let (int_chars, frac_chars, has_point) = parse_numeric_format(fmt)?;
    let frac_n = frac_chars.len();
    let int_n = int_chars.len();
    if int_n == 0 && frac_n == 0 {
        return Err(Error::Unsupported(format!(
            "to_char(): numeric format `{fmt}` has no digit positions"
        )));
    }
    // Round/scale the value to the fractional picture width, then split its digits into the integer
    // and fractional parts.
    let scaled = dec
        .rescale(u8::try_from(frac_n).map_err(|_| Error::IntegerOutOfRange)?)
        .ok_or(Error::IntegerOutOfRange)?;
    let neg = scaled.mantissa < 0;
    let mut all_digits = scaled.mantissa.unsigned_abs().to_string();
    // Guarantee at least `frac_n + 1` digits so the split always yields a fractional slice (and a
    // placeholder integer `0` for a value `< 1`, stripped below).
    while all_digits.len() <= frac_n {
        all_digits.insert(0, '0');
    }
    let split = all_digits.len() - frac_n;
    let (int_digits, frac_digits) = all_digits.split_at(split);
    // A value `< 1` has no integer digit to show — drop the placeholder `0` so the sign floats to
    // the decimal point (matching `to_char(-0.1, '99.99')` = `'  -.10'`).
    let int_digits = int_digits.trim_start_matches('0');

    let overflow = int_digits.len() > int_n;
    let mut out = numeric_integer_area(int_digits, &int_chars, neg, overflow);
    if has_point {
        out.push('.');
        if overflow {
            out.extend(std::iter::repeat_n('#', frac_n));
        } else {
            out.push_str(frac_digits);
        }
    }
    Ok(V::Text(out))
}

/// Parse a `TO_CHAR` numeric format into `(integer picture, fractional picture, has decimal point)`.
/// Only `9`, `0`, and `.` are accepted; a second `.` or any other character is rejected.
fn parse_numeric_format(fmt: &str) -> Result<(Vec<char>, Vec<char>, bool), Error> {
    let mut int_chars = Vec::new();
    let mut frac_chars = Vec::new();
    let mut has_point = false;
    for ch in fmt.chars() {
        match ch {
            '9' | '0' if has_point => frac_chars.push(ch),
            '9' | '0' => int_chars.push(ch),
            '.' if has_point => {
                return Err(Error::Unsupported(
                    "to_char(): a numeric format may have only one decimal point".to_owned(),
                ));
            },
            '.' => has_point = true,
            other => {
                return Err(Error::Unsupported(format!(
                    "to_char(): unsupported numeric format character `{other}` \
                     (supported: `9`, `0`, `.`)"
                )));
            },
        }
    }
    Ok((int_chars, frac_chars, has_point))
}

/// Render the sign + integer part of a `TO_CHAR` number: a digit field `int_chars.len()` wide (real
/// digits right-aligned, a `0` picture position forced to `0`, a `9` position left blank when
/// unused) preceded by a floating sign column. The sign sits just before the first shown character
/// (or at the far right, next to the decimal point, when the integer part is entirely blank). On
/// overflow every position is `#`.
fn numeric_integer_area(int_digits: &str, int_chars: &[char], neg: bool, overflow: bool) -> String {
    let int_n = int_chars.len();
    let sign = if neg { '-' } else { ' ' };
    if overflow {
        let mut s = String::with_capacity(int_n + 1);
        s.push(sign);
        s.extend(std::iter::repeat_n('#', int_n));
        return s;
    }
    // Build the int_n-wide digit field by right-aligning the digits: first the leading filler
    // positions (`0` for a `0` picture char, a space for a `9`), then the real digits.
    let dlen = int_digits.chars().count();
    let pad = int_n.saturating_sub(dlen);
    let mut field: Vec<char> = Vec::with_capacity(int_n);
    for &pic in int_chars.iter().take(pad) {
        field.push(if pic == '0' { '0' } else { ' ' });
    }
    field.extend(int_digits.chars());
    // Float the sign to just before the first non-blank position (or the far right if all blank).
    let first_sig = field.iter().position(|&c| c != ' ').unwrap_or(int_n);
    let mut s = String::with_capacity(int_n + 1);
    s.extend(field.iter().take(first_sig));
    s.push(sign);
    s.extend(field.iter().skip(first_sig));
    s
}

/// `TO_DATE(text, fmt)` — parse `text` per `fmt` into a `DATE`. A mismatch is an honest
/// [`Error::InvalidValue`] rather than a wrong or NULL result.
fn to_date_value(text: &str, fmt: &str) -> Result<ast::Value, Error> {
    let micros =
        crate::temporal::parse_with_pattern(text, fmt).ok_or_else(|| Error::InvalidValue {
            ty: nusadb_core::ColumnType::Date,
            value: text.to_owned(),
        })?;
    let days = i32::try_from(micros.div_euclid(super::clock::MICROS_PER_DAY)).map_err(|_| {
        Error::InvalidValue {
            ty: nusadb_core::ColumnType::Date,
            value: text.to_owned(),
        }
    })?;
    Ok(ast::Value::Date(days))
}

/// `TO_TIMESTAMP(text, fmt)` — parse `text` per `fmt` into a `TIMESTAMP`.
fn to_timestamp_value(text: &str, fmt: &str) -> Result<ast::Value, Error> {
    let micros =
        crate::temporal::parse_with_pattern(text, fmt).ok_or_else(|| Error::InvalidValue {
            ty: nusadb_core::ColumnType::Timestamp,
            value: text.to_owned(),
        })?;
    Ok(ast::Value::Timestamp(micros))
}

/// `to_timestamp(epoch_seconds)` — a UNIX epoch (seconds since 1970-01-01 UTC, fractional allowed)
/// as a `TIMESTAMPTZ`. Integer seconds are exact; a `FLOAT`/`NUMERIC` is scaled to micros
/// and rounded half-away-from-zero. Overflow of the micro range is rejected, not wrapped.
fn to_timestamp_epoch(secs: &ast::Value) -> Result<ast::Value, Error> {
    const MICROS_PER_SEC: i64 = 1_000_000;
    fn overflow() -> Error {
        Error::Unsupported("to_timestamp(): epoch is out of range".to_owned())
    }
    // Scale fractional seconds to micros, rounding half-away-from-zero, and reject any value outside
    // the representable `[i64::MIN, 2^63)` micro range rather than saturating to a wrong instant.
    fn from_secs_f64(v: f64) -> Result<i64, Error> {
        let scaled = (v * MICROS_PER_SEC as f64).round();
        if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled >= 9_223_372_036_854_775_808.0
        {
            return Err(overflow());
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "range-checked to [i64::MIN, 2^63) above"
        )]
        Ok(scaled as i64)
    }
    let micros = match secs {
        ast::Value::Int(s) => s.checked_mul(MICROS_PER_SEC).ok_or_else(overflow)?,
        ast::Value::Float(f) => from_secs_f64(*f)?,
        ast::Value::Numeric(d) => from_secs_f64(d.to_f64())?,
        // NULL and any non-numeric argument (the analyzer rejects the latter) propagate NULL.
        _ => return Ok(ast::Value::Null),
    };
    Ok(ast::Value::TimestampTz(micros))
}

/// `TO_NUMBER(text, format)` — parse a formatted number into a `NUMERIC` (B-fn).
///
/// The format string declares intent but is read leniently: the value is recovered by scanning
/// `text` for its digits, sign, and decimal point while ignoring group separators, currency symbols,
/// spaces, and padding. `.` is the decimal point and `,` the group separator (default locale); a
/// leading/trailing `-`, or surrounding `<`/`(` of the `PR`/parenthesised-negative forms, makes the
/// result negative. Text with no digits is rejected.
fn to_number_value(text: &str) -> Result<ast::Value, Error> {
    let invalid = || Error::InvalidValue {
        ty: nusadb_core::ColumnType::Numeric {
            precision: 0,
            scale: 0,
        },
        value: text.to_owned(),
    };
    let mut digits = String::with_capacity(text.len());
    let mut seen_dot = false;
    let mut negative = false;
    for c in text.chars() {
        match c {
            '0'..='9' => digits.push(c),
            '.' if !seen_dot => {
                seen_dot = true;
                digits.push('.');
            },
            '-' | '<' | '(' => negative = true,
            // Group separators, currency, padding, sign-plus, and the closing `>`/`)` are ignored.
            _ => {},
        }
    }
    if !digits.bytes().any(|b| b.is_ascii_digit()) {
        return Err(invalid());
    }
    let decimal = crate::numeric::Decimal::parse(&digits).ok_or_else(invalid)?;
    Ok(ast::Value::Numeric(if negative {
        decimal.neg()
    } else {
        decimal
    }))
}

/// Compile `pattern` with standard regex flag characters: `i` (case-insensitive), `m` (multi-line), `s`
/// (`.` matches newline), `x` (ignore whitespace), and `g` (replace all — only meaningful for
/// `REGEXP_REPLACE`). Returns the compiled regex and whether `g` was set. An unsupported flag or a
/// malformed pattern surfaces as [`Error::InvalidRegex`] (a user error, not a panic).
fn compile_regex(pattern: &str, flags: &str) -> Result<(regex::Regex, bool), Error> {
    let mut builder = regex::RegexBuilder::new(pattern);
    let mut global = false;
    for f in flags.chars() {
        match f {
            'i' => {
                builder.case_insensitive(true);
            },
            'm' => {
                builder.multi_line(true);
            },
            's' => {
                builder.dot_matches_new_line(true);
            },
            'x' => {
                builder.ignore_whitespace(true);
            },
            'g' => global = true,
            other => return Err(Error::InvalidRegex(format!("unsupported flag '{other}'"))),
        }
    }
    let re = builder
        .build()
        .map_err(|e| Error::InvalidRegex(e.to_string()))?;
    Ok((re, global))
}

/// Translate a standard replacement string into the `regex` crate's syntax: `\1`..`\9` and `\&`
/// become `${N}` group references, `\\` a literal backslash, any other `\x` the bare `x`, and a
/// literal `$` is escaped to `$$` so it is not read as a group reference.
fn translate_replacement(replacement: &str) -> String {
    let mut out = String::with_capacity(replacement.len());
    let mut chars = replacement.chars();
    while let Some(c) = chars.next() {
        match c {
            '$' => out.push_str("$$"),
            '\\' => match chars.next() {
                Some(d) if d.is_ascii_digit() => {
                    out.push_str("${");
                    out.push(d);
                    out.push('}');
                },
                Some('&') => out.push_str("${0}"),
                Some(other) => out.push(other),
                None => out.push('\\'),
            },
            other => out.push(other),
        }
    }
    out
}

/// `REGEXP_REPLACE(s, pattern, replacement [, flags])` — replace the first match (or all with the
/// `g` flag) of `pattern` in `s`, honouring `\1`..`\9`/`\&` backreferences.
fn regexp_replace(
    source: &str,
    pattern: &str,
    replacement: &str,
    flags: &str,
) -> Result<String, Error> {
    let (re, global) = compile_regex(pattern, flags)?;
    let repl = translate_replacement(replacement);
    let out = if global {
        re.replace_all(source, repl.as_str())
    } else {
        re.replace(source, repl.as_str())
    };
    Ok(out.into_owned())
}

/// `SUBSTRING(s FROM 'pattern')` — the POSIX-regex form: the first capture
/// group of the first match when the pattern has capture groups (a non-participating group is
/// `NULL`), otherwise the whole first match; `NULL` when there is no match.
fn substring_regex(source: &str, pattern: &str) -> Result<ast::Value, Error> {
    let (re, _global) = compile_regex(pattern, "")?;
    Ok(re.captures(source).map_or(ast::Value::Null, |caps| {
        if re.captures_len() > 1 {
            caps.get(1).map_or(ast::Value::Null, |m| {
                ast::Value::Text(m.as_str().to_owned())
            })
        } else {
            ast::Value::Text(
                caps.get(0)
                    .map_or_else(String::new, |m| m.as_str().to_owned()),
            )
        }
    }))
}

/// `REGEXP_MATCH(s, pattern [, flags])` — the first match's capture groups as `TEXT[]` (or the whole
/// match when the pattern has no groups; non-participating groups become `NULL`), or `NULL` if there
/// is no match.
fn regexp_match(source: &str, pattern: &str, flags: &str) -> Result<ast::Value, Error> {
    let (re, _global) = compile_regex(pattern, flags)?;
    Ok(re
        .captures(source)
        .map_or(ast::Value::Null, |caps| captures_to_text_array(&re, &caps)))
}

/// One regex match's capture groups as a `TEXT[]` value: the whole match when the pattern has no
/// capture groups, otherwise one element per group with a non-participating group as `NULL`. Shared
/// by `REGEXP_MATCH` (scalar) and `REGEXP_MATCHES` (set-returning).
fn captures_to_text_array(re: &regex::Regex, caps: &regex::Captures) -> ast::Value {
    let elems: Vec<ast::Value> = if re.captures_len() <= 1 {
        // No explicit capture groups → the whole match is the single element.
        vec![ast::Value::Text(
            caps.get(0)
                .map_or_else(String::new, |m| m.as_str().to_owned()),
        )]
    } else {
        (1..re.captures_len())
            .map(|i| {
                caps.get(i).map_or(ast::Value::Null, |m| {
                    ast::Value::Text(m.as_str().to_owned())
                })
            })
            .collect()
    };
    ast::Value::Array(elems)
}

/// `REGEXP_MATCHES(s, pattern [, flags])` — the set-returning form of `REGEXP_MATCH`: one `TEXT[]` row
/// per match's capture groups. With the `g` flag every non-overlapping match is returned; without it
/// only the first match (0 or 1 rows). No match yields no rows.
pub(super) fn regexp_all_matches(
    source: &str,
    pattern: &str,
    flags: &str,
) -> Result<Vec<ast::Value>, Error> {
    let (re, global) = compile_regex(pattern, flags)?;
    if global {
        Ok(re
            .captures_iter(source)
            .map(|caps| captures_to_text_array(&re, &caps))
            .collect())
    } else {
        Ok(re
            .captures(source)
            .map(|caps| captures_to_text_array(&re, &caps))
            .into_iter()
            .collect())
    }
}

/// `REGEXP_LIKE(s, pattern [, flags])` — whether `pattern` matches anywhere in `s`.
fn regexp_like(source: &str, pattern: &str, flags: &str) -> Result<bool, Error> {
    let (re, _global) = compile_regex(pattern, flags)?;
    Ok(re.is_match(source))
}

/// `REGEXP_COUNT(s, pattern [, flags])` — number of non-overlapping matches of `pattern` in `s`,
/// saturating at `i64::MAX` (no string holds that many matches).
fn regexp_count(source: &str, pattern: &str, flags: &str) -> Result<i64, Error> {
    let (re, _global) = compile_regex(pattern, flags)?;
    Ok(i64::try_from(re.find_iter(source).count()).unwrap_or(i64::MAX))
}

/// `REGEXP_INSTR(s, pattern [, flags])` — the 1-based character position of the first match of
/// `pattern` in `s`, or `0` if there is no match. Counts Unicode scalar values, not bytes.
fn regexp_instr(source: &str, pattern: &str, flags: &str) -> Result<i64, Error> {
    let (re, _global) = compile_regex(pattern, flags)?;
    let Some(m) = re.find(source) else {
        return Ok(0);
    };
    // Convert the byte offset of the match to a 1-based character index.
    let char_index = source.get(..m.start()).map_or(0, |p| p.chars().count());
    Ok(i64::try_from(char_index)
        .unwrap_or(i64::MAX)
        .saturating_add(1))
}

/// Split `s` on the literal separator `sep` — the shared core of `STRING_TO_ARRAY` and the
/// set-returning `STRING_TO_TABLE`. An empty separator yields a single piece holding `s`.
pub(super) fn split_on_literal(s: &str, sep: &str) -> Vec<String> {
    if sep.is_empty() {
        vec![s.to_owned()]
    } else {
        s.split(sep).map(str::to_owned).collect()
    }
}

/// `REGEXP_SUBSTR(s, pattern [, flags])` — the first substring of `s` matching `pattern`, or `NULL`
/// if there is no match.
fn regexp_substr(source: &str, pattern: &str, flags: &str) -> Result<ast::Value, Error> {
    let (re, _global) = compile_regex(pattern, flags)?;
    Ok(re.find(source).map_or(ast::Value::Null, |m| {
        ast::Value::Text(m.as_str().to_owned())
    }))
}

/// Split `source` on each match of `pattern` into owned pieces — the shared core of
/// `REGEXP_SPLIT_TO_ARRAY` and the set-returning `REGEXP_SPLIT_TO_TABLE`. A
/// non-matching pattern yields a single piece holding `source` unchanged.
pub(super) fn regexp_split_pieces(
    source: &str,
    pattern: &str,
    flags: &str,
) -> Result<Vec<String>, Error> {
    let (re, _global) = compile_regex(pattern, flags)?;
    Ok(re.split(source).map(str::to_owned).collect())
}

/// `REGEXP_SPLIT_TO_ARRAY(s, pattern [, flags])` — split `s` on each match of `pattern` into a
/// `TEXT[]`. A non-matching pattern yields a single-element array holding `s` unchanged.
fn regexp_split_to_array(source: &str, pattern: &str, flags: &str) -> Result<ast::Value, Error> {
    let parts = regexp_split_pieces(source, pattern, flags)?
        .into_iter()
        .map(ast::Value::Text)
        .collect();
    Ok(ast::Value::Array(parts))
}

/// `CONCAT(a, b, ...)` — concatenate the text of every non-`NULL` argument; `NULL`s contribute
/// nothing and the result is never `NULL` (`''` when all arguments are `NULL`).
fn eval_concat(args: &[TypedExpr], row: &Row) -> Result<ast::Value, Error> {
    let mut out = String::new();
    for arg in args {
        match eval(arg, row)? {
            ast::Value::Null => {},
            value => out.push_str(&text_output(value)?),
        }
    }
    Ok(ast::Value::Text(out))
}

/// The text OUTPUT of a value for `||`/`CONCAT` coercion — the reference
/// output function, not the cast: booleans render `t`/`f` (a CAST renders `true`/`false`),
/// everything else shares the cast-to-text rendering (one source of truth).
fn text_output(value: ast::Value) -> Result<String, Error> {
    match value {
        ast::Value::Text(s) => Ok(s),
        ast::Value::Bool(b) => Ok(if b { "t" } else { "f" }.to_owned()),
        other => match cast_value(other, ColumnType::Text)? {
            ast::Value::Text(s) => Ok(s),
            _ => Err(Error::Unsupported(
                "internal: cast to TEXT did not yield text".to_owned(),
            )),
        },
    }
}

/// `NUM_NONNULLS(...)` / `NUM_NULLS(...)` — count the arguments that are (`count_nulls`) or are not
/// (`!count_nulls`) `NULL`. The result is always a non-`NULL` `INT`.
fn eval_num_nulls(args: &[TypedExpr], row: &Row, count_nulls: bool) -> Result<ast::Value, Error> {
    let mut count: i64 = 0;
    for arg in args {
        if matches!(eval(arg, row)?, ast::Value::Null) == count_nulls {
            count += 1;
        }
    }
    Ok(ast::Value::Int(count))
}

/// `CONCAT_WS(sep, a, b, ...)` — join the non-`NULL` data arguments with `sep`. A `NULL` separator
/// yields `NULL`; `NULL` data arguments are skipped (and produce no extra separators).
fn eval_concat_ws(args: &[TypedExpr], row: &Row) -> Result<ast::Value, Error> {
    let mut it = args.iter();
    let sep = match it.next() {
        Some(sep_expr) => match eval(sep_expr, row)? {
            ast::Value::Text(s) => s,
            // NULL (or, defensively, a non-text) separator → NULL.
            _ => return Ok(ast::Value::Null),
        },
        // No separator: the analyzer requires ≥1 argument, so this is unreachable.
        None => return Ok(ast::Value::Text(String::new())),
    };
    let mut parts = Vec::new();
    for arg in it {
        match eval(arg, row)? {
            ast::Value::Null => {},
            value => parts.push(text_output(value)?),
        }
    }
    Ok(ast::Value::Text(parts.join(&sep)))
}

/// `NULLIF(a, b)` — `NULL` when `a = b` is true, otherwise `a`. Equivalent to
/// `CASE WHEN a = b THEN NULL ELSE a END`, so a `NULL` operand (where `a = b` is unknown) returns `a`.
fn eval_nullif(args: &[TypedExpr], row: &Row) -> Result<ast::Value, Error> {
    let (Some(a_expr), Some(b_expr)) = (args.first(), args.get(1)) else {
        return Ok(ast::Value::Null);
    };
    let a = eval(a_expr, row)?;
    let b = eval(b_expr, row)?;
    if !matches!(a, ast::Value::Null)
        && !matches!(b, ast::Value::Null)
        && compare(&a, &b) == Ordering::Equal
    {
        return Ok(ast::Value::Null);
    }
    Ok(a)
}

/// `GREATEST`/`LEAST` — the largest (`greatest = true`) or smallest non-`NULL` argument, or `NULL`
/// when every argument is `NULL` (SQL skips `NULL`s rather than propagating).
fn eval_greatest_least(args: &[TypedExpr], row: &Row, greatest: bool) -> Result<ast::Value, Error> {
    let mut best: Option<ast::Value> = None;
    for arg in args {
        let v = eval(arg, row)?;
        if matches!(v, ast::Value::Null) {
            continue;
        }
        best = Some(match best {
            None => v,
            Some(cur) => {
                let pick_new = if greatest {
                    compare(&v, &cur) == Ordering::Greater
                } else {
                    compare(&v, &cur) == Ordering::Less
                };
                if pick_new { v } else { cur }
            },
        });
    }
    Ok(best.unwrap_or(ast::Value::Null))
}

/// `JSONB_SET(target, path, new_value, create_missing)`: set the value at `path` in the JSON
/// `target`. The target arrives as `Json` (or coerced `Text`); the path is a `TEXT[]` of object keys /
/// array indices; the new value is `Json`/`Text` and embeds as JSON.
fn json_set(
    target: &ast::Value,
    path: &[ast::Value],
    new: &ast::Value,
    create_missing: bool,
) -> ast::Value {
    let (ast::Value::Json(target) | ast::Value::Text(target)) = target else {
        return ast::Value::Null;
    };
    let keys: Vec<String> = path
        .iter()
        .map(|p| match p {
            ast::Value::Text(s) => s.clone(),
            other => crate::display::value_text(other),
        })
        .collect();
    let new_json = crate::json::value_to_json(new);
    crate::json::set_path(target, &keys, new_json, create_missing)
        .map_or(ast::Value::Null, ast::Value::Json)
}

/// `JSONB_INSERT(target, path, new_value [, insert_after])` — insert `new_value` at `path` without
/// overwriting. Mirrors [`json_set`] but never replaces an existing value.
fn json_insert(
    target: &ast::Value,
    path: &[ast::Value],
    new: &ast::Value,
    insert_after: bool,
) -> ast::Value {
    let (ast::Value::Json(target) | ast::Value::Text(target)) = target else {
        return ast::Value::Null;
    };
    let keys: Vec<String> = path
        .iter()
        .map(|p| match p {
            ast::Value::Text(s) => s.clone(),
            other => crate::display::value_text(other),
        })
        .collect();
    let new_json = crate::json::value_to_json(new);
    crate::json::insert_path(target, &keys, new_json, insert_after)
        .map_or(ast::Value::Null, ast::Value::Json)
}

/// `ARRAY_APPEND(arr, elem)` / `ARRAY_PREPEND(elem, arr)` / `ARRAY_CAT(a, b)`. A `NULL` array
/// operand is treated as an empty array; a `NULL` element is stored as a `NULL` element. A non-array,
/// non-NULL array operand is impossible after analysis and falls back to `NULL`.
fn eval_array_mutate(
    func: ast::ScalarFunc,
    args: &[TypedExpr],
    row: &Row,
) -> Result<ast::Value, Error> {
    use ast::ScalarFunc as F;
    let [a_expr, b_expr] = args else {
        return Ok(ast::Value::Null);
    };
    let a = eval(a_expr, row)?;
    let b = eval(b_expr, row)?;
    // Decode an array operand into its element vector; a NULL array is an empty vector.
    let as_items = |v: ast::Value| match v {
        ast::Value::Array(items) => Some(items),
        ast::Value::Null => Some(Vec::new()),
        _ => None,
    };
    Ok(match func {
        F::ArrayAppend => as_items(a).map_or(ast::Value::Null, |mut items| {
            items.push(b);
            ast::Value::Array(items)
        }),
        F::ArrayPrepend => as_items(b).map_or(ast::Value::Null, |mut items| {
            items.insert(0, a);
            ast::Value::Array(items)
        }),
        F::ArrayCat => match (as_items(a), as_items(b)) {
            (Some(mut items), Some(rest)) => {
                items.extend(rest);
                ast::Value::Array(items)
            },
            _ => ast::Value::Null,
        },
        // ARRAY_POSITION: 1-based index of the first element equal to `b`, or NULL.
        F::ArrayPosition => match a {
            ast::Value::Array(items) => items
                .iter()
                .position(|it| value_eq(it, &b))
                .map_or(ast::Value::Null, |i| {
                    ast::Value::Int(i64::try_from(i + 1).unwrap_or(i64::MAX))
                }),
            _ => ast::Value::Null,
        },
        // ARRAY_POSITIONS: an INT[] of every 1-based index where an element equals `b` (NULL `b`
        // finds NULL elements); an empty array if none, NULL if the array operand is NULL.
        F::ArrayPositions => match a {
            ast::Value::Array(items) => ast::Value::Array(
                items
                    .iter()
                    .enumerate()
                    .filter(|(_, it)| value_eq(it, &b))
                    .map(|(i, _)| ast::Value::Int(i64::try_from(i + 1).unwrap_or(i64::MAX)))
                    .collect(),
            ),
            _ => ast::Value::Null,
        },
        // ARRAY_REMOVE: drop every element equal to `b` (NULL `b` removes the NULL elements).
        F::ArrayRemove => match a {
            ast::Value::Array(items) => {
                ast::Value::Array(items.into_iter().filter(|it| !value_eq(it, &b)).collect())
            },
            _ => ast::Value::Null,
        },
        _ => ast::Value::Null,
    })
}

/// `ARRAY_REPLACE(arr, from, to)` — `arr` with every element equal to `from` replaced by `to` (B-fn).
/// A NULL `from` matches the array's NULL elements; a NULL array operand yields NULL.
fn eval_array_replace(args: &[TypedExpr], row: &Row) -> Result<ast::Value, Error> {
    let [arr_expr, from_expr, to_expr] = args else {
        return Ok(ast::Value::Null);
    };
    let arr = eval(arr_expr, row)?;
    let from = eval(from_expr, row)?;
    let to = eval(to_expr, row)?;
    Ok(match arr {
        ast::Value::Array(items) => ast::Value::Array(
            items
                .into_iter()
                .map(|it| if value_eq(&it, &from) { to.clone() } else { it })
                .collect(),
        ),
        _ => ast::Value::Null,
    })
}

/// Element equality for the array-search functions: two `NULL`s are equal (so `ARRAY_POSITION`/
/// `ARRAY_REMOVE` can find/remove `NULL` elements), a `NULL` matches nothing else, otherwise the
/// values are compared by the standard ordering.
fn value_eq(a: &ast::Value, b: &ast::Value) -> bool {
    match (a, b) {
        (ast::Value::Null, ast::Value::Null) => true,
        (ast::Value::Null, _) | (_, ast::Value::Null) => false,
        (x, y) => compare(x, y) == Ordering::Equal,
    }
}

/// `FORMAT(fmt, ...)` (B-fn): substitute the trailing arguments into the format string's specifiers.
///
/// Supported specifiers: `%s` (the argument as plain text; a NULL renders as the empty string), `%I`
/// (as a SQL identifier, like `QUOTE_IDENT`; a NULL is an error), `%L` (as a SQL literal, like
/// `QUOTE_LITERAL`; a NULL renders as the unquoted word `NULL`), and `%%` (a literal `%`). Arguments
/// are consumed left to right; a specifier with no remaining argument, or an unknown specifier, is an
/// error. A NULL format string yields NULL.
fn eval_format(args: &[TypedExpr], row: &Row) -> Result<ast::Value, Error> {
    let Some((fmt_expr, rest)) = args.split_first() else {
        return Err(Error::Unsupported(
            "format() expects at least 1 argument".to_owned(),
        ));
    };
    let fmt = match eval(fmt_expr, row)? {
        ast::Value::Null => return Ok(ast::Value::Null),
        ast::Value::Text(s) => s,
        other => crate::display::value_text(&other),
    };
    // Pre-evaluate the substitution arguments once, consumed in order by the specifiers.
    let mut values = Vec::with_capacity(rest.len());
    for arg in rest {
        values.push(eval(arg, row)?);
    }
    let mut out = String::with_capacity(fmt.len());
    let mut next_arg = values.into_iter();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('%') => out.push('%'),
            Some(spec @ ('s' | 'I' | 'L')) => {
                let value = next_arg.next().ok_or_else(|| {
                    Error::Unsupported(format!("format(): too few arguments for %{spec}"))
                })?;
                match (spec, &value) {
                    ('s', ast::Value::Null) => {},
                    ('s', v) => out.push_str(&crate::display::value_text(v)),
                    ('I', ast::Value::Null) => {
                        return Err(Error::Unsupported(
                            "format(): NULL is not allowed for the %I specifier".to_owned(),
                        ));
                    },
                    ('I', v) => out.push_str(&quote_ident(&crate::display::value_text(v))),
                    ('L', ast::Value::Null) => out.push_str("NULL"),
                    ('L', v) => out.push_str(&quote_literal(&crate::display::value_text(v))),
                    _ => unreachable!("specifier guarded to s/I/L above"),
                }
            },
            Some(other) => {
                return Err(Error::Unsupported(format!(
                    "format(): unrecognized format specifier %{other}"
                )));
            },
            None => {
                return Err(Error::Unsupported(
                    "format(): dangling % at end of format string".to_owned(),
                ));
            },
        }
    }
    Ok(ast::Value::Text(out))
}

/// `TO_JSON(value)` / `TO_JSONB(value)`: serialize the argument to canonical JSON text. A NULL
/// argument becomes JSON `null`.
fn eval_to_json(args: &[TypedExpr], row: &Row) -> Result<ast::Value, Error> {
    let [arg] = args else {
        return Err(Error::Unsupported(
            "to_json() expects 1 argument".to_owned(),
        ));
    };
    let value = eval(arg, row)?;
    let json = crate::json::to_text(&crate::json::value_to_json(&value));
    Ok(ast::Value::Json(json))
}

/// `ROW_TO_JSON(...)`: serialize a row to a JSON object. The analyzer lowers both forms —
/// `row_to_json(row(a, b))` (positional field names `f1`, `f2`, …) and `row_to_json(t)` (a table /
/// alias, keyed by real column names) — to the same interleaved `key, value, key, value, …`
/// argument list, so this builds the object uniformly. A NULL value becomes JSON `null` (it does not
/// propagate NULL).
///
/// The object text is assembled in argument order by hand: `serde_json::Map` (the JSON builder's
/// backing store) sorts keys, which would both mis-order `f10` before `f2` and lose a table's column
/// order, so the declared field order is preserved explicitly here.
fn eval_row_to_json(args: &[TypedExpr], row: &Row) -> Result<ast::Value, Error> {
    use std::fmt::Write as _;
    let mut out = String::from("{");
    for (i, pair) in args.chunks_exact(2).enumerate() {
        let [key_expr, val_expr] = pair else { continue };
        let key = match eval(key_expr, row)? {
            ast::Value::Null => {
                return Err(Error::Unsupported(
                    "row_to_json(): field name must not be NULL".to_owned(),
                ));
            },
            other => crate::display::value_text(&other),
        };
        let value = crate::json::value_to_json(&eval(val_expr, row)?);
        let sep = if i == 0 { "" } else { "," };
        // Writing to a String is infallible.
        let _ = write!(
            out,
            "{sep}{}:{}",
            crate::json::string_literal(&key),
            crate::json::to_text(&value)
        );
    }
    out.push('}');
    Ok(ast::Value::Json(out))
}

/// `JSON_BUILD_OBJECT(k1, v1, ...)`: build a JSON object from alternating key/value arguments.
/// Keys are coerced to text and must not be NULL; values serialize to JSON (a NULL value → `null`).
fn eval_json_build_object(args: &[TypedExpr], row: &Row) -> Result<ast::Value, Error> {
    let mut pairs = Vec::with_capacity(args.len() / 2);
    for pair in args.chunks_exact(2) {
        let [key_expr, val_expr] = pair else {
            continue;
        };
        let key = match eval(key_expr, row)? {
            ast::Value::Null => {
                return Err(Error::Unsupported(
                    "json_build_object(): object key must not be NULL".to_owned(),
                ));
            },
            other => crate::display::value_text(&other),
        };
        let val = crate::json::value_to_json(&eval(val_expr, row)?);
        pairs.push((key, val));
    }
    Ok(ast::Value::Json(crate::json::build_object(pairs)))
}

/// `json_build_array(v1, v2, ...)`: build a JSON array from the arguments in order. A NULL
/// argument becomes JSON `null` (it does not propagate NULL), so this skips the NULL-strict path.
fn eval_json_build_array(args: &[TypedExpr], row: &Row) -> Result<ast::Value, Error> {
    let mut items = Vec::with_capacity(args.len());
    for arg in args {
        items.push(crate::json::value_to_json(&eval(arg, row)?));
    }
    Ok(ast::Value::Json(crate::json::build_array(items)))
}

/// `LEFT(s, n)` — first `n` characters; a negative `n` drops the last `|n|` characters.
fn left(s: &str, n: i64) -> String {
    let len = char_len(s);
    let take = if n < 0 { (len + n).max(0) } else { n.min(len) };
    s.chars().take(usize::try_from(take).unwrap_or(0)).collect()
}

/// `RIGHT(s, n)` — last `n` characters; a negative `n` drops the first `|n|` characters.
fn right(s: &str, n: i64) -> String {
    let len = char_len(s);
    let skip = if n < 0 {
        n.saturating_neg().min(len)
    } else {
        (len - n).max(0)
    };
    s.chars().skip(usize::try_from(skip).unwrap_or(0)).collect()
}

/// `SPLIT_PART(s, delim, n)` — the 1-based `n`th field of `s` split on `delim`, or `''` when `n` is
/// out of range (`n < 1` always yields `''`). An empty `delim` treats `s` as a single field.
fn split_part(s: &str, delim: &str, n: i64) -> String {
    if n < 1 {
        return String::new();
    }
    let idx = usize::try_from(n - 1).unwrap_or(usize::MAX);
    if delim.is_empty() {
        return if idx == 0 {
            s.to_owned()
        } else {
            String::new()
        };
    }
    s.split(delim).nth(idx).unwrap_or("").to_owned()
}

/// `REVERSE(s)` — the characters of `s` in reverse order.
fn reverse(s: &str) -> String {
    s.chars().rev().collect()
}

/// `OVERLAY(s PLACING r FROM start [FOR len])` — replace `len` characters of `s` (default: the
/// character length of `r`) starting at 1-based `start` with `r`. Operates on Unicode scalar
/// values, not bytes. Out-of-range `start`/`len` are clamped to the string, so the result is always
/// well-defined and never panics.
fn overlay(s: &str, replacement: &str, start: i64, for_len: Option<i64>) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    // 1-based `start` → 0-based, clamped to `[0, n]`.
    let start0 = if start <= 1 {
        0
    } else {
        usize::try_from(start - 1).unwrap_or(usize::MAX).min(n)
    };
    // The number of characters to remove: the explicit FOR length (a negative value removes none),
    // else the character length of the replacement (the SQL-standard default).
    let remove = for_len.map_or_else(
        || replacement.chars().count(),
        |len| usize::try_from(len).unwrap_or(0),
    );
    let end0 = start0.saturating_add(remove).min(n);
    let mut out = String::with_capacity(s.len() + replacement.len());
    // Both slices are clamped to `[0, n]`, so `get` always succeeds; `unwrap_or` is a belt-and-braces
    // guard that keeps the function panic-free.
    out.extend(chars.get(..start0).unwrap_or(&[]).iter());
    out.push_str(replacement);
    out.extend(chars.get(end0..).unwrap_or(&[]).iter());
    out
}

/// `QUOTE_LITERAL(s)` — wrap `s` as a single-quoted SQL string literal, doubling any embedded single
/// quote so the result re-parses to the original text.
fn quote_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push('\'');
        }
        out.push(c);
    }
    out.push('\'');
    out
}

/// `QUOTE_IDENT(s)` — quote `s` for use as a SQL identifier. A string that is already a safe
/// unquoted identifier (`[a-z_][a-z0-9_]*`) is returned unchanged; otherwise it is wrapped in double
/// quotes with any embedded double quote doubled. v1 does not also quote reserved words.
fn quote_ident(s: &str) -> String {
    let is_safe = {
        let mut chars = s.chars();
        // `next()` yields `Some` only for a non-empty string, so an empty `s` is never "safe".
        chars
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
            && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    };
    if is_safe {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        if c == '"' {
            out.push('"');
        }
        out.push(c);
    }
    out.push('"');
    out
}

/// `INITCAP(s)` — upper-case the first letter of each word and lower-case the rest. A word
/// boundary is any non-alphanumeric character.
fn initcap(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut at_word_start = true;
    for c in s.chars() {
        if c.is_alphanumeric() {
            if at_word_start {
                out.extend(c.to_uppercase());
            } else {
                out.extend(c.to_lowercase());
            }
            at_word_start = false;
        } else {
            out.push(c);
            at_word_start = true;
        }
    }
    out
}

/// `TRANSLATE(s, from, to)` — replace each character of `s` found in `from` with the character at the
/// same position in `to`, dropping it when `to` is shorter. The first occurrence in `from`
/// wins for a repeated character.
fn translate(s: &str, from: &str, to: &str) -> String {
    let to: Vec<char> = to.chars().collect();
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match from.chars().position(|f| f == c) {
            Some(i) => {
                if let Some(repl) = to.get(i) {
                    out.push(*repl);
                }
            },
            None => out.push(c),
        }
    }
    out
}

/// The `i64` value of math argument `i` (`0` for a non-integer / missing argument, which the analyzer
/// rules out for `GCD`/`LCM`).
fn int_arg(vals: &[ast::Value], i: usize) -> i64 {
    match vals.get(i) {
        Some(ast::Value::Int(n)) => *n,
        _ => 0,
    }
}

/// Greatest common divisor of two integers (always non-negative); `gcd(0, 0) = 0`.
fn int_gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    i64::try_from(a).unwrap_or(i64::MAX)
}

/// Least common multiple of two integers (non-negative); `0` if either operand is `0`. The
/// product saturates rather than overflowing.
fn int_lcm(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        return 0;
    }
    let g = int_gcd(a, b);
    (a.saturating_abs() / g).saturating_mul(b.saturating_abs())
}

/// Evaluate a numeric math built-in. Arguments are already NULL-checked and numeric. The
/// power/transcendental/trig functions compute in `f64` and return `FLOAT`; the type-preserving
/// functions (`ABS`/`CEIL`/`FLOOR`/`SIGN`/`ROUND`/`MOD`) dispatch on the operand's value type.
fn eval_math(func: ast::ScalarFunc, vals: &[ast::Value]) -> Result<ast::Value, Error> {
    use ast::ScalarFunc as F;
    use ast::Value::Float;
    // `f64` view of argument `i` (0.0 for a missing argument, which arity-checking rules out).
    let f = |i: usize| vals.get(i).map_or(0.0, to_f64);
    // A math function out of its domain raises (matching the standard) rather than returning a silent
    // `NaN`/`±inf` that would propagate through later arithmetic.
    let domain = |msg: &str| Error::ArgumentOutOfDomain(msg.to_owned());
    // The standard logarithm-domain message: zero and negative inputs differ.
    let log_domain = |x: f64| {
        domain(if x == 0.0 {
            "cannot take logarithm of zero"
        } else {
            "cannot take logarithm of a negative number"
        })
    };
    Ok(match func {
        F::Sqrt => {
            let x = f(0);
            if x < 0.0 {
                return Err(domain("cannot take square root of a negative number"));
            }
            Float(x.sqrt())
        },
        F::Ln => {
            let x = f(0);
            if x <= 0.0 {
                return Err(log_domain(x));
            }
            Float(x.ln())
        },
        F::Exp => Float(f(0).exp()),
        F::Sin => Float(f(0).sin()),
        F::Cos => Float(f(0).cos()),
        F::Tan => Float(f(0).tan()),
        // ASIN/ACOS are defined only on [-1, 1]; outside it the standard raises.
        F::Asin => {
            let x = f(0);
            if !(-1.0..=1.0).contains(&x) {
                return Err(domain("input is out of range"));
            }
            Float(x.asin())
        },
        F::Acos => {
            let x = f(0);
            if !(-1.0..=1.0).contains(&x) {
                return Err(domain("input is out of range"));
            }
            Float(x.acos())
        },
        F::Atan => Float(f(0).atan()),
        F::Atan2 => Float(f(0).atan2(f(1))),
        // COT = 1/tan; tan(0) is finite so this yields ±inf at multiples of pi, matching f64 semantics.
        F::Cot => Float(f(0).tan().recip()),
        F::Cbrt => Float(f(0).cbrt()),
        F::Sinh => Float(f(0).sinh()),
        F::Cosh => Float(f(0).cosh()),
        F::Tanh => Float(f(0).tanh()),
        F::Asinh => Float(f(0).asinh()),
        F::Acosh => Float(f(0).acosh()),
        F::Atanh => Float(f(0).atanh()),
        // GCD/LCM operate on the integer values; non-Int operands are impossible after analysis.
        F::Gcd => ast::Value::Int(int_gcd(int_arg(vals, 0), int_arg(vals, 1))),
        F::Lcm => ast::Value::Int(int_lcm(int_arg(vals, 0), int_arg(vals, 1))),
        // DIV(a, b) is the integer quotient truncated toward zero; a zero divisor is an error, and
        // the `i64::MIN / -1` overflow errors rather than wrapping.
        F::Div => {
            let divisor = int_arg(vals, 1);
            if divisor == 0 {
                return Err(Error::DivisionByZero);
            }
            let quotient = int_arg(vals, 0)
                .checked_div(divisor)
                .ok_or(Error::IntegerOutOfRange)?;
            ast::Value::Int(quotient)
        },
        F::Degrees => Float(f(0).to_degrees()),
        F::Radians => Float(f(0).to_radians()),
        F::Power => Float(f(0).powf(f(1))),
        // LOG(x) = base-10; LOG(b, x) = base-b (args ordered base then value, per SQL). A non-positive
        // value is out of domain; a base ≤ 0 or = 1 has no logarithm (base 1 would divide by zero).
        F::Log if vals.len() <= 1 => {
            let x = f(0);
            if x <= 0.0 {
                return Err(log_domain(x));
            }
            Float(x.log10())
        },
        F::Log => {
            let (base, x) = (f(0), f(1));
            if x <= 0.0 {
                return Err(log_domain(x));
            }
            if base <= 0.0 || (base - 1.0).abs() < f64::EPSILON {
                return Err(domain("logarithm base must be positive and not 1"));
            }
            Float(x.log(base))
        },
        F::Abs => math_abs(vals.first())?,
        F::Sign => math_sign(vals.first()),
        F::Ceil => math_floor_ceil(vals.first(), true),
        F::Floor => math_floor_ceil(vals.first(), false),
        F::Round => math_round(vals)?,
        F::Trunc => math_trunc(vals),
        F::Mod => math_mod(vals.first(), vals.get(1))?,
        // Non-math func routed here is impossible after analysis.
        _ => ast::Value::Null,
    })
}

/// `ABS` preserving the numeric type. `ABS(i64::MIN)` overflows (no positive `i64` for it) and
/// errors rather than wrapping to a negative result; `FLOAT`/`NUMERIC` are exact.
fn math_abs(v: Option<&ast::Value>) -> Result<ast::Value, Error> {
    Ok(match v {
        Some(ast::Value::Int(i)) => {
            ast::Value::Int(i.checked_abs().ok_or(Error::IntegerOutOfRange)?)
        },
        Some(ast::Value::Float(x)) => ast::Value::Float(x.abs()),
        Some(ast::Value::Numeric(d)) => {
            ast::Value::Numeric(if d.mantissa < 0 { d.neg() } else { *d })
        },
        _ => ast::Value::Null,
    })
}

/// `SIGN` (`-1`/`0`/`1`) preserving the numeric type. `SIGN(0.0)` is `0.0` and `SIGN(NaN)` is `NaN`
/// (f64's own `signum` returns `±1.0` for these, which SQL does not want).
fn math_sign(v: Option<&ast::Value>) -> ast::Value {
    match v {
        Some(ast::Value::Int(i)) => ast::Value::Int(i.signum()),
        Some(ast::Value::Float(x)) => ast::Value::Float(if *x == 0.0 || x.is_nan() {
            *x
        } else {
            x.signum()
        }),
        Some(ast::Value::Numeric(d)) => {
            let s = match d.compare(&crate::numeric::Decimal::ZERO) {
                Ordering::Less => -1,
                Ordering::Equal => 0,
                Ordering::Greater => 1,
            };
            ast::Value::Numeric(crate::numeric::Decimal::from_i64(s))
        },
        _ => ast::Value::Null,
    }
}

/// `CEIL`/`FLOOR` preserving the numeric type. Integers are returned unchanged; floats use the libm
/// rounding; `NUMERIC` rounds toward `+∞`/`-∞` exactly via Euclidean division of the mantissa.
fn math_floor_ceil(v: Option<&ast::Value>, ceil: bool) -> ast::Value {
    match v {
        Some(ast::Value::Int(i)) => ast::Value::Int(*i),
        Some(ast::Value::Float(x)) => ast::Value::Float(if ceil { x.ceil() } else { x.floor() }),
        Some(ast::Value::Numeric(d)) => numeric_floor_ceil(d, ceil),
        _ => ast::Value::Null,
    }
}

/// Round a [`crate::numeric::Decimal`] toward `+∞` (ceil) or `-∞` (floor) to a whole number.
fn numeric_floor_ceil(d: &crate::numeric::Decimal, ceil: bool) -> ast::Value {
    use crate::numeric::Decimal;
    if d.scale == 0 {
        return ast::Value::Numeric(*d);
    }
    let Some(p) = 10i128.checked_pow(u32::from(d.scale)) else {
        return ast::Value::Numeric(*d);
    };
    // Euclidean division floors toward -∞ for a positive divisor; ceil is `-floor(-x)`.
    let q = if ceil {
        (-d.mantissa)
            .div_euclid(p)
            .checked_neg()
            .unwrap_or(i128::MAX)
    } else {
        d.mantissa.div_euclid(p)
    };
    ast::Value::Numeric(Decimal {
        mantissa: q,
        scale: 0,
    })
}

/// `ROUND(x [, d])` preserving the numeric type. Default 0 places; a `NUMERIC` rescales exactly, a
/// `FLOAT` rounds via scaling, an `INT` is returned unchanged.
fn math_round(vals: &[ast::Value]) -> Result<ast::Value, Error> {
    let places = match vals.get(1) {
        Some(ast::Value::Int(d)) => *d,
        _ => 0,
    };
    Ok(match vals.first() {
        Some(ast::Value::Int(i)) => ast::Value::Int(*i),
        Some(ast::Value::Float(x)) => {
            let scale = 10f64.powi(i32::try_from(places).unwrap_or(0));
            ast::Value::Float((x * scale).round() / scale)
        },
        Some(ast::Value::Numeric(d)) => {
            let target = u8::try_from(places.max(0)).unwrap_or(crate::numeric::MAX_SCALE);
            ast::Value::Numeric(
                d.rescale(target)
                    .ok_or_else(|| Error::Unsupported("numeric round overflow".to_owned()))?,
            )
        },
        _ => ast::Value::Null,
    })
}

/// `TRUNC(x [, d])` truncating toward zero to `d` decimal places (default 0), preserving the numeric
/// type. Unlike [`math_round`] the discarded fraction is dropped, not rounded. Like `ROUND`, a
/// negative `d` is clamped to `0` for the `NUMERIC` path (the `FLOAT` path honours it via scaling).
fn math_trunc(vals: &[ast::Value]) -> ast::Value {
    let places = match vals.get(1) {
        Some(ast::Value::Int(d)) => *d,
        _ => 0,
    };
    match vals.first() {
        Some(ast::Value::Int(i)) => ast::Value::Int(*i),
        Some(ast::Value::Float(x)) => {
            let scale = 10f64.powi(i32::try_from(places).unwrap_or(0));
            ast::Value::Float((x * scale).trunc() / scale)
        },
        Some(ast::Value::Numeric(d)) => numeric_trunc(d, places),
        _ => ast::Value::Null,
    }
}

/// Truncate a [`crate::numeric::Decimal`] toward zero to `places` fractional digits. `i128` division
/// truncates toward zero — exactly the `TRUNC` semantics. A negative `places` is clamped to `0`.
fn numeric_trunc(d: &crate::numeric::Decimal, places: i64) -> ast::Value {
    use crate::numeric::Decimal;
    let target = u8::try_from(places.max(0)).unwrap_or(u8::MAX);
    if d.scale <= target {
        return ast::Value::Numeric(*d);
    }
    let Some(div) = 10i128.checked_pow(u32::from(d.scale - target)) else {
        return ast::Value::Numeric(*d);
    };
    ast::Value::Numeric(Decimal {
        mantissa: d.mantissa / div,
        scale: target,
    })
}

/// `MOD(x, y)` preserving the unified numeric type: `FLOAT` remainder, exact `NUMERIC` remainder, or
/// integer remainder. Division by zero is rejected.
fn math_mod(a: Option<&ast::Value>, b: Option<&ast::Value>) -> Result<ast::Value, Error> {
    use crate::numeric::Decimal;
    let (Some(a), Some(b)) = (a, b) else {
        return Ok(ast::Value::Null);
    };
    if matches!(a, ast::Value::Float(_)) || matches!(b, ast::Value::Float(_)) {
        let divisor = to_f64(b);
        if divisor == 0.0 {
            return Err(Error::DivisionByZero);
        }
        return Ok(ast::Value::Float(to_f64(a) % divisor));
    }
    if matches!(a, ast::Value::Numeric(_)) || matches!(b, ast::Value::Numeric(_)) {
        let to_dec = |v: &ast::Value| match v {
            ast::Value::Numeric(d) => Some(*d),
            ast::Value::Int(i) => Some(Decimal::from_i64(*i)),
            _ => None,
        };
        let (Some(da), Some(db)) = (to_dec(a), to_dec(b)) else {
            return Ok(ast::Value::Null);
        };
        if db.is_zero() {
            return Err(Error::DivisionByZero);
        }
        return Ok(ast::Value::Numeric(da.checked_rem(&db).ok_or_else(
            || Error::Unsupported("numeric mod overflow".to_owned()),
        )?));
    }
    if let (ast::Value::Int(x), ast::Value::Int(y)) = (a, b) {
        if *y == 0 {
            return Err(Error::DivisionByZero);
        }
        // `i64::MIN % -1` overflows (the quotient does); error rather than wrap.
        return Ok(ast::Value::Int(
            x.checked_rem(*y).ok_or(Error::IntegerOutOfRange)?,
        ));
    }
    Ok(ast::Value::Null)
}

/// `FACTORIAL(n)` — `n!` as an `INT`. A negative argument is undefined (error), and any
/// `n > 20` overflows `i64` (`checked_mul` surfaces this as an error rather than wrapping).
fn factorial(n: i64) -> Result<ast::Value, Error> {
    if n < 0 {
        return Err(Error::Unsupported(
            "factorial of a negative number is undefined".to_owned(),
        ));
    }
    let mut acc: i64 = 1;
    for k in 2..=n {
        acc = acc.checked_mul(k).ok_or_else(|| {
            Error::Unsupported(format!("factorial({n}) overflows a 64-bit integer"))
        })?;
    }
    Ok(ast::Value::Int(acc))
}

/// `WIDTH_BUCKET(operand, low, high, count)` — the 1-based histogram bucket `operand` lands in
/// across `count` equi-width buckets spanning `[low, high)` (SQL:2003). Returns `0` below the
/// range and `count + 1` at/above it; the bounds may be given in descending order. Errors on a
/// non-positive `count`, equal bounds, or a NaN argument (matching the standard's domain rules).
fn width_bucket(operand: f64, low: f64, high: f64, count: i64) -> Result<ast::Value, Error> {
    if count <= 0 {
        return Err(Error::Unsupported(
            "width_bucket: count must be greater than zero".to_owned(),
        ));
    }
    if operand.is_nan() || low.is_nan() || high.is_nan() {
        return Err(Error::Unsupported(
            "width_bucket: operand and bounds must not be NaN".to_owned(),
        ));
    }
    if low == high {
        return Err(Error::Unsupported(
            "width_bucket: lower bound must not equal upper bound".to_owned(),
        ));
    }
    // The interior buckets land in 1..=count, so the `floor()` result is bounded by `count` and the
    // `as i64` truncation cannot lose magnitude.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "bucket index is bounded by count; floor() stays within i64"
    )]
    let bucket = if low < high {
        if operand < low {
            0
        } else if operand >= high {
            count + 1
        } else {
            1 + ((operand - low) / (high - low) * count as f64).floor() as i64
        }
    } else if operand > low {
        0
    } else if operand <= high {
        count + 1
    } else {
        1 + ((low - operand) / (low - high) * count as f64).floor() as i64
    };
    Ok(ast::Value::Int(bucket))
}

/// Character count of `s` as an `i64` (saturating; strings never approach `i64::MAX` characters).
fn char_len(s: &str) -> i64 {
    i64::try_from(s.chars().count()).unwrap_or(i64::MAX)
}

/// 1-based `SUBSTRING` over characters. For the 3-argument form the result window is
/// `[start, start + length)`; positions before 1 are clipped but still consume the window, matching
/// SQL semantics (`SUBSTRING('abcdef', -2, 5)` → `'ab'`). A negative `length` is an error.
fn substring(s: &str, start: i64, length: Option<i64>) -> Result<ast::Value, Error> {
    let chars: Vec<char> = s.chars().collect();
    let n = char_len(s);
    let end_excl = match length {
        Some(l) => {
            if l < 0 {
                return Err(Error::Unsupported(
                    "SUBSTRING length must be non-negative".to_owned(),
                ));
            }
            start.saturating_add(l)
        },
        None => n.saturating_add(1),
    };
    let real_start = start.max(1);
    let real_end = end_excl.min(n.saturating_add(1));
    if real_end <= real_start {
        return Ok(ast::Value::Text(String::new()));
    }
    let lo = usize::try_from(real_start - 1).unwrap_or(0);
    let hi = usize::try_from(real_end - 1).unwrap_or(0);
    let out: String = chars
        .get(lo..hi)
        .map(|sl| sl.iter().collect())
        .unwrap_or_default();
    Ok(ast::Value::Text(out))
}

/// Replace every occurrence of `from` in `s` with `to`. An empty `from` leaves `s` unchanged (SQL
/// semantics; avoids the per-character insertion `str::replace` would otherwise do).
fn replace(s: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return s.to_owned();
    }
    s.replace(from, to)
}

/// 1-based character index of the first occurrence of `sub` in `hay`, or `0` if absent. An empty
/// `sub` matches at position 1 (SQL semantics).
fn position(sub: &str, hay: &str) -> i64 {
    if sub.is_empty() {
        return 1;
    }
    hay.find(sub).map_or(0, |byte_idx| {
        hay.get(..byte_idx).map_or(0, |prefix| {
            i64::try_from(prefix.chars().count() + 1).unwrap_or(i64::MAX)
        })
    })
}

/// Pad (or truncate) `s` to `target_len` characters using `fill`. A non-positive length yields the
/// empty string; an input at least as long as the target is truncated to its first `target_len`
/// characters; an empty `fill` cannot extend a short input, so it is returned unchanged.
fn pad(s: &str, target_len: i64, fill: &str, side: PadSide) -> String {
    let target = usize::try_from(target_len).unwrap_or(0);
    let chars: Vec<char> = s.chars().collect();
    if chars.len() >= target {
        return chars.into_iter().take(target).collect();
    }
    let fill_chars: Vec<char> = fill.chars().collect();
    if fill_chars.is_empty() {
        return chars.into_iter().collect();
    }
    let pad_needed = target - chars.len();
    let padding = fill_chars.into_iter().cycle().take(pad_needed);
    match side {
        PadSide::Left => padding.chain(chars).collect(),
        PadSide::Right => chars.into_iter().chain(padding).collect(),
    }
}

/// Strip characters from the leading/trailing side(s) of `s`. With no `set`, Unicode whitespace is
/// stripped; otherwise any character contained in `set` is stripped from the relevant side(s).
fn trim(s: &str, set: Option<&str>, side: TrimSide) -> String {
    set.map_or_else(
        || {
            match side {
                TrimSide::Left => s.trim_start(),
                TrimSide::Right => s.trim_end(),
                TrimSide::Both => s.trim(),
            }
            .to_owned()
        },
        |set| {
            let in_set = |c: char| set.contains(c);
            match side {
                TrimSide::Left => s.trim_start_matches(in_set),
                TrimSide::Right => s.trim_end_matches(in_set),
                TrimSide::Both => s.trim_matches(in_set),
            }
            .to_owned()
        },
    )
}

/// Return the first non-NULL argument; `NULL` if every argument is `NULL`.
/// Widen a numeric value to a type-unifying node's declared numeric type.
///
/// A `CASE`/`COALESCE` whose branches mix numerics is typed by `unify_result_ty` (FLOAT dominates,
/// then NUMERIC over INT), but each branch evaluates to its own value variant — so the chosen value
/// can be a different numeric variant than the node's declared type (e.g. a `FLOAT`-typed
/// `COALESCE(f, 0.5)` returning the `NUMERIC` literal `0.5`). Coercing here makes the value match
/// the declared type, so the row and vectorized paths agree and downstream typed-column conversion
/// sees the expected variant. A no-op when the value already matches (the common case) or for
/// non-numeric types.
fn coerce_numeric_to(v: ast::Value, ty: ColumnType) -> ast::Value {
    match (&v, ty) {
        (ast::Value::Int(i), ColumnType::Float) => ast::Value::Float(*i as f64),
        (ast::Value::Numeric(d), ColumnType::Float) => ast::Value::Float(d.to_f64()),
        (ast::Value::Int(i), ColumnType::Numeric { .. }) => {
            ast::Value::Numeric(crate::numeric::Decimal::from_i64(*i))
        },
        _ => v,
    }
}

fn eval_coalesce(args: &[TypedExpr], row: &Row) -> Result<ast::Value, Error> {
    for arg in args {
        let v = eval(arg, row)?;
        if !matches!(v, ast::Value::Null) {
            return Ok(v);
        }
    }
    Ok(ast::Value::Null)
}

/// Evaluate a 1-based array subscript `base[index]`. A `NULL` array or index yields `NULL`,
/// and an out-of-range index (including `< 1`) yields `NULL` per SQL array semantics.
fn eval_subscript(base: &TypedExpr, index: &TypedExpr, row: &Row) -> Result<ast::Value, Error> {
    let base_v = eval(base, row)?;
    let index_v = eval(index, row)?;
    let (ast::Value::Array(items), ast::Value::Int(i)) = (&base_v, &index_v) else {
        // NULL array / NULL index (or a non-array post-analyzer) → NULL.
        return Ok(ast::Value::Null);
    };
    // SQL arrays are 1-based; index 0 or negative, or past the end, is NULL (not an error).
    // `checked_sub` keeps `i == i64::MIN` from overflowing the 1-based → 0-based conversion.
    let element = (*i)
        .checked_sub(1)
        .and_then(|j| usize::try_from(j).ok())
        .and_then(|j| items.get(j))
        .cloned()
        .unwrap_or(ast::Value::Null);
    Ok(element)
}

/// Evaluate a `base[lower:upper]` array slice (B-fn): a 1-based, inclusive sub-array. An omitted bound
/// defaults to the array's first (`lower`) / last (`upper`) element; bounds are clamped to the array
/// so an out-of-range slice yields the in-range overlap (or an empty array), matching SQL array
/// semantics. A `NULL` array or a present-but-`NULL` bound yields `NULL`.
fn eval_array_slice(
    base: &TypedExpr,
    lower: Option<&TypedExpr>,
    upper: Option<&TypedExpr>,
    row: &Row,
) -> Result<ast::Value, Error> {
    let ast::Value::Array(items) = eval(base, row)? else {
        return Ok(ast::Value::Null);
    };
    // Resolve an optional bound to a concrete 1-based index; a present-but-NULL bound makes the whole
    // slice NULL (like a NULL subscript), an absent bound uses `default`.
    let resolve = |b: Option<&TypedExpr>, default: i64| -> Result<Option<i64>, Error> {
        match b {
            None => Ok(Some(default)),
            Some(expr) => match eval(expr, row)? {
                ast::Value::Int(n) => Ok(Some(n)),
                ast::Value::Null => Ok(None),
                _ => Ok(Some(default)),
            },
        }
    };
    let len = i64::try_from(items.len()).unwrap_or(i64::MAX);
    let (Some(lo), Some(hi)) = (resolve(lower, 1)?, resolve(upper, len)?) else {
        return Ok(ast::Value::Null);
    };
    // Clamp to the 1-based `[1, len]` range, then take the inclusive span; `lo > hi` is empty.
    let lo = lo.max(1);
    let hi = hi.min(len);
    let slice = if lo > hi {
        Vec::new()
    } else {
        // `lo >= 1` and `hi <= len`, so both conversions are in range.
        let start = usize::try_from(lo - 1).unwrap_or(0);
        let end = usize::try_from(hi).unwrap_or(0);
        items.get(start..end).map(<[_]>::to_vec).unwrap_or_default()
    };
    Ok(ast::Value::Array(slice))
}

/// Runtime type conversion. `NULL` is preserved across any cast; otherwise
/// the supported coercions are: same-type identity; numeric widening / narrowing
/// between `Int` and `Float`; `Bool ↔ Int`; conversion to `Text` for any
/// scalar; and parsing `Text` into `Bool`/`Int`/`Float`. Unsupported pairs
/// (e.g. `Bool → Float`, anything involving `Bytes`/`Timestamp` non-trivially)
/// surface as `Error::Unsupported`.
fn eval_cast(
    expr: &TypedExpr,
    target: ColumnType,
    row: &Row,
    try_cast: bool,
) -> Result<ast::Value, Error> {
    let value = eval(expr, row)?;
    match cast_value(value, target) {
        // TRY_CAST/SAFE_CAST swallow a failed *conversion* (not an error from evaluating the
        // operand, which already propagated above) and yield NULL instead.
        Err(_) if try_cast => Ok(ast::Value::Null),
        result => result,
    }
}

/// Cast a single already-evaluated [`ast::Value`] to `target`.
///
/// Shared by `CAST(expr AS type)` evaluation and the `ALTER COLUMN … TYPE` row
/// rewrite. `NULL` casts to `NULL` for every target; an impossible conversion
/// (e.g. non-numeric text into `INT`) surfaces as a typed error rather than a
/// silent default.
#[allow(
    clippy::too_many_lines,
    reason = "one arm per (value, target) type pair; flatter than dispatching to per-type helpers"
)]
pub(super) fn cast_value(value: ast::Value, target: ColumnType) -> Result<ast::Value, Error> {
    if matches!(value, ast::Value::Null) {
        return Ok(ast::Value::Null);
    }
    // The value is converted against the physical target (a length-checked `VARCHAR(n)`/`CHAR(n)` cast
    // is desugared to `substring(cast(x AS text), 1, n)` in the parser and never reaches here, so
    // collapsing VarChar/Char → Text is harmless). The *declared* width is kept so a narrowing integer
    // cast can enforce its range below.
    let declared = target;
    let target = target.physical();
    let result = match (&value, target) {
        // Identity casts (incl. the temporal + UUID types).
        (ast::Value::Bool(_), ColumnType::Bool)
        | (ast::Value::Int(_), ColumnType::Int)
        | (ast::Value::Float(_), ColumnType::Float)
        | (ast::Value::Text(_), ColumnType::Text)
        | (ast::Value::Date(_), ColumnType::Date)
        | (ast::Value::Time(_), ColumnType::Time)
        | (ast::Value::TimeTz(_), ColumnType::TimeTz)
        | (ast::Value::Timestamp(_), ColumnType::Timestamp)
        | (ast::Value::TimestampTz(_), ColumnType::TimestampTz)
        | (ast::Value::Uuid(_), ColumnType::Uuid)
        | (ast::Value::Json(_), ColumnType::Json)
        | (ast::Value::Interval(_), ColumnType::Interval)
        | (ast::Value::Bytes(_), ColumnType::Bytes)
        | (ast::Value::Array(_), ColumnType::Array(_)) => Ok(value),
        // Numeric widening / narrowing.
        (ast::Value::Int(i), ColumnType::Float) => Ok(ast::Value::Float(*i as f64)),
        (ast::Value::Float(f), ColumnType::Int) => {
            // Round half-away-from-zero, then bounds-check. A bare `*f as i64` saturates —
            // NaN→0, ±inf/overflow→i64::MIN/MAX — yielding a *wrong value as success*; SQL requires a
            // non-finite or out-of-range float→int cast to error. `[i64::MIN, 2^63)` is the exact
            // representable range; `2^63` itself (the rounded-up `i64::MAX as f64`) is out of range.
            let rounded = f.round();
            if !rounded.is_finite()
                || rounded < i64::MIN as f64
                || rounded >= 9_223_372_036_854_775_808.0
            {
                return Err(invalid_cast(&f.to_string(), ColumnType::Int));
            }
            Ok(ast::Value::Int(rounded as i64))
        },
        // Bool ↔ Int.
        (ast::Value::Bool(b), ColumnType::Int) => Ok(ast::Value::Int(i64::from(*b))),
        (ast::Value::Int(i), ColumnType::Bool) => Ok(ast::Value::Bool(*i != 0)),
        // To Text.
        (ast::Value::Bool(b), ColumnType::Text) => Ok(ast::Value::Text(b.to_string())),
        (ast::Value::Int(i), ColumnType::Text) => Ok(ast::Value::Text(i.to_string())),
        (ast::Value::Float(f), ColumnType::Text) => Ok(ast::Value::Text(f.to_string())),
        // From Text — parsing failures become a typed error.
        (ast::Value::Text(s), ColumnType::Int) => s
            .trim()
            .parse::<i64>()
            .map(ast::Value::Int)
            .map_err(|_| Error::TypeMismatch {
                context: format!("CAST `{s}` AS INT"),
                expected: ColumnType::Int,
                found: ColumnType::Text,
            }),
        (ast::Value::Text(s), ColumnType::Float) => s
            .trim()
            .parse::<f64>()
            .map(ast::Value::Float)
            .map_err(|_| Error::TypeMismatch {
                context: format!("CAST `{s}` AS FLOAT"),
                expected: ColumnType::Float,
                found: ColumnType::Text,
            }),
        // The full set of the reference engine's boolean string inputs (case-insensitive, surrounding whitespace ignored):
        // true/false, t/f, yes/no, y/n, on/off, 1/0.
        (ast::Value::Text(s), ColumnType::Bool) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "t" | "yes" | "y" | "on" | "1" => Ok(ast::Value::Bool(true)),
            "false" | "f" | "no" | "n" | "off" | "0" => Ok(ast::Value::Bool(false)),
            _ => Err(Error::TypeMismatch {
                context: format!("CAST `{s}` AS BOOL"),
                expected: ColumnType::Bool,
                found: ColumnType::Text,
            }),
        },
        // Parse from text (`'2024-01-15'::DATE`, `CAST('…' AS UUID)`, …)
        (ast::Value::Text(s), ColumnType::Date) => crate::temporal::parse_date(s)
            .map(ast::Value::Date)
            .ok_or_else(|| invalid_cast(s, ColumnType::Date)),
        (ast::Value::Text(s), ColumnType::Time) => crate::temporal::parse_time(s)
            .map(ast::Value::Time)
            .ok_or_else(|| invalid_cast(s, ColumnType::Time)),
        (ast::Value::Text(s), ColumnType::TimeTz) => crate::temporal::parse_timetz(s)
            .map(ast::Value::TimeTz)
            .ok_or_else(|| invalid_cast(s, ColumnType::TimeTz)),
        (ast::Value::Text(s), ColumnType::Timestamp) => crate::temporal::parse_timestamp(s)
            .map(ast::Value::Timestamp)
            .ok_or_else(|| invalid_cast(s, ColumnType::Timestamp)),
        (ast::Value::Text(s), ColumnType::TimestampTz) => crate::temporal::parse_timestamptz(s)
            .map(ast::Value::TimestampTz)
            .ok_or_else(|| invalid_cast(s, ColumnType::TimestampTz)),
        (ast::Value::Text(s), ColumnType::Uuid) => crate::temporal::parse_uuid(s)
            .map(ast::Value::Uuid)
            .ok_or_else(|| invalid_cast(s, ColumnType::Uuid)),
        // Render temporal + UUID back to their canonical text form.
        (ast::Value::Date(d), ColumnType::Text) => {
            Ok(ast::Value::Text(crate::temporal::format_date(*d)))
        },
        (ast::Value::Time(t), ColumnType::Text) => {
            Ok(ast::Value::Text(crate::temporal::format_time(*t)))
        },
        (ast::Value::TimeTz(t), ColumnType::Text) => {
            Ok(ast::Value::Text(crate::temporal::format_timetz(*t)))
        },
        (ast::Value::Timestamp(t), ColumnType::Text) => {
            Ok(ast::Value::Text(crate::temporal::format_timestamp(*t)))
        },
        (ast::Value::TimestampTz(t), ColumnType::Text) => {
            Ok(ast::Value::Text(crate::temporal::format_timestamptz(*t)))
        },
        (ast::Value::Uuid(u), ColumnType::Text) => {
            Ok(ast::Value::Text(crate::temporal::format_uuid(u)))
        },
        // Temporal narrowing/widening (QA category-D): a TIMESTAMP[TZ] splits into its DATE
        // (floor of whole days since the epoch) and TIME-of-day (micros within the day); a DATE
        // widens to midnight. `div_euclid`/`rem_euclid` floor toward negative infinity so dates
        // before the epoch land on the correct day with a non-negative time-of-day.
        (ast::Value::Timestamp(t) | ast::Value::TimestampTz(t), ColumnType::Date) => {
            i32::try_from(t.div_euclid(MICROS_PER_DAY))
                .map(ast::Value::Date)
                .map_err(|_| invalid_cast(&t.to_string(), ColumnType::Date))
        },
        (ast::Value::Timestamp(t) | ast::Value::TimestampTz(t), ColumnType::Time) => {
            Ok(ast::Value::Time(t.rem_euclid(MICROS_PER_DAY)))
        },
        (ast::Value::Date(d), ColumnType::Timestamp) => i64::from(*d)
            .checked_mul(MICROS_PER_DAY)
            .map(ast::Value::Timestamp)
            .ok_or_else(|| invalid_cast(&crate::temporal::format_date(*d), ColumnType::Timestamp)),
        (ast::Value::Date(d), ColumnType::TimestampTz) => i64::from(*d)
            .checked_mul(MICROS_PER_DAY)
            .map(ast::Value::TimestampTz)
            .ok_or_else(|| {
                invalid_cast(&crate::temporal::format_date(*d), ColumnType::TimestampTz)
            }),
        // NUMERIC: cast into NUMERIC (rescaling to the target scale when constrained), and
        // out of NUMERIC to Int / Float / Text.
        (ast::Value::Numeric(d), ColumnType::Numeric { precision, scale }) => {
            Ok(ast::Value::Numeric(cast_to_numeric(*d, precision, scale)?))
        },
        (ast::Value::Int(i), ColumnType::Numeric { precision, scale }) => Ok(ast::Value::Numeric(
            cast_to_numeric(crate::numeric::Decimal::from_i64(*i), precision, scale)?,
        )),
        (ast::Value::Float(f), ColumnType::Numeric { precision, scale }) => {
            let d = crate::numeric::from_f64_text(*f)
                .ok_or_else(|| invalid_cast(&f.to_string(), target))?;
            Ok(ast::Value::Numeric(cast_to_numeric(d, precision, scale)?))
        },
        (ast::Value::Text(s), ColumnType::Numeric { precision, scale }) => {
            let d = crate::numeric::Decimal::parse(s).ok_or_else(|| invalid_cast(s, target))?;
            Ok(ast::Value::Numeric(cast_to_numeric(d, precision, scale)?))
        },
        // SQL `CAST(numeric AS integer)` rounds half-away-from-zero (`2.6 -> 3`, `-2.5 -> -3`),
        // matching the float -> int cast above — not truncation toward zero (QA differential).
        (ast::Value::Numeric(d), ColumnType::Int) => d
            .to_i64_rounded()
            .map(ast::Value::Int)
            .ok_or_else(|| invalid_cast(&d.format(), ColumnType::Int)),
        (ast::Value::Numeric(d), ColumnType::Float) => Ok(ast::Value::Float(d.to_f64())),
        (ast::Value::Numeric(d), ColumnType::Text) => Ok(ast::Value::Text(d.format())),
        // JSON: text -> json (parse + canonicalize), json -> text (canonical).
        (ast::Value::Text(s), ColumnType::Json) => crate::json::canonicalize(s)
            .map(ast::Value::Json)
            .ok_or_else(|| invalid_cast(s, ColumnType::Json)),
        // `json::text` renders the spaced display form (`{"a": 1}`), like the standard jsonb output.
        (ast::Value::Json(s), ColumnType::Text) => {
            Ok(ast::Value::Text(crate::json::display_form(s)))
        },
        // INTERVAL: text -> interval (parse), interval -> text (canonical form).
        (ast::Value::Text(s), ColumnType::Interval) => crate::interval::Interval::parse(s)
            .map(ast::Value::Interval)
            .ok_or_else(|| invalid_cast(s, ColumnType::Interval)),
        (ast::Value::Interval(iv), ColumnType::Text) => Ok(ast::Value::Text(iv.format())),
        // ARRAY: array -> text (`{...}` form); identity handled above.
        (ast::Value::Array(items), ColumnType::Text) => {
            Ok(ast::Value::Text(crate::display::array_text(items)))
        },
        // text `{...}` -> array: parse, then cast each element to the array's element type.
        (ast::Value::Text(s), ColumnType::Array(elem)) => {
            Ok(ast::Value::Array(parse_text_array(s, elem.column_type())?))
        },
        // VECTOR: `'[..]'::VECTOR(n)` text→vector with a dimension check; identity;
        // vector→text renders the `[..]` form.
        (ast::Value::Vector(v), ColumnType::Vector(dim)) => {
            check_vector_dim(v.len(), dim)?;
            Ok(value)
        },
        (ast::Value::Text(s), ColumnType::Vector(dim)) => {
            let v = crate::vector::parse(s).ok_or_else(|| invalid_cast(s, target))?;
            check_vector_dim(v.len(), dim).map(|()| ast::Value::Vector(v))
        },
        (ast::Value::Vector(v), ColumnType::Text) => Ok(ast::Value::Text(crate::vector::format(v))),
        // BYTEA: text `\x<hex>` → bytes; bytes → `\x<hex>` text. (Identity handled above.)
        (ast::Value::Text(s), ColumnType::Bytes) => parse_bytea(s)
            .map(ast::Value::Bytes)
            .ok_or_else(|| invalid_cast(s, target)),
        (ast::Value::Bytes(b), ColumnType::Text) => {
            Ok(ast::Value::Text(crate::display::bytea_hex(b)))
        },
        // JSON scalar casts: a bare JSON number or boolean re-parses
        // through the ordinary text cast to the numeric/bool target — `('[1,2]'::jsonb->0)::int`.
        // Objects, arrays, quoted strings, and json `null` are refused loudly (as the reference
        // does: a jsonb string does not cast to a number).
        (
            ast::Value::Json(j),
            ColumnType::Int
            | ColumnType::SmallInt
            | ColumnType::BigInt
            | ColumnType::Float
            | ColumnType::Real
            | ColumnType::Numeric { .. }
            | ColumnType::Bool,
        ) => {
            let t = j.trim();
            if t.is_empty() || t == "null" || t.starts_with(['{', '[', '"']) {
                return Err(invalid_cast(j, target));
            }
            cast_value(ast::Value::Text(t.to_owned()), target)
        },
        _ => Err(Error::Unsupported(format!(
            "CAST from {:?} to {:?} not supported",
            runtime_type(&value),
            target,
        ))),
    }?;
    // A narrowing integer cast (`::int`/`::int4`/`::smallint`/`::int2`) errors when the value overflows
    // the target width, like the reference engine (`9999999999::int`) and the storage-side int range check.
    // BIGINT and non-integer targets impose no extra bound.
    if let ast::Value::Int(i) = result
        && int_value_bounds(declared).is_some_and(|(lo, hi)| i < lo || i > hi)
    {
        return Err(Error::IntegerOutOfRange);
    }
    Ok(result)
}

/// Parse a `{...}` array text literal and cast each element to `elem_ty`. Shared
/// by `CAST(... AS T[])` and the row encoder's text→array coercion (one parse+cast path).
pub(super) fn parse_text_array(s: &str, elem_ty: ColumnType) -> Result<Vec<ast::Value>, Error> {
    crate::executor::row::parse_array_text(s)
        .ok_or_else(|| invalid_cast(s, ColumnType::Array(elem_array_marker(elem_ty))))?
        .into_iter()
        .map(|tok| cast_value(tok, elem_ty))
        .collect()
}

/// Best-effort element marker for an array error message; defaults to `Int` for non-scalar.
fn elem_array_marker(elem_ty: ColumnType) -> nusadb_core::engine::ArrayElem {
    nusadb_core::engine::ArrayElem::from_column_type(elem_ty)
        .unwrap_or(nusadb_core::engine::ArrayElem::Int)
}

/// Reject a vector whose component count does not match the column's declared `VECTOR(dim)`.
fn check_vector_dim(got: usize, dim: u32) -> Result<(), Error> {
    if got as u64 == u64::from(dim) {
        Ok(())
    } else {
        Err(Error::InvalidValue {
            ty: ColumnType::Vector(dim),
            value: format!("{got}-dimensional vector"),
        })
    }
}

/// Build the error for a text value that does not parse as a temporal / UUID `target`.
fn invalid_cast(s: &str, target: ColumnType) -> Error {
    Error::InvalidValue {
        ty: target,
        value: s.to_owned(),
    }
}

/// Rescale a decimal to a NUMERIC cast target: to the declared `scale` when constrained
/// (`precision > 0`, also enforcing the precision), otherwise unchanged.
fn cast_to_numeric(
    d: crate::numeric::Decimal,
    precision: u8,
    scale: u8,
) -> Result<crate::numeric::Decimal, Error> {
    let ty = ColumnType::Numeric { precision, scale };
    if precision == 0 {
        return Ok(d);
    }
    let rescaled = d
        .rescale(scale)
        .ok_or_else(|| invalid_cast(&d.format(), ty))?;
    if rescaled.required_precision() > u32::from(precision) {
        return Err(invalid_cast(&d.format(), ty));
    }
    Ok(rescaled)
}

/// `ENCODE(bytea, format)` — render raw bytes as text in the `hex` (lowercase, no `\x` prefix) or
/// `escape` (printable bytes literal, others as `\nnn` octal, backslash doubled) format (B-fn).
fn encode_bytea(bytes: &[u8], format: &str) -> Result<String, Error> {
    match format.to_ascii_lowercase().as_str() {
        "hex" => {
            let mut out = String::with_capacity(bytes.len() * 2);
            for b in bytes {
                out.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
                out.push(char::from_digit(u32::from(b & 0x0f), 16).unwrap_or('0'));
            }
            Ok(out)
        },
        "escape" => {
            let mut out = String::with_capacity(bytes.len());
            for &b in bytes {
                match b {
                    b'\\' => out.push_str("\\\\"),
                    0x20..=0x7e => out.push(b as char),
                    other => {
                        out.push('\\');
                        out.push(char::from_digit(u32::from(other >> 6) & 0x7, 8).unwrap_or('0'));
                        out.push(char::from_digit(u32::from(other >> 3) & 0x7, 8).unwrap_or('0'));
                        out.push(char::from_digit(u32::from(other) & 0x7, 8).unwrap_or('0'));
                    },
                }
            }
            Ok(out)
        },
        "base64" => Ok(base64_encode(bytes)),
        other => Err(Error::Unsupported(format!(
            "ENCODE format {other:?} (supported: hex, escape, base64)"
        ))),
    }
}

/// The standard base-64 alphabet (RFC 4648 §4): `A-Z`, `a-z`, `0-9`, `+`, `/`, with `=` padding.
const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode raw bytes as standard base-64 text (RFC 4648, with `=` padding) — the `base64` form of
/// `ENCODE` (B-fn). Each 3-byte group becomes 4 output characters; a 1- or 2-byte tail is padded.
fn base64_encode(bytes: &[u8]) -> String {
    // Map a 6-bit group to its alphabet symbol. The `& 0x3f` keeps the index in `0..64`, so the
    // lookup always hits; the `unwrap_or` is unreachable and only satisfies the no-indexing lint.
    let sym = |six: u32| {
        BASE64_ALPHABET
            .get(six as usize & 0x3f)
            .copied()
            .unwrap_or(b'A') as char
    };
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk.first().copied().unwrap_or(0));
        let b1 = chunk.get(1).copied().map_or(0, u32::from);
        let b2 = chunk.get(2).copied().map_or(0, u32::from);
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(sym(n >> 18));
        out.push(sym(n >> 12));
        out.push(if chunk.len() > 1 { sym(n >> 6) } else { '=' });
        out.push(if chunk.len() > 2 { sym(n) } else { '=' });
    }
    out
}

/// Decode standard base-64 text (RFC 4648) into raw bytes — the `base64` form of `DECODE` (B-fn).
/// ASCII whitespace between characters is ignored; any other non-alphabet byte is an error.
fn base64_decode(text: &str) -> Result<Vec<u8>, Error> {
    /// Map an alphabet byte to its 6-bit value, or `None` for a non-alphabet byte.
    fn sextet(b: u8) -> Option<u32> {
        match b {
            b'A'..=b'Z' => Some(u32::from(b - b'A')),
            b'a'..=b'z' => Some(u32::from(b - b'a') + 26),
            b'0'..=b'9' => Some(u32::from(b - b'0') + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut acc: u32 = 0;
    let mut bits = 0_u32;
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    for &b in text.as_bytes() {
        if b == b'=' || b.is_ascii_whitespace() {
            continue;
        }
        let v = sextet(b).ok_or_else(|| {
            Error::Unsupported(format!("DECODE base64: invalid character {:?}", b as char))
        })?;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            // The mask constrains the value to `0..256`, so `try_from` never fails.
            out.push(u8::try_from((acc >> bits) & 0xff).unwrap_or(0));
        }
    }
    Ok(out)
}

/// `DECODE(text, format)` — parse text in the `hex` or `escape` format into raw bytes; the inverse of
/// [`encode_bytea`] (B-fn).
fn decode_bytea(s: &str, format: &str) -> Result<Vec<u8>, Error> {
    match format.to_ascii_lowercase().as_str() {
        "hex" => {
            let hex: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
            if !hex.len().is_multiple_of(2) {
                return Err(Error::Unsupported(
                    "DECODE hex: odd number of digits".to_owned(),
                ));
            }
            let mut out = Vec::with_capacity(hex.len() / 2);
            let mut chars = hex.chars();
            while let (Some(a), Some(b)) = (chars.next(), chars.next()) {
                let hi = a
                    .to_digit(16)
                    .ok_or_else(|| Error::Unsupported("DECODE hex: invalid digit".to_owned()))?;
                let lo = b
                    .to_digit(16)
                    .ok_or_else(|| Error::Unsupported("DECODE hex: invalid digit".to_owned()))?;
                out.push(u8::try_from(hi * 16 + lo).unwrap_or(0));
            }
            Ok(out)
        },
        "escape" => {
            let mut out = Vec::with_capacity(s.len());
            let mut chars = s.chars().peekable();
            while let Some(c) = chars.next() {
                if c != '\\' {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    continue;
                }
                match chars.peek() {
                    Some('\\') => {
                        chars.next();
                        out.push(b'\\');
                    },
                    Some(d) if d.is_digit(8) => {
                        let mut val = 0u32;
                        for _ in 0..3 {
                            match chars.peek().and_then(|c| c.to_digit(8)) {
                                Some(d) => {
                                    val = val * 8 + d;
                                    chars.next();
                                },
                                None => break,
                            }
                        }
                        out.push(u8::try_from(val).map_err(|_| {
                            Error::Unsupported("DECODE escape: octal value out of range".to_owned())
                        })?);
                    },
                    _ => out.push(b'\\'),
                }
            }
            Ok(out)
        },
        "base64" => base64_decode(s),
        other => Err(Error::Unsupported(format!(
            "DECODE format {other:?} (supported: hex, escape, base64)"
        ))),
    }
}

/// Parse a `BYTEA` text value in the standard `\x<hex>` input form into raw bytes. Returns
/// `None` for a missing `\x` prefix, an odd hex length, or a non-hex digit. The inverse of
/// [`crate::display::bytea_hex`].
pub(crate) fn parse_bytea(s: &str) -> Option<Vec<u8>> {
    let hex = s.strip_prefix("\\x")?;
    if hex.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut chars = hex.chars();
    while let (Some(a), Some(b)) = (chars.next(), chars.next()) {
        let hi = a.to_digit(16)?;
        let lo = b.to_digit(16)?;
        out.push(u8::try_from(hi * 16 + lo).ok()?);
    }
    Some(out)
}

fn runtime_type(v: &ast::Value) -> ColumnType {
    match v {
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
        ast::Value::Array(items) => ColumnType::Array(crate::executor::row::array_elem_of(items)),
        #[allow(
            clippy::cast_possible_truncation,
            reason = "vector dim fits u32 by construction"
        )]
        ast::Value::Vector(v) => ColumnType::Vector(v.len() as u32),
        ast::Value::Bytes(_) => ColumnType::Bytes,
    }
}

/// Evaluate a `CASE` expression. Simple form: compare `operand` against each
/// branch's `when`; first equal wins. Searched form: first `when` that
/// evaluates to `Bool(true)` wins. No match + no `ELSE` → `NULL`.
fn eval_case(
    operand: Option<&TypedExpr>,
    branches: &[TypedCaseBranch],
    default: Option<&TypedExpr>,
    row: &Row,
) -> Result<ast::Value, Error> {
    let operand_value = match operand {
        Some(expr) => Some(eval(expr, row)?),
        None => None,
    };
    for branch in branches {
        let when_value = eval(&branch.when, row)?;
        let matched = operand_value.as_ref().map_or_else(
            // Searched form: only literal Bool(true) qualifies; NULL is "no".
            || matches!(when_value, ast::Value::Bool(true)),
            // Simple form: NULL on either side is "no match" (SQL semantics).
            |op| {
                !matches!(op, ast::Value::Null)
                    && !matches!(when_value, ast::Value::Null)
                    && compare(op, &when_value) == Ordering::Equal
            },
        );
        if matched {
            return eval(&branch.then, row);
        }
    }
    default.map_or(Ok(ast::Value::Null), |expr| eval(expr, row))
}

/// `expr LIKE pattern`: `NULL` in either operand → `NULL`. Otherwise match
/// `%` (any run of chars) and `_` (one char) literally. `negated` flips the
/// result for `NOT LIKE`.
fn eval_like(
    expr: &TypedExpr,
    pattern: &TypedExpr,
    negated: bool,
    escape: Option<char>,
    case_insensitive: bool,
    row: &Row,
) -> Result<ast::Value, Error> {
    let subject = eval(expr, row)?;
    let pat = eval(pattern, row)?;
    if matches!(subject, ast::Value::Null) || matches!(pat, ast::Value::Null) {
        return Ok(ast::Value::Null);
    }
    let (ast::Value::Text(subject_s), ast::Value::Text(pat_s)) = (&subject, &pat) else {
        // Type-incompatible post-analyzer — defensive fallback.
        return Ok(ast::Value::Null);
    };
    let matched = like_match(subject_s, pat_s, escape, case_insensitive);
    Ok(ast::Value::Bool(matched ^ negated))
}

/// Evaluate `subject ~ pattern` / `~*` / `!~` / `!~*`: `NULL` in either operand → `NULL`.
/// The pattern is compiled with the `i` flag for the case-insensitive forms; an invalid pattern
/// surfaces as [`Error::InvalidRegex`]. The match result is flipped for the negated forms.
fn eval_regex_match(
    expr: &TypedExpr,
    pattern: &TypedExpr,
    case_sensitive: bool,
    negated: bool,
    row: &Row,
) -> Result<ast::Value, Error> {
    let subject = eval(expr, row)?;
    let pat = eval(pattern, row)?;
    if matches!(subject, ast::Value::Null) || matches!(pat, ast::Value::Null) {
        return Ok(ast::Value::Null);
    }
    let (ast::Value::Text(subject_s), ast::Value::Text(pat_s)) = (&subject, &pat) else {
        // Type-incompatible post-analyzer — defensive fallback.
        return Ok(ast::Value::Null);
    };
    let flags = if case_sensitive { "" } else { "i" };
    let (re, _) = compile_regex(pat_s, flags)?;
    Ok(ast::Value::Bool(re.is_match(subject_s) ^ negated))
}

/// Evaluate `expr [NOT] SIMILAR TO pattern`. `NULL` on either side yields `NULL`. The SQL
/// `SIMILAR TO` pattern is translated to an anchored POSIX regex (see [`similar_to_regex`]) and
/// matched against the whole subject; `negated` flips the result.
fn eval_similar_to(
    expr: &TypedExpr,
    pattern: &TypedExpr,
    negated: bool,
    row: &Row,
) -> Result<ast::Value, Error> {
    let subject = eval(expr, row)?;
    let pat = eval(pattern, row)?;
    if matches!(subject, ast::Value::Null) || matches!(pat, ast::Value::Null) {
        return Ok(ast::Value::Null);
    }
    let (ast::Value::Text(subject_s), ast::Value::Text(pat_s)) = (&subject, &pat) else {
        // Type-incompatible post-analyzer — defensive fallback.
        return Ok(ast::Value::Null);
    };
    let (re, _) = compile_regex(&similar_to_regex(pat_s), "")?;
    Ok(ast::Value::Bool(re.is_match(subject_s) ^ negated))
}

/// Translate a SQL `SIMILAR TO` pattern into an **anchored** POSIX regex (`^(?:…)$`), so a match
/// covers the whole subject. `_`→`.` and `%`→`.*`; the regex metacharacters
/// `| * + ? ( ) { }` pass through with their regex meaning; a `[ … ]` bracket expression is copied
/// verbatim; the default escape character `\` makes the next character a literal; every other
/// character is emitted as a regex literal (so `.`, `^`, `$`, etc. match themselves).
fn similar_to_regex(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len() + 6);
    out.push_str("^(?:");
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            // Escape: the next character is taken literally (default escape `\`).
            '\\' => match chars.next() {
                Some(next) => out.push_str(&regex::escape(&next.to_string())),
                None => out.push_str("\\\\"),
            },
            '%' => out.push_str(".*"),
            '_' => out.push('.'),
            // SIMILAR-TO operators that map 1:1 onto regex syntax.
            '|' | '*' | '+' | '?' | '(' | ')' | '{' | '}' => out.push(c),
            // Bracket expression: copy through the closing `]` (regex char-class syntax matches).
            '[' => {
                out.push('[');
                for b in chars.by_ref() {
                    out.push(b);
                    if b == ']' {
                        break;
                    }
                }
            },
            // Any other character is a literal (escape it if the regex engine treats it specially).
            other => out.push_str(&regex::escape(&other.to_string())),
        }
    }
    out.push_str(")$");
    out
}

/// Two-pointer pattern matcher: `%` matches zero or more chars, `_` matches
/// exactly one char, every other character is literal. `O(n·m)` worst case,
/// linear on typical patterns.
///
/// `escape` is the `ESCAPE 'c'` character: in the pattern, `c` makes the next character a
/// literal (so `c%`, `c_`, `cc` match a literal `%`/`_`/`c`); a trailing `c` is itself a literal.
/// `None` (no `ESCAPE` clause) means no escaping — `%`/`_` are always wildcards.
///
/// Bounded-index discipline: every `pat[..]` / `text[..]` access is gated behind a `pi < pat.len()`
/// or `ti < text.len()` test in the same condition, so the indexing is provably in range —
/// `clippy::indexing_slicing` is muted with that justification.
#[allow(
    clippy::indexing_slicing,
    reason = "every index access is guarded by an explicit length check in the same condition"
)]
fn like_match(subject: &str, pattern: &str, escape: Option<char>, case_insensitive: bool) -> bool {
    let text: Vec<char> = subject.chars().collect();
    // Expand the pattern into `(char, is_literal)`: an escaped character is forced literal so it is
    // matched verbatim rather than as a `%`/`_` wildcard. With `escape == None` nothing is escaped,
    // reproducing the original wildcard-only behavior exactly.
    let raw: Vec<char> = pattern.chars().collect();
    let mut pat: Vec<(char, bool)> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if Some(raw[i]) == escape && i + 1 < raw.len() {
            pat.push((raw[i + 1], true));
            i += 2;
        } else {
            // A non-escape char, or a trailing escape char with nothing to escape: a literal.
            pat.push((raw[i], false));
            i += 1;
        }
    }
    let is_wild = |c: (char, bool), w: char| c == (w, false);
    // For `ILIKE`, a literal pattern character matches a source character case-insensitively. This is
    // folded per character (not by lower-casing the whole strings), so a `_` still matches exactly one
    // source character even when a letter's lowercase form has a different length, and the `ESCAPE`
    // character keeps its case (deep-gate #12).
    let char_eq =
        |p: char, s: char| p == s || (case_insensitive && p.to_lowercase().eq(s.to_lowercase()));
    let (mut ti, mut pi) = (0usize, 0usize);
    let (mut star_text, mut star_pat): (Option<usize>, Option<usize>) = (None, None);

    while ti < text.len() {
        if pi < pat.len() && is_wild(pat[pi], '%') {
            star_text = Some(ti);
            star_pat = Some(pi);
            pi += 1;
        } else if pi < pat.len() && (is_wild(pat[pi], '_') || char_eq(pat[pi].0, text[ti])) {
            ti += 1;
            pi += 1;
        } else if let (Some(s_ti), Some(s_pi)) = (star_text, star_pat) {
            ti = s_ti + 1;
            star_text = Some(s_ti + 1);
            pi = s_pi + 1;
        } else {
            return false;
        }
    }
    while pi < pat.len() && is_wild(pat[pi], '%') {
        pi += 1;
    }
    pi == pat.len()
}

/// Three-valued IN: TRUE if any item equals; FALSE if no items equal and no
/// NULL items; otherwise NULL. `negated` flips TRUE↔FALSE (NOT IN).
fn eval_in_list(
    expr: &TypedExpr,
    list: &[TypedExpr],
    negated: bool,
    row: &Row,
) -> Result<ast::Value, Error> {
    let target = eval(expr, row)?;
    if matches!(target, ast::Value::Null) {
        return Ok(ast::Value::Null);
    }
    let mut saw_null = false;
    for item in list {
        let candidate = eval(item, row)?;
        if matches!(candidate, ast::Value::Null) {
            saw_null = true;
            continue;
        }
        if compare(&target, &candidate) == Ordering::Equal {
            return Ok(ast::Value::Bool(!negated));
        }
    }
    if saw_null {
        Ok(ast::Value::Null)
    } else {
        Ok(ast::Value::Bool(negated))
    }
}

/// `expr BETWEEN low AND high` ≡ `low <= expr AND expr <= high`. NULL in any
/// operand → NULL. `negated` flips the final boolean.
fn eval_between(
    expr: &TypedExpr,
    low: &TypedExpr,
    high: &TypedExpr,
    negated: bool,
    row: &Row,
) -> Result<ast::Value, Error> {
    let v = eval(expr, row)?;
    let l = eval(low, row)?;
    let h = eval(high, row)?;
    if matches!(v, ast::Value::Null)
        || matches!(l, ast::Value::Null)
        || matches!(h, ast::Value::Null)
    {
        return Ok(ast::Value::Null);
    }
    let in_range =
        !matches!(compare(&v, &l), Ordering::Less) && !matches!(compare(&v, &h), Ordering::Greater);
    Ok(ast::Value::Bool(in_range ^ negated))
}

/// A *total* order over `f64` matching SQL semantics for `NaN`: every `NaN` is equal to every other
/// `NaN` and greater than every non-`NaN` value, and `-0.0` equals `0.0`. Unlike a bare `partial_cmp`
/// (which returns `None` for `NaN`, previously collapsed to `Equal` — making `NaN` compare equal to
/// *everything*, an intransitive non-order), this is consistent across `ORDER BY` / `DISTINCT` /
/// `GROUP BY` / `MIN` / `MAX`.
fn float_total_cmp(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        // Defined for all finite/infinite pairs; `-0.0`/`0.0` compare `Equal` (IEEE), as the standard
        // wants. The `unwrap_or` is unreachable here (neither side is `NaN`).
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// Compare two values for `ORDER BY` / equality semantics, treating `NULL`
/// as greater than every concrete value — this gives `NULLs LAST` in
/// ascending sorts (and `NULLs FIRST` in descending, because the sort caller
/// reverses the ordering for `DESC`).
pub(crate) fn compare(left: &ast::Value, right: &ast::Value) -> Ordering {
    use crate::numeric::Decimal;
    use ast::Value::{
        Array, Bool, Date, Float, Int, Interval, Json, Null, Numeric, Text, Time, TimeTz,
        Timestamp, TimestampTz, Uuid,
    };
    match (left, right) {
        (Null, Null) => Ordering::Equal,
        (Null, _) => Ordering::Greater,
        (_, Null) => Ordering::Less,
        (Bool(a), Bool(b)) => a.cmp(b),
        (Float(a), Float(b)) => float_total_cmp(*a, *b),
        (Int(a), Float(b)) => float_total_cmp(*a as f64, *b),
        (Float(a), Int(b)) => float_total_cmp(*a, *b as f64),
        // Text and JSON both compare by their (canonical) string.
        (Text(a), Text(b)) | (Json(a), Json(b)) => a.cmp(b),
        // NUMERIC compares exactly with itself and with integers; against a float it falls back
        // to f64.
        (Numeric(a), Numeric(b)) => a.compare(b),
        (Numeric(a), Int(b)) => a.compare(&Decimal::from_i64(*b)),
        (Int(a), Numeric(b)) => Decimal::from_i64(*a).compare(b),
        (Numeric(a), Float(b)) => float_total_cmp(a.to_f64(), *b),
        (Float(a), Numeric(b)) => float_total_cmp(*a, b.to_f64()),
        // Temporal + UUID order by their backing representation. Only same-type
        // comparisons are well-defined; the analyzer rejects cross-type comparisons earlier. INT and
        // the i64-backed temporals share this arm (identical `i64` ordering).
        (Date(a), Date(b)) => a.cmp(b),
        (Int(a), Int(b))
        | (Time(a), Time(b))
        | (TimeTz(a), TimeTz(b))
        | (Timestamp(a), Timestamp(b))
        | (TimestampTz(a), TimestampTz(b)) => a.cmp(b),
        (Uuid(a), Uuid(b)) => a.cmp(b),
        // BYTEA orders lexicographically by raw byte.
        (ast::Value::Bytes(a), ast::Value::Bytes(b)) => a.cmp(b),
        (Interval(a), Interval(b)) => a.compare(b),
        // Arrays order lexicographically by element, then by length. Elements are
        // homogeneous (the column's element type), so element-wise `compare` is well-defined; a
        // cross-type element pair falls to the type-rank arm below but cannot occur here.
        (Array(a), Array(b)) => a
            .iter()
            .zip(b)
            .map(|(x, y)| compare(x, y))
            .find(|&o| o != Ordering::Equal)
            .unwrap_or_else(|| a.len().cmp(&b.len())),
        // Vectors order lexicographically by component (total order via `f32::total_cmp`), then by
        // length — used by DISTINCT / GROUP BY / ORDER BY on a VECTOR column.
        (ast::Value::Vector(a), ast::Value::Vector(b)) => a
            .iter()
            .zip(b)
            .map(|(x, y)| x.total_cmp(y))
            .find(|&o| o != Ordering::Equal)
            .unwrap_or_else(|| a.len().cmp(&b.len())),
        // Different value types: order by a stable per-variant rank rather than collapsing to
        // `Equal`. Collapsing made `compare` non-antisymmetric, so two genuinely distinct
        // values (e.g. `1` and `'a'`) looked equal and were wrongly merged by DISTINCT / GROUP BY /
        // deduped in an ORDER BY. The analyzer rejects most cross-type comparisons earlier; this is
        // the last-resort total order for any pair that still reaches here.
        _ => type_rank(left).cmp(&type_rank(right)),
    }
}

/// Compare two `ORDER BY` key values honoring `ASC`/`DESC` and explicit `NULLS FIRST`/`LAST`.
///
/// With [`NullOrdering::Default`](ast::NullOrdering::Default) the result matches [`compare`] flipped
/// for `DESC` — i.e. the SQL default (NULLs last for `ASC`, first for `DESC`). An explicit
/// `FIRST`/`LAST` pins `NULL` placement regardless of the sort direction.
pub(crate) fn compare_order_key(
    a: &ast::Value,
    b: &ast::Value,
    ascending: bool,
    nulls: ast::NullOrdering,
) -> Ordering {
    let a_null = matches!(a, ast::Value::Null);
    let b_null = matches!(b, ast::Value::Null);
    if a_null != b_null && matches!(nulls, ast::NullOrdering::First | ast::NullOrdering::Last) {
        // Exactly one side is NULL and the clause pins where NULLs go — independent of ASC/DESC.
        let nulls_first = matches!(nulls, ast::NullOrdering::First);
        return if a_null == nulls_first {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let ord = compare(a, b);
    if ascending { ord } else { ord.reverse() }
}

/// Stable ordering rank per [`ast::Value`] variant, so [`compare`] is a total order across
/// different value types instead of collapsing them to `Equal`.
const fn type_rank(v: &ast::Value) -> u8 {
    use ast::Value::{
        Array, Bool, Date, Float, Int, Interval, Json, Null, Numeric, Text, Time, TimeTz,
        Timestamp, TimestampTz, Uuid,
    };
    match v {
        Null => 0,
        Bool(_) => 1,
        Int(_) => 2,
        Float(_) => 3,
        Numeric(_) => 4,
        Text(_) => 5,
        Json(_) => 6,
        Date(_) => 7,
        Time(_) => 8,
        Timestamp(_) => 9,
        TimestampTz(_) => 10,
        Uuid(_) => 11,
        Interval(_) => 12,
        Array(_) => 13,
        TimeTz(_) => 14,
        ast::Value::Vector(_) => 15,
        ast::Value::Bytes(_) => 16,
    }
}

fn apply_binary(
    op: ast::BinaryOp,
    left: &ast::Value,
    right: &ast::Value,
    result_ty: ColumnType,
) -> Result<ast::Value, Error> {
    use ast::BinaryOp as Op;
    match op {
        Op::And => Ok(apply_and(left, right)),
        Op::Or => Ok(apply_or(left, right)),
        Op::Eq | Op::NotEq | Op::Lt | Op::LtEq | Op::Gt | Op::GtEq => {
            Ok(apply_comparison(op, left, right))
        },
        Op::Plus | Op::Minus | Op::Multiply | Op::Divide | Op::Modulo => {
            apply_arithmetic(op, left, right, result_ty)
        },
        Op::BitAnd | Op::BitOr | Op::BitXor | Op::ShiftLeft | Op::ShiftRight => {
            Ok(apply_bitwise(op, left, right))
        },
        Op::ArrayOverlap => Ok(apply_array_overlap(left, right)),
        Op::Concat => apply_concat(left, right),
        // `@>` / `<@` containment — over arrays or JSON (the analyzer picks the operand domain).
        Op::JsonContains | Op::JsonContainedBy => Ok(containment_op(op, left, right)),
        Op::JsonGet | Op::JsonGetText | Op::JsonGetPath | Op::JsonGetPathText => {
            Ok(json_op(op, left, right))
        },
        Op::VectorDistance => vector_distance_op(left, right),
        Op::TsMatch => ts_match_op(left, right),
    }
}

/// Evaluate `@@` (F1): the left operand is the `tsvector` text form and the right the `tsquery`
/// text form; if that orientation fails to parse, the swapped orientation (`tsquery @@ tsvector`,
/// which the reference engine also accepts) is tried before reporting the original error. A `NULL` operand yields
/// `NULL`. The reference engine's `text @@ text` implicit-conversion overload (tokenize both sides) is not
/// implemented — a plain-text side fails the tsquery parse loudly rather than being reinterpreted.
/// Wrap a `real` (float4) result as a runtime [`ast::Value::Float`] whose text rendering matches
/// the reference engine's float4 output. `Value::Float` is `f64`, so the value is taken from the `f32`'s shortest
/// round-trip decimal — parsing that decimal to the nearest `f64` renders identically, whereas a
/// plain widening (`f64::from`) would print spurious trailing digits.
fn real_value(r: f32) -> ast::Value {
    ast::Value::Float(
        r.to_string()
            .parse::<f64>()
            .unwrap_or_else(|_| f64::from(r)),
    )
}

/// The Reciprocal Rank Fusion contribution `1/(k + rank)`. A rank below 1 (`RANK()`
/// always starts at 1) or a negative `k` is a loud domain error so a misplaced argument cannot
/// silently skew a fused score.
#[allow(
    clippy::cast_precision_loss,
    reason = "ranks and k are small counts, far inside f64's exact-integer range"
)]
fn rrf_score(rank: i64, k: i64) -> Result<ast::Value, Error> {
    if rank < 1 {
        return Err(Error::ArgumentOutOfDomain(format!(
            "rrf_score: rank must be >= 1 (RANK() starts at 1), got {rank}"
        )));
    }
    if k < 0 {
        return Err(Error::ArgumentOutOfDomain(format!(
            "rrf_score: k must be >= 0, got {k}"
        )));
    }
    Ok(ast::Value::Float(1.0 / (k as f64 + rank as f64)))
}

fn ts_match_op(left: &ast::Value, right: &ast::Value) -> Result<ast::Value, Error> {
    use ast::Value::Text;
    let (Text(l), Text(r)) = (left, right) else {
        return Ok(ast::Value::Null);
    };
    match crate::fts::ts_match(l, r) {
        Ok(matched) => Ok(ast::Value::Bool(matched)),
        Err(first) => crate::fts::ts_match(r, l)
            .map(ast::Value::Bool)
            .map_err(|_| first),
    }
}

/// Evaluate an integer bitwise operator `&` / `|` / `<<` / `>>` (B-fn). A `NULL` operand yields
/// `NULL`; otherwise both operands are `INT` (the analyzer enforces this). The shift amount is taken
/// modulo 64 (the `i64` bit width) so a large or negative count is total rather than undefined,
/// matching the wrapping shift of common SQL engines. `>>` is an arithmetic (sign-preserving) shift.
fn apply_bitwise(op: ast::BinaryOp, left: &ast::Value, right: &ast::Value) -> ast::Value {
    use ast::BinaryOp as Op;
    let (ast::Value::Int(a), ast::Value::Int(b)) = (left, right) else {
        return ast::Value::Null;
    };
    // `rem_euclid(64)` is in `0..64`, so the `u32` conversion is exact and the shift is defined.
    let shift = || u32::try_from(b.rem_euclid(64)).unwrap_or(0);
    let result = match op {
        Op::BitAnd => a & b,
        Op::BitOr => a | b,
        Op::BitXor => a ^ b,
        Op::ShiftLeft => a.wrapping_shl(shift()),
        Op::ShiftRight => a.wrapping_shr(shift()),
        // Unreachable: only the four bitwise/shift ops are routed here. A NULL is a safe fallback
        // (rather than silently applying a shift) should a future op be misrouted.
        _ => return ast::Value::Null,
    };
    ast::Value::Int(result)
}

/// Evaluate array overlap `a && b` (B-fn): `TRUE` if the two arrays share at least one element. A
/// `NULL` array operand yields `NULL`; element comparison treats two `NULL` elements as unequal (a
/// `NULL` element never makes an overlap), matching `ARRAY_POSITION`'s search semantics.
fn apply_array_overlap(left: &ast::Value, right: &ast::Value) -> ast::Value {
    let (ast::Value::Array(a), ast::Value::Array(b)) = (left, right) else {
        return ast::Value::Null;
    };
    let overlap = a
        .iter()
        .any(|x| !matches!(x, ast::Value::Null) && b.iter().any(|y| value_eq(x, y)));
    ast::Value::Bool(overlap)
}

/// Evaluate `a <=> b` — cosine distance between two vectors, as `FLOAT`. A `NULL` operand
/// yields `NULL`; a dimension mismatch is a typed error (the analyzer rejects most cases earlier, but
/// a runtime mismatch is still possible across rows of differently-sized literals).
fn vector_distance_op(left: &ast::Value, right: &ast::Value) -> Result<ast::Value, Error> {
    match (left, right) {
        (ast::Value::Null, _) | (_, ast::Value::Null) => Ok(ast::Value::Null),
        (ast::Value::Vector(a), ast::Value::Vector(b)) => crate::vector::cosine_distance(a, b)
            .map(ast::Value::Float)
            .ok_or_else(|| vector_dim_mismatch(a.len(), b.len())),
        _ => Err(Error::TypeMismatch {
            context: "`<=>` vector distance".to_owned(),
            expected: ColumnType::Vector(0),
            found: runtime_type(right),
        }),
    }
}

/// Evaluate a vector distance function (`l2_distance` / `cosine_distance` / `inner_product`) on two
/// equal-length vectors, as `FLOAT`. The analyzer guarantees two `VECTOR` arguments; a
/// dimension mismatch is a typed runtime error.
fn eval_vector_distance(func: ast::ScalarFunc, args: &[ast::Value]) -> Result<ast::Value, Error> {
    use ast::ScalarFunc as F;
    let [ast::Value::Vector(a), ast::Value::Vector(b)] = args else {
        return Ok(ast::Value::Null);
    };
    let metric = match func {
        F::L2Distance => crate::vector::l2_distance,
        F::CosineDistance => crate::vector::cosine_distance,
        _ => crate::vector::inner_product,
    };
    metric(a, b)
        .map(ast::Value::Float)
        .ok_or_else(|| vector_dim_mismatch(a.len(), b.len()))
}

/// The runtime error for applying a vector op to two different-dimension vectors.
fn vector_dim_mismatch(a: usize, b: usize) -> Error {
    Error::Unsupported(format!(
        "vector distance requires equal dimensions, got {a} and {b}"
    ))
}

/// Evaluate `||` string concatenation: NULL if either operand is NULL, otherwise the two
/// text operands joined. The analyzer guarantees both operands are text.
fn apply_concat(left: &ast::Value, right: &ast::Value) -> Result<ast::Value, Error> {
    use ast::Value::{Array, Bytes, Null, Text};
    Ok(match (left, right) {
        // NULL-strict: a NULL operand yields NULL (matches the analyzer's `||` typing).
        (Null, _) | (_, Null) => Null,
        (Text(a), Text(b)) => Text(format!("{a}{b}")),
        // One text side coerces the other scalar to its text output — the
        // analyzer's `check_concat` admits exactly the textout-able scalar set.
        (Text(a), other) if !matches!(other, Array(_) | Bytes(_)) => {
            Text(format!("{a}{}", text_output(other.clone())?))
        },
        (other, Text(b)) if !matches!(other, Array(_) | Bytes(_)) => {
            Text(format!("{}{b}", text_output(other.clone())?))
        },
        // BYTEA concatenation: append the two byte strings.
        (Bytes(a), Bytes(b)) => {
            let mut out = Vec::with_capacity(a.len() + b.len());
            out.extend_from_slice(a);
            out.extend_from_slice(b);
            Bytes(out)
        },
        // Array concatenation: merge two arrays, or append/prepend a single element.
        (Array(a), Array(b)) => {
            let mut out = a.clone();
            out.extend(b.iter().cloned());
            Array(out)
        },
        (Array(a), elem) => {
            let mut out = a.clone();
            out.push(elem.clone());
            Array(out)
        },
        (elem, Array(b)) => {
            let mut out = Vec::with_capacity(b.len() + 1);
            out.push(elem.clone());
            out.extend(b.iter().cloned());
            Array(out)
        },
        _ => Null,
    })
}

/// Evaluate `IS [NOT] DISTINCT FROM`: two NULLs are *not* distinct, exactly one NULL *is*
/// distinct, otherwise distinctness is inequality. Always boolean (never NULL); `negated` flips it.
fn apply_is_distinct_from(left: &ast::Value, right: &ast::Value, negated: bool) -> ast::Value {
    use ast::Value::Null;
    let distinct = match (left, right) {
        (Null, Null) => false,
        (Null, _) | (_, Null) => true,
        _ => compare(left, right) != Ordering::Equal,
    };
    ast::Value::Bool(distinct ^ negated)
}

/// Evaluate `IS [NOT] {TRUE|FALSE|UNKNOWN}` under three-valued logic. Always boolean (never
/// NULL): `UNKNOWN` matches a NULL operand; `negated` flips the result.
const fn apply_is_bool(value: &ast::Value, truth: ast::TruthValue, negated: bool) -> ast::Value {
    let base = match truth {
        ast::TruthValue::True => matches!(value, ast::Value::Bool(true)),
        ast::TruthValue::False => matches!(value, ast::Value::Bool(false)),
        ast::TruthValue::Unknown => matches!(value, ast::Value::Null),
    };
    ast::Value::Bool(base ^ negated)
}

/// Evaluate `@>` (contains) / `<@` (contained-by) over arrays or JSON. A NULL operand yields NULL.
/// For arrays, `a @> b` is true when every element of `b` is present in `a` (and `<@` is the mirror,
/// `b @> a`); element membership is by value equality. For JSON it delegates to [`json_op`]'s `@>`
/// with the container on the left (swapped for `<@`).
fn containment_op(op: ast::BinaryOp, left: &ast::Value, right: &ast::Value) -> ast::Value {
    if matches!(left, ast::Value::Null) || matches!(right, ast::Value::Null) {
        return ast::Value::Null;
    }
    // Orient the operands so `container` is the side expected to contain `contained`.
    let (container, contained) = if op == ast::BinaryOp::JsonContains {
        (left, right)
    } else {
        (right, left)
    };
    if let (ast::Value::Array(big), ast::Value::Array(small)) = (container, contained) {
        return ast::Value::Bool(small.iter().all(|elem| big.contains(elem)));
    }
    // JSON containment: reuse the `@>` path with the container as the left document.
    json_op(ast::BinaryOp::JsonContains, container, contained)
}

/// Evaluate a JSON navigation operator: `->` (get as JSON), `->>` (get as text), `#>`/`#>>`
/// (path). A NULL operand or a missing field/element yields NULL. (`@>`/`<@` → [`containment_op`].)
fn json_op(op: ast::BinaryOp, left: &ast::Value, right: &ast::Value) -> ast::Value {
    let ast::Value::Json(doc) = left else {
        return ast::Value::Null;
    };
    match op {
        ast::BinaryOp::JsonGet => match right {
            ast::Value::Text(k) => {
                crate::json::get_field(doc, k).map_or(ast::Value::Null, ast::Value::Json)
            },
            ast::Value::Int(i) => {
                crate::json::get_index(doc, *i).map_or(ast::Value::Null, ast::Value::Json)
            },
            _ => ast::Value::Null,
        },
        ast::BinaryOp::JsonGetText => match right {
            ast::Value::Text(k) => {
                crate::json::get_field_text(doc, k).map_or(ast::Value::Null, ast::Value::Text)
            },
            ast::Value::Int(i) => {
                crate::json::get_index_text(doc, *i).map_or(ast::Value::Null, ast::Value::Text)
            },
            _ => ast::Value::Null,
        },
        ast::BinaryOp::JsonContains => match right {
            // `contains` parses both sides, so a raw text right operand works directly.
            ast::Value::Json(b) | ast::Value::Text(b) => {
                crate::json::contains(doc, b).map_or(ast::Value::Null, ast::Value::Bool)
            },
            _ => ast::Value::Null,
        },
        // `#>` / `#>>` — navigate a `text[]` path. The path is a `text[]`, but a bare text
        // value (`'{a,b}'`) is coerced to one here (SQL-standard). A non-text path element → NULL.
        ast::BinaryOp::JsonGetPath | ast::BinaryOp::JsonGetPathText => {
            let parsed;
            let items: &[ast::Value] = match right {
                ast::Value::Array(items) => items,
                ast::Value::Text(s) => match crate::executor::row::parse_array_text(s) {
                    Some(v) => {
                        parsed = v;
                        &parsed
                    },
                    None => return ast::Value::Null,
                },
                _ => return ast::Value::Null,
            };
            let mut path = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    ast::Value::Text(s) => path.push(s.as_str()),
                    _ => return ast::Value::Null,
                }
            }
            if op == ast::BinaryOp::JsonGetPath {
                crate::json::get_path(doc, &path).map_or(ast::Value::Null, ast::Value::Json)
            } else {
                crate::json::get_path_text(doc, &path).map_or(ast::Value::Null, ast::Value::Text)
            }
        },
        _ => ast::Value::Null,
    }
}

const fn apply_and(left: &ast::Value, right: &ast::Value) -> ast::Value {
    use ast::Value::{Bool, Null};
    match (left, right) {
        (Bool(false), _) | (_, Bool(false)) => Bool(false),
        (Bool(true), Bool(true)) => Bool(true),
        _ => Null,
    }
}

const fn apply_or(left: &ast::Value, right: &ast::Value) -> ast::Value {
    use ast::Value::{Bool, Null};
    match (left, right) {
        (Bool(true), _) | (_, Bool(true)) => Bool(true),
        (Bool(false), Bool(false)) => Bool(false),
        _ => Null,
    }
}

fn apply_comparison(op: ast::BinaryOp, left: &ast::Value, right: &ast::Value) -> ast::Value {
    if matches!(left, ast::Value::Null) || matches!(right, ast::Value::Null) {
        return ast::Value::Null;
    }
    let ordering = compare(left, right);
    let result = match op {
        ast::BinaryOp::Eq => ordering == Ordering::Equal,
        ast::BinaryOp::NotEq => ordering != Ordering::Equal,
        ast::BinaryOp::Lt => ordering == Ordering::Less,
        ast::BinaryOp::LtEq => ordering != Ordering::Greater,
        ast::BinaryOp::Gt => ordering == Ordering::Greater,
        ast::BinaryOp::GtEq => ordering != Ordering::Less,
        _ => return ast::Value::Null,
    };
    ast::Value::Bool(result)
}

/// Microseconds in a day (for DATE → micros promotion in interval arithmetic).
const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;

/// INTERVAL / temporal `+`/`-`: `interval ± interval`, and `temporal ± interval` (DATE
/// promotes to TIMESTAMP via calendar-aware addition). Returns `None` for non-interval cases so the
/// caller falls back to numeric arithmetic; `Some(Err(_))` when the interval arithmetic overflows
/// — a wrong duration must not be produced as success.
fn interval_arith(
    op: ast::BinaryOp,
    left: &ast::Value,
    right: &ast::Value,
) -> Option<Result<ast::Value, Error>> {
    use ast::Value::{Date, Int, Interval, Time, Timestamp, TimestampTz};
    let overflow = || Error::Unsupported("INTERVAL arithmetic overflow".to_owned());
    // `interval * integer` (commutative) scales every component.
    if matches!(op, ast::BinaryOp::Multiply) {
        return match (left, right) {
            (Interval(iv), Int(n)) | (Int(n), Interval(iv)) => {
                Some(iv.checked_mul(*n).map(Interval).ok_or_else(overflow))
            },
            _ => None,
        };
    }
    let add = matches!(op, ast::BinaryOp::Plus);
    if !add && !matches!(op, ast::BinaryOp::Minus) {
        return None;
    }
    let date_overflow = || Error::Unsupported("DATE arithmetic out of range".to_owned());
    // `date ± integer` (whole days, result DATE) and `date - date` (day count, result INTEGER).
    // A DATE is `i32` days since the epoch; the day arithmetic is done in `i64` then range-checked.
    let date_plus_days = |d: i32, days: i64| -> Result<ast::Value, Error> {
        i64::from(d)
            .checked_add(days)
            .and_then(|v| i32::try_from(v).ok())
            .map(Date)
            .ok_or_else(date_overflow)
    };
    match (left, right) {
        (Date(d), Int(n)) => {
            let days = if add { Some(*n) } else { n.checked_neg() };
            return Some(
                days.ok_or_else(date_overflow)
                    .and_then(|days| date_plus_days(*d, days)),
            );
        },
        (Int(n), Date(d)) if add => return Some(date_plus_days(*d, *n)),
        (Date(a), Date(b)) if !add => return Some(Ok(Int(i64::from(*a) - i64::from(*b)))),
        // `timestamp - timestamp` (same kind) → INTERVAL: whole days split out of the micro span,
        // months left zero (the elapsed time is calendar-agnostic), matching the standard form.
        (Timestamp(a), Timestamp(b)) | (TimestampTz(a), TimestampTz(b)) if !add => {
            return Some(a.checked_sub(*b).ok_or_else(overflow).and_then(|micros| {
                let days = i32::try_from(micros / MICROS_PER_DAY).map_err(|_| overflow())?;
                Ok(Interval(crate::interval::Interval {
                    months: 0,
                    days,
                    micros: micros % MICROS_PER_DAY,
                }))
            }));
        },
        _ => {},
    }
    // The interval applied to a temporal, with sign per `op` (negation can itself overflow).
    let apply = |ts: i64, iv: &crate::interval::Interval| -> Result<i64, Error> {
        let iv = if add {
            *iv
        } else {
            iv.checked_neg().ok_or_else(overflow)?
        };
        Ok(crate::temporal::add_interval_to_micros(
            ts, iv.months, iv.days, iv.micros,
        ))
    };
    match (left, right) {
        (Interval(a), Interval(b)) => Some(
            if add {
                a.checked_add(b)
            } else {
                a.checked_sub(b)
            }
            .map(Interval)
            .ok_or_else(overflow),
        ),
        (Timestamp(t), Interval(iv)) => Some(apply(*t, iv).map(Timestamp)),
        (TimestampTz(t), Interval(iv)) => Some(apply(*t, iv).map(TimestampTz)),
        (Date(d), Interval(iv)) => Some(apply(i64::from(*d) * MICROS_PER_DAY, iv).map(Timestamp)),
        // `interval + temporal` (commutative; only for `+`).
        (Interval(iv), Timestamp(t)) if add => Some(apply(*t, iv).map(Timestamp)),
        (Interval(iv), TimestampTz(t)) if add => Some(apply(*t, iv).map(TimestampTz)),
        (Interval(iv), Date(d)) if add => {
            Some(apply(i64::from(*d) * MICROS_PER_DAY, iv).map(Timestamp))
        },
        // `time ± interval` wraps within the 24-hour clock, using only the interval's sub-day
        // microseconds — a whole-day (or month) component contributes nothing to a clock time,
        // matching the reference. `time - time` is the elapsed
        // interval, signed.
        // The interval is reduced modulo one day FIRST (audit catch: `iv.micros` is an
        // unbounded i64, so adding it raw could overflow into wrap-garbage — a silent wrong
        // clock time in release); both operands are then < MICROS_PER_DAY and the sum cannot
        // overflow. This is also the stated semantics exactly: only the sub-day part matters.
        (Time(t), Interval(iv)) => {
            let sub_day = iv.micros.rem_euclid(MICROS_PER_DAY);
            let delta = if add {
                sub_day
            } else {
                MICROS_PER_DAY - sub_day
            };
            Some(Ok(Time((t + delta).rem_euclid(MICROS_PER_DAY))))
        },
        (Interval(iv), Time(t)) if add => {
            let sub_day = iv.micros.rem_euclid(MICROS_PER_DAY);
            Some(Ok(Time((t + sub_day).rem_euclid(MICROS_PER_DAY))))
        },
        (Time(a), Time(b)) if !add => Some(Ok(Interval(crate::interval::Interval {
            months: 0,
            days: 0,
            micros: a - b,
        }))),
        _ => None,
    }
}

fn apply_arithmetic(
    op: ast::BinaryOp,
    left: &ast::Value,
    right: &ast::Value,
    result_ty: ColumnType,
) -> Result<ast::Value, Error> {
    if matches!(left, ast::Value::Null) || matches!(right, ast::Value::Null) {
        return Ok(ast::Value::Null);
    }
    // INTERVAL / temporal arithmetic dispatches on operand types.
    if let Some(result) = interval_arith(op, left, right) {
        return result;
    }
    if matches!(result_ty, ColumnType::Numeric { .. }) {
        numeric_op(op, left, right)
    } else if result_ty == ColumnType::Float {
        let value = float_op(op, to_f64(left), to_f64(right))?;
        Ok(ast::Value::Float(value))
    } else {
        // `int_op` guards i64 overflow; a result type narrower than `BIGINT` additionally enforces
        // its declared width, so `int4 + int4` overflows at 2^31 exactly as the reference engine does.
        let value = int_op(op, to_i64(left)?, to_i64(right)?)?;
        if int_value_bounds(result_ty).is_some_and(|(lo, hi)| value < lo || value > hi) {
            return Err(Error::IntegerOutOfRange);
        }
        Ok(ast::Value::Int(value))
    }
}

/// The closed range `[lo, hi]` a value of integer type `ty` must fit, or `None` for `BIGINT` (the full
/// `i64`, already guarded against overflow by `int_op`) and any non-integer type. `SMALLINT`/`INT`
/// expressions enforce the same 16-/32-bit bounds the reference engine applies in arithmetic and casts — the bound the
/// storage layer already enforces on write.
fn int_value_bounds(ty: ColumnType) -> Option<(i64, i64)> {
    match ty {
        ColumnType::SmallInt => Some((i64::from(i16::MIN), i64::from(i16::MAX))),
        ColumnType::Int => Some((i64::from(i32::MIN), i64::from(i32::MAX))),
        _ => None,
    }
}

/// Exact decimal arithmetic for NUMERIC operands. Integers coerce to scale-0 decimals;
/// overflow surfaces as an error rather than wrapping, and division/modulo by zero is rejected.
fn numeric_op(
    op: ast::BinaryOp,
    left: &ast::Value,
    right: &ast::Value,
) -> Result<ast::Value, Error> {
    use crate::numeric::Decimal;
    use ast::BinaryOp as Op;
    let to_dec = |v: &ast::Value| -> Option<Decimal> {
        match v {
            ast::Value::Numeric(d) => Some(*d),
            ast::Value::Int(i) => Some(Decimal::from_i64(*i)),
            _ => None,
        }
    };
    // The exact result would exceed NUMERIC's 38-significant-digit capacity (`i128` mantissa —
    // the cap most engines besides the reference engine's arbitrary-precision NUMERIC declare). Loud, with the
    // standard out-of-range code, so an app sees a documented limit rather than a generic error
    // (lifting the cap is an the design design decision).
    let overflow = || Error::Coded {
        message: "numeric value exceeds the 38-digit precision limit".to_owned(),
        sqlstate: "22003", // numeric_value_out_of_range
    };
    // Defensive: the analyzer only routes Numeric/Int operands here.
    let (Some(a), Some(b)) = (to_dec(left), to_dec(right)) else {
        return Ok(ast::Value::Null);
    };
    let result = match op {
        Op::Plus => a.checked_add(&b).ok_or_else(overflow)?,
        Op::Minus => a.checked_sub(&b).ok_or_else(overflow)?,
        Op::Multiply => a.checked_mul(&b).ok_or_else(overflow)?,
        Op::Divide => {
            if b.is_zero() {
                return Err(Error::DivisionByZero);
            }
            a.checked_div(&b).ok_or_else(overflow)?
        },
        Op::Modulo => {
            if b.is_zero() {
                return Err(Error::DivisionByZero);
            }
            a.checked_rem(&b).ok_or_else(overflow)?
        },
        _ => return Ok(ast::Value::Null),
    };
    Ok(ast::Value::Numeric(result))
}

const fn float_op(op: ast::BinaryOp, l: f64, r: f64) -> Result<f64, Error> {
    use ast::BinaryOp as Op;
    Ok(match op {
        Op::Plus => l + r,
        Op::Minus => l - r,
        Op::Multiply => l * r,
        Op::Divide => {
            if r == 0.0 {
                return Err(Error::DivisionByZero);
            }
            l / r
        },
        Op::Modulo => {
            if r == 0.0 {
                return Err(Error::DivisionByZero);
            }
            l % r
        },
        _ => 0.0,
    })
}

fn int_op(op: ast::BinaryOp, l: i64, r: i64) -> Result<i64, Error> {
    use ast::BinaryOp as Op;
    // Integer arithmetic errors on overflow rather than wrapping silently — a wrapped
    // counter/financial value is catastrophic-but-silent. `checked_div`/`checked_rem` also catch
    // the `i64::MIN / -1` overflow; division by zero keeps its dedicated error.
    match op {
        Op::Plus => l.checked_add(r).ok_or(Error::IntegerOutOfRange),
        Op::Minus => l.checked_sub(r).ok_or(Error::IntegerOutOfRange),
        Op::Multiply => l.checked_mul(r).ok_or(Error::IntegerOutOfRange),
        Op::Divide => {
            if r == 0 {
                return Err(Error::DivisionByZero);
            }
            l.checked_div(r).ok_or(Error::IntegerOutOfRange)
        },
        Op::Modulo => {
            if r == 0 {
                return Err(Error::DivisionByZero);
            }
            l.checked_rem(r).ok_or(Error::IntegerOutOfRange)
        },
        _ => Ok(0),
    }
}

fn apply_unary(op: ast::UnaryOp, value: &ast::Value) -> Result<ast::Value, Error> {
    if matches!(value, ast::Value::Null) {
        return Ok(ast::Value::Null);
    }
    Ok(match op {
        ast::UnaryOp::Not => match value {
            ast::Value::Bool(b) => ast::Value::Bool(!*b),
            _ => ast::Value::Null,
        },
        ast::UnaryOp::Negate => match value {
            // `-i64::MIN` overflows (no positive counterpart); error rather than wrap.
            ast::Value::Int(i) => ast::Value::Int(i.checked_neg().ok_or(Error::IntegerOutOfRange)?),
            ast::Value::Float(f) => ast::Value::Float(-*f),
            ast::Value::Numeric(d) => ast::Value::Numeric(d.neg()),
            _ => ast::Value::Null,
        },
    })
}

fn to_f64(v: &ast::Value) -> f64 {
    match v {
        ast::Value::Int(i) => *i as f64,
        ast::Value::Float(f) => *f,
        ast::Value::Numeric(d) => d.to_f64(),
        _ => 0.0,
    }
}

fn to_i64(v: &ast::Value) -> Result<i64, Error> {
    match v {
        ast::Value::Int(i) => Ok(*i),
        // A NUMERIC whose integer part exceeds i64 must error, not silently become 0.
        // (Latent today: integer arithmetic only ever sees Int operands, since a NUMERIC operand
        // makes the result NUMERIC — but mask-to-zero would be a silent-wrong answer if reached.)
        ast::Value::Numeric(d) => d.to_i64().ok_or_else(|| {
            Error::Unsupported("NUMERIC value out of i64 range in integer arithmetic".to_owned())
        }),
        // The analyzer only routes integer operands here; anything else is a bug, not a silent zero.
        other => Err(Error::Unsupported(format!(
            "integer arithmetic on a non-integer value: {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::eval;
    use crate::ast::{BinaryOp, UnaryOp, Value};
    use crate::error::Error;
    use crate::planner::{TypedExpr, TypedExprKind};
    use nusadb_core::ColumnType;

    const fn lit(value: Value, ty: ColumnType) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Literal(value),
            ty,
        }
    }
    const fn lit_int(i: i64) -> TypedExpr {
        lit(Value::Int(i), ColumnType::Int)
    }
    const fn lit_float(f: f64) -> TypedExpr {
        lit(Value::Float(f), ColumnType::Float)
    }
    const fn lit_bool(b: bool) -> TypedExpr {
        lit(Value::Bool(b), ColumnType::Bool)
    }
    const fn col(index: usize, ty: ColumnType) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Column(index),
            ty,
        }
    }
    fn bin(left: TypedExpr, op: BinaryOp, right: TypedExpr, ty: ColumnType) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            },
            ty,
        }
    }
    fn unary(op: UnaryOp, inner: TypedExpr, ty: ColumnType) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::Unary {
                op,
                expr: Box::new(inner),
            },
            ty,
        }
    }
    fn is_null(inner: TypedExpr, negated: bool) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::IsNull {
                expr: Box::new(inner),
                negated,
            },
            ty: ColumnType::Bool,
        }
    }

    #[test]
    fn literal_passes_through() {
        let row = vec![];
        assert_eq!(eval(&lit_int(42), &row).unwrap(), Value::Int(42));
    }

    #[test]
    fn column_reads_from_row() {
        let row = vec![Value::Int(7), Value::Text("hi".to_owned())];
        assert_eq!(
            eval(&col(1, ColumnType::Text), &row).unwrap(),
            Value::Text("hi".to_owned()),
        );
    }

    #[test]
    fn integer_arithmetic() {
        let expr = bin(lit_int(6), BinaryOp::Plus, lit_int(7), ColumnType::Int);
        assert_eq!(eval(&expr, &vec![]).unwrap(), Value::Int(13));
    }

    #[test]
    fn integer_widens_to_float() {
        let expr = bin(
            lit_int(3),
            BinaryOp::Plus,
            lit_float(0.5),
            ColumnType::Float,
        );
        assert_eq!(eval(&expr, &vec![]).unwrap(), Value::Float(3.5));
    }

    #[test]
    fn string_concat_joins_text() {
        let expr = bin(
            lit(Value::Text("foo".to_owned()), ColumnType::Text),
            BinaryOp::Concat,
            lit(Value::Text("bar".to_owned()), ColumnType::Text),
            ColumnType::Text,
        );
        assert_eq!(
            eval(&expr, &vec![]).unwrap(),
            Value::Text("foobar".to_owned())
        );
    }

    #[test]
    fn string_concat_propagates_null() {
        let expr = bin(
            lit(Value::Text("x".to_owned()), ColumnType::Text),
            BinaryOp::Concat,
            lit(Value::Null, ColumnType::Text),
            ColumnType::Text,
        );
        assert_eq!(eval(&expr, &vec![]).unwrap(), Value::Null);
    }

    fn is_distinct(left: Value, right: Value, negated: bool) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::IsDistinctFrom {
                left: Box::new(lit(left, ColumnType::Int)),
                right: Box::new(lit(right, ColumnType::Int)),
                negated,
            },
            ty: ColumnType::Bool,
        }
    }

    #[test]
    fn is_distinct_from_treats_null_as_value() {
        // distinct; equal; one NULL -> distinct; both NULL -> not distinct.
        assert_eq!(
            eval(&is_distinct(Value::Int(1), Value::Int(2), false), &vec![]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval(&is_distinct(Value::Int(1), Value::Int(1), false), &vec![]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            eval(&is_distinct(Value::Null, Value::Int(1), false), &vec![]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval(&is_distinct(Value::Null, Value::Null, false), &vec![]).unwrap(),
            Value::Bool(false)
        );
        // Negated form is NULL-safe equality: both NULL -> true.
        assert_eq!(
            eval(&is_distinct(Value::Null, Value::Null, true), &vec![]).unwrap(),
            Value::Bool(true)
        );
    }

    fn is_bool(value: Value, truth: crate::ast::TruthValue, negated: bool) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::IsBool {
                expr: Box::new(lit(value, ColumnType::Bool)),
                truth,
                negated,
            },
            ty: ColumnType::Bool,
        }
    }

    #[test]
    fn is_bool_three_valued() {
        use crate::ast::TruthValue::{False, True, Unknown};
        let run = |v: Value, t, n| eval(&is_bool(v, t, n), &vec![]).unwrap();
        assert_eq!(run(Value::Bool(true), True, false), Value::Bool(true));
        assert_eq!(run(Value::Bool(false), True, false), Value::Bool(false));
        assert_eq!(run(Value::Null, True, false), Value::Bool(false));
        // NULL is UNKNOWN.
        assert_eq!(run(Value::Null, Unknown, false), Value::Bool(true));
        assert_eq!(run(Value::Bool(false), False, false), Value::Bool(true));
        // Negated: NULL IS NOT TRUE -> true.
        assert_eq!(run(Value::Null, True, true), Value::Bool(true));
    }

    #[test]
    fn integer_division_by_zero_is_rejected() {
        let expr = bin(lit_int(1), BinaryOp::Divide, lit_int(0), ColumnType::Int);
        assert!(matches!(eval(&expr, &vec![]), Err(Error::DivisionByZero)));
    }

    #[test]
    fn float_division_by_zero_is_rejected() {
        let expr = bin(
            lit_int(1),
            BinaryOp::Divide,
            lit_float(0.0),
            ColumnType::Float,
        );
        assert!(matches!(eval(&expr, &vec![]), Err(Error::DivisionByZero)));
    }

    #[test]
    fn comparison_with_null_is_null() {
        let expr = bin(
            lit_int(1),
            BinaryOp::Eq,
            lit(Value::Null, ColumnType::Int),
            ColumnType::Bool,
        );
        assert_eq!(eval(&expr, &vec![]).unwrap(), Value::Null);
    }

    #[test]
    fn three_valued_and() {
        // FALSE AND NULL = FALSE
        let expr = bin(
            lit_bool(false),
            BinaryOp::And,
            lit(Value::Null, ColumnType::Bool),
            ColumnType::Bool,
        );
        assert_eq!(eval(&expr, &vec![]).unwrap(), Value::Bool(false));

        // TRUE AND NULL = NULL
        let expr = bin(
            lit_bool(true),
            BinaryOp::And,
            lit(Value::Null, ColumnType::Bool),
            ColumnType::Bool,
        );
        assert_eq!(eval(&expr, &vec![]).unwrap(), Value::Null);
    }

    #[test]
    fn three_valued_or() {
        // TRUE OR NULL = TRUE
        let expr = bin(
            lit_bool(true),
            BinaryOp::Or,
            lit(Value::Null, ColumnType::Bool),
            ColumnType::Bool,
        );
        assert_eq!(eval(&expr, &vec![]).unwrap(), Value::Bool(true));

        // FALSE OR NULL = NULL
        let expr = bin(
            lit_bool(false),
            BinaryOp::Or,
            lit(Value::Null, ColumnType::Bool),
            ColumnType::Bool,
        );
        assert_eq!(eval(&expr, &vec![]).unwrap(), Value::Null);
    }

    #[test]
    fn unary_not_and_negate() {
        assert_eq!(
            eval(
                &unary(UnaryOp::Not, lit_bool(true), ColumnType::Bool),
                &vec![]
            )
            .unwrap(),
            Value::Bool(false),
        );
        assert_eq!(
            eval(
                &unary(UnaryOp::Negate, lit_int(5), ColumnType::Int),
                &vec![]
            )
            .unwrap(),
            Value::Int(-5),
        );
    }

    #[test]
    fn is_null_and_is_not_null() {
        let row = vec![Value::Null, Value::Int(1)];
        assert_eq!(
            eval(&is_null(col(0, ColumnType::Int), false), &row).unwrap(),
            Value::Bool(true),
        );
        assert_eq!(
            eval(&is_null(col(0, ColumnType::Int), true), &row).unwrap(),
            Value::Bool(false),
        );
        assert_eq!(
            eval(&is_null(col(1, ColumnType::Int), false), &row).unwrap(),
            Value::Bool(false),
        );
    }

    #[test]
    fn comparison_with_mixed_numerics() {
        // 3 < 3.5
        let expr = bin(lit_int(3), BinaryOp::Lt, lit_float(3.5), ColumnType::Bool);
        assert_eq!(eval(&expr, &vec![]).unwrap(), Value::Bool(true));
    }

    #[test]
    fn text_comparison() {
        let expr = bin(
            lit(Value::Text("alice".to_owned()), ColumnType::Text),
            BinaryOp::Eq,
            lit(Value::Text("alice".to_owned()), ColumnType::Text),
            ColumnType::Bool,
        );
        assert_eq!(eval(&expr, &vec![]).unwrap(), Value::Bool(true));
    }

    // ----- scalar string functions -----

    use crate::ast::ScalarFunc;

    fn scalar(func: ScalarFunc, args: Vec<TypedExpr>, ty: ColumnType) -> TypedExpr {
        TypedExpr {
            kind: TypedExprKind::ScalarFunction { func, args },
            ty,
        }
    }
    fn txt(s: &str) -> TypedExpr {
        lit(Value::Text(s.to_owned()), ColumnType::Text)
    }
    /// Evaluate a text-returning scalar function and unwrap the resulting `String`.
    fn run_text(func: ScalarFunc, args: Vec<TypedExpr>) -> Value {
        eval(&scalar(func, args, ColumnType::Text), &vec![]).unwrap()
    }

    #[test]
    fn to_char_numeric_digit_pictures_match_pg() {
        // Integer + fraction picture: digits fit exactly, fraction rounded/padded, leading sign space.
        assert_eq!(
            run_text(ScalarFunc::ToChar, vec![num("1234.5"), txt("9999.99")]),
            Value::Text(" 1234.50".to_owned())
        );
        // Leading `9` positions with no digit are suppressed to spaces (plus the sign column).
        assert_eq!(
            run_text(ScalarFunc::ToChar, vec![num("12"), txt("9999.99")]),
            Value::Text("   12.00".to_owned())
        );
        // A negative borrows the sign column, floating it next to the first digit.
        assert_eq!(
            run_text(ScalarFunc::ToChar, vec![num("-1234.5"), txt("9999.99")]),
            Value::Text("-1234.50".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::ToChar, vec![lit_int(485), txt("999")]),
            Value::Text(" 485".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::ToChar, vec![lit_int(485), txt("9999")]),
            Value::Text("  485".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::ToChar, vec![lit_int(-485), txt("9999")]),
            Value::Text(" -485".to_owned())
        );
        // A value `< 1` suppresses the whole integer part; the sign floats to the decimal point
        // (the canonical `to_char(-0.1, '99.99')` = `'  -.10'`).
        assert_eq!(
            run_text(ScalarFunc::ToChar, vec![num("0.5"), txt("9.99")]),
            Value::Text("  .50".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::ToChar, vec![num("-0.1"), txt("99.99")]),
            Value::Text("  -.10".to_owned())
        );
        // `0` picture positions are forced to a zero digit.
        assert_eq!(
            run_text(ScalarFunc::ToChar, vec![lit_int(12), txt("0000")]),
            Value::Text(" 0012".to_owned())
        );
        // A value too wide for the integer positions renders as `#` fill.
        assert_eq!(
            run_text(ScalarFunc::ToChar, vec![lit_int(1234), txt("99")]),
            Value::Text(" ##".to_owned())
        );
        // An unsupported format character is rejected, never silently mis-formatted.
        for bad in ["FM999", "9,999", "S999", "999D99", "999PR"] {
            assert!(
                eval(
                    &scalar(
                        ScalarFunc::ToChar,
                        vec![lit_int(5), txt(bad)],
                        ColumnType::Text
                    ),
                    &vec![]
                )
                .is_err(),
                "format `{bad}` should be rejected"
            );
        }
    }

    #[test]
    fn length_counts_characters_not_bytes() {
        // 'héllo' is 5 chars but 6 UTF-8 bytes — LENGTH is character-based.
        let v = eval(
            &scalar(ScalarFunc::Length, vec![txt("héllo")], ColumnType::Int),
            &vec![],
        )
        .unwrap();
        assert_eq!(v, Value::Int(5));
    }

    #[test]
    fn upper_lower_fold_case() {
        assert_eq!(
            run_text(ScalarFunc::Upper, vec![txt("MiXeD")]),
            Value::Text("MIXED".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::Lower, vec![txt("MiXeD")]),
            Value::Text("mixed".to_owned())
        );
    }

    #[test]
    fn substring_two_and_three_arg() {
        // 1-based: SUBSTRING('abcdef', 2) -> 'bcdef'; (.., 2, 3) -> 'bcd'.
        assert_eq!(
            run_text(ScalarFunc::Substring, vec![txt("abcdef"), lit_int(2)]),
            Value::Text("bcdef".to_owned())
        );
        assert_eq!(
            run_text(
                ScalarFunc::Substring,
                vec![txt("abcdef"), lit_int(2), lit_int(3)]
            ),
            Value::Text("bcd".to_owned())
        );
    }

    #[test]
    fn substring_clips_start_before_one_but_window_still_counts() {
        // SUBSTRING('abcdef' FROM -2 FOR 5): window [-2, 3) clips to positions 1..2 -> 'ab'.
        assert_eq!(
            run_text(
                ScalarFunc::Substring,
                vec![txt("abcdef"), lit_int(-2), lit_int(5)]
            ),
            Value::Text("ab".to_owned())
        );
        // Zero-length / past-end windows yield the empty string.
        assert_eq!(
            run_text(
                ScalarFunc::Substring,
                vec![txt("abc"), lit_int(2), lit_int(0)]
            ),
            Value::Text(String::new())
        );
        assert_eq!(
            run_text(ScalarFunc::Substring, vec![txt("abc"), lit_int(9)]),
            Value::Text(String::new())
        );
    }

    #[test]
    fn substring_negative_length_is_rejected() {
        let r = eval(
            &scalar(
                ScalarFunc::Substring,
                vec![txt("abc"), lit_int(1), lit_int(-1)],
                ColumnType::Text,
            ),
            &vec![],
        );
        assert!(matches!(r, Err(Error::Unsupported(_))));
    }

    #[test]
    fn replace_all_occurrences_and_empty_from() {
        assert_eq!(
            run_text(ScalarFunc::Replace, vec![txt("a.b.c"), txt("."), txt("-")]),
            Value::Text("a-b-c".to_owned())
        );
        // Empty search string leaves the input unchanged (not char-by-char insertion).
        assert_eq!(
            run_text(ScalarFunc::Replace, vec![txt("abc"), txt(""), txt("x")]),
            Value::Text("abc".to_owned())
        );
    }

    #[test]
    fn position_is_one_based_with_zero_for_absent() {
        let pos = |sub, hay| {
            eval(
                &scalar(
                    ScalarFunc::Position,
                    vec![txt(sub), txt(hay)],
                    ColumnType::Int,
                ),
                &vec![],
            )
            .unwrap()
        };
        assert_eq!(pos("cd", "abcde"), Value::Int(3));
        assert_eq!(pos("zz", "abcde"), Value::Int(0));
        assert_eq!(pos("", "abc"), Value::Int(1)); // empty needle matches at 1
    }

    #[test]
    fn lpad_rpad_pad_truncate_and_cycle_fill() {
        // Pad to width with default space; cycle a multi-char fill; truncate when too long.
        assert_eq!(
            run_text(ScalarFunc::Lpad, vec![txt("ab"), lit_int(5)]),
            Value::Text("   ab".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::Rpad, vec![txt("ab"), lit_int(5)]),
            Value::Text("ab   ".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::Lpad, vec![txt("ab"), lit_int(5), txt("xy")]),
            Value::Text("xyxab".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::Rpad, vec![txt("abcdef"), lit_int(3)]),
            Value::Text("abc".to_owned())
        );
        // Empty fill cannot extend; returns the (untruncated, since shorter) input.
        assert_eq!(
            run_text(ScalarFunc::Lpad, vec![txt("ab"), lit_int(5), txt("")]),
            Value::Text("ab".to_owned())
        );
    }

    #[test]
    fn trim_default_whitespace_and_custom_set() {
        assert_eq!(
            run_text(ScalarFunc::BTrim, vec![txt("  hi  ")]),
            Value::Text("hi".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::LTrim, vec![txt("  hi  ")]),
            Value::Text("hi  ".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::RTrim, vec![txt("  hi  ")]),
            Value::Text("  hi".to_owned())
        );
        // Custom trim set: strip any of the listed characters from the relevant side.
        assert_eq!(
            run_text(ScalarFunc::BTrim, vec![txt("xxhixx"), txt("x")]),
            Value::Text("hi".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::LTrim, vec![txt("xyxhi"), txt("xy")]),
            Value::Text("hi".to_owned())
        );
    }

    #[test]
    fn scalar_function_propagates_null() {
        // Any NULL argument yields NULL (all functions are NULL-strict).
        assert_eq!(
            run_text(ScalarFunc::Upper, vec![lit(Value::Null, ColumnType::Text)]),
            Value::Null
        );
        assert_eq!(
            run_text(
                ScalarFunc::Substring,
                vec![txt("abc"), lit(Value::Null, ColumnType::Int)]
            ),
            Value::Null
        );
    }

    // ----- string functions -----

    fn null_text() -> TypedExpr {
        lit(Value::Null, ColumnType::Text)
    }

    #[test]
    fn concat_joins_and_skips_null() {
        assert_eq!(
            run_text(ScalarFunc::Concat, vec![txt("a"), txt("b"), txt("c")]),
            Value::Text("abc".to_owned())
        );
        // NULL arguments contribute nothing (not NULL-strict).
        assert_eq!(
            run_text(ScalarFunc::Concat, vec![txt("a"), null_text(), txt("c")]),
            Value::Text("ac".to_owned())
        );
        // All-NULL yields the empty string, never NULL.
        assert_eq!(
            run_text(ScalarFunc::Concat, vec![null_text(), null_text()]),
            Value::Text(String::new())
        );
    }

    #[test]
    fn concat_ws_joins_with_separator_and_skips_null() {
        assert_eq!(
            run_text(
                ScalarFunc::ConcatWs,
                vec![txt("-"), txt("a"), txt("b"), txt("c")]
            ),
            Value::Text("a-b-c".to_owned())
        );
        // NULL data args are skipped (no doubled separator).
        assert_eq!(
            run_text(
                ScalarFunc::ConcatWs,
                vec![txt("-"), txt("a"), null_text(), txt("c")]
            ),
            Value::Text("a-c".to_owned())
        );
        // A NULL separator makes the whole call NULL.
        assert_eq!(
            run_text(ScalarFunc::ConcatWs, vec![null_text(), txt("a"), txt("b")]),
            Value::Null
        );
    }

    #[test]
    fn left_right_handle_negative_and_overflow() {
        assert_eq!(
            run_text(ScalarFunc::Left, vec![txt("abcdef"), lit_int(3)]),
            Value::Text("abc".to_owned())
        );
        // Negative n drops the last |n| characters.
        assert_eq!(
            run_text(ScalarFunc::Left, vec![txt("abcdef"), lit_int(-2)]),
            Value::Text("abcd".to_owned())
        );
        // n beyond the length returns the whole string.
        assert_eq!(
            run_text(ScalarFunc::Left, vec![txt("ab"), lit_int(9)]),
            Value::Text("ab".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::Left, vec![txt("abc"), lit_int(0)]),
            Value::Text(String::new())
        );

        assert_eq!(
            run_text(ScalarFunc::Right, vec![txt("abcdef"), lit_int(2)]),
            Value::Text("ef".to_owned())
        );
        // Negative n drops the first |n| characters.
        assert_eq!(
            run_text(ScalarFunc::Right, vec![txt("abcdef"), lit_int(-2)]),
            Value::Text("cdef".to_owned())
        );
        assert_eq!(
            run_text(ScalarFunc::Right, vec![txt("ab"), lit_int(9)]),
            Value::Text("ab".to_owned())
        );
    }

    #[test]
    fn split_part_is_one_based_with_empty_out_of_range() {
        let sp = |s, d, n| run_text(ScalarFunc::SplitPart, vec![txt(s), txt(d), lit_int(n)]);
        assert_eq!(sp("a,b,c", ",", 2), Value::Text("b".to_owned()));
        assert_eq!(sp("a,b,c", ",", 1), Value::Text("a".to_owned()));
        assert_eq!(sp("a,b,c", ",", 9), Value::Text(String::new())); // out of range
        assert_eq!(sp("a,b,c", ",", 0), Value::Text(String::new())); // n < 1
        assert_eq!(sp("a##b", "##", 2), Value::Text("b".to_owned())); // multi-char delim
    }

    #[test]
    fn reverse_is_character_based() {
        assert_eq!(
            run_text(ScalarFunc::Reverse, vec![txt("abc")]),
            Value::Text("cba".to_owned())
        );
        // Reverses by characters, not bytes (é stays intact).
        assert_eq!(
            run_text(ScalarFunc::Reverse, vec![txt("héllo")]),
            Value::Text("olléh".to_owned())
        );
    }

    // ----- regex functions -----

    /// Evaluate a scalar function, returning the `Result` (for error cases).
    fn try_scalar(func: ScalarFunc, args: Vec<TypedExpr>, ty: ColumnType) -> Result<Value, Error> {
        eval(&scalar(func, args, ty), &vec![])
    }

    #[test]
    fn regexp_replace_first_match_global_and_backrefs() {
        // First match only by default.
        assert_eq!(
            run_text(
                ScalarFunc::RegexpReplace,
                vec![txt("foo bar foo"), txt("foo"), txt("X")]
            ),
            Value::Text("X bar foo".to_owned())
        );
        // Global flag replaces all.
        assert_eq!(
            run_text(
                ScalarFunc::RegexpReplace,
                vec![txt("foo bar foo"), txt("foo"), txt("X"), txt("g")]
            ),
            Value::Text("X bar X".to_owned())
        );
        // Standard \N backreferences reorder capture groups.
        assert_eq!(
            run_text(
                ScalarFunc::RegexpReplace,
                vec![
                    txt("2024-01-15"),
                    txt(r"(\d+)-(\d+)-(\d+)"),
                    txt(r"\3/\2/\1")
                ]
            ),
            Value::Text("15/01/2024".to_owned())
        );
        // Case-insensitive flag.
        assert_eq!(
            run_text(
                ScalarFunc::RegexpReplace,
                vec![txt("Hello"), txt("hello"), txt("hi"), txt("i")]
            ),
            Value::Text("hi".to_owned())
        );
    }

    #[test]
    fn regexp_replace_rejects_bad_pattern_and_flag() {
        // Malformed pattern → InvalidRegex.
        assert!(matches!(
            try_scalar(
                ScalarFunc::RegexpReplace,
                vec![txt("x"), txt("("), txt("y")],
                ColumnType::Text
            ),
            Err(Error::InvalidRegex(_))
        ));
        // Unsupported flag character → InvalidRegex.
        assert!(matches!(
            try_scalar(
                ScalarFunc::RegexpReplace,
                vec![txt("x"), txt("x"), txt("y"), txt("z")],
                ColumnType::Text
            ),
            Err(Error::InvalidRegex(_))
        ));
    }

    #[test]
    fn regexp_match_returns_groups_or_whole_match_or_null() {
        // No groups → array with the whole match.
        assert_eq!(
            try_scalar(
                ScalarFunc::RegexpMatch,
                vec![txt("abc123"), txt("[a-z]+")],
                ColumnType::Array(nusadb_core::engine::ArrayElem::Text)
            )
            .unwrap(),
            Value::Array(vec![Value::Text("abc".to_owned())])
        );
        // Capture groups → array of the captured substrings.
        assert_eq!(
            try_scalar(
                ScalarFunc::RegexpMatch,
                vec![txt("2024-01"), txt(r"(\d+)-(\d+)")],
                ColumnType::Array(nusadb_core::engine::ArrayElem::Text)
            )
            .unwrap(),
            Value::Array(vec![
                Value::Text("2024".to_owned()),
                Value::Text("01".to_owned())
            ])
        );
        // No match → NULL.
        assert_eq!(
            try_scalar(
                ScalarFunc::RegexpMatch,
                vec![txt("abc"), txt(r"\d+")],
                ColumnType::Array(nusadb_core::engine::ArrayElem::Text)
            )
            .unwrap(),
            Value::Null
        );
    }

    // ----- niladic clock functions -----

    #[test]
    fn now_returns_a_timestamptz() {
        super::super::clock::set_statement_now();
        let v = eval(
            &scalar(ScalarFunc::Now, vec![], ColumnType::TimestampTz),
            &vec![],
        )
        .unwrap();
        let Value::TimestampTz(micros) = v else {
            panic!("expected TimestampTz, got {v:?}");
        };
        // Well past the epoch and not absurdly far in the future (sanity, not a flaky exact match).
        assert!(micros > 1_577_836_800_000_000); // 2020-01-01
    }

    #[test]
    fn clock_functions_are_statement_stable() {
        // After pinning the statement clock, repeated reads — and the date/time split — must all
        // agree on one instant (SQL statement stability).
        super::super::clock::set_statement_now();
        let now1 = eval(
            &scalar(ScalarFunc::Now, vec![], ColumnType::TimestampTz),
            &vec![],
        )
        .unwrap();
        let now2 = eval(
            &scalar(
                ScalarFunc::CurrentTimestamp,
                vec![],
                ColumnType::TimestampTz,
            ),
            &vec![],
        )
        .unwrap();
        assert_eq!(now1, now2);

        let Value::TimestampTz(micros) = now1 else {
            panic!("expected TimestampTz");
        };
        let date = eval(
            &scalar(ScalarFunc::CurrentDate, vec![], ColumnType::Date),
            &vec![],
        )
        .unwrap();
        let time = eval(
            &scalar(ScalarFunc::CurrentTime, vec![], ColumnType::Time),
            &vec![],
        )
        .unwrap();
        // The date and time are the floor-div / mod-euclid split of the same instant.
        let micros_per_day = super::super::clock::MICROS_PER_DAY;
        assert_eq!(
            date,
            Value::Date(i32::try_from(micros.div_euclid(micros_per_day)).unwrap())
        );
        assert_eq!(time, Value::Time(micros.rem_euclid(micros_per_day)));
    }

    // ----- math functions -----

    fn numv(s: &str) -> Value {
        Value::Numeric(crate::numeric::Decimal::parse(s).unwrap())
    }
    fn num(s: &str) -> TypedExpr {
        lit(
            numv(s),
            ColumnType::Numeric {
                precision: 0,
                scale: 0,
            },
        )
    }
    /// Evaluate a math function over already-typed literal args (result ty is irrelevant to eval).
    fn math(func: ScalarFunc, args: Vec<TypedExpr>) -> Value {
        eval(&scalar(func, args, ColumnType::Float), &vec![]).unwrap()
    }
    fn approx(v: Value, want: f64) {
        match v {
            Value::Float(got) => assert!((got - want).abs() < 1e-9, "got {got}, want {want}"),
            other => panic!("expected Float, got {other:?}"),
        }
    }

    #[test]
    fn abs_sign_preserve_numeric_type() {
        assert_eq!(math(ScalarFunc::Abs, vec![lit_int(-5)]), Value::Int(5));
        assert_eq!(
            math(ScalarFunc::Abs, vec![lit_float(-2.5)]),
            Value::Float(2.5)
        );
        assert_eq!(math(ScalarFunc::Abs, vec![num("-1.50")]), numv("1.50"));
        // SIGN: type-preserving; SIGN(0.0) is 0.0, not f64::signum's 1.0.
        assert_eq!(math(ScalarFunc::Sign, vec![lit_int(-3)]), Value::Int(-1));
        assert_eq!(
            math(ScalarFunc::Sign, vec![lit_float(0.0)]),
            Value::Float(0.0)
        );
        assert_eq!(
            math(ScalarFunc::Sign, vec![lit_float(2.5)]),
            Value::Float(1.0)
        );
    }

    #[test]
    fn ceil_floor_round_int_float_numeric() {
        // INT is returned unchanged; FLOAT uses libm; NUMERIC is exact.
        assert_eq!(math(ScalarFunc::Ceil, vec![lit_int(4)]), Value::Int(4));
        assert_eq!(
            math(ScalarFunc::Ceil, vec![lit_float(2.1)]),
            Value::Float(3.0)
        );
        assert_eq!(
            math(ScalarFunc::Floor, vec![lit_float(2.9)]),
            Value::Float(2.0)
        );
        // NUMERIC ceil toward +inf, floor toward -inf.
        assert_eq!(math(ScalarFunc::Ceil, vec![num("2.10")]), numv("3"));
        assert_eq!(math(ScalarFunc::Floor, vec![num("-2.10")]), numv("-3"));
        // ROUND: default 0 places, and to d places.
        assert_eq!(
            math(ScalarFunc::Round, vec![lit_float(2.567)]),
            Value::Float(3.0)
        );
        assert_eq!(
            math(ScalarFunc::Round, vec![lit_float(2.567), lit_int(2)]),
            Value::Float(2.57)
        );
        assert_eq!(
            math(ScalarFunc::Round, vec![num("2.567"), lit_int(2)]),
            numv("2.57")
        );
    }

    #[test]
    fn mod_preserves_type_and_rejects_zero() {
        assert_eq!(
            math(ScalarFunc::Mod, vec![lit_int(7), lit_int(3)]),
            Value::Int(1)
        );
        assert_eq!(
            math(ScalarFunc::Mod, vec![lit_float(7.5), lit_float(2.0)]),
            Value::Float(1.5)
        );
        assert_eq!(
            math(ScalarFunc::Mod, vec![num("7.5"), num("2")]),
            numv("1.5")
        );
        assert!(matches!(
            eval(
                &scalar(
                    ScalarFunc::Mod,
                    vec![lit_int(7), lit_int(0)],
                    ColumnType::Int
                ),
                &vec![]
            ),
            Err(Error::DivisionByZero)
        ));
    }

    #[test]
    fn power_root_log_trig_compute_in_float() {
        assert_eq!(
            math(ScalarFunc::Power, vec![lit_int(2), lit_int(10)]),
            Value::Float(1024.0)
        );
        assert_eq!(math(ScalarFunc::Sqrt, vec![lit_int(9)]), Value::Float(3.0));
        approx(math(ScalarFunc::Ln, vec![lit_int(1)]), 0.0);
        approx(math(ScalarFunc::Exp, vec![lit_int(0)]), 1.0);
        approx(math(ScalarFunc::Log, vec![lit_int(1000)]), 3.0); // log10
        approx(math(ScalarFunc::Log, vec![lit_int(2), lit_int(8)]), 3.0); // base-2 of 8
        approx(math(ScalarFunc::Sin, vec![lit_float(0.0)]), 0.0);
        approx(math(ScalarFunc::Cos, vec![lit_float(0.0)]), 1.0);
        approx(
            math(ScalarFunc::Atan2, vec![lit_float(1.0), lit_float(1.0)]),
            std::f64::consts::FRAC_PI_4,
        );
    }

    #[test]
    fn math_propagates_null() {
        assert_eq!(
            math(ScalarFunc::Abs, vec![lit(Value::Null, ColumnType::Int)]),
            Value::Null
        );
        assert_eq!(
            math(
                ScalarFunc::Power,
                vec![lit_int(2), lit(Value::Null, ColumnType::Int)]
            ),
            Value::Null
        );
    }

    // ----- conditional functions -----

    #[test]
    fn nullif_returns_null_on_equal_else_first() {
        let nullif = |a, b| {
            eval(
                &scalar(ScalarFunc::Nullif, vec![a, b], ColumnType::Int),
                &vec![],
            )
            .unwrap()
        };
        assert_eq!(nullif(lit_int(5), lit_int(5)), Value::Null);
        assert_eq!(nullif(lit_int(5), lit_int(3)), Value::Int(5));
        // NULL operand: a = b is unknown (not true) → returns a.
        assert_eq!(
            nullif(lit(Value::Null, ColumnType::Int), lit_int(5)),
            Value::Null
        );
        assert_eq!(
            nullif(lit_int(5), lit(Value::Null, ColumnType::Int)),
            Value::Int(5)
        );
    }

    #[test]
    fn greatest_least_skip_null() {
        let g = |args| {
            eval(
                &scalar(ScalarFunc::Greatest, args, ColumnType::Int),
                &vec![],
            )
            .unwrap()
        };
        let l = |args| eval(&scalar(ScalarFunc::Least, args, ColumnType::Int), &vec![]).unwrap();
        assert_eq!(g(vec![lit_int(1), lit_int(5), lit_int(3)]), Value::Int(5));
        assert_eq!(l(vec![lit_int(1), lit_int(5), lit_int(3)]), Value::Int(1));
        // NULL arguments are skipped (not propagated).
        assert_eq!(
            g(vec![
                lit_int(1),
                lit(Value::Null, ColumnType::Int),
                lit_int(3)
            ]),
            Value::Int(3)
        );
        assert_eq!(
            l(vec![lit(Value::Null, ColumnType::Int), lit_int(2)]),
            Value::Int(2)
        );
        // All-NULL → NULL.
        assert_eq!(
            g(vec![
                lit(Value::Null, ColumnType::Int),
                lit(Value::Null, ColumnType::Int)
            ]),
            Value::Null
        );
    }

    // ----- RANDOM / SETSEED -----

    #[test]
    fn random_in_range_and_setseed_returns_bool() {
        // SETSEED returns BOOL true once applied.
        assert_eq!(
            eval(
                &scalar(ScalarFunc::Setseed, vec![lit_float(0.5)], ColumnType::Bool),
                &vec![]
            )
            .unwrap(),
            Value::Bool(true)
        );
        // RANDOM() (no args) is a FLOAT in [0, 1).
        match eval(
            &scalar(ScalarFunc::Random, vec![], ColumnType::Float),
            &vec![],
        )
        .unwrap()
        {
            Value::Float(v) => assert!((0.0..1.0).contains(&v), "RANDOM() out of range: {v}"),
            other => panic!("expected Float, got {other:?}"),
        }
    }

    // ----- regex match operator -----

    fn regex(subject: &str, pattern: &str, case_sensitive: bool, negated: bool) -> Value {
        let node = TypedExpr {
            kind: TypedExprKind::RegexMatch {
                expr: Box::new(lit(Value::Text(subject.to_owned()), ColumnType::Text)),
                pattern: Box::new(lit(Value::Text(pattern.to_owned()), ColumnType::Text)),
                case_sensitive,
                negated,
            },
            ty: ColumnType::Bool,
        };
        eval(&node, &vec![]).unwrap()
    }

    #[test]
    fn regex_match_operator_semantics() {
        // `~` case-sensitive match / non-match.
        assert_eq!(regex("hello", "ell", true, false), Value::Bool(true));
        assert_eq!(regex("hello", "^x", true, false), Value::Bool(false));
        // `!~` negates.
        assert_eq!(regex("hello", "ell", true, true), Value::Bool(false));
        // `~*` is case-insensitive; `~` is not.
        assert_eq!(regex("HELLO", "hello", false, false), Value::Bool(true));
        assert_eq!(regex("HELLO", "hello", true, false), Value::Bool(false));
        // Anchors + character classes work (real regex, not LIKE).
        assert_eq!(regex("abc123", "[0-9]+$", true, false), Value::Bool(true));
    }

    #[test]
    fn regex_match_null_and_invalid_pattern() {
        // NULL operand -> NULL.
        let null_node = TypedExpr {
            kind: TypedExprKind::RegexMatch {
                expr: Box::new(lit(Value::Null, ColumnType::Text)),
                pattern: Box::new(lit(Value::Text("a".to_owned()), ColumnType::Text)),
                case_sensitive: true,
                negated: false,
            },
            ty: ColumnType::Bool,
        };
        assert_eq!(eval(&null_node, &vec![]).unwrap(), Value::Null);
        // Invalid pattern -> InvalidRegex error.
        let bad = TypedExpr {
            kind: TypedExprKind::RegexMatch {
                expr: Box::new(lit(Value::Text("x".to_owned()), ColumnType::Text)),
                pattern: Box::new(lit(Value::Text("(".to_owned()), ColumnType::Text)),
                case_sensitive: true,
                negated: false,
            },
            ty: ColumnType::Bool,
        };
        assert!(matches!(eval(&bad, &vec![]), Err(Error::InvalidRegex(_))));
    }

    // ----- JSON path operators #> / #>> -----

    fn json_path(doc: &str, path: &[&str], text_form: bool) -> Value {
        let elems = path.iter().map(|p| Value::Text((*p).to_owned())).collect();
        let op = if text_form {
            BinaryOp::JsonGetPathText
        } else {
            BinaryOp::JsonGetPath
        };
        let node = bin(
            lit(Value::Json(doc.to_owned()), ColumnType::Json),
            op,
            lit(
                Value::Array(elems),
                ColumnType::Array(nusadb_core::engine::ArrayElem::Text),
            ),
            if text_form {
                ColumnType::Text
            } else {
                ColumnType::Json
            },
        );
        eval(&node, &vec![]).unwrap()
    }

    #[test]
    fn json_path_operators() {
        let doc = r#"{"a":{"b":42},"arr":[10,20,30]}"#;
        // #> returns JSON; #>> returns text.
        assert_eq!(
            json_path(doc, &["a", "b"], false),
            Value::Json("42".to_owned())
        );
        assert_eq!(
            json_path(doc, &["a", "b"], true),
            Value::Text("42".to_owned())
        );
        // Array index via text path element.
        assert_eq!(
            json_path(doc, &["arr", "1"], false),
            Value::Json("20".to_owned())
        );
        // Missing path -> NULL.
        assert_eq!(json_path(doc, &["a", "z"], false), Value::Null);
        // String leaf via #>> is unquoted.
        assert_eq!(
            json_path(r#"{"k":"hi"}"#, &["k"], true),
            Value::Text("hi".to_owned())
        );
    }

    // ----- ENCODE / DECODE base64 (B-fn) -----

    #[test]
    fn base64_encode_matches_rfc4648_vectors() {
        use super::base64_encode;
        // The classic RFC 4648 §10 test vectors exercise every tail length (0/1/2 bytes).
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_decode_inverts_encode_and_ignores_whitespace() {
        use super::{base64_decode, base64_encode};
        for raw in [
            &b""[..],
            b"f",
            b"fo",
            b"foo",
            b"foobar",
            &[0u8, 255, 128, 1, 254],
        ] {
            let text = base64_encode(raw);
            assert_eq!(base64_decode(&text).expect("round-trip"), raw);
        }
        // Embedded ASCII whitespace (newlines/spaces) is skipped, matching the standard decoder.
        assert_eq!(base64_decode("Zm9v\nYmFy").expect("decode"), b"foobar");
        // A non-alphabet character is rejected.
        assert!(base64_decode("Zm9v*").is_err());
    }

    // ----- TO_NUMBER (B-fn) -----

    #[test]
    fn to_number_reads_digits_sign_and_decimal() {
        use super::to_number_value;
        let num = |s: &str| match to_number_value(s).expect("parse") {
            crate::ast::Value::Numeric(d) => d.format(),
            other => panic!("expected numeric, got {other:?}"),
        };
        // Group separators, currency, and padding are dropped; the decimal point is kept.
        assert_eq!(num("12,345.67"), "12345.67");
        assert_eq!(num("$1,234.50"), "1234.50");
        assert_eq!(num("  48  "), "48");
        // Sign forms: leading '-', and the angle-bracket / parenthesised PR negatives.
        assert_eq!(num("-42"), "-42");
        assert_eq!(num("<12>"), "-12");
        assert_eq!(num("(99)"), "-99");
        // Only the first '.' is the decimal point; later dots are ignored as separators.
        assert_eq!(num("1.234.5"), "1.2345");
        // Text with no digits is an error, not a silent zero.
        assert!(to_number_value("abc").is_err());
    }

    // ----- LIKE ... ESCAPE -----

    #[test]
    fn like_match_without_escape_is_wildcard_only() {
        use super::like_match;
        // `%`/`_` are wildcards; no character escapes them.
        assert!(like_match("abc", "a%", None, false));
        assert!(like_match("axc", "a_c", None, false));
        assert!(!like_match("axc", "a_d", None, false));
    }

    #[test]
    fn like_match_honors_escape_char() {
        use super::like_match;
        // `!%` / `!_` match a *literal* `%` / `_`; wildcards still work alongside.
        assert!(like_match("a%b", "a!%b", Some('!'), false));
        assert!(!like_match("axb", "a!%b", Some('!'), false)); // literal % required, 'x' ≠ '%'
        assert!(like_match("a_b", "a!_b", Some('!'), false));
        assert!(!like_match("aXb", "a!_b", Some('!'), false)); // literal _ required, 'X' ≠ '_'
        // `!!` is a literal escape char; ordinary `%` still globs.
        assert!(like_match("a!b", "a!!b", Some('!'), false));
        assert!(like_match("axyzb", "a%b", Some('!'), false));
        // A trailing escape char has nothing to escape → it is itself a literal.
        assert!(like_match("a!", "a!", Some('!'), false));
    }

    // Deep-gate #12: ILIKE folds case per character, keeping `_` and an alphabetic `ESCAPE` correct.
    #[test]
    fn ilike_folds_case_per_character_and_keeps_escape_and_underscore() {
        use super::like_match;
        // Case-insensitive: letters match regardless of case; `%`/`_` unaffected.
        assert!(like_match("HELLO", "hello", None, true));
        assert!(like_match("HeLLo", "h%O", None, true));
        assert!(!like_match("HELLO", "hxllo", None, true)); // a non-letter mismatch still fails
        // (a) A `_` matches exactly one source character even when its lowercase form is longer:
        // `'İ'` (U+0130) lowercases to two code points, which a `LOWER()`-based ILIKE would break.
        assert!(like_match("İ", "_", None, true));
        // (b) An alphabetic `ESCAPE` keeps its case: `X` escapes the next char (a literal `z`), and the
        // letter still matches case-insensitively — a `LOWER()`-based ILIKE would lose the escape.
        assert!(like_match("aZ", "aXz", Some('X'), true));
        assert!(like_match("AZ", "aXz", Some('X'), true));
        assert!(!like_match("ay", "aXz", Some('X'), true)); // the escaped `z` is required
    }

    // ----- array subscript -----

    #[test]
    fn subscript_min_index_is_null_not_panic() {
        use nusadb_core::engine::ArrayElem;
        // `i64::MIN` must not overflow the 1-based → 0-based conversion (checked_sub).
        let array = TypedExpr {
            kind: TypedExprKind::ArrayLiteral(vec![lit_int(10), lit_int(20)]),
            ty: ColumnType::Array(ArrayElem::Int),
        };
        let subscript = TypedExpr {
            kind: TypedExprKind::Subscript {
                base: Box::new(array),
                index: Box::new(lit_int(i64::MIN)),
            },
            ty: ColumnType::Int,
        };
        assert_eq!(eval(&subscript, &vec![]).unwrap(), Value::Null);
    }
}
