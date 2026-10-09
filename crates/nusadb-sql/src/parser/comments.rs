//! Comments around and inside a statement.
//!
//! Several statements are recognized from their leading words before the generic parser runs, and
//! a few of those read the statement's last word too. A comment in front of the statement (a
//! migration file's header), after it, or between its words must not change what it is, so
//! [`blank_comments`] turns every comment into spaces before any of that looks at the text.

use std::borrow::Cow;

/// `sql` with every comment (`-- …` to the end of the line, and `/* … */`, which nests) replaced
/// by spaces. Line breaks inside a comment are kept, so the remaining tokens keep their line and
/// column in an error message.
///
/// A comment marker inside a string (`'…'` with `''` doubling, `E'…'` with backslash escapes), a
/// quoted identifier (`"…"`) or a dollar-quoted body (`$$ … $$`, `$tag$ … $tag$`) is text, not a
/// comment, and is left untouched. An unterminated block comment leaves `sql` unchanged so the
/// parser reports it.
pub(super) fn blank_comments(sql: &str) -> Cow<'_, str> {
    if !sql.contains("--") && !sql.contains("/*") {
        return Cow::Borrowed(sql);
    }
    let bytes = sql.as_bytes();
    let mut out = bytes.to_vec();
    let mut i = 0;
    while let Some(&b) = bytes.get(i) {
        let next = bytes.get(i + 1).copied();
        i = match b {
            b'\'' => skip_quoted(bytes, i, b'\'', is_escape_string(bytes, i)),
            b'"' => skip_quoted(bytes, i, b'"', false),
            b'$' if !follows_identifier(bytes, i) => skip_dollar_quoted(bytes, i),
            b'-' if next == Some(b'-') => {
                let end = bytes
                    .iter()
                    .skip(i)
                    .position(|&c| c == b'\n')
                    .map_or(bytes.len(), |p| i + p);
                blank(&mut out, i, end);
                end
            },
            b'/' if next == Some(b'*') => {
                let Some(end) = block_comment_end(bytes, i) else {
                    return Cow::Borrowed(sql);
                };
                blank(&mut out, i, end);
                end
            },
            _ => i + 1,
        };
    }
    // Every byte of a comment became an ASCII space or stayed a line break, and a comment starts
    // and ends on ASCII bytes, so no multi-byte character was split.
    String::from_utf8(out).map_or(Cow::Borrowed(sql), Cow::Owned)
}

/// `sql` from its first word, with leading whitespace and comments skipped.
///
/// A check of a statement's first word then sees the statement even when a comment comes first.
/// An unterminated block comment is returned as is.
#[must_use]
pub fn skip_leading_comments(sql: &str) -> &str {
    let mut rest = sql.trim_start();
    loop {
        if let Some(after) = rest.strip_prefix("--") {
            rest = after
                .find('\n')
                .and_then(|p| after.get(p..))
                .unwrap_or("")
                .trim_start();
        } else if rest.starts_with("/*") {
            let Some(end) = block_comment_end(rest.as_bytes(), 0) else {
                return rest;
            };
            rest = rest.get(end..).unwrap_or("").trim_start();
        } else {
            return rest;
        }
    }
}

/// Overwrite `out[start..end]` with spaces, keeping line breaks.
fn blank(out: &mut [u8], start: usize, end: usize) {
    for byte in out.iter_mut().take(end).skip(start) {
        if *byte != b'\n' && *byte != b'\r' {
            *byte = b' ';
        }
    }
}

/// The offset just past the `*/` closing the block comment opened at `start`, counting nested
/// `/* … */` pairs, or `None` when it is never closed.
fn block_comment_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut j = start;
    loop {
        match (bytes.get(j)?, bytes.get(j + 1)) {
            (b'/', Some(b'*')) => {
                depth += 1;
                j += 2;
            },
            (b'*', Some(b'/')) => {
                depth -= 1;
                j += 2;
                if depth == 0 {
                    return Some(j);
                }
            },
            _ => j += 1,
        }
    }
}

const fn is_identifier_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$' || !b.is_ascii()
}

/// Whether the byte before `i` continues an identifier (so a `$` at `i` is part of it).
fn follows_identifier(bytes: &[u8], i: usize) -> bool {
    i.checked_sub(1)
        .and_then(|p| bytes.get(p))
        .is_some_and(|&b| is_identifier_byte(b))
}

/// Whether the quote at `i` opens an `E'…'` string, where a backslash escapes the next character.
fn is_escape_string(bytes: &[u8], i: usize) -> bool {
    i.checked_sub(1)
        .and_then(|p| bytes.get(p))
        .is_some_and(|&b| b == b'e' || b == b'E')
        && !follows_identifier(bytes, i - 1)
}

/// The offset just past the string or quoted identifier opened by `quote` at `start` (a doubled
/// quote stays inside it), or the end of `bytes` when it is never closed.
fn skip_quoted(bytes: &[u8], start: usize, quote: u8, backslash_escapes: bool) -> usize {
    let mut j = start + 1;
    while let Some(&b) = bytes.get(j) {
        if backslash_escapes && b == b'\\' {
            j += 2;
        } else if b == quote {
            if bytes.get(j + 1) == Some(&quote) {
                j += 2;
            } else {
                return j + 1;
            }
        } else {
            j += 1;
        }
    }
    bytes.len()
}

/// The offset just past the dollar-quoted body opened at `start` (`$$` or `$tag$`, the tag an
/// identifier that does not start with a digit, so `$1` is a parameter), or `start + 1` when no
/// dollar quote opens there.
fn skip_dollar_quoted(bytes: &[u8], start: usize) -> usize {
    let tag_len = bytes
        .iter()
        .skip(start + 1)
        .position(|&b| !(b.is_ascii_alphanumeric() || b == b'_' || !b.is_ascii()));
    let Some(tag_len) = tag_len else {
        return start + 1;
    };
    let close = start + 1 + tag_len;
    let starts_with_digit = bytes.get(start + 1).is_some_and(u8::is_ascii_digit);
    if bytes.get(close) != Some(&b'$') || (tag_len > 0 && starts_with_digit) {
        return start + 1;
    }
    let Some(delimiter) = bytes.get(start..=close) else {
        return start + 1;
    };
    let body = close + 1;
    bytes
        .get(body..)
        .and_then(|rest| rest.windows(delimiter.len()).position(|w| w == delimiter))
        .map_or(bytes.len(), |p| body + p + delimiter.len())
}

#[cfg(test)]
mod tests {
    use super::{blank_comments, skip_leading_comments};
    use crate::parse;

    #[test]
    fn comments_become_spaces_and_keep_their_line_breaks() {
        let sql = "-- head\nSELECT 1 /* a /* nested */ b */ + 2 -- tail";
        let blanked = blank_comments(sql);
        assert_eq!(blanked.len(), sql.len());
        assert_eq!(
            blanked.split_whitespace().collect::<Vec<_>>(),
            ["SELECT", "1", "+", "2"]
        );
        // Every remaining token keeps its offset.
        assert_eq!(blanked.find('+'), sql.find('+'));
        assert_eq!(blanked.lines().count(), 2);
        let block = "SELECT 1 /* one\ntwo */ + 2";
        assert_eq!(blank_comments(block).lines().count(), 2);
        let multibyte = "SELECT 1 /* ünïcødé */";
        assert_eq!(blank_comments(multibyte).trim_end(), "SELECT 1");
    }

    #[test]
    fn markers_inside_quotes_are_text() {
        for sql in [
            "SELECT '-- not a comment', 'a /* b */ c'",
            "SELECT 'it''s -- still text'",
            "SELECT E'\\' -- still text'",
            "SELECT \"col -- name\" FROM t",
            "CREATE FUNCTION f() RETURNS INT AS $$ SELECT 1 -- body $$",
            "CREATE FUNCTION f() RETURNS INT AS $fn$ SELECT '$$' /* x */ $fn$",
        ] {
            assert_eq!(blank_comments(sql), sql, "{sql}");
        }
        // `$1` is a parameter and `a$b` an identifier, not the start of a dollar-quoted body.
        assert_eq!(blank_comments("SELECT $1 -- x").trim_end(), "SELECT $1");
        assert_eq!(blank_comments("SELECT a$b$ -- x").trim_end(), "SELECT a$b$");
        // Not an escape string: the `e` ends an identifier.
        assert_eq!(
            blank_comments("SELECT name'x' -- c").trim_end(),
            "SELECT name'x'"
        );
    }

    #[test]
    fn an_unterminated_block_comment_is_left_for_the_parser() {
        let sql = "SELECT 1 /* never closed";
        assert_eq!(blank_comments(sql), sql);
        assert!(parse(sql).is_err());
        assert_eq!(skip_leading_comments("/* open"), "/* open");
    }

    #[test]
    fn skip_leading_comments_finds_the_first_word() {
        assert_eq!(
            skip_leading_comments("  -- a\n /* b /* c */ */\n\tBEGIN"),
            "BEGIN"
        );
        assert_eq!(skip_leading_comments("-- only a comment"), "");
        assert_eq!(
            skip_leading_comments("COPY t FROM STDIN"),
            "COPY t FROM STDIN"
        );
    }

    #[test]
    fn every_statement_kind_parses_the_same_with_comments_around_it() {
        let statements = [
            "CREATE TABLE c (id INT PRIMARY KEY, v TEXT)",
            "SELECT * FROM c",
            "CREATE FUNCTION f(x INT) RETURNS INT LANGUAGE sql AS $$ SELECT x + 1 $$",
            "CREATE TRIGGER tg AFTER INSERT ON c FOR EACH ROW EXECUTE FUNCTION g()",
            "CREATE ROLE r LOGIN",
            "GRANT SELECT ON c TO r",
            "REVOKE SELECT ON c FROM r",
            "CREATE TYPE e AS ENUM ('a', 'b')",
            "CREATE PROCEDURE p() LANGUAGE sql AS $$ SELECT 1 $$",
            "CALL p()",
            "COMMENT ON TABLE c IS 'x -- y'",
            "CREATE POLICY pol ON c USING (true)",
            "VACUUM",
            "CHECKPOINT",
            "LISTEN ch",
            "NOTIFY ch",
            "SET ROLE r",
            "RESET ROLE",
            "DO $$ BEGIN END $$",
            "BEGIN",
            "COMMIT",
            "DROP TRIGGER tg ON c",
            "DROP FUNCTION f",
            "DROP ROLE r",
            "COPY c FROM STDIN",
        ];
        for sql in statements {
            let plain = parse(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
            for commented in [
                format!("-- leading\n{sql}"),
                format!("/* leading */ {sql}"),
                format!("/* a /* nested */ b */\n{sql}"),
                format!("-- one\n-- two\n/* three */\n{sql}"),
                format!("{sql} -- trailing"),
                format!("{sql} /* trailing */"),
            ] {
                assert_eq!(parse(&commented).ok(), Some(plain.clone()), "{commented}");
            }
        }
    }
}
