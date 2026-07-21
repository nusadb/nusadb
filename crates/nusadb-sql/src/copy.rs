//! Text-format codec for the `COPY` sub-protocol.
//!
//! `COPY` streams rows as newline-separated *data lines*; within a line, fields are separated by a
//! delimiter (default tab) and SQL `NULL` is written as a marker token (default `\N`). Special
//! characters inside a field are backslash-escaped. This module is the single place that parses a
//! data line into fields ([`parse_text_row`]) and renders a row back to a data line
//! ([`format_text_row`]); the analyzer/executor and the wire server share it so load and export
//! round-trip exactly.

/// Parse one text-format data line into its fields.
///
/// Fields are split on unescaped `delimiter` characters. A field whose raw text equals `null`
/// (e.g. `\N`) is SQL `NULL` (`None`); otherwise the field's backslash escapes are decoded
/// (`\t`/`\n`/`\r`/`\\`, and `\x` for any other `x` yields `x`). A trailing `\r` (from CRLF line
/// endings) is not special-cased here — callers split on `\n` and may trim `\r` first.
#[must_use]
pub fn parse_text_row(line: &str, delimiter: char, null: &str) -> Vec<Option<String>> {
    split_raw_fields(line, delimiter)
        .into_iter()
        .map(|raw| {
            if raw == null {
                None
            } else {
                Some(unescape(&raw))
            }
        })
        .collect()
}

/// Render a row of optional field values as one text-format data line (no trailing newline).
///
/// `None` becomes the `null` marker; every other field has its delimiter, backslash, tab, newline,
/// and carriage-return escaped so [`parse_text_row`] recovers it exactly.
#[must_use]
pub fn format_text_row(fields: &[Option<&str>], delimiter: char, null: &str) -> String {
    let mut out = String::new();
    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            out.push(delimiter);
        }
        match field {
            None => out.push_str(null),
            Some(value) => escape_into(value, delimiter, &mut out),
        }
    }
    out
}

/// Split a line into raw (still-escaped) fields on unescaped `delimiter` characters. A backslash
/// makes the following character literal for splitting, so an escaped delimiter stays in the field.
fn split_raw_fields(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            // Keep the backslash and the escaped char together for the field; `unescape` decodes it.
            current.push('\\');
            if let Some(escaped) = chars.next() {
                current.push(escaped);
            }
        } else if ch == delimiter {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    fields.push(current);
    fields
}

/// Decode backslash escapes in a single raw field.
fn unescape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                // `\\` and any other escaped character are taken literally (text-format rule).
                Some(other) => out.push(other),
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Escape a field value's special characters into `out`.
fn escape_into(value: &str, delimiter: char, out: &mut String) {
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == delimiter => {
                out.push('\\');
                out.push(c);
            },
            c => out.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAB: char = '\t';
    const NULL: &str = "\\N";

    #[test]
    fn parses_tab_separated_fields() {
        assert_eq!(
            parse_text_row("1\talice\t30", TAB, NULL),
            vec![
                Some("1".to_owned()),
                Some("alice".to_owned()),
                Some("30".to_owned())
            ],
        );
    }

    #[test]
    fn null_marker_becomes_none() {
        assert_eq!(
            parse_text_row("1\t\\N\t3", TAB, NULL),
            vec![Some("1".to_owned()), None, Some("3".to_owned())],
        );
    }

    #[test]
    fn decodes_escapes_and_escaped_delimiter() {
        // `a\tb` (escaped tab) is one field containing a literal tab, not two fields.
        assert_eq!(
            parse_text_row("a\\tb\tc", TAB, NULL),
            vec![Some("a\tb".to_owned()), Some("c".to_owned())],
        );
        assert_eq!(
            parse_text_row("x\\\\y", TAB, NULL),
            vec![Some("x\\y".to_owned())],
        );
    }

    #[test]
    fn empty_field_is_empty_string_not_null() {
        assert_eq!(
            parse_text_row("\t", TAB, NULL),
            vec![Some(String::new()), Some(String::new())],
        );
    }

    #[test]
    fn format_round_trips_through_parse() {
        let row = vec![Some("a\tb"), None, Some("plain"), Some("back\\slash")];
        let line = format_text_row(&row, TAB, NULL);
        let parsed = parse_text_row(&line, TAB, NULL);
        assert_eq!(
            parsed,
            vec![
                Some("a\tb".to_owned()),
                None,
                Some("plain".to_owned()),
                Some("back\\slash".to_owned()),
            ],
        );
    }

    #[test]
    fn format_writes_null_marker_and_escapes() {
        assert_eq!(
            format_text_row(&[Some("x"), None, Some("a\tb")], TAB, NULL),
            "x\t\\N\ta\\tb",
        );
    }
}
