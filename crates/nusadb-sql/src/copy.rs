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

/// A streaming iterator over the CSV records in a COPY payload, yielding one record's fields at a
/// time so a multi-million-row load is parsed incrementally (bounded memory) rather than up front.
///
/// CSV differs from the text format: fields are separated by `delimiter` (default `,`) and a field
/// may be *quoted* with `quote` (default `"`) to carry the delimiter, a newline, or a quote — a
/// literal quote is written by doubling it, or after `escape` when `escape` differs from `quote`. A
/// record ends at an unquoted `\n`, `\r\n`, or `\r`, or at end of input. Whether each field was
/// quoted is tracked so the `null` test applies only to *unquoted* fields: an unquoted field equal
/// to `null` is SQL `NULL` (`None`), while a quoted field is always a value (so a quoted empty
/// string is `Some("")`, distinct from an unquoted empty NULL).
#[derive(Debug, Clone)]
pub struct CsvRecords<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    delimiter: char,
    quote: char,
    escape: char,
    null: &'a str,
}

impl<'a> CsvRecords<'a> {
    /// Start parsing `data` as CSV with the given field/quote/escape characters and NULL marker.
    #[must_use]
    pub fn new(data: &'a str, delimiter: char, quote: char, escape: char, null: &'a str) -> Self {
        Self {
            chars: data.chars().peekable(),
            delimiter,
            quote,
            escape,
            null,
        }
    }

    /// Parse one field, returning its decoded text and whether it was quoted. A quoted field stops
    /// right after its closing quote; an unquoted field stops at the next delimiter, newline, or EOF
    /// (leaving that terminator for the record loop).
    fn parse_field(&mut self) -> Result<(String, bool), String> {
        if self.chars.peek() == Some(&self.quote) {
            self.chars.next(); // opening quote
            let mut buf = String::new();
            loop {
                let Some(ch) = self.chars.next() else {
                    return Err("unterminated quoted field in CSV data".to_owned());
                };
                if ch == self.escape && self.escape != self.quote {
                    // A distinct escape character makes the next character literal.
                    match self.chars.next() {
                        Some(next) => buf.push(next),
                        None => return Err("CSV escape character at end of input".to_owned()),
                    }
                } else if ch == self.quote {
                    // A quote is either a doubled quote (one literal quote) or the closing quote.
                    if self.chars.peek() == Some(&self.quote) {
                        self.chars.next();
                        buf.push(self.quote);
                    } else {
                        return Ok((buf, true));
                    }
                } else {
                    buf.push(ch);
                }
            }
        } else {
            // Unquoted: raw text up to the next delimiter / record terminator; CSV applies no escapes
            // to unquoted fields.
            let mut buf = String::new();
            while let Some(&ch) = self.chars.peek() {
                if ch == self.delimiter || ch == '\n' || ch == '\r' {
                    break;
                }
                buf.push(ch);
                self.chars.next();
            }
            Ok((buf, false))
        }
    }
}

impl Iterator for CsvRecords<'_> {
    type Item = Result<Vec<Option<String>>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // No more input → no more records (so a trailing record terminator yields nothing extra).
        self.chars.peek()?;
        let mut fields = Vec::new();
        loop {
            let (text, quoted) = match self.parse_field() {
                Ok(field) => field,
                Err(e) => return Some(Err(e)),
            };
            // Only an unquoted field can be NULL; a quoted field is always a value.
            fields.push(if !quoted && text == self.null {
                None
            } else {
                Some(text)
            });
            match self.chars.peek().copied() {
                Some(ch) if ch == self.delimiter => {
                    self.chars.next(); // consume the delimiter; parse the next field
                },
                Some('\r') => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'\n') {
                        self.chars.next();
                    }
                    return Some(Ok(fields));
                },
                Some('\n') => {
                    self.chars.next();
                    return Some(Ok(fields));
                },
                None => return Some(Ok(fields)),
                Some(other) => {
                    // Reachable only just after a quoted field (an unquoted field always stops exactly
                    // at a delimiter/newline/EOF): stray data after the closing quote.
                    return Some(Err(format!(
                        "unexpected character {other:?} after a quoted CSV field"
                    )));
                },
            }
        }
    }
}

/// Render a row of optional field values as one CSV data line (no trailing newline).
///
/// `None` is written as the `null` marker, unquoted. A value is quoted with `quote` when it contains
/// the delimiter, a quote, or a newline/carriage-return, or when leaving it unquoted would make it
/// read back as `NULL` (an empty string when `null` is empty, or a value equal to the `null` marker).
///
/// Inside a quoted value an embedded quote is written by doubling it when `escape == quote` (the
/// default), or as `escape`+`quote` otherwise; a non-default `escape` character is itself escaped
/// (`escape`+`escape`) so it is not misread as an escape on the way back. Round-trips through
/// [`CsvRecords`].
#[must_use]
pub fn format_csv_row(
    fields: &[Option<&str>],
    delimiter: char,
    quote: char,
    escape: char,
    null: &str,
) -> String {
    let mut out = String::new();
    for (i, field) in fields.iter().enumerate() {
        if i > 0 {
            out.push(delimiter);
        }
        let Some(value) = field else {
            out.push_str(null);
            continue;
        };
        let ambiguous_null = *value == null;
        let needs_quote = ambiguous_null
            || value
                .chars()
                .any(|c| c == delimiter || c == quote || c == '\n' || c == '\r');
        if needs_quote {
            out.push(quote);
            for c in value.chars() {
                if escape == quote {
                    // Default CSV: a quote is escaped by doubling it; there is no separate escape.
                    if c == quote {
                        out.push(quote);
                    }
                } else if c == quote || c == escape {
                    // A distinct escape char escapes both the quote and itself.
                    out.push(escape);
                }
                out.push(c);
            }
            out.push(quote);
        } else {
            out.push_str(value);
        }
    }
    out
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

    // --- CSV codec -------------------------------------------------------------------------------

    /// Parse a whole CSV payload into records with the standard defaults (`,` / `"` quote+escape /
    /// empty NULL), unwrapping any parse error.
    fn csv(data: &str) -> Vec<Vec<Option<String>>> {
        CsvRecords::new(data, ',', '"', '"', "")
            .map(Result::unwrap)
            .collect()
    }

    #[test]
    fn csv_splits_plain_comma_fields() {
        assert_eq!(
            csv("1,alice,30\n2,bob,25\n"),
            vec![
                vec![Some("1".into()), Some("alice".into()), Some("30".into())],
                vec![Some("2".into()), Some("bob".into()), Some("25".into())],
            ],
        );
    }

    #[test]
    fn csv_unquoted_empty_is_null_but_quoted_empty_is_a_string() {
        // `a,,b` — the middle unquoted-empty field is NULL; `a,"",b` — the quoted empty is "".
        assert_eq!(
            csv("a,,b"),
            vec![vec![Some("a".into()), None, Some("b".into())]],
        );
        assert_eq!(
            csv("a,\"\",b"),
            vec![vec![
                Some("a".into()),
                Some(String::new()),
                Some("b".into())
            ]],
        );
    }

    #[test]
    fn csv_quoted_field_carries_delimiter_newline_and_doubled_quote() {
        // One record: a quoted field containing a comma and a newline, and a doubled quote → `"`.
        assert_eq!(
            csv("\"a,b\nc\",\"she said \"\"hi\"\"\",z"),
            vec![vec![
                Some("a,b\nc".into()),
                Some("she said \"hi\"".into()),
                Some("z".into()),
            ]],
        );
    }

    #[test]
    fn csv_handles_crlf_and_final_line_without_newline() {
        assert_eq!(
            csv("1,2\r\n3,4"),
            vec![
                vec![Some("1".into()), Some("2".into())],
                vec![Some("3".into()), Some("4".into())],
            ],
        );
    }

    #[test]
    fn csv_custom_null_marker_only_applies_unquoted() {
        // NULL marker `\N`: an unquoted `\N` is NULL, a quoted `"\N"` is the literal text.
        let recs: Vec<_> = CsvRecords::new("\\N,\"\\N\"", ',', '"', '"', "\\N")
            .map(Result::unwrap)
            .collect();
        assert_eq!(recs, vec![vec![None, Some("\\N".into())]]);
    }

    #[test]
    fn csv_custom_escape_character() {
        // ESCAPE `\`: `\"` inside a quoted field is a literal quote.
        let recs: Vec<_> = CsvRecords::new("\"a\\\"b\"", ',', '"', '\\', "")
            .map(Result::unwrap)
            .collect();
        assert_eq!(recs, vec![vec![Some("a\"b".into())]]);
    }

    #[test]
    fn csv_unterminated_quote_is_an_error() {
        let mut it = CsvRecords::new("\"oops", ',', '"', '"', "");
        assert!(it.next().unwrap().is_err());
    }

    #[test]
    fn csv_format_quotes_only_when_needed_and_round_trips() {
        // plain stays bare; delimiter/quote/newline force quoting; NULL → empty; ""→quoted empty.
        let row = vec![
            Some("plain"),
            Some("a,b"),
            Some("she \"said\""),
            None,
            Some(""),
        ];
        let line = format_csv_row(&row, ',', '"', '"', "");
        assert_eq!(line, "plain,\"a,b\",\"she \"\"said\"\"\",,\"\"");
        // Round-trips: the empty NULL and the quoted empty string come back distinct.
        assert_eq!(
            csv(&line),
            vec![vec![
                Some("plain".into()),
                Some("a,b".into()),
                Some("she \"said\"".into()),
                None,
                Some(String::new()),
            ]],
        );
    }

    #[test]
    fn csv_format_with_custom_escape_round_trips() {
        // ESCAPE `\` (≠ quote): a value that needs quoting and contains both the quote and the escape
        // char must escape both so the reader recovers it exactly.
        let row = vec![Some("a,\"b\\c")];
        let line = format_csv_row(&row, ',', '"', '\\', "");
        // The comma forces quoting; the embedded quote and backslash are each escaped with `\`.
        assert_eq!(line, "\"a,\\\"b\\\\c\"");
        let back: Vec<_> = CsvRecords::new(&line, ',', '"', '\\', "")
            .map(Result::unwrap)
            .collect();
        assert_eq!(back, vec![vec![Some("a,\"b\\c".into())]]);
    }
}
