//! Stream codecs for the `COPY` sub-protocol: text, CSV, and binary.
//!
//! In the text format `COPY` streams rows as newline-separated *data lines*; within a line, fields
//! are separated by a delimiter (default tab) and SQL `NULL` is written as a marker token (default
//! `\N`). Special characters inside a field are backslash-escaped. This module is the single place
//! that parses a data line into fields ([`parse_text_row`]) and renders a row back to a data line
//! ([`format_text_row`]); the analyzer/executor and the wire server share it so load and export
//! round-trip exactly. CSV has [`CsvRecords`] and [`format_csv_row`].
//!
//! The binary format frames rows with length prefixes instead of delimiters:
//!
//! ```text
//! header   signature (11 bytes: "NUSACOPY" 0xFF '\r' '\n')
//!          flags: u32 (0)  extension length: u32, then that many extension bytes
//! row      field count: i16, then per field: length i32 (-1 = NULL) and that many bytes
//! trailer  field count -1 as i16
//! ```
//!
//! Every integer is big-endian. Each field's bytes are the value's binary layout
//! ([`crate::binary`]), the same bytes a binary result column carries. [`BinaryRecords`] reads a
//! stream and [`binary_header`] / [`push_binary_row`] / [`binary_trailer`] write one.

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

/// The 11-byte signature that opens a binary `COPY` stream. The `0xFF` byte catches a channel that
/// strips the high bit, and the `\r\n` pair one that rewrites line endings.
pub const BINARY_SIGNATURE: [u8; 11] = *b"NUSACOPY\xff\r\n";

/// Write the binary stream header: the signature, a zero flags word, and an empty header extension.
pub fn binary_header(out: &mut Vec<u8>) {
    out.extend_from_slice(&BINARY_SIGNATURE);
    out.extend_from_slice(&0_u32.to_be_bytes()); // flags
    out.extend_from_slice(&0_u32.to_be_bytes()); // header extension length
}

/// Append one binary row: the field count, then each field as a length and its bytes (`None` is
/// SQL `NULL`, written as length `-1` with no bytes).
///
/// # Errors
/// A message when the row has more fields than an `i16` count holds or a field is longer than an
/// `i32` length holds. Nothing is appended in that case.
pub fn push_binary_row(out: &mut Vec<u8>, fields: &[Option<Vec<u8>>]) -> Result<(), String> {
    let count = i16::try_from(fields.len())
        .map_err(|_| format!("a binary COPY row holds at most {} fields", i16::MAX))?;
    let mut row = Vec::with_capacity(2 + fields.len() * 4);
    row.extend_from_slice(&count.to_be_bytes());
    for field in fields {
        match field {
            None => row.extend_from_slice(&(-1_i32).to_be_bytes()),
            Some(bytes) => {
                let len = i32::try_from(bytes.len())
                    .map_err(|_| format!("a binary COPY field holds at most {} bytes", i32::MAX))?;
                row.extend_from_slice(&len.to_be_bytes());
                row.extend_from_slice(bytes);
            },
        }
    }
    out.extend_from_slice(&row);
    Ok(())
}

/// Write the binary stream trailer: a field count of `-1`.
pub fn binary_trailer(out: &mut Vec<u8>) {
    out.extend_from_slice(&(-1_i16).to_be_bytes());
}

/// A streaming reader over the rows of a binary `COPY` payload.
///
/// [`BinaryRecords::new`] checks the header; each step then yields one row's fields as borrowed
/// byte slices (`None` for SQL `NULL`), so a large load is decoded row by row. The stream must end
/// with the trailer and nothing after it; a payload that stops early, a negative length other than
/// `-1`, or a length running past the end is an error, after which the reader yields nothing more.
#[derive(Debug, Clone)]
pub struct BinaryRecords<'a> {
    data: &'a [u8],
    pos: usize,
    done: bool,
}

impl<'a> BinaryRecords<'a> {
    /// Check the header of `data` and position the reader at the first row.
    ///
    /// # Errors
    /// A message when the signature does not match, the header is cut short, a flag bit is set
    /// (none are defined), or the header extension runs past the end.
    pub fn new(data: &'a [u8]) -> Result<Self, String> {
        if data.get(..BINARY_SIGNATURE.len()) != Some(&BINARY_SIGNATURE[..]) {
            return Err("binary COPY data does not start with the NUSACOPY signature".to_owned());
        }
        let mut reader = Self {
            data,
            pos: BINARY_SIGNATURE.len(),
            done: false,
        };
        let flags = reader.u32("header flags")?;
        if flags != 0 {
            return Err(format!(
                "binary COPY header has unknown flags 0x{flags:08X}"
            ));
        }
        let extension = reader.u32("header extension length")?;
        reader.read_bytes(
            usize::try_from(extension).unwrap_or(usize::MAX),
            "header extension",
        )?;
        Ok(reader)
    }

    /// Consume `n` bytes, or report what was cut short.
    fn read_bytes(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.data.len());
        let Some(end) = end else {
            return Err(format!(
                "binary COPY data ends inside the {what} at byte {}",
                self.pos
            ));
        };
        let bytes = self.data.get(self.pos..end).unwrap_or_default();
        self.pos = end;
        Ok(bytes)
    }

    fn u32(&mut self, what: &str) -> Result<u32, String> {
        let bytes = self.read_bytes(4, what)?;
        Ok(u32::from_be_bytes(bytes.try_into().unwrap_or_default()))
    }

    fn i32(&mut self, what: &str) -> Result<i32, String> {
        let bytes = self.read_bytes(4, what)?;
        Ok(i32::from_be_bytes(bytes.try_into().unwrap_or_default()))
    }

    fn i16(&mut self, what: &str) -> Result<i16, String> {
        let bytes = self.read_bytes(2, what)?;
        Ok(i16::from_be_bytes(bytes.try_into().unwrap_or_default()))
    }

    /// Read the next row, or `None` after the trailer.
    fn row(&mut self) -> Result<Option<Vec<Option<&'a [u8]>>>, String> {
        if self.pos == self.data.len() {
            return Err("binary COPY data ends without its trailer".to_owned());
        }
        let count = self.i16("field count")?;
        if count == -1 {
            if self.pos != self.data.len() {
                return Err(format!(
                    "binary COPY data continues for {} byte(s) after its trailer",
                    self.data.len() - self.pos
                ));
            }
            return Ok(None);
        }
        let count = usize::try_from(count)
            .map_err(|_| format!("binary COPY row has a negative field count {count}"))?;
        let mut fields = Vec::with_capacity(count);
        for _ in 0..count {
            let len = self.i32("field length")?;
            fields.push(match len {
                -1 => None,
                len => {
                    let len = usize::try_from(len)
                        .map_err(|_| format!("binary COPY field has a negative length {len}"))?;
                    Some(self.read_bytes(len, "field data")?)
                },
            });
        }
        Ok(Some(fields))
    }
}

impl<'a> Iterator for BinaryRecords<'a> {
    type Item = Result<Vec<Option<&'a [u8]>>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.row() {
            Ok(Some(fields)) => Some(Ok(fields)),
            Ok(None) => {
                self.done = true;
                None
            },
            Err(e) => {
                self.done = true;
                Some(Err(e))
            },
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

    // --- binary codec ----------------------------------------------------------------------------

    /// A binary stream holding `rows`, with the header and trailer.
    fn binary(rows: &[Vec<Option<Vec<u8>>>]) -> Vec<u8> {
        let mut out = Vec::new();
        binary_header(&mut out);
        for row in rows {
            push_binary_row(&mut out, row).unwrap();
        }
        binary_trailer(&mut out);
        out
    }

    #[test]
    fn binary_round_trips_values_nulls_and_empty_fields() {
        let rows = vec![
            vec![Some(b"ab".to_vec()), None, Some(Vec::new())],
            vec![None, Some(vec![0xFF]), None],
        ];
        let data = binary(&rows);
        assert_eq!(&data[..11], b"NUSACOPY\xff\r\n");
        let back: Vec<Vec<Option<Vec<u8>>>> = BinaryRecords::new(&data)
            .unwrap()
            .map(|r| {
                r.unwrap()
                    .into_iter()
                    .map(|f| f.map(<[u8]>::to_vec))
                    .collect()
            })
            .collect();
        assert_eq!(back, rows);
    }

    #[test]
    fn binary_header_is_checked() {
        let mut data = binary(&[]);
        data[0] = b'X';
        assert!(BinaryRecords::new(&data).is_err());
        assert!(BinaryRecords::new(&BINARY_SIGNATURE[..5]).is_err());
        let mut flagged = binary(&[]);
        flagged[14] = 1; // low byte of the flags word
        assert!(BinaryRecords::new(&flagged).is_err());
    }

    #[test]
    fn binary_header_extension_is_skipped() {
        let mut data = BINARY_SIGNATURE.to_vec();
        data.extend_from_slice(&0_u32.to_be_bytes());
        data.extend_from_slice(&3_u32.to_be_bytes());
        data.extend_from_slice(b"xyz");
        binary_trailer(&mut data);
        assert_eq!(BinaryRecords::new(&data).unwrap().count(), 0);
    }

    #[test]
    fn binary_missing_trailer_and_trailing_bytes_are_errors() {
        let mut data = binary(&[vec![Some(b"a".to_vec())]]);
        data.truncate(data.len() - 2);
        let mut it = BinaryRecords::new(&data).unwrap();
        assert!(it.next().unwrap().is_ok());
        assert!(it.next().unwrap().is_err());
        assert!(it.next().is_none());

        let mut data = binary(&[]);
        data.push(0);
        assert!(BinaryRecords::new(&data).unwrap().next().unwrap().is_err());
    }

    #[test]
    fn binary_length_past_end_and_negative_length_are_errors() {
        let mut data = Vec::new();
        binary_header(&mut data);
        data.extend_from_slice(&1_i16.to_be_bytes());
        data.extend_from_slice(&10_i32.to_be_bytes());
        data.extend_from_slice(b"abc");
        assert!(BinaryRecords::new(&data).unwrap().next().unwrap().is_err());

        let mut data = Vec::new();
        binary_header(&mut data);
        data.extend_from_slice(&1_i16.to_be_bytes());
        data.extend_from_slice(&(-2_i32).to_be_bytes());
        binary_trailer(&mut data);
        assert!(BinaryRecords::new(&data).unwrap().next().unwrap().is_err());
    }
}
