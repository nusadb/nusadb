//! XML value support for the `XML` column type.
//!
//! An `xml` value stores its **original text verbatim** — unlike `jsonb`, the reference engine does
//! not canonicalize XML. The only work at input time is a well-formedness check: markup must be
//! balanced and properly nested. Two modes match the reference engine's `XMLPARSE`:
//!
//! - `CONTENT` (also the `text::xml` cast): a well-formed fragment — any number of top-level nodes,
//!   including bare text and multiple elements.
//! - `DOCUMENT`: exactly one top-level element (an XML document), with no top-level character data.

use quick_xml::events::Event;
use quick_xml::reader::Reader;

/// Which well-formedness rules to apply when validating an `xml` value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum XmlMode {
    /// `XMLPARSE(DOCUMENT …)`: a single-rooted XML document.
    Document,
    /// `XMLPARSE(CONTENT …)` / the `text::xml` cast: a well-formed content fragment.
    Content,
}

/// Validate that `input` is well-formed XML under `mode`.
///
/// Returns `Ok(())` when valid; `Err(reason)` with a short message otherwise. The input text is never
/// modified — the caller stores it as-is.
///
/// # Errors
///
/// A malformed fragment (unbalanced or mis-nested tags, bad markup), or — in [`XmlMode::Document`] —
/// content that is not a single rooted element.
pub fn validate(input: &str, mode: XmlMode) -> Result<(), String> {
    let mut reader = Reader::from_str(input);
    reader.config_mut().check_end_names = true;
    let mut depth: i32 = 0;
    let mut top_level_elements: u32 = 0;
    loop {
        match reader.read_event() {
            Ok(Event::Eof) => break,
            Ok(Event::Start(_)) => {
                if depth == 0 {
                    top_level_elements += 1;
                }
                depth += 1;
            },
            Ok(Event::End(_)) => depth -= 1,
            Ok(Event::Empty(_)) => {
                if depth == 0 {
                    top_level_elements += 1;
                }
            },
            // Character data outside any element is not allowed in a document (it is in content).
            Ok(Event::Text(text)) => {
                if mode == XmlMode::Document
                    && depth == 0
                    && text.iter().any(|b| !b.is_ascii_whitespace())
                {
                    return Err("XML document must have a single root element".to_owned());
                }
            },
            Ok(Event::CData(_)) if mode == XmlMode::Document && depth == 0 => {
                return Err("XML document must have a single root element".to_owned());
            },
            // Comments, processing instructions, the XML declaration, DOCTYPE, and CDATA inside an
            // element are all well-formed markup.
            Ok(_) => {},
            Err(why) => return Err(why.to_string()),
        }
    }
    // A streaming reader reaches EOF happily with tags still open, so require every element closed.
    if depth != 0 {
        return Err("XML has an unclosed element".to_owned());
    }
    if mode == XmlMode::Document && top_level_elements != 1 {
        return Err("XML document must have a single root element".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{XmlMode, validate};

    #[test]
    fn content_accepts_fragments_and_bare_text() {
        assert!(validate("<a>1</a>", XmlMode::Content).is_ok());
        assert!(validate(r#"<a x="1">t</a>"#, XmlMode::Content).is_ok());
        assert!(validate("<a/>", XmlMode::Content).is_ok());
        // A fragment: multiple top-level nodes and bare text are valid content.
        assert!(validate("a<b/>c", XmlMode::Content).is_ok());
        assert!(validate("notxml", XmlMode::Content).is_ok());
    }

    #[test]
    fn document_requires_a_single_root() {
        assert!(validate("<r><c/></r>", XmlMode::Document).is_ok());
        // A fragment / bare text is not a document.
        assert!(validate("a<b/>c", XmlMode::Document).is_err());
        assert!(validate("<a/><b/>", XmlMode::Document).is_err());
        assert!(validate("text", XmlMode::Document).is_err());
    }

    #[test]
    fn malformed_markup_is_rejected_in_both_modes() {
        for mode in [XmlMode::Content, XmlMode::Document] {
            assert!(validate("<a>", mode).is_err(), "unclosed tag in {mode:?}");
            assert!(
                validate("<a></b>", mode).is_err(),
                "mismatched tag in {mode:?}"
            );
        }
    }
}
