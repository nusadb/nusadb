//! Full-text search (F1): the `simple` text-search configuration, canonical
//! `tsvector`/`tsquery` text values, and the `@@` match.
//!
//! Values are carried as `TEXT` in the reference engine's canonical form — a `tsvector` is
//! `'brown':3 'fox':4 'quick':2` (lexemes sorted, always quoted, positions ascending) and a
//! `tsquery` is `'fox' & !'lazy'` — so no storage/type-system change is needed; a dedicated
//! `ColumnType` can later adopt the same encoding.
//!
//! Two configurations are implemented, both aiming for byte-exact parity with the reference engine on plain-word
//! input: **`simple`** (lowercase only) and **`english`** (the Snowball stopword list plus a
//! from-scratch Snowball English / Porter2 stemmer, `english` being the reference engine's default configuration).
//! Honest scope (anti-silent-wrong): other configurations, the reference engine's compound token classes
//! (email/url/host/file, hyphenated compounds), phrase search (`<->`), and weight/prefix suffixes
//! are rejected loudly rather than producing near-but-not-equal results.

use crate::error::Error;

/// The reference engine caps a lexeme position at 16383; larger positions are clamped.
const MAX_POSITION: u32 = 16_383;
/// The reference engine stores at most 256 positions per lexeme; the rest are dropped.
const MAX_POSITIONS_PER_LEXEME: usize = 256;

/// A syntax error in a `tsquery`, with the reference engine's SQLSTATE (42601 `syntax_error`).
fn syntax_error(input: &str) -> Error {
    Error::Coded {
        message: format!("syntax error in tsquery: {input:?}"),
        sqlstate: "42601",
    }
}

/// A supported text-search configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Config {
    /// Lowercase only — no stemming, no stopwords.
    Simple,
    /// The reference engine's `english`: the Snowball stopword list plus the Snowball English (Porter2) stemmer.
    English,
}

/// Resolve a configuration name; any other name is a loud reject (not near-parity).
fn check_config(config: &str) -> Result<Config, Error> {
    if config.eq_ignore_ascii_case("simple") {
        Ok(Config::Simple)
    } else if config.eq_ignore_ascii_case("english") {
        Ok(Config::English)
    } else {
        Err(Error::Unsupported(format!(
            "text search configuration {config:?} is not implemented (supported: 'simple', \
             'english')"
        )))
    }
}

/// Normalize one lowercased token under `config`: `None` drops it (a stopword), `Some` keeps the
/// (possibly stemmed) lexeme. A token containing a digit maps to the reference engine's `numword` class, which the
/// `english` configuration sends to the *simple* dictionary — no stopword check, no stemming.
fn normalize_lexeme(config: Config, lexeme: &str) -> Option<String> {
    match config {
        Config::Simple => Some(lexeme.to_owned()),
        Config::English => {
            if lexeme.chars().any(|c| c.is_ascii_digit()) {
                Some(lexeme.to_owned())
            } else if is_stopword(lexeme) {
                None
            } else {
                Some(porter2::stem(lexeme))
            }
        },
    }
}

/// The Snowball English stopword list, exactly as the reference engine's `english.stop` ships it (sorted for binary
/// search). The lone `s`/`t`/`don` entries are the apostrophe fragments (`it's` tokenizes to
/// `it` + `s`).
const STOPWORDS: &[&str] = &[
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
];

/// Whether `word` (lowercased) is an English stopword.
fn is_stopword(word: &str) -> bool {
    STOPWORDS.binary_search(&word).is_ok()
}

/// The Snowball English (Porter2) stemmer.
///
/// A from-scratch implementation of the Snowball English stemming algorithm (the successor to the
/// Porter stemmer; public specification at snowballstem.org), which is what the reference engine's `english_stem`
/// dictionary runs — so `to_tsvector('english', …)` output can be compared byte-for-byte with the reference engine.
/// `y` is marked as the consonant `Y` when word-initial or after a vowel; R1/R2 are the standard
/// regions (with the special `gener`/`commun`/`arsen` R1 prefixes); the steps follow the published
/// algorithm, including its two exception lists.
mod porter2 {
    #![allow(
        clippy::indexing_slicing,
        reason = "every index and slice is bounds-guarded by a length check in the same expression"
    )]

    /// Doubled consonant endings undone by step 1b.
    const DOUBLES: &[&str] = &["bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt"];

    /// Whether `w[i]` is a vowel (`a e i o u y` — the marked consonant `Y` is not).
    fn is_vowel(w: &[char], i: usize) -> bool {
        matches!(w.get(i), Some('a' | 'e' | 'i' | 'o' | 'u' | 'y'))
    }

    /// Whether the word ends with `suffix`; if so, returns the index where the suffix starts.
    fn suffix_at(w: &[char], suffix: &str) -> Option<usize> {
        let n = suffix.chars().count();
        let start = w.len().checked_sub(n)?;
        if w[start..].iter().copied().eq(suffix.chars()) {
            Some(start)
        } else {
            None
        }
    }

    /// The longest suffix of `w` among `suffixes`; returns `(index into suffixes, start)`.
    fn longest_suffix(w: &[char], suffixes: &[&str]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, suffix) in suffixes.iter().enumerate() {
            if let Some(start) = suffix_at(w, suffix)
                && best.is_none_or(|(_, s)| start < s)
            {
                best = Some((i, start));
            }
        }
        best
    }

    /// Replace the tail of `w` from `start` with `replacement`.
    fn replace_from(w: &mut Vec<char>, start: usize, replacement: &str) {
        w.truncate(start);
        w.extend(replacement.chars());
    }

    /// Compute R1 and R2: R1 begins after the first non-vowel that follows a vowel (with the special
    /// `gener`/`commun`/`arsen` prefixes), R2 the same rule applied within R1.
    fn mark_regions(w: &[char]) -> (usize, usize) {
        let region_after = |from: usize| -> usize {
            let mut i = from;
            while i < w.len() && !is_vowel(w, i) {
                i += 1;
            }
            while i < w.len() && is_vowel(w, i) {
                i += 1;
            }
            // `i` sits on the first non-vowel after the vowel run; the region starts after it.
            if i < w.len() { i + 1 } else { w.len() }
        };
        let r1 = ["gener", "commun", "arsen"]
            .iter()
            .find_map(|prefix| {
                let n = prefix.chars().count();
                (w.len() >= n && w[..n].iter().copied().eq(prefix.chars())).then_some(n)
            })
            .unwrap_or_else(|| region_after(0));
        let r2 = region_after(r1);
        (r1, r2)
    }

    /// Whether the word ends in a short syllable: a vowel followed by a non-vowel other than `w`/`x`/`Y`
    /// and preceded by a non-vowel — or, for a two-letter word, a vowel followed by a non-vowel.
    fn ends_short_syllable(w: &[char]) -> bool {
        let n = w.len();
        if n == 2 {
            return is_vowel(w, 0) && !is_vowel(w, 1);
        }
        n >= 3
            && !is_vowel(w, n - 3)
            && is_vowel(w, n - 2)
            && !is_vowel(w, n - 1)
            && !matches!(w[n - 1], 'w' | 'x' | 'Y')
    }

    /// Step 1a: plural-ish `s` suffixes.
    fn step_1a(w: &mut Vec<char>) {
        if let Some(start) = suffix_at(w, "sses") {
            replace_from(w, start, "ss");
        } else if let Some(start) = suffix_at(w, "ied").or_else(|| suffix_at(w, "ies")) {
            // More than one letter before the suffix -> `i`, else `ie` (`ties` -> `tie`, `cries` -> `cri`).
            let replacement = if start > 1 { "i" } else { "ie" };
            replace_from(w, start, replacement);
        } else if suffix_at(w, "us").is_some() || suffix_at(w, "ss").is_some() {
            // Leave as-is.
        } else if let Some(start) = suffix_at(w, "s") {
            // Delete if a vowel exists before the letter immediately preceding the `s`.
            if (0..start.saturating_sub(1)).any(|i| is_vowel(w, i)) {
                w.truncate(start);
            }
        }
    }

    /// Step 1b: `eed`/`ed`/`ing` families.
    fn step_1b(w: &mut Vec<char>, r1: usize) {
        const SUFFIXES: &[&str] = &["ingly", "eedly", "edly", "eed", "ing", "ed"];
        let Some((idx, start)) = longest_suffix(w, SUFFIXES) else {
            return;
        };
        if SUFFIXES[idx] == "eed" || SUFFIXES[idx] == "eedly" {
            if start >= r1 {
                replace_from(w, start, "ee");
            }
            return;
        }
        // ed / edly / ing / ingly: delete when the preceding part contains a vowel, then repair.
        if !(0..start).any(|i| is_vowel(w, i)) {
            return;
        }
        w.truncate(start);
        if ["at", "bl", "iz"].iter().any(|s| suffix_at(w, s).is_some()) {
            w.push('e');
        } else if DOUBLES.iter().any(|d| suffix_at(w, d).is_some()) {
            w.pop();
        } else if r1 >= w.len() && ends_short_syllable(w) {
            w.push('e');
        }
    }

    /// Step 1c: final `y`/`Y` -> `i` after a non-vowel that is not the first letter.
    fn step_1c(w: &mut [char]) {
        let n = w.len();
        if n >= 3
            && matches!(w[n - 1], 'y' | 'Y')
            && !is_vowel(w, n - 2)
            && let Some(last) = w.last_mut()
        {
            *last = 'i';
        }
    }

    /// Step 2 suffix table: `(suffix, replacement)`; the special `ogi` and `li` rows are handled apart.
    const STEP2: &[(&str, &str)] = &[
        ("ization", "ize"),
        ("ational", "ate"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("tional", "tion"),
        ("biliti", "ble"),
        ("lessli", "less"),
        ("entli", "ent"),
        ("ation", "ate"),
        ("alism", "al"),
        ("aliti", "al"),
        ("ousli", "ous"),
        ("iviti", "ive"),
        ("fulli", "ful"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("abli", "able"),
        ("izer", "ize"),
        ("ator", "ate"),
        ("alli", "al"),
        ("bli", "ble"),
        ("ogi", "og"),
        ("li", ""),
    ];

    /// Step 2: derivational suffixes, longest match first, in R1.
    fn step_2(w: &mut Vec<char>, r1: usize) {
        let suffixes: Vec<&str> = STEP2.iter().map(|(s, _)| *s).collect();
        let Some((idx, start)) = longest_suffix(w, &suffixes) else {
            return;
        };
        if start < r1 {
            return; // longest match wins; a failed condition does not fall back to a shorter suffix
        }
        let (suffix, replacement) = STEP2[idx];
        match suffix {
            // `ogi` -> `og` only when preceded by `l`; `li` deleted only after a valid li-ending.
            "ogi" => {
                if start > 0 && w.get(start - 1) == Some(&'l') {
                    replace_from(w, start, "og");
                }
            },
            "li" => {
                if start > 0
                    && matches!(
                        w.get(start - 1),
                        Some('c' | 'd' | 'e' | 'g' | 'h' | 'k' | 'm' | 'n' | 'r' | 't')
                    )
                {
                    w.truncate(start);
                }
            },
            _ => replace_from(w, start, replacement),
        }
    }

    /// Step 3: more derivational suffixes, longest match first, in R1 (`ative` needs R2).
    fn step_3(w: &mut Vec<char>, r1: usize, r2: usize) {
        const STEP3: &[(&str, &str)] = &[
            ("ational", "ate"),
            ("tional", "tion"),
            ("alize", "al"),
            ("icate", "ic"),
            ("iciti", "ic"),
            ("ative", ""),
            ("ical", "ic"),
            ("ness", ""),
            ("ful", ""),
        ];
        let suffixes: Vec<&str> = STEP3.iter().map(|(s, _)| *s).collect();
        let Some((idx, start)) = longest_suffix(w, &suffixes) else {
            return;
        };
        if start < r1 {
            return;
        }
        let (suffix, replacement) = STEP3[idx];
        if suffix == "ative" {
            if start >= r2 {
                w.truncate(start);
            }
            return;
        }
        replace_from(w, start, replacement);
    }

    /// Step 4: residual suffixes, longest match first, in R2 (`ion` needs a preceding `s`/`t`).
    fn step_4(w: &mut Vec<char>, r2: usize) {
        const STEP4: &[&str] = &[
            "ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ism", "ate", "iti",
            "ous", "ive", "ize", "ion", "al", "er", "ic",
        ];
        let Some((idx, start)) = longest_suffix(w, STEP4) else {
            return;
        };
        if start < r2 {
            return;
        }
        if STEP4[idx] == "ion" {
            if start > 0 && matches!(w.get(start - 1), Some('s' | 't')) {
                w.truncate(start);
            }
            return;
        }
        w.truncate(start);
    }

    /// Step 5: final `e`/`l` cleanup.
    fn step_5(w: &mut Vec<char>, r1: usize, r2: usize) {
        let n = w.len();
        if n == 0 {
            return;
        }
        if w[n - 1] == 'e' {
            // Delete a final `e` in R2, or in R1 when not preceded by a short syllable.
            if n > r2 || (n > r1 && !ends_short_syllable(&w[..n - 1])) {
                w.pop();
            }
        } else if w[n - 1] == 'l' && n > r2 && n >= 2 && w[n - 2] == 'l' {
            w.pop();
        }
    }

    /// The word-level exception list applied before the algorithm proper.
    fn exception1(word: &str) -> Option<&str> {
        let mapped = match word {
            "skis" => "ski",
            "skies" => "sky",
            "dying" => "die",
            "lying" => "lie",
            "tying" => "tie",
            "idly" => "idl",
            "gently" => "gentl",
            "ugly" => "ugli",
            "early" => "earli",
            "only" => "onli",
            "singly" => "singl",
            // Invariant forms the algorithm must not touch.
            "sky" | "news" | "howe" | "atlas" | "cosmos" | "bias" | "andes" => word,
            _ => return None,
        };
        Some(mapped)
    }

    /// The invariant forms recognized after step 1a; the algorithm stops on them.
    fn is_exception2(w: &[char]) -> bool {
        let word: String = w.iter().collect();
        matches!(
            word.as_str(),
            "inning"
                | "outing"
                | "canning"
                | "herring"
                | "earring"
                | "proceed"
                | "exceed"
                | "succeed"
        )
    }

    /// Stem one lowercased word with the Snowball English (Porter2) algorithm.
    pub(super) fn stem(word: &str) -> String {
        if word.chars().count() <= 2 {
            return word.to_owned();
        }
        if let Some(exception) = exception1(word) {
            return exception.to_owned();
        }
        let mut w: Vec<char> = word.chars().collect();
        // Prelude: strip one leading apostrophe; mark consonant `y` as `Y` (word-initial or post-vowel).
        if w.first() == Some(&'\'') {
            w.remove(0);
        }
        if w.first() == Some(&'y') {
            w[0] = 'Y';
        }
        for i in 1..w.len() {
            if w[i] == 'y' && is_vowel(&w, i - 1) {
                w[i] = 'Y';
            }
        }
        let (r1, r2) = mark_regions(&w);
        // Step 0: possessive apostrophe suffixes (longest of `'s'`, `'s`, `'`).
        for suffix in ["'s'", "'s", "'"] {
            if let Some(start) = suffix_at(&w, suffix) {
                w.truncate(start);
                break;
            }
        }
        step_1a(&mut w);
        if is_exception2(&w) {
            return w.into_iter().collect();
        }
        step_1b(&mut w, r1);
        step_1c(&mut w);
        step_2(&mut w, r1);
        step_3(&mut w, r1, r2);
        step_4(&mut w, r2);
        step_5(&mut w, r1, r2);
        w.into_iter()
            .map(|c| if c == 'Y' { 'y' } else { c })
            .collect()
    }
}

/// Tokenize `text` under the `simple` configuration: maximal alphanumeric runs, lowercased, with
/// 1-based positions. Everything else (punctuation, whitespace, `_`) separates tokens.
fn tokenize_simple(text: &str) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    let mut position = 0u32;
    let mut current = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            current.extend(c.to_lowercase());
        } else if !current.is_empty() {
            position = position.saturating_add(1);
            out.push((std::mem::take(&mut current), position.min(MAX_POSITION)));
        }
    }
    if !current.is_empty() {
        position = position.saturating_add(1);
        out.push((current, position.min(MAX_POSITION)));
    }
    out
}

/// Quote a lexeme for the canonical output form: always single-quoted, interior `'` doubled.
fn quote_lexeme(lexeme: &str) -> String {
    let mut out = String::with_capacity(lexeme.len() + 2);
    out.push('\'');
    for c in lexeme.chars() {
        if c == '\'' {
            out.push('\'');
        }
        out.push(c);
    }
    out.push('\'');
    out
}

/// `to_tsvector(config, text)` — tokenize `text` under `config` into the reference engine's canonical `tsvector` form.
///
/// Canonical: lexemes sorted (byte order), each quoted, positions ascending and deduplicated. Under
/// `english`, stopwords are dropped but still consume positions (`a fat cat` -> `'cat':3 'fat':2`),
/// and the remaining lexemes are stemmed — both like the reference engine.
///
/// # Errors
/// [`Error::Unsupported`] for an unimplemented configuration.
pub fn to_tsvector(config: &str, text: &str) -> Result<String, Error> {
    let config = check_config(config)?;
    let mut by_lexeme: std::collections::BTreeMap<String, Vec<u32>> =
        std::collections::BTreeMap::new();
    for (token, position) in tokenize_simple(text) {
        let Some(lexeme) = normalize_lexeme(config, &token) else {
            continue; // a stopword: dropped, but its position stays consumed
        };
        let positions = by_lexeme.entry(lexeme).or_default();
        // Positions arrive ascending; dedup (a clamped tail repeats 16383) and cap the count.
        if positions.len() < MAX_POSITIONS_PER_LEXEME && positions.last() != Some(&position) {
            positions.push(position);
        }
    }
    let mut out = String::new();
    for (lexeme, positions) in &by_lexeme {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&quote_lexeme(lexeme));
        out.push(':');
        for (i, position) in positions.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&position.to_string());
        }
    }
    Ok(out)
}

/// A parsed `tsquery`: the boolean structure over lexemes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TsQuery {
    Lexeme(String),
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

impl TsQuery {
    /// Binding strength for canonical printing: `|` loosest, then `&`, then `!`/atoms.
    const fn precedence(&self) -> u8 {
        match self {
            Self::Or(..) => 1,
            Self::And(..) => 2,
            Self::Not(_) | Self::Lexeme(_) => 3,
        }
    }

    /// Render in the reference engine's canonical form: children parenthesized (with interior spaces, `( 'a' | 'b' )`)
    /// only where precedence requires.
    fn render(&self, out: &mut String) {
        match self {
            Self::Lexeme(lexeme) => out.push_str(&quote_lexeme(lexeme)),
            Self::Not(inner) => {
                out.push('!');
                if inner.precedence() < 3 {
                    out.push_str("( ");
                    inner.render(out);
                    out.push_str(" )");
                } else {
                    inner.render(out);
                }
            },
            Self::And(left, right) | Self::Or(left, right) => {
                let (my_prec, op) = if matches!(self, Self::And(..)) {
                    (2, " & ")
                } else {
                    (1, " | ")
                };
                for (i, side) in [left, right].into_iter().enumerate() {
                    if i > 0 {
                        out.push_str(op);
                    }
                    if side.precedence() < my_prec {
                        out.push_str("( ");
                        side.render(out);
                        out.push_str(" )");
                    } else {
                        side.render(out);
                    }
                }
            },
        }
    }

    /// Whether the query matches a set membership test over `lexemes`.
    fn matches(&self, lexemes: &std::collections::HashSet<String>) -> bool {
        match self {
            Self::Lexeme(lexeme) => lexemes.contains(lexeme),
            Self::Not(inner) => !inner.matches(lexemes),
            Self::And(left, right) => left.matches(lexemes) && right.matches(lexemes),
            Self::Or(left, right) => left.matches(lexemes) || right.matches(lexemes),
        }
    }

    /// Normalize every lexeme under `config`, eliding dropped stopwords the way the reference engine does: an elided
    /// operand of `&`/`|` collapses the node to its other side, an elided `!` argument drops the
    /// negation, and a fully-elided query becomes `None` (the empty query).
    fn normalize(self, config: Config) -> Option<Self> {
        match self {
            Self::Lexeme(lexeme) => normalize_lexeme(config, &lexeme).map(Self::Lexeme),
            Self::Not(inner) => inner.normalize(config).map(|q| Self::Not(Box::new(q))),
            Self::And(left, right) => match (left.normalize(config), right.normalize(config)) {
                (Some(l), Some(r)) => Some(Self::And(Box::new(l), Box::new(r))),
                (one, other) => one.or(other),
            },
            Self::Or(left, right) => match (left.normalize(config), right.normalize(config)) {
                (Some(l), Some(r)) => Some(Self::Or(Box::new(l), Box::new(r))),
                (one, other) => one.or(other),
            },
        }
    }
}

/// A token of the `tsquery` input grammar.
enum QueryToken {
    Lexeme(String),
    And,
    Or,
    Not,
    Open,
    Close,
}

/// Lex a `tsquery` input: `&`/`|`/`!`/parens, `'...'` quoted lexemes (with `''` escapes), and bare
/// words (alphanumeric runs). `<->`/`<N>` phrase operators and `:...` weight suffixes are rejected
/// loudly (follow-ups).
fn lex_tsquery(input: &str) -> Result<Vec<QueryToken>, Error> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            c if c.is_whitespace() => {
                chars.next();
            },
            '&' => {
                chars.next();
                tokens.push(QueryToken::And);
            },
            '|' => {
                chars.next();
                tokens.push(QueryToken::Or);
            },
            '!' => {
                chars.next();
                tokens.push(QueryToken::Not);
            },
            '(' => {
                chars.next();
                tokens.push(QueryToken::Open);
            },
            ')' => {
                chars.next();
                tokens.push(QueryToken::Close);
            },
            '<' => {
                return Err(Error::Unsupported(
                    "tsquery phrase operator `<->` is not implemented (follow-up)".to_owned(),
                ));
            },
            ':' => {
                return Err(Error::Unsupported(
                    "tsquery weight/prefix suffix `:` is not implemented (follow-up)".to_owned(),
                ));
            },
            '\'' => {
                chars.next();
                let mut lexeme = String::new();
                loop {
                    match chars.next() {
                        Some('\'') => {
                            if chars.peek() == Some(&'\'') {
                                chars.next();
                                lexeme.push('\'');
                            } else {
                                break;
                            }
                        },
                        Some(c) => lexeme.push(c),
                        None => return Err(syntax_error(input)),
                    }
                }
                if lexeme.is_empty() {
                    return Err(syntax_error(input));
                }
                tokens.push(QueryToken::Lexeme(lexeme.to_lowercase()));
            },
            c if c.is_alphanumeric() => {
                let mut lexeme = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() {
                        lexeme.extend(c.to_lowercase());
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(QueryToken::Lexeme(lexeme));
            },
            _ => return Err(syntax_error(input)),
        }
    }
    Ok(tokens)
}

/// Recursive-descent parser over the lexed tokens, the reference engine's precedence: `|` < `&` < `!`.
struct QueryParser<'a> {
    tokens: &'a [QueryToken],
    pos: usize,
    input: &'a str,
}

impl QueryParser<'_> {
    fn or_expr(&mut self) -> Result<TsQuery, Error> {
        let mut left = self.and_expr()?;
        while matches!(self.tokens.get(self.pos), Some(QueryToken::Or)) {
            self.pos += 1;
            let right = self.and_expr()?;
            left = TsQuery::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> Result<TsQuery, Error> {
        let mut left = self.unary()?;
        while matches!(self.tokens.get(self.pos), Some(QueryToken::And)) {
            self.pos += 1;
            let right = self.unary()?;
            left = TsQuery::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<TsQuery, Error> {
        match self.tokens.get(self.pos) {
            Some(QueryToken::Not) => {
                self.pos += 1;
                Ok(TsQuery::Not(Box::new(self.unary()?)))
            },
            Some(QueryToken::Open) => {
                self.pos += 1;
                let inner = self.or_expr()?;
                if !matches!(self.tokens.get(self.pos), Some(QueryToken::Close)) {
                    return Err(syntax_error(self.input));
                }
                self.pos += 1;
                Ok(inner)
            },
            Some(QueryToken::Lexeme(lexeme)) => {
                let lexeme = lexeme.clone();
                self.pos += 1;
                Ok(TsQuery::Lexeme(lexeme))
            },
            _ => Err(syntax_error(self.input)),
        }
    }
}

/// Parse a `tsquery` from its input (or canonical) text form.
fn parse_tsquery(input: &str) -> Result<TsQuery, Error> {
    let tokens = lex_tsquery(input)?;
    if tokens.is_empty() {
        return Err(syntax_error(input));
    }
    let mut parser = QueryParser {
        tokens: &tokens,
        pos: 0,
        input,
    };
    let query = parser.or_expr()?;
    if parser.pos != tokens.len() {
        return Err(syntax_error(input));
    }
    Ok(query)
}

/// `to_tsquery(config, text)` — parse a boolean lexeme query into the reference engine's canonical text form.
///
/// The grammar is `&`/`|`/`!`/parens over quoted or bare lexemes. Under `english` each lexeme is
/// stemmed and stopwords are elided from the boolean structure (`'the & fat'` -> `'fat'`); a query
/// left empty by elision yields the empty query, which matches nothing — like the reference engine.
///
/// # Errors
/// [`Error::Unsupported`] for an unimplemented configuration or a phrase/weight operator;
/// a `42601`-coded [`Error::Coded`] for a malformed query (like the reference engine's `syntax error in tsquery`).
pub fn to_tsquery(config: &str, text: &str) -> Result<String, Error> {
    let config = check_config(config)?;
    let Some(query) = parse_tsquery(text)?.normalize(config) else {
        return Ok(String::new());
    };
    let mut out = String::new();
    query.render(&mut out);
    Ok(out)
}

/// `plainto_tsquery(config, text)` — tokenize plain text and AND the lexemes together.
///
/// Operators in the input are plain text, not syntax (`a & b` means the three lexemes `a`, `&`
/// dropped, `b`). Under `english` the lexemes are stemmed and stopwords dropped. Empty input (or
/// all-stopword input) yields the empty query, which matches nothing — like the reference engine.
///
/// # Errors
/// [`Error::Unsupported`] for an unimplemented configuration.
pub fn plainto_tsquery(config: &str, text: &str) -> Result<String, Error> {
    let config = check_config(config)?;
    let mut out = String::new();
    for (token, _) in tokenize_simple(text) {
        let Some(lexeme) = normalize_lexeme(config, &token) else {
            continue;
        };
        if !out.is_empty() {
            out.push_str(" & ");
        }
        out.push_str(&quote_lexeme(&lexeme));
    }
    Ok(out)
}

/// Parse a `tsvector` text form into its lexeme set (positions and any `A`-`D` weight suffixes are
/// accepted and ignored — they do not affect a boolean match). Accepts both the canonical quoted
/// form and bare lexemes (`fox:1 quick`), like the reference engine's `::tsvector` cast.
fn tsvector_lexemes(input: &str) -> Result<std::collections::HashSet<String>, Error> {
    let bad = || Error::Coded {
        message: format!("syntax error in tsvector: {input:?}"),
        sqlstate: "42601",
    };
    let mut lexemes = std::collections::HashSet::new();
    let mut chars = input.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        let mut lexeme = String::new();
        if c == '\'' {
            chars.next();
            loop {
                match chars.next() {
                    Some('\'') => {
                        if chars.peek() == Some(&'\'') {
                            chars.next();
                            lexeme.push('\'');
                        } else {
                            break;
                        }
                    },
                    Some(c) => lexeme.push(c),
                    None => return Err(bad()),
                }
            }
        } else {
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == ':' || c == '\'' {
                    break;
                }
                lexeme.push(c);
                chars.next();
            }
        }
        if lexeme.is_empty() {
            return Err(bad());
        }
        // Optional `:positions` — digits, commas, and A-D weight letters; ignored for matching.
        if chars.peek() == Some(&':') {
            chars.next();
            let mut any = false;
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() || c == ',' || matches!(c, 'A'..='D' | 'a'..='d') || c == '*'
                {
                    any = true;
                    chars.next();
                } else {
                    break;
                }
            }
            if !any {
                return Err(bad());
            }
        }
        lexemes.insert(lexeme);
    }
    Ok(lexemes)
}

/// The `@@` match: does `tsvector` (text form) satisfy `tsquery` (text form)? An empty query
/// matches nothing; an empty tsvector can still satisfy a negated query (`!'x'`), like the reference engine.
///
/// # Errors
/// A `42601`-coded [`Error::Coded`] for a malformed tsvector/tsquery, or [`Error::Unsupported`]
/// for the unimplemented phrase/weight forms.
pub fn ts_match(tsvector: &str, tsquery: &str) -> Result<bool, Error> {
    if tsquery.trim().is_empty() {
        return Ok(false);
    }
    let lexemes = tsvector_lexemes(tsvector)?;
    let query = parse_tsquery(tsquery)?;
    Ok(query.matches(&lexemes))
}

// ===== Relevance ranking: `ts_rank` / `ts_rank_cd` =====
//
// Both rankers reproduce the reference engine's published algorithms (the `ts_rank` term-frequency score and the
// `ts_rank_cd` cover-density score) from their public description — no third-party or the reference
// engine's source is reused. `ts_rank`/`ts_rank_cd` return `real` (float4), so the arithmetic runs in `f32` (with the
// same `f64` intermediates the reference engine's C uses) to keep the result bit-for-bit comparable; the caller
// renders it as the shortest float4 decimal, matching the reference engine's output.

/// The default term weight for a weight class (`D`/`C`/`B`/`A`, numbered 0..=3), in the reference engine's order
/// `{D:0.1, C:0.2, B:0.4, A:1.0}`. Used unless an explicit `{D,C,B,A}` array is given — that
/// four-argument overload is a follow-up.
const fn default_weight(class: u8) -> f32 {
    match class {
        3 => 1.0,
        2 => 0.4,
        1 => 0.2,
        _ => 0.1,
    }
}

/// Map a `tsvector` weight letter to its class number (0..=3), defaulting to `D` (0).
const fn weight_class(letter: char) -> u8 {
    match letter {
        'A' | 'a' => 3,
        'B' | 'b' => 2,
        'C' | 'c' => 1,
        _ => 0,
    }
}

/// One past the largest representable position; the stand-in distance the reference engine uses when two occurrences
/// share a position.
const MAXENTRYPOS: u32 = MAX_POSITION + 1;

/// The single position-0, `D`-weight occurrence a position-less operand contributes to a rank
/// (the reference engine's `POSNULL`, shared by the OR and AND paths).
const POSNULL: [WordPos; 1] = [WordPos { pos: 0, weight: 0 }];

/// The reference engine's word-distance weighting for the AND (proximity) score: adjacent occurrences score near 1
/// and the value decays logistically with the gap — `1 / (1.005 + 0.05·e^(d/1.5 − 2))` — bottoming
/// out at essentially nothing past 100 positions. (an earlier `1/d²`
/// stand-in diverged from the reference engine by ~4.6× at distance 9; the QA differential pinned this form —
/// adjacent 0.09910322, distance-2 0.098500855, distance-9 0.05174401.)
#[allow(
    clippy::suboptimal_flops,
    reason = "the fused multiply-add would change the double rounding versus the reference engine's separate \
              multiply and add, breaking float4 parity"
)]
fn word_distance(dist: u32) -> f32 {
    if dist > 100 {
        return 1e-30;
    }
    (1.0 / (1.005 + 0.05 * (f64::from(dist) / 1.5 - 2.0).exp())) as f32
}

/// One occurrence of a lexeme in a `tsvector`: its position (1-based, clamped to [`MAX_POSITION`])
/// and weight class.
#[derive(Clone, Copy)]
struct WordPos {
    pos: u32,
    weight: u8,
}

/// A parsed `tsvector` entry: the lexeme and its positions in stored (ascending) order. A lexeme
/// with no position list has an empty vector.
struct TsvEntry {
    lexeme: String,
    positions: Vec<WordPos>,
}

/// A lexeme lookup over a parsed `tsvector`: lexeme -> its positions.
type DocMap<'a> = std::collections::HashMap<&'a str, &'a [WordPos]>;

/// Parse a `tsvector` text form into its entries, keeping positions and weights (unlike
/// [`tsvector_lexemes`], which only needs the lexeme set). Accepts the canonical quoted form and
/// bare lexemes; a `:positions` list is `<number><weight?>` items separated by commas.
fn parse_tsvector_entries(input: &str) -> Result<Vec<TsvEntry>, Error> {
    let bad = || Error::Coded {
        message: format!("syntax error in tsvector: {input:?}"),
        sqlstate: "42601",
    };
    let mut entries = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        let mut lexeme = String::new();
        if c == '\'' {
            chars.next();
            loop {
                match chars.next() {
                    Some('\'') => {
                        if chars.peek() == Some(&'\'') {
                            chars.next();
                            lexeme.push('\'');
                        } else {
                            break;
                        }
                    },
                    Some(ch) => lexeme.push(ch),
                    None => return Err(bad()),
                }
            }
        } else {
            while let Some(&ch) = chars.peek() {
                if ch.is_whitespace() || ch == ':' || ch == '\'' {
                    break;
                }
                lexeme.push(ch);
                chars.next();
            }
        }
        if lexeme.is_empty() {
            return Err(bad());
        }
        let mut positions = Vec::new();
        if chars.peek() == Some(&':') {
            chars.next();
            let mut any = false;
            loop {
                let mut num: u32 = 0;
                let mut have_digit = false;
                while let Some(d) = chars.peek().and_then(|c| c.to_digit(10)) {
                    have_digit = true;
                    num = num.saturating_mul(10).saturating_add(d);
                    chars.next();
                }
                let wclass = match chars.peek().copied() {
                    Some(ch @ ('A'..='D' | 'a'..='d')) => {
                        chars.next();
                        weight_class(ch)
                    },
                    _ => 0,
                };
                if chars.peek() == Some(&'*') {
                    chars.next();
                }
                if have_digit {
                    any = true;
                    positions.push(WordPos {
                        pos: num.min(MAX_POSITION),
                        weight: wclass,
                    });
                }
                if chars.peek() == Some(&',') {
                    chars.next();
                } else {
                    break;
                }
            }
            if !any {
                return Err(bad());
            }
        }
        entries.push(TsvEntry { lexeme, positions });
    }
    Ok(entries)
}

/// The count of words in a `tsvector`: the sum of each entry's position count, treating a
/// position-less entry as one word (the reference engine's `cnt_length`).
fn cnt_length(entries: &[TsvEntry]) -> i32 {
    entries
        .iter()
        .map(|e| i32::try_from(e.positions.len()).unwrap_or(i32::MAX).max(1))
        .sum()
}

/// Build the lexeme lookup for the entries.
fn build_map(entries: &[TsvEntry]) -> DocMap<'_> {
    entries
        .iter()
        .map(|e| (e.lexeme.as_str(), e.positions.as_slice()))
        .collect()
}

/// Collect the query's operand lexemes (deduplicated). Operands under `!` are included — the reference engine's rank
/// counts every operand regardless of negation.
fn collect_operands(q: &TsQuery, out: &mut std::collections::BTreeSet<String>) {
    match q {
        TsQuery::Lexeme(l) => {
            out.insert(l.clone());
        },
        TsQuery::Not(inner) => collect_operands(inner, out),
        TsQuery::And(a, b) | TsQuery::Or(a, b) => {
            collect_operands(a, out);
            collect_operands(b, out);
        },
    }
}

/// The `ts_rank` score for a query with no top-level `&`/phrase: the average over operands of a
/// per-operand term-frequency contribution (dominated by the highest-weighted occurrence, with a
/// diminishing bonus for repeats).
#[allow(
    clippy::cast_precision_loss,
    reason = "positions and operand counts are small; the float4 score is defined in these widths"
)]
fn calc_rank_or(map: &DocMap<'_>, operands: &[String]) -> f32 {
    let mut res: f32 = 0.0;
    for op in operands {
        let posvec: &[WordPos] = match map.get(op.as_str()).copied() {
            None => continue,
            Some([]) => &POSNULL,
            Some(p) => p,
        };
        let mut resj: f32 = 0.0;
        let mut wjm: f32 = -1.0;
        let mut jm: usize = 0;
        for (j, wp) in posvec.iter().enumerate() {
            let w = default_weight(wp.weight);
            let idx = (j + 1) as f32;
            resj += w / (idx * idx);
            if w > wjm {
                wjm = w;
                jm = j;
            }
        }
        let jmd = (jm + 1) as f32;
        let numer: f32 = wjm + resj - wjm / (jmd * jmd);
        // The `/1.6449…` (limit of sum 1/i^2) and the running total go through `f64`, matching the reference engine's
        // mixed float4/double arithmetic, then round back to float4.
        let term: f64 = f64::from(numer) / 1.644_934_066_85_f64;
        res = (f64::from(res) + term) as f32;
    }
    let size = operands.len();
    if size > 0 {
        res /= size as f32;
    }
    res
}

/// The `ts_rank` score for a query whose top operator is `&`: proximity-weighted, combining every
/// cross-term occurrence pair as `sqrt(w1·w2·word_distance(dist))` via `1 - prod(1 - curw)`.
/// Returns the `-1.0` sentinel when no pair ever combined (e.g. only one operand present) — the
/// caller clamps that to the reference engine's `1e-20` floor.
#[allow(
    clippy::suboptimal_flops,
    reason = "the fused multiply-add would change the double rounding versus the reference engine's separate \
              multiply and subtract, breaking float4 parity"
)]
fn calc_rank_and(map: &DocMap<'_>, operands: &[String]) -> f32 {
    if operands.len() < 2 {
        return calc_rank_or(map, operands);
    }
    let slots: Vec<Option<(&[WordPos], bool)>> = operands
        .iter()
        .map(|op| match map.get(op.as_str()).copied() {
            None => None,
            Some([]) => Some((&POSNULL[..], true)),
            Some(p) => Some((p, false)),
        })
        .collect();
    let mut res: f32 = -1.0;
    for (i, si) in slots.iter().enumerate() {
        let Some((posi, nulli)) = *si else { continue };
        for sk in slots.iter().take(i) {
            let Some((posk, nullk)) = *sk else { continue };
            for wl in posi {
                for wp in posk {
                    let signed = i64::from(wl.pos) - i64::from(wp.pos);
                    let mut dist = signed.unsigned_abs() as u32;
                    if dist != 0 || nulli || nullk {
                        if dist == 0 {
                            dist = MAXENTRYPOS;
                        }
                        let arg: f32 = default_weight(wl.weight)
                            * default_weight(wp.weight)
                            * word_distance(dist);
                        let curw: f32 = f64::from(arg).sqrt() as f32;
                        res = if res < 0.0 {
                            curw
                        } else {
                            (1.0 - (1.0 - f64::from(res)) * (1.0 - f64::from(curw))) as f32
                        };
                    }
                }
            }
        }
    }
    res
}

/// Apply the ranking normalization bits (shared bit layout; bit 4 `EXTDIST` is `ts_rank_cd` only)
/// to a `ts_rank` score. `log`-based bits use log base 2, matching the reference engine's `ts_rank`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    reason = "word/lexeme counts are small; the float4 score is defined in these widths"
)]
fn calc_rank(entries: &[TsvEntry], query: &TsQuery, method: i32) -> f32 {
    if entries.is_empty() {
        return 0.0;
    }
    let mut ops = std::collections::BTreeSet::new();
    collect_operands(query, &mut ops);
    if ops.is_empty() {
        return 0.0;
    }
    let operands: Vec<String> = ops.into_iter().collect();
    let map = build_map(entries);
    let mut res = if matches!(query, TsQuery::And(..)) {
        calc_rank_and(&map, &operands)
    } else {
        calc_rank_or(&map, &operands)
    };
    if res < 0.0 {
        res = 1e-20_f32;
    }
    let clen = cnt_length(entries);
    let size = entries.len() as i32;
    if method & 0x01 != 0 {
        res = (f64::from(res) / (f64::from(clen + 1).ln() / std::f64::consts::LN_2)) as f32;
    }
    if method & 0x02 != 0 && clen > 0 {
        res /= clen as f32;
    }
    if method & 0x08 != 0 {
        res /= size as f32;
    }
    if method & 0x10 != 0 {
        res = (f64::from(res) / (f64::from(size + 1).ln() / std::f64::consts::LN_2)) as f32;
    }
    if method & 0x20 != 0 {
        res /= res + 1.0;
    }
    res
}

/// `ts_rank(tsvector, tsquery [, normalization])` — the term-frequency relevance score as a `real`.
///
/// An empty query (e.g. one that stemming/stopword-elision emptied) or an empty document ranks 0,
/// like the reference engine.
///
/// # Errors
/// A `42601`-coded [`Error::Coded`] for a malformed `tsvector`/`tsquery`, or [`Error::Unsupported`]
/// for an unimplemented `tsquery` phrase/weight form.
pub fn ts_rank(tsvector: &str, tsquery: &str, method: i32) -> Result<f32, Error> {
    if tsquery.trim().is_empty() {
        return Ok(0.0);
    }
    let entries = parse_tsvector_entries(tsvector)?;
    let query = parse_tsquery(tsquery)?;
    Ok(calc_rank(&entries, &query, method))
}

/// One matched occurrence in the document representation used by cover-density ranking.
struct DocRep {
    pos: u32,
    weight: u8,
    /// Index into the operand list of the query lexeme this occurrence matched.
    item: usize,
}

/// Build the position-sorted document representation: one entry per (operand, occurrence) pair.
fn get_docrep(map: &DocMap<'_>, operands: &[String]) -> Vec<DocRep> {
    let mut doc = Vec::new();
    for (idx, op) in operands.iter().enumerate() {
        if let Some(positions) = map.get(op.as_str()) {
            if positions.is_empty() {
                doc.push(DocRep {
                    pos: 0,
                    weight: 0,
                    item: idx,
                });
            } else {
                for wp in *positions {
                    doc.push(DocRep {
                        pos: wp.pos,
                        weight: wp.weight,
                        item: idx,
                    });
                }
            }
        }
    }
    doc.sort_by_key(|d| d.pos);
    doc
}

/// Whether the query is satisfied by the set of operand lexemes seen so far in a candidate cover.
/// `!` is treated as satisfied (the reference engine does not evaluate negation when scanning for covers).
fn cover_eval(q: &TsQuery, present: &std::collections::HashSet<&str>) -> bool {
    match q {
        TsQuery::Lexeme(l) => present.contains(l.as_str()),
        TsQuery::Not(_) => true,
        TsQuery::And(a, b) => cover_eval(a, present) && cover_eval(b, present),
        TsQuery::Or(a, b) => cover_eval(a, present) || cover_eval(b, present),
    }
}

/// A cover extent: `[begin, end]` index range in the doc representation with position span `[p, q]`.
struct Ext {
    pos: usize,
    p: u32,
    q: u32,
    begin: usize,
    end: usize,
}

/// Advance to the next minimal cover (a shortest run of the document that satisfies the query),
/// returning `false` when none remains. Mirrors the reference engine's `Cover`: extend upward to the first satisfying
/// position, then contract from that top downward to the tightest left edge.
fn next_cover(doc: &[DocRep], operands: &[String], query: &TsQuery, ext: &mut Ext) -> bool {
    loop {
        let mut present: std::collections::HashSet<&str> = std::collections::HashSet::new();
        ext.p = u32::MAX;
        ext.q = 0;
        let mut found = false;
        let mut lastpos = ext.pos;
        let mut i = ext.pos;
        while i < doc.len() {
            let Some(d) = doc.get(i) else { break };
            let Some(name) = operands.get(d.item) else {
                break;
            };
            present.insert(name.as_str());
            if cover_eval(query, &present) {
                ext.q = d.pos;
                ext.end = i;
                lastpos = i;
                found = true;
                break;
            }
            i += 1;
        }
        if !found {
            return false;
        }
        present.clear();
        let mut begin = ext.pos;
        let mut j = lastpos;
        loop {
            if let Some((pos, name)) = doc
                .get(j)
                .and_then(|d| operands.get(d.item).map(|n| (d.pos, n.as_str())))
            {
                present.insert(name);
                if cover_eval(query, &present) {
                    ext.p = pos;
                    ext.begin = j;
                    begin = j;
                    break;
                }
            }
            if j == ext.pos {
                break;
            }
            j -= 1;
        }
        if ext.p <= ext.q {
            ext.pos = begin + 1;
            return true;
        }
        ext.pos += 1;
    }
}

/// The `ts_rank_cd` cover-density score: sum over minimal covers of `coverlen / sum(1/weight)`
/// discounted by the noise in the cover, optionally normalized. `log`-based bits use natural log,
/// matching the reference engine's `ts_rank_cd`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    reason = "cover lengths, positions, and word counts are small; the score is defined in these widths"
)]
fn calc_rank_cd(entries: &[TsvEntry], query: &TsQuery, method: i32) -> f32 {
    if entries.is_empty() {
        return 0.0;
    }
    let mut ops = std::collections::BTreeSet::new();
    collect_operands(query, &mut ops);
    if ops.is_empty() {
        return 0.0;
    }
    let operands: Vec<String> = ops.into_iter().collect();
    let map = build_map(entries);
    let doc = get_docrep(&map, &operands);
    if doc.is_empty() {
        return 0.0;
    }
    let mut wdoc: f64 = 0.0;
    let mut sumdist: f64 = 0.0;
    let mut prev_ext: f64 = 0.0;
    let mut nextent: i32 = 0;
    let mut ext = Ext {
        pos: 0,
        p: 0,
        q: 0,
        begin: 0,
        end: 0,
    };
    while next_cover(&doc, &operands, query, &mut ext) {
        let mut invsum: f64 = 0.0;
        for k in ext.begin..=ext.end {
            if let Some(d) = doc.get(k) {
                invsum += 1.0 / f64::from(default_weight(d.weight));
            }
        }
        let coverlen = (ext.end - ext.begin + 1) as f64;
        let cpos = if invsum > 0.0 { coverlen / invsum } else { 0.0 };
        let span = i64::from(ext.q) - i64::from(ext.p);
        let idxspan = (ext.end - ext.begin) as i64;
        let mut nnoise = span - idxspan;
        if nnoise < 0 {
            nnoise = idxspan / 2;
        }
        wdoc += cpos / ((1 + nnoise) as f64);
        let cur_ext = f64::from(ext.p + ext.q) / 2.0;
        if nextent > 0 && cur_ext > prev_ext {
            sumdist += 1.0 / (cur_ext - prev_ext);
        }
        prev_ext = cur_ext;
        nextent += 1;
    }
    let clen = cnt_length(entries);
    let size = entries.len() as i32;
    if method & 0x01 != 0 && size > 0 {
        wdoc /= f64::from(clen + 1).ln();
    }
    if method & 0x02 != 0 && clen > 0 {
        wdoc /= f64::from(clen);
    }
    if method & 0x04 != 0 && nextent > 0 && sumdist > 0.0 {
        wdoc /= f64::from(nextent) / sumdist;
    }
    if method & 0x08 != 0 && size > 0 {
        wdoc /= f64::from(size);
    }
    if method & 0x10 != 0 && size > 0 {
        wdoc /= f64::from(size + 1).ln();
    }
    if method & 0x20 != 0 {
        wdoc /= wdoc + 1.0;
    }
    wdoc as f32
}

/// `ts_rank_cd(tsvector, tsquery [, normalization])` — the cover-density relevance score as a
/// `real`. An empty query or document ranks 0, like the reference engine.
///
/// # Errors
/// A `42601`-coded [`Error::Coded`] for a malformed `tsvector`/`tsquery`, or [`Error::Unsupported`]
/// for an unimplemented `tsquery` phrase/weight form.
pub fn ts_rank_cd(tsvector: &str, tsquery: &str, method: i32) -> Result<f32, Error> {
    if tsquery.trim().is_empty() {
        return Ok(0.0);
    }
    let entries = parse_tsvector_entries(tsvector)?;
    let query = parse_tsquery(tsquery)?;
    Ok(calc_rank_cd(&entries, &query, method))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_tsvector_matches_pg_simple_output() {
        // The reference engine: SELECT to_tsvector('simple', 'The quick brown fox') -> 'brown':3 'fox':4 'quick':2 'the':1
        assert_eq!(
            to_tsvector("simple", "The quick brown fox").unwrap(),
            "'brown':3 'fox':4 'quick':2 'the':1"
        );
        // Repeats collect ascending positions on one lexeme; punctuation separates.
        assert_eq!(
            to_tsvector("simple", "a b, a. b a").unwrap(),
            "'a':1,3,5 'b':2,4"
        );
        // Digits are lexemes too; case folds.
        assert_eq!(
            to_tsvector("simple", "Abc123 42").unwrap(),
            "'42':2 'abc123':1"
        );
        // Empty input -> empty tsvector.
        assert_eq!(to_tsvector("simple", " .,! ").unwrap(), "");
        // An interior quote is doubled in the canonical form.
        assert_eq!(to_tsvector("simple", "it's").unwrap(), "'it':1 's':2");
    }

    #[test]
    fn unknown_configuration_is_rejected() {
        assert!(matches!(
            to_tsvector("french", "x"),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            to_tsquery("french", "x"),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            plainto_tsquery("french", "x"),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn english_stemmer_matches_snowball() {
        // A battery of known Snowball-English (Porter2) pairs exercising every step: plurals (1a),
        // ed/ing with the at/bl/iz, double-consonant, and short-word repairs (1b), y->i (1c), the
        // derivational tables (2/3/4), final-e/l cleanup (5), both exception lists, and the
        // gener/commun region prefixes.
        for (word, stem) in [
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "tie"),
            ("cries", "cri"),
            ("gaps", "gap"),
            ("gas", "gas"),
            ("kiwis", "kiwi"),
            ("foxes", "fox"),
            ("dogs", "dog"),
            ("cats", "cat"),
            ("rats", "rat"),
            ("agreed", "agre"),
            ("feed", "feed"),
            ("jumped", "jump"),
            ("running", "run"),
            ("hopping", "hop"),
            ("hoping", "hope"),
            ("filing", "file"),
            ("sing", "sing"),
            ("singing", "sing"),
            ("skating", "skate"),
            ("sized", "size"),
            ("controlling", "control"),
            ("lazy", "lazi"),
            ("happy", "happi"),
            ("cry", "cri"),
            ("by", "by"),
            ("say", "say"),
            ("quickly", "quick"),
            ("knightly", "knight"),
            ("relational", "relat"),
            ("conditional", "condit"),
            ("national", "nation"),
            ("university", "univers"),
            ("communication", "communic"),
            ("generously", "generous"),
            ("apple", "appl"),
            ("ate", "ate"),
            // Exception list 1 and the post-1a invariants.
            ("dying", "die"),
            ("lying", "lie"),
            ("skies", "sky"),
            ("news", "news"),
            ("early", "earli"),
            ("only", "onli"),
            ("inning", "inning"),
            ("proceed", "proceed"),
            ("succeed", "succeed"),
            // Words of <= 2 letters are unchanged.
            ("ox", "ox"),
            ("a", "a"),
        ] {
            assert_eq!(porter2::stem(word), stem, "stem({word:?})");
        }
    }

    #[test]
    fn english_tsvector_drops_stopwords_but_keeps_their_positions() {
        // The canonical example from the reference engine's own documentation.
        assert_eq!(
            to_tsvector("english", "a fat  cat sat on a mat - it ate a fat rats").unwrap(),
            "'ate':9 'cat':3 'fat':2,11 'mat':7 'rat':12 'sat':4"
        );
        assert_eq!(
            to_tsvector("english", "The quick brown foxes jumped over the lazy dogs").unwrap(),
            "'brown':3 'dog':9 'fox':4 'jump':5 'lazi':8 'quick':2"
        );
        // All-stopword input -> empty tsvector; digit tokens skip the stopword/stem pipeline.
        assert_eq!(to_tsvector("english", "the of and").unwrap(), "");
        assert_eq!(
            to_tsvector("english", "the42 12 the").unwrap(),
            "'12':2 'the42':1"
        );
    }

    #[test]
    fn english_tsquery_stems_and_elides_stopwords() {
        assert_eq!(
            to_tsquery("english", "The & Fat & Rats").unwrap(),
            "'fat' & 'rat'"
        );
        // A stopword operand collapses the node to its other side; a negated stopword drops.
        assert_eq!(to_tsquery("english", "fat & the").unwrap(), "'fat'");
        assert_eq!(to_tsquery("english", "fat & !the").unwrap(), "'fat'");
        assert_eq!(to_tsquery("english", "the | fat").unwrap(), "'fat'");
        // An all-stopword query is the empty query (matches nothing).
        assert_eq!(to_tsquery("english", "the").unwrap(), "");
        assert!(!ts_match("'fat':1", &to_tsquery("english", "the").unwrap()).unwrap());
        // plainto stems and drops stopwords too.
        assert_eq!(
            plainto_tsquery("english", "The Fat Rats").unwrap(),
            "'fat' & 'rat'"
        );
        // The full english round trip matches.
        let tv = to_tsvector("english", "The quick brown foxes jumped").unwrap();
        assert!(ts_match(&tv, &to_tsquery("english", "foxes & jumping").unwrap()).unwrap());
    }

    #[test]
    fn to_tsquery_canonicalizes_like_pg() {
        // The reference engine: SELECT to_tsquery('simple', 'fox & quick') -> 'fox' & 'quick'
        assert_eq!(
            to_tsquery("simple", "fox & quick").unwrap(),
            "'fox' & 'quick'"
        );
        assert_eq!(to_tsquery("simple", "Fox").unwrap(), "'fox'");
        // Parens kept only where precedence needs them (the reference engine prints interior spaces).
        assert_eq!(
            to_tsquery("simple", "a & (b | c)").unwrap(),
            "'a' & ( 'b' | 'c' )"
        );
        assert_eq!(to_tsquery("simple", "(a | b)").unwrap(), "'a' | 'b'");
        assert_eq!(
            to_tsquery("simple", "a | b & c").unwrap(),
            "'a' | 'b' & 'c'"
        );
        assert_eq!(to_tsquery("simple", "!a & b").unwrap(), "!'a' & 'b'");
        assert_eq!(to_tsquery("simple", "!(a | b)").unwrap(), "!( 'a' | 'b' )");
        // Quoted lexemes may hold spaces/quotes.
        assert_eq!(
            to_tsquery("simple", "'it''s' | fine").unwrap(),
            "'it''s' | 'fine'"
        );
    }

    #[test]
    fn malformed_tsquery_is_a_42601_syntax_error() {
        for bad in ["", "&", "a &", "& a", "(a", "a)", "a b", "''"] {
            match to_tsquery("simple", bad) {
                Err(Error::Coded { sqlstate, .. }) => {
                    assert_eq!(sqlstate, "42601", "input {bad:?}");
                },
                other => panic!("expected 42601 for {bad:?}, got {other:?}"),
            }
        }
        // Phrase / weight forms are loud Unsupported, not silent misparse.
        assert!(matches!(
            to_tsquery("simple", "a <-> b"),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            to_tsquery("simple", "a:*"),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn plainto_tsquery_ands_the_lexemes() {
        assert_eq!(
            plainto_tsquery("simple", "The Fat Rats").unwrap(),
            "'the' & 'fat' & 'rats'"
        );
        // Operators are plain text here, not syntax; empty input -> empty query.
        assert_eq!(plainto_tsquery("simple", "a & b").unwrap(), "'a' & 'b'");
        assert_eq!(plainto_tsquery("simple", "").unwrap(), "");
    }

    #[test]
    fn ts_match_evaluates_the_boolean_structure() {
        let tv = to_tsvector("simple", "The quick brown fox").unwrap();
        assert!(ts_match(&tv, "'fox' & 'quick'").unwrap());
        assert!(!ts_match(&tv, "'fox' & 'lazy'").unwrap());
        assert!(ts_match(&tv, "'fox' | 'lazy'").unwrap());
        assert!(ts_match(&tv, "'fox' & !'lazy'").unwrap());
        assert!(!ts_match(&tv, "!'fox'").unwrap());
        assert!(ts_match(&tv, "!( 'lazy' | 'dog' )").unwrap());
        // The empty query matches nothing; an empty tsvector still satisfies a negation.
        assert!(!ts_match(&tv, "").unwrap());
        assert!(ts_match("", "!'fox'").unwrap());
        // Bare and positioned tsvector forms are accepted; weights are ignored for matching.
        assert!(ts_match("fox:1 quick", "'fox' & 'quick'").unwrap());
        assert!(ts_match("'fox':1A 'dog':2B", "'dog'").unwrap());
    }

    /// Assert two `real` scores are equal to within a float4 epsilon.
    fn approx(got: f32, want: f32) {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }

    #[test]
    fn ts_rank_scores_match_the_pg_algorithm() {
        // The canonical single-term score the reference engine documents: 0.1 / (pi^2/6).
        approx(ts_rank("'fox':1", "'fox'", 0).unwrap(), 0.060_792_71);
        // It renders as the shortest float4 decimal, exactly like the reference engine.
        assert_eq!(
            super::super::display::value_text(&real_score("'fox':1", "'fox'")),
            "0.06079271"
        );

        // AND (proximity) scores, every expectation measured on the reference engine itself (the QA differential
        // fornot hand-derived, which is how the original 1/d² error
        // slipped through): sqrt(0.1·0.1·word_distance(d)).
        approx(
            ts_rank("'a':1 'b':2", "'a' & 'b'", 0).unwrap(),
            0.099_103_22,
        );
        approx(
            ts_rank("'a':2 'b':4", "'a' & 'b'", 0).unwrap(),
            0.098_500_855,
        );
        approx(
            ts_rank("'a':1 'b':10", "'a' & 'b'", 0).unwrap(),
            0.051_744_01,
        );
        // Multiple pairs combine as 1 - prod(1 - curw): cat@1,3 dog@2 → two adjacent pairs.
        approx(
            ts_rank("'cat':1,3 'dog':2", "'cat' & 'dog'", 0).unwrap(),
            0.188_385,
        );
        // An AND with no co-occurring pair (one operand missing) floors at the reference engine's 1e-20 sentinel.
        approx(ts_rank("'cat':1", "'cat' & 'dog'", 0).unwrap(), 1e-20);

        // Repeats raise the OR score above a single occurrence.
        let once = ts_rank("'fox':1", "'fox'", 0).unwrap();
        let twice = ts_rank("'fox':1,2", "'fox'", 0).unwrap();
        assert!(twice > once, "{twice} !> {once}");

        // A non-matching document and an empty query both rank 0.
        approx(ts_rank("'cat':1", "'fox'", 0).unwrap(), 0.0);
        approx(ts_rank("'fox':1", "", 0).unwrap(), 0.0);

        // Normalization bit 32 (rank/(rank+1)) maps the score into (0, 1); the reference engine yields 0.09016734.
        let normed = ts_rank("'a':1 'b':2", "'a' & 'b'", 32).unwrap();
        approx(normed, 0.090_167_34);
    }

    #[test]
    fn ts_rank_cd_rewards_cover_density() {
        // A single adjacent cover of two default-weight terms: coverlen/sum(1/w) = 2/20 = 0.1.
        approx(ts_rank_cd("'a':1 'b':2", "'a' & 'b'", 0).unwrap(), 0.1);
        // The same pair spread out is discounted by the noise inside the cover: 0.1 / (1 + 8).
        approx(
            ts_rank_cd("'a':1 'b':10", "'a' & 'b'", 0).unwrap(),
            0.1 / 9.0,
        );
        // No cover (one term missing) ranks 0; the empty query ranks 0.
        approx(ts_rank_cd("'a':1", "'a' & 'b'", 0).unwrap(), 0.0);
        approx(ts_rank_cd("'a':1 'b':2", "", 0).unwrap(), 0.0);
    }

    /// Helper: rank `'fox'` in a document and wrap the score the way the executor renders it.
    fn real_score(tsvector: &str, tsquery: &str) -> crate::ast::Value {
        let r = ts_rank(tsvector, tsquery, 0).unwrap();
        crate::ast::Value::Float(
            r.to_string()
                .parse::<f64>()
                .unwrap_or_else(|_| f64::from(r)),
        )
    }
}
