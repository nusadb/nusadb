//! Notices: messages a statement sends its client besides its result, such as `DROP TABLE IF
//! EXISTS` of a table that is not there.
//!
//! A statement runs on one thread from start to finish (see [`crate::cancel`]), so the notices it
//! raises collect in a thread-local list, in the order raised, without a handle threaded through
//! every operator. Whoever runs the statement takes them when it is done: the embedded
//! [`Session`](crate::Session) keeps them for [`Session::take_notices`](crate::Session::take_notices),
//! and the wire server sends them ahead of the statement's `CommandComplete` to a client that
//! negotiated protocol minor 3. A runner clears the list before a statement, so nothing raised on
//! the thread by earlier work reaches the wrong client.

use std::cell::RefCell;

/// How serious a notice is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Something the client may want to know; the statement did what it was asked.
    Notice,
    /// Something likely to be a mistake.
    Warning,
    /// Information the client asked for.
    Info,
}

impl Severity {
    /// The severity as the protocol spells it: `NOTICE`, `WARNING` or `INFO`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Notice => "NOTICE",
            Self::Warning => "WARNING",
            Self::Info => "INFO",
        }
    }
}

/// One notice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notice {
    /// How serious it is.
    pub severity: Severity,
    /// The five-character SQLSTATE it carries (`00000` for plain information).
    pub code: String,
    /// The human-readable text.
    pub message: String,
}

/// The SQLSTATE of a notice that reports nothing wrong.
pub const SUCCESSFUL_COMPLETION: &str = "00000";

/// The most notices one statement sends, and the most an embedded session keeps between takes.
///
/// Past it further notices are dropped (a `RAISE NOTICE` in a long loop would otherwise grow
/// without bound) and one closing `WARNING` (`01000`) says how many were.
pub const MAX_NOTICES: usize = 1000;

/// Notices kept in the order raised, at most [`MAX_NOTICES`] of them, with a count of the rest.
#[derive(Debug, Default)]
pub struct NoticeBuffer {
    kept: Vec<Notice>,
    dropped: u64,
}

impl NoticeBuffer {
    /// An empty buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            kept: Vec::new(),
            dropped: 0,
        }
    }

    /// Keep `notice`, or count it as dropped once the buffer is full.
    pub fn push(&mut self, notice: Notice) {
        if self.kept.len() < MAX_NOTICES {
            self.kept.push(notice);
        } else {
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    /// Keep every notice of `notices` the buffer has room for, in order.
    pub fn extend(&mut self, notices: impl IntoIterator<Item = Notice>) {
        for notice in notices {
            self.push(notice);
        }
    }

    /// Keep what `other` kept, as far as there is room, and count everything it dropped.
    pub fn absorb(&mut self, other: Self) {
        self.extend(other.kept);
        self.dropped = self.dropped.saturating_add(other.dropped);
    }

    /// The kept notices, followed by a `WARNING` that counts the dropped ones when there were any.
    #[must_use]
    pub fn into_vec(self) -> Vec<Notice> {
        let mut out = self.kept;
        if self.dropped > 0 {
            out.push(Notice {
                severity: Severity::Warning,
                code: "01000".to_owned(),
                message: format!(
                    "{} more notices were dropped (at most {MAX_NOTICES} are kept)",
                    self.dropped
                ),
            });
        }
        out
    }
}

thread_local! {
    static PENDING: RefCell<NoticeBuffer> = const { RefCell::new(NoticeBuffer::new()) };
}

/// Raise a notice from the statement running on this thread.
pub fn raise(severity: Severity, code: &str, message: impl Into<String>) {
    let notice = Notice {
        severity,
        code: code.to_owned(),
        message: message.into(),
    };
    PENDING.with(|pending| pending.borrow_mut().push(notice));
}

/// Raise a `NOTICE` with SQLSTATE `00000`.
pub fn notice(message: impl Into<String>) {
    raise(Severity::Notice, SUCCESSFUL_COMPLETION, message);
}

/// The notice an `IF EXISTS` raises for a missing object: `NOTICE 00000`
/// `<what> does not exist, skipping`, where `what` names it (`table "t"`, `column "c" of relation
/// "t"`).
pub fn skipped_missing(what: &str) {
    notice(format!("{what} does not exist, skipping"));
}

/// The notice an `IF NOT EXISTS` raises for an object already there.
///
/// It reads `<what> already exists, skipping` and carries the code the error would have carried
/// (`42P07` for a relation, `42P06` for a schema, `42701` for a column, `42710` otherwise).
pub fn skipped_existing(code: &str, what: &str) {
    raise(
        Severity::Notice,
        code,
        format!("{what} already exists, skipping"),
    );
}

/// Take every notice raised on this thread since the last take, in the order raised (at most
/// [`MAX_NOTICES`], then the warning that counts the rest).
#[must_use]
pub fn take() -> Vec<Notice> {
    PENDING.with(|pending| std::mem::take(&mut *pending.borrow_mut()).into_vec())
}

/// [`take`] as a buffer, so a holder that keeps notices across statements can
/// [`absorb`](NoticeBuffer::absorb) it without counting a summary warning as a notice.
#[must_use]
pub fn take_buffer() -> NoticeBuffer {
    PENDING.with(|pending| std::mem::take(&mut *pending.borrow_mut()))
}

/// Drop every notice raised on this thread since the last take.
pub fn clear() {
    PENDING.with(|pending| *pending.borrow_mut() = NoticeBuffer::new());
}

/// Put `notices` back in front of whatever has been raised since they were taken: a nested runner
/// that took the outer statement's notices to isolate its own returns them this way.
pub fn restore(mut notices: NoticeBuffer) {
    PENDING.with(|pending| {
        let mut pending = pending.borrow_mut();
        notices.absorb(std::mem::take(&mut *pending));
        *pending = notices;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notices_are_taken_in_order_and_once() {
        clear();
        notice("one");
        raise(Severity::Warning, "01000", "two");
        let taken = take();
        assert_eq!(
            taken
                .iter()
                .map(|n| (n.severity.as_str(), n.code.as_str(), n.message.as_str()))
                .collect::<Vec<_>>(),
            vec![("NOTICE", "00000", "one"), ("WARNING", "01000", "two")]
        );
        assert!(take().is_empty());
    }

    #[test]
    fn a_flood_keeps_the_first_notices_and_counts_the_rest() {
        clear();
        for i in 0..MAX_NOTICES + 25 {
            notice(format!("n{i}"));
        }
        let taken = take();
        assert_eq!(taken.len(), MAX_NOTICES + 1);
        assert_eq!(taken.first().map(|n| n.message.as_str()), Some("n0"));
        let last = taken.last().unwrap();
        assert_eq!(
            (last.severity, last.code.as_str()),
            (Severity::Warning, "01000")
        );
        assert!(
            last.message.starts_with("25 more notices"),
            "{}",
            last.message
        );
        assert!(take().is_empty());
    }

    #[test]
    fn restore_puts_earlier_notices_first() {
        clear();
        notice("outer");
        let outer = take_buffer();
        notice("inner");
        restore(outer);
        let all: Vec<String> = take().into_iter().map(|n| n.message).collect();
        assert_eq!(all, vec!["outer".to_owned(), "inner".to_owned()]);
    }
}
