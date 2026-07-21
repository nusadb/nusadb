//! LISTEN/NOTIFY asynchronous notification registry (async pub/sub).
//!
//! Each connection registers an [`UnboundedSender`] under its unique backend `pid` and records which
//! channels it is listening on (`LISTEN`). `NOTIFY channel[, payload]` looks up every connection
//! listening on that channel *in the same database* and pushes a [`Notification`] down its sender;
//! the connection's query loop drains the receiver and writes a `NotificationResponse` frame while it
//! is idle (between statements).
//!
//! Delivery is scoped per database (the physical engine-per-database model), like the reference engine: a session only
//! receives notifications sent from its own database. Self-notifications are delivered too (a backend
//! that `LISTEN`s and then `NOTIFY`s the same channel hears itself), also like the reference engine.
//!
//! The registry is process-global — the same shape as the cancel-key registry ([`crate::cancel`]) —
//! so the query loop reaches it without threading a handle through every `serve_*` entry point. An
//! RAII [`Registration`] removes a connection's entry (and all its subscriptions) when it ends.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use tokio::sync::mpsc::UnboundedSender;

/// A notification delivered to a listening connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notification {
    /// Backend pid of the connection that issued the `NOTIFY`.
    pub pid: u32,
    /// The channel the notification was sent on.
    pub channel: String,
    /// The payload (empty when `NOTIFY` carried none).
    pub payload: String,
}

/// One registered connection: where to push its notifications, which database it is bound to, and the
/// set of channels it currently listens on.
struct Listener {
    database: String,
    sender: UnboundedSender<Notification>,
    channels: HashSet<String>,
}

static REGISTRY: OnceLock<Mutex<HashMap<u32, Listener>>> = OnceLock::new();

/// Access the registry map, recovering from a poisoned lock (a panic elsewhere must not wedge
/// notification delivery for every other connection).
fn with_registry<R>(f: impl FnOnce(&mut HashMap<u32, Listener>) -> R) -> R {
    let mutex = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(&mut guard)
}

/// Register a fresh connection for notifications.
///
/// `pid` is bound to `database` with the `sender` its query loop drains. Returns an RAII
/// [`Registration`] that removes the connection — and all its subscriptions — when dropped (at the
/// end of the connection).
#[must_use]
pub fn register(pid: u32, database: String, sender: UnboundedSender<Notification>) -> Registration {
    with_registry(|map| {
        map.insert(
            pid,
            Listener {
                database,
                sender,
                channels: HashSet::new(),
            },
        );
    });
    Registration { pid }
}

/// Subscribe connection `pid` to `channel` (`LISTEN channel`). A repeat `LISTEN` on the same channel
/// is a no-op, like the reference engine.
pub fn listen(pid: u32, channel: String) {
    with_registry(|map| {
        if let Some(listener) = map.get_mut(&pid) {
            listener.channels.insert(channel);
        }
    });
}

/// Unsubscribe connection `pid`: from one channel (`Some`, `UNLISTEN channel`) or from all of them
/// (`None`, `UNLISTEN *`). Unlistening a channel that was never listened to is a no-op.
pub fn unlisten(pid: u32, channel: Option<&str>) {
    with_registry(|map| {
        if let Some(listener) = map.get_mut(&pid) {
            match channel {
                Some(name) => {
                    listener.channels.remove(name);
                },
                None => listener.channels.clear(),
            }
        }
    });
}

/// Deliver a notification to every listener of `channel` in `database`.
///
/// Includes the sender itself if it listens. Returns the number of connections the notification was
/// queued for; a connection whose receiver has been dropped (it is tearing down) is skipped.
pub fn notify(database: &str, channel: &str, notification: &Notification) -> usize {
    with_registry(|map| {
        let mut delivered = 0;
        for listener in map.values() {
            if listener.database == database
                && listener.channels.contains(channel)
                && listener.sender.send(notification.clone()).is_ok()
            {
                delivered += 1;
            }
        }
        delivered
    })
}

/// Removes a connection's registry entry (and all its subscriptions) on drop.
#[derive(Debug)]
pub struct Registration {
    pid: u32,
}

impl Drop for Registration {
    fn drop(&mut self) {
        with_registry(|map| map.remove(&self.pid));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(pid: u32, channel: &str, payload: &str) -> Notification {
        Notification {
            pid,
            channel: channel.to_owned(),
            payload: payload.to_owned(),
        }
    }

    #[test]
    fn notify_reaches_only_same_database_listeners_on_the_channel() {
        let (tx_a, mut rx_a) = tokio::sync::mpsc::unbounded_channel();
        let (tx_b, mut rx_b) = tokio::sync::mpsc::unbounded_channel();
        let (tx_c, mut rx_c) = tokio::sync::mpsc::unbounded_channel();
        let _ra = register(9001, "shop".to_owned(), tx_a);
        let _rb = register(9002, "shop".to_owned(), tx_b);
        let _rc = register(9003, "other".to_owned(), tx_c); // different database

        listen(9001, "orders".to_owned());
        listen(9003, "orders".to_owned()); // same channel, wrong database
        // 9002 listens on a different channel, so it must not receive "orders".
        listen(9002, "shipments".to_owned());

        let n = note(9002, "orders", "row-1");
        assert_eq!(notify("shop", "orders", &n), 1);

        assert_eq!(rx_a.try_recv(), Ok(n));
        assert!(rx_b.try_recv().is_err()); // wrong channel
        assert!(rx_c.try_recv().is_err()); // wrong database
    }

    #[test]
    fn self_notification_is_delivered() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let _r = register(9101, "db".to_owned(), tx);
        listen(9101, "ch".to_owned());
        let n = note(9101, "ch", "hi");
        assert_eq!(notify("db", "ch", &n), 1);
        assert_eq!(rx.try_recv(), Ok(n));
    }

    #[test]
    fn unlisten_stops_delivery() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let _r = register(9201, "db".to_owned(), tx);
        listen(9201, "a".to_owned());
        listen(9201, "b".to_owned());

        unlisten(9201, Some("a"));
        assert_eq!(notify("db", "a", &note(1, "a", "")), 0);
        assert_eq!(notify("db", "b", &note(1, "b", "")), 1);
        let _ = rx.try_recv();

        unlisten(9201, None); // unlisten *
        assert_eq!(notify("db", "b", &note(1, "b", "")), 0);
    }

    #[test]
    fn dropping_the_registration_removes_the_listener() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        {
            let _r = register(9301, "db".to_owned(), tx);
            listen(9301, "ch".to_owned());
            assert_eq!(notify("db", "ch", &note(1, "ch", "")), 1);
        }
        // After the registration drops, the listener is gone.
        assert_eq!(notify("db", "ch", &note(1, "ch", "")), 0);
    }
}
