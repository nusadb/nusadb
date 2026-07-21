//! Server metrics: in-process atomic counters + Prometheus text rendering.
//!
//! A [`Metrics`] is shared across connections via `Arc`. Counters use relaxed atomics — exact
//! ordering between independent counters does not matter for monitoring. The server binary scrapes
//! [`Metrics::render_prometheus`] over a small HTTP endpoint; OTLP push export is a later add-on
//! (it would pull in the `opentelemetry` stack).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

/// Server-wide counters (connections + queries). Cheap, lock-free, shared via `Arc`.
#[derive(Debug, Default)]
pub struct Metrics {
    connections_total: AtomicU64,
    connections_active: AtomicU64,
    queries_total: AtomicU64,
    query_errors_total: AtomicU64,
}

impl Metrics {
    /// A fresh zeroed metrics set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an accepted connection (bumps the total and the active gauge).
    pub(crate) fn connection_opened(&self) {
        self.connections_total.fetch_add(1, Ordering::Relaxed);
        self.connections_active.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a closed connection (decrements the active gauge).
    pub(crate) fn connection_closed(&self) {
        self.connections_active.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record an executed query; `ok` is false when the server returned an error to the client.
    pub(crate) fn query(&self, ok: bool) {
        self.queries_total.fetch_add(1, Ordering::Relaxed);
        if !ok {
            self.query_errors_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Total connections accepted since start.
    #[must_use]
    pub fn connections_total(&self) -> u64 {
        self.connections_total.load(Ordering::Relaxed)
    }

    /// Connections currently open.
    #[must_use]
    pub fn connections_active(&self) -> u64 {
        self.connections_active.load(Ordering::Relaxed)
    }

    /// Total simple queries executed since start.
    #[must_use]
    pub fn queries_total(&self) -> u64 {
        self.queries_total.load(Ordering::Relaxed)
    }

    /// Total queries that returned an error to the client.
    #[must_use]
    pub fn query_errors_total(&self) -> u64 {
        self.query_errors_total.load(Ordering::Relaxed)
    }

    /// Render every counter in the Prometheus text exposition format (v0.0.4).
    #[must_use]
    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(512);
        let mut metric = |name: &str, kind: &str, help: &str, value: u64| {
            // Each write targets a String, so the formatting cannot fail.
            let _ = writeln!(out, "# HELP {name} {help}");
            let _ = writeln!(out, "# TYPE {name} {kind}");
            let _ = writeln!(out, "{name} {value}");
        };
        metric(
            "nusadb_connections_total",
            "counter",
            "Total client connections accepted.",
            self.connections_total(),
        );
        metric(
            "nusadb_connections_active",
            "gauge",
            "Client connections currently open.",
            self.connections_active(),
        );
        metric(
            "nusadb_queries_total",
            "counter",
            "Total simple queries executed.",
            self.queries_total(),
        );
        metric(
            "nusadb_query_errors_total",
            "counter",
            "Total queries that returned an error to the client.",
            self.query_errors_total(),
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::Metrics;

    #[test]
    fn counters_track_connections_and_queries() {
        let m = Metrics::new();
        m.connection_opened();
        m.connection_opened();
        m.connection_closed();
        m.query(true);
        m.query(false);

        assert_eq!(m.connections_total(), 2);
        assert_eq!(m.connections_active(), 1);
        assert_eq!(m.queries_total(), 2);
        assert_eq!(m.query_errors_total(), 1);
    }

    #[test]
    fn prometheus_rendering_has_help_type_and_value_lines() {
        let m = Metrics::new();
        m.connection_opened();
        m.query(false);
        let text = m.render_prometheus();

        assert!(text.contains("# TYPE nusadb_connections_total counter"));
        assert!(text.contains("nusadb_connections_total 1"));
        assert!(text.contains("# TYPE nusadb_connections_active gauge"));
        assert!(text.contains("nusadb_connections_active 1"));
        assert!(text.contains("nusadb_queries_total 1"));
        assert!(text.contains("nusadb_query_errors_total 1"));
        // Well-formed exposition: every line is a comment or `name value`.
        for line in text.lines() {
            assert!(
                line.starts_with('#') || line.split(' ').count() == 2,
                "malformed exposition line: {line:?}"
            );
        }
    }
}
