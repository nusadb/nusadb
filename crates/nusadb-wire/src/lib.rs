//! L2 — Nusa Wire Protocol.
//!
//! Async (tokio) TCP server. Framing: `[type:u8][len:u32][payload]`. TLS via rustls
//! (TLS 1.3 only by default). Auth: SCRAM-SHA-256 (RFC 7677) or mTLS. Extended query
//! protocol (Parse / Bind / Execute / Sync) for prepared statements.
//!
//! Async lives **only** in this crate. Calls into [`nusadb-sql`](nusadb_sql) are
//! synchronous, dispatched via `tokio::task::spawn_blocking`.

#![warn(missing_docs)]

pub mod auth;
pub mod cancel;
pub mod cluster;
pub mod error;
pub mod frame;
pub mod messages;
pub mod metrics;
pub mod notify;
pub mod server;
pub mod tls;
pub mod value;

pub use auth::AuthStore;
pub use cluster::{ClusterError, DatabaseCluster, SingleDatabase, is_valid_database_name};
pub use error::WireError;
pub use frame::Frame;
pub use messages::{BackendMessage, DescribeTarget, FrontendMessage, TxnStatus};
pub use metrics::Metrics;
pub use server::{
    Connection, ServerConfig, handle_client, handle_client_with, serve,
    serve_cluster_with_shutdown, serve_with_shutdown,
};
pub use value::encode_binary;

/// Maximum frame length accepted by the server. 256 MiB hard cap to prevent
/// memory-exhaustion DoS via crafted length-prefix.
pub const MAX_FRAME_LEN: u32 = 256 * 1024 * 1024;

/// Protocol magic — ASCII `"NUSA"` (`0x4E55_5341`). The first four bytes of every Startup
/// payload; identifies the original Nusa Wire Protocol (NusaDB's own wire format).
pub const PROTOCOL_MAGIC: u32 = 0x4E55_5341;

/// Current protocol version `(major, minor)`.
///
/// A major bump is an incompatible change; the server rejects a Startup whose major version it does
/// not support. A minor bump is additive: an old client requesting `minor = 0` keeps the exact 1.0
/// byte behaviour. **1.1** adds the typed
/// [`RowDescriptionTyped`](messages::BackendMessage::RowDescriptionTyped) message, sent only when
/// the connection negotiated `minor >= 1`. **1.2** carries the *element* type of an `ARRAY` column in
/// its type tag (`0x80 | element_tag`) so a client decodes the elements at their real type rather than
/// as text; a `minor < 2` connection keeps the plain `ARRAY` tag (`0x0F`).
pub const PROTOCOL_VERSION: (u16, u16) = (1, 2);

/// The high bit marking an element-typed `ARRAY` tag (protocol 1.2).
///
/// Set on an `ARRAY` column's type tag so the low 7 bits carry the element's own
/// [`column_type_tag`] — `INT[]` is `0x80 | 0x02 = 0x82`. Set only on a connection that negotiated
/// `minor >= 2`; otherwise an array column keeps the plain `0x0F` tag.
pub const ARRAY_ELEMENT_TAG_FLAG: u8 = 0x80;

/// Per-column type tag for [`RowDescriptionTyped`](messages::BackendMessage::RowDescriptionTyped)
/// (protocol 1.1).
///
/// The §11.2 type taxonomy as a single byte, mirroring [`nusadb_core::ColumnType`]. `0x00` is
/// reserved for an unresolved type (clients treat it as `TEXT`). `ARRAY`/`VECTOR` collapse to one
/// tag — their element type / dimension are not carried in the tag (the values stay canonical text,
/// per §11.2).
#[must_use]
pub const fn column_type_tag(ty: nusadb_core::ColumnType) -> u8 {
    use nusadb_core::ColumnType;
    match ty {
        ColumnType::Bool => 0x01,
        // SMALLINT/BIGINT are INT on the wire — the declared width is not carried in the tag.
        ColumnType::Int | ColumnType::SmallInt | ColumnType::BigInt => 0x02,
        // REAL is FLOAT on the wire (the single/double distinction is not carried in the tag).
        ColumnType::Float | ColumnType::Real => 0x03,
        ColumnType::Numeric { .. } => 0x04,
        // VARCHAR/CHAR are TEXT on the wire — the declared length is not carried in the tag.
        ColumnType::Text | ColumnType::VarChar(_) | ColumnType::Char(_) => 0x05,
        ColumnType::Bytes => 0x06,
        ColumnType::Date => 0x07,
        ColumnType::Time => 0x08,
        ColumnType::TimeTz => 0x09,
        ColumnType::Timestamp => 0x0A,
        ColumnType::TimestampTz => 0x0B,
        ColumnType::Interval => 0x0C,
        ColumnType::Uuid => 0x0D,
        // JSONB is JSON on the wire (both are canonical text; the tag does not distinguish them).
        ColumnType::Json | ColumnType::Jsonb => 0x0E,
        ColumnType::Array(_) => 0x0F,
        ColumnType::Vector(_) => 0x10,
    }
}

/// The per-column type tag for a `minor >= 2` (protocol 1.2) connection.
///
/// Identical to [`column_type_tag`] except an `ARRAY` carries its element type in the tag —
/// `ARRAY_ELEMENT_TAG_FLAG | <element tag>` (e.g. `INT[]` → `0x82`) — so the client decodes the
/// elements at their real type instead of as text. A `minor < 2` connection must use
/// [`column_type_tag`] (plain `0x0F`) for backward compatibility.
#[must_use]
pub const fn column_type_tag_v2(ty: nusadb_core::ColumnType) -> u8 {
    match ty {
        nusadb_core::ColumnType::Array(elem) => {
            ARRAY_ELEMENT_TAG_FLAG | column_type_tag(elem.column_type())
        },
        other => column_type_tag(other),
    }
}

/// The element type tag of an array column tag, or `None` if `tag` is not an element-typed array tag
/// (protocol 1.2). `0x82` (`INT[]`) → `Some(0x02)`; a plain scalar or `0x0F` array tag → `None`.
#[must_use]
pub const fn array_element_tag(tag: u8) -> Option<u8> {
    if tag & ARRAY_ELEMENT_TAG_FLAG != 0 {
        Some(tag & !ARRAY_ELEMENT_TAG_FLAG)
    } else {
        None
    }
}

/// The canonical type name for a
/// [`RowDescriptionTyped`](messages::BackendMessage::RowDescriptionTyped) `type_tag` byte.
///
/// The inverse of [`column_type_tag`] (wire-protocol.md §9.2). An unknown or `0x00` tag maps to
/// `"UNKNOWN"` (clients treat it as text). Lets a Rust client surface a column's type without
/// re-encoding the taxonomy.
#[must_use]
pub const fn column_type_name(tag: u8) -> &'static str {
    // An element-typed array tag (protocol 1.2, `0x80 | element`) still names as `ARRAY`; the element
    // type is recovered separately with `array_element_tag`.
    if array_element_tag(tag).is_some() {
        return "ARRAY";
    }
    match tag {
        0x01 => "BOOL",
        0x02 => "INT",
        0x03 => "FLOAT",
        0x04 => "NUMERIC",
        0x05 => "TEXT",
        0x06 => "BYTES",
        0x07 => "DATE",
        0x08 => "TIME",
        0x09 => "TIMETZ",
        0x0A => "TIMESTAMP",
        0x0B => "TIMESTAMPTZ",
        0x0C => "INTERVAL",
        0x0D => "UUID",
        0x0E => "JSON",
        0x0F => "ARRAY",
        0x10 => "VECTOR",
        _ => "UNKNOWN",
    }
}

#[cfg(test)]
mod tag_tests {
    use super::*;
    use nusadb_core::{ArrayElem, ColumnType};

    #[test]
    fn array_tag_carries_element_type_in_v2_only() {
        // 1.1 collapses every array to the plain ARRAY tag; 1.2 carries the element type.
        let int_array = ColumnType::Array(ArrayElem::Int);
        assert_eq!(column_type_tag(int_array), 0x0F);
        assert_eq!(column_type_tag_v2(int_array), 0x82); // 0x80 | INT(0x02)
        assert_eq!(
            column_type_tag_v2(ColumnType::Array(ArrayElem::Numeric)),
            0x84
        ); // 0x80 | NUMERIC(0x04)
        assert_eq!(column_type_tag_v2(ColumnType::Array(ArrayElem::Text)), 0x85);
        assert_eq!(
            column_type_tag_v2(ColumnType::Array(ArrayElem::TimestampTz)),
            0x8B
        );
        // A scalar is unchanged between v1 and v2.
        assert_eq!(
            column_type_tag_v2(ColumnType::Int),
            column_type_tag(ColumnType::Int)
        );
    }

    #[test]
    fn array_element_tag_round_trips_and_names() {
        assert_eq!(array_element_tag(0x82), Some(0x02)); // INT[]
        assert_eq!(
            array_element_tag(column_type_tag_v2(ColumnType::Array(ArrayElem::Uuid))),
            Some(0x0D)
        );
        // A plain scalar / 0x0F array tag is not element-typed.
        assert_eq!(array_element_tag(0x02), None);
        assert_eq!(array_element_tag(0x0F), None);
        // Both the plain and element-typed array tags name as ARRAY.
        assert_eq!(column_type_name(0x0F), "ARRAY");
        assert_eq!(column_type_name(0x82), "ARRAY");
        assert_eq!(column_type_name(0x02), "INT");
    }
}
