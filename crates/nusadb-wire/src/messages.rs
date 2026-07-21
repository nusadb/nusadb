//! Nusa Wire Protocol message taxonomy (synchronous encode/decode).
//!
//! Messages are direction-scoped: a [`FrontendMessage`] travels client→server, a
//! [`BackendMessage`] server→client. Each maps to a [`Frame`] — a type byte plus a payload of
//! length-prefixed fields (`[len:u32][bytes]` for strings, network byte order throughout).
//!
//! Type bytes are reused across directions (e.g. `S` is Startup outbound but never appears
//! inbound), which is unambiguous because a connection always knows which side it is decoding.
//!
//! This batch covers the simple-query path and the success leg of auth; the extended-query
//! messages (Parse/Bind/Execute/Sync), SCRAM challenge frames, and COPY follow.

use bytes::{Buf, BufMut, Bytes, BytesMut};

use crate::PROTOCOL_MAGIC;
use crate::error::WireError;
use crate::frame::Frame;

// Frontend (client → server) type bytes.
const T_STARTUP: u8 = b'S';
const T_QUERY: u8 = b'Q';
const T_TERMINATE: u8 = b'X';
// Out-of-band cancel request, sent on a fresh connection in place of Startup.
const T_CANCEL_REQUEST: u8 = b'K';
// Extended query.
const T_PARSE: u8 = b'P';
const T_BIND: u8 = b'B';
const T_DESCRIBE: u8 = b'D';
const T_EXECUTE: u8 = b'E';
const T_SYNC: u8 = b'Y';
const T_CLOSE: u8 = b'C';
// COPY sub-protocol, client → server.
const T_COPY_DATA: u8 = b'd';
const T_COPY_DONE: u8 = b'c';
const T_COPY_FAIL: u8 = b'f';
// SCRAM SASL, client → server.
const T_SASL_INITIAL: u8 = b'p';
const T_SASL_RESPONSE: u8 = b'r';

// Backend cancellation key, server → client.
const T_BACKEND_KEY_DATA: u8 = b'K';

// Backend (server → client) type bytes.
const T_AUTH_OK: u8 = b'R';
// Authentication sub-codes carried in the `R` message's leading `u32` (standard wire
// numbering): `0` = ok, `10` = offer SASL, `11` = SASL continue, `12` = SASL final.
const AUTH_OK: u32 = 0;
const AUTH_SASL: u32 = 10;
const AUTH_SASL_CONTINUE: u32 = 11;
const AUTH_SASL_FINAL: u32 = 12;
const T_READY: u8 = b'Z';
// Server → client run-time parameter report (`name\0value\0`), sent during the startup handshake so a
// client can read `server_version` / `client_encoding` / etc. The `S` tag is the backend message
// space (distinct from the frontend `T_STARTUP`, which is decoded by a separate enum).
const T_PARAMETER_STATUS: u8 = b'S';
const T_COMMAND_COMPLETE: u8 = b'C';
const T_ERROR: u8 = b'E';
const T_ROW_DESCRIPTION: u8 = b'T';
// Typed row description (protocol 1.1). A distinct message byte (not a version-gated `T`
// layout) so a client disambiguates typed vs untyped by message type, with no extra negotiation.
const T_ROW_DESCRIPTION_TYPED: u8 = b'y';
const T_DATA_ROW: u8 = b'D';
// Extended query replies.
const T_PARSE_COMPLETE: u8 = b'1';
const T_BIND_COMPLETE: u8 = b'2';
const T_CLOSE_COMPLETE: u8 = b'3';
const T_PARAMETER_DESCRIPTION: u8 = b't';
const T_NO_DATA: u8 = b'n';
const T_PORTAL_SUSPENDED: u8 = b'z';
// COPY sub-protocol, server → client.
const T_COPY_IN_RESPONSE: u8 = b'G';
const T_COPY_OUT_RESPONSE: u8 = b'H';
// CopyData / CopyDone flow both ways; the backend forms (`COPY ... TO STDOUT`) reuse the letters.
const T_COPY_OUT_DATA: u8 = b'd';
const T_COPY_OUT_DONE: u8 = b'c';
// Asynchronous LISTEN/NOTIFY delivery, server → client. Unsolicited: the server pushes it to a
// listening connection between statements (when the connection is idle), carrying the notifying
// backend's pid, the channel name, and its (possibly empty) payload.
const T_NOTIFICATION_RESPONSE: u8 = b'A';

/// What a [`FrontendMessage::Describe`] or [`FrontendMessage::Close`] targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescribeTarget {
    /// A prepared statement (created by `Parse`).
    Statement,
    /// A bound portal (created by `Bind`).
    Portal,
}

impl DescribeTarget {
    const fn tag(self) -> u8 {
        match self {
            Self::Statement => b'S',
            Self::Portal => b'P',
        }
    }

    const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            b'S' => Some(Self::Statement),
            b'P' => Some(Self::Portal),
            _ => None,
        }
    }
}

/// Transaction status carried by [`BackendMessage::ReadyForQuery`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnStatus {
    /// Not in a transaction.
    Idle,
    /// Inside an open transaction.
    InTransaction,
    /// Inside a transaction that has errored and must be rolled back.
    Failed,
}

impl TxnStatus {
    const fn tag(self) -> u8 {
        match self {
            Self::Idle => b'I',
            Self::InTransaction => b'T',
            Self::Failed => b'E',
        }
    }

    const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            b'I' => Some(Self::Idle),
            b'T' => Some(Self::InTransaction),
            b'E' => Some(Self::Failed),
            _ => None,
        }
    }
}

/// A message sent from a client to the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrontendMessage {
    /// First message: protocol magic + version + the user and database to connect to.
    Startup {
        /// Requested protocol major version.
        major: u16,
        /// Requested protocol minor version.
        minor: u16,
        /// Authenticating user.
        user: String,
        /// Target database.
        database: String,
    },
    /// A one-off SQL string (simple query).
    Query {
        /// The SQL text.
        sql: String,
    },
    /// Extended query: create a prepared statement `name` from `sql`. An empty `name` is
    /// the unnamed statement. `param_types` are placeholder type hints (empty until parameters
    /// land in a follow-up).
    Parse {
        /// Prepared-statement name (empty = unnamed).
        name: String,
        /// SQL text of the statement.
        sql: String,
        /// Declared parameter type tags, in order (currently always empty).
        param_types: Vec<u8>,
    },
    /// Extended query: bind `statement` into a portal `portal` with `params`. Parameter
    /// values are raw bytes (text format), `None` for SQL `NULL`.
    Bind {
        /// Destination portal name (empty = unnamed).
        portal: String,
        /// Source prepared-statement name (empty = unnamed).
        statement: String,
        /// Bound parameter values, in order.
        params: Vec<Option<Vec<u8>>>,
        /// Result-column format codes: `0` = text, `1` = binary. Empty means every column
        /// is text; a single entry applies to all columns; otherwise one code per output column.
        result_formats: Vec<u16>,
    },
    /// Extended query: request the metadata of a statement or portal.
    Describe {
        /// Whether `name` is a statement or a portal.
        target: DescribeTarget,
        /// The statement/portal name.
        name: String,
    },
    /// Extended query: run portal `portal`, returning at most `max_rows` rows (`0` = all).
    Execute {
        /// Portal to run (empty = unnamed).
        portal: String,
        /// Row cap; `0` means unlimited.
        max_rows: u32,
    },
    /// Extended query: close the current pipeline; the server replies `ReadyForQuery`.
    Sync,
    /// Extended query: close (free) a statement or portal.
    Close {
        /// Whether `name` is a statement or a portal.
        target: DescribeTarget,
        /// The statement/portal name.
        name: String,
    },
    /// SCRAM: the client's chosen SASL mechanism + `client-first` message.
    SaslInitialResponse {
        /// The selected SASL mechanism (`SCRAM-SHA-256`).
        mechanism: String,
        /// The `client-first-message` bytes.
        data: Vec<u8>,
    },
    /// SCRAM: a subsequent SASL message — the `client-final` message.
    SaslResponse {
        /// The `client-final-message` bytes.
        data: Vec<u8>,
    },
    /// COPY sub-protocol: a chunk of `COPY ... FROM STDIN` data. One message may carry any
    /// number of whole or partial data lines; the server reassembles them.
    CopyData {
        /// Raw bytes of the data stream chunk.
        data: Vec<u8>,
    },
    /// COPY sub-protocol: the client finished sending `COPY FROM` data.
    CopyDone,
    /// COPY sub-protocol: the client aborts the in-progress `COPY FROM`.
    CopyFail {
        /// Why the client aborted.
        message: String,
    },
    /// Out-of-band cancel request: sent on a *fresh* connection (in place of `Startup`) to
    /// cancel the in-flight statement of another connection identified by its backend key.
    CancelRequest {
        /// The target connection's backend process id (from its `BackendKeyData`).
        pid: u32,
        /// The target connection's secret key — proves the requester saw the original key.
        secret: u32,
    },
    /// Politely close the connection.
    Terminate,
}

impl FrontendMessage {
    /// Encode this message into a [`Frame`].
    ///
    /// # Errors
    /// [`WireError::FieldTooLarge`] if a string, field, or count exceeds the width its on-wire
    /// length prefix reserves (it cannot be sent without truncating the prefix and desyncing the
    /// stream).
    pub fn encode(&self) -> Result<Frame, WireError> {
        let mut p = BytesMut::new();
        let ty = match self {
            Self::Startup {
                major,
                minor,
                user,
                database,
            } => {
                p.put_u32(PROTOCOL_MAGIC);
                p.put_u16(*major);
                p.put_u16(*minor);
                put_str(&mut p, user)?;
                put_str(&mut p, database)?;
                T_STARTUP
            },
            Self::Query { sql } => {
                put_str(&mut p, sql)?;
                T_QUERY
            },
            Self::SaslInitialResponse { mechanism, data } => {
                put_str(&mut p, mechanism)?;
                put_u32_len(&mut p, data.len())?;
                p.put_slice(data);
                T_SASL_INITIAL
            },
            Self::SaslResponse { data } => {
                p.put_slice(data);
                T_SASL_RESPONSE
            },
            Self::CopyData { data } => {
                p.put_slice(data);
                T_COPY_DATA
            },
            Self::CopyDone => T_COPY_DONE,
            Self::CopyFail { message } => {
                put_str(&mut p, message)?;
                T_COPY_FAIL
            },
            Self::CancelRequest { pid, secret } => {
                p.put_u32(*pid);
                p.put_u32(*secret);
                T_CANCEL_REQUEST
            },
            Self::Parse {
                name,
                sql,
                param_types,
            } => {
                put_str(&mut p, name)?;
                put_str(&mut p, sql)?;
                put_u16_count(&mut p, param_types.len())?;
                p.put_slice(param_types);
                T_PARSE
            },
            Self::Bind {
                portal,
                statement,
                params,
                result_formats,
            } => {
                put_str(&mut p, portal)?;
                put_str(&mut p, statement)?;
                put_fields(&mut p, params)?;
                put_u16_count(&mut p, result_formats.len())?;
                for code in result_formats {
                    p.put_u16(*code);
                }
                T_BIND
            },
            Self::Describe { target, name } => {
                p.put_u8(target.tag());
                put_str(&mut p, name)?;
                T_DESCRIBE
            },
            Self::Execute { portal, max_rows } => {
                put_str(&mut p, portal)?;
                p.put_u32(*max_rows);
                T_EXECUTE
            },
            Self::Sync => T_SYNC,
            Self::Close { target, name } => {
                p.put_u8(target.tag());
                put_str(&mut p, name)?;
                T_CLOSE
            },
            Self::Terminate => T_TERMINATE,
        };
        Ok(Frame::new(ty, p.freeze()))
    }

    /// Decode a frame received from a client.
    ///
    /// # Errors
    /// [`WireError`] for an unknown type, a truncated payload, bad magic, or invalid UTF-8.
    pub fn decode(frame: &Frame) -> Result<Self, WireError> {
        let mut p = frame.payload.clone();
        match frame.message_type {
            T_STARTUP => {
                let magic = get_u32(&mut p, T_STARTUP)?;
                if magic != PROTOCOL_MAGIC {
                    return Err(WireError::BadMagic(magic));
                }
                let major = get_u16(&mut p, T_STARTUP)?;
                let minor = get_u16(&mut p, T_STARTUP)?;
                let user = get_str(&mut p, T_STARTUP)?;
                let database = get_str(&mut p, T_STARTUP)?;
                Ok(Self::Startup {
                    major,
                    minor,
                    user,
                    database,
                })
            },
            T_QUERY => Ok(Self::Query {
                sql: get_str(&mut p, T_QUERY)?,
            }),
            T_PARSE => {
                let name = get_str(&mut p, T_PARSE)?;
                let sql = get_str(&mut p, T_PARSE)?;
                let n = get_u16(&mut p, T_PARSE)? as usize;
                if p.remaining() < n {
                    return Err(WireError::TruncatedPayload(T_PARSE));
                }
                Ok(Self::Parse {
                    name,
                    sql,
                    param_types: p.split_to(n).to_vec(),
                })
            },
            T_BIND => {
                let portal = get_str(&mut p, T_BIND)?;
                let statement = get_str(&mut p, T_BIND)?;
                let params = get_fields(&mut p, T_BIND)?;
                let format_count = get_u16(&mut p, T_BIND)? as usize;
                // Bound the speculative allocation by what the remaining bytes can actually hold:
                // each result format is a u16 (2 bytes), so a client's count cannot exceed
                // remaining/2. Without this a (bounded but wasteful) ~128 KiB Vec could be reserved
                // from an unvalidated u16 before the loop hits a truncated payload.
                let mut result_formats = Vec::with_capacity(format_count.min(p.remaining() / 2));
                for _ in 0..format_count {
                    result_formats.push(get_u16(&mut p, T_BIND)?);
                }
                Ok(Self::Bind {
                    portal,
                    statement,
                    params,
                    result_formats,
                })
            },
            T_DESCRIBE => Ok(Self::Describe {
                target: DescribeTarget::from_tag(get_u8(&mut p, T_DESCRIBE)?)
                    .ok_or(WireError::MalformedPayload(T_DESCRIBE))?,
                name: get_str(&mut p, T_DESCRIBE)?,
            }),
            T_EXECUTE => Ok(Self::Execute {
                portal: get_str(&mut p, T_EXECUTE)?,
                max_rows: get_u32(&mut p, T_EXECUTE)?,
            }),
            T_SYNC => Ok(Self::Sync),
            T_CLOSE => Ok(Self::Close {
                target: DescribeTarget::from_tag(get_u8(&mut p, T_CLOSE)?)
                    .ok_or(WireError::MalformedPayload(T_CLOSE))?,
                name: get_str(&mut p, T_CLOSE)?,
            }),
            T_SASL_INITIAL => {
                let mechanism = get_str(&mut p, T_SASL_INITIAL)?;
                let len = get_u32(&mut p, T_SASL_INITIAL)? as usize;
                if p.remaining() < len {
                    return Err(WireError::TruncatedPayload(T_SASL_INITIAL));
                }
                Ok(Self::SaslInitialResponse {
                    mechanism,
                    data: p.split_to(len).to_vec(),
                })
            },
            T_SASL_RESPONSE => Ok(Self::SaslResponse { data: p.to_vec() }),
            T_CANCEL_REQUEST => Ok(Self::CancelRequest {
                pid: get_u32(&mut p, T_CANCEL_REQUEST)?,
                secret: get_u32(&mut p, T_CANCEL_REQUEST)?,
            }),
            T_COPY_DATA => Ok(Self::CopyData { data: p.to_vec() }),
            T_COPY_DONE => Ok(Self::CopyDone),
            T_COPY_FAIL => Ok(Self::CopyFail {
                message: get_str(&mut p, T_COPY_FAIL)?,
            }),
            T_TERMINATE => Ok(Self::Terminate),
            other => Err(WireError::UnknownMessageType(other)),
        }
    }
}

/// A message sent from the server to a client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendMessage {
    /// Authentication succeeded; the connection may proceed.
    AuthOk,
    /// SCRAM: offer the SASL mechanisms the server supports.
    AuthSasl {
        /// Supported SASL mechanism names (`SCRAM-SHA-256`).
        mechanisms: Vec<String>,
    },
    /// SCRAM: a SASL challenge — the `server-first` message.
    AuthSaslContinue {
        /// The `server-first-message` bytes.
        data: Vec<u8>,
    },
    /// SCRAM: the final SASL message — the `server-final` (verifier) message.
    AuthSaslFinal {
        /// The `server-final-message` bytes.
        data: Vec<u8>,
    },
    /// A run-time parameter report (`name = value`), sent during the startup handshake (and, in
    /// principle, whenever a reported parameter changes). Lets a client read `server_version`,
    /// `client_encoding`, `standard_conforming_strings`, etc. without a round-trip query.
    ParameterStatus {
        /// The parameter name (e.g. `server_version`).
        name: String,
        /// Its current value.
        value: String,
    },
    /// The server is idle and ready for the next query.
    ReadyForQuery(TxnStatus),
    /// A statement finished; `tag` is e.g. `"SELECT 3"` or `"INSERT 1"`.
    CommandComplete {
        /// Completion tag.
        tag: String,
    },
    /// A structured error.
    Error {
        /// 5-character SQLSTATE code.
        code: String,
        /// Human-readable message.
        message: String,
    },
    /// Column metadata preceding the rows of a result set.
    RowDescription {
        /// Output column names, in order.
        columns: Vec<String>,
    },
    /// Typed column metadata preceding the rows of a result set (protocol 1.1). Sent in
    /// place of [`RowDescription`](Self::RowDescription) only when the connection negotiated
    /// `minor >= 1`; each column carries its name plus a 1-byte type tag (the §11.2 taxonomy, see
    /// [`crate::column_type_tag`]).
    RowDescriptionTyped {
        /// Output columns, in order: `(name, type_tag)`.
        columns: Vec<(String, u8)>,
    },
    /// One result row; each field is `Some(bytes)` or `None` for SQL `NULL`.
    DataRow {
        /// Field values, one per column.
        values: Vec<Option<Vec<u8>>>,
    },
    /// Extended query: a `Parse` succeeded.
    ParseComplete,
    /// Extended query: a `Bind` succeeded.
    BindComplete,
    /// Extended query: a `Close` succeeded.
    CloseComplete,
    /// Extended query: the parameter types of a described statement.
    ParameterDescription {
        /// Number of parameters the statement takes.
        count: u16,
    },
    /// Extended query: a described statement/portal produces no result rows.
    NoData,
    /// Extended query: `Execute` hit its `max_rows` cap with rows still pending.
    PortalSuspended,
    /// COPY sub-protocol: the server is ready to receive `COPY ... FROM STDIN` data. The
    /// client replies with a stream of `CopyData` ending in `CopyDone` (or `CopyFail`). `columns`
    /// is the number of target columns (text format for every column).
    CopyInResponse {
        /// Number of target columns.
        columns: u16,
    },
    /// COPY sub-protocol: the server will now stream `COPY ... TO STDOUT` rows. It follows
    /// with `CopyData` messages and a final `CopyDone`. `columns` is the number of output columns
    /// (text format for every column).
    CopyOutResponse {
        /// Number of output columns.
        columns: u16,
    },
    /// COPY sub-protocol: a chunk of `COPY ... TO STDOUT` output.
    CopyData {
        /// Raw bytes of the output chunk.
        data: Vec<u8>,
    },
    /// COPY sub-protocol: the server finished streaming `COPY ... TO STDOUT` rows.
    CopyDone,
    /// The connection's cancellation key, sent once after authentication. A client wanting
    /// to cancel an in-flight statement opens a new connection and sends a `CancelRequest` with
    /// these values.
    BackendKeyData {
        /// This connection's backend process id.
        pid: u32,
        /// This connection's secret key.
        secret: u32,
    },
    /// Asynchronous LISTEN/NOTIFY delivery: the server pushes this, unsolicited, to a connection
    /// listening on `channel` when some backend runs `NOTIFY channel[, payload]`. Delivered only
    /// while the connection is idle (between statements), never mid-result.
    NotificationResponse {
        /// Backend process id of the connection that issued the `NOTIFY`.
        pid: u32,
        /// The channel name the notification was sent on.
        channel: String,
        /// The notification payload (empty string when `NOTIFY` carried none).
        payload: String,
    },
}

impl BackendMessage {
    /// Encode this message into a [`Frame`].
    ///
    /// # Errors
    /// [`WireError::FieldTooLarge`] if a string, field, or column/field count exceeds the width its
    /// on-wire length prefix reserves — e.g. a result with more than 65 535 columns. Emitting it
    /// would truncate the prefix and desync the client, so it is reported instead.
    #[allow(
        clippy::too_many_lines,
        reason = "a flat one-arm-per-message-type dispatch; splitting it would obscure the 1:1 \
                  mapping between a message and its wire encoding"
    )]
    pub fn encode(&self) -> Result<Frame, WireError> {
        let mut p = BytesMut::new();
        let ty = match self {
            Self::AuthOk => {
                p.put_u32(AUTH_OK);
                T_AUTH_OK
            },
            Self::AuthSasl { mechanisms } => {
                p.put_u32(AUTH_SASL);
                put_u16_count(&mut p, mechanisms.len())?;
                for mechanism in mechanisms {
                    put_str(&mut p, mechanism)?;
                }
                T_AUTH_OK
            },
            Self::AuthSaslContinue { data } => {
                p.put_u32(AUTH_SASL_CONTINUE);
                p.put_slice(data);
                T_AUTH_OK
            },
            Self::AuthSaslFinal { data } => {
                p.put_u32(AUTH_SASL_FINAL);
                p.put_slice(data);
                T_AUTH_OK
            },
            Self::ParameterStatus { name, value } => {
                put_str(&mut p, name)?;
                put_str(&mut p, value)?;
                T_PARAMETER_STATUS
            },
            Self::ReadyForQuery(status) => {
                p.put_u8(status.tag());
                T_READY
            },
            Self::CommandComplete { tag } => {
                put_str(&mut p, tag)?;
                T_COMMAND_COMPLETE
            },
            Self::Error { code, message } => {
                put_str(&mut p, code)?;
                put_str(&mut p, message)?;
                T_ERROR
            },
            Self::RowDescription { columns } => {
                put_u16_count(&mut p, columns.len())?;
                for column in columns {
                    put_str(&mut p, column)?;
                }
                T_ROW_DESCRIPTION
            },
            Self::RowDescriptionTyped { columns } => {
                put_u16_count(&mut p, columns.len())?;
                for (name, type_tag) in columns {
                    put_str(&mut p, name)?;
                    p.put_u8(*type_tag);
                }
                T_ROW_DESCRIPTION_TYPED
            },
            Self::DataRow { values } => {
                put_fields(&mut p, values)?;
                T_DATA_ROW
            },
            Self::ParseComplete => T_PARSE_COMPLETE,
            Self::BindComplete => T_BIND_COMPLETE,
            Self::CloseComplete => T_CLOSE_COMPLETE,
            Self::ParameterDescription { count } => {
                p.put_u16(*count);
                T_PARAMETER_DESCRIPTION
            },
            Self::NoData => T_NO_DATA,
            Self::PortalSuspended => T_PORTAL_SUSPENDED,
            Self::CopyInResponse { columns } => {
                // Format byte 0 = text for the overall copy and every column (mirrored implicitly).
                p.put_u8(0);
                p.put_u16(*columns);
                T_COPY_IN_RESPONSE
            },
            Self::CopyOutResponse { columns } => {
                p.put_u8(0); // 0 = text format
                p.put_u16(*columns);
                T_COPY_OUT_RESPONSE
            },
            Self::CopyData { data } => {
                p.put_slice(data);
                T_COPY_OUT_DATA
            },
            Self::CopyDone => T_COPY_OUT_DONE,
            Self::BackendKeyData { pid, secret } => {
                p.put_u32(*pid);
                p.put_u32(*secret);
                T_BACKEND_KEY_DATA
            },
            Self::NotificationResponse {
                pid,
                channel,
                payload,
            } => {
                p.put_u32(*pid);
                put_str(&mut p, channel)?;
                put_str(&mut p, payload)?;
                T_NOTIFICATION_RESPONSE
            },
        };
        Ok(Frame::new(ty, p.freeze()))
    }

    /// Decode a frame received from the server.
    ///
    /// # Errors
    /// [`WireError`] for an unknown type, a truncated payload, an invalid status tag, or
    /// invalid UTF-8.
    pub fn decode(frame: &Frame) -> Result<Self, WireError> {
        let mut p = frame.payload.clone();
        match frame.message_type {
            T_AUTH_OK => match get_u32(&mut p, T_AUTH_OK)? {
                AUTH_OK => Ok(Self::AuthOk),
                AUTH_SASL => {
                    let n = get_u16(&mut p, T_AUTH_OK)? as usize;
                    let mut mechanisms = Vec::with_capacity(n);
                    for _ in 0..n {
                        mechanisms.push(get_str(&mut p, T_AUTH_OK)?);
                    }
                    Ok(Self::AuthSasl { mechanisms })
                },
                AUTH_SASL_CONTINUE => Ok(Self::AuthSaslContinue { data: p.to_vec() }),
                AUTH_SASL_FINAL => Ok(Self::AuthSaslFinal { data: p.to_vec() }),
                _ => Err(WireError::MalformedPayload(T_AUTH_OK)),
            },
            T_READY => {
                let tag = get_u8(&mut p, T_READY)?;
                let status =
                    TxnStatus::from_tag(tag).ok_or(WireError::MalformedPayload(T_READY))?;
                Ok(Self::ReadyForQuery(status))
            },
            T_PARAMETER_STATUS => Ok(Self::ParameterStatus {
                name: get_str(&mut p, T_PARAMETER_STATUS)?,
                value: get_str(&mut p, T_PARAMETER_STATUS)?,
            }),
            T_COMMAND_COMPLETE => Ok(Self::CommandComplete {
                tag: get_str(&mut p, T_COMMAND_COMPLETE)?,
            }),
            T_ERROR => Ok(Self::Error {
                code: get_str(&mut p, T_ERROR)?,
                message: get_str(&mut p, T_ERROR)?,
            }),
            T_ROW_DESCRIPTION => {
                let n = get_u16(&mut p, T_ROW_DESCRIPTION)? as usize;
                let mut columns = Vec::with_capacity(n);
                for _ in 0..n {
                    columns.push(get_str(&mut p, T_ROW_DESCRIPTION)?);
                }
                Ok(Self::RowDescription { columns })
            },
            T_ROW_DESCRIPTION_TYPED => {
                let n = get_u16(&mut p, T_ROW_DESCRIPTION_TYPED)? as usize;
                let mut columns = Vec::with_capacity(n);
                for _ in 0..n {
                    let name = get_str(&mut p, T_ROW_DESCRIPTION_TYPED)?;
                    let type_tag = get_u8(&mut p, T_ROW_DESCRIPTION_TYPED)?;
                    columns.push((name, type_tag));
                }
                Ok(Self::RowDescriptionTyped { columns })
            },
            T_DATA_ROW => Ok(Self::DataRow {
                values: get_fields(&mut p, T_DATA_ROW)?,
            }),
            T_PARSE_COMPLETE => Ok(Self::ParseComplete),
            T_BIND_COMPLETE => Ok(Self::BindComplete),
            T_CLOSE_COMPLETE => Ok(Self::CloseComplete),
            T_PARAMETER_DESCRIPTION => Ok(Self::ParameterDescription {
                count: get_u16(&mut p, T_PARAMETER_DESCRIPTION)?,
            }),
            T_NO_DATA => Ok(Self::NoData),
            T_PORTAL_SUSPENDED => Ok(Self::PortalSuspended),
            T_COPY_IN_RESPONSE => {
                let _format = get_u8(&mut p, T_COPY_IN_RESPONSE)?;
                Ok(Self::CopyInResponse {
                    columns: get_u16(&mut p, T_COPY_IN_RESPONSE)?,
                })
            },
            T_COPY_OUT_RESPONSE => {
                let _format = get_u8(&mut p, T_COPY_OUT_RESPONSE)?;
                Ok(Self::CopyOutResponse {
                    columns: get_u16(&mut p, T_COPY_OUT_RESPONSE)?,
                })
            },
            T_COPY_OUT_DATA => Ok(Self::CopyData { data: p.to_vec() }),
            T_COPY_OUT_DONE => Ok(Self::CopyDone),
            T_BACKEND_KEY_DATA => Ok(Self::BackendKeyData {
                pid: get_u32(&mut p, T_BACKEND_KEY_DATA)?,
                secret: get_u32(&mut p, T_BACKEND_KEY_DATA)?,
            }),
            T_NOTIFICATION_RESPONSE => Ok(Self::NotificationResponse {
                pid: get_u32(&mut p, T_NOTIFICATION_RESPONSE)?,
                channel: get_str(&mut p, T_NOTIFICATION_RESPONSE)?,
                payload: get_str(&mut p, T_NOTIFICATION_RESPONSE)?,
            }),
            other => Err(WireError::UnknownMessageType(other)),
        }
    }
}

/// Write a `u16` count prefix, erroring (not truncating) when `len > u16::MAX` (N1 / G21).
fn put_u16_count(buf: &mut BytesMut, len: usize) -> Result<(), WireError> {
    buf.put_u16(u16::try_from(len).map_err(|_| WireError::FieldTooLarge)?);
    Ok(())
}

/// Write a `u32` length prefix, erroring (not truncating) when `len > u32::MAX` (N1 / G21).
fn put_u32_len(buf: &mut BytesMut, len: usize) -> Result<(), WireError> {
    buf.put_u32(u32::try_from(len).map_err(|_| WireError::FieldTooLarge)?);
    Ok(())
}

fn put_str(buf: &mut BytesMut, s: &str) -> Result<(), WireError> {
    put_u32_len(buf, s.len())?;
    buf.put_slice(s.as_bytes());
    Ok(())
}

/// Encode a list of optional byte fields: `[count:u16]` then per field a present-tag
/// (`1` + `[len:u32][bytes]`, or `0` for `NULL`). Shared by `DataRow` and `Bind`.
fn put_fields(buf: &mut BytesMut, fields: &[Option<Vec<u8>>]) -> Result<(), WireError> {
    put_u16_count(buf, fields.len())?;
    for field in fields {
        match field {
            Some(bytes) => {
                buf.put_u8(1);
                put_u32_len(buf, bytes.len())?;
                buf.put_slice(bytes);
            },
            None => buf.put_u8(0),
        }
    }
    Ok(())
}

/// Decode the field list written by [`put_fields`].
fn get_fields(buf: &mut Bytes, ty: u8) -> Result<Vec<Option<Vec<u8>>>, WireError> {
    let n = get_u16(buf, ty)? as usize;
    // Bound the speculative allocation: every field is at least one byte (its present/NULL tag),
    // so a client's count cannot exceed the remaining byte length. This caps the reservation made
    // from an unvalidated u16 before the loop reaches a truncated payload.
    let mut fields = Vec::with_capacity(n.min(buf.remaining()));
    for _ in 0..n {
        if get_u8(buf, ty)? == 0 {
            fields.push(None);
        } else {
            let len = get_u32(buf, ty)? as usize;
            if buf.remaining() < len {
                return Err(WireError::TruncatedPayload(ty));
            }
            fields.push(Some(buf.split_to(len).to_vec()));
        }
    }
    Ok(fields)
}

fn get_u8(buf: &mut Bytes, ty: u8) -> Result<u8, WireError> {
    if buf.remaining() < 1 {
        return Err(WireError::TruncatedPayload(ty));
    }
    Ok(buf.get_u8())
}

fn get_u16(buf: &mut Bytes, ty: u8) -> Result<u16, WireError> {
    if buf.remaining() < 2 {
        return Err(WireError::TruncatedPayload(ty));
    }
    Ok(buf.get_u16())
}

fn get_u32(buf: &mut Bytes, ty: u8) -> Result<u32, WireError> {
    if buf.remaining() < 4 {
        return Err(WireError::TruncatedPayload(ty));
    }
    Ok(buf.get_u32())
}

fn get_str(buf: &mut Bytes, ty: u8) -> Result<String, WireError> {
    let len = get_u32(buf, ty)? as usize;
    if buf.remaining() < len {
        return Err(WireError::TruncatedPayload(ty));
    }
    let bytes = buf.split_to(len);
    String::from_utf8(bytes.to_vec()).map_err(|_| WireError::InvalidString)
}
