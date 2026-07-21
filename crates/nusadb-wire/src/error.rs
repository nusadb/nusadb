//! Wire-protocol decode errors.

/// An error decoding a frame or message off the wire.
#[derive(Debug, thiserror::Error)]
pub enum WireError {
    /// A frame's declared length exceeds [`MAX_FRAME_LEN`](crate::MAX_FRAME_LEN).
    #[error("frame length {0} exceeds MAX_FRAME_LEN")]
    FrameTooLarge(u32),
    /// A frame's declared length is smaller than the fixed header.
    #[error("malformed frame: declared length {0} is below the header size")]
    MalformedFrame(u32),
    /// The Startup payload did not begin with [`PROTOCOL_MAGIC`](crate::PROTOCOL_MAGIC).
    #[error("bad protocol magic {0:#010x} (not the Nusa Wire Protocol)")]
    BadMagic(u32),
    /// A message type byte that is not part of the taxonomy for its direction.
    #[error("unknown message type byte {0:#04x}")]
    UnknownMessageType(u8),
    /// A message payload ended before all expected fields were read.
    #[error("truncated payload for message type {0:#04x}")]
    TruncatedPayload(u8),
    /// A payload field held an invalid value (e.g. an unknown enum tag).
    #[error("malformed payload for message type {0:#04x}")]
    MalformedPayload(u8),
    /// A protocol string field was not valid UTF-8.
    #[error("invalid UTF-8 in a protocol string")]
    InvalidString,
    /// A count or length field exceeds the width its on-wire prefix reserves — e.g. more than
    /// 65 535 columns/fields in one message, or a field longer than `u32::MAX` — so it cannot be
    /// encoded without truncating the prefix and desyncing the stream (N1 G21).
    #[error("a message field is too large to encode within its length prefix")]
    FieldTooLarge,
    /// A TLS server configuration could not be built (bad certificate or key).
    #[error("TLS configuration error: {0}")]
    Tls(String),
}

impl From<WireError> for std::io::Error {
    fn from(e: WireError) -> Self {
        Self::other(e)
    }
}
