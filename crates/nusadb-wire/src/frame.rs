//! TCP frame codec: `[type:u8][len:u32][payload]`, network byte order (big-endian).
//!
//! `len` is the **total** frame size (header + payload), so a reader knows exactly how many
//! bytes to expect. Frames larger than [`MAX_FRAME_LEN`] are rejected at
//! decode to bound memory. This codec is synchronous and buffer-based; the async server feeds
//! it bytes read from the socket.

// Byte codec: every index below is guarded by a preceding length check.
#![allow(clippy::indexing_slicing)]

use bytes::{Buf, BufMut, Bytes, BytesMut};

use crate::MAX_FRAME_LEN;
use crate::error::WireError;

/// `type:u8` + `len:u32`.
const HEADER_LEN: usize = 5;

/// One wire-protocol frame: a message type byte plus its (already length-validated) payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Single-byte message type identifier (direction-scoped — see [`messages`](crate::messages)).
    pub message_type: u8,
    /// The frame body, with the `[type][len]` header stripped.
    pub payload: Bytes,
}

impl Frame {
    /// Build a frame from a type byte and payload.
    #[must_use]
    pub const fn new(message_type: u8, payload: Bytes) -> Self {
        Self {
            message_type,
            payload,
        }
    }

    /// Append this frame's wire encoding to `out`.
    pub fn encode(&self, out: &mut BytesMut) {
        let total = (self.payload.len() + HEADER_LEN) as u32;
        out.put_u8(self.message_type);
        out.put_u32(total);
        out.put_slice(&self.payload);
    }

    /// Try to decode one frame from the front of `buf`, consuming its bytes on success.
    ///
    /// Returns `Ok(None)` when `buf` does not yet hold a complete frame (the caller should read
    /// more bytes and retry).
    ///
    /// # Errors
    /// [`WireError::MalformedFrame`] if the declared length is below the header, or
    /// [`WireError::FrameTooLarge`] if it exceeds [`MAX_FRAME_LEN`].
    pub fn decode(buf: &mut BytesMut) -> Result<Option<Self>, WireError> {
        if buf.len() < HEADER_LEN {
            return Ok(None);
        }
        let message_type = buf[0];
        let total = u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]);
        if total < HEADER_LEN as u32 {
            return Err(WireError::MalformedFrame(total));
        }
        if total > MAX_FRAME_LEN {
            return Err(WireError::FrameTooLarge(total));
        }
        let total = total as usize;
        if buf.len() < total {
            return Ok(None); // frame not fully received yet
        }
        buf.advance(HEADER_LEN);
        let payload = buf.split_to(total - HEADER_LEN).freeze();
        Ok(Some(Self {
            message_type,
            payload,
        }))
    }
}
