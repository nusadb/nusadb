//! Authenticated encryption for data at rest.
//!
//! [`Sealer`] seals byte strings with AES-256-GCM-SIV (RustCrypto `aes-gcm-siv`), a
//! nonce-misuse-resistant AEAD. A sealed string is `nonce (12) ‖ ciphertext ‖ tag (16)`, so it is
//! [`SEAL_OVERHEAD`] bytes longer than its plaintext. The caller passes additional authenticated
//! data (AAD) that binds the string to where it belongs (a log position, a page id and slot): a
//! string moved elsewhere fails its tag check exactly like a tampered one.
//!
//! # Nonces
//!
//! Every [`Sealer`] draws a random 96-bit base from the operating system when it is created and
//! hands out `base + n` for its `n`-th seal (a 64-bit counter added to the low eight bytes). One
//! sealer never repeats a nonce. Two sealers under the same key (two runs of the server) can
//! repeat one only if their nonce ranges overlap: with `S` sealers of at most `n` seals each, the
//! chance that any two overlap is at most about `S² · n / 2⁹⁶` (for a million runs of a billion
//! seals each, about 2⁻³⁶). The nonce is
//! stored with the string, so nothing about it has to be derived again when it is read, and a
//! position that is written twice (a log truncated and appended again) still gets a fresh nonce.
//! Should two seals ever share a nonce, GCM-SIV reveals only whether their plaintexts were equal.
//!
//! The WAL uses it through [`RecordSeal`]: [`WalWriter::with_seal`](crate::WalWriter::with_seal)
//! seals each framed record body, and the CRC32 of the frame then covers the ciphertext.

#![allow(
    clippy::doc_markdown,
    reason = "cipher names like AES-256-GCM-SIV read better unbackticked"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use aes_gcm_siv::aead::{AeadInPlace, KeyInit};
use aes_gcm_siv::{Aes256GcmSiv, Key, Nonce, Tag};
use nusadb_core::{Error, Result};

/// Bytes of the nonce stored in front of every sealed string.
pub const NONCE_LEN: usize = 12;
/// Bytes of the authentication tag stored after every sealed string.
pub const TAG_LEN: usize = 16;
/// How much longer a sealed string is than its plaintext.
pub const SEAL_OVERHEAD: usize = NONCE_LEN + TAG_LEN;
/// The length of a key: AES-256.
pub const KEY_LEN: usize = 32;

/// Fill `buf` from the operating system's random source.
///
/// # Errors
/// The random source failed.
pub fn random_bytes(buf: &mut [u8]) -> Result<()> {
    getrandom::fill(buf).map_err(|e| {
        Error::Io(std::io::Error::other(format!(
            "the operating system's random source failed: {e}"
        )))
    })
}

/// Seals and opens byte strings under one 32-byte key. See the module docs.
pub struct Sealer {
    aead: Aes256GcmSiv,
    /// The random nonce base; the `n`-th seal uses `base + n`.
    base: [u8; NONCE_LEN],
    /// Seals handed out so far.
    counter: AtomicU64,
}

impl std::fmt::Debug for Sealer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never leak key material through `Debug`.
        f.debug_struct("Sealer").finish_non_exhaustive()
    }
}

impl Sealer {
    /// A sealer for `key`, with a fresh random nonce base.
    ///
    /// # Errors
    /// The random source failed.
    pub fn new(key: &[u8; KEY_LEN]) -> Result<Self> {
        let mut base = [0u8; NONCE_LEN];
        random_bytes(&mut base)?;
        Ok(Self {
            aead: Aes256GcmSiv::new(Key::<Aes256GcmSiv>::from_slice(key)),
            base,
            counter: AtomicU64::new(0),
        })
    }

    /// A sealer for a random key that exists only in this process: for scratch files that are
    /// never read after the process ends.
    ///
    /// # Errors
    /// The random source failed.
    pub fn ephemeral() -> Result<Self> {
        let mut key = zeroize::Zeroizing::new([0u8; KEY_LEN]);
        random_bytes(key.as_mut_slice())?;
        Self::new(&key)
    }

    /// The nonce for the next seal: the base plus the number of seals before it.
    fn next_nonce(&self) -> [u8; NONCE_LEN] {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut nonce = self.base;
        let (low, _) = nonce.split_at_mut(8);
        let mut word = [0u8; 8];
        word.copy_from_slice(low);
        low.copy_from_slice(&u64::from_le_bytes(word).wrapping_add(n).to_le_bytes());
        nonce
    }

    /// Append `plaintext` sealed under `aad` to `out`: nonce, ciphertext, tag.
    ///
    /// # Errors
    /// The AEAD refused the input (a plaintext beyond its size limit).
    pub fn seal_into(&self, aad: &[u8], plaintext: &[u8], out: &mut Vec<u8>) -> Result<()> {
        let nonce = self.next_nonce();
        out.reserve(plaintext.len() + SEAL_OVERHEAD);
        out.extend_from_slice(&nonce);
        let start = out.len();
        out.extend_from_slice(plaintext);
        let body = out.get_mut(start..).unwrap_or_default();
        let tag = self
            .aead
            .encrypt_in_place_detached(Nonce::from_slice(&nonce), aad, body)
            .map_err(|_| Error::Io(std::io::Error::other("encryption failed")))?;
        out.extend_from_slice(&tag);
        Ok(())
    }

    /// `plaintext` sealed under `aad`.
    ///
    /// # Errors
    /// As [`seal_into`](Self::seal_into).
    pub fn seal(&self, aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(plaintext.len() + SEAL_OVERHEAD);
        self.seal_into(aad, plaintext, &mut out)?;
        Ok(out)
    }

    /// Open `sealed` under `aad` into `out`, which must be exactly the plaintext's length
    /// (`sealed.len() - SEAL_OVERHEAD`). Returns `false`, leaving `out` unspecified, when the
    /// string is too short, the lengths disagree, or the tag does not verify: a wrong key,
    /// tampered bytes, or bytes that belong somewhere else.
    pub fn open_into(&self, aad: &[u8], sealed: &[u8], out: &mut [u8]) -> bool {
        let Some(body_len) = sealed.len().checked_sub(SEAL_OVERHEAD) else {
            return false;
        };
        if out.len() != body_len {
            return false;
        }
        let (Some(nonce), Some(body), Some(tag)) = (
            sealed.get(..NONCE_LEN),
            sealed.get(NONCE_LEN..NONCE_LEN + body_len),
            sealed.get(NONCE_LEN + body_len..),
        ) else {
            return false;
        };
        out.copy_from_slice(body);
        self.aead
            .decrypt_in_place_detached(Nonce::from_slice(nonce), aad, out, Tag::from_slice(tag))
            .is_ok()
    }

    /// Open `sealed` under `aad`, or `None` when it does not verify (see
    /// [`open_into`](Self::open_into)).
    pub fn open(&self, aad: &[u8], sealed: &[u8]) -> Option<Vec<u8>> {
        let mut out = vec![0u8; sealed.len().checked_sub(SEAL_OVERHEAD)?];
        self.open_into(aad, sealed, &mut out).then_some(out)
    }
}

/// How a log's records are sealed: the sealer, and the context bound into every record's AAD
/// together with its position, so a record cannot be moved to another position or into another
/// kind of file.
#[derive(Debug, Clone)]
pub struct RecordSeal {
    /// The sealer of the database's data key.
    pub sealer: Arc<Sealer>,
    /// What kind of file the records belong to (the live log and its archived segments, or a
    /// checkpoint image's records).
    pub domain: u8,
    /// A further number the records are bound to (for an image, the log position it covers).
    pub scope: u64,
}

/// [`RecordSeal::domain`] of the live log and its archived and shipped segments.
pub const DOMAIN_LOG: u8 = 1;
/// [`RecordSeal::domain`] of a checkpoint image's records.
pub const DOMAIN_IMAGE: u8 = 2;

impl RecordSeal {
    /// The AAD of the record at `lsn`: domain, scope, position.
    #[allow(
        clippy::missing_const_for_fn,
        reason = "slice splitting is not const; the AAD is built once per record"
    )]
    pub(crate) fn aad(&self, lsn: u64) -> [u8; 17] {
        let mut aad = [0u8; 17];
        let (domain, rest) = aad.split_at_mut(1);
        domain.copy_from_slice(&[self.domain]);
        let (scope, position) = rest.split_at_mut(8);
        scope.copy_from_slice(&self.scope.to_le_bytes());
        position.copy_from_slice(&lsn.to_le_bytes());
        aad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_open_round_trips() {
        let s = Sealer::new(&[7u8; 32]).unwrap();
        let pt = b"a record payload".to_vec();
        let sealed = s.seal(b"aad", &pt).unwrap();
        assert_eq!(sealed.len(), pt.len() + SEAL_OVERHEAD);
        assert!(!sealed.windows(pt.len()).any(|w| w == pt.as_slice()));
        assert_eq!(s.open(b"aad", &sealed).unwrap(), pt);
    }

    #[test]
    fn a_second_sealer_with_the_same_key_opens_it() {
        let sealed = Sealer::new(&[3u8; 32]).unwrap().seal(b"a", b"x").unwrap();
        assert_eq!(
            Sealer::new(&[3u8; 32])
                .unwrap()
                .open(b"a", &sealed)
                .unwrap(),
            b"x"
        );
    }

    #[test]
    fn wrong_key_fails() {
        let sealed = Sealer::new(&[1u8; 32])
            .unwrap()
            .seal(b"", b"secret")
            .unwrap();
        assert!(
            Sealer::new(&[2u8; 32])
                .unwrap()
                .open(b"", &sealed)
                .is_none()
        );
    }

    #[test]
    fn other_aad_fails() {
        // The AAD binds the string to its place: opened anywhere else it must not verify.
        let s = Sealer::new(&[9u8; 32]).unwrap();
        let sealed = s.seal(b"page 5", b"payload").unwrap();
        assert!(s.open(b"page 6", &sealed).is_none());
    }

    #[test]
    fn tampered_byte_fails() {
        let s = Sealer::new(&[4u8; 32]).unwrap();
        let sealed = s.seal(b"", b"payload").unwrap();
        for i in 0..sealed.len() {
            let mut bad = sealed.clone();
            bad[i] ^= 0x01;
            assert!(
                s.open(b"", &bad).is_none(),
                "flipping byte {i} went unnoticed"
            );
        }
        assert!(s.open(b"", &sealed[..sealed.len() - 1]).is_none());
    }

    #[test]
    fn every_seal_uses_a_new_nonce() {
        // The same plaintext under the same AAD, sealed twice, must give different strings:
        // their nonces differ. A repeated nonce would make them identical.
        let s = Sealer::new(&[5u8; 32]).unwrap();
        let a = s.seal(b"aad", b"same plaintext").unwrap();
        let b = s.seal(b"aad", b"same plaintext").unwrap();
        assert_ne!(a[..NONCE_LEN], b[..NONCE_LEN]);
        assert_ne!(a, b);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            let sealed = s.seal(b"", b"").unwrap();
            assert!(
                seen.insert(sealed[..NONCE_LEN].to_vec()),
                "a nonce repeated"
            );
        }
    }

    #[test]
    fn two_sealers_start_from_different_bases() {
        let a = Sealer::new(&[5u8; 32]).unwrap().seal(b"", b"").unwrap();
        let b = Sealer::new(&[5u8; 32]).unwrap().seal(b"", b"").unwrap();
        assert_ne!(a[..NONCE_LEN], b[..NONCE_LEN]);
    }

    #[test]
    fn record_aad_binds_domain_scope_and_position() {
        let seal = RecordSeal {
            sealer: Arc::new(Sealer::new(&[1u8; 32]).unwrap()),
            domain: DOMAIN_LOG,
            scope: 0,
        };
        let image = RecordSeal {
            domain: DOMAIN_IMAGE,
            ..seal.clone()
        };
        assert_ne!(seal.aad(1), seal.aad(2));
        assert_ne!(seal.aad(1), image.aad(1));
    }
}
