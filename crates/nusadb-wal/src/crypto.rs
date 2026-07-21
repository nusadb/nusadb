//! Record-level encryption at rest for the WAL.
//!
//! A [`WalWriter`](crate::WalWriter) opened with a key compresses each record's payload (lz4) and
//! then seals the compressed bytes with AES-256-GCM-SIV (RustCrypto `aes-gcm-siv`), a
//! nonce-misuse-resistant AEAD; the CRC32 in the frame header then covers the ciphertext. A
//! [`WalReader`](crate::WalReader) opened with the same key reverses it (CRC check, decrypt,
//! decompress). A wrong key or tampered bytes fail the AEAD tag check, so recovery stops at that
//! record rather than replaying garbage.
//!
//! # Nonce derivation
//!
//! Each record's nonce is derived from its LSN. `WAL` truncation resets the LSN to 1, so an LSN can
//! recur across log generations under the same key — AES-GCM-SIV is misuse-resistant, so a repeat
//! is safe (it would at worst reveal that two identical plaintexts were sealed under the same
//! nonce, never compromising the key). A domain byte distinct from the at-rest data-block cipher's
//! nonce keeps the two from colliding even when the same data-encryption key backs both.

#![allow(
    clippy::doc_markdown,
    reason = "cipher names like AES-256-GCM-SIV read better unbackticked"
)]

use aes_gcm_siv::aead::{Aead, KeyInit};
use aes_gcm_siv::{Aes256GcmSiv, Key, Nonce};

/// Domain separator placed in the WAL nonce so a WAL record and an at-rest data block
/// never share a nonce under the same data-encryption key. The data-block cipher
/// leaves these high bytes zero.
const WAL_NONCE_DOMAIN: u8 = 0x01;

/// An AES-256-GCM-SIV cipher bound to the WAL's 32-byte data-encryption key, sealing/opening each
/// record under an LSN-derived nonce.
pub(crate) struct WalCipher {
    cipher: Aes256GcmSiv,
}

impl std::fmt::Debug for WalCipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never leak key material through `Debug`.
        f.debug_struct("WalCipher").finish_non_exhaustive()
    }
}

impl WalCipher {
    /// Build the cipher for the raw 32-byte data-encryption key.
    pub(crate) fn new(key: &[u8; 32]) -> Self {
        Self {
            cipher: Aes256GcmSiv::new(Key::<Aes256GcmSiv>::from_slice(key)),
        }
    }

    /// Seal `plaintext` (an lz4-compressed record payload) under the nonce derived from `lsn`,
    /// returning ciphertext followed by the 16-byte AEAD tag.
    ///
    /// # Errors
    /// `Err(())` on the AEAD's (pathological) encryption failure; the caller maps it to an error.
    pub(crate) fn seal(&self, lsn: u64, plaintext: &[u8]) -> Result<Vec<u8>, ()> {
        self.cipher
            .encrypt(Nonce::from_slice(&nonce_for(lsn)), plaintext)
            .map_err(|_| ())
    }

    /// Open `sealed` (ciphertext ‖ tag) under the nonce derived from `lsn`, returning the recovered
    /// lz4-compressed payload.
    ///
    /// # Errors
    /// `Err(())` if the AEAD tag check fails — a wrong key or tampered bytes.
    pub(crate) fn open(&self, lsn: u64, sealed: &[u8]) -> Result<Vec<u8>, ()> {
        self.cipher
            .decrypt(Nonce::from_slice(&nonce_for(lsn)), sealed)
            .map_err(|_| ())
    }
}

/// Derive the 12-byte AEAD nonce for the record at `lsn`: the LSN little-endian in the low 8 bytes,
/// a WAL domain marker in byte 8, the rest zero. Distinct LSNs yield distinct nonces, and the
/// domain marker keeps WAL nonces disjoint from at-rest data-block nonces under a shared key.
const fn nonce_for(lsn: u64) -> [u8; 12] {
    let l = lsn.to_le_bytes();
    [
        l[0],
        l[1],
        l[2],
        l[3],
        l[4],
        l[5],
        l[6],
        l[7],
        WAL_NONCE_DOMAIN,
        0,
        0,
        0,
    ]
}

#[cfg(test)]
mod tests {
    use super::{WAL_NONCE_DOMAIN, WalCipher, nonce_for};

    #[test]
    fn seal_open_round_trips() {
        let c = WalCipher::new(&[7u8; 32]);
        let pt = b"a compressed wal payload".to_vec();
        let sealed = c.seal(3, &pt).unwrap();
        assert_ne!(sealed, pt);
        assert_eq!(c.open(3, &sealed).unwrap(), pt);
    }

    #[test]
    fn wrong_key_fails() {
        let sealed = WalCipher::new(&[1u8; 32]).seal(0, b"secret").unwrap();
        assert!(WalCipher::new(&[2u8; 32]).open(0, &sealed).is_err());
    }

    #[test]
    fn wrong_lsn_fails() {
        // The LSN feeds the nonce, which the AEAD binds into the tag, so opening under a different
        // LSN must fail rather than return data.
        let c = WalCipher::new(&[9u8; 32]);
        let sealed = c.seal(5, b"payload").unwrap();
        assert!(c.open(6, &sealed).is_err());
    }

    #[test]
    fn tampered_byte_fails() {
        let c = WalCipher::new(&[4u8; 32]);
        let mut sealed = c.seal(0, b"payload").unwrap();
        sealed[0] ^= 0x01;
        assert!(c.open(0, &sealed).is_err());
    }

    #[test]
    fn distinct_lsns_give_distinct_nonces() {
        assert_ne!(nonce_for(0), nonce_for(1));
        assert_ne!(nonce_for(1), nonce_for(2));
    }

    #[test]
    fn nonce_carries_the_wal_domain_marker() {
        // Byte 8 marks the WAL domain so a WAL nonce can't collide with a data-block nonce
        // (which leaves byte 8 zero) under a shared key.
        assert_eq!(nonce_for(42)[8], WAL_NONCE_DOMAIN);
    }
}
