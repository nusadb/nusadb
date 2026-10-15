//! Sealing of spill files when the server encrypts data at rest.
//!
//! Spilled rows are sealed with AES-256-GCM-SIV under a key drawn from the operating system when
//! [`set_spill_encryption`] is called and never written anywhere: a spill file is scratch, read
//! back only by the query that wrote it, so the key dies with the process and with it any chance
//! of reading a file left behind by a crash. Each record is `nonce (12) ‖ ciphertext ‖ tag (16)`;
//! the nonce is a random base drawn with the key plus a counter, so it never repeats within the
//! process.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use aes_gcm_siv::aead::{AeadInPlace, KeyInit};
use aes_gcm_siv::{Aes256GcmSiv, Key, Nonce, Tag};

/// Bytes a sealed record adds to its plaintext: the nonce and the tag.
pub(super) const OVERHEAD: usize = 28;
/// What every sealed spill record is bound to.
const AAD: &[u8] = b"nusadb spill record";

/// The process's spill key.
pub(super) struct SpillSeal {
    aead: Aes256GcmSiv,
    base: [u8; 12],
    counter: AtomicU64,
}

static SEAL: OnceLock<SpillSeal> = OnceLock::new();

/// Seal every spill file written from now on (the server calls this at start-up when data at
/// rest is encrypted). Calling it again keeps the key already drawn.
///
/// # Errors
/// The operating system's random source failed.
pub fn set_spill_encryption() -> Result<(), String> {
    if SEAL.get().is_some() {
        return Ok(());
    }
    let _ = SEAL.set(SpillSeal::random()?);
    Ok(())
}

/// The spill key, when spill files are sealed.
pub(super) fn active() -> Option<&'static SpillSeal> {
    SEAL.get()
}

impl SpillSeal {
    /// A key and nonce base drawn from the operating system.
    fn random() -> Result<Self, String> {
        let mut key = zeroize::Zeroizing::new([0u8; 32]);
        let mut base = [0u8; 12];
        getrandom::fill(key.as_mut_slice()).map_err(|e| e.to_string())?;
        getrandom::fill(&mut base).map_err(|e| e.to_string())?;
        Ok(Self {
            aead: Aes256GcmSiv::new(Key::<Aes256GcmSiv>::from_slice(key.as_slice())),
            base,
            counter: AtomicU64::new(0),
        })
    }

    /// `plaintext` sealed: nonce, ciphertext, tag.
    pub(super) fn seal(&self, plaintext: &[u8]) -> Option<Vec<u8>> {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut nonce = self.base;
        let mut low = [0u8; 8];
        low.copy_from_slice(nonce.get(..8)?);
        nonce
            .get_mut(..8)?
            .copy_from_slice(&u64::from_le_bytes(low).wrapping_add(n).to_le_bytes());
        let mut out = Vec::with_capacity(plaintext.len() + OVERHEAD);
        out.extend_from_slice(&nonce);
        out.extend_from_slice(plaintext);
        let tag = self
            .aead
            .encrypt_in_place_detached(Nonce::from_slice(&nonce), AAD, out.get_mut(12..)?)
            .ok()?;
        out.extend_from_slice(&tag);
        Some(out)
    }

    /// The plaintext of a sealed record, or `None` when it does not verify.
    pub(super) fn open(&self, sealed: &[u8]) -> Option<Vec<u8>> {
        let body_len = sealed.len().checked_sub(OVERHEAD)?;
        let nonce = sealed.get(..12)?;
        let mut body = sealed.get(12..12 + body_len)?.to_vec();
        let tag = sealed.get(12 + body_len..)?;
        self.aead
            .decrypt_in_place_detached(
                Nonce::from_slice(nonce),
                AAD,
                &mut body,
                Tag::from_slice(tag),
            )
            .ok()?;
        Some(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sealed_record_hides_its_bytes_and_opens_only_intact() {
        let seal = SpillSeal::random().unwrap();
        let row = b"ZEBRA-SPILLED-ROW".to_vec();
        let a = seal.seal(&row).unwrap();
        let b = seal.seal(&row).unwrap();
        assert_eq!(a.len(), row.len() + OVERHEAD);
        assert!(!a.windows(row.len()).any(|w| w == row.as_slice()));
        assert_ne!(a, b, "every record gets its own nonce");
        assert_eq!(seal.open(&a).unwrap(), row);
        let mut bad = a.clone();
        bad[15] ^= 1;
        assert!(seal.open(&bad).is_none());
        assert!(SpillSeal::random().unwrap().open(&a).is_none());
    }
}
