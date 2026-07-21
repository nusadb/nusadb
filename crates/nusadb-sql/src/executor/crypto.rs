//! Column-value encryption for the SQL `encrypt`/`decrypt` functions
//! (column-level, in the SQL layer, **opaque to the
//! engine** so ADR 003's tuple-bytes seam stays intact).
//!
//! The cipher is AES-256-GCM-SIV (RustCrypto `aes-gcm-siv`), a
//! nonce-misuse-resistant AEAD. With a fixed nonce the ciphertext is
//! **deterministic** — the same `(key, plaintext)` always produces the same
//! bytes — yet distinct plaintexts stay secure (unlike plain GCM, where a fixed
//! nonce is catastrophic). Determinism is what lets encrypted values round-trip,
//! keeps `.slt` output stable, and keeps deterministic simulation reproducible.
//! The trade-off is the standard one for deterministic encryption: equal
//! plaintexts yield equal ciphertexts, so equality is observable.
//!
//! The 32-byte AES key is derived from the caller's key string with SHA-256.
//! Ciphertext (AEAD output = ciphertext ‖ 16-byte tag) is hex-encoded so it
//! lives inside an ordinary `TEXT` value the engine stores without
//! interpretation. `decrypt` re-checks the tag, so a wrong key or any tampering
//! surfaces as [`Error::Decryption`] rather than returning garbage.

#![allow(
    clippy::doc_markdown,
    reason = "cipher names like AES-256-GCM-SIV read better unbackticked"
)]

use aes_gcm_siv::aead::{Aead, KeyInit};
use aes_gcm_siv::{Aes256GcmSiv, Key, Nonce};
use sha2::{Digest, Sha256, Sha512};

use crate::error::Error;

/// `SHA256(text)` — the 64-character lowercase-hex SHA-256 digest of the UTF-8 bytes of `input`.
pub(super) fn sha256_hex(input: &str) -> String {
    to_hex(&Sha256::digest(input.as_bytes()))
}

/// `SHA512(text)` — the 128-character lowercase-hex SHA-512 digest of the UTF-8 bytes of `input`.
pub(super) fn sha512_hex(input: &str) -> String {
    to_hex(&Sha512::digest(input.as_bytes()))
}

/// `MD5(text)` — the 32-character lowercase-hex MD5 digest of the UTF-8 bytes of `input`.
///
/// MD5 is **cryptographically broken** and is provided only as a non-security data fingerprint
/// (ETL dedup, cache keys, checksums) — use [`sha256_hex`] for anything
/// security-relevant. The algorithm is implemented inline (RFC 1321) rather than via a crate: the
/// `md5`/`md-5` crates are banned in production by `deny.toml`, and pulling a broken-hash dependency
/// for a compatibility shim is exactly what that ban exists to prevent.
pub(super) fn md5_hex(input: &str) -> String {
    to_hex(&md5_digest(input.as_bytes()))
}

/// Per-round left-rotation amounts (RFC 1321 §3.4).
#[rustfmt::skip]
const MD5_S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// Per-round additive constants `K[i] = floor(2^32 · |sin(i + 1)|)` (RFC 1321 §3.4).
#[rustfmt::skip]
const MD5_K: [u32; 64] = [
    0xd76a_a478, 0xe8c7_b756, 0x2420_70db, 0xc1bd_ceee,
    0xf57c_0faf, 0x4787_c62a, 0xa830_4613, 0xfd46_9501,
    0x6980_98d8, 0x8b44_f7af, 0xffff_5bb1, 0x895c_d7be,
    0x6b90_1122, 0xfd98_7193, 0xa679_438e, 0x49b4_0821,
    0xf61e_2562, 0xc040_b340, 0x265e_5a51, 0xe9b6_c7aa,
    0xd62f_105d, 0x0244_1453, 0xd8a1_e681, 0xe7d3_fbc8,
    0x21e1_cde6, 0xc337_07d6, 0xf4d5_0d87, 0x455a_14ed,
    0xa9e3_e905, 0xfcef_a3f8, 0x676f_02d9, 0x8d2a_4c8a,
    0xfffa_3942, 0x8771_f681, 0x6d9d_6122, 0xfde5_380c,
    0xa4be_ea44, 0x4bde_cfa9, 0xf6bb_4b60, 0xbebf_bc70,
    0x289b_7ec6, 0xeaa1_27fa, 0xd4ef_3085, 0x0488_1d05,
    0xd9d4_d039, 0xe6db_99e5, 0x1fa2_7cf8, 0xc4ac_5665,
    0xf429_2244, 0x432a_ff97, 0xab94_23a7, 0xfc93_a039,
    0x655b_59c3, 0x8f0c_cc92, 0xffef_f47d, 0x8584_5dd1,
    0x6fa8_7e4f, 0xfe2c_e6e0, 0xa301_4314, 0x4e08_11a1,
    0xf753_7e82, 0xbd3a_f235, 0x2ad7_d2bb, 0xeb86_d391,
];

/// The 16-byte MD5 digest of `message` (RFC 1321). All arithmetic wraps mod 2^32 by definition.
#[allow(
    clippy::many_single_char_names,
    reason = "a/b/c/d are the RFC 1321 state-word names; renaming would obscure the spec mapping"
)]
fn md5_digest(message: &[u8]) -> [u8; 16] {
    let (mut a0, mut b0, mut c0, mut d0) = (
        0x6745_2301u32,
        0xefcd_ab89u32,
        0x98ba_dcfeu32,
        0x1032_5476u32,
    );

    // Pad with a 1 bit (0x80), zeros until the length is 56 mod 64, then the original message length
    // in bits as a 64-bit little-endian integer.
    let bit_len = u64::try_from(message.len())
        .unwrap_or(u64::MAX)
        .wrapping_mul(8);
    let mut padded = message.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_le_bytes());

    for block in padded.chunks_exact(64) {
        // Decode the block into sixteen little-endian 32-bit words.
        let mut m = [0u32; 16];
        for (word, bytes) in m.iter_mut().zip(block.chunks_exact(4)) {
            *word = u32::from_le_bytes(bytes.try_into().unwrap_or_default());
        }

        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for (i, (&shift, &k)) in MD5_S.iter().zip(MD5_K.iter()).enumerate() {
            let (mix, g) = if i < 16 {
                ((b & c) | (!b & d), i)
            } else if i < 32 {
                ((d & b) | (!d & c), (5 * i + 1) % 16)
            } else if i < 48 {
                (b ^ c ^ d, (3 * i + 5) % 16)
            } else {
                (c ^ (b | !d), (7 * i) % 16)
            };
            let rotated = mix
                .wrapping_add(a)
                .wrapping_add(k)
                .wrapping_add(m.get(g).copied().unwrap_or(0))
                .rotate_left(shift);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(rotated);
        }

        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    // The digest is a0‖b0‖c0‖d0, each word little-endian.
    let mut out = [0u8; 16];
    let words = a0
        .to_le_bytes()
        .into_iter()
        .chain(b0.to_le_bytes())
        .chain(c0.to_le_bytes())
        .chain(d0.to_le_bytes());
    for (slot, byte) in out.iter_mut().zip(words) {
        *slot = byte;
    }
    out
}

/// Fixed nonce. GCM-SIV is misuse-resistant, so a constant nonce gives
/// deterministic output without the key-recovery a fixed nonce causes in plain
/// GCM. The synthetic IV GCM-SIV derives from the plaintext still differs
/// between distinct plaintexts.
const NONCE: [u8; 12] = [0u8; 12];

/// Build the AES-256-GCM-SIV cipher for `key`, deriving the 32-byte key with
/// SHA-256 of the key string.
fn cipher(key: &str) -> Aes256GcmSiv {
    let digest = Sha256::digest(key.as_bytes());
    Aes256GcmSiv::new(Key::<Aes256GcmSiv>::from_slice(&digest))
}

/// Encrypt `plaintext` under `key`, returning lowercase-hex ciphertext.
pub(super) fn encrypt(plaintext: &str, key: &str) -> Result<String, Error> {
    let sealed = cipher(key)
        .encrypt(Nonce::from_slice(&NONCE), plaintext.as_bytes())
        .map_err(|_| Error::Decryption("encryption failed"))?;
    Ok(to_hex(&sealed))
}

/// Decrypt hex ciphertext produced by [`encrypt`] under `key`. A wrong key or
/// tampered ciphertext fails the AEAD tag check → [`Error::Decryption`].
pub(super) fn decrypt(ciphertext_hex: &str, key: &str) -> Result<String, Error> {
    let sealed =
        from_hex(ciphertext_hex).ok_or(Error::Decryption("ciphertext is not valid hex"))?;
    let plaintext = cipher(key)
        .decrypt(Nonce::from_slice(&NONCE), sealed.as_slice())
        .map_err(|_| Error::Decryption("wrong key or tampered ciphertext"))?;
    String::from_utf8(plaintext).map_err(|_| Error::Decryption("plaintext is not valid UTF-8"))
}

/// Lowercase-hex encode.
fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        // `b >> 4` and `b & 0x0f` are always `< 16`, so `from_digit` succeeds.
        out.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
        out.push(char::from_digit(u32::from(b & 0x0f), 16).unwrap_or('0'));
    }
    out
}

/// Decode lowercase/uppercase hex; `None` on odd length or a non-hex digit.
fn from_hex(s: &str) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let [hi, lo] = pair else { return None };
        out.push((hex_val(*hi)? << 4) | hex_val(*lo)?);
    }
    Some(out)
}

const fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{decrypt, encrypt, md5_hex};

    #[test]
    fn md5_matches_rfc1321_vectors() {
        // The RFC 1321 §A.5 test suite.
        assert_eq!(md5_hex(""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex("a"), "0cc175b9c0f1b6a831c399e269772661");
        assert_eq!(md5_hex("abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            md5_hex("message digest"),
            "f96b697d7cb7938d525a2f31aaf161d0"
        );
        assert_eq!(
            md5_hex("abcdefghijklmnopqrstuvwxyz"),
            "c3fcd3d76192e4007dfb496cca67e13b"
        );
        assert_eq!(
            md5_hex("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
            "d174ab98d277d9f5a5611c2c9f419d9f"
        );
    }

    #[test]
    fn md5_spans_multiple_blocks() {
        // 80 bytes forces a second 64-byte block plus a padding-only block.
        assert_eq!(
            md5_hex(
                "12345678901234567890123456789012345678901234567890123456789012345678901234567890"
            ),
            "57edf4a22be3c955ac49da2e2107b67a"
        );
    }

    #[test]
    fn round_trip_recovers_plaintext() {
        let ct = encrypt("ssn-123-45-6789", "secret-key").unwrap();
        assert_eq!(decrypt(&ct, "secret-key").unwrap(), "ssn-123-45-6789");
    }

    #[test]
    fn ciphertext_is_deterministic() {
        let a = encrypt("hello", "k").unwrap();
        let b = encrypt("hello", "k").unwrap();
        assert_eq!(a, b);
        // ...and is not the plaintext.
        assert_ne!(a, "hello");
    }

    #[test]
    fn distinct_plaintexts_differ_under_same_key() {
        assert_ne!(encrypt("a", "k").unwrap(), encrypt("b", "k").unwrap());
    }

    #[test]
    fn wrong_key_is_rejected() {
        let ct = encrypt("secret", "right-key").unwrap();
        assert!(decrypt(&ct, "wrong-key").is_err());
    }

    #[test]
    fn tampered_ciphertext_is_rejected() {
        let mut ct = encrypt("secret", "k").unwrap();
        // Flip the last hex nibble.
        let last = ct.pop().unwrap();
        ct.push(if last == 'a' { 'b' } else { 'a' });
        assert!(decrypt(&ct, "k").is_err());
    }

    #[test]
    fn non_hex_ciphertext_is_rejected() {
        assert!(decrypt("not hex!", "k").is_err());
    }

    #[test]
    fn empty_plaintext_round_trips() {
        let ct = encrypt("", "k").unwrap();
        assert_eq!(decrypt(&ct, "k").unwrap(), "");
    }
}
