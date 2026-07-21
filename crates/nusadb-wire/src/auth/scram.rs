//! SCRAM-SHA-256 message layer (RFC 5802 / RFC 7677) — the `client-first` / `server-first`
//! exchange.
//!
//! This module is the wire-format half of the handshake: it parses the client's first
//! message (GS2 header, username, client nonce) and builds the server's first message
//! (combined nonce, base64 salt, iteration count). The cryptographic proof exchange
//! (`client-final` / `server-final`, PBKDF2 + HMAC + constant-time verify) is
//!
//! The functions are deliberately string-in / string-out and hold no I/O or catalog state,
//! so they are exercised directly against the RFC test vectors. Server nonces come from a
//! CSPRNG ([`generate_nonce`]); everything else is deterministic and so is testable with a
//! fixed nonce, exactly as the RFC examples present them.

use core::num::NonZeroU32;

use ring::rand::{SecureRandom, SystemRandom};
use ring::{digest, hmac, pbkdf2};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

use super::base64;

/// Length of a SHA-256 output / HMAC-SHA-256 tag in bytes.
const SHA256_LEN: usize = 32;

/// Bytes of CSPRNG entropy behind each server nonce. 18 bytes → 24 base64 characters with no
/// padding, comfortably above the RFC's "at least 16 bytes" guidance for the combined nonce.
const SERVER_NONCE_BYTES: usize = 18;

/// An error processing a SCRAM-SHA-256 message.
///
/// `Malformed` deliberately does not echo the offending bytes: a SCRAM message carries the
/// username and nonces, and reflecting it into an error string risks leaking it into a log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ScramError {
    /// A message did not conform to the SCRAM grammar (bad framing, attribute, or base64).
    #[error("malformed SCRAM message")]
    Malformed,
    /// A SCRAM extension (e.g. mandatory `m=`) the server does not implement.
    #[error("unsupported SCRAM extension")]
    UnsupportedExtension,
    /// A nonce was empty or contained a byte outside the printable, comma-free range.
    #[error("invalid SCRAM nonce")]
    InvalidNonce,
    /// The CSPRNG failed to produce a server nonce.
    #[error("SCRAM nonce generation failed")]
    Rng,
    /// The client proof did not verify against the stored credentials.
    #[error("SCRAM authentication failed")]
    AuthenticationFailed,
}

/// The GS2 channel-binding flag from a `client-first-message` (RFC 5802 §5.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelBinding {
    /// `n` — the client does not support channel binding.
    NotSupported,
    /// `y` — the client supports channel binding but believes the server does not.
    SupportedNotUsed,
    /// `p=<name>` — the client requires channel binding of the named type.
    Required(String),
}

/// A parsed `client-first-message` (RFC 5802 §7).
///
/// `bare` and `gs2_header` are retained verbatim because the proof step needs the
/// `client-first-message-bare` for the `AuthMessage` and the GS2 header to validate the
/// client's `c=` channel-binding echo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientFirst {
    /// The GS2 channel-binding flag.
    pub channel_binding: ChannelBinding,
    /// The optional SASL authorization identity (`a=`), unescaped. Usually `None`.
    pub authzid: Option<String>,
    /// The SCRAM username (`n=`), with `=2C` / `=3D` unescaped.
    pub username: String,
    /// The client nonce (`r=`).
    pub client_nonce: String,
    /// The `client-first-message-bare` (`n=...,r=...`), verbatim.
    pub bare: String,
    /// The GS2 header (`<cbind-flag>,<authzid>,`), verbatim.
    pub gs2_header: String,
}

impl ClientFirst {
    /// Parse a `client-first-message`, e.g. `n,,n=user,r=fyko+d2lbbFgONRv9qkxdawL`.
    ///
    /// # Errors
    /// [`ScramError::Malformed`] for bad framing or attribute order, [`ScramError::InvalidNonce`]
    /// for a malformed nonce, [`ScramError::UnsupportedExtension`] for a mandatory extension.
    pub fn parse(message: &str) -> Result<Self, ScramError> {
        // gs2-header = gs2-cbind-flag "," [ authzid ] ","
        let (cbind_flag, after_flag) = message.split_once(',').ok_or(ScramError::Malformed)?;
        let (authzid_field, bare) = after_flag.split_once(',').ok_or(ScramError::Malformed)?;
        let gs2_header = format!("{cbind_flag},{authzid_field},");

        let channel_binding = parse_cbind_flag(cbind_flag)?;
        let authzid = match authzid_field {
            "" => None,
            field => Some(unescape_saslname(
                field.strip_prefix("a=").ok_or(ScramError::Malformed)?,
            )?),
        };

        // client-first-message-bare = [reserved-mext ","] username "," nonce ["," extensions]
        let mut fields = bare.split(',');
        let username_field = fields.next().ok_or(ScramError::Malformed)?;
        if username_field.starts_with("m=") {
            return Err(ScramError::UnsupportedExtension);
        }
        let username = unescape_saslname(
            username_field
                .strip_prefix("n=")
                .ok_or(ScramError::Malformed)?,
        )?;

        let nonce_field = fields.next().ok_or(ScramError::Malformed)?;
        let client_nonce = nonce_field
            .strip_prefix("r=")
            .ok_or(ScramError::Malformed)?;
        validate_nonce(client_nonce)?;

        Ok(Self {
            channel_binding,
            authzid,
            username,
            client_nonce: client_nonce.to_owned(),
            bare: bare.to_owned(),
            gs2_header,
        })
    }
}

/// A `server-first-message` (RFC 5802 §7): the combined nonce, the user's salt, and the
/// iteration count the client must use to derive the salted password.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerFirst {
    /// The combined nonce: the client nonce with the server nonce appended.
    pub combined_nonce: String,
    /// The user's salt (raw bytes; base64-encoded on the wire).
    pub salt: Vec<u8>,
    /// The PBKDF2 iteration count.
    pub iterations: u32,
}

impl ServerFirst {
    /// Build a `server-first-message` by appending a freshly generated `server_nonce` to the
    /// client nonce from `client_first`.
    pub fn build(
        client_first: &ClientFirst,
        salt: Vec<u8>,
        iterations: u32,
        server_nonce: &str,
    ) -> Self {
        Self {
            combined_nonce: format!("{}{}", client_first.client_nonce, server_nonce),
            salt,
            iterations,
        }
    }

    /// Render the wire form: `r=<combined-nonce>,s=<base64-salt>,i=<iterations>`.
    pub fn to_message(&self) -> String {
        format!(
            "r={},s={},i={}",
            self.combined_nonce,
            base64::encode(&self.salt),
            self.iterations
        )
    }

    /// Parse the wire form produced by [`to_message`](Self::to_message). Used by tests and by
    /// any client-side driver.
    ///
    /// # Errors
    /// [`ScramError::Malformed`] for bad framing, base64, or iteration count;
    /// [`ScramError::InvalidNonce`] for a malformed combined nonce.
    pub fn parse(message: &str) -> Result<Self, ScramError> {
        let mut fields = message.split(',');
        let combined_nonce = fields
            .next()
            .and_then(|f| f.strip_prefix("r="))
            .ok_or(ScramError::Malformed)?;
        validate_nonce(combined_nonce)?;
        let salt_b64 = fields
            .next()
            .and_then(|f| f.strip_prefix("s="))
            .ok_or(ScramError::Malformed)?;
        let iterations: u32 = fields
            .next()
            .and_then(|f| f.strip_prefix("i="))
            .ok_or(ScramError::Malformed)?
            .parse()
            .map_err(|_| ScramError::Malformed)?;
        Ok(Self {
            combined_nonce: combined_nonce.to_owned(),
            salt: base64::decode(salt_b64)?,
            iterations,
        })
    }
}

/// Generate a fresh server nonce: 18 bytes of CSPRNG entropy, base64-encoded.
///
/// The base64 alphabet is a subset of the printable, comma-free `c-nonce` character set, so
/// the result is always a valid SCRAM nonce.
///
/// # Errors
/// [`ScramError::Rng`] if the system CSPRNG is unavailable.
pub fn generate_nonce() -> Result<String, ScramError> {
    let rng = SystemRandom::new();
    let mut buf = [0u8; SERVER_NONCE_BYTES];
    rng.fill(&mut buf).map_err(|_| ScramError::Rng)?;
    Ok(base64::encode(&buf))
}

/// Generate a fresh 16-byte per-user salt (CSPRNG) for [`derive_credentials`].
///
/// # Errors
/// [`ScramError::Rng`] if the system CSPRNG fails.
pub fn generate_salt() -> Result<Vec<u8>, ScramError> {
    let rng = SystemRandom::new();
    let mut buf = [0u8; 16];
    rng.fill(&mut buf).map_err(|_| ScramError::Rng)?;
    Ok(buf.to_vec())
}

/// Parse a GS2 channel-binding flag: `n`, `y`, or `p=<name>`.
fn parse_cbind_flag(flag: &str) -> Result<ChannelBinding, ScramError> {
    match flag {
        "n" => Ok(ChannelBinding::NotSupported),
        "y" => Ok(ChannelBinding::SupportedNotUsed),
        other => match other.strip_prefix("p=") {
            Some(name) if !name.is_empty() => Ok(ChannelBinding::Required(name.to_owned())),
            _ => Err(ScramError::Malformed),
        },
    }
}

/// Unescape a SCRAM `saslname`: `=2C` → `,` and `=3D` → `=`. A bare `,` or any other `=`
/// sequence is malformed.
fn unescape_saslname(value: &str) -> Result<String, ScramError> {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        match c {
            '=' => match (chars.next(), chars.next()) {
                (Some('2'), Some('C')) => out.push(','),
                (Some('3'), Some('D')) => out.push('='),
                _ => return Err(ScramError::Malformed),
            },
            ',' => return Err(ScramError::Malformed),
            other => out.push(other),
        }
    }
    Ok(out)
}

/// Validate a SCRAM nonce: non-empty and every byte printable ASCII (`0x21..=0x7E`) except the
/// comma (`0x2C`), per the `c-nonce` / `s-nonce` production.
fn validate_nonce(nonce: &str) -> Result<(), ScramError> {
    if !nonce.is_empty()
        && nonce
            .bytes()
            .all(|b| (0x21..=0x7e).contains(&b) && b != b',')
    {
        Ok(())
    } else {
        Err(ScramError::InvalidNonce)
    }
}

// === client-final / server-final proof exchange ==================

/// A parsed `client-final-message` (RFC 5802 §7): `c=<cbind>,r=<nonce>,p=<proof>`.
///
/// `without_proof` retains the `c=…,r=…` prefix verbatim because it is the
/// `client-final-message-without-proof` term of the `AuthMessage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientFinal {
    /// The decoded `c=` channel-binding data (the GS2 header bytes the client echoes).
    pub channel_binding: Vec<u8>,
    /// The `r=` combined nonce (must equal the server's `combined_nonce`).
    pub combined_nonce: String,
    /// The decoded `p=` client proof (32 bytes for SHA-256).
    pub proof: Vec<u8>,
    /// `client-final-message-without-proof` (`c=…,r=…`), for the `AuthMessage`.
    pub without_proof: String,
}

impl ClientFinal {
    /// Parse a `client-final-message`. `p=` is the last attribute; any field between `r=` and
    /// `p=` is a SCRAM extension the server does not implement and is rejected.
    ///
    /// # Errors
    /// [`ScramError::Malformed`] for bad framing or base64; [`ScramError::InvalidNonce`] for a
    /// malformed nonce; [`ScramError::UnsupportedExtension`] for an extension field.
    pub fn parse(message: &str) -> Result<Self, ScramError> {
        let (without_proof, proof_b64) = message.rsplit_once(",p=").ok_or(ScramError::Malformed)?;
        let proof = base64::decode(proof_b64)?;
        let mut fields = without_proof.split(',');
        let cbind_b64 = fields
            .next()
            .and_then(|f| f.strip_prefix("c="))
            .ok_or(ScramError::Malformed)?;
        let channel_binding = base64::decode(cbind_b64)?;
        let combined_nonce = fields
            .next()
            .and_then(|f| f.strip_prefix("r="))
            .ok_or(ScramError::Malformed)?;
        validate_nonce(combined_nonce)?;
        if fields.next().is_some() {
            return Err(ScramError::UnsupportedExtension);
        }
        Ok(Self {
            channel_binding,
            combined_nonce: combined_nonce.to_owned(),
            proof,
            without_proof: without_proof.to_owned(),
        })
    }
}

/// Server-side stored SCRAM credentials (RFC 5802 §3) — what the catalog persists per user,
/// never the plaintext password. Derived once via [`derive_credentials`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredCredentials {
    /// `StoredKey = H(ClientKey)` — used to verify the client proof.
    pub stored_key: [u8; SHA256_LEN],
    /// `ServerKey = HMAC(SaltedPassword, "Server Key")` — signs the server-final message.
    pub server_key: [u8; SHA256_LEN],
    /// The per-user salt (base64-encoded on the wire in `server-first`).
    pub salt: Vec<u8>,
    /// The PBKDF2 iteration count.
    pub iterations: u32,
}

/// Compute one HMAC-SHA-256 tag as a fixed 32-byte array.
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; SHA256_LEN] {
    let tag = hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, key), data);
    let mut out = [0u8; SHA256_LEN];
    out.copy_from_slice(tag.as_ref());
    out
}

/// Derive [`StoredCredentials`] from a plaintext password (RFC 5802 §3).
///
/// `SaltedPassword = PBKDF2-HMAC-SHA256(password, salt, iterations)`, then
/// `StoredKey = H(HMAC(SaltedPassword, "Client Key"))` and
/// `ServerKey = HMAC(SaltedPassword, "Server Key")`. Called once at user-creation
/// time; only the result (not the password) is stored.
///
/// # Errors
/// [`ScramError::Malformed`] if `iterations` is zero.
pub fn derive_credentials(
    password: &str,
    salt: Vec<u8>,
    iterations: u32,
) -> Result<StoredCredentials, ScramError> {
    let iters = NonZeroU32::new(iterations).ok_or(ScramError::Malformed)?;
    // `salted` (SaltedPassword) and `client_key` are secret-derived; zeroize on drop so they do
    // not linger on the stack after the keys are computed.
    let mut salted = Zeroizing::new([0u8; SHA256_LEN]);
    pbkdf2::derive(
        pbkdf2::PBKDF2_HMAC_SHA256,
        iters,
        &salt,
        password.as_bytes(),
        salted.as_mut(),
    );
    let client_key = Zeroizing::new(hmac_sha256(&*salted, b"Client Key"));
    let mut stored_key = [0u8; SHA256_LEN];
    stored_key.copy_from_slice(digest::digest(&digest::SHA256, &*client_key).as_ref());
    let server_key = hmac_sha256(&*salted, b"Server Key");
    Ok(StoredCredentials {
        stored_key,
        server_key,
        salt,
        iterations,
    })
}

/// Compute the client proof `ClientProof = ClientKey XOR HMAC(StoredKey, AuthMessage)` (RFC 5802
/// §3, client side). Used by a client to build its `client-final` message.
///
/// # Errors
/// [`ScramError::Malformed`] if `iterations` is zero.
pub fn client_proof(
    password: &str,
    salt: &[u8],
    iterations: u32,
    auth_message: &str,
) -> Result<Vec<u8>, ScramError> {
    let iters = NonZeroU32::new(iterations).ok_or(ScramError::Malformed)?;
    let mut salted = Zeroizing::new([0u8; SHA256_LEN]);
    pbkdf2::derive(
        pbkdf2::PBKDF2_HMAC_SHA256,
        iters,
        salt,
        password.as_bytes(),
        salted.as_mut(),
    );
    let client_key = Zeroizing::new(hmac_sha256(&*salted, b"Client Key"));
    let mut stored_key = [0u8; SHA256_LEN];
    stored_key.copy_from_slice(digest::digest(&digest::SHA256, &*client_key).as_ref());
    let client_signature = hmac_sha256(&stored_key, auth_message.as_bytes());
    Ok(client_key
        .iter()
        .zip(client_signature.iter())
        .map(|(k, s)| k ^ s)
        .collect())
}

/// Build a complete `client-final-message` (with proof) for a SCRAM exchange (client side).
///
/// The result is `c=<base64 gs2-header>,r=<combined-nonce>,p=<base64 proof>`; the client builds it
/// after parsing the `server-first` message.
///
/// # Errors
/// [`ScramError::Malformed`] if the server's iteration count is zero.
pub fn client_final_message(
    password: &str,
    gs2_header: &str,
    client_first_bare: &str,
    server_first_msg: &str,
    server_first: &ServerFirst,
) -> Result<String, ScramError> {
    let channel_binding = base64::encode(gs2_header.as_bytes());
    let without_proof = format!("c={channel_binding},r={}", server_first.combined_nonce);
    let auth_msg = auth_message(client_first_bare, server_first_msg, &without_proof);
    let proof = client_proof(
        password,
        &server_first.salt,
        server_first.iterations,
        &auth_msg,
    )?;
    Ok(format!("{without_proof},p={}", base64::encode(&proof)))
}

/// Build the `AuthMessage` (RFC 5802 §3): `client-first-bare + "," + server-first + "," +
/// client-final-without-proof`. All three terms are the verbatim wire messages.
#[must_use]
pub fn auth_message(
    client_first_bare: &str,
    server_first: &str,
    client_final_without_proof: &str,
) -> String {
    format!("{client_first_bare},{server_first},{client_final_without_proof}")
}

/// Verify a client proof against stored credentials and return the `server-final-message`
/// (`v=<base64 ServerSignature>`) on success.
///
/// Recovers `ClientKey = ClientProof XOR HMAC(StoredKey, AuthMessage)`, then checks
/// `H(ClientKey) == StoredKey` in **constant time** (no early-exit timing leak). On success the
/// `ServerSignature = HMAC(ServerKey, AuthMessage)` is returned for the client to verify.
///
/// # Errors
/// [`ScramError::AuthenticationFailed`] if the proof is the wrong length or does not verify.
pub fn verify_client_proof(
    creds: &StoredCredentials,
    auth_message: &str,
    proof: &[u8],
) -> Result<String, ScramError> {
    if proof.len() != SHA256_LEN {
        return Err(ScramError::AuthenticationFailed);
    }
    let client_signature = hmac_sha256(&creds.stored_key, auth_message.as_bytes());
    // recovered ClientKey = ClientProof XOR ClientSignature; secret, so zeroize on drop.
    let mut client_key = Zeroizing::new(client_signature);
    for (k, p) in client_key.iter_mut().zip(proof) {
        *k ^= p;
    }
    let computed_stored_key = digest::digest(&digest::SHA256, &*client_key);
    // Constant-time compare: a timing-variable check would leak how many leading bytes matched.
    if computed_stored_key
        .as_ref()
        .ct_eq(&creds.stored_key)
        .unwrap_u8()
        != 1
    {
        return Err(ScramError::AuthenticationFailed);
    }
    let server_signature = hmac_sha256(&creds.server_key, auth_message.as_bytes());
    Ok(format!("v={}", base64::encode(&server_signature)))
}

/// Verify the server's `server-final-message` on the client side (client, RFC 5802 §3).
///
/// This is the mutual-authentication half: recompute `ServerSignature = HMAC(ServerKey,
/// AuthMessage)` from the password and the server-supplied `salt`/`iterations`, then compare it
/// against the server's `v=<base64>` claim in constant time. A match proves the server actually
/// holds the user's credentials (defeats a rogue/relaying server), not merely that the client's
/// proof was accepted.
///
/// # Errors
/// [`ScramError::Malformed`] if `server_final` is not a `v=<base64>` field or `iterations` is zero;
/// [`ScramError::AuthenticationFailed`] if the signature does not match.
pub fn verify_server_signature(
    password: &str,
    salt: &[u8],
    iterations: u32,
    auth_message: &str,
    server_final: &str,
) -> Result<(), ScramError> {
    let claimed = server_final
        .strip_prefix("v=")
        .ok_or(ScramError::Malformed)?;
    let claimed = base64::decode(claimed)?;
    let iters = NonZeroU32::new(iterations).ok_or(ScramError::Malformed)?;
    let mut salted = Zeroizing::new([0u8; SHA256_LEN]);
    pbkdf2::derive(
        pbkdf2::PBKDF2_HMAC_SHA256,
        iters,
        salt,
        password.as_bytes(),
        salted.as_mut(),
    );
    let server_key = hmac_sha256(&*salted, b"Server Key");
    let expected = hmac_sha256(&server_key, auth_message.as_bytes());
    // Constant-time compare (subtle's slice `ct_eq` is `0` on a length mismatch, also in constant
    // time), so neither the bytes nor the length of the recomputed signature leak.
    if expected[..].ct_eq(&claimed).unwrap_u8() != 1 {
        return Err(ScramError::AuthenticationFailed);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ChannelBinding, ClientFinal, ClientFirst, ScramError, ServerFirst, auth_message, base64,
        derive_credentials, generate_nonce, verify_client_proof, verify_server_signature,
    };

    #[test]
    fn parses_rfc7677_client_first() {
        // RFC 7677 §3 worked example.
        let cf = ClientFirst::parse("n,,n=user,r=rOprNGfwEbeRWgbNEkqO").unwrap();
        assert_eq!(cf.channel_binding, ChannelBinding::NotSupported);
        assert_eq!(cf.authzid, None);
        assert_eq!(cf.username, "user");
        assert_eq!(cf.client_nonce, "rOprNGfwEbeRWgbNEkqO");
        assert_eq!(cf.bare, "n=user,r=rOprNGfwEbeRWgbNEkqO");
        assert_eq!(cf.gs2_header, "n,,");
    }

    #[test]
    fn parses_channel_binding_flags() {
        assert_eq!(
            ClientFirst::parse("y,,n=u,r=abc").unwrap().channel_binding,
            ChannelBinding::SupportedNotUsed
        );
        assert_eq!(
            ClientFirst::parse("p=tls-server-end-point,,n=u,r=abc")
                .unwrap()
                .channel_binding,
            ChannelBinding::Required("tls-server-end-point".to_owned())
        );
    }

    #[test]
    fn parses_authzid_and_unescapes_username() {
        let cf = ClientFirst::parse("n,a=admin,n=a=2Cb=3Dc,r=xyz").unwrap();
        assert_eq!(cf.authzid.as_deref(), Some("admin"));
        assert_eq!(cf.username, "a,b=c");
        // The bare message keeps the on-wire (escaped) username.
        assert_eq!(cf.bare, "n=a=2Cb=3Dc,r=xyz");
    }

    #[test]
    fn rejects_malformed_client_first() {
        for bad in [
            "n,,n=user",            // missing nonce
            "n,,r=abc,n=user",      // wrong attribute order
            "n,n=user,r=abc",       // authzid field not "a=..."
            "x,,n=user,r=abc",      // bad cbind flag
            "p=,,n=user,r=abc",     // empty channel-binding name
            "n,,m=ext,n=user,r=ab", // mandatory extension
            "n,,n=user,r=",         // empty nonce
            "n,,n=us=XYer,r=abc",   // bad saslname escape
        ] {
            assert!(ClientFirst::parse(bad).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn mandatory_extension_is_distinct_error() {
        assert_eq!(
            ClientFirst::parse("n,,m=ext,n=user,r=abc"),
            Err(ScramError::UnsupportedExtension)
        );
    }

    #[test]
    fn builds_server_first_with_appended_nonce() {
        let cf = ClientFirst::parse("n,,n=user,r=rOprNGfwEbeRWgbNEkqO").unwrap();
        // RFC 7677 §3: salt "==", server nonce suffix, i=4096.
        let salt = super::base64::decode("W22ZaJ0SNY7soEsUEjb6gQ==").unwrap();
        let sf = ServerFirst::build(&cf, salt, 4096, "%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0");
        assert_eq!(
            sf.to_message(),
            "r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,\
             s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4096"
        );
        assert!(sf.combined_nonce.starts_with(&cf.client_nonce));
    }

    #[test]
    fn server_first_round_trips() {
        let cf = ClientFirst::parse("n,,n=user,r=clientNONCE").unwrap();
        let sf = ServerFirst::build(&cf, vec![1, 2, 3, 4, 5, 6, 7, 8], 8192, "serverNONCE");
        let reparsed = ServerFirst::parse(&sf.to_message()).unwrap();
        assert_eq!(reparsed, sf);
    }

    #[test]
    fn server_first_parse_rejects_garbage() {
        for bad in [
            "",
            "x=1,s=AAAA,i=1",
            "r=abc,s=AAAA",
            "r=abc,s=AAAA,i=notnum",
        ] {
            assert!(ServerFirst::parse(bad).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn generated_nonces_are_valid_and_unique() {
        let a = generate_nonce().unwrap();
        let b = generate_nonce().unwrap();
        assert_ne!(a, b, "CSPRNG nonces must not repeat");
        // A generated nonce must itself round-trip through client-first parsing.
        let msg = format!("n,,n=user,r={a}");
        assert_eq!(ClientFirst::parse(&msg).unwrap().client_nonce, a);
    }

    // --- client-final / proof exchange ---------------------------

    /// The RFC 7677 §3 worked example, reused across the proof tests.
    const CLIENT_FIRST_BARE: &str = "n=user,r=rOprNGfwEbeRWgbNEkqO";
    const SERVER_FIRST: &str = "r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,\
                                s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4096";
    const CLIENT_FINAL: &str = "c=biws,r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,\
                                p=dHzbZapWIk4jUhN+Ute9ytag9zjfMHgsqmmiz7AndVQ=";

    #[test]
    fn client_final_parses_rfc7677() {
        let cf = ClientFinal::parse(CLIENT_FINAL).unwrap();
        assert_eq!(cf.channel_binding, b"n,,"); // base64("n,,") = "biws"
        assert_eq!(
            cf.combined_nonce,
            "rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0"
        );
        assert_eq!(cf.proof.len(), 32);
        assert_eq!(
            cf.without_proof,
            "c=biws,r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0"
        );
    }

    #[test]
    fn client_final_rejects_malformed() {
        for bad in [
            "c=biws,r=abc",            // no proof
            "r=abc,p=AAAA",            // missing channel binding
            "c=biws,p=AAAA",           // missing nonce
            "c=biws,r=abc,x=1,p=AAAA", // unsupported extension before proof
        ] {
            assert!(ClientFinal::parse(bad).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn verify_proof_accepts_rfc7677_and_returns_server_signature() {
        // Derive credentials from the RFC 7677 password + salt, then verify the example proof.
        let salt = base64::decode("W22ZaJ0SNY7soEsUEjb6gQ==").unwrap();
        let creds = derive_credentials("pencil", salt, 4096).unwrap();
        let cf = ClientFinal::parse(CLIENT_FINAL).unwrap();
        let msg = auth_message(CLIENT_FIRST_BARE, SERVER_FIRST, &cf.without_proof);
        let server_final = verify_client_proof(&creds, &msg, &cf.proof).unwrap();
        // RFC 7677 §3 server-final-message.
        assert_eq!(
            server_final,
            "v=6rriTRBi23WpRR/wtup+mMhUZUn/dB5nLTJRsjl95G4="
        );
    }

    #[test]
    fn verify_proof_rejects_wrong_proof_and_bad_length() {
        let salt = base64::decode("W22ZaJ0SNY7soEsUEjb6gQ==").unwrap();
        let creds = derive_credentials("pencil", salt, 4096).unwrap();
        let cf = ClientFinal::parse(CLIENT_FINAL).unwrap();
        let msg = auth_message(CLIENT_FIRST_BARE, SERVER_FIRST, &cf.without_proof);
        // Flip one proof byte → must fail authentication.
        let mut tampered = cf.proof.clone();
        tampered[0] ^= 0x01;
        assert_eq!(
            verify_client_proof(&creds, &msg, &tampered),
            Err(ScramError::AuthenticationFailed)
        );
        // Wrong-length proof → rejected without panicking.
        assert_eq!(
            verify_client_proof(&creds, &msg, &[0u8; 16]),
            Err(ScramError::AuthenticationFailed)
        );
        // Wrong password derives different credentials → proof fails.
        let salt2 = base64::decode("W22ZaJ0SNY7soEsUEjb6gQ==").unwrap();
        let wrong = derive_credentials("wrong", salt2, 4096).unwrap();
        assert_eq!(
            verify_client_proof(&wrong, &msg, &cf.proof),
            Err(ScramError::AuthenticationFailed)
        );
    }

    #[test]
    fn client_verifies_genuine_server_signature_and_rejects_a_forged_one() {
        // RFC 7677 §3: the client recomputes the server signature from the password it knows.
        let salt = base64::decode("W22ZaJ0SNY7soEsUEjb6gQ==").unwrap();
        let cf = ClientFinal::parse(CLIENT_FINAL).unwrap();
        let msg = auth_message(CLIENT_FIRST_BARE, SERVER_FIRST, &cf.without_proof);
        let genuine = "v=6rriTRBi23WpRR/wtup+mMhUZUn/dB5nLTJRsjl95G4=";
        assert!(verify_server_signature("pencil", &salt, 4096, &msg, genuine).is_ok());
        // A forged signature (one byte flipped in the decoded value) must be rejected.
        let forged = "v=7rriTRBi23WpRR/wtup+mMhUZUn/dB5nLTJRsjl95G4=";
        assert_eq!(
            verify_server_signature("pencil", &salt, 4096, &msg, forged),
            Err(ScramError::AuthenticationFailed)
        );
        // The right password against the wrong server-final, and a malformed field, both fail.
        assert_eq!(
            verify_server_signature("wrong-pw", &salt, 4096, &msg, genuine),
            Err(ScramError::AuthenticationFailed)
        );
        assert_eq!(
            verify_server_signature("pencil", &salt, 4096, &msg, "no-v-prefix"),
            Err(ScramError::Malformed)
        );
        assert_eq!(
            verify_server_signature("pencil", &salt, 0, &msg, genuine),
            Err(ScramError::Malformed)
        );
    }

    #[test]
    fn derive_credentials_rejects_zero_iterations() {
        assert_eq!(
            derive_credentials("pencil", vec![1, 2, 3], 0),
            Err(ScramError::Malformed)
        );
    }
}
