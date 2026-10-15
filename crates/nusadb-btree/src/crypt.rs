//! Encryption at rest: the key hierarchy and what is sealed with it.
//!
//! Two keys are involved:
//!
//! - The **master key** ([`MasterKey`]) is supplied by the operator: 32 random bytes, written as
//!   64 hex digits in a key file (or an environment variable). It never touches the data
//!   directory. Its only use is to wrap data keys.
//! - Each database has a **data key** ([`DataKey`]): 32 random bytes generated when an encrypted
//!   database is created, stored beside its log in the key envelope `<wal>.key`, sealed under the
//!   master key. Everything the database writes (the log, the checkpoint image's page directory
//!   and records, the page segments) is sealed under it with AES-256-GCM-SIV (see
//!   [`nusadb_wal::crypto`]). Copies of those files (the checkpoint archive, backups, a
//!   standby's seed) carry a copy of the envelope, so they open with the same master key.
//!
//! Changing the master key re-seals the data key in every envelope ([`rewrap`]); the data itself
//! is not rewritten.
//!
//! # The envelope
//!
//! ```text
//! nusadb data key 1
//! id <32 hex digits>         random, names the data key; archive and database must agree
//! key <16 hex digits>        check value of the master key that sealed it
//! sealed <hex>               nonce, the data key sealed under the master key, tag
//! ```
//!
//! The check value lets a wrong master key be reported as such, before any decryption is tried.
//! It is a truncated HMAC-SHA-256 under the master key of a domain string and the envelope's id,
//! so it reveals nothing usable about the key, and two installations under the same master key
//! (different ids) store unrelated values. The fingerprint printed for an operator
//! ([`MasterKey::fingerprint`]) is a different value that is never stored. A copy in progress
//! records [`MasterKey::copy_check`], another truncated HMAC under the key with its own domain
//! string, to refuse being finished under another key.
//!
//! # What each sealed string is bound to (its AAD)
//!
//! | Sealed | AAD |
//! | --- | --- |
//! | log record (live log, archived and shipped segments) | domain 1, 0, its log position |
//! | checkpoint image record | domain 2, the position the image covers, its position in the image |
//! | page in a segment | domain 3, page id, slot, segment name |
//! | image page directory | domain 4, the image header, its segment names |
//! | page in the spill file | domain 5, page id, write generation (with a key that never leaves the process) |
//! | data key in an envelope | the envelope header line and the data key's id |
//!
//! A database created empty gets a data key of its own, so nothing sealed for one such database
//! opens in another. A database restored from a checkpoint archive and a standby seeded from one
//! keep the archive's data key (they continue its history and write into it), so between a
//! database, its restores and its standbys a page or record is bound to its place but not to the
//! copy it belongs to: one of them can be passed off as another, much as an older copy of the
//! files can be put back.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nusadb_core::traits::Page;
use nusadb_core::{Error, PAGE_SIZE, Result};
use nusadb_wal::crypto::{DOMAIN_IMAGE, DOMAIN_LOG, KEY_LEN, SEAL_OVERHEAD, random_bytes};
use nusadb_wal::{RecordSeal, Sealer};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

/// The bytes one sealed page takes in a segment or the spill file.
pub const SEALED_PAGE_SIZE: usize = PAGE_SIZE + SEAL_OVERHEAD;

/// AAD domain of a page in a segment.
const DOMAIN_PAGE: u8 = 3;
/// AAD domain of a checkpoint image's page directory.
pub(crate) const DOMAIN_DIRECTORY: u8 = 4;
/// AAD domain of a page in the spill file.
const DOMAIN_SPILL: u8 = 5;

/// The first line of an envelope.
const ENVELOPE_HEADER: &str = "nusadb data key 1";

fn invalid(message: String) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message,
    ))
}

fn refused(message: String) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        message,
    ))
}

/// Lowercase hex of `bytes`.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// The bytes of a hex string, or `None` if it is not one.
fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digit = |c: u8| (c as char).to_digit(16);
            u8::try_from(digit(*pair.first()?)? * 16 + digit(*pair.get(1)?)?).ok()
        })
        .collect()
}

/// The operator's master key: 32 bytes that seal and unseal the data keys of every database.
pub struct MasterKey {
    key: Zeroizing<[u8; KEY_LEN]>,
}

impl std::fmt::Debug for MasterKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MasterKey")
            .field("fingerprint", &self.fingerprint())
            .finish_non_exhaustive()
    }
}

impl MasterKey {
    /// The key written as `text`: 64 hex digits, surrounding whitespace ignored, and lines
    /// starting with `#` skipped.
    ///
    /// # Errors
    /// Anything else.
    pub fn from_text(text: &str) -> Result<Self> {
        let digits: Zeroizing<String> = Zeroizing::new(
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .collect(),
        );
        let bytes = Zeroizing::new(unhex(&digits).unwrap_or_default());
        let key: [u8; KEY_LEN] = bytes.as_slice().try_into().map_err(|_| {
            invalid(
                "an encryption key must be 64 hex digits (32 random bytes); generate one with \
                 `nusadb-server --generate-encryption-key <file>`"
                    .to_owned(),
            )
        })?;
        Ok(Self {
            key: Zeroizing::new(key),
        })
    }

    /// Read the key file at `path`. On Unix the file must not be writable by group or others,
    /// nor readable by others: anyone who can read it can read every database, and anyone who
    /// can replace it can lock the server out.
    ///
    /// # Errors
    /// The file cannot be read, has loose permissions, or does not hold a key.
    pub fn from_file(path: &Path) -> Result<Self> {
        let metadata = std::fs::metadata(path).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!(
                    "cannot read the encryption key file {}: {e}",
                    path.display()
                ),
            ))
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = metadata.permissions().mode() & 0o777;
            if mode & 0o026 != 0 {
                return Err(refused(format!(
                    "the encryption key file {} has mode {mode:03o}; it must not be writable by \
                     group or others nor readable by others. Run `chmod 600 {}` (or 640 to let \
                     a group read it)",
                    path.display(),
                    path.display()
                )));
            }
        }
        if !metadata.is_file() {
            return Err(invalid(format!(
                "the encryption key file {} is not a regular file",
                path.display()
            )));
        }
        let text = Zeroizing::new(std::fs::read_to_string(path)?);
        Self::from_text(&text).map_err(|e| invalid(format!("{}: {e}", path.display())))
    }

    /// Create a key file at `path` holding a new random key, readable and writable by its owner
    /// only. An existing file is never overwritten.
    ///
    /// # Errors
    /// The file exists, or I/O and random-source errors.
    pub fn generate_file(path: &Path) -> Result<Self> {
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        random_bytes(key.as_mut_slice())?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(path).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("cannot create the key file {}: {e}", path.display()),
            ))
        })?;
        let digits = Zeroizing::new(hex(key.as_slice()));
        let text = Zeroizing::new(format!(
            "# nusadb encryption master key. Keep a copy somewhere safe and apart from the data:\n\
             # without it the data cannot be read.\n{}\n",
            digits.as_str()
        ));
        std::io::Write::write_all(&mut file, text.as_bytes())?;
        file.sync_all()?;
        Ok(Self { key })
    }

    /// A short public name for the key: 16 hex digits of a SHA-256 of a domain string and the
    /// key. Lets two keys be told apart in messages without revealing either.
    pub fn fingerprint(&self) -> String {
        hex(self.fingerprint_bytes().as_slice())
    }

    fn fingerprint_bytes(&self) -> [u8; 8] {
        let mut hasher = Sha256::new();
        hasher.update(b"nusadb master key fingerprint\0");
        hasher.update(self.key.as_ref());
        let digest = hasher.finalize();
        let mut out = [0u8; 8];
        out.copy_from_slice(digest.get(..8).unwrap_or(&[0u8; 8]));
        out
    }

    /// The check value a copy in progress records for the key it writes under: 16 hex digits of
    /// an HMAC-SHA-256 under the key of a domain string. Like an envelope's check value it
    /// reveals nothing usable about the key, and it differs from both that value and the
    /// [`fingerprint`](Self::fingerprint).
    pub fn copy_check(&self) -> String {
        let mac = hmac_sha256(self.key.as_slice(), b"nusadb encrypted copy check\0");
        hex(mac.get(..8).unwrap_or(&[0u8; 8]))
    }

    /// The check value an envelope with id `id` stores for this key: HMAC-SHA-256 under the
    /// key of a domain string and the id, truncated to 8 bytes.
    fn check_value(&self, id: &[u8; 16]) -> [u8; 8] {
        let mac = hmac_sha256(
            self.key.as_slice(),
            &[b"nusadb data key check\0".as_slice(), id.as_slice()].concat(),
        );
        let mut out = [0u8; 8];
        out.copy_from_slice(mac.get(..8).unwrap_or(&[0u8; 8]));
        out
    }

    fn sealer(&self) -> Result<Sealer> {
        Sealer::new(&self.key)
    }
}

/// HMAC-SHA-256 (RFC 2104) of `message` under `key`.
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut block = Zeroizing::new([0u8; BLOCK]);
    let digest;
    let key = if key.len() > BLOCK {
        digest = Sha256::digest(key);
        digest.as_slice()
    } else {
        key
    };
    // `key` is at most one block long here.
    let (head, _) = block.split_at_mut(key.len());
    head.copy_from_slice(key);
    let pad = |byte: u8| -> Zeroizing<[u8; BLOCK]> {
        let mut out = Zeroizing::new([0u8; BLOCK]);
        for (o, k) in out.iter_mut().zip(block.iter()) {
            *o = k ^ byte;
        }
        out
    };
    let inner = Sha256::new()
        .chain_update(pad(0x36).as_slice())
        .chain_update(message)
        .finalize();
    Sha256::new()
        .chain_update(pad(0x5c).as_slice())
        .chain_update(inner)
        .finalize()
        .into()
}

/// A database's data key, unsealed: everything the database writes is sealed under it.
#[derive(Debug)]
pub struct DataKey {
    id: [u8; 16],
    sealer: Arc<Sealer>,
}

/// A data key as stored in an envelope.
struct Envelope {
    id: [u8; 16],
    /// [`MasterKey::check_value`] of the key that sealed it.
    check: [u8; 8],
    sealed: Vec<u8>,
}

/// The envelope beside a database's log: `<wal>.key`.
pub fn envelope_path(wal: &Path) -> PathBuf {
    let mut p = wal.as_os_str().to_owned();
    p.push(".key");
    p.into()
}

/// The envelope of a checkpoint archive: `<archive>/key`.
pub fn archive_envelope_path(archive: &Path) -> PathBuf {
    archive.join("key")
}

/// The AAD of a sealed data key.
fn envelope_aad(id: &[u8; 16]) -> Vec<u8> {
    let mut aad = ENVELOPE_HEADER.as_bytes().to_vec();
    aad.extend_from_slice(id);
    aad
}

impl Envelope {
    fn text(&self) -> String {
        format!(
            "{ENVELOPE_HEADER}\nid {}\nkey {}\nsealed {}\n",
            hex(&self.id),
            hex(&self.check),
            hex(&self.sealed)
        )
    }

    fn parse(text: &str) -> Option<Self> {
        let mut lines = text.lines();
        if lines.next()? != ENVELOPE_HEADER {
            return None;
        }
        let mut field = |name: &str| unhex(lines.next()?.strip_prefix(name)?.strip_prefix(' ')?);
        Some(Self {
            id: field("id")?.try_into().ok()?,
            check: field("key")?.try_into().ok()?,
            sealed: field("sealed")?,
        })
    }

    /// The envelope at `path`, `None` when there is none.
    fn read(path: &Path) -> Result<Option<Self>> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text).map(Some).ok_or_else(|| {
                invalid(format!(
                    "{} is not a readable key envelope; restore it from a backup of the database \
                     (without it the data cannot be read)",
                    path.display()
                ))
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Write the envelope to `path` atomically: a scratch file, synced, renamed over it, the
    /// directory synced.
    fn write(&self, path: &Path) -> Result<()> {
        write_atomically(path, self.text().as_bytes())
    }

    /// Unseal the data key with `master`.
    fn open(&self, master: &MasterKey, path: &Path) -> Result<DataKey> {
        if self.check != master.check_value(&self.id) {
            return Err(refused(format!(
                "the encryption key given (fingerprint {}) is not the one {} was sealed with; \
                 start with the right key file. Nothing was changed",
                master.fingerprint(),
                path.display(),
            )));
        }
        let key = Zeroizing::new(
            master
                .sealer()?
                .open(&envelope_aad(&self.id), &self.sealed)
                .ok_or_else(|| {
                    invalid(format!(
                        "the key envelope {} does not open with the given key although its \
                         check value matches: the envelope is damaged. Restore it from a backup",
                        path.display()
                    ))
                })?,
        );
        let key: &[u8; KEY_LEN] = key
            .as_slice()
            .try_into()
            .map_err(|_| invalid(format!("the key envelope {} is damaged", path.display())))?;
        Ok(DataKey {
            id: self.id,
            sealer: Arc::new(Sealer::new(key)?),
        })
    }

    fn seal(id: [u8; 16], key: &[u8; KEY_LEN], master: &MasterKey) -> Result<Self> {
        Ok(Self {
            id,
            check: master.check_value(&id),
            sealed: master.sealer()?.seal(&envelope_aad(&id), key)?,
        })
    }
}

/// Create `path` as a new file of its own and open it for writing.
///
/// Whatever is at `path` is removed first: a link is removed, never followed, and a file with
/// other names (a hard link) keeps its contents under them. The new file is then created
/// exclusively, so a link put there meanwhile makes this fail instead of writing through it.
/// For scratch files, markers and journals, which are never written in place.
///
/// # Errors
/// I/O errors, including a link or file that appears at `path` between the removal and the
/// creation.
pub fn create_fresh(path: &Path) -> Result<std::fs::File> {
    match std::fs::remove_file(path) {
        Ok(()) => {},
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
        Err(e) => return Err(e.into()),
    }
    Ok(std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?)
}

/// Write `bytes` to `path` as a new file of its own ([`create_fresh`]) and make them durable.
///
/// # Errors
/// I/O errors.
pub fn write_fresh(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = create_fresh(path)?;
    std::io::Write::write_all(&mut file, bytes)?;
    file.sync_all()?;
    Ok(())
}

/// Write `bytes` to `path` atomically, through a scratch file of its own.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut scratch = path.as_os_str().to_owned();
    scratch.push(".tmp");
    let scratch = PathBuf::from(scratch);
    write_fresh(&scratch, bytes)?;
    std::fs::rename(&scratch, path)?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        #[cfg(unix)]
        std::fs::File::open(dir)?.sync_all()?;
        #[cfg(not(unix))]
        let _ = dir;
    }
    Ok(())
}

impl DataKey {
    /// Generate a new data key, seal it under `master` and write the envelope to `path`.
    ///
    /// # Errors
    /// I/O and random-source errors.
    pub fn create(path: &Path, master: &MasterKey) -> Result<Self> {
        let mut id = [0u8; 16];
        random_bytes(&mut id)?;
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        random_bytes(key.as_mut_slice())?;
        Envelope::seal(id, &key, master)?.write(path)?;
        Ok(Self {
            id,
            sealer: Arc::new(Sealer::new(&key)?),
        })
    }

    /// The data key in the envelope at `path`, unsealed with `master`; `None` when there is no
    /// envelope there.
    ///
    /// # Errors
    /// A wrong master key, a damaged envelope, or I/O errors.
    pub fn open(path: &Path, master: &MasterKey) -> Result<Option<Self>> {
        Envelope::read(path)?
            .map(|envelope| envelope.open(master, path))
            .transpose()
    }

    /// The id of the data key in the envelope at `path`, read without any key; `None` when
    /// there is no envelope.
    ///
    /// # Errors
    /// A damaged envelope or I/O errors.
    pub fn id_at(path: &Path) -> Result<Option<String>> {
        Ok(Envelope::read(path)?.map(|e| hex(&e.id)))
    }

    /// This key's id, as hex.
    pub fn id(&self) -> String {
        hex(&self.id)
    }

    /// How the live log's records (and archived and shipped segments) are sealed.
    pub fn log_seal(&self) -> RecordSeal {
        RecordSeal {
            sealer: Arc::clone(&self.sealer),
            domain: DOMAIN_LOG,
            scope: 0,
        }
    }

    /// How the records of the checkpoint image covering `covered_lsn` are sealed.
    pub fn image_seal(&self, covered_lsn: u64) -> RecordSeal {
        RecordSeal {
            sealer: Arc::clone(&self.sealer),
            domain: DOMAIN_IMAGE,
            scope: covered_lsn,
        }
    }

    /// The sealer itself, for strings sealed under an AAD built elsewhere.
    pub(crate) fn sealer(&self) -> &Sealer {
        &self.sealer
    }

    /// Append page `id`, stored at `slot` of the segment `segment`, sealed, to `out`.
    ///
    /// # Errors
    /// The AEAD refused the input.
    pub(crate) fn seal_page(
        &self,
        id: u64,
        slot: u32,
        segment: &str,
        page: &Page,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        self.sealer
            .seal_into(&page_aad(id, slot, segment), page, out)
    }

    /// Open the sealed page `id` read from `slot` of the segment `segment` into `page`; `false`
    /// when it does not verify.
    pub(crate) fn open_page(
        &self,
        id: u64,
        slot: u32,
        segment: &str,
        sealed: &[u8],
        page: &mut Page,
    ) -> bool {
        self.sealer
            .open_into(&page_aad(id, slot, segment), sealed, page)
    }
}

/// The AAD of a page in a segment: domain, page id, slot, segment name.
fn page_aad(id: u64, slot: u32, segment: &str) -> Vec<u8> {
    let mut aad = Vec::with_capacity(13 + segment.len());
    aad.push(DOMAIN_PAGE);
    aad.extend_from_slice(&id.to_le_bytes());
    aad.extend_from_slice(&slot.to_le_bytes());
    aad.extend_from_slice(segment.as_bytes());
    aad
}

/// The AAD of a page in the spill file: domain, page id, write generation.
#[allow(
    clippy::missing_const_for_fn,
    reason = "slice splitting is not const; the AAD is built once per spilled page"
)]
pub(crate) fn spill_aad(id: u64, generation: u64) -> [u8; 17] {
    let mut aad = [0u8; 17];
    let (domain, rest) = aad.split_at_mut(1);
    domain.copy_from_slice(&[DOMAIN_SPILL]);
    let (page, generation_bytes) = rest.split_at_mut(8);
    page.copy_from_slice(&id.to_le_bytes());
    generation_bytes.copy_from_slice(&generation.to_le_bytes());
    aad
}

/// The marker an encrypted copy (see `BtreeEngine::copy_encrypted`) leaves beside the log until
/// it is complete: `<wal>.incomplete`. A database with it is refused.
pub fn incomplete_path(wal: &Path) -> PathBuf {
    let mut p = wal.as_os_str().to_owned();
    p.push(".incomplete");
    p.into()
}

/// Mark the database whose log is `wal` as an encrypted copy in progress.
///
/// Writes [`incomplete_path`] durably as a new file ([`write_fresh`]: a link there is replaced,
/// never written through), creating its directory if needed.
/// A marker that is already there as a plain file is kept as it is. Until the marker is
/// removed, no open accepts the database.
///
/// # Errors
/// I/O errors.
pub fn mark_incomplete(wal: &Path) -> Result<()> {
    if let Some(parent) = wal.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let marker = incomplete_path(wal);
    // A marker already there stays: removing it to write it again would leave a moment, which a
    // crash could make permanent, where the directory is not marked. Anything else at its name
    // (a link above all) is replaced, never written through.
    if !std::fs::symlink_metadata(&marker).is_ok_and(|m| m.file_type().is_file()) {
        write_fresh(&marker, b"an encrypted copy is being written here\n")?;
    }
    if let Some(dir) = wal.parent().filter(|d| !d.as_os_str().is_empty()) {
        #[cfg(unix)]
        std::fs::File::open(dir)?.sync_all()?;
        #[cfg(not(unix))]
        let _ = dir;
    }
    Ok(())
}

/// Whether the database whose log is `wal` holds anything: a log with records, a checkpoint
/// image, or page segments. A directory that holds none of them can start encrypted.
pub fn holds_data(wal: &Path) -> bool {
    let log = std::fs::metadata(wal).is_ok_and(|m| m.len() > 0);
    let mut ckpt = wal.as_os_str().to_owned();
    ckpt.push(".ckpt");
    let mut pages = wal.as_os_str().to_owned();
    pages.push(".pages");
    let segments = std::fs::read_dir(PathBuf::from(pages)).is_ok_and(|mut entries| {
        entries.any(|e| {
            e.is_ok_and(|e| {
                e.file_name().to_str().is_some_and(|n| {
                    std::path::Path::new(n)
                        .extension()
                        .is_some_and(|ext| ext == "seg")
                })
            })
        })
    });
    log || PathBuf::from(ckpt).exists() || segments
}

/// What opening the database at `wal` with `master` (or without a key) will find, decided
/// without writing anything.
#[derive(Debug)]
pub enum KeyCheck {
    /// Not encrypted, and no key given: opens as before.
    Plain,
    /// Encrypted, and the key opens its data key.
    Encrypted(DataKey),
    /// A key was given and the database holds nothing yet: it will be created encrypted.
    New,
}

/// Decide, without writing anything, how the database at `wal` opens with `master`.
///
/// Refused: an encrypted database without a key or with the wrong one, and an unencrypted
/// database that holds data when a key is given (it would stay readable on disk while the server
/// claims encryption).
///
/// # Errors
/// The refusals above, each saying what to do, or I/O errors.
pub fn check_database(wal: &Path, master: Option<&MasterKey>) -> Result<KeyCheck> {
    let path = envelope_path(wal);
    let marker = incomplete_path(wal);
    if marker.exists() {
        return Err(refused(format!(
            "the database at {} is an encrypted copy that never finished ({} is still there); \
             run the copy (--encrypt-from) again, or remove the database directory. Nothing was \
             changed",
            wal.display(),
            marker.display()
        )));
    }
    let envelope = Envelope::read(&path)?;
    // An envelope beside data that was written unencrypted does not belong to it: the data
    // would be read as sealed and refused, or worse, the directory recorded as encrypted.
    if envelope.is_some()
        && holds_data(wal)
        && crate::format::read_format(wal)? != Some(crate::format::ENCRYPTED_FORMAT_VERSION)
    {
        return Err(invalid(format!(
            "the database at {} holds unencrypted data (its data format is not {}) but a key \
             envelope {} lies beside it; the envelope does not belong to this database. Remove \
             it to open the database unencrypted. Nothing was changed",
            wal.display(),
            crate::format::ENCRYPTED_FORMAT_VERSION,
            path.display()
        )));
    }
    match (envelope, master) {
        (Some(envelope), Some(master)) => Ok(KeyCheck::Encrypted(envelope.open(master, &path)?)),
        (Some(_), None) => Err(refused(format!(
            "the database at {} is encrypted (it has the key envelope {}); start the server with \
             --encryption-key-file. Nothing was changed",
            wal.display(),
            path.display()
        ))),
        (None, Some(_)) if holds_data(wal) => Err(refused(format!(
            "the database at {} is not encrypted, and the server was started with an encryption \
             key; its data would stay readable on disk. Either start without the key, or copy it \
             into a new encrypted data directory with --encrypt-from (see the encryption section \
             of docs/deployment.md). Nothing was changed",
            wal.display()
        ))),
        (None, Some(_)) => Ok(KeyCheck::New),
        (None, None) => Ok(KeyCheck::Plain),
    }
}

/// What [`rewrap`] did to one envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rewrapped {
    /// Sealed again under the new key.
    Done,
    /// Already sealed under the new key (a rotation run again after an interruption).
    Already,
}

/// Seal the data key in the envelope at `path` under `new` instead of `old`.
///
/// The data itself is not touched: it stays sealed under the same data key. The envelope is
/// replaced atomically, so an interrupted rotation leaves each envelope under one key or the
/// other, and running it again finishes it.
///
/// # Errors
/// The envelope opens with neither key, is damaged, or I/O errors.
pub fn rewrap(path: &Path, old: &MasterKey, new: &MasterKey) -> Result<Rewrapped> {
    let envelope = Envelope::read(path)?
        .ok_or_else(|| invalid(format!("{} holds no key envelope", path.display())))?;
    if envelope.check == new.check_value(&envelope.id) {
        // Prove it really opens with the new key before calling it done.
        envelope.open(new, path)?;
        return Ok(Rewrapped::Already);
    }
    let key = Zeroizing::new(
        old.sealer()?
            .open(&envelope_aad(&envelope.id), &envelope.sealed)
            .filter(|_| envelope.check == old.check_value(&envelope.id))
            .ok_or_else(|| {
                refused(format!(
                    "{} is sealed under neither the old key (fingerprint {}) nor the new one \
                     (fingerprint {}); it was left unchanged",
                    path.display(),
                    old.fingerprint(),
                    new.fingerprint()
                ))
            })?,
    );
    let key: &[u8; KEY_LEN] = key
        .as_slice()
        .try_into()
        .map_err(|_| invalid(format!("the key envelope {} is damaged", path.display())))?;
    Envelope::seal(envelope.id, key, new)?.write(path)?;
    Ok(Rewrapped::Done)
}

/// Whether `path` holds a key envelope (its first line is the envelope header).
pub fn is_envelope(path: &Path) -> bool {
    std::fs::read_to_string(path).is_ok_and(|text| text.lines().next() == Some(ENVELOPE_HEADER))
}

/// Copy the envelope at `from` to `to`, atomically.
///
/// # Errors
/// I/O errors.
pub(crate) fn copy_envelope(from: &Path, to: &Path) -> Result<()> {
    let bytes = std::fs::read(from)?;
    write_atomically(to, &bytes)
}

/// The data key of the checkpoint archive at `archive`, unsealed with `master`. `None` for an
/// unencrypted archive read without a key.
///
/// # Errors
/// An encrypted archive without a key or with the wrong one, or an unencrypted archive read with
/// a key (what it restores or seeds would not be encrypted).
pub fn archive_key(archive: &Path, master: Option<&MasterKey>) -> Result<Option<DataKey>> {
    let path = archive_envelope_path(archive);
    match (Envelope::read(&path)?, master) {
        (Some(envelope), Some(master)) => Ok(Some(envelope.open(master, &path)?)),
        (Some(_), None) => Err(refused(format!(
            "the archive {} is encrypted; pass the encryption key",
            archive.display()
        ))),
        (None, Some(_)) => Err(refused(format!(
            "the archive {} is not encrypted, and an encryption key was given: what it restores \
             would not be encrypted. Use it without the key",
            archive.display()
        ))),
        (None, None) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(byte: u8) -> MasterKey {
        MasterKey::from_text(&hex(&[byte; 32])).unwrap()
    }

    /// A file outside the database's directory, its contents, and a dangling path beside it.
    #[cfg(unix)]
    fn victim(dir: &Path) -> (PathBuf, PathBuf) {
        let file = dir.join("victim");
        std::fs::write(&file, b"victim bytes").unwrap();
        (file, dir.join("nowhere"))
    }

    #[cfg(unix)]
    #[test]
    fn a_marker_is_never_written_through_a_link_or_into_a_hard_linked_file() {
        let outside = tempfile::tempdir().unwrap();
        let (file, nowhere) = victim(outside.path());
        let db = tempfile::tempdir().unwrap();
        let wal = db.path().join("btree.wal");
        let marker = incomplete_path(&wal);
        for how in ["link", "dangling", "hard"] {
            let _ = std::fs::remove_file(&marker);
            match how {
                "link" => std::os::unix::fs::symlink(&file, &marker).unwrap(),
                "dangling" => std::os::unix::fs::symlink(&nowhere, &marker).unwrap(),
                _ => std::fs::hard_link(&file, &marker).unwrap(),
            }
            mark_incomplete(&wal).unwrap();
            assert_eq!(std::fs::read(&file).unwrap(), b"victim bytes", "{how}");
            assert!(!nowhere.exists(), "{how}");
            let meta = std::fs::symlink_metadata(&marker).unwrap();
            assert!(meta.file_type().is_file(), "{how}");
            if how != "hard" {
                // A link is replaced by a marker of its own.
                assert!(
                    std::fs::read(&marker)
                        .unwrap()
                        .starts_with(b"an encrypted copy")
                );
            }
        }
        // A marker already there is kept, never removed and written again.
        std::fs::remove_file(&marker).unwrap();
        mark_incomplete(&wal).unwrap();
        std::fs::write(&marker, b"left by an earlier run").unwrap();
        let inode = std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&marker).unwrap());
        mark_incomplete(&wal).unwrap();
        assert_eq!(std::fs::read(&marker).unwrap(), b"left by an earlier run");
        assert_eq!(
            std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&marker).unwrap()),
            inode
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_envelope_scratch_file_is_never_written_through_a_link_on_create_or_rotation() {
        let outside = tempfile::tempdir().unwrap();
        let (file, nowhere) = victim(outside.path());
        let db = tempfile::tempdir().unwrap();
        let path = db.path().join("btree.wal.key");
        let scratch = db.path().join("btree.wal.key.tmp");
        std::os::unix::fs::symlink(&file, &scratch).unwrap();
        DataKey::create(&path, &key(1)).unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"victim bytes");
        // Rotation, with a dangling link and then a hard link at the scratch name.
        std::os::unix::fs::symlink(&nowhere, &scratch).unwrap();
        assert_eq!(rewrap(&path, &key(1), &key(2)).unwrap(), Rewrapped::Done);
        assert!(!nowhere.exists());
        std::fs::hard_link(&file, &scratch).unwrap();
        assert_eq!(rewrap(&path, &key(2), &key(3)).unwrap(), Rewrapped::Done);
        assert_eq!(std::fs::read(&file).unwrap(), b"victim bytes");
        assert!(DataKey::open(&path, &key(3)).unwrap().is_some());
        assert!(!scratch.exists());
    }

    #[test]
    fn key_text_accepts_hex_with_comments_and_refuses_anything_else() {
        let text = format!("# a comment\n  {}  \n", hex(&[7u8; 32]));
        assert_eq!(
            MasterKey::from_text(&text).unwrap().fingerprint(),
            key(7).fingerprint()
        );
        assert!(MasterKey::from_text("abcd").is_err());
        assert!(MasterKey::from_text(&"zz".repeat(32)).is_err());
        assert!(MasterKey::from_text(&hex(&[1u8; 33])).is_err());
        assert_ne!(key(1).fingerprint(), key(2).fingerprint());
        // The stored check value depends on the envelope's id, and differs from the fingerprint.
        assert_ne!(key(1).check_value(&[1; 16]), key(1).check_value(&[2; 16]));
        assert_ne!(hex(&key(1).check_value(&[1; 16])), key(1).fingerprint());
        assert_ne!(key(1).copy_check(), key(1).fingerprint());
        assert_ne!(key(1).copy_check(), key(2).copy_check());
        assert_eq!(key(1).copy_check().len(), 16);
    }

    #[test]
    fn an_envelope_opens_with_its_key_only_and_rewraps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("btree.wal.key");
        let created = DataKey::create(&path, &key(1)).unwrap();
        let opened = DataKey::open(&path, &key(1)).unwrap().unwrap();
        assert_eq!(created.id(), opened.id());
        let sealed = created.sealer().seal(b"x", b"row").unwrap();
        assert_eq!(opened.sealer().open(b"x", &sealed).unwrap(), b"row");
        let err = DataKey::open(&path, &key(2)).unwrap_err().to_string();
        assert!(err.contains("not the one"), "{err}");

        assert_eq!(rewrap(&path, &key(1), &key(2)).unwrap(), Rewrapped::Done);
        assert_eq!(rewrap(&path, &key(1), &key(2)).unwrap(), Rewrapped::Already);
        assert!(DataKey::open(&path, &key(1)).is_err());
        let rotated = DataKey::open(&path, &key(2)).unwrap().unwrap();
        assert_eq!(rotated.id(), created.id());
        assert_eq!(rotated.sealer().open(b"x", &sealed).unwrap(), b"row");
        // A third key opens nothing and changes nothing.
        let before = std::fs::read(&path).unwrap();
        assert!(rewrap(&path, &key(3), &key(4)).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn hmac_matches_the_rfc_4231_vectors() {
        // Test case 2: key "Jefe".
        assert_eq!(
            hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // Test case 6: a key longer than the block.
        assert_eq!(
            hex(&hmac_sha256(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn a_damaged_envelope_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k");
        DataKey::create(&path, &key(1)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let line = text.lines().nth(3).unwrap().to_owned();
        let last = line.chars().last().unwrap();
        let flipped = if last == '0' { '1' } else { '0' };
        let damaged = text.replace(&line, &format!("{}{flipped}", &line[..line.len() - 1]));
        std::fs::write(&path, damaged).unwrap();
        let err = DataKey::open(&path, &key(1)).unwrap_err().to_string();
        assert!(err.contains("damaged"), "{err}");
    }

    #[test]
    fn a_page_opens_only_where_it_was_sealed() {
        let dir = tempfile::tempdir().unwrap();
        let data = DataKey::create(&dir.path().join("k"), &key(1)).unwrap();
        let page = [9u8; PAGE_SIZE];
        let mut sealed = Vec::new();
        data.seal_page(4, 0, "seg", &page, &mut sealed).unwrap();
        assert_eq!(sealed.len(), SEALED_PAGE_SIZE);
        let mut out = [0u8; PAGE_SIZE];
        assert!(data.open_page(4, 0, "seg", &sealed, &mut out));
        assert_eq!(out, page);
        assert!(!data.open_page(5, 0, "seg", &sealed, &mut out));
        assert!(!data.open_page(4, 1, "seg", &sealed, &mut out));
        assert!(!data.open_page(4, 0, "other", &sealed, &mut out));
    }

    #[cfg(unix)]
    #[test]
    fn a_key_file_with_loose_permissions_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("master.key");
        let generated = MasterKey::generate_file(&path).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            MasterKey::from_file(&path).unwrap().fingerprint(),
            generated.fingerprint()
        );
        assert!(
            MasterKey::generate_file(&path).is_err(),
            "never overwritten"
        );
        for mode in [0o644, 0o620, 0o602] {
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            assert!(MasterKey::from_file(&path).is_err(), "mode {mode:o}");
        }
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        assert!(MasterKey::from_file(&path).is_ok());
    }
}
