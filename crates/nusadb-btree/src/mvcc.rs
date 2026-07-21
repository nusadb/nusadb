//! MVCC for the clustered engine: version stamps in the leaf, old versions in an undo
//! arena, visibility through **read views**.
//!
//! The newest version of a row lives in its leaf entry, prefixed by a fixed [`RowMeta`] header:
//! `xmin` (creating transaction), `xmax` (deleting transaction, `0` = none), and the arena index
//! of the **previous** version (`u64::MAX` = none). An `UPDATE` pushes the old header+tuple into
//! the arena and installs the new version in the leaf; a `DELETE` stamps `xmax` in place. A
//! reader that cannot see the leaf version walks the chain newest→oldest and takes the first
//! version whose creator its [`ReadView`] can see (then that version's `xmax` decides deletion).
//!
//! Rolled-back transactions leave **no versions behind** — rollback physically restores the
//! previous leaf bytes (the undo-op discipline, now carrying the whole encoded entry) — so
//! any version present in a chain was written by a transaction that either committed or is
//! still active. Visibility therefore needs only the view's active-set, not a commit log.
//!
//! Purge reclaims arena slots whose chain no view can reach; freed slots are `None` and
//! their indices are recycled through the engine's free list.

use std::collections::HashSet;

/// `xmax` value meaning "not deleted".
pub const NO_XMAX: u64 = 0;
/// Undo-pointer value meaning "no previous version".
pub const NO_UNDO: u64 = u64::MAX;
/// Bytes the [`RowMeta`] header occupies in front of the tuple in a leaf value.
pub const META: usize = 24;

/// The fixed per-version header stored in front of the tuple bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowMeta {
    /// Transaction that created this version.
    pub xmin: u64,
    /// Transaction that deleted the row at this version ([`NO_XMAX`] = live).
    pub xmax: u64,
    /// Arena index of the previous version ([`NO_UNDO`] = none).
    pub undo: u64,
}

impl RowMeta {
    /// A fresh version created by `xmin`: live, no history.
    #[must_use]
    pub const fn fresh(xmin: u64) -> Self {
        Self {
            xmin,
            xmax: NO_XMAX,
            undo: NO_UNDO,
        }
    }
}

/// One superseded version parked in the undo arena: the full header + the tuple bytes it had.
#[derive(Debug, Clone)]
pub struct UndoVersion {
    /// The version's header as it was in the leaf when superseded.
    pub meta: RowMeta,
    /// The tuple bytes of that version.
    pub tuple: Vec<u8>,
}

/// Encode a leaf value: header then tuple bytes.
#[must_use]
pub fn encode_row(meta: RowMeta, tuple: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(META + tuple.len());
    out.extend_from_slice(&meta.xmin.to_le_bytes());
    out.extend_from_slice(&meta.xmax.to_le_bytes());
    out.extend_from_slice(&meta.undo.to_le_bytes());
    out.extend_from_slice(tuple);
    out
}

/// Decode a leaf value into its header and tuple bytes, or `None` if it is too short to carry a
/// header (a corruption-class condition the caller reports loudly).
#[must_use]
pub fn decode_row(value: &[u8]) -> Option<(RowMeta, &[u8])> {
    let xmin = u64::from_le_bytes(value.get(0..8)?.try_into().ok()?);
    let xmax = u64::from_le_bytes(value.get(8..16)?.try_into().ok()?);
    let undo = u64::from_le_bytes(value.get(16..24)?.try_into().ok()?);
    Some((RowMeta { xmin, xmax, undo }, value.get(META..)?))
}

/// A transaction's consistent snapshot: which OTHER transactions' effects it may observe.
///
/// Taken at `BEGIN` for `REPEATABLE READ`/`SERIALIZABLE` and afresh at every read for
/// `READ COMMITTED` (statement-level snapshots — the same discipline the predecessor engine's
/// `scan_snapshot` uses).
#[derive(Debug, Clone)]
pub struct ReadView {
    /// The observing transaction: its own writes are always visible to it.
    pub own: u64,
    /// Transactions active (begun, not yet ended) when the view was taken — invisible.
    pub active: HashSet<u64>,
    /// First transaction id NOT yet assigned when the view was taken — ids at or beyond this
    /// began later and are invisible.
    pub horizon: u64,
}

impl ReadView {
    /// Whether a version stamp (an `xmin` or a non-zero `xmax`) is visible under this view:
    /// the observer itself, or a transaction that began before the horizon and had already
    /// ended (any version left behind by an ended transaction is committed — rollback erases
    /// its versions physically).
    #[must_use]
    pub fn sees(&self, txn: u64) -> bool {
        txn == self.own || (txn < self.horizon && !self.active.contains(&txn))
    }
}

/// Resolve the version of a row visible under `view`, walking the undo chain newest→oldest.
///
/// `arena` is the engine's undo arena (indexes stored in [`RowMeta::undo`]). Returns the visible
/// tuple bytes, or `None` when no version is visible (created after the view, or deleted by a
/// transaction the view sees).
#[must_use]
pub fn visible_tuple<'a>(
    leaf_meta: RowMeta,
    leaf_tuple: &'a [u8],
    arena: &'a [Option<UndoVersion>],
    view: &ReadView,
) -> Option<&'a [u8]> {
    let mut meta = leaf_meta;
    let mut tuple = leaf_tuple;
    loop {
        if view.sees(meta.xmin) {
            // This is the newest version whose creator the view sees; its xmax decides.
            let deleted = meta.xmax != NO_XMAX && view.sees(meta.xmax);
            return if deleted { None } else { Some(tuple) };
        }
        if meta.undo == NO_UNDO {
            return None; // The row did not exist for this view.
        }
        // A freed (purged) slot is unreachable by construction — purge only frees chains no
        // view can walk — so hitting `None` here means the row did not exist for this view.
        let prev = arena.get(usize::try_from(meta.undo).ok()?)?.as_ref()?;
        meta = prev.meta;
        tuple = prev.tuple.as_slice();
    }
}
