//! Tests for `catalog` (`src/catalog.rs`) — the on-disk table/column metadata store. These
//! create real files (catalog persistence + restart), so they live in `tests/` and use only the
//! public `Catalog` / `DiskManager` API.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "integration test harness asserts via unwrap; corruption test indexes a known offset"
)]

use nusadb_core::{ColumnDef, ColumnType, Error, PageId, PageStore, TableDef, TableId};
use nusadb_storage::{Catalog, DiskManager};

/// Catalog metadata lives on page 0 (`CATALOG_HEAD` inside the crate); referenced by literal here.
const CATALOG_HEAD: PageId = PageId(0);

fn users_def() -> TableDef {
    TableDef {
        schema: "public".to_owned(),
        name: "users".into(),
        columns: vec![
            ColumnDef {
                name: "id".into(),
                ty: ColumnType::Int,
                nullable: false,
            },
            ColumnDef {
                name: "name".into(),
                ty: ColumnType::Text,
                nullable: true,
            },
        ],
    }
}

#[test]
fn create_lookup_and_duplicate() {
    let dir = nusadb_test_utils::temp_dir();
    let dm = DiskManager::open(dir.path().join("cat.db")).unwrap();
    let mut cat = Catalog::create(&dm).unwrap();
    let id = cat.create_table(&users_def()).unwrap();
    assert_eq!(id, TableId(1));

    let schema = cat.lookup("users").unwrap();
    assert_eq!(schema.id, TableId(1));
    assert_eq!(schema.columns.len(), 2);
    assert_eq!(schema.columns[0].ty, ColumnType::Int);
    assert!(!schema.columns[0].nullable);
    assert!(schema.columns[1].nullable);

    // duplicate name is rejected
    assert!(matches!(
        cat.create_table(&users_def()),
        Err(Error::TableExists { .. })
    ));
    assert!(cat.lookup("missing").is_none());
}

#[test]
fn survives_restart() {
    let dir = nusadb_test_utils::temp_dir();
    let path = dir.path().join("cat.db");
    {
        let dm = DiskManager::open(&path).unwrap();
        let mut cat = Catalog::create(&dm).unwrap();
        cat.create_table(&users_def()).unwrap();
        cat.create_table(&TableDef {
            schema: "public".to_owned(),
            name: "orders".into(),
            columns: vec![ColumnDef {
                name: "total".into(),
                ty: ColumnType::Float,
                nullable: false,
            }],
        })
        .unwrap();
    }
    // Reopen the file from scratch — metadata must still be there.
    let dm = DiskManager::open(&path).unwrap();
    let cat = Catalog::open(&dm).unwrap();
    assert_eq!(cat.tables().len(), 2);
    let users = cat.lookup("users").unwrap();
    assert_eq!(users.id, TableId(1));
    assert_eq!(users.columns[1].name, "name");
    let orders = cat.lookup("orders").unwrap();
    assert_eq!(orders.id, TableId(2));
    assert_eq!(orders.columns[0].ty, ColumnType::Float);
}

#[test]
fn drop_then_reopen() {
    let dir = nusadb_test_utils::temp_dir();
    let path = dir.path().join("cat.db");
    let id = {
        let dm = DiskManager::open(&path).unwrap();
        let mut cat = Catalog::create(&dm).unwrap();
        let id = cat.create_table(&users_def()).unwrap();
        cat.drop_table(id).unwrap();
        assert!(cat.lookup("users").is_none());
        assert!(matches!(
            cat.drop_table(id),
            Err(Error::TableNotFound { .. })
        ));
        id
    };
    let dm = DiskManager::open(&path).unwrap();
    let cat = Catalog::open(&dm).unwrap();
    assert!(cat.lookup_id(id).is_none());
    assert_eq!(cat.tables().len(), 0);
}

#[test]
fn drop_reclaims_surplus_catalog_pages() {
    // G14: when the catalog blob shrinks (e.g. after dropping many tables) the tail pages it no
    // longer needs must be deallocated, not left allocated and stamped CATALOG_MAGIC where the
    // free-list scan can never reclaim them (a permanent orphan leak on every DDL).
    let dir = nusadb_test_utils::temp_dir();
    let path = dir.path().join("cat.db");
    let dm = DiskManager::open(&path).unwrap();
    let mut cat = Catalog::create(&dm).unwrap();

    // Each table carries a long name so the serialized blob crosses page boundaries quickly.
    let mut ids = Vec::new();
    let mut i = 0u32;
    while dm.page_count() < 3 {
        let def = TableDef {
            schema: "public".to_owned(),
            name: format!("table_{i:04}_{}", "x".repeat(300)),
            columns: vec![ColumnDef {
                name: "c".into(),
                ty: ColumnType::Int,
                nullable: true,
            }],
        };
        ids.push(cat.create_table(&def).unwrap());
        i += 1;
    }
    let grown_pages = dm.page_count();
    assert!(grown_pages >= 3, "catalog should span multiple pages");
    assert_eq!(dm.free_count(), 0, "no pages freed while growing");

    // Drop every table but the first; the blob collapses back to a single page.
    for &id in &ids[1..] {
        cat.drop_table(id).unwrap();
    }
    // The surplus catalog pages were released to the free list (reclaimable), not orphaned.
    assert!(
        dm.free_count() >= grown_pages as usize - 1,
        "dropped catalog pages must be reclaimed (free_count={}, grown_pages={grown_pages})",
        dm.free_count()
    );

    // Reopen from disk: the surviving table is intact and the orphan tail is gone.
    drop(cat);
    let dm2 = DiskManager::open(&path).unwrap();
    let cat2 = Catalog::open(&dm2).unwrap();
    assert_eq!(cat2.tables().len(), 1);
    assert!(cat2.lookup_id(ids[0]).is_some());
}

#[test]
fn bad_magic_is_detected() {
    let dir = nusadb_test_utils::temp_dir();
    let path = dir.path().join("cat.db");
    {
        let dm = DiskManager::open(&path).unwrap();
        Catalog::create(&dm).unwrap();
    }
    // Corrupt page 0's magic.
    let dm = DiskManager::open(&path).unwrap();
    let mut page = dm.read_page(CATALOG_HEAD).unwrap();
    page[0] ^= 0xFF;
    dm.write_page(CATALOG_HEAD, &page).unwrap();
    dm.fsync().unwrap();
    assert!(matches!(
        Catalog::open(&dm),
        Err(Error::InvalidMagic { .. })
    ));
}

#[test]
fn cyclic_page_chain_is_rejected_not_looped_forever() {
    // A corrupt `next` pointer (offset 8) that points back at the head would make `open` walk the
    // chain forever; it must surface a clean `InvalidData` error instead.
    let dir = nusadb_test_utils::temp_dir();
    let path = dir.path().join("cat.db");
    {
        let dm = DiskManager::open(&path).unwrap();
        Catalog::create(&dm).unwrap();
    }
    let dm = DiskManager::open(&path).unwrap();
    let mut page = dm.read_page(CATALOG_HEAD).unwrap();
    page[8..16].copy_from_slice(&CATALOG_HEAD.0.to_le_bytes()); // next := head → self-cycle
    dm.write_page(CATALOG_HEAD, &page).unwrap();
    dm.fsync().unwrap();
    let err = Catalog::open(&dm).unwrap_err();
    assert!(
        matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::InvalidData),
        "expected InvalidData, got {err:?}"
    );
}
