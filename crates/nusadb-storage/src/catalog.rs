//! Catalog / meta store — table and column metadata persisted on dedicated pages.
//!
//! The catalog is a small key set (one entry per table) that must survive restart, so it
//! is serialized to a chain of pages anchored at a well-known location ([`CATALOG_HEAD`],
//! page 0). On every DDL change the whole catalog is re-serialized and re-written, then
//! `fsync`-ed — durability over write-efficiency, which is the right trade-off for
//! rarely-changing metadata.
//!
//! Per-page layout (distinct from the [`Page`](crate::page::Page) heap format):
//!
//! ```text
//! 0   4   magic (0x4341_544C = "CATL")
//! 4   2   reserved
//! 6   2   used_len  (data bytes used in this page)
//! 8   8   next      (page id of the continuation, or NO_NEXT)
//! 16  ..  data
//! ```

// Byte-level catalog codec: bounded offset arithmetic + slicing.
#![allow(clippy::indexing_slicing)]

use nusadb_core::engine::ArrayElem;
use nusadb_core::{
    ColumnDef, ColumnType, Error, PAGE_SIZE, PUBLIC_SCHEMA, PageId, PageStore, Result, TableDef,
    TableId, TableSchema,
};

/// Magic identifying a catalog page: ASCII `"CATL"`.
pub const CATALOG_MAGIC: u32 = 0x4341_544C;

/// The catalog always begins at page 0 so it can be found on reopen.
pub const CATALOG_HEAD: PageId = PageId(0);

const PAGE_HDR: usize = 16;
const DATA_CAP: usize = PAGE_SIZE - PAGE_HDR;
const NO_NEXT: u64 = u64::MAX;

/// Persistent table/column metadata store over a [`PageStore`].
#[derive(Debug)]
pub struct Catalog<'s, S: PageStore> {
    store: &'s S,
    tables: Vec<TableSchema>,
    chain: Vec<PageId>, // pages backing the catalog; chain[0] == CATALOG_HEAD
    next_table_id: u64,
}

impl<'s, S: PageStore> Catalog<'s, S> {
    /// Initialize an empty catalog on a **fresh** store (it must claim page 0).
    ///
    /// # Errors
    /// Propagates storage errors, or [`Error::Io`] if page 0 is already taken (the catalog
    /// must be the first thing created in a database file).
    pub fn create(store: &'s S) -> Result<Self> {
        let head = store.allocate_page()?;
        if head != CATALOG_HEAD {
            return Err(Error::Io(std::io::Error::other(
                "catalog must be initialized on a fresh store (expected page 0)",
            )));
        }
        let mut catalog = Self {
            store,
            tables: Vec::new(),
            chain: vec![head],
            next_table_id: 1,
        };
        catalog.persist()?;
        Ok(catalog)
    }

    /// Load an existing catalog by following the page chain from [`CATALOG_HEAD`].
    ///
    /// # Errors
    /// [`Error::InvalidMagic`] if a catalog page is unrecognizable, or a propagated
    /// storage/parse error.
    pub fn open(store: &'s S) -> Result<Self> {
        let mut chain = Vec::new();
        let mut blob = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut id = CATALOG_HEAD;
        loop {
            // Guard the chain walk against a corrupt `next` pointer: a self- or back-reference
            // would otherwise spin forever, and a `used` byte count past the page would panic the
            // slice. Both become a clean `InvalidData` error.
            if !seen.insert(id) {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("corrupt catalog: cyclic page chain at {id:?}"),
                )));
            }
            let page = store.read_page(id)?;
            if rd_u32_at(&page, 0) != CATALOG_MAGIC {
                return Err(Error::InvalidMagic { page_id: id });
            }
            let used = rd_u16_at(&page, 6) as usize;
            if used > DATA_CAP {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("corrupt catalog: used_len {used} exceeds page capacity at {id:?}"),
                )));
            }
            chain.push(id);
            blob.extend_from_slice(&page[PAGE_HDR..PAGE_HDR + used]);
            let next = rd_u64_at(&page, 8);
            if next == NO_NEXT {
                break;
            }
            id = PageId(next);
        }
        let tables = Self::deserialize(&blob)?;
        let next_table_id = tables.iter().map(|t| t.id.0).max().map_or(1, |m| m + 1);
        Ok(Self {
            store,
            tables,
            chain,
            next_table_id,
        })
    }

    /// All tables, in creation order.
    #[must_use]
    pub fn tables(&self) -> &[TableSchema] {
        &self.tables
    }

    /// Resolve a table name to its schema.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<TableSchema> {
        self.tables.iter().find(|t| t.name == name).cloned()
    }

    /// Resolve a table id to its schema.
    #[must_use]
    pub fn lookup_id(&self, id: TableId) -> Option<TableSchema> {
        self.tables.iter().find(|t| t.id == id).cloned()
    }

    /// Register a new table, returning its assigned id.
    ///
    /// # Errors
    /// [`Error::TableExists`] if the name is taken; propagates storage errors.
    pub fn create_table(&mut self, def: &TableDef) -> Result<TableId> {
        if self.tables.iter().any(|t| t.name == def.name) {
            return Err(Error::TableExists {
                name: def.name.clone(),
            });
        }
        let id = TableId(self.next_table_id);
        self.next_table_id += 1;
        self.tables.push(TableSchema {
            id,
            schema: def.schema.clone(),
            name: def.name.clone(),
            columns: def.columns.clone(),
        });
        self.persist()?;
        Ok(id)
    }

    /// Remove a table from the catalog.
    ///
    /// # Errors
    /// [`Error::TableNotFound`] if no table has that id; propagates storage errors.
    pub fn drop_table(&mut self, id: TableId) -> Result<()> {
        let before = self.tables.len();
        self.tables.retain(|t| t.id != id);
        if self.tables.len() == before {
            return Err(Error::TableNotFound {
                name: format!("id {}", id.0),
            });
        }
        self.persist()
    }

    // ── persistence ──────────────────────────────────────────────────────────

    fn persist(&mut self) -> Result<()> {
        let blob = self.serialize();
        let pages_needed = blob.len().div_ceil(DATA_CAP).max(1);
        while self.chain.len() < pages_needed {
            let new_page = self.store.allocate_page()?;
            self.chain.push(new_page);
        }
        for i in 0..pages_needed {
            let start = i * DATA_CAP;
            let end = ((i + 1) * DATA_CAP).min(blob.len());
            let chunk = &blob[start..end];
            let next = if i + 1 < pages_needed {
                self.chain[i + 1].0
            } else {
                NO_NEXT
            };

            let mut page = [0u8; PAGE_SIZE];
            page[0..4].copy_from_slice(&CATALOG_MAGIC.to_le_bytes());
            page[6..8].copy_from_slice(&(chunk.len() as u16).to_le_bytes());
            page[8..16].copy_from_slice(&next.to_le_bytes());
            page[PAGE_HDR..PAGE_HDR + chunk.len()].copy_from_slice(chunk);
            self.store.write_page(self.chain[i], &page)?;
        }
        // Release any surplus tail pages when the blob shrank (e.g. after `drop_table`).
        // The last written page now terminates the next-chain (`NO_NEXT`), so on the next open the
        // reload stops before these pages — left allocated and still stamped `CATALOG_MAGIC` they
        // would be unreclaimable orphans (the free-list scan only reclaims `FREE_PAGE_MAGIC`). Drain
        // first so the `&mut self.chain` borrow is released before deallocating through `self.store`.
        let surplus: Vec<PageId> = self.chain.drain(pages_needed..).collect();
        for page in surplus {
            self.store.deallocate_page(page)?;
        }
        self.store.fsync()
    }

    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.tables.len() as u32).to_le_bytes());
        for t in &self.tables {
            buf.extend_from_slice(&t.id.0.to_le_bytes());
            wr_str(&mut buf, &t.name);
            buf.extend_from_slice(&(t.columns.len() as u32).to_le_bytes());
            for c in &t.columns {
                buf.push(coltype_to_u8(c.ty));
                // NUMERIC carries precision + scale; ARRAY carries its element tag.
                match c.ty {
                    ColumnType::Numeric { precision, scale } => {
                        buf.push(precision);
                        buf.push(scale);
                    },
                    ColumnType::Array(elem) => buf.push(coltype_to_u8(elem.column_type())),
                    // VECTOR carries its dimension as a u32 after the tag.
                    ColumnType::Vector(dim) => buf.extend_from_slice(&dim.to_le_bytes()),
                    // VARCHAR(n)/CHAR(n) carry their declared length as a u32 after the tag, so the
                    // declared type round-trips in DDL.
                    ColumnType::VarChar(n) | ColumnType::Char(n) => {
                        buf.extend_from_slice(&n.to_le_bytes());
                    },
                    _ => {},
                }
                buf.push(u8::from(c.nullable));
                wr_str(&mut buf, &c.name);
            }
        }
        buf
    }

    fn deserialize(blob: &[u8]) -> Result<Vec<TableSchema>> {
        let mut cur = 0usize;
        let count = rd_u32(blob, &mut cur)?;
        let mut tables = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let id = TableId(rd_u64(blob, &mut cur)?);
            let name = rd_str(blob, &mut cur)?;
            let ncol = rd_u32(blob, &mut cur)?;
            let mut columns = Vec::with_capacity(ncol as usize);
            for _ in 0..ncol {
                let tag = rd_u8(blob, &mut cur)?;
                // NUMERIC (tag 10) stores precision + scale; ARRAY (tag 12) stores element tag.
                let ty = if tag == 10 {
                    ColumnType::Numeric {
                        precision: rd_u8(blob, &mut cur)?,
                        scale: rd_u8(blob, &mut cur)?,
                    }
                } else if tag == 12 {
                    let elem = ArrayElem::from_column_type(u8_to_coltype(rd_u8(blob, &mut cur)?)?)
                        .ok_or_else(corrupt)?;
                    ColumnType::Array(elem)
                } else if tag == 15 {
                    // VECTOR (tag 15) stores its dimension as a u32.
                    ColumnType::Vector(rd_u32(blob, &mut cur)?)
                } else if tag == 16 {
                    // VARCHAR (tag 16) stores its declared length as a u32.
                    ColumnType::VarChar(rd_u32(blob, &mut cur)?)
                } else if tag == 17 {
                    // CHAR (tag 17) stores its declared length as a u32.
                    ColumnType::Char(rd_u32(blob, &mut cur)?)
                } else {
                    u8_to_coltype(tag)?
                };
                let nullable = rd_u8(blob, &mut cur)? != 0;
                let cname = rd_str(blob, &mut cur)?;
                columns.push(ColumnDef {
                    name: cname,
                    ty,
                    nullable,
                });
            }
            // The on-disk catalog format predates schemas; loaded tables default to `public`.
            tables.push(TableSchema {
                id,
                schema: PUBLIC_SCHEMA.to_owned(),
                name,
                columns,
            });
        }
        Ok(tables)
    }
}

// ── byte helpers ──────────────────────────────────────────────────────────────

fn corrupt() -> Error {
    Error::Io(std::io::Error::other("catalog: truncated or corrupt entry"))
}

fn rd_u32_at(b: &[u8], o: usize) -> u32 {
    bytemuck::pod_read_unaligned::<u32>(&b[o..o + 4])
}
fn rd_u16_at(b: &[u8], o: usize) -> u16 {
    bytemuck::pod_read_unaligned::<u16>(&b[o..o + 2])
}
fn rd_u64_at(b: &[u8], o: usize) -> u64 {
    bytemuck::pod_read_unaligned::<u64>(&b[o..o + 8])
}

fn rd_u8(b: &[u8], cur: &mut usize) -> Result<u8> {
    let byte = *b.get(*cur).ok_or_else(corrupt)?;
    *cur += 1;
    Ok(byte)
}
fn rd_u32(b: &[u8], cur: &mut usize) -> Result<u32> {
    if *cur + 4 > b.len() {
        return Err(corrupt());
    }
    let v = rd_u32_at(b, *cur);
    *cur += 4;
    Ok(v)
}
fn rd_u64(b: &[u8], cur: &mut usize) -> Result<u64> {
    if *cur + 8 > b.len() {
        return Err(corrupt());
    }
    let v = rd_u64_at(b, *cur);
    *cur += 8;
    Ok(v)
}
fn rd_str(b: &[u8], cur: &mut usize) -> Result<String> {
    let len = rd_u32(b, cur)? as usize;
    if *cur + len > b.len() {
        return Err(corrupt());
    }
    let s = String::from_utf8(b[*cur..*cur + len].to_vec()).map_err(|_| corrupt())?;
    *cur += len;
    Ok(s)
}
fn wr_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

const fn coltype_to_u8(t: ColumnType) -> u8 {
    match t {
        ColumnType::Bool => 0,
        ColumnType::Int => 1,
        ColumnType::Float => 2,
        ColumnType::Text => 3,
        ColumnType::Bytes => 4,
        ColumnType::Timestamp => 5,
        ColumnType::Date => 6,
        ColumnType::Time => 7,
        ColumnType::TimestampTz => 8,
        ColumnType::Uuid => 9,
        // Tag 10; precision + scale are serialized separately.
        ColumnType::Numeric { .. } => 10,
        ColumnType::Json => 11,
        // Tag 12; the element type tag is serialized separately.
        ColumnType::Array(_) => 12,
        ColumnType::Interval => 13,
        ColumnType::TimeTz => 14,
        // Tag 15; the dimension is serialized separately.
        ColumnType::Vector(_) => 15,
        // Tags 16/17; the declared length is serialized separately.
        ColumnType::VarChar(_) => 16,
        ColumnType::Char(_) => 17,
        // Tags 18/19; stored identically to Int (the declared width is metadata only).
        ColumnType::SmallInt => 18,
        ColumnType::BigInt => 19,
        // Tags 20/21; REAL stored as Float, JSONB as Json (declared type is metadata only).
        ColumnType::Real => 20,
        ColumnType::Jsonb => 21,
    }
}
fn u8_to_coltype(b: u8) -> Result<ColumnType> {
    Ok(match b {
        0 => ColumnType::Bool,
        1 => ColumnType::Int,
        2 => ColumnType::Float,
        3 => ColumnType::Text,
        4 => ColumnType::Bytes,
        5 => ColumnType::Timestamp,
        6 => ColumnType::Date,
        7 => ColumnType::Time,
        8 => ColumnType::TimestampTz,
        9 => ColumnType::Uuid,
        11 => ColumnType::Json,
        13 => ColumnType::Interval,
        14 => ColumnType::TimeTz,
        // SMALLINT/BIGINT carry no payload (stored as Int); decode them directly.
        18 => ColumnType::SmallInt,
        19 => ColumnType::BigInt,
        // REAL (stored as Float) / JSONB (stored as Json) carry no payload.
        20 => ColumnType::Real,
        21 => ColumnType::Jsonb,
        // An array's NUMERIC element is unconstrained (single tag byte, no precision/scale); a
        // column-level NUMERIC (tag 10) is decoded with its precision/scale before this fallback.
        10 => ColumnType::Numeric {
            precision: 0,
            scale: 0,
        },
        _ => return Err(corrupt()),
    })
}
