//! The column list of a foreign key's `ON DELETE SET NULL (cols)` action.
//!
//! The storage treaty's foreign key definition carries the action but no column list, so the list
//! lives in an engine-scoped SQL-layer catalog, `nusadb_fk_set_columns`, one row per foreign key
//! that has one: `(constraint name, columns)`. A foreign key's name is unique in the engine's
//! catalog, so the name alone identifies it; it is also stable across the operations that rebuild
//! a key under the same name (`TRUNCATE`, a column rename). Every creation of a foreign key writes
//! or clears its row ([`declare`]), so a dropped key's leftover row can never give a later key of
//! the same name a list; the drops forget the row too ([`forget`]).
//!
//! A key without a row sets every referencing column, the plain `SET NULL`.
//!
//! # Trust
//!
//! The catalog sits in the reserved `nusadb_*` namespace, which only a superuser may read or write
//! by name. A superuser can still write anything into it, so what it holds is never trusted:
//! a parent delete uses a list only when it names distinct referencing columns of the key, and
//! refuses the delete with an error naming the key otherwise ([`set_null_ordinals`]), rather than
//! null a column outside the key (which would leave the child pointing at a deleted parent). A
//! table of the catalog's name that does not have the catalog's shape is never read or written:
//! it holds no lists, and declaring a key with a list is refused while it is there.
//!
//! # Concurrency
//!
//! The catalog is created on first use, inside the transaction declaring the first key with a
//! list. Two transactions doing that at once both try to create it; the second gets the engine's
//! serialization conflict (`40001`) and can retry, by when the catalog exists.
#![allow(clippy::wildcard_imports)]

use nusadb_core::Tid;

use super::*;

/// Engine-scoped catalog of `ON DELETE SET` column lists.
const CATALOG: &str = "nusadb_fk_set_columns";

/// The catalog's column names, in order; both `TEXT NOT NULL`.
const CATALOG_COLUMNS: [&str; 2] = ["constraint_name", "columns"];

/// `(constraint_name, columns)`, both text.
const CATALOG_SCHEMA: [ColumnType; 2] = [ColumnType::Text; 2];

/// Encode a column list as one text value: each name as its length in bytes, a `:` and the name,
/// so any name (a quoted one holding a comma, say) round-trips.
fn encode(columns: &[String]) -> String {
    let mut out = String::new();
    for c in columns {
        out.push_str(&c.len().to_string());
        out.push(':');
        out.push_str(c);
    }
    out
}

/// Decode what [`encode`] wrote; `None` for a malformed value.
fn decode(mut text: &str) -> Option<Vec<String>> {
    let mut out = Vec::new();
    while !text.is_empty() {
        let (len, rest) = text.split_once(':')?;
        let len: usize = len.parse().ok()?;
        let name = rest.get(..len)?;
        out.push(name.to_owned());
        text = rest.get(len..)?;
    }
    Some(out)
}

/// What the catalog's name resolves to.
enum Found {
    /// No table of that name: no key has a list.
    Missing,
    /// The catalog.
    Catalog(nusadb_core::TableId),
    /// Some other table under the catalog's name: holds no lists, and is never written.
    Foreign,
}

/// Whether `table` has the catalog's shape.
fn is_catalog_shape(table: &TableSchema) -> bool {
    table.columns.len() == CATALOG_COLUMNS.len()
        && table
            .columns
            .iter()
            .zip(CATALOG_COLUMNS)
            .all(|(c, name)| c.name == name && c.ty == ColumnType::Text && !c.nullable)
}

fn find(engine: &dyn StorageEngine, txn: TxnId) -> Result<Found, Error> {
    Ok(match engine.lookup_table_as_of(txn, CATALOG)? {
        None => Found::Missing,
        Some(t) if is_catalog_shape(&t) => Found::Catalog(t.id),
        Some(_) => Found::Foreign,
    })
}

/// One catalog row: its address, the key it names, and its column list (`None` when the stored
/// value does not decode).
type CatalogRow = (Tid, String, Option<Vec<String>>);

/// The catalog rows `txn` sees whose constraint name satisfies `pick`.
fn rows(
    engine: &dyn StorageEngine,
    txn: TxnId,
    pick: impl Fn(&str) -> bool,
) -> Result<Vec<CatalogRow>, Error> {
    let Found::Catalog(catalog) = find(engine, txn)? else {
        return Ok(Vec::new());
    };
    let mut scan = engine.scan(txn, catalog)?;
    let mut out = Vec::new();
    while let Some((tid, bytes)) = scan.try_next()? {
        let row = row::decode(&bytes, &CATALOG_SCHEMA)?;
        if let Some(ast::Value::Text(name)) = row.first()
            && pick(name)
        {
            let columns = match row.get(1) {
                Some(ast::Value::Text(columns)) => decode(columns),
                _ => None,
            };
            out.push((tid, name.clone(), columns));
        }
    }
    Ok(out)
}

fn ensure_catalog(engine: &dyn StorageEngine, txn: TxnId) -> Result<nusadb_core::TableId, Error> {
    match find(engine, txn)? {
        Found::Catalog(id) => return Ok(id),
        Found::Foreign => {
            return Err(Error::Coded {
                message: format!(
                    "cannot record an ON DELETE SET NULL column list: a table named \
                     \"{CATALOG}\" exists and is not the catalog this engine keeps under that \
                     name; rename or drop it"
                ),
                sqlstate: "42P07",
            });
        },
        Found::Missing => {},
    }
    let columns = CATALOG_COLUMNS
        .into_iter()
        .map(|name| ColumnDef {
            name: name.to_owned(),
            ty: ColumnType::Text,
            nullable: false,
        })
        .collect();
    let def = TableDef {
        schema: nusadb_core::PUBLIC_SCHEMA.to_owned(),
        name: CATALOG.to_owned(),
        columns,
    };
    Ok(engine.create_table(txn, &def)?)
}

/// Record the `ON DELETE SET` column list of the foreign key `name` at its creation: a key with a
/// list gets a row, any other has none.
///
/// # Errors
/// Storage errors; `42P07` when a table that is not the catalog holds its name and `columns` is
/// not empty.
pub(super) fn declare(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
    columns: &[String],
) -> Result<(), Error> {
    forget(engine, txn, name)?;
    if columns.is_empty() {
        return Ok(());
    }
    let catalog = ensure_catalog(engine, txn)?;
    let row = [
        ast::Value::Text(name.to_owned()),
        ast::Value::Text(encode(columns)),
    ];
    engine.insert(txn, catalog, &row::encode(&row, &CATALOG_SCHEMA)?)?;
    Ok(())
}

/// What the catalog holds for one foreign key.
enum Stored {
    /// No row: the key sets every referencing column.
    None,
    /// A list that decodes.
    List(Vec<String>),
    /// A row that does not decode, or more than one row.
    Invalid,
}

/// What the catalog holds for the foreign key `name`.
fn stored(engine: &dyn StorageEngine, txn: TxnId, name: &str) -> Result<Stored, Error> {
    let mut found = rows(engine, txn, |n| n == name)?;
    Ok(match (found.len(), found.pop()) {
        (0, _) => Stored::None,
        (1, Some((_, _, Some(columns)))) => Stored::List(columns),
        // An undecodable value, or more than one row for one key (which this module never writes).
        _ => Stored::Invalid,
    })
}

/// The drop of the foreign key `name`: forget its row.
///
/// # Errors
/// Storage errors.
pub(super) fn forget(engine: &dyn StorageEngine, txn: TxnId, name: &str) -> Result<(), Error> {
    let victims = rows(engine, txn, |n| n == name)?;
    let Found::Catalog(catalog) = find(engine, txn)? else {
        return Ok(());
    };
    for (tid, _, _) in victims {
        engine.delete(txn, catalog, tid)?;
    }
    Ok(())
}

/// The error for a foreign key whose stored column list cannot be used.
fn invalid_list(fk: &str, why: &str) -> Error {
    Error::Coded {
        message: format!(
            "foreign key \"{fk}\": its ON DELETE SET NULL column list in \"{CATALOG}\" is \
             invalid ({why}); the delete is refused rather than null columns outside the key. \
             Drop and re-create the foreign key to repair it"
        ),
        sqlstate: "55000",
    }
}

/// A copy of the foreign key `from` made as `to` (on a partition): the copy sets the same columns.
///
/// # Errors
/// Storage errors; `55000` when `from`'s stored list does not decode.
pub(super) fn copy(
    engine: &dyn StorageEngine,
    txn: TxnId,
    from: &str,
    to: &str,
) -> Result<(), Error> {
    let columns = match stored(engine, txn, from)? {
        Stored::None => Vec::new(),
        Stored::List(columns) => columns,
        Stored::Invalid => return Err(invalid_list(from, "it does not decode")),
    };
    declare(engine, txn, to, &columns)
}

/// A referencing column of the foreign key `name` renamed from `from` to `to`: its list follows.
/// A list that does not decode is left as it is (a delete refuses it).
///
/// # Errors
/// Storage errors.
pub(super) fn rename_column(
    engine: &dyn StorageEngine,
    txn: TxnId,
    name: &str,
    from: &str,
    to: &str,
) -> Result<(), Error> {
    let Stored::List(columns) = stored(engine, txn, name)? else {
        return Ok(());
    };
    if !columns.iter().any(|c| c == from) {
        return Ok(());
    }
    let renamed: Vec<String> = columns
        .into_iter()
        .map(|c| if c == from { to.to_owned() } else { c })
        .collect();
    declare(engine, txn, name, &renamed)
}

/// The ordinals (into the child table) of the columns the foreign key `fk`'s `ON DELETE SET NULL` sets:
/// its column list when it has one, else every referencing column (`fk_ordinals`, the child's
/// ordinals of `fk.child_columns`, in order).
///
/// The stored list is used only when it decodes and names distinct referencing columns of `fk`:
/// anything else refuses the delete (`55000`, naming the key), never nulls a column outside it.
///
/// # Errors
/// Storage errors; `55000` for an unusable stored list.
pub(super) fn set_null_ordinals(
    fk: &nusadb_core::ForeignKeyDef,
    fk_ordinals: &[usize],
    engine: &dyn StorageEngine,
    txn: TxnId,
) -> Result<Vec<usize>, Error> {
    let listed = match stored(engine, txn, &fk.name)? {
        Stored::None => return Ok(fk_ordinals.to_vec()),
        Stored::Invalid => return Err(invalid_list(&fk.name, "it does not decode")),
        Stored::List(listed) => listed,
    };
    if listed.is_empty() {
        return Err(invalid_list(&fk.name, "it is empty"));
    }
    let mut ordinals = Vec::with_capacity(listed.len());
    for name in &listed {
        let Some(at) = fk.child_columns.iter().position(|c| c == name) else {
            return Err(invalid_list(
                &fk.name,
                &format!("\"{name}\" is not a referencing column of the key"),
            ));
        };
        let Some(&ordinal) = fk_ordinals.get(at) else {
            return Err(invalid_list(&fk.name, "the key's columns do not resolve"));
        };
        if ordinals.contains(&ordinal) {
            return Err(invalid_list(
                &fk.name,
                &format!("\"{name}\" is listed twice"),
            ));
        }
        ordinals.push(ordinal);
    }
    Ok(ordinals)
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};

    #[test]
    fn column_lists_round_trip() {
        for cols in [
            vec![],
            vec!["a".to_owned()],
            vec![
                "d_id".to_owned(),
                "x,y".to_owned(),
                "12:3".to_owned(),
                "é".to_owned(),
            ],
        ] {
            assert_eq!(decode(&encode(&cols)), Some(cols));
        }
        assert_eq!(decode("3:ab"), None);
        assert_eq!(decode("x:ab"), None);
    }
}
