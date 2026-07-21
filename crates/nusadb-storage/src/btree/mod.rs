//! B-tree index — one node per 8 KiB page.
//!
//! Nodes split at 50% capacity. Keys are byte-comparable. Used both as the primary
//! storage shape for catalog tables and as a building block for secondary indexes.

pub mod node;
pub mod tree;

pub use node::{Node, NodeType};
pub use tree::BTree;
