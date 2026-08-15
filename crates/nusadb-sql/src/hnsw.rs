//! In-memory HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index for
//! `VECTOR(n)` columns — **increment 1: the standalone algorithm core + its recall oracle**.
//!
//! The current KNN path (`ORDER BY v <=> q LIMIT k`) is an exact `O(n·dim)` scan per query. HNSW
//! trades a small, bounded loss of recall for roughly logarithmic search by navigating a multi-layer
//! proximity graph: upper layers are sparse "express lanes" for coarse approach, the dense bottom
//! layer (layer 0) holds every point for fine search. Build links each new point to its `m` nearest
//! neighbours per layer (`m0 = 2·m` at layer 0) chosen by the diversity heuristic of Malkov & Yashunin
//! (Algorithm 4), and search is a greedy descent through the layers followed by an `ef`-width beam
//! search at layer 0.
//!
//! This increment is deliberately self-contained and dependency-free: a pure in-memory index over
//! `Vec<f32>` points using the distance metrics in [`crate::vector`], plus a recall test that pins it
//! against a brute-force exact oracle. Level assignment uses a seeded PRNG so a given (seed, insert
//! order) reproduces the same graph — important for deterministic tests and debugging. **Not yet
//! wired in:** on-disk page-backed persistence and routing a planner/executor KNN query to the index
//! are later increments; this commit lands and proves the algorithm first.

#![allow(
    clippy::wildcard_imports,
    clippy::indexing_slicing,
    reason = "node ids and per-layer indices are construction invariants of this index: every id \
              originates from `self.nodes`, and every layer index is bounded by the owning node's \
              own neighbour-list length, so the indexing cannot go out of bounds"
)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "the PRNG mantissa extraction and the geometric level arithmetic intentionally cast \
              between integer and float; the values involved (a small degree bound, a level capped \
              at 31, non-negative `-ln(u)`) are well within range"
)]

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use crate::error::Error;

/// Hasher for the `u32` node ids the beam search dedupes on.
///
/// [`HashSet`]'s default is `SipHash` — keyed and collision-resistant, priced for keys an attacker
/// might choose. These keys are dense integers this module mints itself, and the set is hit once per
/// node the search visits, which is the hottest thing a build does. A `splitmix64` finalizer spreads
/// them across buckets just as evenly for a fraction of the work.
///
/// The set is only ever asked for membership, never iterated, so the bucket order this changes is not
/// observable — the built graph is unaffected (pinned by
/// `cosine_build_graph_matches_its_pinned_digest`).
#[derive(Default)]
struct NodeIdHasher(u64);

impl Hasher for NodeIdHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write_u32(&mut self, n: u32) {
        let mut z = u64::from(n).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        // Mix into the running state rather than replacing it, so a future composite key does not
        // silently degenerate to "last field wins". For the single-`u32` key used here the state is
        // still zero on entry, so this is the plain finalizer.
        self.0 ^= z ^ (z >> 31);
    }

    /// Unreachable for a `u32` key (`Hash for u32` calls [`Self::write_u32`]), but a `Hasher` must
    /// define it; fold the bytes so it stays a valid hash rather than a constant.
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// A node-id set hashed by [`NodeIdHasher`].
type NodeIdSet = HashSet<u32, BuildHasherDefault<NodeIdHasher>>;

/// The distance metric an index is built and searched under.
///
/// Every variant returns a *distance* (smaller is closer), one per vector-distance operator: `<=>`
/// (cosine), `<->` (L2), `<#>` (negative inner product), `<+>` (L1).
///
/// An index serves exactly the metric it was built under. Nearest neighbours differ between metrics,
/// so a graph built for one cannot answer another's query — see [`Metric::for_operator`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Euclidean distance.
    L2,
    /// Cosine distance `1 − cosθ`.
    Cosine,
    /// Negative inner product (so a larger dot product is a smaller distance).
    InnerProduct,
    /// Manhattan (L1, taxicab) distance.
    L1,
}

impl Metric {
    /// The metric an index must have been built under to answer a query written with `op`, or `None`
    /// if `op` is not a vector-distance operator.
    ///
    /// Nearest neighbours are metric-specific: the closest vectors under L2 are not the closest under
    /// cosine. So this is what lets the planner refuse to answer one distance's query from another
    /// distance's graph — routing on the operator alone would return confidently wrong rows.
    #[must_use]
    pub const fn for_operator(op: crate::ast::BinaryOp) -> Option<Self> {
        match op {
            crate::ast::BinaryOp::VectorDistance => Some(Self::Cosine),
            crate::ast::BinaryOp::VectorL2Distance => Some(Self::L2),
            crate::ast::BinaryOp::VectorNegInnerProduct => Some(Self::InnerProduct),
            crate::ast::BinaryOp::VectorL1Distance => Some(Self::L1),
            _ => None,
        }
    }

    /// The operator-class name that selects this metric in `CREATE INDEX ... USING hnsw (col <name>)`.
    #[must_use]
    pub const fn operator_class(self) -> &'static str {
        match self {
            Self::L2 => "vector_l2_ops",
            Self::Cosine => "vector_cosine_ops",
            Self::InnerProduct => "vector_ip_ops",
            Self::L1 => "vector_l1_ops",
        }
    }

    /// The metric an operator-class name selects, or `None` if the name is not one of them.
    #[must_use]
    pub fn from_operator_class(name: &str) -> Option<Self> {
        [Self::L2, Self::Cosine, Self::InnerProduct, Self::L1]
            .into_iter()
            .find(|m| name.eq_ignore_ascii_case(m.operator_class()))
    }

    /// Distance between two vectors under this metric, or `None` if their dimensions differ.
    ///
    /// This is the metric's definition. The index's own internal wrapper turns the mismatch into
    /// `+∞` so such a pair is never chosen as a neighbour; an exact scan wants the `None` instead, so
    /// it can skip the row rather than rank it as infinitely far.
    #[must_use]
    pub fn exact_distance(self, a: &[f32], b: &[f32]) -> Option<f64> {
        match self {
            Self::L2 => crate::vector::l2_distance(a, b),
            Self::Cosine => crate::vector::cosine_distance(a, b),
            Self::InnerProduct => crate::vector::neg_inner_product(a, b),
            Self::L1 => crate::vector::l1_distance(a, b),
        }
    }

    /// Distance between two equal-length vectors. A dimension mismatch (which the index guards
    /// against at insert time) maps to `+∞` so such a pair is never selected as a neighbour.
    fn distance(self, a: &[f32], b: &[f32]) -> f64 {
        self.exact_distance(a, b).unwrap_or(f64::INFINITY)
    }

    /// The per-vector constant this metric would otherwise recompute on every comparison, cached on
    /// the node at insert time. Only cosine has one (the norm); the others return `0.0`, which they
    /// never read.
    fn cached_term(self, v: &[f32]) -> f64 {
        match self {
            Self::Cosine => crate::vector::norm(v),
            Self::L2 | Self::InnerProduct | Self::L1 => 0.0,
        }
    }

    /// [`Self::distance`] with both operands' [`Self::cached_term`] supplied.
    ///
    /// Bit-identical to `distance(a, b)` when the terms are the ones `cached_term` produces for `a`
    /// and `b` — for cosine it is literally the same expression with the two norm reductions hoisted
    /// out, and the other metrics ignore the terms and call straight through.
    fn distance_cached(self, a: &[f32], a_term: f64, b: &[f32], b_term: f64) -> f64 {
        match self {
            Self::Cosine => crate::vector::cosine_distance_with_norms(a, a_term, b, b_term)
                .unwrap_or(f64::INFINITY),
            Self::L2 | Self::InnerProduct | Self::L1 => self.distance(a, b),
        }
    }
}

/// Build/search tunables. `m` is the neighbour degree of the upper layers; layer 0 uses `2·m`.
/// `ef_construction` is the beam width while building (higher = better graph, slower build).
#[derive(Clone, Copy, Debug)]
pub struct HnswParams {
    /// Max neighbours per node on layers ≥ 1.
    pub m: usize,
    /// Beam width during construction.
    pub ef_construction: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
        }
    }
}

/// One indexed point: its vector and, per layer it participates in, the ids of its neighbours.
/// `neighbours.len() - 1` is the node's top layer.
#[derive(Debug)]
struct Node {
    vector: Vec<f32>,
    /// [`Metric::cached_term`] of `vector` — for cosine, its norm. A build compares each node against
    /// hundreds of others, and without this every one of those comparisons would recompute the norms
    /// of *both* sides from scratch. Derived purely from `vector`, so it is recomputed on load rather
    /// than persisted (see [`HnswIndex::deserialize`]) and the on-disk blob format is unchanged.
    term: f64,
    neighbours: Vec<Vec<u32>>,
}

impl Node {
    /// The only constructor — `term` is derived here and nowhere else, so it cannot be built out of
    /// step with `vector`. Every site that materializes a node (a fresh insert, a deserialized blob,
    /// a reload from persisted parts) goes through this, and `vector` is never mutated afterwards.
    fn new(metric: Metric, vector: Vec<f32>, neighbours: Vec<Vec<u32>>) -> Self {
        Self {
            term: metric.cached_term(&vector),
            vector,
            neighbours,
        }
    }
}

/// A candidate during search/build: a node id tagged with its distance to the focus point. Ordered
/// by distance (total order via `f64::total_cmp`, id as a stable tie-break) so it can drive both a
/// nearest-first min-heap (via [`Reverse`]) and a farthest-first max-heap.
#[derive(Clone, Copy)]
struct Candidate {
    dist: f64,
    id: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.id.cmp(&other.id))
    }
}

/// An in-memory HNSW index over fixed-dimension `f32` vectors.
#[derive(Debug)]
pub struct HnswIndex {
    dim: usize,
    metric: Metric,
    params: HnswParams,
    /// Reciprocal of `ln(m)` — the level-assignment scale (`mL` in the paper).
    level_mult: f64,
    nodes: Vec<Node>,
    entry: Option<u32>,
    /// `xorshift64*` state for level assignment (seeded for reproducible builds).
    rng: u64,
}

impl HnswIndex {
    /// Create an empty index over `dim`-dimensional vectors under `metric`, with `seed` driving the
    /// (otherwise random) level assignment so a build is reproducible.
    #[must_use]
    pub fn new(dim: usize, metric: Metric, params: HnswParams, seed: u64) -> Self {
        let m = params.m.max(2);
        Self {
            dim,
            metric,
            params: HnswParams { m, ..params },
            level_mult: 1.0 / (m as f64).ln(),
            nodes: Vec::new(),
            entry: None,
            // Avoid a zero state (xorshift's fixed point); fold in a constant.
            rng: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Number of indexed points.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// The dimension every indexed vector has.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// The distance metric this graph was built under. A rebuild must reuse it, or the rebuilt graph
    /// would answer a different question than the one the index was declared for.
    #[must_use]
    pub const fn metric(&self) -> Metric {
        self.metric
    }

    /// Whether the index holds no points.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// A `(0, 1]` uniform draw from the index PRNG (`xorshift64*`).
    fn next_unit(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        // Top 53 bits → a double in [0, 1); shift to (0, 1] so ln() is finite.
        let u = ((v >> 11) as f64) / ((1u64 << 53) as f64);
        if u <= 0.0 { f64::MIN_POSITIVE } else { u }
    }

    /// Draw a node's top layer from the geometric distribution `floor(-ln(U)·mL)`, capped so a
    /// pathological draw can't allocate an absurd number of layers.
    fn random_level(&mut self) -> usize {
        let r = -self.next_unit().ln() * self.level_mult;
        (r as usize).min(31)
    }

    /// Distance from `probe` (with its cached term) to the stored node `id`.
    fn distance_to(&self, probe: &[f32], probe_term: f64, id: u32) -> f64 {
        let node = &self.nodes[id as usize];
        self.metric
            .distance_cached(probe, probe_term, &node.vector, node.term)
    }

    /// The max neighbour degree for `layer` (`2·m` at layer 0, else `m`).
    const fn max_degree(&self, layer: usize) -> usize {
        if layer == 0 {
            self.params.m * 2
        } else {
            self.params.m
        }
    }

    /// Insert `vector`. Returns its node id.
    ///
    /// # Errors
    /// [`Error::Unsupported`] if `vector`'s length differs from the index dimension.
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<u32, Error> {
        self.insert_reporting(vector).map(|(id, _)| id)
    }

    /// Insert `vector`, returning its node id **and** the ids of the existing nodes whose neighbour
    /// lists this insert changed (its chosen neighbours, which are linked back and possibly pruned).
    /// Together with the new id, that is exactly the set of nodes whose persisted adjacency is now
    /// stale — so an incremental persister can rewrite only those rows instead of the whole graph.
    ///
    /// # Errors
    /// [`Error::Unsupported`] if `vector`'s length differs from the index dimension.
    pub fn insert_reporting(&mut self, vector: Vec<f32>) -> Result<(u32, Vec<u32>), Error> {
        if vector.len() != self.dim {
            return Err(Error::Unsupported(format!(
                "HNSW expects dimension {}, got {}",
                self.dim,
                vector.len()
            )));
        }
        let level = self.random_level();
        let id = u32::try_from(self.nodes.len())
            .map_err(|_| Error::Unsupported("HNSW index is full (u32 node ids)".to_owned()))?;
        let node = Node::new(self.metric, vector, vec![Vec::new(); level + 1]);
        self.nodes.push(node);

        let Some(entry) = self.entry else {
            // First point becomes the entry point.
            self.entry = Some(id);
            return Ok((id, Vec::new()));
        };

        let query = self.nodes[id as usize].vector.clone();
        let query_term = self.nodes[id as usize].term;
        let top = self.top_level();

        // Phase 1: greedily descend the layers above this node's top, narrowing to one entry point.
        let mut ep = entry;
        for layer in (level + 1..=top).rev() {
            ep = self.greedy_nearest(&query, query_term, ep, layer);
        }

        // Phase 2: from this node's top down to layer 0, beam-search, pick neighbours, link both ways.
        let mut touched: Vec<u32> = Vec::new();
        let mut entry_points = vec![ep];
        for layer in (0..=level.min(top)).rev() {
            let found = self.search_layer(
                &query,
                query_term,
                &entry_points,
                self.params.ef_construction,
                layer,
            );
            let degree = self.max_degree(layer);
            let chosen = self.select_neighbours(&found, degree);
            touched.extend_from_slice(&chosen);
            self.connect(id, &chosen, layer);
            entry_points = found.into_iter().map(|c| c.id).collect();
            if entry_points.is_empty() {
                entry_points.push(ep);
            }
        }

        // A taller new node becomes the entry point.
        if level > top {
            self.entry = Some(id);
        }
        touched.sort_unstable();
        touched.dedup();
        Ok((id, touched))
    }

    /// The top layer index of the current entry point (0 if the index has a single layer).
    fn top_level(&self) -> usize {
        self.entry
            .map_or(0, |e| self.nodes[e as usize].neighbours.len() - 1)
    }

    /// Walk greedily to the node nearest `query` on `layer`, starting from `from`.
    fn greedy_nearest(&self, query: &[f32], query_term: f64, from: u32, layer: usize) -> u32 {
        let mut best = from;
        let mut best_dist = self.distance_to(query, query_term, from);
        loop {
            let mut improved = false;
            if let Some(node) = self.nodes.get(best as usize)
                && let Some(neighbours) = node.neighbours.get(layer)
            {
                for &n in neighbours {
                    let d = self.distance_to(query, query_term, n);
                    if d < best_dist {
                        best_dist = d;
                        best = n;
                        improved = true;
                    }
                }
            }
            if !improved {
                return best;
            }
        }
    }

    /// Beam search on `layer`: return up to `ef` nodes nearest `query`, reachable from
    /// `entry_points`. Classic HNSW `SEARCH-LAYER` with a nearest-first candidate min-heap and a
    /// farthest-first result max-heap bounded to `ef`.
    fn search_layer(
        &self,
        query: &[f32],
        query_term: f64,
        entry_points: &[u32],
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited = NodeIdSet::default();
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        let mut result: BinaryHeap<Candidate> = BinaryHeap::new();

        for &ep in entry_points {
            if visited.insert(ep) {
                let d = self.distance_to(query, query_term, ep);
                candidates.push(Reverse(Candidate { dist: d, id: ep }));
                result.push(Candidate { dist: d, id: ep });
            }
        }
        // Keep the result set bounded to `ef` from the start.
        while result.len() > ef {
            result.pop();
        }

        while let Some(Reverse(current)) = candidates.pop() {
            let farthest = result.peek().map_or(f64::INFINITY, |c| c.dist);
            if current.dist > farthest && result.len() >= ef {
                break; // every remaining candidate is farther than our worst keeper
            }
            // Borrow the neighbour list in place. Everything in this loop — `distance_to`, the heaps,
            // the visited set — reads `self` immutably or touches only locals, so there is no need to
            // copy the list out; doing so would allocate once per node the search visits.
            let neighbours: &[u32] = self
                .nodes
                .get(current.id as usize)
                .and_then(|node| node.neighbours.get(layer))
                .map_or(&[], Vec::as_slice);
            for &n in neighbours {
                if !visited.insert(n) {
                    continue;
                }
                let d = self.distance_to(query, query_term, n);
                let worst = result.peek().map_or(f64::INFINITY, |c| c.dist);
                if d < worst || result.len() < ef {
                    candidates.push(Reverse(Candidate { dist: d, id: n }));
                    result.push(Candidate { dist: d, id: n });
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = result.into_vec();
        out.sort_unstable(); // nearest first
        out
    }

    /// The neighbour-selection heuristic (Malkov & Yashunin Algorithm 4): from `candidates` (nearest
    /// first), keep a node only if it is closer to the query than to every already-kept neighbour,
    /// up to `m`. This spreads links across directions instead of clumping on the nearest cluster,
    /// which is what gives HNSW its high recall.
    ///
    /// A kept neighbour sitting at distance **0** from the query point is excluded from that test,
    /// and it has to be. Such a neighbour is an exact duplicate of the query point, so
    /// `dist(c, kept)` equals `dist(c, query)` for *every* candidate `c` and the strict `<` never
    /// holds — one twin vetoes the entire candidate list. When this runs from [`Self::connect`] to
    /// prune a node's links, the node is left with a single link, to its own twin. That pair is a
    /// sink: a greedy search reaching either can only step to the other, and when one of them is the
    /// entry point *every* search ends there. A single duplicated row made the whole index answer
    /// every query with that row, silently. A twin carries no directional information — it covers
    /// all directions equally — so it has no business vetoing anything.
    ///
    /// Excluding only the zero-distance case is deliberate. Refilling the list from the rejected
    /// candidates instead (the reference implementation's `keepPrunedConnections`) also clears the
    /// collapse, but it pins every node at the degree cap: measured against this build, 6-14x the
    /// build time and ~10% larger graphs, for a recall gain of about 0.01. The exclusion costs
    /// nothing — on data with no exact duplicates no kept neighbour is ever at distance 0, so the
    /// graph it produces is byte-identical to before (the pinned digest test proves it).
    fn select_neighbours(&self, candidates: &[Candidate], m: usize) -> Vec<u32> {
        let mut kept: Vec<Candidate> = Vec::with_capacity(m);
        for &cand in candidates {
            if kept.len() >= m {
                break;
            }
            let cand_node = &self.nodes[cand.id as usize];
            let (cand_vec, cand_term) = (&cand_node.vector, cand_node.term);
            let closer_to_query_than_to_kept = kept
                .iter()
                .filter(|k| k.dist > 0.0)
                .all(|k| cand.dist < self.distance_to(cand_vec, cand_term, k.id));
            if closer_to_query_than_to_kept {
                kept.push(cand);
            }
        }
        kept.into_iter().map(|c| c.id).collect()
    }

    /// Link `id` to each of `neighbours` on `layer` (both directions), pruning any neighbour whose
    /// degree now exceeds the layer cap by re-running the selection heuristic over its links.
    fn connect(&mut self, id: u32, neighbours: &[u32], layer: usize) {
        for &n in neighbours {
            self.nodes[id as usize].neighbours[layer].push(n);
            self.nodes[n as usize].neighbours[layer].push(id);
        }
        let degree = self.max_degree(layer);
        for &n in neighbours {
            if self.nodes[n as usize].neighbours[layer].len() <= degree {
                continue;
            }
            let n_vec = self.nodes[n as usize].vector.clone();
            let n_term = self.nodes[n as usize].term;
            let mut cands: Vec<Candidate> = self.nodes[n as usize].neighbours[layer]
                .iter()
                .map(|&x| Candidate {
                    dist: self.distance_to(&n_vec, n_term, x),
                    id: x,
                })
                .collect();
            cands.sort_unstable(); // nearest first
            let pruned = self.select_neighbours(&cands, degree);
            self.nodes[n as usize].neighbours[layer] = pruned;
        }
    }

    /// Approximate `k` nearest neighbours of `query`, nearest first, as `(id, distance)`. `ef`
    /// (clamped to ≥ `k`) is the search beam width — larger trades latency for recall.
    ///
    /// # Errors
    /// [`Error::Unsupported`] if `query`'s length differs from the index dimension.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(u32, f64)>, Error> {
        if query.len() != self.dim {
            return Err(Error::Unsupported(format!(
                "HNSW expects dimension {}, got {}",
                self.dim,
                query.len()
            )));
        }
        let Some(entry) = self.entry else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }

        // The query is external, so its term is computed once here and reused for every comparison
        // this search makes — the same saving the stored nodes get from their cached `term`.
        let query_term = self.metric.cached_term(query);
        let top = self.top_level();
        let mut ep = entry;
        for layer in (1..=top).rev() {
            ep = self.greedy_nearest(query, query_term, ep, layer);
        }
        let beam = ef.max(k);
        let found = self.search_layer(query, query_term, &[ep], beam, 0);
        Ok(found.into_iter().take(k).map(|c| (c.id, c.dist)).collect())
    }

    /// Serialize the built graph to a compact little-endian byte blob (see [`Self::deserialize`]).
    /// Captures every field needed to reconstruct an identical index without rebuilding: dimension,
    /// metric, params, RNG state, entry point, and each node's vector and per-layer neighbour lists.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "dimension, param, and node/neighbour counts are far below u32::MAX for any real index"
    )]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.push(match self.metric {
            Metric::L2 => 0,
            Metric::Cosine => 1,
            Metric::InnerProduct => 2,
            Metric::L1 => 3,
        });
        out.extend_from_slice(&(self.params.m as u32).to_le_bytes());
        out.extend_from_slice(&(self.params.ef_construction as u32).to_le_bytes());
        out.extend_from_slice(&self.rng.to_le_bytes());
        match self.entry {
            Some(e) => {
                out.push(1);
                out.extend_from_slice(&e.to_le_bytes());
            },
            None => out.push(0),
        }
        out.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        for node in &self.nodes {
            for &x in &node.vector {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out.extend_from_slice(&(node.neighbours.len() as u32).to_le_bytes());
            for layer in &node.neighbours {
                out.extend_from_slice(&(layer.len() as u32).to_le_bytes());
                for &nb in layer {
                    out.extend_from_slice(&nb.to_le_bytes());
                }
            }
        }
        out
    }

    /// Reconstruct an index from [`Self::serialize`] output, or `None` if the bytes are truncated,
    /// carry trailing garbage, or name an unknown metric. The result is the exact index that was
    /// serialized, so a search over it returns identical results without any rebuild.
    #[must_use]
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        let mut r = ByteReader::new(bytes);
        let dim = r.u32()? as usize;
        let metric = match r.u8()? {
            0 => Metric::L2,
            1 => Metric::Cosine,
            2 => Metric::InnerProduct,
            3 => Metric::L1,
            _ => return None,
        };
        let m = (r.u32()? as usize).max(2);
        let ef_construction = r.u32()? as usize;
        let rng = r.u64()?;
        let entry = match r.u8()? {
            0 => None,
            1 => Some(r.u32()?),
            _ => return None,
        };
        let node_count = r.u32()? as usize;
        // Cap the pre-allocation by the remaining byte count so a corrupt count cannot request a
        // pathological allocation (each node needs at least its vector + a layer count).
        let mut nodes = Vec::with_capacity(node_count.min(bytes.len()));
        for _ in 0..node_count {
            let mut vector = Vec::with_capacity(dim.min(bytes.len()));
            for _ in 0..dim {
                vector.push(r.f32()?);
            }
            let layer_count = r.u32()? as usize;
            let mut neighbours = Vec::with_capacity(layer_count.min(bytes.len()));
            for _ in 0..layer_count {
                let nb_count = r.u32()? as usize;
                let mut layer = Vec::with_capacity(nb_count.min(bytes.len()));
                for _ in 0..nb_count {
                    layer.push(r.u32()?);
                }
                neighbours.push(layer);
            }
            // `term` is derived from `vector`, so it is recomputed here rather than stored — the blob
            // format is untouched and an older blob loads unchanged.
            nodes.push(Node::new(metric, vector, neighbours));
        }
        if !r.at_end() {
            return None;
        }
        Some(Self {
            dim,
            metric,
            params: HnswParams { m, ef_construction },
            level_mult: 1.0 / (m as f64).ln(),
            nodes,
            entry,
            rng,
        })
    }

    /// Serialize one node's per-layer neighbour lists to bytes, for storing that node on its own.
    /// See [`Self::deserialize_adjacency`].
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "a node's layer and neighbour counts are far below u32::MAX"
    )]
    pub fn serialize_adjacency(neighbours: &[Vec<u32>]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(neighbours.len() as u32).to_le_bytes());
        for layer in neighbours {
            out.extend_from_slice(&(layer.len() as u32).to_le_bytes());
            for &nb in layer {
                out.extend_from_slice(&nb.to_le_bytes());
            }
        }
        out
    }

    /// Parse bytes written by [`Self::serialize_adjacency`], or `None` if truncated / trailing.
    #[must_use]
    pub fn deserialize_adjacency(bytes: &[u8]) -> Option<Vec<Vec<u32>>> {
        let mut r = ByteReader::new(bytes);
        let layer_count = r.u32()? as usize;
        let mut neighbours = Vec::with_capacity(layer_count.min(bytes.len()));
        for _ in 0..layer_count {
            let count = r.u32()? as usize;
            let mut layer = Vec::with_capacity(count.min(bytes.len()));
            for _ in 0..count {
                layer.push(r.u32()?);
            }
            neighbours.push(layer);
        }
        if !r.at_end() {
            return None;
        }
        Some(neighbours)
    }

    /// A node's per-layer neighbour lists, or `None` if `id` is out of range — to persist one node.
    #[must_use]
    pub fn node_neighbours(&self, id: usize) -> Option<&[Vec<u32>]> {
        self.nodes.get(id).map(|node| node.neighbours.as_slice())
    }

    /// Reconstruct an index from its parts — each node as `(vector, per-layer neighbours)`, in node-id
    /// order, as persisted per node. The entry point is recomputed as a highest-layer node and the RNG
    /// is reset; neither affects search results, only the layer assignment of *future* inserts.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the node count is bounded by u32 (insert caps ids at u32::MAX)"
    )]
    pub fn from_parts(
        dim: usize,
        metric: Metric,
        params: HnswParams,
        nodes: Vec<(Vec<f32>, Vec<Vec<u32>>)>,
    ) -> Self {
        let m = params.m.max(2);
        let mut index = Self {
            dim,
            metric,
            params: HnswParams {
                m,
                ef_construction: params.ef_construction,
            },
            level_mult: 1.0 / (m as f64).ln(),
            nodes: nodes
                .into_iter()
                .map(|(vector, neighbours)| Node::new(metric, vector, neighbours))
                .collect(),
            entry: None,
            // A fresh non-zero state (matching `new`'s fold constant) so future inserts advance
            // reproducibly; the reloaded graph's own search does not depend on it.
            rng: 0x9E37_79B9_7F4A_7C15,
        };
        // Recompute the entry as the *lowest-id* node on the highest layer — exactly the entry a
        // fresh build ends with (a new node only becomes entry when strictly taller, so the first
        // node to reach the top layer stays the entry). Matching it makes a reloaded graph search
        // byte-identically, not merely correctly.
        index.entry = index
            .nodes
            .iter()
            .enumerate()
            .max_by_key(|(id, node)| (node.neighbours.len(), std::cmp::Reverse(*id)))
            .map(|(id, _)| id as u32);
        index
    }
}

/// A bounds-checked little-endian cursor: every read returns `None` past the end rather than
/// panicking, so a truncated blob is rejected cleanly. Shared with the vector-index graph
/// persistence wrapper in the executor, which frames a graph blob with its own header.
pub(crate) struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    /// Start reading at the front of `bytes`.
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    /// Whether every byte has been consumed (used to reject trailing garbage).
    pub(crate) const fn at_end(&self) -> bool {
        self.pos == self.bytes.len()
    }
    pub(crate) fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    pub(crate) fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    pub(crate) fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    pub(crate) fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small seeded `xorshift64*` so the test dataset is reproducible without an external crate.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((v >> 40) as f32) / ((1u32 << 24) as f32) // [0, 1)
        }
    }

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Rng(seed ^ 0xDEAD_BEEF);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
            .collect()
    }

    /// Exact top-`k` ids by L2 distance — the recall oracle.
    fn brute_force(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(f64, u32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    crate::vector::l2_distance(v, query).unwrap_or(f64::INFINITY),
                    i as u32,
                )
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    #[test]
    fn recall_beats_threshold_against_brute_force() {
        let (n, dim, k) = (600, 16, 10);
        let data = random_vectors(n, dim, 1);
        let mut index = HnswIndex::new(dim, Metric::L2, HnswParams::default(), 42);
        for v in &data {
            index.insert(v.clone()).expect("insert");
        }
        assert_eq!(index.len(), n);

        let queries = random_vectors(40, dim, 2);
        let mut total_recall = 0.0;
        for q in &queries {
            let exact: HashSet<u32> = brute_force(&data, q, k).into_iter().collect();
            let approx = index.search(q, k, 64).expect("search");
            assert_eq!(approx.len(), k, "must return k results when n >= k");
            // Results must be sorted nearest-first.
            for w in approx.windows(2) {
                assert!(w[0].1 <= w[1].1, "results not sorted by distance");
            }
            let hits = approx.iter().filter(|(id, _)| exact.contains(id)).count();
            total_recall += hits as f64 / k as f64;
        }
        let recall = total_recall / queries.len() as f64;
        assert!(
            recall >= 0.90,
            "HNSW recall@{k} = {recall:.3} fell below 0.90 — graph build/search regressed"
        );
    }

    /// Exact top-`k` ids under an arbitrary metric — the per-metric recall oracle. Ties broken by
    /// id so the ground truth is deterministic.
    fn brute_force_metric(data: &[Vec<f32>], query: &[f32], k: usize, metric: Metric) -> Vec<u32> {
        let mut scored: Vec<(f64, u32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    metric.exact_distance(v, query).unwrap_or(f64::INFINITY),
                    i as u32,
                )
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    /// Recall must clear a high bar for EVERY metric, not just L2 (the metric is baked into the
    /// index since the operator-class binding, so each is a distinct build+search path). The
    /// recall oracle uses the same metric the index was built with — the standard definition of
    /// recall against exact search.
    #[test]
    fn recall_beats_threshold_for_every_metric() {
        let (n, dim, k, ef) = (1500, 64, 10, 96);
        let data = random_vectors(n, dim, 7);
        let queries = random_vectors(50, dim, 8);
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct, Metric::L1] {
            let mut index = HnswIndex::new(dim, metric, HnswParams::default(), 42);
            for v in &data {
                index.insert(v.clone()).expect("insert");
            }
            let mut total_recall = 0.0;
            for q in &queries {
                let exact: HashSet<u32> = brute_force_metric(&data, q, k, metric)
                    .into_iter()
                    .collect();
                let approx = index.search(q, k, ef).expect("search");
                assert_eq!(approx.len(), k, "must return k results when n >= k");
                for w in approx.windows(2) {
                    assert!(
                        w[0].1 <= w[1].1,
                        "{metric:?}: results not sorted by distance"
                    );
                }
                let hits = approx.iter().filter(|(id, _)| exact.contains(id)).count();
                total_recall += hits as f64 / k as f64;
            }
            let recall = total_recall / queries.len() as f64;
            eprintln!("VECRECALL {metric:?} recall@{k} = {recall:.3}");
            assert!(
                recall >= 0.85,
                "HNSW recall@{k} for {metric:?} = {recall:.3} fell below 0.85"
            );
        }
    }

    /// Measurement (not a pass/fail gate): the build-time-vs-recall trade-off of `ef_construction`,
    /// the one build-cost lever that does not touch the distance kernel. Bit-identical layout changes
    /// to the build path were measured and did not pay off, so the remaining lever is
    /// result-affecting: a lower `ef_construction` builds faster but can lower recall. This prints
    /// build time and recall@k per `ef_construction` so the trade-off is a measured table, not a
    /// guess. `#[ignore]` — run manually
    /// (`cargo test -p nusadb-sql ef_construction_build_time_vs_recall -- --ignored --nocapture`);
    /// criterion measures the time half but cannot measure recall, so both live here.
    #[test]
    #[ignore = "measurement: ef_construction build-time vs recall trade-off (run manually)"]
    fn ef_construction_build_time_vs_recall_tradeoff() {
        use std::time::Instant;
        let (k, ef_search) = (10, 96);
        // Two configs: an easy one, and a harder one (more points, higher dimension — where a coarser
        // graph is likelier to cost recall), so the trade-off is judged on more than one regime.
        for (n, dim) in [(1500usize, 64usize), (2000, 128)] {
            let data = random_vectors(n, dim, 7);
            let queries = random_vectors(50, dim, 8);
            eprintln!("EFTRADEOFF --- n={n} dim={dim} k={k} ef_search={ef_search} metric=L2 ---");
            eprintln!("EFTRADEOFF ef_construction | build_ms | recall@{k}");
            for ef_c in [40usize, 64, 100, 150] {
                let params = HnswParams {
                    m: 16,
                    ef_construction: ef_c,
                };
                let start = Instant::now();
                let mut index = HnswIndex::new(dim, Metric::L2, params, 42);
                for v in &data {
                    index.insert(v.clone()).expect("insert");
                }
                let build_ms = start.elapsed().as_secs_f64() * 1000.0;
                let mut total_recall = 0.0;
                for q in &queries {
                    let exact: HashSet<u32> = brute_force(&data, q, k).into_iter().collect();
                    let approx = index.search(q, k, ef_search).expect("search");
                    let hits = approx.iter().filter(|(id, _)| exact.contains(id)).count();
                    total_recall += hits as f64 / k as f64;
                }
                let recall = total_recall / queries.len() as f64;
                eprintln!("EFTRADEOFF {ef_c:>15} | {build_ms:8.1} | {recall:.3}");
            }
        }
    }

    /// Each metric must rank the *right* vector nearest — a guard against a sign/direction error
    /// that self-consistent recall cannot catch (a flipped inner product would still recall 1.0
    /// against its own oracle while returning the *farthest* vector). The expected nearest here is
    /// worked out from each metric's definition by hand, independent of the implementation:
    /// - **L2 / L1**: the closer point in space is nearer (`[1,0]` beats `[3,0]` from the origin).
    /// - **Cosine**: the same-direction vector is nearer (`[5,0]` beats `[0,9]` for query `[1,0]`,
    ///   regardless of magnitude — cosine ignores it).
    /// - **Inner product**: a *larger* dot product with the query is *more* similar (nearer), so
    ///   `[2,0]` (dot 2) must beat `[0.5,0]` (dot 0.5) for query `[1,0]`. A flipped sign would pick
    ///   `[0.5,0]` — exactly the bug this pins.
    #[test]
    fn each_metric_ranks_the_correct_vector_nearest() {
        let cases = [
            (
                Metric::L2,
                vec![vec![1.0, 0.0], vec![3.0, 0.0]],
                vec![0.0, 0.0],
                0u32,
            ),
            (
                Metric::L1,
                vec![vec![1.0, 0.0], vec![0.0, 3.0]],
                vec![0.0, 0.0],
                0,
            ),
            (
                Metric::Cosine,
                vec![vec![5.0, 0.0], vec![0.0, 9.0]],
                vec![1.0, 0.0],
                0,
            ),
            (
                Metric::InnerProduct,
                vec![vec![2.0, 0.0], vec![0.5, 0.0]],
                vec![1.0, 0.0],
                0,
            ),
        ];
        for (metric, points, query, expected_nearest) in cases {
            let mut index = HnswIndex::new(2, metric, HnswParams::default(), 1);
            for p in &points {
                index.insert(p.clone()).expect("insert");
            }
            let got = index.search(&query, 1, 16).expect("search");
            assert_eq!(
                got[0].0, expected_nearest,
                "{metric:?}: expected id {expected_nearest} nearest to {query:?}, got {got:?}"
            );
        }
    }

    #[test]
    fn search_is_exact_on_a_tiny_index() {
        // With few points and a wide beam, HNSW degenerates to exact search.
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
        ];
        let mut index = HnswIndex::new(2, Metric::L2, HnswParams::default(), 7);
        for v in &data {
            index.insert(v.clone()).expect("insert");
        }
        let got = index.search(&[0.1, 0.1], 3, 16).expect("search");
        let ids: Vec<u32> = got.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![0, 1, 2], "nearest three of (0.1,0.1)");
    }

    #[test]
    fn a_duplicate_vector_does_not_collapse_the_graph() {
        // One row whose vector is an exact copy of another used to be enough to break *every*
        // query. Pruning a node's links keeps a candidate only when it is closer to the node than
        // to each already-kept neighbour; against a twin at distance 0 that test is `d < d`, false
        // for every candidate, so the node kept exactly one link — to the twin. The pair became a
        // sink, and when one of them was the entry point every search ended there. The reported
        // symptom was all twenty probes returning the duplicated row's id.
        //
        // The probes below deliberately target rows that are *not* part of the duplicate pair: the
        // damage was global, not confined to the twins.
        let dim = 32;
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1_u64 << 53) as f32
        };
        let mut vectors: Vec<Vec<f32>> = (0..400)
            .map(|_| (0..dim).map(|_| next()).collect())
            .collect();
        vectors.push(vectors[0].clone()); // the one duplicate

        let mut index = HnswIndex::new(dim, Metric::L2, HnswParams::default(), 42);
        for v in &vectors {
            index.insert(v.clone()).expect("insert");
        }

        // No node may be left with a single link on layer 0 — that is the collapse itself, and it
        // is what disconnects the graph regardless of which query runs.
        let min_degree = (0..vectors.len())
            .filter_map(|i| index.node_neighbours(i).and_then(<[Vec<u32>]>::first))
            .map(Vec::len)
            .min()
            .expect("a built graph has nodes");
        assert!(
            min_degree > 1,
            "a node was pruned down to {min_degree} link(s) on layer 0 — the duplicate collapsed it"
        );

        // Every unique row still finds itself: an exact copy of the query is at distance 0, so an
        // index that cannot return it is wrong, not approximate.
        for want in [17_usize, 93, 155, 231, 307, 399] {
            let hit = index.search(&vectors[want], 1, 64).expect("search");
            assert_eq!(
                hit.first().map(|(id, _)| *id as usize),
                Some(want),
                "self-retrieval of unique row {want} with a duplicate pair present"
            );
        }
    }

    #[test]
    fn serialize_round_trip_reproduces_search() {
        let (n, dim, k) = (300, 12, 10);
        let data = random_vectors(n, dim, 5);
        let mut index = HnswIndex::new(dim, Metric::Cosine, HnswParams::default(), 99);
        for v in &data {
            index.insert(v.clone()).expect("insert");
        }
        let blob = index.serialize();
        let restored = HnswIndex::deserialize(&blob).expect("deserialize a valid blob");
        assert_eq!(restored.len(), index.len());

        // A search over the restored graph is identical to the original — same ids, same distances.
        for q in random_vectors(20, dim, 6) {
            let before = index.search(&q, k, 64).expect("search original");
            let after = restored.search(&q, k, 64).expect("search restored");
            assert_eq!(before, after, "restored graph must search identically");
        }
    }

    #[test]
    fn from_parts_round_trip_reproduces_search() {
        let (n, dim, k) = (300, 12, 10);
        let data = random_vectors(n, dim, 7);
        let mut index = HnswIndex::new(dim, Metric::Cosine, HnswParams::default(), 55);
        for v in &data {
            index.insert(v.clone()).expect("insert");
        }
        // Persist each node as (vector, adjacency-bytes), then reconstruct via from_parts — the
        // per-node persistence path. Vectors come from the (base-table stand-in) `data` by node id.
        let mut parts = Vec::new();
        for (id, vector) in data.iter().enumerate() {
            let adjacency = index.node_neighbours(id).expect("node in range");
            let bytes = HnswIndex::serialize_adjacency(adjacency);
            let neighbours =
                HnswIndex::deserialize_adjacency(&bytes).expect("adjacency round-trip");
            parts.push((vector.clone(), neighbours));
        }
        let restored = HnswIndex::from_parts(dim, Metric::Cosine, HnswParams::default(), parts);
        assert_eq!(restored.len(), index.len());
        for q in random_vectors(20, dim, 8) {
            let before = index.search(&q, k, 64).expect("search original");
            let after = restored.search(&q, k, 64).expect("search restored");
            assert_eq!(before, after, "from_parts must search identically");
        }
    }

    #[test]
    fn insert_reporting_touches_only_existing_neighbours() {
        let mut index = HnswIndex::new(4, Metric::L2, HnswParams::default(), 3);
        // The first insert has no neighbours to touch.
        let (id0, touched0) = index
            .insert_reporting(vec![0.0, 0.0, 0.0, 0.0])
            .expect("insert");
        assert_eq!(id0, 0);
        assert!(touched0.is_empty());
        // Later inserts only ever report ids below the new one (existing nodes), sorted and unique.
        for step in 1..30u32 {
            let v = vec![step as f32, 0.0, 0.0, 0.0];
            let (id, touched) = index.insert_reporting(v).expect("insert");
            assert_eq!(id, step);
            assert!(
                touched.iter().all(|&t| t < id),
                "touched are existing nodes"
            );
            assert!(touched.windows(2).all(|w| w[0] < w[1]), "sorted + deduped");
        }
    }

    #[test]
    fn deserialize_rejects_truncated_or_garbage_blobs() {
        let mut index = HnswIndex::new(4, Metric::L2, HnswParams::default(), 3);
        for v in [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]] {
            index.insert(v.to_vec()).expect("insert");
        }
        let blob = index.serialize();
        // A prefix (truncated) blob is rejected, not accepted as a smaller index.
        assert!(HnswIndex::deserialize(&blob[..blob.len() - 1]).is_none());
        // Trailing bytes are rejected.
        let mut extra = blob;
        extra.push(0xAB);
        assert!(HnswIndex::deserialize(&extra).is_none());
        // Empty input is rejected.
        assert!(HnswIndex::deserialize(&[]).is_none());
    }

    #[test]
    fn empty_and_degenerate_cases() {
        let mut index = HnswIndex::new(3, Metric::Cosine, HnswParams::default(), 1);
        assert!(index.is_empty());
        assert!(
            index
                .search(&[1.0, 0.0, 0.0], 5, 16)
                .expect("search")
                .is_empty()
        );
        index.insert(vec![1.0, 0.0, 0.0]).expect("insert");
        assert_eq!(
            index.search(&[1.0, 0.0, 0.0], 5, 16).expect("search").len(),
            1
        );
        // k = 0 yields nothing; a dimension mismatch is a clean error, not a panic.
        assert!(
            index
                .search(&[1.0, 0.0, 0.0], 0, 16)
                .expect("search")
                .is_empty()
        );
        assert!(index.insert(vec![1.0, 0.0]).is_err());
        assert!(index.search(&[1.0, 0.0], 1, 16).is_err());
    }

    #[test]
    fn build_is_deterministic_for_a_seed() {
        let data = random_vectors(120, 8, 3);
        let build = || {
            let mut idx = HnswIndex::new(8, Metric::L2, HnswParams::default(), 99);
            for v in &data {
                idx.insert(v.clone()).expect("insert");
            }
            let q = vec![0.5_f32; 8];
            idx.search(&q, 10, 50).expect("search")
        };
        assert_eq!(
            build(),
            build(),
            "same seed + insert order must reproduce the graph"
        );
    }

    /// A blob round-trips under every metric, and the metric survives it — a graph reloaded as the
    /// wrong distance would answer plausible-looking nonsense. Also pins that an unrecognized metric
    /// byte is refused rather than mapped onto some default, so a blob from a future build that added
    /// a metric is rebuilt instead of misread.
    #[test]
    fn every_metric_survives_a_blob_round_trip() {
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct, Metric::L1] {
            let mut index = HnswIndex::new(8, metric, HnswParams::default(), 7);
            for v in random_vectors(24, 8, 0xB10B) {
                index.insert(v).expect("insert");
            }
            let blob = index.serialize();
            let back = HnswIndex::deserialize(&blob).expect("round trip");
            assert_eq!(back.metric(), metric, "metric lost in the blob");
            let q = vec![0.25_f32; 8];
            assert_eq!(
                index.search(&q, 5, 32).expect("search"),
                back.search(&q, 5, 32).expect("search"),
                "{metric:?} reloaded graph answers differently"
            );
        }
        // The metric is the 5th byte (u32 dim, then the metric tag). An unknown tag is refused.
        let mut index = HnswIndex::new(4, Metric::L2, HnswParams::default(), 1);
        index.insert(vec![1.0, 0.0, 0.0, 0.0]).expect("insert");
        let mut blob = index.serialize();
        blob[4] = 200;
        assert!(
            HnswIndex::deserialize(&blob).is_none(),
            "an unknown metric tag must be refused, not defaulted"
        );
    }

    /// A 64-bit FNV-1a digest — enough to pin an exact byte sequence without pulling in a hash crate.
    fn digest(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// The cached per-vector term must give a **bit**-identical distance to recomputing it, for every
    /// metric — that identity is the entire justification for caching, so it is asserted rather than
    /// assumed. Checked on values that are not round numbers, where any reassociation of the
    /// arithmetic would surface in the low mantissa bits.
    #[test]
    fn cached_term_distance_is_bit_identical_to_recomputing() {
        let data = random_vectors(64, 96, 0x1234);
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            for pair in data.windows(2) {
                let (a, b) = (&pair[0], &pair[1]);
                let cached =
                    metric.distance_cached(a, metric.cached_term(a), b, metric.cached_term(b));
                assert_eq!(
                    cached.to_bits(),
                    metric.distance(a, b).to_bits(),
                    "{metric:?} distance diverged once the per-vector term was cached"
                );
            }
        }
        // The zero vector has no direction: cosine answers 1.0 by convention, and the cached path
        // must reach that same guard rather than dividing by a cached zero norm. The guard reads
        // `a_norm == 0.0 || b_norm == 0.0`, so check the zero on *either* side.
        let zero = vec![0.0_f32; 96];
        let other = &data[0];
        let m = Metric::Cosine;
        for (a, b) in [(&zero, other), (other, &zero)] {
            assert_eq!(
                m.distance_cached(a, m.cached_term(a), b, m.cached_term(b))
                    .to_bits(),
                m.distance(a, b).to_bits(),
                "zero-vector guard diverged"
            );
        }

        // A dimension mismatch has no distance; both paths must reach `+∞` so such a pair is never
        // chosen as a neighbour, rather than one of them producing a finite number.
        let short = vec![1.0_f32; 8];
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            assert_eq!(
                metric
                    .distance_cached(
                        other,
                        metric.cached_term(other),
                        &short,
                        metric.cached_term(&short)
                    )
                    .to_bits(),
                f64::INFINITY.to_bits(),
                "{metric:?} dimension mismatch must be +∞"
            );
        }
    }

    /// Pins the exact graph a cosine build produces at a realistic dimension.
    ///
    /// Caching each vector's norm is an arithmetic identity, not an approximation, so it must leave
    /// the built graph **byte-identical** — same links, same order, same distances, hence the same
    /// recall. A digest states that in the bluntest available way: if any of it shifts, this fails.
    /// Written against the public API only, so the same test can be checked out onto an earlier
    /// revision to confirm the pinned value did not move.
    ///
    /// A dimension of 128 is deliberate: an earlier round of vector work shipped a regression that a
    /// `VECTOR(3)`-sized test could not see.
    ///
    /// **If this fails, do not re-pin it.** A new number means the graph changed, and recall has to
    /// be re-validated before the value moves. The one benign way it could shift is the platform's
    /// `f64::ln`, which `random_level` uses and which libm does not guarantee bit-for-bit across
    /// targets: a 1-ulp difference only changes a level when `-ln(u)·mL` lands within an ulp of an
    /// integer, so on a different libc this is worth ruling out first — but the distance kernels
    /// themselves are exact and host-independent, so nothing else here should drift.
    #[test]
    fn cosine_build_graph_matches_its_pinned_digest() {
        let mut index = HnswIndex::new(128, Metric::Cosine, HnswParams::default(), 42);
        for v in random_vectors(400, 128, 0xC057) {
            index.insert(v).expect("insert");
        }
        assert_eq!(digest(&index.serialize()), 16_565_488_365_752_518_164);
    }
}
