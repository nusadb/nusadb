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

use crate::error::Error;

/// The distance metric an index is built and searched under. Every variant returns a *distance*
/// (smaller is closer), matching the SQL operators: `<=>` (cosine), `<#>` (negative inner product),
/// and L2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean distance.
    L2,
    /// Cosine distance `1 − cosθ`.
    Cosine,
    /// Negative inner product (so a larger dot product is a smaller distance).
    InnerProduct,
}

impl Metric {
    /// Distance between two equal-length vectors. A dimension mismatch (which the index guards
    /// against at insert time) maps to `+∞` so such a pair is never selected as a neighbour.
    fn distance(self, a: &[f32], b: &[f32]) -> f64 {
        let d = match self {
            Self::L2 => crate::vector::l2_distance(a, b),
            Self::Cosine => crate::vector::cosine_distance(a, b),
            Self::InnerProduct => crate::vector::inner_product(a, b),
        };
        d.unwrap_or(f64::INFINITY)
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
    neighbours: Vec<Vec<u32>>,
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

    fn distance(&self, a: &[f32], b: &[f32]) -> f64 {
        self.metric.distance(a, b)
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
        self.nodes.push(Node {
            vector,
            neighbours: vec![Vec::new(); level + 1],
        });

        let Some(entry) = self.entry else {
            // First point becomes the entry point.
            self.entry = Some(id);
            return Ok(id);
        };

        let query = self.nodes[id as usize].vector.clone();
        let top = self.top_level();

        // Phase 1: greedily descend the layers above this node's top, narrowing to one entry point.
        let mut ep = entry;
        for layer in (level + 1..=top).rev() {
            ep = self.greedy_nearest(&query, ep, layer);
        }

        // Phase 2: from this node's top down to layer 0, beam-search, pick neighbours, link both ways.
        let mut entry_points = vec![ep];
        for layer in (0..=level.min(top)).rev() {
            let found =
                self.search_layer(&query, &entry_points, self.params.ef_construction, layer);
            let degree = self.max_degree(layer);
            let chosen = self.select_neighbours(&query, &found, degree);
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
        Ok(id)
    }

    /// The top layer index of the current entry point (0 if the index has a single layer).
    fn top_level(&self) -> usize {
        self.entry
            .map_or(0, |e| self.nodes[e as usize].neighbours.len() - 1)
    }

    /// Walk greedily to the node nearest `query` on `layer`, starting from `from`.
    fn greedy_nearest(&self, query: &[f32], from: u32, layer: usize) -> u32 {
        let mut best = from;
        let mut best_dist = self.distance(query, &self.nodes[from as usize].vector);
        loop {
            let mut improved = false;
            if let Some(node) = self.nodes.get(best as usize)
                && let Some(neighbours) = node.neighbours.get(layer)
            {
                for &n in neighbours {
                    let d = self.distance(query, &self.nodes[n as usize].vector);
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
        entry_points: &[u32],
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<u32> = HashSet::new();
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        let mut result: BinaryHeap<Candidate> = BinaryHeap::new();

        for &ep in entry_points {
            if visited.insert(ep) {
                let d = self.distance(query, &self.nodes[ep as usize].vector);
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
            let neighbours = self
                .nodes
                .get(current.id as usize)
                .map_or_else(Vec::new, |node| {
                    node.neighbours.get(layer).cloned().unwrap_or_default()
                });
            for n in neighbours {
                if !visited.insert(n) {
                    continue;
                }
                let d = self.distance(query, &self.nodes[n as usize].vector);
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
    fn select_neighbours(&self, query: &[f32], candidates: &[Candidate], m: usize) -> Vec<u32> {
        let _ = query; // candidates already carry their distance to the query
        let mut kept: Vec<Candidate> = Vec::with_capacity(m);
        for &cand in candidates {
            if kept.len() >= m {
                break;
            }
            let cand_vec = &self.nodes[cand.id as usize].vector;
            let closer_to_query_than_to_kept = kept
                .iter()
                .all(|k| cand.dist < self.distance(cand_vec, &self.nodes[k.id as usize].vector));
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
            let mut cands: Vec<Candidate> = self.nodes[n as usize].neighbours[layer]
                .iter()
                .map(|&x| Candidate {
                    dist: self.distance(&n_vec, &self.nodes[x as usize].vector),
                    id: x,
                })
                .collect();
            cands.sort_unstable(); // nearest first
            let pruned = self.select_neighbours(&n_vec, &cands, degree);
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

        let top = self.top_level();
        let mut ep = entry;
        for layer in (1..=top).rev() {
            ep = self.greedy_nearest(query, ep, layer);
        }
        let beam = ef.max(k);
        let found = self.search_layer(query, &[ep], beam, 0);
        Ok(found.into_iter().take(k).map(|c| (c.id, c.dist)).collect())
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
}
