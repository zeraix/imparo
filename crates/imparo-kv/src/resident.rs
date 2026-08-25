//! Resident multi-conversation bookkeeping: which sealed units live in device
//! blocks, whose tables point at them, and what gets evicted when blocks run out.
//!
//! The one-copy rule made mechanical: a conversation's table is a VIEW over pooled
//! blocks. Sharing a preamble = two tables naming the same blocks (refcounted via
//! the unit index); switching conversations = swapping which table the kernels
//! read; nothing full-attention ever moves. Windowed state is per-conversation and
//! bounded, so it is NOT pooled (design: "What the pool manages") — the engine
//! keeps it as per-conversation checkpoints.
//!
//! Model-agnostic: layers are opaque ids; a unit spans UNIT_TOKENS positions =
//! `UNIT_BLOCKS` 64-cell blocks on every layer.

use std::collections::BTreeMap;

use crate::alloc::{BlockAllocator, BlockIdx};
use crate::identity::{UNIT_TOKENS, UnitHash};
use crate::index::{ConversationId, Pool, UnitId};

/// 64-cell blocks per content unit (256 / 64).
pub const UNIT_BLOCKS: usize = UNIT_TOKENS / 64;

/// Per-layer physical placement of one unit.
pub type UnitPlacement = BTreeMap<u32, [BlockIdx; UNIT_BLOCKS]>;

/// A windowed layer's ring, written as a BLOCK TABLE.
///
/// The two addressing rules the engine has -- ring and paged -- are not two rules:
///
/// ```text
///   ring    slot = base + (pos & (ring - 1))
///   paged   slot = table[pos >> 6] * 64 + (pos & 63)
/// ```
///
/// A ring is a power of two and at least 64, so every 64-position block lands on one
/// contiguous 64-slot run inside it, and `table[j] = (base + (64j & (ring - 1))) / 64`
/// reproduces the ring exactly. Proven by `ring_and_paged_addressing_agree`, not by
/// this comment.
///
/// Why it matters: the ring is the one piece of KV the pool does NOT place, and that
/// is why a windowed model's answer depends on what else is resident -- a neighbour's
/// cells sit inside the tiles a request reduces (docs/kv-pool-review.md). Expressing
/// the ring as a table is what lets each conversation own a REGION of it, with `base`
/// as the only difference between them, and with no kernel that has to learn a third
/// way to find a slot.
///
/// `base_slot` must be a multiple of 64; `ring` a power of two at least 64.
#[must_use]
pub fn ring_page_table(base_slot: usize, ring: usize, logical_blocks: usize) -> Vec<u32> {
    assert!(ring.is_power_of_two() && ring >= 64, "ring {ring} must be a power of two >= 64");
    assert!(base_slot % 64 == 0, "region base {base_slot} must be block aligned");
    (0..logical_blocks)
        .map(|j| ((base_slot + ((j * 64) & (ring - 1))) / 64) as u32)
        .collect()
}

pub struct ResidentKv {
    pub pool: Pool,
    layers: Vec<u32>,
    alloc: BTreeMap<u32, BlockAllocator>,
    placed: BTreeMap<UnitId, UnitPlacement>,
    /// Least-recently-probed order for whole-conversation eviction.
    lru: Vec<ConversationId>,
}

impl ResidentKv {
    /// `layers`: the full-attention layer ids; `capacity_blocks`: 64-cell blocks
    /// each layer's device buffer holds.
    #[must_use]
    pub fn new(layers: &[u32], capacity_blocks: u32) -> Self {
        Self {
            pool: Pool::new(),
            layers: layers.to_vec(),
            alloc: layers
                .iter()
                .map(|&l| (l, BlockAllocator::new(capacity_blocks, 8)))
                .collect(),
            placed: BTreeMap::new(),
            lru: Vec::new(),
        }
    }

    fn touch(&mut self, conv: &ConversationId) {
        self.lru.retain(|c| c != conv);
        self.lru.push(conv.clone());
    }

    /// Adopt-on-lookup against RESIDENT units: returns the number of leading units
    /// served in place (their blocks are already on device) and records the
    /// conversation's unit list.
    pub fn probe(&mut self, conv: &ConversationId, hashes: &[UnitHash]) -> usize {
        self.touch(conv);
        let p = self.pool.probe(conv, hashes);
        p.hit_units
    }

    /// Seal the conversation's next unit IN PLACE: the blocks holding positions
    /// [boundary, boundary + UNIT_TOKENS) were already written by prefill through
    /// the conversation's table; sealing just names them. Merge-on-seal frees the
    /// duplicate's blocks when the hash is already resident (equal hash = equal
    /// bytes, so the survivor serves both).
    pub fn seal_unit(
        &mut self,
        conv: &ConversationId,
        hash: UnitHash,
        placement: &UnitPlacement,
    ) -> UnitId {
        let id = self.pool.begin_fill(conv);
        let survivor = self.pool.seal(id, hash);
        if survivor == id {
            self.placed.insert(id, placement.clone());
        } else if let std::collections::btree_map::Entry::Vacant(e) =
            self.placed.entry(survivor)
        {
            // the survivor's backing was evicted earlier; this fresh placement
            // becomes its new home
            e.insert(placement.clone());
        } else {
            // duplicate content: this conversation's freshly written blocks are
            // byte-identical to the survivor's; free them and share
            for (&layer, blocks) in placement {
                if let Some(a) = self.alloc.get_mut(&layer) {
                    for &b in blocks {
                        a.free(b);
                    }
                }
            }
        }
        survivor
    }

    /// Allocate one unit's blocks on every layer (for a filling tail region), or
    /// None when a layer is out of blocks (caller evicts and retries, or defers).
    pub fn alloc_unit(&mut self) -> Option<UnitPlacement> {
        let mut out = UnitPlacement::new();
        for &layer in &self.layers.clone() {
            let a = self.alloc.get_mut(&layer)?;
            let mut got: Vec<BlockIdx> = Vec::with_capacity(UNIT_BLOCKS);
            while got.len() < UNIT_BLOCKS {
                if let Some(b) = a.alloc() {
                    got.push(b);
                } else {
                    // roll back this layer's partial run and every prior layer
                    for &b in &got {
                        a.free(b);
                    }
                    for (&l2, b2) in &out {
                        if let Some(a2) = self.alloc.get_mut(&l2) {
                            for &b in b2 {
                                a2.free(b);
                            }
                        }
                    }
                    return None;
                }
            }
            let mut blocks = [0 as BlockIdx; UNIT_BLOCKS];
            blocks.copy_from_slice(&got);
            out.insert(layer, blocks);
        }
        Some(out)
    }

    /// The per-layer table (block entries, one per 64 positions) for a
    /// conversation's sealed units followed by `tail` extra unit placements
    /// (the unsealed filling region).
    #[must_use]
    pub fn table_for(
        &self,
        conv: &ConversationId,
        layer: u32,
        tail: &[UnitPlacement],
    ) -> Vec<u32> {
        let mut t = Vec::new();
        if let Some(units) = self.pool.conversation(conv) {
            for u in units {
                if let Some(pl) = self.placed.get(u) {
                    if let Some(blocks) = pl.get(&layer) {
                        t.extend(blocks.iter().copied());
                    }
                }
            }
        }
        for pl in tail {
            if let Some(blocks) = pl.get(&layer) {
                t.extend(blocks.iter().copied());
            }
        }
        t
    }

    /// Free every refcount-zero sealed unit's blocks (eviction candidates in
    /// deterministic order). Returns the freed placements so the caller can hand
    /// their pages back to the OS. Durability is the disk tier's job (spilled
    /// before residency is dropped), so dropping residency loses nothing.
    pub fn evict_unreferenced(&mut self) -> Vec<UnitPlacement> {
        let victims = self.pool.evictable();
        let mut freed = Vec::new();
        for id in victims {
            if let Some(pl) = self.placed.remove(&id) {
                for (&layer, blocks) in &pl {
                    if let Some(a) = self.alloc.get_mut(&layer) {
                        for &b in blocks {
                            a.free(b);
                        }
                    }
                }
                freed.push(pl);
            }
            self.pool.remove_unit(id);
        }
        freed
    }

    /// Drop the least-recently-probed conversation (not `keep`) entirely,
    /// returning whether anything was dropped. The safety valve when allocation
    /// still fails after evict_unreferenced.
    pub fn drop_lru_conversation(&mut self, keep: &ConversationId) -> bool {
        let victim = self.lru.iter().find(|c| *c != keep).cloned();
        let Some(v) = victim else { return false };
        self.pool.forget(&v);
        self.lru.retain(|c| c != &v);
        let _ = self.evict_unreferenced();
        true
    }

    #[must_use]
    pub fn free_blocks(&self, layer: u32) -> u32 {
        self.alloc
            .get(&layer)
            .map_or(0, BlockAllocator::free_blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ConfigRoot, unit_hashes};

    fn cid(s: &str) -> ConversationId {
        ConversationId(s.to_string())
    }

    fn hashes(tokens: &[u32]) -> Vec<UnitHash> {
        unit_hashes(&ConfigRoot::new(b"m", (1, 1), b"g"), tokens)
    }

    #[test]
    fn shared_preamble_is_one_copy_and_switching_moves_nothing() {
        let layers = [5u32, 11];
        let mut r = ResidentKv::new(&layers, 32);
        let pre: Vec<u32> = (0..2 * UNIT_TOKENS as u32).collect();
        let h = hashes(&pre);

        // conversation A prefills the preamble: 2 units allocated + sealed in place
        assert_eq!(r.probe(&cid("a"), &h), 0);
        let mut a_units = Vec::new();
        for hu in &h {
            let pl = r.alloc_unit().unwrap();
            a_units.push(r.seal_unit(&cid("a"), *hu, &pl));
        }
        let used_after_a = 32 - r.free_blocks(5);
        assert_eq!(used_after_a, 2 * UNIT_BLOCKS as u32);

        // B probes the same preamble: full resident hit, ZERO new blocks
        assert_eq!(r.probe(&cid("b"), &h), 2);
        assert_eq!(32 - r.free_blocks(5), used_after_a);

        // both tables name the same physical blocks: one copy, two views
        let ta = r.table_for(&cid("a"), 5, &[]);
        let tb = r.table_for(&cid("b"), 5, &[]);
        assert_eq!(ta, tb);
        assert_eq!(ta.len(), 2 * UNIT_BLOCKS);
    }

    #[test]
    fn concurrent_fill_merges_and_frees_the_duplicate() {
        let mut r = ResidentKv::new(&[3], 32);
        let toks: Vec<u32> = (0..UNIT_TOKENS as u32).collect();
        let h = hashes(&toks)[0];
        r.probe(&cid("a"), &[]);
        r.probe(&cid("b"), &[]);
        let pa = r.alloc_unit().unwrap();
        let pb = r.alloc_unit().unwrap();
        assert_eq!(32 - r.free_blocks(3), 2 * UNIT_BLOCKS as u32);
        let ua = r.seal_unit(&cid("a"), h, &pa);
        let ub = r.seal_unit(&cid("b"), h, &pb);
        assert_eq!(ua, ub);
        // the duplicate's blocks went back to the pool
        assert_eq!(32 - r.free_blocks(3), UNIT_BLOCKS as u32);
    }

    #[test]
    fn eviction_frees_only_the_unreferenced() {
        let mut r = ResidentKv::new(&[3], 32);
        let toks: Vec<u32> = (0..2 * UNIT_TOKENS as u32).collect();
        let h = hashes(&toks);
        r.probe(&cid("a"), &h);
        for hu in &h {
            let pl = r.alloc_unit().unwrap();
            r.seal_unit(&cid("a"), *hu, &pl);
        }
        r.probe(&cid("b"), &h); // B shares
        assert!(r.evict_unreferenced().is_empty()); // everything referenced
        r.pool.forget(&cid("a"));
        assert!(r.evict_unreferenced().is_empty()); // B still holds both
        r.pool.forget(&cid("b"));
        assert_eq!(r.evict_unreferenced().len(), 2); // now they go
        assert_eq!(r.free_blocks(3), 32);
    }
}


#[cfg(test)]
mod ring_tests {
    use super::ring_page_table;

    /// The claim the region change rests on: a ring and a block table addressing the
    /// same positions agree, slot for slot. If this ever fails, regions cannot be
    /// built out of the paged path and the kernels would need a third rule.
    #[test]
    fn ring_and_paged_addressing_agree() {
        for ring in [64_usize, 128, 256, 1024, 4096] {
            for base in [0_usize, ring, 3 * ring] {
                let blocks = 40;
                let pt = ring_page_table(base, ring, blocks);
                for pos in 0..blocks * 64 {
                    let by_ring = base + (pos & (ring - 1));
                    let by_table = pt[pos >> 6] as usize * 64 + (pos & 63);
                    assert_eq!(
                        by_ring, by_table,
                        "ring={ring} base={base} pos={pos}"
                    );
                }
            }
        }
    }

    /// Two conversations differ only by their base, and their slots never collide.
    #[test]
    fn regions_do_not_overlap() {
        let (ring, blocks) = (1024_usize, 32);
        let a = ring_page_table(0, ring, blocks);
        let b = ring_page_table(ring, ring, blocks);
        for (x, y) in a.iter().zip(&b) {
            assert_ne!(x, y);
            assert_eq!(*y as usize, *x as usize + ring / 64);
        }
    }
}
