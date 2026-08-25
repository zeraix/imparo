//! Deterministic block allocation: LOWEST FREE BLOCK, always.
//!
//! Allocation must be a pure function of which blocks are currently free — never
//! of history. The fork found the alternative the hard way: a free list used as a
//! stack re-issues most-recently-freed first, physical order changes the attention
//! reduction order, and at low-precision KV a last-bit difference crosses a
//! quantisation boundary — a flipped token whose value depended on what ran
//! earlier in the process. A request's output must not depend on what ran before
//! it; lowest-free-block keeps that invariant, and as a side effect fills low
//! arenas first, so the live-arena count tracks the real working set (design:
//! "Returning memory").

use std::collections::BTreeSet;

/// Block index within one tier's arena space.
pub type BlockIdx = u32;

#[derive(Debug)]
pub struct BlockAllocator {
    free: BTreeSet<BlockIdx>,
    capacity: u32,
    /// Blocks per arena — the unit of release back to the tier's owner.
    arena_blocks: u32,
}

impl BlockAllocator {
    /// `capacity` blocks, grouped into arenas of `arena_blocks`.
    ///
    /// # Panics
    /// Panics when `arena_blocks` is zero.
    #[must_use]
    pub fn new(capacity: u32, arena_blocks: u32) -> Self {
        assert!(arena_blocks > 0, "arena_blocks must be nonzero");
        Self {
            free: (0..capacity).collect(),
            capacity,
            arena_blocks,
        }
    }

    /// The lowest free block, or None when full. Pure in the free set.
    pub fn alloc(&mut self) -> Option<BlockIdx> {
        let idx = *self.free.iter().next()?;
        self.free.remove(&idx);
        Some(idx)
    }

    /// # Panics
    /// Panics on double-free — a bookkeeping bug upstream, never a runtime state.
    pub fn free(&mut self, idx: BlockIdx) {
        assert!(idx < self.capacity, "free of out-of-range block {idx}");
        assert!(self.free.insert(idx), "double free of block {idx}");
    }

    #[must_use]
    pub fn free_blocks(&self) -> u32 {
        self.free.len() as u32
    }

    /// Arenas holding at least one ALLOCATED block — what must stay committed.
    /// Everything above the highest live arena is releasable to the tier.
    #[must_use]
    pub fn live_arenas(&self) -> u32 {
        let arenas = self.capacity.div_ceil(self.arena_blocks);
        (0..arenas)
            .filter(|a| {
                let lo = a * self.arena_blocks;
                let hi = (lo + self.arena_blocks).min(self.capacity);
                // an arena is live if any block in it is NOT free
                (lo..hi).any(|b| !self.free.contains(&b))
            })
            .count() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_pure_in_the_free_set() {
        // Two different histories arriving at the same free set must allocate
        // identically — this is the invariant, not an optimisation.
        let mut a = BlockAllocator::new(8, 4);
        let mut b = BlockAllocator::new(8, 4);

        // history A: take 0..4, free 1 and 3
        for _ in 0..4 {
            a.alloc();
        }
        a.free(3);
        a.free(1);
        // history B: take 0..4 then free 1, realloc it, free 1 and 3 in the
        // opposite order
        for _ in 0..4 {
            b.alloc();
        }
        b.free(1);
        assert_eq!(b.alloc(), Some(1));
        b.free(1);
        b.free(3);

        // same free set now; the next allocations must match exactly
        assert_eq!(a.alloc(), b.alloc()); // 1, the lowest
        assert_eq!(a.alloc(), b.alloc()); // 3
        assert_eq!(a.alloc(), b.alloc()); // 4
    }

    #[test]
    fn ascending_after_churn() {
        let mut al = BlockAllocator::new(8, 4);
        let x: Vec<_> = (0..4).map(|_| al.alloc().unwrap()).collect();
        assert_eq!(x, vec![0, 1, 2, 3]);
        al.free(2);
        al.free(0);
        // stack behavior would hand out 0 then 2? no: MRU-first would hand 0 (last
        // freed) then 2 — either way, lowest-first is what we demand:
        assert_eq!(al.alloc(), Some(0));
        assert_eq!(al.alloc(), Some(2));
        assert_eq!(al.alloc(), Some(4));
    }

    #[test]
    fn live_arenas_track_the_working_set() {
        let mut al = BlockAllocator::new(16, 4);
        assert_eq!(al.live_arenas(), 0);
        let blocks: Vec<_> = (0..6).map(|_| al.alloc().unwrap()).collect();
        assert_eq!(al.live_arenas(), 2); // blocks 0..6 span arenas 0 and 1
        for b in blocks {
            al.free(b);
        }
        assert_eq!(al.live_arenas(), 0);
    }

    #[test]
    #[should_panic(expected = "double free")]
    fn double_free_is_a_bug_not_a_state() {
        let mut al = BlockAllocator::new(4, 4);
        let b = al.alloc().unwrap();
        al.free(b);
        al.free(b);
    }
}
