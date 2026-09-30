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
    /// Free blocks BELOW `high`. Every block at or above `high` is free as well, so the set
    /// holds only the holes a free left behind: its size follows the churn, not the tier.
    holes: BTreeSet<BlockIdx>,
    capacity: u32,
    /// One past the highest allocated block: how far the backend's storage must reach.
    high: u32,
    /// Blocks per arena — the unit of release back to the tier's owner.
    arena_blocks: u32,
    /// Blocks held back for memory the tier holds outside its blocks (a co-batch slot's
    /// rings and recurrent state): counted as used, never handed out. No block stands
    /// behind them, so `high_water` does not move.
    withheld: u32,
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
            holes: BTreeSet::new(),
            capacity,
            high: 0,
            arena_blocks,
            withheld: 0,
        }
    }

    /// The lowest free block, or None when full. Pure in the free set: the lowest free
    /// block is the lowest hole when there is one, and `high` otherwise.
    pub fn alloc(&mut self) -> Option<BlockIdx> {
        if self.used() + self.withheld >= self.capacity {
            return None;
        }
        if let Some(&idx) = self.holes.iter().next() {
            self.holes.remove(&idx);
            return Some(idx);
        }
        if self.high < self.capacity {
            self.high += 1;
            return Some(self.high - 1);
        }
        None
    }

    /// # Panics
    /// Panics on double-free — a bookkeeping bug upstream, never a runtime state.
    pub fn free(&mut self, idx: BlockIdx) {
        assert!(idx < self.capacity, "free of out-of-range block {idx}");
        assert!(
            idx < self.high && !self.holes.contains(&idx),
            "double free of block {idx}"
        );
        if idx + 1 == self.high {
            // Freed at the top: walk down past the holes to the next block still in use.
            self.high = idx;
            while self.high > 0 && self.holes.remove(&(self.high - 1)) {
                self.high -= 1;
            }
        } else {
            self.holes.insert(idx);
        }
    }

    /// One past the highest allocated block, 0 when none is: the part of the tier the
    /// backend must hold committed. Lowest-free allocation keeps it close to the blocks in
    /// use, so the storage above it can be given back.
    #[must_use]
    pub fn high_water(&self) -> u32 {
        self.high
    }

    #[must_use]
    pub fn free_blocks(&self) -> u32 {
        self.capacity.saturating_sub(self.used() + self.withheld)
    }

    /// Blocks allocated now.
    #[must_use]
    pub fn used(&self) -> u32 {
        self.high - self.holes.len() as u32
    }

    /// Holds back `n` blocks from allocation, replacing the amount held back before. False,
    /// and nothing changed, when the blocks in use leave fewer than `n` unallocated.
    pub fn set_withheld(&mut self, n: u32) -> bool {
        if u64::from(self.used()) + u64::from(n) > u64::from(self.capacity) {
            return false;
        }
        self.withheld = n;
        true
    }

    /// Blocks held back from allocation (`set_withheld`).
    #[must_use]
    pub fn withheld(&self) -> u32 {
        self.withheld
    }

    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Arenas holding at least one ALLOCATED block — what must stay committed.
    /// Everything above the highest live arena is releasable to the tier.
    #[must_use]
    pub fn live_arenas(&self) -> u32 {
        let arenas = self.high.div_ceil(self.arena_blocks);
        (0..arenas)
            .filter(|a| {
                let lo = a * self.arena_blocks;
                let hi = (lo + self.arena_blocks).min(self.high);
                // an arena is live if any block in it is NOT free
                (lo..hi).any(|b| !self.holes.contains(&b))
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

    /// Withheld blocks count as used for allocation and for `free_blocks`, take no block, and
    /// cannot be more than the blocks not in use.
    #[test]
    fn withheld_blocks_are_used_without_a_block() {
        let mut al = BlockAllocator::new(8, 4);
        for _ in 0..3 {
            al.alloc();
        }
        assert!(al.set_withheld(4));
        assert_eq!(al.free_blocks(), 1);
        assert_eq!(al.alloc(), Some(3));
        assert_eq!(al.alloc(), None);
        assert_eq!(al.high_water(), 4);
        // Four in use: withholding five would take a block in use.
        assert!(!al.set_withheld(5));
        assert_eq!(al.withheld(), 4);
        assert!(al.set_withheld(0));
        assert_eq!(al.alloc(), Some(4));
        assert_eq!(al.free_blocks(), 3);
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

    /// The same allocations as a full free set would give, in every order of frees, and
    /// `free_blocks` / `high_water` agree with a reference model of the free set.
    #[test]
    fn holes_below_high_allocate_like_a_full_free_set() {
        let cap = 64_u32;
        let mut al = BlockAllocator::new(cap, 8);
        let mut reference: BTreeSet<u32> = (0..cap).collect();
        let mut live: Vec<u32> = Vec::new();
        // A fixed pseudo-random schedule of allocs and frees.
        let mut x = 0x9e37_79b9_u32;
        for _ in 0..2000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if x % 3 != 0 || live.is_empty() {
                let want = reference.iter().next().copied();
                let got = al.alloc();
                assert_eq!(got, want);
                if let Some(b) = got {
                    reference.remove(&b);
                    live.push(b);
                }
            } else {
                let b = live.swap_remove(x as usize % live.len());
                al.free(b);
                reference.insert(b);
            }
            assert_eq!(al.free_blocks() as usize, reference.len());
            let high = (0..cap)
                .rev()
                .find(|b| !reference.contains(b))
                .map_or(0, |b| b + 1);
            assert_eq!(al.high_water(), high);
        }
    }

    /// A tier sized from the fit can hold tens of thousands of blocks per layer: building
    /// the allocator must not touch each of them.
    #[test]
    fn a_large_tier_costs_nothing_until_used() {
        let mut al = BlockAllocator::new(1 << 30, 8);
        assert_eq!(al.free_blocks(), 1 << 30);
        assert_eq!(al.alloc(), Some(0));
        assert_eq!(al.high_water(), 1);
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
