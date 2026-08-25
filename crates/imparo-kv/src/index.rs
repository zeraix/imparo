//! The no-slots bookkeeping: conversations are lists of unit ids; units carry
//! refcounts and seal state; the content index maps hash -> unit.
//!
//! ```text
//! conversation id     ->  ordered list of unit ids
//! unit id             ->  refcount, seal state, hash
//! unit hash           ->  unit id                     <- the content index
//! ```
//!
//! Sharing happens two ways (design: "Two ways a unit is shared"):
//! - ON LOOKUP: probe the request's chained hashes in order; the first miss ends
//!   the reuse; everything before it is adopted.
//! - ON SEAL: a unit whose hash is already resident drops its own copy and
//!   re-tables onto the existing one — concurrent sub-agent startup then shares
//!   instead of duplicating.
//!
//! A filling unit belongs to exactly one conversation (no write contention);
//! sealed units are immutable (divergence is allocation, not copy-on-write).

use std::collections::BTreeMap;

use crate::identity::UnitHash;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct UnitId(pub u64);

/// Optional label for deletion/pinning accounting only — NEVER consulted for
/// correctness or reuse (design: "The conversation-id decision").
#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct ConversationId(pub String);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Seal {
    Filling,
    Sealed(UnitHash),
}

#[derive(Debug)]
struct Unit {
    refcount: u32,
    seal: Seal,
}

/// The result of an adopt-on-lookup probe.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Probe {
    /// Units adopted, in prefix order; refcounts already incremented.
    pub adopted: Vec<UnitId>,
    /// Number of leading WHOLE units served from the index; prefill starts at
    /// token `hit_units * UNIT_TOKENS`.
    pub hit_units: usize,
}

/// Resident bookkeeping. Owns identity and sharing; owns no memory — placement
/// (device/tier/offset per (unit, layer) slice) arrives with the later steps.
#[derive(Debug, Default)]
pub struct Pool {
    units: BTreeMap<UnitId, Unit>,
    by_hash: BTreeMap<UnitHash, UnitId>,
    conversations: BTreeMap<ConversationId, Vec<UnitId>>,
    next_id: u64,
}

impl Pool {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn bump(&mut self, id: UnitId) {
        if let Some(u) = self.units.get_mut(&id) {
            u.refcount += 1;
        }
    }

    fn drop_ref(&mut self, id: UnitId) {
        if let Some(u) = self.units.get_mut(&id) {
            u.refcount = u.refcount.saturating_sub(1);
        }
    }

    /// Adopt-on-lookup: walk `hashes` in order against the content index; stop at
    /// the first miss. Adopted units join `conversation`'s list (created if new).
    pub fn probe(
        &mut self,
        conversation: &ConversationId,
        hashes: &[UnitHash],
    ) -> Probe {
        let mut adopted = Vec::new();
        for h in hashes {
            match self.by_hash.get(h) {
                Some(&id) => adopted.push(id),
                None => break,
            }
        }
        for &id in &adopted {
            self.bump(id);
        }
        // The conversation IS the list; re-probing replaces it (old refs released).
        let hit_units = adopted.len();
        if let Some(old) = self
            .conversations
            .insert(conversation.clone(), adopted.clone())
        {
            for id in old {
                self.drop_ref(id);
            }
        }
        Probe { adopted, hit_units }
    }

    /// A new filling unit for `conversation`'s tail. Refcount 1 (its one writer).
    pub fn begin_fill(&mut self, conversation: &ConversationId) -> UnitId {
        let id = UnitId(self.next_id);
        self.next_id += 1;
        self.units.insert(
            id,
            Unit {
                refcount: 1,
                seal: Seal::Filling,
            },
        );
        self.conversations
            .entry(conversation.clone())
            .or_default()
            .push(id);
        id
    }

    /// Merge-on-seal: seal `id` under `hash`. If the hash is already tabled, the
    /// duplicate drops its copy and every list naming it re-tables onto the
    /// existing unit; the survivor's id is returned (callers release the
    /// duplicate's memory when placement lands).
    pub fn seal(&mut self, id: UnitId, hash: UnitHash) -> UnitId {
        if let Some(&existing) = self.by_hash.get(&hash) {
            if existing != id {
                let dup_refs = self.units.get(&id).map_or(0, |u| u.refcount);
                for list in self.conversations.values_mut() {
                    for slot in list.iter_mut().filter(|s| **s == id) {
                        *slot = existing;
                    }
                }
                if let Some(u) = self.units.get_mut(&existing) {
                    u.refcount += dup_refs;
                }
                self.units.remove(&id);
                return existing;
            }
        }
        if let Some(u) = self.units.get_mut(&id) {
            u.seal = Seal::Sealed(hash);
        }
        self.by_hash.insert(hash, id);
        id
    }

    /// Drop a conversation's references (the deletion accounting path; disk-side
    /// manifest sweep arrives with the disk tier).
    pub fn forget(&mut self, conversation: &ConversationId) {
        if let Some(list) = self.conversations.remove(conversation) {
            for id in list {
                self.drop_ref(id);
            }
        }
    }

    /// Remove a unit entirely (eviction: its backing is gone, so its identity
    /// must leave the index — a hash resolving to a placement-less ghost is how
    /// a re-seal once merged onto nothing).
    pub fn remove_unit(&mut self, id: UnitId) {
        if let Some(u) = self.units.remove(&id) {
            if let Seal::Sealed(h) = u.seal {
                self.by_hash.remove(&h);
            }
        }
        for list in self.conversations.values_mut() {
            list.retain(|x| *x != id);
        }
    }

    /// Sealed units with refcount zero, lowest id first — the eviction candidates,
    /// in deterministic order.
    #[must_use]
    pub fn evictable(&self) -> Vec<UnitId> {
        self.units
            .iter()
            .filter(|(_, u)| u.refcount == 0 && matches!(u.seal, Seal::Sealed(_)))
            .map(|(&id, _)| id)
            .collect()
    }

    #[must_use]
    pub fn refcount(&self, id: UnitId) -> u32 {
        self.units.get(&id).map_or(0, |u| u.refcount)
    }

    #[must_use]
    pub fn conversation(&self, c: &ConversationId) -> Option<&[UnitId]> {
        self.conversations.get(c).map(Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ConfigRoot, UNIT_TOKENS, unit_hashes};

    fn cid(s: &str) -> ConversationId {
        ConversationId(s.to_string())
    }

    fn hashes(tokens: &[u32]) -> Vec<UnitHash> {
        unit_hashes(&ConfigRoot::new(b"m", (1, 1), b"g"), tokens)
    }

    #[test]
    fn two_agents_share_the_preamble() {
        let preamble: Vec<u32> = (0..2 * UNIT_TOKENS as u32).collect();
        let mut pool = Pool::new();

        // agent A prefills the preamble cold: probe misses, fill + seal two units
        let ha = hashes(&preamble);
        let p = pool.probe(&cid("a"), &ha);
        assert_eq!(p.hit_units, 0);
        let u0 = pool.begin_fill(&cid("a"));
        let u0 = pool.seal(u0, ha[0]);
        let u1 = pool.begin_fill(&cid("a"));
        let u1 = pool.seal(u1, ha[1]);

        // agent B probes the same preamble: full hit, no new units, refcounts rise
        let p = pool.probe(&cid("b"), &ha);
        assert_eq!(p.hit_units, 2);
        assert_eq!(p.adopted, vec![u0, u1]);
        assert_eq!(pool.refcount(u0), 2);

        // B ends; the preamble stays alive through A
        pool.forget(&cid("b"));
        assert_eq!(pool.refcount(u0), 1);
        assert!(pool.evictable().is_empty());
        pool.forget(&cid("a"));
        assert_eq!(pool.evictable(), vec![u0, u1]);
    }

    #[test]
    fn concurrent_fill_merges_on_seal() {
        let toks: Vec<u32> = (0..UNIT_TOKENS as u32).collect();
        let h = hashes(&toks);
        let mut pool = Pool::new();
        // both agents miss and fill independently
        pool.probe(&cid("a"), &h);
        pool.probe(&cid("b"), &h);
        let ua = pool.begin_fill(&cid("a"));
        let ub = pool.begin_fill(&cid("b"));
        let sa = pool.seal(ua, h[0]);
        let sb = pool.seal(ub, h[0]);
        // second seal re-tables onto the first; one unit carries both references
        assert_eq!(sa, sb);
        assert_eq!(pool.refcount(sa), 2);
        assert_eq!(pool.conversation(&cid("b")).unwrap(), &[sa]);
    }

    #[test]
    fn divergence_is_allocation_not_duplication() {
        let shared: Vec<u32> = (0..UNIT_TOKENS as u32).collect();
        let mut a = shared.clone();
        a.extend(1_000..1_000 + UNIT_TOKENS as u32);
        let mut b = shared.clone();
        b.extend(2_000..2_000 + UNIT_TOKENS as u32);
        let (ha, hb) = (hashes(&a), hashes(&b));
        assert_eq!(ha[0], hb[0]);
        assert_ne!(ha[1], hb[1]);

        let mut pool = Pool::new();
        pool.probe(&cid("a"), &ha);
        let f = pool.begin_fill(&cid("a"));
        let s = pool.seal(f, ha[0]);
        let f = pool.begin_fill(&cid("a"));
        let a1 = pool.seal(f, ha[1]);
        // b adopts the shared unit only, then fills its own divergent tail
        let p = pool.probe(&cid("b"), &hb);
        assert_eq!(p.hit_units, 1);
        assert_eq!(p.adopted, vec![s]);
        let f = pool.begin_fill(&cid("b"));
        let b1 = pool.seal(f, hb[1]);
        assert_ne!(a1, b1);
        assert_eq!(pool.refcount(s), 2);
    }
}
