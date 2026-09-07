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
    /// token `hit_units * grid_tokens()`.
    pub hit_units: usize,
}

/// A read-only content-index lookup. Calling [`Pool::lookup_prefix`] never
/// changes a conversation or a unit refcount; [`Pool::adopt_units`] is the
/// explicit commit point.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Lookup {
    pub units: Vec<UnitId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdoptError {
    UnknownUnit(UnitId),
    UnsealedUnit(UnitId),
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

    /// Walk `hashes` in order against the content index and stop at the first
    /// miss. This is deliberately read-only so a caller can inspect residency
    /// and prepare all required transfers before committing conversation refs.
    #[must_use]
    pub fn lookup_prefix(&self, hashes: &[UnitHash]) -> Lookup {
        let mut units = Vec::new();
        for h in hashes {
            match self.by_hash.get(h) {
                Some(&id) => units.push(id),
                None => break,
            }
        }
        Lookup { units }
    }

    /// Cut a conversation's adopted list down to its first `keep` units, releasing the
    /// rest.
    ///
    /// The caller already probed to get that list; re-probing a PREFIX of the same
    /// hashes would resolve them a second time, which is what the pool did to truncate
    /// to a checkpoint boundary. Nothing is looked up here -- the list is already the
    /// answer.
    pub fn truncate(&mut self, conversation: &ConversationId, keep: usize) {
        let Some(units) = self.conversations.get_mut(conversation) else {
            // Nothing adopted: `keep` of nothing is nothing, and an absent entry and an
            // empty one mean the same thing to every reader.
            return;
        };
        if keep >= units.len() {
            return;
        }
        let dropped: Vec<UnitId> = units.split_off(keep);
        for id in dropped {
            self.drop_ref(id);
        }
    }

    /// Transactionally replace `conversation`'s unit list with a previously
    /// looked-up prefix. Every id is validated before refs or the conversation
    /// map change, so an invalid adoption is a no-op.
    pub fn adopt_units(
        &mut self,
        conversation: &ConversationId,
        units: &[UnitId],
    ) -> Result<Probe, AdoptError> {
        for &id in units {
            let Some(unit) = self.units.get(&id) else {
                return Err(AdoptError::UnknownUnit(id));
            };
            if !matches!(unit.seal, Seal::Sealed(_)) {
                return Err(AdoptError::UnsealedUnit(id));
            }
        }

        for &id in units {
            self.bump(id);
        }
        // The conversation IS the list; re-probing replaces it (old refs released).
        let adopted = units.to_vec();
        let hit_units = units.len();
        if let Some(old) = self
            .conversations
            .insert(conversation.clone(), adopted.clone())
        {
            for id in old {
                self.drop_ref(id);
            }
        }
        Ok(Probe { adopted, hit_units })
    }

    /// Compatibility composition of read-only lookup plus explicit adoption.
    /// The lookup result came from this pool, so adoption cannot fail unless the
    /// implementation is internally inconsistent.
    pub fn probe(
        &mut self,
        conversation: &ConversationId,
        hashes: &[UnitHash],
    ) -> Probe {
        let lookup = self.lookup_prefix(hashes);
        self.adopt_units(conversation, &lookup.units)
            .expect("content-index lookup returned an invalid unit")
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

    /// Return the content identity of a sealed resident unit.
    ///
    /// Filling and unknown units have no durable content identity yet and
    /// therefore return `None`. This is the only reverse lookup exposed to
    /// tier movers; callers cannot mutate the content index through it.
    #[must_use]
    pub fn unit_hash(&self, id: UnitId) -> Option<UnitHash> {
        let unit = self.units.get(&id)?;
        match unit.seal {
            Seal::Sealed(hash) => Some(hash),
            Seal::Filling => None,
        }
    }

    #[must_use]
    pub fn conversation(&self, c: &ConversationId) -> Option<&[UnitId]> {
        self.conversations.get(c).map(Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ConfigRoot, grid_tokens, unit_ids};

    fn cid(s: &str) -> ConversationId {
        ConversationId(s.to_string())
    }

    fn hashes(tokens: &[u32]) -> Vec<UnitHash> {
        unit_ids(&ConfigRoot::new(b"m", (1, 1), b"g"), tokens)
    }

    /// `truncate` must release exactly what re-probing a shorter prefix released.
    ///
    /// That re-probe is what it replaced -- the pool cut its adopted list to a
    /// checkpoint boundary by resolving a prefix of the same hashes a second time. If
    /// the two disagree about refcounts, units either leak or are evicted while a
    /// conversation still points at them. So run both and compare.
    #[test]
    fn truncate_releases_what_a_shorter_probe_would_have() {
        let toks: Vec<u32> = (0..3 * grid_tokens() as u32).collect();
        let h = hashes(&toks);
        // A owns three units; B adopts all three, then keeps only the first.
        let build = || {
            let mut pool = Pool::new();
            for hh in &h {
                let u = pool.begin_fill(&cid("a"));
                pool.seal(u, *hh);
            }
            let adopted = pool.probe(&cid("b"), &h).adopted;
            assert_eq!(adopted.len(), 3, "B must adopt all three first");
            (pool, adopted)
        };
        let counts = |pool: &Pool, ids: &[UnitId]| -> Vec<u32> {
            ids.iter().map(|&u| pool.refcount(u)).collect()
        };

        let (mut by_probe, ids) = build();
        by_probe.probe(&cid("b"), &h[..1]); // the old way
        let (mut by_trunc, ids2) = build();
        by_trunc.truncate(&cid("b"), 1); // the new way
        assert_eq!(ids, ids2, "the same units in the same order");
        assert_eq!(
            counts(&by_trunc, &ids),
            counts(&by_probe, &ids),
            "truncate must leave the refcounts a shorter probe leaves"
        );
        // Concretely: the kept unit is still shared, the released two fall back to A.
        assert_eq!(counts(&by_trunc, &ids), vec![2, 1, 1]);

        // Truncating to 0 releases the rest; past the end is a no-op.
        by_trunc.truncate(&cid("b"), 0);
        assert_eq!(counts(&by_trunc, &ids), vec![1, 1, 1]);
        by_trunc.truncate(&cid("b"), 99);
        assert_eq!(counts(&by_trunc, &ids), vec![1, 1, 1]);
    }

    #[test]
    fn two_agents_share_the_preamble() {
        let preamble: Vec<u32> = (0..2 * grid_tokens() as u32).collect();
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
        let toks: Vec<u32> = (0..grid_tokens() as u32).collect();
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
        let shared: Vec<u32> = (0..grid_tokens() as u32).collect();
        let mut a = shared.clone();
        a.extend(1_000..1_000 + grid_tokens() as u32);
        let mut b = shared.clone();
        b.extend(2_000..2_000 + grid_tokens() as u32);
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

    #[test]
    fn lookup_is_read_only_until_adoption_commit() {
        let toks: Vec<u32> = (0..grid_tokens() as u32).collect();
        let h = hashes(&toks);
        let mut pool = Pool::new();
        pool.probe(&cid("owner"), &[]);
        let id = pool.begin_fill(&cid("owner"));
        let id = pool.seal(id, h[0]);
        assert_eq!(pool.refcount(id), 1);

        let lookup = pool.lookup_prefix(&h);
        assert_eq!(lookup.units, vec![id]);
        assert_eq!(pool.refcount(id), 1);
        assert!(pool.conversation(&cid("reader")).is_none());

        let committed = pool.adopt_units(&cid("reader"), &lookup.units).unwrap();
        assert_eq!(committed.adopted, vec![id]);
        assert_eq!(pool.refcount(id), 2);
    }

    #[test]
    fn invalid_adoption_is_transactional() {
        let mut pool = Pool::new();
        pool.probe(&cid("reader"), &[]);
        let filling = pool.begin_fill(&cid("writer"));
        assert_eq!(
            pool.adopt_units(&cid("reader"), &[filling]),
            Err(AdoptError::UnsealedUnit(filling))
        );
        assert_eq!(pool.conversation(&cid("reader")).unwrap(), &[]);
        assert_eq!(pool.refcount(filling), 1);

        let missing = UnitId(u64::MAX);
        assert_eq!(
            pool.adopt_units(&cid("reader"), &[missing]),
            Err(AdoptError::UnknownUnit(missing))
        );
        assert_eq!(pool.conversation(&cid("reader")).unwrap(), &[]);
        assert_eq!(pool.refcount(filling), 1);
    }

    #[test]
    fn reverse_lookup_only_exposes_sealed_content_identity() {
        let toks: Vec<u32> = (0..grid_tokens() as u32).collect();
        let h = hashes(&toks);
        let mut pool = Pool::new();
        let filling = pool.begin_fill(&cid("writer"));

        assert_eq!(pool.unit_hash(filling), None);
        assert_eq!(pool.unit_hash(UnitId(u64::MAX)), None);

        let sealed = pool.seal(filling, h[0]);
        assert_eq!(pool.unit_hash(sealed), Some(h[0]));

        pool.remove_unit(sealed);
        assert_eq!(pool.unit_hash(sealed), None);
    }
}
