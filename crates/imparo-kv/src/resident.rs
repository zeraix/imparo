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
//! Model-agnostic: layers are opaque ids; one entry is ONE 64-cell block on every
//! layer -- `grid_tokens()` positions, the same grid identity and the disk cut use.
//!
//! It used to group four blocks per entry, so residency tiled by 256 while
//! everything else was on 64, and the restore path translated between them. Every
//! translation was a place to be wrong, and several were. The block table handed to
//! the kernel is unchanged either way: `kv_slot` indexes `pt[gp >> 6]`, one entry
//! per 64 positions, and the grouping never reached the device.

use crate::state::{LayerStateGeom, StateKind};
use imparo_backend::Backend;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use crate::alloc::{BlockAllocator, BlockIdx};
use crate::identity::UnitHash;
use crate::index::{ConversationId, Pool, UnitId};
use imparo_backend::KvHostHandle;

/// Per-layer physical placement of one entry: one block per layer.
pub type UnitPlacement = BTreeMap<u32, BlockIdx>;

/// Physical pages per resident identity extent on each layer.
///
/// Identity and placement now share the backend-selected page grid, so one
/// resident unit is represented by exactly one page-table entry.
pub const UNIT_BLOCKS: usize = 1;

/// How many windowed conversations may be live at once, each with its own ring.
///
/// A windowed layer keeps ONE ring, so the conversation that ran last owns it and every
/// switch captures the outgoing window and restores the incoming one. A ring per
/// conversation removes that, and is what makes two conversations in one forward
/// expressible at all. The extra rings are address space: the caches are zero-fill
/// allocations that commit a page on first write, measured at E4B as 38.1 -> 125.2 MiB
/// allocated with the resident footprint unchanged.
#[must_use]
pub fn window_regions() -> usize {
    std::env::var("IMPARO_KV_REGIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

/// TEST INSTRUMENT (the isolation invariant's mechanical half): scatter every
/// full-attention layer's placement by pair-swapping 64-cell blocks. Output
/// must be byte-identical to identity placement -- physical location must be
/// invisible. Pair-swap breaks block adjacency at every crossing, so the
/// kernels' 8-run fast/slow paths are both exercised.
pub fn set_scrambled_tables(
    be: &dyn Backend,
    geom: &[LayerStateGeom],
    slots: usize,
    capacity: usize,
) {
    // Clamped to CAPACITY: buffers are sized min(kv_round(pos), capacity), so a
    // swap must never map into blocks the allocation clamped away (the first
    // version did, at n=65, and produced wild stores -- in the TEST, not the
    // engine).
    let backed = slots.min(capacity);
    for geom in geom {
        if !matches!(geom.kind, StateKind::Full) {
            continue;
        }
        let ident = std::env::var("IMPARO_SCRAMBLE_IDENTITY").is_ok();
        let table = scrambled_page_table(backed, ident);
        be.set_kv_page_table(geom.layer, &table);
    }
}

fn scrambled_page_table(backed: usize, identity: bool) -> Vec<u32> {
    // The backend allocates exactly ceil(logical_slots / page) entries. The old probe
    // appended eight speculative identity entries past that capacity; the stricter v2
    // native table correctly rejected them. Pair-swap only complete page pairs and keep
    // the real, possibly partial tail page identity-mapped.
    let page = crate::page_cells();
    let page_count = backed.div_ceil(page);
    let swappable = (backed / page) & !1;
    (0..page_count as u32)
        .map(|entry| {
            if identity || entry >= swappable as u32 {
                entry
            } else {
                entry ^ 1
            }
        })
        .collect()
}

/// Backend-owned persistent Host storage for exactly one sealed content unit.
/// The handle is opaque; only the server/backend seam may perform I/O or free it.
#[derive(Debug, Eq, PartialEq)]
pub struct HostResidency {
    handle: KvHostHandle,
    bytes: u64,
}

impl HostResidency {
    #[must_use]
    pub fn handle(&self) -> KvHostHandle {
        self.handle
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// One stable residency for a sealed unit. Sharing changes refs and tables, not
/// this value: every equal hash names this single Device or Host copy.
#[derive(Debug, Eq, PartialEq)]
pub enum UnitResidency {
    Device(UnitPlacement),
    Host(HostResidency),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidencyError {
    MissingResidency(UnitId),
    AlreadyDevice(UnitId),
    AlreadyHost(UnitId),
    ZeroHostBytes,
    NoDeviceCapacity,
    StalePlan(UnitId),
    StaleLookup,
    PromotionRequired(UnitId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ExpectedResidency {
    Device(UnitPlacement),
    Host { handle: KvHostHandle, bytes: u64 },
}

/// Read-only result of content lookup plus tier classification. It carries
/// enough identity to reject a stale commit after external Host -> Device I/O.
#[derive(Debug)]
pub struct ResidentProbePlan {
    hashes: Vec<UnitHash>,
    expected: Vec<ExpectedResidency>,
    pub indexed: Vec<UnitId>,
    pub device_prefix: usize,
    pub host_hits: Vec<UnitId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentProbe {
    /// Whole-unit prefix found in the content index, regardless of tier.
    pub indexed_units: usize,
    /// Leading units immediately usable before this transaction's promotions.
    pub device_prefix: usize,
    /// Units promoted by the I/O caller before commit.
    pub host_hits: Vec<UnitId>,
    /// The exact unit list committed to the conversation.
    pub adopted: Vec<UnitId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TableError {
    UnknownLayer(u32),
    MissingConversation,
    MissingResidency(UnitId),
    HostUnit(UnitId),
    UnitMissingLayer { unit: UnitId, layer: u32 },
    TailMissingLayer { tail: usize, layer: u32 },
}

/// Device -> Host transaction prepared without changing authoritative residency.
/// The caller owns the proposed Host allocation until commit.
#[derive(Debug)]
pub struct DemotionPlan {
    unit: UnitId,
    device: UnitPlacement,
    host: HostResidency,
}

impl DemotionPlan {
    #[must_use]
    pub fn unit(&self) -> UnitId {
        self.unit
    }

    #[must_use]
    pub fn device(&self) -> &UnitPlacement {
        &self.device
    }

    #[must_use]
    pub fn host(&self) -> &HostResidency {
        &self.host
    }
}

/// Host -> Device transaction. Preparing it allocates and emits `Reuse`, while
/// the authoritative Host copy remains installed until commit.
#[derive(Debug)]
pub struct PromotionPlan {
    unit: UnitId,
    expected_handle: KvHostHandle,
    expected_bytes: u64,
    device: UnitPlacement,
}

impl PromotionPlan {
    #[must_use]
    pub fn unit(&self) -> UnitId {
        self.unit
    }

    #[must_use]
    pub fn host_handle(&self) -> KvHostHandle {
        self.expected_handle
    }

    #[must_use]
    pub fn host_bytes(&self) -> u64 {
        self.expected_bytes
    }

    #[must_use]
    pub fn device(&self) -> &UnitPlacement {
        &self.device
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct SealOutcome {
    pub unit: UnitId,
    /// Present only when fresh identical Device bytes replace a Host survivor.
    /// The caller must release this backend handle after observing the outcome.
    pub host_release: Option<HostResidency>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum EvictedResidency {
    Device(UnitPlacement),
    /// The caller must release this backend handle explicitly.
    Host(HostResidency),
}

#[derive(Debug, Eq, PartialEq)]
pub struct DroppedConversation {
    pub conversation: ConversationId,
    pub evicted: Vec<EvictedResidency>,
}

/// A placement transition emitted by [`ResidentKv`], the sole owner of block
/// allocation state.
///
/// Consumers must apply these in order. `Reuse` is emitted only after every
/// layer of a unit was allocated successfully and therefore precedes any write
/// into that placement. `Free` is emitted exactly once, after the blocks have
/// returned to the deterministic allocator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlacementLifecycle {
    Reuse(UnitPlacement),
    Free(UnitPlacement),
}

/// A windowed layer's ring, written as a BLOCK TABLE.
///
/// The two addressing rules the engine has -- ring and paged -- are not two rules:
///
/// ```text
///   ring    slot = base + (pos & (ring - 1))
///   paged   slot = table[pos / page] * page + pos % page
/// ```
///
/// A ring is a power of two and at least one page, so every page-sized block lands on
/// one contiguous run inside it, and `table[j] = (base + (page*j & (ring - 1))) / page`
/// reproduces the ring exactly. Proven by `ring_and_paged_addressing_agree`, not by
/// this comment.
///
/// NOT WIRED IN. It is the designed answer to window isolation and it has its proof,
/// but nothing calls it yet: windowed layers still address through the ring directly.
///
/// Why it matters: the ring is the one piece of KV the pool does NOT place, and that
/// is why a windowed model's answer depends on what else is resident -- a neighbour's
/// cells sit inside the tiles a request reduces (docs/kv-pool-review.md). Expressing
/// the ring as a table is what lets each conversation own a REGION of it, with `base`
/// as the only difference between them, and with no kernel that has to learn a third
/// way to find a slot.
///
/// `base_slot` must be a multiple of the page; `ring` a power of two at least one page.
///
/// The page comes from `grid_tokens()` rather than a literal 64: a backend whose page
/// differs (CUDA at 128) would otherwise get a table addressing the wrong slots.
#[must_use]
pub fn ring_page_table(
    base_slot: usize,
    ring: usize,
    logical_blocks: usize,
) -> Vec<u32> {
    let page = crate::grid_tokens();
    assert!(
        ring.is_power_of_two() && ring >= page,
        "ring {ring} must be a power of two >= the page {page}"
    );
    assert!(
        base_slot % page == 0,
        "region base {base_slot} must be page aligned"
    );
    (0..logical_blocks)
        .map(|j| ((base_slot + ((j * page) & (ring - 1))) / page) as u32)
        .collect()
}

pub struct ResidentKv {
    pub pool: Pool,
    layers: Vec<u32>,
    alloc: BTreeMap<u32, BlockAllocator>,
    residency: BTreeMap<UnitId, UnitResidency>,
    /// Least-recently-probed order for whole-conversation eviction.
    lru: Vec<ConversationId>,
    /// Ordered allocator transitions waiting for the backend owner to apply.
    lifecycle: Vec<PlacementLifecycle>,
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
            residency: BTreeMap::new(),
            lru: Vec::new(),
            lifecycle: Vec::new(),
        }
    }

    fn release_placement(&mut self, placement: UnitPlacement) {
        for (&layer, &block) in &placement {
            let a = self
                .alloc
                .get_mut(&layer)
                .unwrap_or_else(|| panic!("release for unknown layer {layer}"));
            a.free(block);
        }
        self.lifecycle.push(PlacementLifecycle::Free(placement));
    }

    /// Return an unsealed tail placement to the allocator. Tail ownership stays
    /// with the caller until it invokes this method; no other code may free its
    /// blocks directly.
    pub fn release_unsealed(&mut self, placement: UnitPlacement) {
        self.release_placement(placement);
    }

    /// Drain allocator transitions in creation order. A consumer should drain
    /// immediately after a mutation; in particular, `Reuse` must be applied
    /// before the first device write into the returned placement.
    pub fn drain_lifecycle(&mut self) -> Vec<PlacementLifecycle> {
        std::mem::take(&mut self.lifecycle)
    }

    fn touch(&mut self, conv: &ConversationId) {
        self.lru.retain(|c| c != conv);
        self.lru.push(conv.clone());
    }

    /// Read content identity and classify tiers without changing LRU, refcounts,
    /// or any conversation. Backend promotion may safely fail after this call.
    pub fn prepare_probe(
        &self,
        hashes: &[UnitHash],
    ) -> Result<ResidentProbePlan, ResidencyError> {
        let lookup = self.pool.lookup_prefix(hashes);
        let mut expected = Vec::with_capacity(lookup.units.len());
        for &unit in &lookup.units {
            match self.residency.get(&unit) {
                Some(UnitResidency::Device(placement)) => {
                    expected.push(ExpectedResidency::Device(placement.clone()));
                }
                Some(UnitResidency::Host(host)) => {
                    expected.push(ExpectedResidency::Host {
                        handle: host.handle,
                        bytes: host.bytes,
                    });
                }
                None => return Err(ResidencyError::MissingResidency(unit)),
            }
        }
        let device_prefix = expected
            .iter()
            .take_while(|residency| matches!(residency, ExpectedResidency::Device(_)))
            .count();
        let host_hits = lookup
            .units
            .iter()
            .zip(&expected)
            .filter_map(|(&unit, residency)| {
                matches!(residency, ExpectedResidency::Host { .. }).then_some(unit)
            })
            .collect();
        Ok(ResidentProbePlan {
            hashes: hashes.to_vec(),
            expected,
            indexed: lookup.units,
            device_prefix,
            host_hits,
        })
    }

    /// Commit conversation refs only after every Host hit in `plan` was promoted.
    /// Device units that were already resident must retain their exact placement.
    pub fn commit_probe(
        &mut self,
        conv: &ConversationId,
        plan: ResidentProbePlan,
    ) -> Result<ResidentProbe, ResidencyError> {
        let current = self.pool.lookup_prefix(&plan.hashes);
        if current.units != plan.indexed {
            return Err(ResidencyError::StaleLookup);
        }
        for (&unit, expected) in plan.indexed.iter().zip(&plan.expected) {
            match (expected, self.residency.get(&unit)) {
                (ExpectedResidency::Device(old), Some(UnitResidency::Device(now)))
                    if old == now => {}
                (ExpectedResidency::Host { .. }, Some(UnitResidency::Device(_))) => {}
                (
                    ExpectedResidency::Host { handle, bytes },
                    Some(UnitResidency::Host(now)),
                ) if now.handle == *handle && now.bytes == *bytes => {
                    return Err(ResidencyError::PromotionRequired(unit));
                }
                (_, None) => return Err(ResidencyError::MissingResidency(unit)),
                _ => return Err(ResidencyError::StalePlan(unit)),
            }
        }
        let adopted = self
            .pool
            .adopt_units(conv, &plan.indexed)
            .expect("resident lookup returned an invalid unit")
            .adopted;
        self.touch(conv);
        Ok(ResidentProbe {
            indexed_units: plan.indexed.len(),
            device_prefix: plan.device_prefix,
            host_hits: plan.host_hits,
            adopted,
        })
    }

    /// Device-only compatibility composition. Explicit-tier callers use
    /// `prepare_probe`, promote every Host hit, then call `commit_probe`.
    pub fn probe(
        &mut self,
        conv: &ConversationId,
        hashes: &[UnitHash],
    ) -> Result<ResidentProbe, ResidencyError> {
        let plan = self.prepare_probe(hashes)?;
        if let Some(&unit) = plan.host_hits.first() {
            return Err(ResidencyError::PromotionRequired(unit));
        }
        self.commit_probe(conv, plan)
    }

    /// Cut this conversation's adopted list to its first `keep` units.
    ///
    /// What the pool wants after deciding a checkpoint boundary is shallower than what
    /// it adopted. It used to re-probe a prefix of the same hashes, which resolved them
    /// against the content index a second time for an answer it already had.
    pub fn truncate(&mut self, conv: &ConversationId, keep: usize) {
        self.touch(conv);
        self.pool.truncate(conv, keep);
    }

    /// Seal the conversation's next unit IN PLACE: the blocks holding positions
    /// [boundary, boundary + grid_tokens()) were already written by prefill through
    /// the conversation's table; sealing just names them. Merge-on-seal frees the
    /// duplicate's blocks when the hash is already resident (equal hash = equal
    /// bytes, so the survivor serves both).
    pub fn seal_unit(
        &mut self,
        conv: &ConversationId,
        hash: UnitHash,
        placement: &UnitPlacement,
    ) -> SealOutcome {
        let id = self.pool.begin_fill(conv);
        let survivor = self.pool.seal(id, hash);
        let mut host_release = None;
        if survivor == id {
            self.residency
                .insert(id, UnitResidency::Device(placement.clone()));
        } else {
            match self.residency.entry(survivor) {
                Entry::Vacant(entry) => {
                    entry.insert(UnitResidency::Device(placement.clone()));
                }
                Entry::Occupied(mut entry) => match entry.get() {
                    UnitResidency::Device(_) => {
                        self.release_placement(placement.clone());
                    }
                    UnitResidency::Host(_) => {
                        let previous =
                            entry.insert(UnitResidency::Device(placement.clone()));
                        let UnitResidency::Host(host) = previous else {
                            unreachable!()
                        };
                        host_release = Some(host);
                    }
                },
            }
        }
        SealOutcome {
            unit: survivor,
            host_release,
        }
    }

    /// Allocate one unit's blocks on every layer (for a filling tail region), or
    /// None when a layer is out of blocks (caller evicts and retries, or defers).
    pub fn alloc_unit(&mut self) -> Option<UnitPlacement> {
        let mut out = UnitPlacement::new();
        for &layer in &self.layers.clone() {
            let a = self.alloc.get_mut(&layer)?;
            if let Some(b) = a.alloc() {
                out.insert(layer, b);
            } else {
                // roll back every layer already served
                for (&l2, &b2) in &out {
                    if let Some(a2) = self.alloc.get_mut(&l2) {
                        a2.free(b2);
                    }
                }
                return None;
            }
        }
        self.lifecycle.push(PlacementLifecycle::Reuse(out.clone()));
        Some(out)
    }

    #[must_use]
    pub fn residency(&self, unit: UnitId) -> Option<&UnitResidency> {
        self.residency.get(&unit)
    }

    /// Describe a Device -> Host move without changing residency or allocator
    /// state. The caller allocates `handle`, performs backend I/O from `device`,
    /// then commits; abort returns the still-caller-owned Host allocation.
    pub fn prepare_demote(
        &self,
        unit: UnitId,
        handle: KvHostHandle,
        bytes: u64,
    ) -> Result<DemotionPlan, ResidencyError> {
        if bytes == 0 {
            return Err(ResidencyError::ZeroHostBytes);
        }
        match self.residency.get(&unit) {
            Some(UnitResidency::Device(device)) => Ok(DemotionPlan {
                unit,
                device: device.clone(),
                host: HostResidency { handle, bytes },
            }),
            Some(UnitResidency::Host(_)) => Err(ResidencyError::AlreadyHost(unit)),
            None => Err(ResidencyError::MissingResidency(unit)),
        }
    }

    /// Commit only after the complete backend demotion batch succeeded. The
    /// resulting Device `Free` is therefore never observable before durable Host
    /// ownership exists. On stale-plan failure, the proposed Host handle is
    /// returned to the caller for release.
    pub fn commit_demote(
        &mut self,
        plan: DemotionPlan,
    ) -> Result<(), (ResidencyError, HostResidency)> {
        let matches = matches!(
            self.residency.get(&plan.unit),
            Some(UnitResidency::Device(current)) if current == &plan.device
        );
        if !matches {
            return Err((ResidencyError::StalePlan(plan.unit), plan.host));
        }
        self.residency
            .insert(plan.unit, UnitResidency::Host(plan.host));
        self.release_placement(plan.device);
        Ok(())
    }

    #[must_use]
    pub fn abort_demote(plan: DemotionPlan) -> HostResidency {
        plan.host
    }

    /// Allocate a destination for a Host unit. This emits Device `Reuse` but
    /// leaves Host authoritative until backend I/O and `commit_promote` succeed.
    pub fn prepare_promote(
        &mut self,
        unit: UnitId,
    ) -> Result<PromotionPlan, ResidencyError> {
        let (expected_handle, expected_bytes) = match self.residency.get(&unit) {
            Some(UnitResidency::Host(host)) => (host.handle, host.bytes),
            Some(UnitResidency::Device(_)) => {
                return Err(ResidencyError::AlreadyDevice(unit));
            }
            None => return Err(ResidencyError::MissingResidency(unit)),
        };
        let device = self.alloc_unit().ok_or(ResidencyError::NoDeviceCapacity)?;
        Ok(PromotionPlan {
            unit,
            expected_handle,
            expected_bytes,
            device,
        })
    }

    /// Make the promoted Device placement authoritative and return the old Host
    /// allocation so the caller can explicitly release it. A stale plan is
    /// rolled back as `Free`; the installed Host residency is left untouched.
    pub fn commit_promote(
        &mut self,
        plan: PromotionPlan,
    ) -> Result<HostResidency, ResidencyError> {
        let matches = matches!(
            self.residency.get(&plan.unit),
            Some(UnitResidency::Host(host))
                if host.handle == plan.expected_handle && host.bytes == plan.expected_bytes
        );
        if !matches {
            self.release_placement(plan.device);
            return Err(ResidencyError::StalePlan(plan.unit));
        }
        let previous = self
            .residency
            .insert(plan.unit, UnitResidency::Device(plan.device));
        let Some(UnitResidency::Host(host)) = previous else {
            unreachable!()
        };
        Ok(host)
    }

    /// Roll back an uncommitted promotion. The Host copy remains authoritative;
    /// the prepared Device placement is returned through ordered `Free`.
    pub fn abort_promote(&mut self, plan: PromotionPlan) -> Result<(), ResidencyError> {
        let matches = matches!(
            self.residency.get(&plan.unit),
            Some(UnitResidency::Host(host))
                if host.handle == plan.expected_handle && host.bytes == plan.expected_bytes
        );
        self.release_placement(plan.device);
        if matches {
            Ok(())
        } else {
            Err(ResidencyError::StalePlan(plan.unit))
        }
    }

    /// Build a complete Device table or fail closed. Host/missing units and
    /// incomplete layer placements can never become a silently shortened table.
    pub fn table_for(
        &self,
        conv: &ConversationId,
        layer: u32,
        tail: &[UnitPlacement],
    ) -> Result<Vec<u32>, TableError> {
        if !self.alloc.contains_key(&layer) {
            return Err(TableError::UnknownLayer(layer));
        }
        let units = self
            .pool
            .conversation(conv)
            .ok_or(TableError::MissingConversation)?;
        let mut table = Vec::new();
        for &unit in units {
            match self.residency.get(&unit) {
                Some(UnitResidency::Device(placement)) => {
                    let block = placement
                        .get(&layer)
                        .ok_or(TableError::UnitMissingLayer { unit, layer })?;
                    table.push(*block);
                }
                Some(UnitResidency::Host(_)) => return Err(TableError::HostUnit(unit)),
                None => return Err(TableError::MissingResidency(unit)),
            }
        }
        for (index, placement) in tail.iter().enumerate() {
            let block = placement
                .get(&layer)
                .ok_or(TableError::TailMissingLayer { tail: index, layer })?;
            table.push(*block);
        }
        Ok(table)
    }

    /// Remove every refcount-zero unit in deterministic order. Device frees are
    /// emitted through lifecycle; Host handles are returned for explicit backend
    /// release. Durability remains the disk tier's responsibility.
    pub fn evict_unreferenced(&mut self) -> Vec<EvictedResidency> {
        let victims = self.pool.evictable();
        let mut freed = Vec::new();
        for id in victims {
            if let Some(residency) = self.residency.remove(&id) {
                match residency {
                    UnitResidency::Device(placement) => {
                        self.release_placement(placement.clone());
                        freed.push(EvictedResidency::Device(placement));
                    }
                    UnitResidency::Host(host) => {
                        freed.push(EvictedResidency::Host(host));
                    }
                }
            }
            self.pool.remove_unit(id);
        }
        freed
    }

    /// Drop the least-recently-probed conversation (not `keep`) entirely,
    /// returning all residency releases. The safety valve when allocation still
    /// fails after unreferenced eviction.
    pub fn drop_lru_conversation(
        &mut self,
        keep: &ConversationId,
    ) -> Option<DroppedConversation> {
        let v = self.lru.iter().find(|c| *c != keep).cloned()?;
        self.pool.forget(&v);
        self.lru.retain(|c| c != &v);
        let evicted = self.evict_unreferenced();
        Some(DroppedConversation {
            conversation: v,
            evicted,
        })
    }

    /// Forget one conversation and release every newly-unreferenced sealed
    /// residency. Host handles in the result must be released by the caller.
    pub fn forget(&mut self, conv: &ConversationId) -> Vec<EvictedResidency> {
        self.pool.forget(conv);
        self.lru.retain(|c| c != conv);
        self.evict_unreferenced()
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
    use crate::identity::{ConfigRoot, grid_tokens, unit_ids};

    fn cid(s: &str) -> ConversationId {
        ConversationId(s.to_string())
    }

    fn hashes(tokens: &[u32]) -> Vec<UnitHash> {
        unit_ids(&ConfigRoot::new(b"m", (1, 1), b"g"), tokens)
    }

    #[test]
    fn shared_preamble_is_one_copy_and_switching_moves_nothing() {
        let layers = [5u32, 11];
        let mut r = ResidentKv::new(&layers, 32);
        let pre: Vec<u32> = (0..2 * grid_tokens() as u32).collect();
        let h = hashes(&pre);

        // conversation A prefills the preamble: 2 units allocated + sealed in place
        assert_eq!(r.probe(&cid("a"), &h).unwrap().device_prefix, 0);
        let mut a_units = Vec::new();
        for hu in &h {
            let pl = r.alloc_unit().unwrap();
            a_units.push(r.seal_unit(&cid("a"), *hu, &pl));
        }
        let used_after_a = 32 - r.free_blocks(5);
        assert_eq!(used_after_a, 2);

        // B probes the same preamble: full resident hit, ZERO new blocks
        assert_eq!(r.probe(&cid("b"), &h).unwrap().device_prefix, 2);
        assert_eq!(32 - r.free_blocks(5), used_after_a);

        // both tables name the same physical blocks: one copy, two views
        let ta = r.table_for(&cid("a"), 5, &[]).unwrap();
        let tb = r.table_for(&cid("b"), 5, &[]).unwrap();
        assert_eq!(ta, tb);
        assert_eq!(ta.len(), 2);
    }

    #[test]
    fn concurrent_fill_merges_and_frees_the_duplicate() {
        let mut r = ResidentKv::new(&[3], 32);
        let toks: Vec<u32> = (0..grid_tokens() as u32).collect();
        let h = hashes(&toks)[0];
        r.probe(&cid("a"), &[]).unwrap();
        r.probe(&cid("b"), &[]).unwrap();
        let pa = r.alloc_unit().unwrap();
        let pb = r.alloc_unit().unwrap();
        assert_eq!(r.drain_lifecycle().len(), 2);
        assert_eq!(32 - r.free_blocks(3), 2 * UNIT_BLOCKS as u32);
        let ua = r.seal_unit(&cid("a"), h, &pa);
        let ub = r.seal_unit(&cid("b"), h, &pb);
        assert_eq!(ua, ub);
        // the duplicate's blocks went back to the pool
        assert_eq!(32 - r.free_blocks(3), UNIT_BLOCKS as u32);
        assert_eq!(r.drain_lifecycle(), vec![PlacementLifecycle::Free(pb)]);
        assert!(r.drain_lifecycle().is_empty());
    }

    #[test]
    fn eviction_frees_only_the_unreferenced() {
        let mut r = ResidentKv::new(&[3], 32);
        let toks: Vec<u32> = (0..2 * grid_tokens() as u32).collect();
        let h = hashes(&toks);
        r.probe(&cid("a"), &h).unwrap();
        for hu in &h {
            let pl = r.alloc_unit().unwrap();
            r.seal_unit(&cid("a"), *hu, &pl);
        }
        r.probe(&cid("b"), &h).unwrap(); // B shares
        assert!(r.evict_unreferenced().is_empty()); // everything referenced
        r.pool.forget(&cid("a"));
        assert!(r.evict_unreferenced().is_empty()); // B still holds both
        r.pool.forget(&cid("b"));
        assert_eq!(r.evict_unreferenced().len(), 2); // now they go
        assert_eq!(r.free_blocks(3), 32);
    }

    #[test]
    fn lifecycle_is_ordered_free_then_reuse_on_retry() {
        let mut r = ResidentKv::new(&[3], UNIT_BLOCKS as u32);
        let h = hashes(&(0..grid_tokens() as u32).collect::<Vec<_>>())[0];
        r.probe(&cid("old"), &[]).unwrap();
        let old = r.alloc_unit().unwrap();
        assert_eq!(
            r.drain_lifecycle(),
            vec![PlacementLifecycle::Reuse(old.clone())]
        );
        r.seal_unit(&cid("old"), h, &old);
        let _ = r.forget(&cid("old"));
        let replacement = r.alloc_unit().unwrap();
        assert_eq!(replacement, old, "lowest-free placement must be reused");
        assert_eq!(
            r.drain_lifecycle(),
            vec![
                PlacementLifecycle::Free(old.clone()),
                PlacementLifecycle::Reuse(old),
            ]
        );
    }

    #[test]
    fn drop_lru_reports_victim_and_one_free_event() {
        let mut r = ResidentKv::new(&[3], 2 * UNIT_BLOCKS as u32);
        let h = hashes(&(0..grid_tokens() as u32).collect::<Vec<_>>())[0];
        r.probe(&cid("old"), &[]).unwrap();
        let pl = r.alloc_unit().unwrap();
        r.drain_lifecycle();
        r.seal_unit(&cid("old"), h, &pl);
        r.probe(&cid("keep"), &[]).unwrap();

        let dropped = r.drop_lru_conversation(&cid("keep")).unwrap();
        assert_eq!(dropped.conversation, cid("old"));
        assert_eq!(dropped.evicted, vec![EvictedResidency::Device(pl.clone())]);
        assert_eq!(r.drain_lifecycle(), vec![PlacementLifecycle::Free(pl)]);
        assert!(r.drain_lifecycle().is_empty());
    }

    fn sealed_one(
        resident: &mut ResidentKv,
        conversation: &str,
    ) -> (UnitId, UnitHash, UnitPlacement) {
        let hash = hashes(&(0..grid_tokens() as u32).collect::<Vec<_>>())[0];
        resident.probe(&cid(conversation), &[]).unwrap();
        let placement = resident.alloc_unit().unwrap();
        let outcome = resident.seal_unit(&cid(conversation), hash, &placement);
        assert!(outcome.host_release.is_none());
        (outcome.unit, hash, placement)
    }

    #[test]
    fn probe_adopts_only_after_host_promotion_commits() {
        let mut resident = ResidentKv::new(&[3], UNIT_BLOCKS as u32);
        let (unit, hash, original) = sealed_one(&mut resident, "owner");
        resident.drain_lifecycle();

        let demote = resident
            .prepare_demote(unit, KvHostHandle(41), 4096)
            .unwrap();
        assert_eq!(
            resident.table_for(&cid("owner"), 3, &[]).unwrap(),
            vec![original[&3]]
        );
        assert!(resident.drain_lifecycle().is_empty());
        resident.commit_demote(demote).unwrap();
        assert_eq!(
            resident.drain_lifecycle(),
            vec![PlacementLifecycle::Free(original.clone())]
        );

        let plan = resident.prepare_probe(&[hash]).unwrap();
        assert_eq!(plan.indexed, vec![unit]);
        assert_eq!(plan.device_prefix, 0);
        assert_eq!(plan.host_hits, vec![unit]);
        assert!(resident.pool.conversation(&cid("reader")).is_none());
        assert_eq!(resident.pool.refcount(unit), 1);
        assert_eq!(
            resident.commit_probe(&cid("reader"), plan),
            Err(ResidencyError::PromotionRequired(unit))
        );
        assert!(resident.pool.conversation(&cid("reader")).is_none());
        assert_eq!(resident.pool.refcount(unit), 1);

        let plan = resident.prepare_probe(&[hash]).unwrap();
        let promotion = resident.prepare_promote(unit).unwrap();
        let destination = promotion.device().clone();
        assert_eq!(destination, original);
        assert_eq!(
            resident.drain_lifecycle(),
            vec![PlacementLifecycle::Reuse(destination)]
        );
        let released = resident.commit_promote(promotion).unwrap();
        assert_eq!(released.handle(), KvHostHandle(41));
        assert_eq!(released.bytes(), 4096);
        let committed = resident.commit_probe(&cid("reader"), plan).unwrap();
        assert_eq!(committed.adopted, vec![unit]);
        assert_eq!(resident.pool.refcount(unit), 2);
        assert_eq!(
            resident.table_for(&cid("reader"), 3, &[]).unwrap(),
            vec![original[&3]]
        );
    }

    #[test]
    fn abort_promotion_emits_reuse_then_free_and_keeps_host() {
        let mut resident = ResidentKv::new(&[3], UNIT_BLOCKS as u32);
        let (unit, _, placement) = sealed_one(&mut resident, "owner");
        resident.drain_lifecycle();
        let demote = resident
            .prepare_demote(unit, KvHostHandle(42), 2048)
            .unwrap();
        resident.commit_demote(demote).unwrap();
        resident.drain_lifecycle();

        let promotion = resident.prepare_promote(unit).unwrap();
        assert_eq!(promotion.device(), &placement);
        resident.abort_promote(promotion).unwrap();
        assert_eq!(
            resident.drain_lifecycle(),
            vec![
                PlacementLifecycle::Reuse(placement.clone()),
                PlacementLifecycle::Free(placement),
            ]
        );
        assert!(matches!(
            resident.residency(unit),
            Some(UnitResidency::Host(host)) if host.handle() == KvHostHandle(42)
        ));
        assert_eq!(
            resident.table_for(&cid("owner"), 3, &[]),
            Err(TableError::HostUnit(unit))
        );
    }

    #[test]
    fn demotion_keeps_identity_and_shared_host_promotes_once() {
        let mut resident = ResidentKv::new(&[3], UNIT_BLOCKS as u32);
        let (unit, hash, original) = sealed_one(&mut resident, "owner");
        resident.probe(&cid("peer"), &[hash]).unwrap();
        assert_eq!(resident.pool.refcount(unit), 2);
        resident.drain_lifecycle();

        let demote = resident
            .prepare_demote(unit, KvHostHandle(43), 1024)
            .unwrap();
        resident.commit_demote(demote).unwrap();
        let lookup = resident.pool.lookup_prefix(&[hash]);
        assert_eq!(lookup.units, vec![unit]);
        assert_eq!(resident.pool.refcount(unit), 2);

        let plan = resident.prepare_probe(&[hash]).unwrap();
        assert_eq!(plan.host_hits, vec![unit]);
        let promotion = resident.prepare_promote(unit).unwrap();
        let released = resident.commit_promote(promotion).unwrap();
        assert_eq!(released.handle(), KvHostHandle(43));
        assert_eq!(
            resident.prepare_promote(unit).unwrap_err(),
            ResidencyError::AlreadyDevice(unit)
        );
        resident.commit_probe(&cid("third"), plan).unwrap();
        assert_eq!(resident.pool.refcount(unit), 3);
        assert_eq!(
            resident.table_for(&cid("owner"), 3, &[]).unwrap(),
            vec![original[&3]]
        );
        assert_eq!(
            resident.table_for(&cid("peer"), 3, &[]).unwrap(),
            vec![original[&3]]
        );
    }

    #[test]
    fn tables_and_probe_fail_closed_on_partial_or_missing_state() {
        let mut resident = ResidentKv::new(&[3], UNIT_BLOCKS as u32);
        resident.probe(&cid("empty"), &[]).unwrap();
        assert_eq!(
            resident.table_for(&cid("missing"), 3, &[]),
            Err(TableError::MissingConversation)
        );
        assert_eq!(
            resident.table_for(&cid("empty"), 99, &[]),
            Err(TableError::UnknownLayer(99))
        );
        assert_eq!(
            resident.table_for(&cid("empty"), 3, &[UnitPlacement::new()]),
            Err(TableError::TailMissingLayer { tail: 0, layer: 3 })
        );

        let (unit, hash, _) = sealed_one(&mut resident, "owner");
        resident.residency.remove(&unit);
        assert_eq!(
            resident.table_for(&cid("owner"), 3, &[]),
            Err(TableError::MissingResidency(unit))
        );
        assert!(resident.pool.conversation(&cid("reader")).is_none());
        assert!(matches!(
            resident.prepare_probe(&[hash]),
            Err(ResidencyError::MissingResidency(id)) if id == unit
        ));
        assert!(resident.pool.conversation(&cid("reader")).is_none());
    }

    #[test]
    fn demotion_abort_and_host_eviction_return_handle_ownership() {
        let mut resident = ResidentKv::new(&[3], UNIT_BLOCKS as u32);
        let (unit, _, _) = sealed_one(&mut resident, "owner");
        resident.drain_lifecycle();
        let aborted = ResidentKv::abort_demote(
            resident
                .prepare_demote(unit, KvHostHandle(44), 512)
                .unwrap(),
        );
        assert_eq!(aborted.handle(), KvHostHandle(44));
        assert!(resident.drain_lifecycle().is_empty());

        let demote = resident
            .prepare_demote(unit, KvHostHandle(45), 512)
            .unwrap();
        resident.commit_demote(demote).unwrap();
        resident.drain_lifecycle();
        let evicted = resident.forget(&cid("owner"));
        assert_eq!(evicted.len(), 1);
        let EvictedResidency::Host(host) = &evicted[0] else {
            panic!("Host eviction must return the backend handle")
        };
        assert_eq!(host.handle(), KvHostHandle(45));
        assert!(resident.drain_lifecycle().is_empty());
    }

    #[test]
    fn partial_allocation_rollback_emits_no_external_transition() {
        let mut r = ResidentKv::new(&[1, 2], (UNIT_BLOCKS - 1) as u32);
        assert!(r.alloc_unit().is_none());
        assert!(r.drain_lifecycle().is_empty());
        assert_eq!(r.free_blocks(1), (UNIT_BLOCKS - 1) as u32);
        assert_eq!(r.free_blocks(2), (UNIT_BLOCKS - 1) as u32);
    }

    #[test]
    #[should_panic(expected = "double free")]
    fn releasing_the_same_tail_twice_is_a_bug() {
        let mut r = ResidentKv::new(&[3], UNIT_BLOCKS as u32);
        let pl = r.alloc_unit().unwrap();
        r.drain_lifecycle();
        r.release_unsealed(pl.clone());
        r.release_unsealed(pl);
    }
}

#[cfg(test)]
mod ring_tests {
    use super::{ring_page_table, scrambled_page_table};

    #[test]
    fn scramble_table_never_exceeds_the_allocated_page_capacity() {
        let page = crate::page_cells();
        let table = scrambled_page_table(8 * page + 9, false);
        assert_eq!(table.len(), 9);
        assert_eq!(&table[..8], &[1, 0, 3, 2, 5, 4, 7, 6]);
        assert_eq!(table[8], 8);
        assert_eq!(
            scrambled_page_table(8 * page + 9, true),
            (0..9).collect::<Vec<_>>()
        );
    }

    /// The claim the region change rests on: a ring and a block table addressing the
    /// same positions agree, slot for slot. If this ever fails, regions cannot be
    /// built out of the paged path and the kernels would need a third rule.
    #[test]
    fn ring_and_paged_addressing_agree() {
        // Expressed in the PAGE, not a literal 64. This test IS the proof that the two
        // addressing rules agree, so writing one of them with the number that happens
        // to be the page today would make it pass for the wrong reason.
        let page = crate::grid_tokens();
        for ring in [page, 2 * page, 4 * page, 16 * page, 64 * page] {
            for base in [0_usize, ring, 3 * ring] {
                let blocks = 40;
                let pt = ring_page_table(base, ring, blocks);
                for pos in 0..blocks * page {
                    let by_ring = base + (pos & (ring - 1));
                    let by_table = pt[pos / page] as usize * page + pos % page;
                    assert_eq!(by_ring, by_table, "ring={ring} base={base} pos={pos}");
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
            assert_eq!(*y as usize, *x as usize + ring / crate::grid_tokens());
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::identity::{ConfigRoot, PrefixHash, grid_tokens, unit_id};

    /// What the RESIDENT bookkeeping costs as a function of how many entries a
    /// conversation is tiled into.
    ///
    /// Residency holds ONE 64-cell block per entry, the same grid the disk cuts on.
    /// It used to group four, so 4x fewer entries; this measures what that grouping
    /// was buying. Entry count is the driver -- the number of BLOCKS is identical
    /// either way, only how many map entries hold them changes.
    ///
    /// IMPARO_RESIDENT_BENCH=1 to run.
    #[test]
    fn what_the_resident_tiling_costs_per_entry_count() {
        if !std::env::var("IMPARO_RESIDENT_BENCH").is_ok_and(|v| v == "1") {
            return;
        }
        let layers: Vec<u32> = (0..4).collect(); // E4B has 4 full-attention layers
        let root = ConfigRoot::new(b"m", (1, 1), b"g");
        for entries in [64_usize, 256, 1024] {
            // One block per entry, so blocks == entries.
            let mut r = ResidentKv::new(&layers, entries as u32 + 8);
            let mut prev = PrefixHash::start(&root);
            let hashes: Vec<UnitHash> = (0..entries)
                .map(|i| {
                    let p = crate::identity::prefix_hashes(
                        &root,
                        &(0..((i + 1) * grid_tokens()) as u32).collect::<Vec<u32>>(),
                    );
                    let end = *p.last().unwrap();
                    let id = unit_id(&prev, &end);
                    prev = end;
                    id
                })
                .collect();
            let a = ConversationId("a".into());
            let t = std::time::Instant::now();
            for h in &hashes {
                let pl = r.alloc_unit().expect("blocks");
                r.seal_unit(&a, *h, &pl);
            }
            let fill = t.elapsed().as_secs_f64() * 1e6;

            // A second conversation adopting the whole prefix: the hot path.
            let b = ConversationId("b".into());
            let t = std::time::Instant::now();
            let hit = r.probe(&b, &hashes).unwrap().indexed_units;
            let probe = t.elapsed().as_secs_f64() * 1e6;

            let t = std::time::Instant::now();
            let table = r.table_for(&a, layers[0], &[]).unwrap();
            let table_us = t.elapsed().as_secs_f64() * 1e6;

            println!(
                "entries={entries:5} fill={fill:8.1} us  probe={probe:7.1} us  \
                 table_for={table_us:6.1} us  (hit={hit}, blocks={})",
                table.len()
            );
        }
    }
}
