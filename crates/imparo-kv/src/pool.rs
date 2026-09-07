//! Pool-mode orchestration: resident multi-conversation KV over imparo-kv's
//! ResidentKv, with the disk tier feeding and fed through the same block tables.
//!
//! The one-copy discipline end to end: full-attention KV is written ONCE by
//! prefill into pooled blocks and never moves; sharing is table aliasing;
//! switching restores only the bounded windowed state; disk restore writes
//! straight into freshly allocated blocks (table-aware -- a linear restore would
//! scatter under paging); disk spill reads unit bytes from their blocks.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::identity::UnitHash;
use crate::page_cells;
#[cfg(test)]
use crate::resident::UNIT_BLOCKS;
use crate::resident::{
    DemotionPlan, EvictedResidency, HostResidency, PlacementLifecycle, PromotionPlan,
    ResidentProbe, UnitPlacement, UnitResidency,
};
use crate::state::KvDelta;
use crate::tenant::PoolTenant;
use crate::{
    ConversationId, KvState, Manifest, ResidentKv, Store, UnitId, grid_tokens,
};
use imparo_backend::{Backend, KvHostHandle, KvTransferSpan, PoolAddressing};

// The kv_* pool methods are defaults on this trait, not inherent methods.

/// A checkpoint's identity: the hash of the last WHOLE content unit at or below
/// it, plus its exact position.
///
/// Two parts because a turn boundary does not land where an extent was cut.
/// The hash says which conversation prefix this is (content-addressed, shared);
/// the position says how far past that prefix the checkpoint reaches. Several
/// checkpoints can share a hash -- two turns inside one unit -- so the position is
/// part of the key, not a value hanging off it.
type CkptKey = (UnitHash, usize);

struct CkptEntry {
    delta: KvDelta,
    /// The predecessor checkpoint (None for an anchor).
    prev: Option<CkptKey>,
    /// The conversation's tokens between the tip unit and the boundary.
    ///
    /// The unit hash proves agreement only up to the unit's end. Everything above
    /// it is proved by these tokens: a borrower may use this checkpoint only if
    /// its own prompt carries the same ones, which is the same test the reference
    /// spells `n_tokens <= lcp`.
    tokens: Vec<u32>,
    /// Whether this boundary starts a user turn. See `crate::store::Ckpt::turn`.
    turn: bool,
}

/// A checkpoint RECORDED but not yet captured: the rings still hold its rows
/// (invariant: filled - boundary <= the model's window slack), so the bytes are
/// copied only if the conversation is switched out -- a superseded record is
/// dropped without ever being captured. This is what makes the hot path
/// zero-copy per turn.
struct PendingCkpt {
    tip: UnitHash,
    prev: Option<CkptKey>,
    boundary: usize,
    from: usize,
    /// Tokens between the tip unit and `boundary`; see `CkptEntry::tokens`.
    tokens: Vec<u32>,
    /// Whether this boundary starts a user turn. See `crate::store::Ckpt::turn`.
    turn: bool,
}

pub struct ConvState {
    pub tokens: Vec<u32>,
    pub sealed_units: usize,
    /// Pre-allocated placements for positions beyond the sealed boundary,
    /// consumed front-first as units seal.
    pub tail: Vec<UnitPlacement>,
    /// The sealed unit-hash chain on the RESIDENT tiling (what residency is keyed on).
    hashes: Vec<UnitHash>,
    /// Where this conversation's requests ended, ascending, each snapped to the
    /// `grid_tokens()` grid. THE DISK CUT: one stored extent per request rather than
    /// one per REQUEST, so a turn writes the rows it produced and no row is
    /// transferred twice. Residency keeps its own tiling above.
    cuts: Vec<usize>,
    /// Checkpoints recorded against the live rings, captured only at switch-out.
    pending: Vec<PendingCkpt>,
    /// Units already written through to disk (prefix of `hashes`).
    spilled_units: usize,
    /// The boundary the on-disk copy of this conversation describes, 0 for none.
    /// Re-committing the same one is pure I/O for no new durability.
    committed_at: usize,
    /// The boundaries this conversation already has on disk, shallowest first. Kept so a
    /// later commit can name them again without rewriting them, and so a restart can
    /// rewind to any of them rather than only the newest.
    disk_ckpts: Vec<crate::store::Ckpt>,
    /// Whether this conversation's LABEL is content-derived (the client sent no id).
    ///
    /// A keyless name moves with the content and is minted fresh on every restart, so
    /// one conversation writes a manifest under several of them and none of the old ones
    /// can ever be asked for again. That is what `adopted` cleans up; an explicit id is
    /// an identity and is never touched implicitly.
    keyless: bool,
    /// The keyless manifest this conversation was restored from, under a name that is
    /// not the one it now commits to. A rewind REPLACES the conversation, so that file
    /// is written once more under the new name and then removed. None when the client
    /// asked for a fork.
    adopted: Option<PathBuf>,
    /// This conversation's recurrent snapshot: `(boundary, bytes)`.
    ///
    /// PER CONVERSATION, not per process. There is one device buffer, so the workflow
    /// holds a scratch copy for whichever conversation is resident -- but the note about
    /// WHICH boundary those bytes belong to is a fact about a conversation, and with
    /// requests in flight for several of them a single note is one conversation's answer
    /// given to another. The pool takes it at switch-out and installs it at switch-in.
    recur_note: Option<(usize, Vec<u8>)>,
}

pub struct PoolMode {
    pub resident: ResidentKv,
    addressing: PoolAddressing,
    host_capacity_bytes: Option<u64>,
    pub convs: BTreeMap<String, ConvState>,
    /// Branch-point window checkpoints, keyed by the unit-chain tip hash at the
    /// boundary -- what lets a sub-agent adopt a shared preamble WITH valid
    /// windowed state. Stored as CHAINED DELTAS (design: "Checkpoint size"):
    /// each entry holds min(gap, window) rows and names its predecessor; restore
    /// assembles the window by walking the chain. Capped LRU that never evicts a
    /// link a live entry still chains from.
    window_ckpts: BTreeMap<CkptKey, CkptEntry>,
    ckpt_order: Vec<CkptKey>,
    ckpt_cap: usize,
    /// Turn boundaries a conversation keeps ON DISK: how far back a rewind can aim after
    /// a restart. Bigger than the resident cap because a disk link costs a turn's slice
    /// of a file, not device rows -- `IMPARO_KV_DISK_CKPT_CAP`, 20 by default. The
    /// retained set is larger than this: every root also needs the ancestors that carry
    /// it back a whole window (`cap_ckpts`).
    disk_ckpt_cap: usize,
    pub active: Option<String>,
    layers: Vec<u32>,
    /// Ring slack (min over windowed layers of ring - window): how far behind
    /// `filled` a boundary can lag and still be captured from the rings.
    slack: usize,
    /// Conversations by recency of use, newest last (the release keep-set).
    recent: Vec<String>,
    /// Bytes of RETAINED conversation KV allowed at rest, beyond the incoming
    /// conversation (which is always kept). Newest-first until the budget runs
    /// out. IMPARO_KV_RESIDENT_MB, default 64 -- at q4 that keeps roughly two
    /// long conversations resident, at f16 one.
    resident_budget: usize,
    /// Full-attention layer strides (k, v) for page-advise math.
    strides: BTreeMap<u32, (usize, usize)>,
    /// Backend supplied by the active tenant. Cached only as the lower-crate
    /// replacement for the model crate's former process-global registry.
    backend: Option<&'static dyn Backend>,
    /// How this tenant's unpooled state is checkpointed. Cached the same way and for
    /// the same reason: `end` and `note_branch` record checkpoints and have no model
    /// in hand, and the answer is a property of the model, not of the request.
    shape: crate::CheckpointShape,
    /// Allocator transitions advised to the backend since this pool was built.
    ///
    /// A PROBE THAT STAYS IN THE BINARY. The invariant it exists for -- one
    /// transition, one advice -- is otherwise unobservable: `advise_placement`
    /// returns on its first line when no backend is installed, which is every
    /// test, which is how a second advice per eviction survived review. Counting
    /// before that early return costs one add per placement and lets a test assert
    /// the property on the same code the server runs.
    advice_applied: usize,
    /// The configuration identity every hash in this pool is rooted in.
    ///
    /// HELD, not passed: `begin` and `end` need the prompt's prefix grid AND its
    /// resident unit ids, and a caller handing those in beside the tokens is handing
    /// in three things that can disagree with each other. With the root here they are
    /// derived from the tokens at the moment they are used.
    root: crate::ConfigRoot,
    /// Which conversation each windowed REGION currently holds. A windowed layer keeps
    /// one ring per region, so a conversation whose region still names it can be resumed
    /// without restoring its window at all -- the bytes never left. One region is the
    /// single-ring layout, where only the active conversation qualifies.
    regions: Vec<Option<String>>,
}

/// How many extents end at or below `b`, and where the last of them ends.
///
/// Under a fixed grid this was a division. Extents
/// end where requests ended, so the answer is a lookup in an ascending list, not a
/// division. `(0, 0)` when nothing ends at or below `b`.
fn extents_below(cuts: &[usize], b: usize) -> (usize, usize) {
    let n = cuts.partition_point(|&p| p <= b);
    (n, n.checked_sub(1).map_or(0, |i| cuts[i]))
}

/// Preserve the durable extent history when a conversation switches back from
/// resident state. A real disk restore is authoritative and replaces that history;
/// a resident resume has no manifest to replace it with.
fn cuts_for_switch_in(prior: &[usize], restored: Option<&Manifest>) -> Vec<usize> {
    restored.map_or_else(
        || prior.to_vec(),
        |manifest| manifest.cuts.iter().map(|cut| cut.end as usize).collect(),
    )
}

fn cid(label: &str) -> ConversationId {
    ConversationId(label.to_string())
}

fn log_host_event(event: &str, units: usize, bytes: u64, spans: usize) {
    if !std::env::var("IMPARO_LOG").is_ok_and(|value| value != "0" && !value.is_empty())
    {
        return;
    }
    eprintln!("[imparo] kv Host {event}: units={units} bytes={bytes} spans={spans}");
}

/// Narrow mover seam: production delegates to Backend while tests can prove
/// transaction ordering without installing a process-global backend.
trait KvHostMover {
    fn host_alloc(&self, bytes: u64) -> Result<KvHostHandle, i32>;
    fn host_free(&self, handle: KvHostHandle) -> Result<(), i32>;
    fn demote(&self, spans: &[KvTransferSpan]) -> Result<(), i32>;
    fn promote(&self, spans: &[KvTransferSpan]) -> Result<(), i32>;
    fn allocated_bytes(&self) -> u64;
    fn host_read(
        &self,
        handle: KvHostHandle,
        off: u64,
        dst: &mut [u8],
    ) -> Result<(), i32>;
}

impl<T: Backend + ?Sized> KvHostMover for T {
    fn host_alloc(&self, bytes: u64) -> Result<KvHostHandle, i32> {
        self.kv_host_alloc(bytes)
    }

    fn host_free(&self, handle: KvHostHandle) -> Result<(), i32> {
        self.kv_host_free(handle)
    }

    fn demote(&self, spans: &[KvTransferSpan]) -> Result<(), i32> {
        self.kv_demote(spans)
    }

    fn promote(&self, spans: &[KvTransferSpan]) -> Result<(), i32> {
        self.kv_promote(spans)
    }

    fn allocated_bytes(&self) -> u64 {
        self.kv_host_allocated_bytes()
    }

    fn host_read(
        &self,
        handle: KvHostHandle,
        off: u64,
        dst: &mut [u8],
    ) -> Result<(), i32> {
        self.kv_host_read(handle, off, dst)
    }
}

impl PoolMode {
    #[must_use]
    pub fn new(
        addressing: PoolAddressing,
        host_capacity_bytes: Option<u64>,
        layers: Vec<u32>,
        capacity_blocks: u32,
        slack: usize,
        strides: BTreeMap<u32, (usize, usize)>,
        root: crate::ConfigRoot,
    ) -> Self {
        let ckpt_cap = std::env::var("IMPARO_KV_CKPT_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        // How deep a rewind SURVIVES A RESTART, which is a different budget from the
        // resident chain above: a resident link is device rows held in RAM, a disk one is
        // a turn's slice of a file. One number for both meant the cheap side was sized by
        // the expensive side's limit.
        let disk_ckpt_cap = std::env::var("IMPARO_KV_DISK_CKPT_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);
        let resident_budget = std::env::var("IMPARO_KV_RESIDENT_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64)
            .saturating_mul(1 << 20);
        Self {
            resident: ResidentKv::new(&layers, capacity_blocks),
            addressing,
            host_capacity_bytes: host_capacity_bytes.filter(|bytes| *bytes != 0),
            convs: BTreeMap::new(),
            window_ckpts: BTreeMap::new(),
            ckpt_order: Vec::new(),
            ckpt_cap,
            disk_ckpt_cap,
            active: None,
            layers,
            slack,
            recent: Vec::new(),
            resident_budget,
            root,
            strides,
            backend: None,
            shape: crate::CheckpointShape::Snapshots,
            advice_applied: 0,
            regions: vec![None; crate::resident::window_regions()],
        }
    }

    /// The region this conversation's windows live in, and whether they are still there.
    ///
    /// Assigned by name rather than allocated: a collision costs the restore that every
    /// switch pays today, so the only bookkeeping is who wrote a region last.
    fn claim_region(&mut self, label: &str) -> (usize, bool) {
        let n = self.regions.len().max(1);
        let r = if n == 1 {
            0
        } else {
            let h = label.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |a, b| {
                (a ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
            });
            (h % n as u64) as usize
        };
        let ours = self.regions[r].as_deref() == Some(label);
        self.regions[r] = Some(label.to_string());
        (r, ours)
    }

    /// Whether this conversation's windows are live on the device right now.
    ///
    /// With one region that is only the conversation that ran last; with regions it is
    /// anyone who still holds theirs.
    fn windows_live(&self, label: &str) -> bool {
        if self.active.as_deref() == Some(label) {
            return true;
        }
        self.regions.len() > 1
            && self.regions.iter().any(|h| h.as_deref() == Some(label))
    }

    /// Page-advise one placement's blocks (release or reuse).
    fn advise_placement(&mut self, pl: &UnitPlacement, free: bool) {
        // Counted BEFORE the backend check, so the count is the same whether or not a
        // device is installed. See `advice_applied`.
        self.advice_applied += 1;
        let Some(be) = self.backend else {
            return;
        };
        // The SAME expansion the Host movers use. It was open-coded here as well, and
        // the two copies had already drifted: this one skipped a layer with no stride
        // geometry, the other panicked on it. They cannot disagree now.
        visit_placement_spans(&self.strides, pl, |layer, is_v, off, len| {
            if free {
                be.kv_advise_free(layer, is_v, off, len);
            } else {
                be.kv_advise_reuse(layer, is_v, off, len);
            }
        });
    }

    /// Apply every allocator transition exactly once, in source order. This is
    /// deliberately the only server path to kv_advise_free/reuse.
    fn drain_lifecycle(&mut self) {
        let events = self.resident.drain_lifecycle();
        for event in events {
            match event {
                PlacementLifecycle::Reuse(pl) => self.advise_placement(&pl, false),
                PlacementLifecycle::Free(pl) => self.advise_placement(&pl, true),
            }
        }
    }

    fn active_mover(&self) -> Result<&'static dyn Backend, String> {
        self.backend
            .ok_or_else(|| "kv pool: no active backend".to_string())
    }

    fn free_hosts_with<M: KvHostMover + ?Sized>(
        mover: &M,
        hosts: impl IntoIterator<Item = HostResidency>,
    ) -> Result<(), String> {
        let mut first_error = None;
        for host in hosts {
            if let Err(code) = mover.host_free(host.handle()) {
                first_error.get_or_insert_with(|| {
                    format!("kv pool: host free {:?} failed ({code})", host.handle())
                });
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn release_evicted_with<M: KvHostMover + ?Sized>(
        &mut self,
        mover: &M,
        evicted: Vec<EvictedResidency>,
    ) -> Result<usize, String> {
        let count = evicted.len();
        let hosts: Vec<HostResidency> = evicted
            .into_iter()
            .filter_map(|residency| match residency {
                EvictedResidency::Device(_) => None,
                EvictedResidency::Host(host) => Some(host),
            })
            .collect();
        if hosts.is_empty() {
            return Ok(count);
        }
        if self.addressing == PoolAddressing::Shared {
            return Err(
                "kv pool invariant: Shared residency returned a Host handle".into()
            );
        }
        Self::free_hosts_with(mover, hosts)?;
        Ok(count)
    }

    fn release_evicted(
        &mut self,
        evicted: Vec<EvictedResidency>,
    ) -> Result<usize, String> {
        let count = evicted.len();
        let has_host = evicted
            .iter()
            .any(|residency| matches!(residency, EvictedResidency::Host(_)));
        if !has_host {
            return Ok(count);
        }
        if self.addressing == PoolAddressing::Shared {
            return Err(
                "kv pool invariant: Shared residency returned a Host handle".into()
            );
        }
        let mover = self.active_mover()?;
        self.release_evicted_with(mover, evicted)
    }

    fn rollback_promotions_with(
        &mut self,
        plans: Vec<PromotionPlan>,
    ) -> Result<(), String> {
        let mut first_error = None;
        for plan in plans {
            if let Err(error) = self.resident.abort_promote(plan) {
                first_error.get_or_insert_with(|| {
                    format!("kv pool: promotion rollback became stale: {error:?}")
                });
            }
        }
        self.drain_lifecycle();
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn promote_probe_with<M: KvHostMover + ?Sized>(
        &mut self,
        mover: &M,
        conv: &ConversationId,
        hashes: &[UnitHash],
    ) -> Result<ResidentProbe, String> {
        let probe = self
            .resident
            .prepare_probe(hashes)
            .map_err(|error| format!("kv pool: probe failed: {error:?}"))?;
        if probe.host_hits.is_empty() {
            return self
                .resident
                .commit_probe(conv, probe)
                .map_err(|error| format!("kv pool: probe commit failed: {error:?}"));
        }

        let mut plans = Vec::with_capacity(probe.host_hits.len());
        for &unit in &probe.host_hits {
            match self.resident.prepare_promote(unit) {
                Ok(plan) => plans.push(plan),
                Err(error) => {
                    self.rollback_promotions_with(plans)?;
                    return Err(format!(
                        "kv pool: promotion reserve failed: {error:?}"
                    ));
                }
            }
        }
        // Every destination page becomes reusable before the one batched H2D call.
        self.drain_lifecycle();
        let spans = match transfer_spans_for_promotions(&self.strides, &plans) {
            Ok(spans) => spans,
            Err(error) => {
                self.rollback_promotions_with(plans)?;
                return Err(error);
            }
        };
        if let Err(code) = mover.promote(&spans) {
            self.rollback_promotions_with(plans)?;
            return Err(format!("kv pool: Host -> Device promotion failed ({code})"));
        }

        let moved_units = plans.len();
        let moved_bytes = plans
            .iter()
            .fold(0_u64, |total, plan| total.saturating_add(plan.host_bytes()));
        // Same shape as the demote commit loop: `?` here dropped both the plans after
        // the failure (still holding their reservation) and the `old_hosts` already
        // collected (Host copies the committed promotions replaced, and ours to free).
        let mut remaining = plans.into_iter();
        let mut old_hosts = Vec::with_capacity(moved_units);
        let mut failure = None;
        for plan in remaining.by_ref() {
            match self.resident.commit_promote(plan) {
                Ok(host) => old_hosts.push(host),
                Err(error) => {
                    failure =
                        Some(format!("kv pool: promotion commit failed: {error:?}"));
                    break;
                }
            }
        }
        if let Some(error) = failure {
            let _ = Self::free_hosts_with(mover, old_hosts);
            self.rollback_promotions_with(remaining.collect())?;
            return Err(error);
        }
        // The Device placements are authoritative now. Release the replaced Host
        // ownership before adopting refs or exposing a table to compute.
        Self::free_hosts_with(mover, old_hosts)?;
        let committed = self
            .resident
            .commit_probe(conv, probe)
            .map_err(|error| format!("kv pool: probe commit failed: {error:?}"))?;
        log_host_event("promote", moved_units, moved_bytes, spans.len());
        Ok(committed)
    }

    fn probe_with<M: KvHostMover + ?Sized>(
        &mut self,
        mover: &M,
        conv: &ConversationId,
        hashes: &[UnitHash],
    ) -> Result<ResidentProbe, String> {
        match self.addressing {
            PoolAddressing::Shared => self
                .resident
                .probe(conv, hashes)
                .map_err(|error| format!("kv pool: probe failed: {error:?}")),
            PoolAddressing::ExplicitHostTransfers => {
                self.promote_probe_with(mover, conv, hashes)
            }
        }
    }

    fn probe(
        &mut self,
        conv: &ConversationId,
        hashes: &[UnitHash],
    ) -> Result<ResidentProbe, String> {
        let mover = self.active_mover()?;
        self.probe_with(mover, conv, hashes)
    }

    fn rollback_demotions_with<M: KvHostMover + ?Sized>(
        mover: &M,
        plans: Vec<DemotionPlan>,
    ) -> Result<(), String> {
        Self::free_hosts_with(mover, plans.into_iter().map(ResidentKv::abort_demote))
    }

    fn demote_units_with<M: KvHostMover + ?Sized>(
        &mut self,
        mover: &M,
        units: impl IntoIterator<Item = UnitId>,
    ) -> Result<(), String> {
        let bytes = u64::try_from(self.unit_bytes())
            .map_err(|_| "kv pool: unit byte size exceeds u64".to_string())?;
        if bytes == 0 {
            return Err("kv pool: cannot demote a zero-byte unit".into());
        }
        let capacity = self.host_capacity_bytes.ok_or_else(|| {
            "kv pool: Host movement disabled because auto-fit supplied no capacity"
                .to_string()
        })?;
        let mut plans = Vec::new();
        for unit in units {
            if !matches!(
                self.resident.residency(unit),
                Some(UnitResidency::Device(_))
            ) {
                continue;
            }
            let required =
                mover.allocated_bytes().checked_add(bytes).ok_or_else(|| {
                    "kv pool: Host allocation accounting overflow".to_string()
                })?;
            if required > capacity {
                Self::rollback_demotions_with(mover, plans)?;
                return Err(format!(
                    "kv pool: Host capacity exhausted ({required} > {capacity} bytes)"
                ));
            }
            let handle = match mover.host_alloc(bytes) {
                Ok(handle) => handle,
                Err(code) => {
                    Self::rollback_demotions_with(mover, plans)?;
                    return Err(format!("kv pool: Host allocation failed ({code})"));
                }
            };
            match self.resident.prepare_demote(unit, handle, bytes) {
                Ok(plan) => plans.push(plan),
                Err(error) => {
                    let _ = mover.host_free(handle);
                    Self::rollback_demotions_with(mover, plans)?;
                    return Err(format!("kv pool: demotion prepare failed: {error:?}"));
                }
            }
        }
        if plans.is_empty() {
            return Ok(());
        }
        let spans = match transfer_spans_for_demotions(&self.strides, &plans) {
            Ok(spans) => spans,
            Err(error) => {
                Self::rollback_demotions_with(mover, plans)?;
                return Err(error);
            }
        };
        if let Err(code) = mover.demote(&spans) {
            Self::rollback_demotions_with(mover, plans)?;
            return Err(format!("kv pool: Device -> Host demotion failed ({code})"));
        }

        let moved_units = plans.len();
        let moved_bytes = plans.iter().fold(0_u64, |total, plan| {
            total.saturating_add(plan.host().bytes())
        });
        // `by_ref`, so a failure leaves the UNCOMMITTED plans in the iterator rather
        // than dropping them. They still own Host allocations this batch made; a
        // plain `for` consumed the Vec and leaked every one after the failure, which
        // the capacity check would then refuse to lend again.
        let mut remaining = plans.into_iter();
        let mut failure = None;
        for plan in remaining.by_ref() {
            if let Err((error, host)) = self.resident.commit_demote(plan) {
                let _ = mover.host_free(host.handle());
                failure = Some(format!("kv pool: demotion commit failed: {error:?}"));
                break;
            }
        }
        if let Some(error) = failure {
            let _ = Self::rollback_demotions_with(mover, remaining.collect());
            return Err(error);
        }
        // Device pages cannot become free until every Host copy is committed.
        self.drain_lifecycle();
        log_host_event("demote", moved_units, moved_bytes, spans.len());
        Ok(())
    }

    /// Return caller-owned (unsealed) tail placements through ResidentKv, then
    /// consume their Free events before control can leave this operation.
    fn release_unsealed(
        &mut self,
        placements: impl IntoIterator<Item = UnitPlacement>,
    ) {
        for pl in placements {
            self.resident.release_unsealed(pl);
        }
        self.drain_lifecycle();
    }

    /// General-path restore/rewind has already copied every row it may need by
    /// the time this is called. Return the old unsealed tail before asking the
    /// allocator for the replacement, otherwise a capacity-sized old tail can
    /// make an otherwise valid same-label rewind fail spuriously.
    fn release_tail_for_replacement(&mut self, label: &str) {
        let tail = self
            .convs
            .get_mut(label)
            .map_or_else(Vec::new, |c| std::mem::take(&mut c.tail));
        self.release_unsealed(tail);
    }

    fn touch_recent(&mut self, label: &str) {
        self.recent.retain(|l| l != label);
        self.recent.push(label.to_string());
    }

    fn remember_delta(
        &mut self,
        tip: UnitHash,
        prev: Option<CkptKey>,
        delta: KvDelta,
        tokens: Vec<u32>,
        turn: bool,
    ) {
        let key = (tip, delta.boundary);
        self.ckpt_order.retain(|h| *h != key);
        self.ckpt_order.push(key);
        self.window_ckpts.insert(
            key,
            CkptEntry {
                delta,
                prev,
                tokens,
                turn,
            },
        );
        while self.window_ckpts.len() > self.ckpt_cap {
            // Deltas depend on their ancestors (design: "eviction is not
            // independent per entry"): evict the oldest entry nothing chains
            // from, and never the entry just inserted.
            let depended: std::collections::BTreeSet<CkptKey> =
                self.window_ckpts.values().filter_map(|e| e.prev).collect();
            let Some(pos) = self.ckpt_order[..self.ckpt_order.len() - 1]
                .iter()
                .position(|h| !depended.contains(h))
            else {
                break; // whole cap is one live chain; allow the overshoot
            };
            let old = self.ckpt_order.remove(pos);
            self.window_ckpts.remove(&old);
        }
    }

    /// Deepest checkpoint at or below `upto` whose prefix `ids` reproduces:
    /// every whole unit below it hashes the same, and the tokens above the last
    /// unit are the same tokens.
    ///
    /// Materialized checkpoints AND `label`'s own pending records both count -- a
    /// pending is captured before anything needs its bytes.
    ///
    /// `upto` bounds the search so a delta never chains from a checkpoint AHEAD of
    /// itself. `hashes` covers the whole prompt, so without it the "previous"
    /// checkpoint for a boundary at 600 could come back as one at 900.
    #[must_use]
    pub fn prev_ckpt_at(
        &self,
        label: &str,
        hashes: &[UnitHash],
        ids: &[u32],
        upto: usize,
    ) -> Option<CkptKey> {
        self.matching_ckpts(Some(label), hashes, ids, upto, usize::MAX)
            .first()
            .copied()
    }

    /// Every checkpoint `ids` can legitimately resume from, deepest first.
    ///
    /// Two tests, one per half of the identity:
    ///
    /// ```text
    ///   units [0, u)      the unit-chain hash at u-1 must match     content-addressed
    ///   rows  [ue, B)     the stored tokens must be these tokens    literal comparison
    /// ```
    ///
    /// `upto` caps the boundary (a delta must chain from something BEHIND it, and
    /// `hashes` covers the whole prompt, so without the cap a checkpoint further
    /// forward would come back as the "previous" one). `max_units` caps how many
    /// units must be resident. `label`, when given, also admits that
    /// conversation's own pending records, which are captured before anything
    /// needs their bytes.
    fn matching_ckpts(
        &self,
        label: Option<&str>,
        hashes: &[UnitHash],
        ids: &[u32],
        upto: usize,
        max_units: usize,
    ) -> Vec<CkptKey> {
        let agrees = |tip: UnitHash, b: usize, tok: &[u32]| {
            let u = b / grid_tokens();
            b <= upto
                && u >= 1
                && u <= max_units
                && hashes.get(u - 1) == Some(&tip)
                && ids.len() >= b
                && ids[u * grid_tokens()..b] == tok[..]
        };
        let mut out: Vec<CkptKey> = self
            .window_ckpts
            .iter()
            .filter(|&(&(tip, b), e)| agrees(tip, b, &e.tokens))
            .map(|(&k, _)| k)
            .collect();
        if let Some(c) = label.and_then(|l| self.convs.get(l)) {
            out.extend(
                c.pending
                    .iter()
                    .filter(|p| agrees(p.tip, p.boundary, &p.tokens))
                    .map(|p| (p.tip, p.boundary)),
            );
        }
        out.sort_by_key(|k| std::cmp::Reverse(k.1));
        out.dedup();
        out
    }

    /// Capture one recorded branch point: the windowed delta from the rings, the
    /// recurrent snapshot, and the full-attention rows above the last whole unit.
    ///
    /// ONE body for both timings -- eager (the boundary would slide out of ring
    /// reach before a switch-out could reach it) and lazy (at switch-out). They
    /// differ in WHEN, never in what a checkpoint is.
    fn capture_ckpt<T: PoolTenant + ?Sized>(
        &mut self,
        model: &T,
        label: &str,
        pk: PendingCkpt,
    ) {
        let mut delta = model.kv_capture_delta(pk.boundary, pk.from);
        let unit_end = pk.boundary / grid_tokens() * grid_tokens();
        if pk.boundary > unit_end {
            let tail = self
                .convs
                .get(label)
                .map_or_else(Vec::new, |c| c.tail.clone());
            let geom = model.kv_state_geometry();
            if let Some(be) = model.backend() {
                delta.tail = self.read_rows(
                    &cid(label),
                    &tail,
                    unit_end,
                    pk.boundary,
                    &geom,
                    be,
                );
                if delta.tail.is_empty() {
                    // rows unreachable: no checkpoint beats a wrong one
                    if crate::log_on() {
                        eprintln!(
                            "[imparo] kv ckpt dropped: no rows for [{unit_end}, {})",
                            pk.boundary
                        );
                    }
                    return;
                }
            } else {
                return;
            }
        }
        if !recurrent_ok(model, &delta.recurrent) {
            if crate::log_on() {
                eprintln!(
                    "[imparo] kv ckpt {} dropped: no recurrent state",
                    pk.boundary
                );
            }
            return;
        }
        self.remember_delta(pk.tip, pk.prev, delta, pk.tokens, pk.turn);
    }

    /// The link at `key` and every ancestor its window still needs, newest first:
    /// each link and whether its boundary starts a user turn.
    ///
    /// WINDOWS ONLY. A link owns `[from, boundary)`. Whoever restores at that boundary
    /// reads `[boundary - window, boundary)`, which no single link covers -- the
    /// ancestors are the only copy of the rows below `from`. Naming just the newest
    /// link was enough while the disk wrote whole windows and is not now.
    ///
    /// A `Snapshots` model has no window to rebuild and never calls this: each of its
    /// checkpoints is the whole state at its own boundary.
    ///
    /// It used to hand back each link's tail TOKENS as well. Nothing reads them: the
    /// manifest's tails are measured from the last extent and derived at commit, so
    /// this was cloning a vector per link for no reader.
    fn ckpt_chain(&self, key: CkptKey, win_max: usize) -> Vec<(KvDelta, bool)> {
        let mut out = Vec::new();
        let floor = key.1.saturating_sub(win_max);
        let mut need = key.1;
        let mut cur = Some(key);
        while let Some(k) = cur {
            let Some(e) = self.window_ckpts.get(&k) else {
                break;
            };
            out.push((e.delta.clone(), e.turn));
            need = need.min(e.delta.from);
            if need <= floor || e.delta.from == 0 || out.len() >= 32 {
                break;
            }
            cur = e.prev;
        }
        out
    }

    /// Assemble the full windowed state at `tip` by walking its delta chain.
    /// None when a needed link was evicted -- the caller treats it as no
    /// checkpoint.
    fn try_assemble<T: PoolTenant + ?Sized>(
        &self,
        model: &T,
        key: CkptKey,
    ) -> Option<KvState> {
        let mut chain: Vec<&KvDelta> = Vec::new();
        let mut cur = Some(key);
        while let Some(t) = cur {
            let Some(e) = self.window_ckpts.get(&t) else {
                break; // missing link: assemble decides if it was needed
            };
            chain.push(&e.delta);
            if chain.len() >= 16 {
                break;
            }
            cur = e.prev;
        }
        model.kv_assemble_chain(&chain)
    }

    fn host_unit_canonical_with<M: KvHostMover + ?Sized>(
        &self,
        mover: &M,
        host: &HostResidency,
    ) -> Result<Vec<u8>, String> {
        let raw_len = usize::try_from(host.bytes())
            .map_err(|_| "kv pool: Host unit exceeds addressable memory".to_string())?;
        let mut raw = vec![0_u8; raw_len];
        mover
            .host_read(host.handle(), 0, &mut raw)
            .map_err(|code| format!("kv pool: Host unit read failed ({code})"))?;
        let mut cursor = 0_usize;
        let mut state = KvState {
            boundary: grid_tokens(),
            full: Vec::with_capacity(self.strides.len()),
            window: Vec::new(),
            recurrent: Vec::new(),
        };
        for (&layer, &(k_stride, v_stride)) in &self.strides {
            let k_len = grid_tokens()
                .checked_mul(k_stride)
                .ok_or_else(|| "kv pool: canonical K length overflow".to_string())?;
            let v_len = grid_tokens()
                .checked_mul(v_stride)
                .ok_or_else(|| "kv pool: canonical V length overflow".to_string())?;
            let k_end = cursor
                .checked_add(k_len)
                .ok_or_else(|| "kv pool: canonical K offset overflow".to_string())?;
            let v_end = k_end
                .checked_add(v_len)
                .ok_or_else(|| "kv pool: canonical V offset overflow".to_string())?;
            if v_end > raw.len() {
                return Err("kv pool: Host unit is shorter than its geometry".into());
            }
            state.full.push(crate::KvLayerState {
                layer,
                base_pos: 0,
                positions: grid_tokens(),
                k: raw[cursor..k_end].to_vec(),
                v: raw[k_end..v_end].to_vec(),
            });
            cursor = v_end;
        }
        if cursor != raw.len() {
            return Err("kv pool: Host unit is longer than its geometry".into());
        }
        crate::state::unit_blobs_at(&state, &[grid_tokens()])
            .into_iter()
            .next()
            .ok_or_else(|| {
                "kv pool: canonical Host unit encoding produced no blob".to_string()
            })
    }

    /// Evict the oldest idle Host-only ownership to Disk. Every fallible read and
    /// durability acknowledgement completes before refs, residency, or handles
    /// change, so a failure leaves the Host copy authoritative.
    fn evict_oldest_host_to_disk_with<M: KvHostMover + ?Sized>(
        &mut self,
        mover: &M,
        keep: &str,
        store: Option<&Store>,
        disk: Option<&crate::disk::DiskQueue>,
    ) -> Result<bool, String> {
        let (Some(store), Some(disk)) = (store, disk) else {
            return Ok(false);
        };
        for label in self.recent.clone() {
            if label == keep || self.active.as_deref() == Some(label.as_str()) {
                continue;
            }
            let Some(units) = self.resident.pool.conversation(&cid(&label)) else {
                continue;
            };
            let referenced_hosts: Vec<UnitId> = units
                .iter()
                .copied()
                .filter(|unit| {
                    matches!(
                        self.resident.residency(*unit),
                        Some(UnitResidency::Host(_))
                    )
                })
                .collect();
            if referenced_hosts.is_empty() {
                continue;
            }
            // A shared Host unit is one physical handle with several conversation
            // refs. Removing an old ref makes progress without I/O; only the final
            // ref owns the last copy and therefore needs a durability handoff.
            let last_copy_hosts: Vec<(UnitId, UnitHash)> = referenced_hosts
                .iter()
                .copied()
                .filter(|unit| self.resident.pool.refcount(*unit) == 1)
                .filter_map(|unit| Some((unit, self.resident.pool.unit_hash(unit)?)))
                .collect();
            let mut durable = Vec::with_capacity(last_copy_hosts.len());
            let mut evicted_bytes = 0_u64;
            for (unit, hash) in &last_copy_hosts {
                let Some(UnitResidency::Host(host)) = self.resident.residency(*unit)
                else {
                    return Err(
                        "kv pool: Host pressure residency changed before I/O".into()
                    );
                };
                evicted_bytes = evicted_bytes.saturating_add(host.bytes());
                durable.push((*hash, self.host_unit_canonical_with(mover, host)?));
            }
            for (hash, bytes) in durable {
                disk.ensure_unit_durable(store, hash, bytes)?;
            }
            // Every last-copy Host unit is durable. Shared handles with surviving
            // refs remain installed; final handles are returned and freed once.
            let tail = self
                .convs
                .remove(&label)
                .map_or_else(Vec::new, |conv| conv.tail);
            let evicted = self.resident.forget(&cid(&label));
            self.release_evicted_with(mover, evicted)?;
            self.release_unsealed(tail);
            self.recent.retain(|recent| recent != &label);
            if last_copy_hosts.is_empty() {
                log_host_event("drop-shared-reference", referenced_hosts.len(), 0, 0);
            } else {
                log_host_event(
                    "evict-to-disk",
                    last_copy_hosts.len(),
                    evicted_bytes,
                    0,
                );
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn make_host_room_with<M: KvHostMover + ?Sized>(
        &mut self,
        mover: &M,
        needed_bytes: u64,
        keep: &str,
        store: Option<&Store>,
        disk: Option<&crate::disk::DiskQueue>,
    ) -> Result<(), String> {
        let capacity = self.host_capacity_bytes.ok_or_else(|| {
            "kv pool: Host movement disabled because auto-fit supplied no capacity"
                .to_string()
        })?;
        loop {
            let required = mover
                .allocated_bytes()
                .checked_add(needed_bytes)
                .ok_or_else(|| {
                    "kv pool: Host allocation accounting overflow".to_string()
                })?;
            if required <= capacity {
                return Ok(());
            }
            if !self.evict_oldest_host_to_disk_with(mover, keep, store, disk)? {
                return Err(format!(
                    "kv pool: Host capacity exhausted ({required} > {capacity} bytes)"
                ));
            }
        }
    }

    /// Free Device capacity without deleting an idle conversation. Every
    /// candidate reached this path after its turn's write-through. Host-cap
    /// pressure spills the oldest idle Host ownership through canonical bytes
    /// and a durability acknowledgement before any handle is released.
    fn demote_idle_for_allocation(
        &mut self,
        keep: &str,
        store: Option<&Store>,
        disk: Option<&crate::disk::DiskQueue>,
    ) -> Result<bool, String> {
        if self.addressing != PoolAddressing::ExplicitHostTransfers {
            return Ok(false);
        }
        let protected: BTreeSet<UnitId> = self
            .resident
            .pool
            .conversation(&cid(keep))
            .into_iter()
            .flatten()
            .copied()
            .collect();
        for label in self.recent.clone() {
            if label == keep || self.active.as_deref() == Some(label.as_str()) {
                continue;
            }
            let Some(units) = self.resident.pool.conversation(&cid(&label)) else {
                continue;
            };
            let candidates: BTreeSet<UnitId> = units
                .iter()
                .copied()
                .filter(|unit| {
                    !protected.contains(unit)
                        && matches!(
                            self.resident.residency(*unit),
                            Some(UnitResidency::Device(_))
                        )
                })
                .collect();
            if candidates.is_empty() {
                continue;
            }
            let mover = self.active_mover()?;
            let unit_bytes = u64::try_from(self.unit_bytes())
                .map_err(|_| "kv pool: unit byte size exceeds u64".to_string())?;
            let needed = u64::try_from(candidates.len())
                .ok()
                .and_then(|count| count.checked_mul(unit_bytes))
                .ok_or_else(|| "kv pool: Host capacity request overflow".to_string())?;
            self.make_host_room_with(mover, needed, keep, store, disk)?;
            self.demote_units_with(mover, candidates)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn alloc_units(
        &mut self,
        n: usize,
        keep: &str,
        store: Option<&Store>,
        disk: Option<&crate::disk::DiskQueue>,
    ) -> Result<Vec<UnitPlacement>, String> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            loop {
                if let Some(pl) = self.resident.alloc_unit() {
                    // Reuse is applied BEFORE this placement escapes to a writer.
                    self.drain_lifecycle();
                    out.push(pl);
                    break;
                }
                let evicted = self.resident.evict_unreferenced();
                let evicted_count = self.release_evicted(evicted)?;
                self.drain_lifecycle();
                if evicted_count != 0 {
                    continue;
                }
                if self.addressing == PoolAddressing::ExplicitHostTransfers {
                    if self.demote_idle_for_allocation(keep, store, disk)? {
                        continue;
                    }
                    self.release_unsealed(std::mem::take(&mut out));
                    return Err(
                        "kv pool exhausted: no idle Device unit can be demoted safely"
                            .into(),
                    );
                }
                if let Some(dropped) = self.resident.drop_lru_conversation(&cid(keep)) {
                    let victim = dropped.conversation.0;
                    self.release_evicted(dropped.evicted)?;
                    let tail =
                        self.convs.remove(&victim).map_or_else(Vec::new, |c| c.tail);
                    self.recent.retain(|l| l != &victim);
                    if self.active.as_deref() == Some(victim.as_str()) {
                        self.active = None;
                    }
                    self.release_unsealed(tail);
                    // Shared-address behavior remains the existing drop path.
                    continue;
                }
                // Earlier units in this allocation request never reached a
                // caller. Roll them back now instead of leaking on Err.
                self.release_unsealed(std::mem::take(&mut out));
                return Err("kv pool exhausted: request exceeds device capacity".into());
            }
        }
        Ok(out)
    }

    fn apply_tables<T: PoolTenant + ?Sized>(
        &self,
        model: &T,
        label: &str,
        tail: &[UnitPlacement],
    ) -> Result<(), String> {
        let mut tables = BTreeMap::new();
        for &layer in &self.layers {
            let table =
                self.resident
                    .table_for(&cid(label), layer, tail)
                    .map_err(|error| {
                        format!("kv pool: incomplete block table: {error:?}")
                    })?;
            tables.insert(layer, table);
        }
        model.kv_apply_tables(&tables);
        Ok(())
    }

    /// Prepare for a request: adopt what is resident, restore what disk holds,
    /// allocate the tail, apply tables and window state. Returns the resume
    /// position (0 = cold). `upper` bounds prompt+generation for tail sizing.
    ///
    /// # Errors
    /// Returns an error when the pool cannot fit the request even after eviction.
    #[allow(clippy::too_many_lines)]
    pub fn begin<T: PoolTenant + ?Sized>(
        &mut self,
        model: &mut T,
        store: Option<&Store>,
        disk: Option<&crate::disk::DiskQueue>,
        label: &str,
        ids: &[u32],
        upper: usize,
        // Whether `label` was derived from the prompt's content because the client sent
        // no conversation id. Passed in rather than sniffed from the string: a client is
        // free to send an id that looks content-derived, and deleting THAT would be a
        // silent data loss.
        keyless: bool,
        // The client said `x-new-conversation: true`. It borrows, it does not continue,
        // so nothing it restores from may be superseded -- even when the prompt agrees
        // with the stored chain to the last token, which is exactly what a fork at the
        // tip looks like from here.
        new_conversation: bool,
    ) -> Result<usize, String> {
        self.backend = model.backend();
        self.shape = crate::CheckpointShape::of(&model.kv_state_geometry());
        // Two views of this prompt, both from its tokens: `grid` is prefix(p) at every
        // multiple of grid_tokens(), which is what a stored boundary is matched against;
        // `hashes` is the resident tiling's ids, which is what residency is keyed on.
        let grid = crate::prefix_hashes(&self.root, ids);
        let hashes =
            crate::unit_ids_at(&self.root, ids, &crate::resident_bounds(ids.len()));
        let hashes = &hashes[..];
        // A different conversation is coming in: the outgoing one's pending
        // checkpoints are captured from the still-intact rings and its state is
        // written through to disk NOW -- the only moment bytes ever move for it.
        // Then residency beyond the keep-set is released.
        if let Some(out) = self.active.clone() {
            if out != label {
                self.switch_out(model, store, disk, &out)?;
                self.release_idle(model, label, hashes, store, disk)?;
            }
        }
        self.touch_recent(label);
        // Point the windowed layers at THIS conversation's ring before anything reads or
        // writes them. `ours` says the ring still holds what this conversation left there.
        let (region, ours) = self.claim_region(label);
        model.kv_apply_region(region);
        // Where this conversation stopped, read before anything below replaces its state.
        let tip_before = self.convs.get(label).map_or(0, |c| c.tokens.len());
        // Fast path: pure append of the ACTIVE conversation -- tables and windows are
        // already live; only the tail may need room. Not extended to any conversation
        // holding its region: a recurrent model's state is one device buffer with a
        // per-conversation note, and this path never restores it.
        if self.active.as_deref() == Some(label) {
            if let Some(c) = self.convs.get(label) {
                if !c.tokens.is_empty()
                    && ids.len() >= c.tokens.len()
                    && ids[..c.tokens.len()] == c.tokens[..]
                {
                    // Rewriting positions below the resume point is benign here: the
                    // token prefix matches, so identical bytes land on identical slots.
                    let resume =
                        crate::identity::resume_point(c.tokens.len(), ids.len());
                    let need_units = upper
                        .div_ceil(grid_tokens())
                        .saturating_sub(c.sealed_units + self.convs[label].tail.len());
                    if need_units > 0 {
                        let more = self.alloc_units(need_units, label, store, disk)?;
                        self.convs.get_mut(label).unwrap().tail.extend(more);
                    }
                    let tail = self.convs[label].tail.clone();
                    let _ = self.probe(&cid(label), hashes)?;
                    self.apply_tables(model, label, &tail)?;
                    return Ok(resume);
                }
            }
        }

        // Same-conversation rewind through the general path: the rings still hold
        // this conversation's state, so its pending checkpoints must be captured
        // BEFORE the restore below overwrites them.
        if self.windows_live(label) {
            self.materialize_pendings(model, label);
        }
        // General path: content probe against resident units.
        let hit = self.probe(&cid(label), hashes)?;

        // Deepest checkpoint this prompt reproduces AND whose chain still assembles.
        // Its boundary is a turn start, so it sits anywhere -- `k` is where the
        // POOLED part of it ends, and everything from there to `k_at` rides in the
        // checkpoint's own tail rows.
        let mut k_at = 0;
        let mut k_state: Option<KvState> = None;
        let mut k_tail: Vec<crate::KvLayerState> = Vec::new();
        for key in self.matching_ckpts(None, hashes, ids, usize::MAX, hit.indexed_units)
        {
            if let Some(st) = self.try_assemble(model, key) {
                k_at = key.1;
                k_tail.clone_from(&self.window_ckpts[&key].delta.tail);
                k_state = Some(st);
                break;
            }
        }
        let k = k_at / grid_tokens();
        if crate::log_on() {
            eprintln!(
                "[imparo] kv begin {label}: prompt_units={} resident_hit={} ckpts={:?} chose={k_at}",
                hashes.len(),
                hit.indexed_units,
                self.window_ckpts.keys().map(|k| k.1).collect::<Vec<_>>()
            );
        }

        // Disk can be deeper than residency (restart, evicted content): restore
        // INTO the pool if its committed chain extends past k.
        let mut from_disk: Option<(Manifest, Vec<Vec<u8>>, Vec<Vec<u8>>)> = None;
        // The manifest a restore stood on: its path, whether its label was
        // content-derived, and the boundaries it named. A keyless one this conversation
        // goes on to cover is superseded at the next commit -- see `commit_to_disk`.
        let mut adopted_from: Option<(PathBuf, bool)> = None;
        if let Some(st) = store {
            if st.best_match(&grid).is_none() && crate::log_on() {
                // How far the stored chain and this prompt agree. A partial match means
                // the prompt DIVERGED from what was stored -- the re-rendered text of an
                // earlier turn does not re-tokenise to the stream that produced it --
                // rather than that nothing related is on disk.
                let best = st
                    .prefix_matches(&grid)
                    .into_iter()
                    .map(|(_, m, n)| (n, m.cuts.len()))
                    .max();
                match best {
                    Some((n, stored)) => eprintln!(
                        "[imparo] kv disk: prompt agrees with a stored chain for {n} of its {stored} units, so nothing is adoptable (prompt has {})",
                        hashes.len()
                    ),
                    None => eprintln!(
                        "[imparo] kv disk: no manifest shares even one unit with this prompt's {}",
                        hashes.len()
                    ),
                }
            }
            // The DEEPEST boundary this prompt can actually stand on. A manifest carries
            // every boundary its conversation still holds, and a prompt agrees with that
            // chain only up to where the two diverge -- so the choice is: among the
            // boundaries below the agreement, take the highest.
            //
            // Both halves matter. Keeping only the newest boundary means a restart cannot
            // rewind or branch to an earlier turn at all. Demanding that the WHOLE stored
            // chain be a prefix means a conversation that grew past the prompt -- which is
            // every conversation whose last turn is being re-asked -- adopts nothing, even
            // when its system prompt and tools match to the token ("agrees for 4 of its 5
            // units, so nothing is adoptable").
            type Candidate =
                (usize, crate::Manifest, Vec<Vec<u8>>, Vec<Vec<u8>>, PathBuf);
            let mut best: Option<Candidate> = None;
            let shape = crate::CheckpointShape::of(&model.kv_state_geometry());
            for (mpath, m, agree) in st.prefix_matches(&grid) {
                let candidates: Vec<&crate::store::Ckpt> = m.ckpts.iter().collect();
                let mcuts: Vec<usize> = m.cuts.iter().map(|c| c.end as usize).collect();
                for c in candidates {
                    let b = c.boundary as usize;
                    let (units, unit_end) = extents_below(&mcuts, b);
                    if units > agree
                        || units > m.cuts.len()
                        // Not on THIS build's grid. A store written where the grid
                        // was coarser or finer is still readable -- prefix(p) is a
                        // hash of the tokens below p, so the match is exact wherever
                        // both have a boundary -- but resuming off the grid does not
                        // reproduce a cold pass. Skip the boundary, keep the store.
                        || b % grid_tokens() != 0
                        || b > ids.len()
                        || b <= k_at
                        || c.tail.len() != b - unit_end
                        || ids[unit_end..b] != c.tail[..]
                    {
                        continue;
                    }
                    if best.as_ref().is_some_and(|(at, ..)| *at >= b) {
                        continue;
                    }
                    // Does the chain below this link reach back a whole window? The
                    // manifest answers it in arithmetic -- each link covers
                    // [from, boundary) -- so a boundary whose ancestors have aged out is
                    // skipped HERE, in favour of a shallower one, instead of failing the
                    // request when the assemble finds the hole.
                    if !m.ckpts.is_empty() {
                        let mut need = b;
                        let floor = shape.chain_floor(b).unwrap_or(b);
                        let mut desc: Vec<&crate::store::Ckpt> = m
                            .ckpts
                            .iter()
                            .filter(|o| o.boundary as usize <= b)
                            .collect();
                        desc.sort_by_key(|o| std::cmp::Reverse(o.boundary));
                        for o in desc {
                            if need <= floor {
                                break;
                            }
                            if o.boundary as usize >= need {
                                need = need.min(o.from as usize);
                            }
                        }
                        if need > floor {
                            continue;
                        }
                    }
                    let Some(u) = m.cuts[..units]
                        .iter()
                        .map(|c| st.get_unit(&c.unit))
                        .collect::<Option<Vec<Vec<u8>>>>()
                    else {
                        continue;
                    };
                    // The link and everything below it, newest first: the assemble walks
                    // down until the window is covered. A store written before links has
                    // one whole-window blob, which reads back as an anchor.
                    let mut chain: Vec<Vec<u8>> = Vec::new();
                    let load = |h: &crate::UnitHash| st.get_checkpoint(h);
                    // `m.ckpts` cannot be empty here: this loop iterates it. The
                    // branch that used to handle the empty case read `<manifest>.ckpt`
                    // through `Store::checkpoint_for` -- a file no writer in this tree
                    // has ever produced -- so it was dead twice over. Both are gone.
                    let Some(target) = load(&c.blob) else {
                        continue;
                    };
                    chain.push(target);
                    let mut older: Vec<&crate::store::Ckpt> =
                        m.ckpts.iter().filter(|o| o.boundary < c.boundary).collect();
                    older.sort_by_key(|o| std::cmp::Reverse(o.boundary));
                    let mut missing = false;
                    for o in older {
                        let Some(bytes) = load(&o.blob) else {
                            // Named but not on disk. A chain with a hole restores
                            // garbage, so drop this candidate and let a shallower
                            // boundary answer instead of failing the request.
                            missing = true;
                            break;
                        };
                        chain.push(bytes);
                    }
                    if missing {
                        continue;
                    }

                    let chosen = crate::Manifest {
                        boundary: c.boundary,
                        cuts: m.cuts[..units].to_vec(),
                        ckpts: m.ckpts.clone(),
                        keyless: m.keyless,
                    };
                    best = Some((b, chosen, u, chain, mpath.clone()));
                }
            }
            if let Some((b, m, u, chain, mpath)) = best {
                if crate::log_on() {
                    eprintln!("[imparo] kv disk: restoring at boundary {b}");
                }
                // Absent the fork header this IS that conversation, being rewound -- so
                // the file it was stored under is replaced rather than left behind. The
                // caller sets `new_conversation` when the client asked for a fork, and
                // then nothing is adopted for replacement.
                adopted_from = Some((mpath, m.keyless));
                from_disk = Some((m, u, chain));
            } else if crate::log_on() {
                // WHY none of them fit, not just that none did. Each boundary is refused
                // for exactly one of four reasons and they mean different things: too
                // deep for what we agree on, past the prompt, already beaten by a
                // resident checkpoint, or its sub-unit tokens are somebody else's.
                for (_, m, agree) in st.prefix_matches(&grid) {
                    let mcuts: Vec<usize> =
                        m.cuts.iter().map(|c| c.end as usize).collect();
                    for c in &m.ckpts {
                        let b = c.boundary as usize;
                        let (units, ue) = extents_below(&mcuts, b);
                        let why = if units > agree || units > m.cuts.len() {
                            "needs more units than we agree on"
                        } else if b % grid_tokens() != 0 {
                            "not on this build's resume grid"
                        } else if b > ids.len() {
                            "past the end of this prompt"
                        } else if b <= k_at {
                            "not deeper than the resident checkpoint"
                        } else if c.tail.len() != b - ue || ids[ue..b] != c.tail[..] {
                            "its sub-unit tokens are not ours"
                        } else {
                            "usable"
                        };
                        eprintln!(
                            "[imparo] kv disk:   boundary {b} (units {units}, agree \
{agree}, chain {}): {why}",
                            m.cuts.len()
                        );
                    }
                }
                let best_n = st
                    .prefix_matches(&grid)
                    .into_iter()
                    .map(|(_, m, n)| (n, m.cuts.len(), m.ckpts.len()))
                    .max();
                match best_n {
                    Some((n, stored, kept)) => eprintln!(
                        "[imparo] kv disk: agrees for {n} of a {stored}-unit chain with \
                         {kept} kept boundaries, none of them usable here (prompt {} units)",
                        hashes.len()
                    ),
                    None => eprintln!(
                        "[imparo] kv disk: no manifest shares a unit with this prompt's {}",
                        hashes.len()
                    ),
                }
            }
        }

        // What the restored manifest says is already on disk, so the next commit names
        // those boundaries again instead of rewriting them.
        let mut restored_ckpts: Vec<crate::store::Ckpt> = Vec::new();
        // Keep the store's historical extent boundaries across an in-process
        // switch-back. Only an actual disk restore has a manifest authoritative enough
        // to replace them. Losing these cuts made the next commit reinterpret a delta
        // as a standalone checkpoint and produced a chain that restart correctly
        // rejected as incomplete.
        let prior_cuts = self
            .convs
            .get(label)
            .map(|c| c.cuts.clone())
            .unwrap_or_default();
        let restored_cuts = cuts_for_switch_in(
            &prior_cuts,
            from_disk.as_ref().map(|(manifest, _, _)| manifest),
        );
        // (tip, link, sub-unit tail tokens) for every link a disk restore assembled from,
        // so the resident chain can be rebuilt from it -- see the comment at the fill.
        let mut restored_links: Vec<(UnitHash, KvDelta, Vec<u32>)> = Vec::new();
        let prior_disk_ckpts = self
            .convs
            .get(label)
            .map(|c| c.disk_ckpts.clone())
            .unwrap_or_default();
        // All old-tail reads and checkpoint captures are complete. Free it
        // before the first fresh allocation in either restore branch.
        self.release_tail_for_replacement(label);
        // The conversation keeps only the units at or below its resume boundary;
        // everything beyond is re-filled into fresh tail blocks (sealed shared
        // units must never be rewritten).
        // (resume position, windowed+recurrent state, full rows above the last unit)
        let (resume, restored_windows, tail_rows): (
            usize,
            Option<KvState>,
            Vec<crate::KvLayerState>,
        ) = if let Some((manifest, unit_bytes, chain_blobs)) = from_disk {
            // DISK extents and RESIDENT units are different counts. The manifest's
            // extents end where requests ended; residency holds one entry per grid
            // step. What
            // residency can take is the whole tiles the extents reach, and the ids
            // are this prompt's own -- the prompt agrees with the manifest to
            // `reach`, so its resident ids there are the same rows by another name.
            let reach = manifest.cuts.last().map_or(0, |c| c.end as usize);
            let n = (reach / grid_tokens()).min(hashes.len());
            let resident_ids = &hashes[..n];
            let boundary = manifest.boundary as usize;
            restored_ckpts.clone_from(&manifest.ckpts);
            // adopt what is already resident among the manifest's units; the
            // rest gets fresh blocks and a table-aware write-in
            let resident_hit = self.probe(&cid(label), resident_ids)?.indexed_units;
            let missing = n - resident_hit;
            let fresh = self.alloc_units(missing, label, store, disk)?;
            let links: Vec<crate::KvDelta> = chain_blobs
                .iter()
                .map(|b| crate::state::delta_from_blob(b))
                .collect::<Result<_, _>>()?;
            let refs: Vec<&crate::KvDelta> = links.iter().collect();
            // The links this restore just parsed ARE the resident chain, and dropping
            // them made the next commit re-capture a whole window: `matching_ckpts`
            // found nothing, so the fallback wrote an anchor (21.0 MiB on E4B f16)
            // where a link is 3.5 MiB -- one whole window per restart per
            // conversation. Rebuilt below, shallowest first, so each `prev` key
            // exists before the link that names it.
            restored_links = links
                .iter()
                .filter_map(|d| {
                    let u = d.boundary / grid_tokens();
                    let tip = *hashes.get(u.checked_sub(1)?)?;
                    // Tile grid, not the manifest's extent grid: these become RESIDENT
                    // records and are compared as such.
                    let tail = ids
                        .get(u * grid_tokens()..d.boundary)
                        .map(<[u32]>::to_vec)?;
                    Some((tip, d.clone(), tail))
                })
                .collect();
            restored_links.sort_by_key(|(_, d, _)| d.boundary);
            // The BOUNDED geometry, the same view the resident chain assembles
            // against: a delta covers a window, not a whole context.
            let wgeom = crate::state::bounded_geometry(&model.kv_state_geometry());
            let state =
                crate::state::state_from_units_and_chain(&unit_bytes, &wgeom, &refs)?;
            // write the missing units' bytes into their blocks
            let be = model.backend().ok_or("no backend")?;
            for (i, pl) in fresh.iter().enumerate() {
                let u = resident_hit + i;
                for ls in &state.full {
                    let stride = ls.k.len() / ls.positions;
                    let vstride = ls.v.len() / ls.positions;
                    let Some(&blk) = pl.get(&ls.layer) else {
                        continue;
                    };
                    {
                        // One extent's rows into one page. The OFFSET already said
                        // `grid_tokens()` while the LENGTH said 64 -- agreeing only
                        // because the grid takes its value from the page. Both say it.
                        let n = grid_tokens();
                        let pos0 = u * n;
                        be.write_kv_bytes(
                            ls.layer,
                            false,
                            (blk as usize * page_cells() * stride) as u64,
                            &ls.k[pos0 * stride..(pos0 + n) * stride],
                        );
                        be.write_kv_bytes(
                            ls.layer,
                            true,
                            (blk as usize * page_cells() * vstride) as u64,
                            &ls.v[pos0 * vstride..(pos0 + n) * vstride],
                        );
                    }
                }
            }
            // seal the written units under their manifest hashes
            for (i, pl) in fresh.into_iter().enumerate() {
                let u = resident_hit + i;
                let _ = self.resident.seal_unit(&cid(label), hashes[u], &pl);
            }
            // Duplicate placement frees must be advised before any later reuse.
            self.drain_lifecycle();
            // Rows above the last whole unit: the checkpoint carried them, and
            // they go into this conversation's OWN tail blocks below -- there is
            // no unit for them to be part of.
            let lo = n * grid_tokens();
            let leftover: Vec<crate::KvLayerState> = state
                .full
                .iter()
                .filter(|_| boundary > lo)
                .map(|ls| {
                    let (ks, vs) =
                        (ls.k.len() / ls.positions, ls.v.len() / ls.positions);
                    crate::KvLayerState {
                        layer: ls.layer,
                        base_pos: lo,
                        positions: boundary - lo,
                        k: ls.k[lo * ks..boundary * ks].to_vec(),
                        v: ls.v[lo * vs..boundary * vs].to_vec(),
                    }
                })
                .collect();
            let windows = KvState {
                boundary: state.boundary,
                full: Vec::new(),
                window: state.window,
                // Carried through: a checkpoint that dropped it would restore a
                // recurrent model with a zero convolution history.
                recurrent: state.recurrent,
            };
            // Keyed on the resident unit the boundary sits in. `n` can be 0 when
            // the manifest's extents do not reach a whole tile, and there is then
            // no resident unit to key an anchor on -- the restore still stands, it
            // just leaves no resident chain behind.
            if let Some(&tip) = n.checked_sub(1).and_then(|i| hashes.get(i)) {
                // remember as an ANCHOR delta (disk holds the full window)
                self.remember_delta(
                    tip,
                    None,
                    KvDelta {
                        boundary: windows.boundary,
                        from: 0,
                        window: clone_windows(&windows).window,
                        tail: leftover.clone(),
                        // The anchor carries the disk checkpoint's recurrent snapshot,
                        // so a chain assembled from it is complete and the pool does
                        // not have to decline the restore for want of one.
                        recurrent: windows.recurrent.clone(),
                    },
                    // The RESIDENT record's tail, on the TILE grid. `manifest.tail`
                    // is the manifest's, measured from the last EXTENT -- handing
                    // that over left a restored conversation with a resident
                    // checkpoint whose tail `matching_ckpts` measures differently, so
                    // it could not resume from what it had just restored.
                    ids.get(boundary / grid_tokens() * grid_tokens()..boundary)
                        .map(<[u32]>::to_vec)
                        .unwrap_or_default(),
                    // Restored from the manifest, whose kept set is already one per
                    // turn: treat it as a turn boundary so a later step collapses
                    // against it rather than beside it.
                    true,
                );
            }
            (boundary, Some(windows), leftover)
        } else if k_at > 0 {
            // truncate the adopted list to the checkpointed boundary
            self.resident.truncate(&cid(label), k);
            (k_at, k_state, k_tail)
        } else {
            self.resident.truncate(&cid(label), 0);
            (0, None, Vec::new())
        };

        let resume_units = resume / grid_tokens();
        // tail for everything beyond the sealed resume boundary
        let need_units = upper.div_ceil(grid_tokens()) - resume_units;
        let tail = self.alloc_units(need_units, label, store, disk)?;
        // The checkpoint's own rows above the last extent go into the
        // FIRST tail unit -- this conversation's blocks, not the donor's.
        if !tail_rows.is_empty() {
            if let Some(be) = model.backend() {
                Self::write_rows(
                    &tail_rows,
                    resume_units * grid_tokens(),
                    &tail,
                    &model.kv_state_geometry(),
                    be,
                );
            }
        }
        self.apply_tables(model, label, &tail)?;
        // A conversation's ATTENTION cache and its RECURRENT state move together or not
        // at all -- so they travel in one `KvState` and land in one call.
        //
        // They used to be two: a restore (which carries both, because KvState holds the
        // recurrent bytes) OR `kv_set_filled(0)`, and then a separate install bolted on
        // afterwards. Adopting resident blocks takes the second branch, which moved the
        // attention cache and nothing else, so the device kept whichever conversation ran
        // last. Two things that must be kept in step are a defect waiting to happen; one
        // call cannot be half-done.
        let stored = self.convs.get(label).and_then(|c| c.recur_note.clone());
        // The window this conversation left in its OWN region is still there, so the
        // restore below would write the same bytes back. It is only the same bytes while
        // the ring still reaches the resume point: the ring holds [tip - ring, tip], the
        // window at `resume` needs [resume - window, resume], and `slack` is ring - window.
        // The full rows and the recurrent state are restored either way.
        let live_window = ours
            && resume <= tip_before
            && tip_before - resume <= self.slack
            && restored_windows.is_some();
        let mut state = restored_windows.unwrap_or(KvState {
            boundary: 0,
            full: Vec::new(),
            window: Vec::new(),
            recurrent: Vec::new(),
        });
        // The stored note wins when it describes the boundary being resumed at; a restore
        // that just wrote the device is already carrying its own.
        if let Some((at, blob)) = stored {
            if at == state.boundary || state.recurrent.is_empty() {
                state.recurrent = blob;
            }
        }
        if live_window {
            state.window.clear();
            if crate::log_on() {
                eprintln!(
                    "[imparo] kv region {region}: {label} keeps its window (tip {tip_before}, resume {resume})"
                );
            }
        }
        if let Err(e) = model.kv_resume(&state) {
            // apply_tables made these placements visible to the failed resume,
            // so consume their Free notifications before returning the error.
            self.release_unsealed(tail);
            return Err(e);
        }
        let prior_note = model.kv_recurrent_note();
        let replaced = self.convs.insert(
            label.to_string(),
            ConvState {
                tokens: Vec::new(),
                sealed_units: resume_units,
                tail,
                hashes: hashes[..resume_units].to_vec(),
                // What the store already holds for this conversation. Re-deriving the
                // cut from tokens would invent request boundaries that never happened.
                cuts: restored_cuts,
                pending: Vec::new(),
                // adopted/restored units are already on disk (or shared with a
                // conversation that will spill them); nothing to write for them
                spilled_units: resume_units,
                committed_at: 0,
                // What this conversation has on disk survives a switch-in. A restore
                // learns it from the manifest; an adoption from resident blocks restores
                // nothing and must NOT forget -- dropping the list here made every commit
                // name one boundary, and the rewind targets vanished one turn later.
                disk_ckpts: if restored_ckpts.is_empty() {
                    prior_disk_ckpts
                } else {
                    restored_ckpts
                },
                keyless,
                // Only a KEYLESS manifest can be superseded, and only by a conversation
                // that is not already writing to it. An explicit id is an identity: the
                // client can ask for it again, so only `erase` removes it.
                // Set only when this prompt CONTINUED the stored conversation (see the
                // candidate loop); a branch leaves it None and the donor's file stands.
                // BOTH sides must be nameless for a replace. This conversation, because
                // an explicit id means the client named something else and that is a
                // FORK by the id rule; and the file's own label, because an explicit id
                // is an identity only `erase` may remove. Same file, nothing to do.
                adopted: adopted_from
                    .filter(|_| keyless && !new_conversation)
                    .and_then(|(p, was_keyless)| {
                        let mine = store.map(|st| st.manifest_path(label));
                        (was_keyless && mine.as_ref() != Some(&p)).then_some(p)
                    }),
                // Whatever this switch-in put on the device: a restore seeds it, and an
                // adoption from resident blocks carries the previous note forward.
                recur_note: prior_note,
            },
        );
        // General-path rewind/restore replaces its old unsealed tail. Those
        // placements are not indexed by Pool and must be returned explicitly.
        if let Some(old) = replaced {
            self.release_unsealed(old.tail);
        }
        // The restored chain becomes the resident one. Shallowest first: `prev` names the
        // link below, and remember_delta's LRU must see that key already present or it
        // would treat the ancestor as evictable.
        let mut prev: Option<CkptKey> = None;
        for (tip, delta, tokens) in std::mem::take(&mut restored_links) {
            let key = (tip, delta.boundary);
            self.remember_delta(tip, prev, delta, tokens, true);
            prev = Some(key);
        }
        self.active = Some(label.to_string());
        // A resume past the prompt is not a slow path, it is a slice out of bounds one
        // frame later. Everything above proves its own boundary against `ids`; this
        // says so once, where it is cheap to check and impossible to miss.
        if resume > ids.len() {
            return Err(format!(
                "kv pool: resume {resume} exceeds the {} prompt tokens",
                ids.len()
            ));
        }
        Ok(resume)
    }

    /// Turn end: seal the new whole units in place (naming blocks -- no copies)
    /// and RECORD the boundary checkpoint. Nothing is captured and nothing is
    /// written to disk here: bytes move only at switch-out, release, or shutdown
    /// (the one-copy law -- the rings and blocks already hold this state).
    ///
    /// # Errors
    /// Returns an error when sealing finds the tail underflowed (a logic bug).
    pub fn end(
        &mut self,
        label: &str,
        final_tokens: Vec<u32>,
        hashes_full: &[UnitHash],
    ) -> Result<(), String> {
        let sealed_target = final_tokens.len() / grid_tokens();
        let filled = final_tokens.len();

        let Some(c) = self.convs.get_mut(label) else {
            return Ok(());
        };
        let mut sealed_placements: Vec<(UnitHash, UnitPlacement)> = Vec::new();
        while c.sealed_units < sealed_target {
            if c.tail.is_empty() {
                return Err("pool: tail underflow at seal".into());
            }
            let pl = c.tail.remove(0);
            sealed_placements.push((hashes_full[c.sealed_units], pl));
            c.sealed_units += 1;
        }
        c.tokens = final_tokens;
        c.hashes = hashes_full[..sealed_target].to_vec();
        // THE DISK CUT: where this request ended, snapped to the grid a resume can
        // land on. One extent per request, so a turn writes the rows it produced and
        // no row is written twice. Cuts at or above it are this stream's no longer --
        // a rewind replaced the tokens there.
        let cut = crate::identity::resume_point(filled, filled);
        c.cuts.retain(|&p| p < cut);
        // THE OTHER CUT is the branch point, recorded by `note_branch` / `note_ckpt`
        // where a user message starts, so that every sealed boundary is also an extent
        // cut. Two positions, both of them boundaries the design already names.
        //
        // A cut at the PROMPT end was tried as a third: it splits the user message off
        // from the generated reply, and the reasoning was that a client re-sends the
        // conversation rather than the raw stream. It earns nothing. The branch point
        // already lands there -- measured on the shape workload, the prompt end and the
        // branch both snapped to 512 -- so with it removed the disk is byte-identical
        // (same 5 extents, same hashes) and every gate number is unchanged.
        record_cut(&mut c.cuts, cut);
        // Superseded records die uncaptured; what stays must remain capturable
        // from the rings (filled - boundary <= slack), which the eager-capture
        // rule at the branch site guarantees for everything recorded this turn.
        let slack = self.slack;
        c.pending.retain(|pk| filled - pk.boundary <= slack);
        let mut host_releases = Vec::new();
        for (h, pl) in &sealed_placements {
            let outcome = self.resident.seal_unit(&cid(label), *h, pl);
            if let Some(host) = outcome.host_release {
                host_releases.push(host);
            }
        }
        // Merge-on-seal may have returned duplicate placements to the allocator.
        self.drain_lifecycle();
        if !host_releases.is_empty() {
            if self.addressing == PoolAddressing::Shared {
                return Err(
                    "kv pool invariant: Shared seal returned a Host handle".into()
                );
            }
            Self::free_hosts_with(self.active_mover()?, host_releases)?;
        }

        // The turn's own resume point, WHERE THIS REQUEST'S EXTENT ENDS. It used to
        // sit at the last resident tile, which is below the cut whenever a request
        // ends off the grid -- and a checkpoint below the cut can never satisfy
        // the tail rule, because the extents already carry those rows. Cut and
        // checkpoint land together: that is the whole point of cutting here.
        let boundary = cut;
        if boundary > 0 && sealed_target > 0 {
            let tip = hashes_full[sealed_target - 1];
            let tokens = self
                .convs
                .get(label)
                .map_or_else(Vec::new, |c| c.tokens.clone());
            let prev = self.chain_from(self.prev_ckpt_at(
                label,
                hashes_full,
                &tokens,
                boundary - 1,
            ));
            if let Some(c) = self.convs.get_mut(label) {
                // The RESIDENT tail: tokens between the last whole TILE and the
                // boundary, which is what `matching_ckpts` compares. It is not the
                // disk tail -- that one counts from the last EXTENT, and commit
                // derives it there. One field for both left this empty while the rule
                // wanted 128 tokens, and every resident match failed.
                let tile_end = boundary / grid_tokens() * grid_tokens();
                let tokens = c
                    .tokens
                    .get(tile_end..boundary)
                    .map(<[u32]>::to_vec)
                    .unwrap_or_default();
                if !c.pending.iter().any(|pk| pk.boundary == boundary) {
                    c.pending.push(PendingCkpt {
                        tip,
                        prev,
                        boundary,
                        from: prev.map_or(0, |(_, b)| b),
                        tokens,
                        // A STEP: where this request stopped, not where a user turn
                        // began. Steps within a turn replace each other.
                        turn: false,
                    });
                }
            }
        }
        self.active = Some(label.to_string());
        Ok(())
    }

    /// Capture `label`'s pending checkpoints from the rings (valid: nothing has
    /// run since this conversation's last forward).
    fn materialize_pendings<T: PoolTenant + ?Sized>(&mut self, model: &T, label: &str) {
        let pending = match self.convs.get_mut(label) {
            Some(c) => std::mem::take(&mut c.pending),
            None => return,
        };
        for pk in pending {
            self.capture_ckpt(model, label, pk);
        }
    }

    /// The moment bytes move: capture `label`'s pending window checkpoints and
    /// write its unspilled units + manifest + checkpoint blob to disk. Called at
    /// switch-out, release, and graceful shutdown.
    ///
    /// # Errors
    /// Returns an error on disk I/O failure.
    pub fn switch_out<T: PoolTenant + ?Sized>(
        &mut self,
        model: &T,
        store: Option<&Store>,
        disk: Option<&crate::disk::DiskQueue>,
        label: &str,
    ) -> Result<(), String> {
        // The model's note describes the conversation being switched OUT -- it is still
        // the resident one at this moment. Keep it with that conversation, so the next
        // one to become resident does not inherit an answer about someone else.
        if let Some(c) = self.convs.get_mut(label) {
            c.recur_note = model.kv_recurrent_note();
        }
        self.commit_to_disk(model, store, disk, label).map(|_| ())
    }

    /// Write this conversation's unwritten units, its manifest and its checkpoint.
    ///
    /// Called at the END OF EVERY TURN (write-through), and again at switch-out,
    /// release and shutdown. Per-turn is what makes the others cheap: by the time a
    /// conversation is evicted or the process is asked to stop, its units are already
    /// on disk and only the checkpoint can have moved -- so a SIGTERM is not the moment
    /// a long conversation discovers it has to write everything it ever computed. The
    /// reference does the same thing at the same point (`write_through` at the end of
    /// its checkpoint creation).
    ///
    /// Returns whether anything was written, so the caller can skip a GC pass that
    /// would find nothing new.
    ///
    /// # Errors
    /// Returns an error on disk I/O failure.
    pub fn commit_to_disk<T: PoolTenant + ?Sized>(
        &mut self,
        model: &T,
        store: Option<&Store>,
        disk: Option<&crate::disk::DiskQueue>,
        label: &str,
    ) -> Result<bool, String> {
        self.backend = model.backend();
        self.shape = crate::CheckpointShape::of(&model.kv_state_geometry());
        // Pendings first: a recorded-but-uncaptured branch point has no bytes, and the
        // rings still hold what it needs at this instant. This is also what makes the
        // per-turn write-through and the switch-out path do the same thing.
        self.materialize_pendings(model, label);
        let Some(c) = self.convs.get(label) else {
            return Ok(false);
        };
        let (sealed, spilled) = (c.sealed_units, c.spilled_units);
        let tokens = c.tokens.clone();
        let cut_at = c.cuts.clone();
        let hashes = c.hashes.clone();
        // A request commonly ends ABOVE the last sealed unit, so the blocks holding
        // its rows are in the tail, not in the sealed table.
        let conv_tail = c.tail.clone();
        // ONE cut list drives both the files written and the manifest that names them.
        // They used to be derived separately, which is two chances to disagree about
        // what is on disk.
        let (cuts, at) = crate::store::Cut::for_stream(&self.root, &tokens, &cut_at);
        // COUNT AGAINST WHAT WAS EMITTED, not against what was asked for. `for_stream`
        // can return fewer entries than `bounds`: it skips a cut below one grid step and
        // stops at one beyond the tokens. Everything downstream slices `cuts` and `at`,
        // so counting extents in `cut_at` and indexing these was a panic waiting for the
        // two to differ.
        //
        // They cannot differ today -- a recorded cut is `resume_point(filled, filled)`
        // and is only pushed when positive, so every one is a positive multiple of the
        // grid. They CAN differ the moment the grid changes, because the grid now takes
        // its value from the backend's page and is not part of the ConfigRoot: a store
        // cut on 64 read by a build whose page is 128 has cuts that `for_stream` drops.
        // The boundary path already skips those (`b % grid_tokens() != 0`); this is the
        // same rule for the cut path.
        let cut_ends: Vec<usize> = cuts.iter().map(|c| c.end as usize).collect();
        if sealed == 0 || hashes.len() < sealed {
            return Ok(false);
        }
        // Nothing has happened since the last write: the tip has not moved, no tile
        // sealed, and no extent was cut above what is already committed. Re-committing
        // is 21 MiB of checkpoint (measured, E4B) for no new durability, and switch-out
        // and shutdown both land here right after a turn end did the work.
        //
        // The cut is tested explicitly rather than inferred from the tiles: a request
        // that ends without sealing a new tile still produced an extent, and "same
        // units" is a statement about memory, not about what disk is missing.
        let newest_cut = cut_at.last().copied().unwrap_or(0);
        if spilled == sealed
            && newest_cut <= c.committed_at
            && c.committed_at == model.kv_runtime().filled
        {
            return Ok(false);
        }
        if let (Some(st), Some(dq)) = (store, disk) {
            let t_write = std::time::Instant::now();
            let be = model.backend().ok_or("no backend")?;
            let geom = model.kv_state_geometry();
            let mut lo = 0_usize;
            let mut extents_written = 0_usize;
            for c in &cuts {
                let hi = c.end as usize;
                // Already durable, or already handed to the writer: reading it off
                // the device again would be a memcpy for a file that exists.
                if dq.holds(st, &c.unit) {
                    lo = hi;
                    continue;
                }
                let Some(bytes) = self.read_unit_canonical(
                    &cid(label),
                    lo,
                    hi,
                    &conv_tail,
                    &geom,
                    be,
                ) else {
                    lo = hi;
                    continue;
                };
                dq.put_unit(c.unit, bytes)?;
                extents_written += 1;
                lo = hi;
            }
            // NOT the conversation's tip. The tip is prompt + the model's own
            // generated tokens, and the next request re-renders those through the
            // template -- so the only positions the next prompt is GUARANTEED to
            // agree with are the ones the client sent, ending at the last user
            // message. Committing the tip made every restart miss: the manifest's
            // tail tokens described text the next prompt spelled differently, the
            // check refused it, and reuse went to 0 where it had been 512.
            //
            // The recorded checkpoints are exactly the right positions -- they sit at
            // turn boundaries, and they already carry their window rows, their
            // recurrent snapshot and their sub-unit tail. So the deepest of those
            // wins, and a fresh capture at the tip is only the fallback for a
            // conversation that has none.
            let ck_at = model.kv_runtime().filled.max(sealed * grid_tokens());
            // The LINK, not the assembled window: the disk holds what the resident
            // chain holds -- `[from, boundary)` per turn -- so a turn's checkpoint
            // costs a turn's slice. try_assemble is still what proves the link's
            // ancestors are reachable; a link whose chain has a hole is no use to a
            // restore, and finding that out here is cheaper than at read time.
            let recorded = self
                .matching_ckpts(None, &hashes, &tokens, ck_at, sealed)
                .into_iter()
                .find_map(|key| {
                    self.try_assemble(model, key)?;
                    self.window_ckpts.get(&key)?;
                    Some(key)
                });
            let shape = crate::CheckpointShape::of(&geom);
            let chain: Vec<(crate::KvDelta, bool)> = if let Some(key) = recorded {
                match shape {
                    // THE SIMPLE CASE. Every checkpoint is the whole state at its
                    // boundary, so what to persist is just the ones this stream
                    // reproduces -- no chain, no floor, no ancestors. `matching_ckpts`
                    // returns them deepest first, which is the order the consumer
                    // wants: it reads `chain.first()` as the newest.
                    crate::CheckpointShape::Snapshots => self
                        .matching_ckpts(None, &hashes, &tokens, ck_at, sealed)
                        .into_iter()
                        .filter_map(|k| {
                            self.window_ckpts.get(&k).map(|e| (e.delta.clone(), e.turn))
                        })
                        .take(32)
                        .collect(),
                    // THE COMPLICATED ONE: a delta is meaningless without the
                    // ancestors that carry the window back.
                    crate::CheckpointShape::WindowDeltas { win_max } => {
                        self.ckpt_chain(key, win_max)
                    }
                }
            } else {
                // Above the last EXTENT, not the last tile: the extents carry
                // everything below `unit_end`, so that is where this capture's own
                // rows have to start or the two would describe the same positions.
                let (_, unit_end) = extents_below(&cut_ends, ck_at);
                let mut st = model.kv_capture_windows(ck_at);
                let toks = tokens
                    .get(unit_end..ck_at)
                    .map(<[u32]>::to_vec)
                    .unwrap_or_default();
                if ck_at > unit_end {
                    let tail_pl = c.tail.clone();
                    st.full = self.read_rows(
                        &cid(label),
                        &tail_pl,
                        unit_end,
                        ck_at,
                        &geom,
                        be,
                    );
                    if st.full.is_empty() || toks.len() != ck_at - unit_end {
                        // Cannot describe the part above the extents: fall back to the
                        // cut rather than claim more than we carry.
                        st = model.kv_capture_windows(unit_end);
                    }
                }
                // No recorded link: this capture owns its whole window, so it is an
                // ANCHOR (`from: 0`) and needs no predecessor.
                let d = crate::KvDelta {
                    boundary: st.boundary,
                    from: 0,
                    window: st.window,
                    tail: crate::state::tail_rows(&st.full, st.boundary, unit_end),
                    recurrent: st.recurrent,
                };
                // A fresh capture at the tip: a STEP, not a turn start. It is the
                // newest boundary, so the per-turn collapse keeps it either way.
                vec![(d, false)]
            };
            // The chain's stored tail tokens are the RESIDENT record's, on the tile
            // grid; the disk tail is derived from the cut below.
            let Some((link, _)) = chain.first().cloned() else {
                return Ok(false);
            };
            let boundary = link.boundary;
            // `unit_end` is unused here now that the manifest carries no tail of its
            // own; each Ckpt derives its own below.
            let (units, _) = extents_below(&cut_ends, boundary);
            if !recurrent_ok(model, &link.recurrent) {
                // A checkpoint whose recurrent half is missing restores a model with a
                // ZERO convolution history -- right shape, wrong numbers. Keep the
                // manifest that is already on disk and let this conversation
                // reprocess instead. (`kv_capture_*` fill in an empty blob when the
                // model cannot produce one for that position, which is how this got
                // committed silently once.)
                eprintln!(
                    "[imparo] kv spill: no recurrent state at {boundary}; not committing"
                );
                return Ok(false);
            }
            // KEEP the older boundaries. They are content addressed, so naming one again
            // costs nothing and only what is new is written -- with rewind and branch
            // surviving a restart.
            let mut kept: Vec<crate::store::Ckpt> = self
                .convs
                .get(label)
                .map(|c| c.disk_ckpts.clone())
                .unwrap_or_default();
            let mut blobs: Vec<(crate::UnitHash, Vec<u8>)> = Vec::new();
            let mut ck_bytes = 0usize;
            // Walk the chain newest first and stop as soon as the manifest covers a whole
            // window back from this boundary. An entry ALREADY on disk that reaches at
            // least as deep wins: it is written, and replacing an anchor with a delta at
            // the same boundary buys nothing and adds a dependency. That is what makes a
            // re-committed boundary cost zero bytes.
            // `None` for a snapshot model: there is no window to cover, so nothing to
            // stop for, and every entry the chain offered is a place a later prompt can
            // land.
            let floor = shape.chain_floor(boundary);
            let mut need = boundary;
            for (i, (d, is_turn)) in chain.iter().enumerate() {
                // The newest link is always named -- it IS the resume point, and it
                // carries the recurrent snapshot and the sub-unit tail. Its ancestors
                // are here to cover a window, so this stops once they do.
                //
                // Under `Snapshots` there is no window to cover and every entry is
                // complete on its own, so there is nothing to stop for: naming them all
                // is simply more places a later prompt can land. `cap_ckpts` bounds the
                // count.
                if i > 0 && floor.is_some_and(|f| need <= f) {
                    break;
                }
                let b = d.boundary;
                // THE EXTENT GRID, like every other tail this manifest carries. `toks`
                // is the resident record's, measured from the last sealed tile, and a
                // manifest that mixes the two writes tails its own reader refuses.
                let (_, ue) = extents_below(&cut_ends, b);
                let t = tokens.get(ue..b).map(<[u32]>::to_vec).unwrap_or_default();
                if t.len() != b - ue {
                    continue; // cannot describe the part above the extents: do not name it
                }
                if let Some(old) = kept.iter().find(|c| c.boundary == b as u64) {
                    if old.from <= d.from as u64 && dq.holds_checkpoint(st, &old.blob) {
                        need = need.min(old.from as usize);
                        continue;
                    }
                }
                // The delta on DISK carries only the rows the extents do not. A
                // recorded delta's tail is cut at the last TILE, because that is what
                // the resident chain needs; written through unchanged it overlaps the
                // extents, and the reader refuses it ("the extents reach 1408, the
                // checkpoint counts its tail from 1280"). Re-cut to the extent grid --
                // empty whenever the boundary IS a cut, which is the common case now.
                let on_disk = crate::KvDelta {
                    tail: crate::state::tail_rows(&d.tail, b, ue),
                    ..d.clone()
                };
                // AND IT MUST CARRY THE ROWS THAT TAIL CLAIMS. `tail_rows` returns an
                // empty vec when the recorded delta cannot reach back to the extent cut,
                // and an empty tail reads at restore as "the cut IS the boundary" -- so
                // naming it made the reader refuse the whole manifest ("the extents reach
                // 1664, the checkpoint counts its tail from 1792") and every keyless
                // restart went cold. Skip the boundary; a shallower one still serves.
                let rows = on_disk.tail.first().map_or(0, |ls| ls.positions);
                if rows != b - ue {
                    if crate::log_on() {
                        eprintln!(
                            "[imparo] kv ckpt {b} not named: {rows} rows for [{ue}, {b})"
                        );
                    }
                    continue;
                }
                let bytes = crate::state::delta_blob(&on_disk);
                let a = crate::store::blob_hash(&bytes);
                kept.retain(|c| c.boundary != b as u64);
                kept.push(crate::store::Ckpt {
                    boundary: b as u64,
                    from: d.from as u64,
                    blob: a,
                    tail: t,
                    turn: *is_turn,
                });
                need = need.min(d.from);
                if !dq.holds_checkpoint(st, &a) {
                    ck_bytes += bytes.len();
                    blobs.push((a, bytes));
                }
            }
            // A REWIND replaces the conversation from its branch point, so the
            // boundaries above it are not part of it any more. Drop them: they are
            // already refused at read time (their sub-unit tokens are not this stream's),
            // and `cap_ckpts` keeps by POSITION -- so a dead boundary sits high and
            // crowds out a live rewind target.
            kept.retain(|c| {
                let b = c.boundary as usize;
                let (_, ue) = extents_below(&cut_ends, b);
                b <= tokens.len()
                    && c.tail.len() == b - ue
                    && tokens.get(ue..b).is_some_and(|t| t == c.tail)
            });
            kept.sort_by_key(|c| c.boundary);
            cap_ckpts(&mut kept, self.disk_ckpt_cap, shape);
            let keyless = self.convs.get(label).is_some_and(|c| c.keyless);
            let m = Manifest {
                boundary: boundary as u64,
                cuts: cuts[..units].to_vec(),
                ckpts: kept.clone(),
                keyless,
            };
            let blobs_written = blobs.len();
            // SUPERSEDE. A keyless conversation is renamed by its own growth and again by
            // every restart, so the manifest it restored from can never be asked for
            // again -- and it holds its checkpoint blobs against the sweep until the
            // store hits its cap. Drop it once THIS commit names every boundary it did,
            // and only then: until that holds it is still the only way back to those
            // turns. Queued behind the commit, so a crash in between leaves the old one
            // standing rather than neither.
            let superseded: Vec<_> = self
                .convs
                .get(label)
                .and_then(|c| c.adopted.clone())
                .into_iter()
                .collect();
            dq.commit(label, m, blobs, at[..units].to_vec(), superseded.clone())?;
            if !superseded.is_empty() {
                if let Some(c) = self.convs.get_mut(label) {
                    c.adopted = None;
                }
                if crate::log_on() {
                    eprintln!(
                        "[imparo] kv disk: superseded the keyless manifest this \
conversation restored from"
                    );
                }
            }
            if let Some(c) = self.convs.get_mut(label) {
                c.committed_at = boundary;
                c.disk_ckpts = kept;
            }
            if crate::log_on() {
                eprintln!(
                    "[imparo] kv write-through: {} extent(s) + {:.1} MiB checkpoint [{}, {boundary}) in {} blob(s), {:.1} ms",
                    extents_written,
                    ck_bytes as f64 / (1 << 20) as f64,
                    link.from,
                    blobs_written,
                    t_write.elapsed().as_secs_f64() * 1e3
                );
            }
        }
        if let Some(c) = self.convs.get_mut(label) {
            c.spilled_units = sealed;
        }
        Ok(true)
    }

    /// Bytes one unit occupies across the full-attention layers (both sides).
    fn unit_bytes(&self) -> usize {
        unit_bytes(&self.strides)
    }

    /// Drop device residency for conversations outside the keep-set (the
    /// incoming one, always, plus the most recent others whose units fit the
    /// resident-bytes budget) and hand the freed block pages back to the OS.
    /// Everything dropped was spilled at its own switch-out, so the disk
    /// restore path serves any return visit.
    fn release_idle<T: PoolTenant + ?Sized>(
        &mut self,
        _model: &T,
        incoming: &str,
        incoming_hashes: &[UnitHash],
        store: Option<&Store>,
        disk: Option<&crate::disk::DiskQueue>,
    ) -> Result<(), String> {
        let mut keep: Vec<String> = vec![incoming.to_string()];
        let ub = self.unit_bytes().max(1);
        let mut budget = self.resident_budget;
        for l in self.recent.iter().rev() {
            if keep.contains(l) {
                continue;
            }
            let cost = self.convs.get(l).map_or(0, |c| c.sealed_units) * ub;
            if cost > budget {
                break; // newest-first: once one no longer fits, stop retaining
            }
            budget -= cost;
            keep.push(l.clone());
        }
        let drop: Vec<String> = self
            .convs
            .keys()
            .filter(|l| !keep.contains(l))
            .cloned()
            .collect();
        if drop.is_empty() {
            return Ok(());
        }

        if self.addressing == PoolAddressing::ExplicitHostTransfers {
            // Protect every unit the incoming request can adopt, including a shared
            // hash not yet referenced by its conversation label, plus all Device
            // keep-set references. Only units used exclusively by idle conversations
            // are eligible for D2H.
            let mut protected: BTreeSet<UnitId> = self
                .resident
                .prepare_probe(incoming_hashes)
                .map_err(|error| format!("kv pool: idle probe failed: {error:?}"))?
                .indexed
                .into_iter()
                .collect();
            for label in &keep {
                if let Some(units) = self.resident.pool.conversation(&cid(label)) {
                    protected.extend(units.iter().copied());
                }
            }
            let mut candidates = BTreeSet::new();
            for label in &drop {
                if let Some(units) = self.resident.pool.conversation(&cid(label)) {
                    candidates.extend(
                        units
                            .iter()
                            .copied()
                            .filter(|unit| !protected.contains(unit)),
                    );
                }
            }
            let mover = self.active_mover()?;
            let needed = u64::try_from(candidates.len())
                .ok()
                .and_then(|count| count.checked_mul(self.unit_bytes() as u64))
                .ok_or_else(|| "kv pool: Host capacity request overflow".to_string())?;
            self.make_host_room_with(mover, needed, incoming, store, disk)?;
            self.demote_units_with(mover, candidates)?;
            // Unsealed tails are mutable and have no content identity: release them
            // only after the sealed transaction commits, and reprocess on return.
            for label in &drop {
                let tail = self
                    .convs
                    .get_mut(label)
                    .map_or_else(Vec::new, |conv| std::mem::take(&mut conv.tail));
                self.release_unsealed(tail);
            }
            if self
                .active
                .as_deref()
                .is_some_and(|label| drop.iter().any(|d| d == label))
            {
                self.active = None;
            }
        } else {
            // Unified/shared behavior is intentionally unchanged: no Host API call.
            for label in &drop {
                let tail = self
                    .convs
                    .remove(label)
                    .map_or_else(Vec::new, |conv| conv.tail);
                self.release_unsealed(tail);
                let evicted = self.resident.forget(&cid(label));
                self.release_evicted(evicted)?;
                self.recent.retain(|recent| recent != label);
                if self.active.as_deref() == Some(label.as_str()) {
                    self.active = None;
                }
            }
            self.drain_lifecycle();
        }
        // This used to advise each returned Device placement directly, beside the
        // queue. `evict_unreferenced` releases through `release_placement`, which
        // ALREADY queues a Free for the same placement, so anything it reclaims here
        // was advised twice -- a second path to the backend that `drain_lifecycle`
        // says does not exist. Measured on the Shared path the count was 1 either
        // way: `forget` above has already evicted and drained those units, so the
        // call below finds nothing. The discrete path does not call `forget`, and
        // needs a backend to reach, so the double there is by inspection only.
        let freed = self.resident.evict_unreferenced();
        self.release_evicted(freed)?;
        self.drain_lifecycle();
        // Checkpoints reachable from kept conversations stay; the rest are on
        // disk (their conv's spill) or re-capturable, so their RAM goes too.
        let mut reach: BTreeSet<CkptKey> = BTreeSet::new();
        for l in &keep {
            let Some(c) = self.convs.get(l.as_str()) else {
                continue;
            };
            // Roots: every checkpoint sitting on one of this conversation's sealed
            // units, plus whatever its uncaptured records chain from. A checkpoint
            // is named by (unit hash, position), so a conversation reaches all the
            // ones whose unit it owns -- there can be more than one per unit.
            let mut roots: Vec<CkptKey> = self
                .window_ckpts
                .keys()
                .copied()
                .filter(|(t, _)| c.hashes.contains(t))
                .collect();
            roots.extend(c.pending.iter().filter_map(|pk| pk.prev));
            for r in roots {
                let mut cur = Some(r);
                while let Some(t) = cur {
                    if !reach.insert(t) {
                        break;
                    }
                    cur = self.window_ckpts.get(&t).and_then(|e| e.prev);
                }
            }
        }
        self.ckpt_order.retain(|h| reach.contains(h));
        self.window_ckpts.retain(|h, _| reach.contains(h));
        Ok(())
    }

    /// Canonical unit blob (same format as kv_io's unit_blobs) read from the
    /// unit's physical blocks.
    /// Full-attention rows for `[from, to)`, read through `conv`'s OWN block
    /// table -- the positions between the last whole unit and a turn boundary.
    ///
    /// Physical, not logical: `read_kv_bytes` addresses a layer's buffer directly,
    /// so every position has to be routed through the table first. Reads run per
    /// contiguous stretch inside one 64-cell block, so this is a few dozen memcpys,
    /// not one per position.
    fn read_rows(
        &self,
        conv: &ConversationId,
        tail: &[UnitPlacement],
        from: usize,
        to: usize,
        geom: &[crate::LayerStateGeom],
        be: &dyn imparo_backend::Backend,
    ) -> Vec<crate::KvLayerState> {
        let page = page_cells();

        let mut out = Vec::new();
        if to <= from {
            return out;
        }
        for g in geom {
            if !matches!(g.kind, crate::StateKind::Full) {
                continue;
            }
            let Ok(table) = self.resident.table_for(conv, g.layer, tail) else {
                return Vec::new();
            };
            if table.len() * page < to {
                return Vec::new(); // placement missing: no tail rows, no checkpoint
            }
            let n = to - from;
            let mut k = vec![0u8; n * g.k_stride];
            let mut v = vec![0u8; n * g.v_stride];
            let mut p = from;
            while p < to {
                let run = (page - p % page).min(to - p);
                let blk = table[p / page] as usize;
                let off = blk * page + p % page;
                let i = p - from;
                be.read_kv_bytes(
                    g.layer,
                    false,
                    (off * g.k_stride) as u64,
                    &mut k[i * g.k_stride..(i + run) * g.k_stride],
                );
                be.read_kv_bytes(
                    g.layer,
                    true,
                    (off * g.v_stride) as u64,
                    &mut v[i * g.v_stride..(i + run) * g.v_stride],
                );
                p += run;
            }
            out.push(crate::KvLayerState {
                layer: g.layer,
                base_pos: from,
                positions: n,
                k,
                v,
            });
        }
        out
    }

    /// Writes checkpoint tail rows into `tail`, whose first unit starts at `base`.
    ///
    /// Into the borrower's OWN blocks, never the donor's: the borrower continues
    /// writing at the boundary, and the positions right below it share a 64-cell
    /// block with the ones right above. Aliasing the donor's block there would
    /// have one conversation's prefill overwrite another's cache.
    fn write_rows(
        rows: &[crate::KvLayerState],
        base: usize,
        tail: &[UnitPlacement],
        geom: &[crate::LayerStateGeom],
        be: &dyn imparo_backend::Backend,
    ) {
        let page = page_cells();
        let grid = grid_tokens();

        for ls in rows {
            let Some(g) = geom.iter().find(|g| g.layer == ls.layer) else {
                continue;
            };
            let mut i = 0;
            while i < ls.positions {
                let p = ls.base_pos + i;
                let run = (page - p % page).min(ls.positions - i);
                // One placement entry per 64 positions, so the entry index IS the
                // block index -- there is no second division to get wrong.
                let unit = (p - base) / grid;
                let Some(&blk) = tail.get(unit).and_then(|pl| pl.get(&ls.layer)) else {
                    return;
                };
                let blk = blk as usize;
                let off = blk * page + p % page;
                be.write_kv_bytes(
                    ls.layer,
                    false,
                    (off * g.k_stride) as u64,
                    &ls.k[i * g.k_stride..(i + run) * g.k_stride],
                );
                be.write_kv_bytes(
                    ls.layer,
                    true,
                    (off * g.v_stride) as u64,
                    &ls.v[i * g.v_stride..(i + run) * g.v_stride],
                );
                i += run;
            }
        }
    }

    /// One stored extent's rows, read off the device and framed as a unit blob.
    ///
    /// `[lo, hi)` in absolute positions, both on the 64 grid. Extents end where
    /// requests ended, so this reads a block RANGE rather than a fixed tile -- and
    /// `tail` is needed because a request commonly ends above the last sealed unit,
    /// in blocks the conversation holds but has not sealed.
    fn read_unit_canonical(
        &self,
        conv: &ConversationId,
        lo: usize,
        hi: usize,
        tail: &[UnitPlacement],
        geom: &[crate::LayerStateGeom],
        be: &dyn imparo_backend::Backend,
    ) -> Option<Vec<u8>> {
        let page = page_cells();

        let full: Vec<&crate::LayerStateGeom> = geom
            .iter()
            .filter(|g| matches!(g.kind, crate::StateKind::Full))
            .collect();
        // Write-through of ONE unit's full-attention rows; nothing here is a
        // conversation checkpoint, so it carries no recurrent state.
        let (b0, b1) = (lo / page, hi / page);
        let rows = hi.checked_sub(lo)?;
        let mut state = KvState {
            boundary: hi,
            full: Vec::new(),
            window: Vec::new(),
            recurrent: Vec::new(),
        };
        for g in full {
            let table = self.resident.table_for(conv, g.layer, tail).ok()?;
            if table.len() < b1 {
                return None; // placement missing (evicted mid-flight): skip write-through
            }
            let blocks = &table[b0..b1];
            let mut k = vec![0u8; rows * g.k_stride];
            let mut v = vec![0u8; rows * g.v_stride];
            for (bi, &blk) in blocks.iter().enumerate() {
                be.read_kv_bytes(
                    g.layer,
                    false,
                    (blk as usize * page * g.k_stride) as u64,
                    &mut k[bi * page * g.k_stride..(bi + 1) * page * g.k_stride],
                );
                be.read_kv_bytes(
                    g.layer,
                    true,
                    (blk as usize * page * g.v_stride) as u64,
                    &mut v[bi * page * g.v_stride..(bi + 1) * page * g.v_stride],
                );
            }
            state.full.push(crate::KvLayerState {
                layer: g.layer,
                base_pos: lo,
                positions: rows,
                k,
                v,
            });
        }
        crate::state::unit_blobs_at(&state, &[hi])
            .into_iter()
            .next()
    }

    /// Free blocks on the first full-attention layer (all layers allocate in
    /// lockstep, so one layer's count describes the pool). For measurement logs.
    #[must_use]
    pub fn free_blocks(&self) -> u32 {
        self.layers
            .first()
            .map_or(0, |&l| self.resident.free_blocks(l))
    }

    /// Record a branch point and capture it NOW (the server does this mid-prefill
    /// only when the boundary would slide out of ring reach before a switch-out
    /// could reach it -- which is always, for a recurrent model).
    pub fn note_ckpt<T: PoolTenant + ?Sized>(
        &mut self,
        model: &T,
        label: &str,
        tip: UnitHash,
        prev: Option<CkptKey>,
        boundary: usize,
        tokens: Vec<u32>,
    ) {
        let chained = self.chain_from(prev);
        // Same rule as `note_branch`: a sealed boundary is an extent cut.
        if let Some(c) = self.convs.get_mut(label) {
            record_cut(&mut c.cuts, boundary);
        }
        self.capture_ckpt(
            model,
            label,
            PendingCkpt {
                tip,
                prev: chained,
                boundary,
                from: chained.map_or(0, |(_, b)| b),
                tokens,
                // A user message begins here: this is where a turn starts.
                turn: true,
            },
        );
    }

    /// The predecessor a new checkpoint chains from, or `None` where checkpoints do
    /// not chain.
    ///
    /// A windowed delta holds min(gap, window) rows and is meaningless without the
    /// ancestors a restore walks to rebuild the window. A snapshot is the whole state
    /// at its boundary, so a predecessor buys no bytes and only adds a dependency that
    /// retention can drop -- which it did: a manifest shipped `ckpt=1408:1280:...` on a
    /// model with no window, a delta whose link survived by luck of what else was kept.
    ///
    /// Every site that records a checkpoint goes through here. There are three, and an
    /// earlier attempt guarded only one.
    fn chain_from(&self, prev: Option<CkptKey>) -> Option<CkptKey> {
        if self.shape.chains() { prev } else { None }
    }

    /// Record a branch point WITHOUT capturing: the rings still cover it, so the
    /// bytes are taken only if the conversation is actually switched out.
    pub fn note_branch(
        &mut self,
        label: &str,
        tip: UnitHash,
        prev: Option<CkptKey>,
        boundary: usize,
        tokens: Vec<u32>,
    ) {
        let chained = self.chain_from(prev);
        if let Some(c) = self.convs.get_mut(label) {
            // CUT AND CHECKPOINT LAND TOGETHER (design: "Cut and checkpoint land
            // together: that is the whole point of cutting here"). A branch point is
            // where a user message starts, which is neither the prompt end nor the
            // request end, so nothing else cuts there -- and a sealed checkpoint whose
            // boundary is not a cut has to carry the rows between the two. It cannot:
            // `capture_ckpt` fills a row tail only when the boundary is OFF the tile
            // grid, and a branch point is on it. Measured: a branch at 1792 with cuts
            // [1280, 1472, 1664, 1856] claimed a 128-token tail and carried 0 rows.
            //
            // Cutting here instead makes the extents reach every boundary the manifest
            // names, so the tail is empty on both sides and the rows are stored once,
            // in the extent, rather than copied into the checkpoint.
            record_cut(&mut c.cuts, boundary);
            if !c.pending.iter().any(|pk| pk.boundary == boundary) {
                c.pending.push(PendingCkpt {
                    tip,
                    prev: chained,
                    boundary,
                    from: chained.map_or(0, |(_, b)| b),
                    tokens,
                    // A user message begins here: this is where a turn starts.
                    turn: true,
                });
            }
        }
    }

    /// The conversation this prompt belongs to, if one is resident.
    ///
    /// For a client that sends no conversation id. Branching from an old turn is ONE
    /// operation with two meanings, and the client picks which:
    ///
    /// ```text
    /// REWIND  branch from an old turn and REPLACE this conversation   the default
    /// FORK    branch from an old turn and start a NEW one             x-new-conversation
    /// ```
    ///
    /// So a prompt that agrees with a resident conversation and then diverges is that
    /// conversation being rewound -- not a stranger. The longest agreement wins and no
    /// threshold is applied: requiring an exact prefix, or agreement past some boundary,
    /// would call a rewind a new conversation and mint a name for it. The caller does not
    /// consult this at all when the client asked for a fork.
    ///
    /// This is also what makes a thinking model ordinary rather than special. Its
    /// recorded stream and its own next prompt differ inside the last turn by
    /// construction; that difference is a rewind to the previous turn boundary, and
    /// nothing here has to know why the two disagree.
    pub fn label_for_prefix(&self, ids: &[u32]) -> Option<String> {
        self.convs
            .iter()
            // This lookup is only for a client that supplied no identity.  A named
            // resident conversation is a fork source, never an identity that a
            // keyless request may silently take over.
            .filter(|(_, c)| c.keyless && !c.tokens.is_empty())
            .map(|(l, c)| {
                (
                    l,
                    c.tokens.iter().zip(ids).take_while(|(a, b)| a == b).count(),
                )
            })
            // A prompt that shares only a preamble with a resident conversation is not
            // that conversation: below one whole unit there is nothing to stand on and
            // nothing to adopt either, so the content name is the honest answer.
            .filter(|(_, n)| *n >= grid_tokens())
            .max_by_key(|(_, n)| *n)
            .map(|(l, _)| l.clone())
    }

    /// IMPARO_KV_DIGEST=1: an FNV-1a over this conversation's full-attention KV, read
    /// THROUGH its own block table, per layer, for the first `positions` positions.
    ///
    /// The instrument for "are two conversations' caches actually the same bytes".
    /// Reuse is supposed to make a resumed conversation's cache identical to a cold
    /// one's; when the answers diverge, this says whether the bytes diverged or the
    /// arithmetic on top of them did.
    pub fn digest<T: PoolTenant + ?Sized>(
        &self,
        model: &T,
        label: &str,
        positions: usize,
    ) {
        if !std::env::var("IMPARO_KV_DIGEST").is_ok_and(|v| v == "1") {
            return;
        }
        let Some(be) = model.backend() else {
            return;
        };
        let geom = model.kv_state_geometry();
        let tail = self
            .convs
            .get(label)
            .map_or_else(Vec::new, |c| c.tail.clone());
        let fnv = |k: &[u8], v: &[u8]| {
            k.iter().chain(v).fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
                (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
            })
        };
        // Pooled layers, through this conversation's block table.
        for ls in self.read_rows(&cid(label), &tail, 0, positions, &geom, be) {
            eprintln!(
                "[imparo] kv digest {label} FULL layer={} pos=0..{positions} h={:016x}",
                ls.layer,
                fnv(&ls.k, &ls.v)
            );
        }
        // Windowed layers, through the ring. These are the ones a checkpoint carries,
        // so this is where a restore that is not byte-exact shows up.
        for g in &geom {
            let crate::StateKind::Window { window, ring } = g.kind else {
                continue;
            };
            let lo = positions.saturating_sub(window);
            let mask = ring - 1;
            let (mut k, mut v) = (Vec::new(), Vec::new());
            for p in lo..positions {
                let slot = p & mask;
                let mut kb = vec![0_u8; g.k_stride];
                let mut vb = vec![0_u8; g.v_stride];
                be.read_kv_bytes(g.layer, false, (slot * g.k_stride) as u64, &mut kb);
                be.read_kv_bytes(g.layer, true, (slot * g.v_stride) as u64, &mut vb);
                k.extend_from_slice(&kb);
                v.extend_from_slice(&vb);
            }
            eprintln!(
                "[imparo] kv digest {label} WNDW layer={} pos={lo}..{positions} h={:016x}",
                g.layer,
                fnv(&k, &v)
            );
        }
    }

    /// Deletion support: drop residency records for erased conversations.
    pub fn forget(&mut self, labels: &[String]) -> Result<(), String> {
        for label in labels {
            let tail = self
                .convs
                .remove(label)
                .map_or_else(Vec::new, |conv| conv.tail);
            self.release_unsealed(tail);
            let evicted = self.resident.forget(&cid(label));
            self.release_evicted(evicted)?;
            self.recent.retain(|recent| recent != label);
            if self.active.as_deref() == Some(label.as_str()) {
                self.active = None;
            }
        }
        self.drain_lifecycle();
        Ok(())
    }
}

fn transfer_spans_for_placement(
    strides: &BTreeMap<u32, (usize, usize)>,
    handle: KvHostHandle,
    placement: &UnitPlacement,
) -> Result<Vec<KvTransferSpan>, String> {
    if placement.len() != strides.len() || placement.keys().ne(strides.keys()) {
        return Err("kv pool: placement/stride layer sets differ".into());
    }
    let mut spans = Vec::with_capacity(strides.len() * 2);
    let mut host_offset = 0_u64;
    for (&layer, &(k_stride, v_stride)) in strides {
        let &block = placement
            .get(&layer)
            .ok_or_else(|| format!("kv pool: placement missing layer {layer}"))?;
        // Canonical Host layout for one resident grid unit: each layer's K page,
        // then its V page. A placement has exactly one page per layer.
        for (is_v, stride) in [(false, k_stride), (true, v_stride)] {
            let len = (page_cells() as u64)
                .checked_mul(stride as u64)
                .ok_or_else(|| "kv pool: transfer span length overflow".to_string())?;
            let device_offset = u64::from(block)
                .checked_mul(len)
                .ok_or_else(|| "kv pool: transfer offset overflow".to_string())?;
            spans.push(KvTransferSpan {
                host_handle: handle.0,
                layer,
                is_v: u32::from(is_v),
                device_offset,
                host_offset,
                len,
            });
            host_offset = host_offset
                .checked_add(len)
                .ok_or_else(|| "kv pool: Host offset overflow".to_string())?;
        }
    }
    Ok(spans)
}

fn transfer_spans_for_promotions(
    strides: &BTreeMap<u32, (usize, usize)>,
    plans: &[PromotionPlan],
) -> Result<Vec<KvTransferSpan>, String> {
    let mut spans = Vec::new();
    for plan in plans {
        let unit =
            transfer_spans_for_placement(strides, plan.host_handle(), plan.device())?;
        if unit.last().map_or(0, |span| span.host_offset + span.len)
            != plan.host_bytes()
        {
            return Err(format!(
                "kv pool: Host unit {:?} has {} bytes but geometry requires {}",
                plan.unit(),
                plan.host_bytes(),
                unit.last().map_or(0, |span| span.host_offset + span.len)
            ));
        }
        spans.extend(unit);
    }
    Ok(spans)
}

fn transfer_spans_for_demotions(
    strides: &BTreeMap<u32, (usize, usize)>,
    plans: &[DemotionPlan],
) -> Result<Vec<KvTransferSpan>, String> {
    let mut spans = Vec::new();
    for plan in plans {
        let unit =
            transfer_spans_for_placement(strides, plan.host().handle(), plan.device())?;
        if unit.last().map_or(0, |span| span.host_offset + span.len)
            != plan.host().bytes()
        {
            return Err(format!(
                "kv pool: Host unit {:?} has {} bytes but geometry requires {}",
                plan.unit(),
                plan.host().bytes(),
                unit.last().map_or(0, |span| span.host_offset + span.len)
            ));
        }
        spans.extend(unit);
    }
    Ok(spans)
}

/// Bytes one RESIDENT unit occupies across the full-attention layers, both sides.
///
/// The unit is a grid step, so this counts `grid_tokens()` rows -- not the 256-token
/// extent. Public because the server sizes the Host tier in these and the pool charges
/// Host capacity in these, and the two were separately written: the server multiplied
/// the same strides by `UNIT_TOKENS`, making its unit four times the one the pool then
/// allocated. One function, so a capacity and the thing it is a capacity for cannot be
/// measured in different units.
#[must_use]
pub fn unit_bytes(strides: &BTreeMap<u32, (usize, usize)>) -> usize {
    strides
        .values()
        .map(|&(k, v)| grid_tokens() * (k + v))
        .sum()
}

/// Visit K then V for each block, in layer/placement order.
///
/// Keeping the address expansion pure is what makes the backend-notification
/// contract testable without installing a process-global backend -- but only while
/// `advise_placement` actually calls it. It was `#[cfg(test)]` and open-coded a
/// second time in `advise_placement`, so the two tests below gated a copy the server
/// never ran, and the copies had already drifted over an unknown layer (skip here,
/// panic there). One implementation, used by both.
fn visit_placement_spans(
    strides: &BTreeMap<u32, (usize, usize)>,
    pl: &UnitPlacement,
    mut visit: impl FnMut(u32, bool, u64, u64),
) {
    for (&layer, &block) in pl {
        let &(ks, vs) = strides.get(&layer).unwrap_or_else(|| {
            panic!("placement has no stride geometry for layer {layer}")
        });
        let page = page_cells() as u64;
        let (ko, vo) = (
            u64::from(block) * page * ks as u64,
            u64::from(block) * page * vs as u64,
        );
        visit(layer, false, ko, page * ks as u64);
        visit(layer, true, vo, page * vs as u64);
    }
}

/// Record an extent cut at `at`, keeping the list ascending and unique.
///
/// ASCENDING BY CONSTRUCTION, because the order is not cosmetic: `extents_below` is a
/// `partition_point` and `Cut::for_stream` builds extents that have to abut, so one
/// out-of-order entry makes the manifest name a cut whose extents stop short. Two of
/// the three cuts can arrive BELOW one already recorded -- the prompt end, when a client
/// re-renders its last turn shorter than the stream that produced it, and a branch point,
/// which sits where a user message starts rather than where the request ended.
fn record_cut(cuts: &mut Vec<usize>, at: usize) {
    if at == 0 {
        return;
    }
    if let Err(i) = cuts.binary_search(&at) {
        cuts.insert(i, at);
    }
}

/// Choose which boundaries a manifest names: the newest `cap` turns, one state each,
/// plus every ancestor their chains still need.
///
/// The ancestor walk is why this cannot be a plain filter. A link covers
/// `[from, boundary)`, so dropping an entry can leave a NEWER boundary unable to
/// reach back a whole window -- it would still be named, and the restore would have
/// to discover the hole.
fn cap_ckpts(
    kept: &mut Vec<crate::store::Ckpt>,
    cap: usize,
    shape: crate::CheckpointShape,
) {
    // ONE STATE PER TURN, AT THE TURN'S END, for every state kind. A turn's end is the
    // next turn's branch point -- the same position named from the other side -- so the
    // branch points ARE the ends of every turn that has finished:
    //
    //   turn 0 ends 100 | turn A: steps 300, 500, ends 700 | turn B: step 900, in flight
    //   kept:      100                                 700                          900
    //
    // A step inside a turn is read by the next rewind and then superseded by the one
    // after it, which always sits further forward, so it is kept resident and never
    // written (design: "Only branch points are sealed"). That is what makes the cap a
    // budget of TURNS: sealing steps too spent it on one chatty turn, where ten tool
    // calls took ten of twenty slots and evicted the older turns it exists to reach.
    //
    // Keeping each turn's last STEP instead -- 500 for turn A -- is one step short of
    // where turn B attaches, so a rewind to turn B re-prefilled turn A's whole reply.
    let mut roots: Vec<u64> = kept
        .iter()
        .rev()
        .filter(|c| c.turn)
        .map(|c| c.boundary)
        .take(cap)
        .collect();
    // AND THE TIP: the END of the turn still in flight, which has no branch point above
    // it yet. It is this commit's own boundary, and it is what a mid-turn eviction has
    // to land on (design: "Eviction mid-turn must capture the tip first" -- sealing the
    // turn's start alone discards everything the turn has done).
    //
    // This does not grow with tool calls: each step's tip replaces the one before it,
    // and when the turn ends the next turn's branch point takes its place. One per turn
    // either way, so the manifest names at most `cap` finished turns plus the live one.
    //
    // A conversation's first turn starts at 0 and records no branch point at all, so
    // this is also the only thing a one-turn conversation seals.
    if let Some(c) = kept.last() {
        if !roots.contains(&c.boundary) {
            roots.push(c.boundary);
        }
    }
    let mut keep: std::collections::BTreeSet<u64> = roots.iter().copied().collect();
    // ANCESTORS ARE A WINDOW'S BUSINESS. A snapshot has none, so the roots above are
    // already the whole answer and this loop does not run at all.
    for r in roots.iter().filter(|_| shape.chains()) {
        let floor = shape.chain_floor(*r as usize).unwrap_or(0) as u64;
        let mut need = *r;
        for c in kept.iter().rev() {
            if need <= floor {
                break;
            }
            if c.boundary <= *r && c.boundary >= need {
                need = need.min(c.from);
                keep.insert(c.boundary);
            }
        }
    }
    kept.retain(|c| keep.contains(&c.boundary));
}

/// Whether a captured state actually carries what this model's recurrent layers
/// need. False means the capture could not describe that position.
fn recurrent_ok<T: PoolTenant + ?Sized>(model: &T, recurrent: &[u8]) -> bool {
    model.recurrent_elems() == 0 || !recurrent.is_empty()
}

#[allow(clippy::items_after_test_module)]
#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[test]
    fn every_block_expands_to_exactly_one_k_and_one_v_notification() {
        let mut strides = BTreeMap::new();
        strides.insert(7, (3, 5));
        let mut pl = UnitPlacement::new();
        pl.insert(7, 2);
        let mut spans = Vec::new();
        visit_placement_spans(&strides, &pl, |l, v, o, n| spans.push((l, v, o, n)));

        let page = page_cells() as u64;
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], (7, false, 2 * page * 3, page * 3));
        assert_eq!(spans[1], (7, true, 2 * page * 5, page * 5));
    }

    #[test]
    #[should_panic(expected = "placement has no stride geometry for layer 9")]
    fn unknown_layer_is_rejected_before_any_notification() {
        let mut pl = UnitPlacement::new();
        pl.insert(9, 0);
        visit_placement_spans(&BTreeMap::new(), &pl, |_, _, _, _| {
            panic!("notification must not run for invalid geometry");
        });
    }

    #[derive(Default)]
    struct FakeMoverState {
        next: u64,
        allocations: BTreeMap<u64, Vec<u8>>,
        frees: Vec<KvHostHandle>,
        demote_batches: Vec<Vec<KvTransferSpan>>,
        promote_batches: Vec<Vec<KvTransferSpan>>,
        fail_demote: bool,
        fail_promote: bool,
        fail_read: bool,
    }

    #[derive(Default)]
    struct FakeMover(std::cell::RefCell<FakeMoverState>);

    impl FakeMover {
        fn set_fail_demote(&self, fail: bool) {
            self.0.borrow_mut().fail_demote = fail;
        }

        fn set_fail_promote(&self, fail: bool) {
            self.0.borrow_mut().fail_promote = fail;
        }

        fn set_fail_read(&self, fail: bool) {
            self.0.borrow_mut().fail_read = fail;
        }
    }

    impl KvHostMover for FakeMover {
        fn host_alloc(&self, bytes: u64) -> Result<KvHostHandle, i32> {
            let mut state = self.0.borrow_mut();
            state.next += 1;
            let handle = KvHostHandle(state.next);
            state.allocations.insert(handle.0, vec![0; bytes as usize]);
            Ok(handle)
        }

        fn host_free(&self, handle: KvHostHandle) -> Result<(), i32> {
            let mut state = self.0.borrow_mut();
            if state.allocations.remove(&handle.0).is_none() {
                return Err(-2);
            }
            state.frees.push(handle);
            Ok(())
        }

        fn demote(&self, spans: &[KvTransferSpan]) -> Result<(), i32> {
            let mut state = self.0.borrow_mut();
            state.demote_batches.push(spans.to_vec());
            if state.fail_demote {
                return Err(-11);
            }
            for span in spans {
                let Some(host) = state.allocations.get_mut(&span.host_handle) else {
                    return Err(-2);
                };
                let start = span.host_offset as usize;
                let end = start + span.len as usize;
                for (index, byte) in host[start..end].iter_mut().enumerate() {
                    *byte = span.device_offset.wrapping_add(index as u64) as u8;
                }
            }
            Ok(())
        }

        fn promote(&self, spans: &[KvTransferSpan]) -> Result<(), i32> {
            let mut state = self.0.borrow_mut();
            state.promote_batches.push(spans.to_vec());
            if state.fail_promote { Err(-12) } else { Ok(()) }
        }

        fn allocated_bytes(&self) -> u64 {
            self.0
                .borrow()
                .allocations
                .values()
                .map(|bytes| bytes.len() as u64)
                .sum()
        }

        fn host_read(
            &self,
            handle: KvHostHandle,
            off: u64,
            dst: &mut [u8],
        ) -> Result<(), i32> {
            let state = self.0.borrow();
            if state.fail_read {
                return Err(-13);
            }
            let Some(host) = state.allocations.get(&handle.0) else {
                return Err(-2);
            };
            let start = off as usize;
            let Some(src) = host.get(start..start + dst.len()) else {
                return Err(-3);
            };
            dst.copy_from_slice(src);
            Ok(())
        }
    }

    fn mode(addressing: PoolAddressing, capacity_units: u32) -> PoolMode {
        let mut strides = BTreeMap::new();
        strides.insert(3, (3, 5));
        PoolMode::new(
            addressing,
            (capacity_units as u64 * grid_tokens() as u64 * 8).into(),
            vec![3],
            capacity_units * UNIT_BLOCKS as u32,
            0,
            strides,
            crate::ConfigRoot::new(b"model", (1, 1), b"geometry"),
        )
    }

    fn seed(mode: &mut PoolMode, label: &str, tag: u8) -> (UnitId, UnitHash) {
        let placement = mode.resident.alloc_unit().expect("seed placement");
        mode.drain_lifecycle();
        let hash = UnitHash([tag; 16]);
        let outcome = mode.resident.seal_unit(&cid(label), hash, &placement);
        assert!(outcome.host_release.is_none());
        (outcome.unit, hash)
    }

    #[test]
    fn transfer_layout_is_layer_k_then_v_and_geometry_checked() {
        let mut strides = BTreeMap::new();
        strides.insert(7, (3, 5));
        let mut placement = UnitPlacement::new();
        placement.insert(7, 2);
        let spans = transfer_spans_for_placement(&strides, KvHostHandle(9), &placement)
            .unwrap();
        assert_eq!(spans.len(), 2 * UNIT_BLOCKS);
        assert!(spans[..UNIT_BLOCKS].iter().all(|span| span.is_v == 0));
        assert!(spans[UNIT_BLOCKS..].iter().all(|span| span.is_v == 1));
        for pair in spans.windows(2) {
            assert_eq!(pair[0].host_offset + pair[0].len, pair[1].host_offset);
        }
        placement.insert(8, 0);
        assert!(
            transfer_spans_for_placement(&strides, KvHostHandle(9), &placement)
                .is_err()
        );
    }

    #[test]
    fn shared_addressing_never_calls_host_mover() {
        let mut mode = mode(PoolAddressing::Shared, 1);
        let (_, hash) = seed(&mut mode, "owner", 1);
        let mover = FakeMover::default();
        let hit = mode
            .probe_with(&mover, &cid("reader"), &[hash])
            .expect("shared probe");
        assert_eq!(hit.indexed_units, 1);
        let state = mover.0.borrow();
        assert!(state.allocations.is_empty());
        assert!(state.demote_batches.is_empty());
        assert!(state.promote_batches.is_empty());
        assert!(state.frees.is_empty());
    }

    #[test]
    fn device_to_host_is_one_conversation_batch() {
        let mut mode = mode(PoolAddressing::ExplicitHostTransfers, 2);
        let (first, _) = seed(&mut mode, "idle", 1);
        let (second, _) = seed(&mut mode, "idle", 2);
        let mover = FakeMover::default();
        mode.demote_units_with(&mover, [first, second]).unwrap();
        let state = mover.0.borrow();
        assert_eq!(state.demote_batches.len(), 1);
        assert_eq!(state.demote_batches[0].len(), 2 * 2 * UNIT_BLOCKS);
        assert!(matches!(
            mode.resident.residency(first),
            Some(UnitResidency::Host(_))
        ));
        assert!(matches!(
            mode.resident.residency(second),
            Some(UnitResidency::Host(_))
        ));
    }

    #[test]
    fn d2h_failure_frees_new_handles_and_keeps_device_authoritative() {
        let mut mode = mode(PoolAddressing::ExplicitHostTransfers, 1);
        let (unit, _) = seed(&mut mode, "idle", 1);
        let mover = FakeMover::default();
        mover.set_fail_demote(true);
        assert!(mode.demote_units_with(&mover, [unit]).is_err());
        assert!(matches!(
            mode.resident.residency(unit),
            Some(UnitResidency::Device(_))
        ));
        assert_eq!(mode.resident.free_blocks(3), 0);
        let state = mover.0.borrow();
        assert_eq!(state.demote_batches.len(), 1);
        assert_eq!(state.frees.len(), 1);
        assert!(state.allocations.is_empty());
    }

    #[test]
    fn h2d_failure_aborts_destinations_without_adoption_or_host_loss() {
        let mut mode = mode(PoolAddressing::ExplicitHostTransfers, 1);
        let (unit, hash) = seed(&mut mode, "owner", 1);
        let mover = FakeMover::default();
        mode.demote_units_with(&mover, [unit]).unwrap();
        mover.set_fail_promote(true);
        assert!(mode.probe_with(&mover, &cid("reader"), &[hash]).is_err());
        assert!(matches!(
            mode.resident.residency(unit),
            Some(UnitResidency::Host(_))
        ));
        assert!(mode.resident.pool.conversation(&cid("reader")).is_none());
        assert_eq!(mode.resident.free_blocks(3), UNIT_BLOCKS as u32);
        let state = mover.0.borrow();
        assert_eq!(state.promote_batches.len(), 1);
        assert!(state.frees.is_empty());
        assert_eq!(state.allocations.len(), 1);
    }

    #[test]
    fn shared_host_unit_promotes_once_before_repeated_use() {
        let mut mode = mode(PoolAddressing::ExplicitHostTransfers, 1);
        let (unit, hash) = seed(&mut mode, "owner", 1);
        let mover = FakeMover::default();
        mode.demote_units_with(&mover, [unit]).unwrap();
        mode.probe_with(&mover, &cid("reader-a"), &[hash]).unwrap();
        mode.probe_with(&mover, &cid("reader-b"), &[hash]).unwrap();
        let state = mover.0.borrow();
        assert_eq!(state.promote_batches.len(), 1);
        assert_eq!(state.frees.len(), 1);
        assert!(state.allocations.is_empty());
    }

    #[test]
    fn incomplete_layer_table_is_rejected() {
        let mut resident = ResidentKv::new(&[3, 4], UNIT_BLOCKS as u32);
        let mut placement = UnitPlacement::new();
        placement.insert(3, 0);
        resident.seal_unit(&cid("bad"), UnitHash([7; 16]), &placement);
        assert!(matches!(
            resident.table_for(&cid("bad"), 4, &[]),
            Err(crate::TableError::UnitMissingLayer { layer: 4, .. })
        ));
    }

    fn remember_sealed(mode: &mut PoolMode, label: &str, hash: UnitHash) {
        mode.convs.insert(
            label.to_string(),
            ConvState {
                tokens: vec![0; grid_tokens()],
                sealed_units: 1,
                tail: Vec::new(),
                hashes: vec![hash],
                pending: Vec::new(),
                spilled_units: 1,
                committed_at: grid_tokens(),
                cuts: Vec::new(),
                disk_ckpts: Vec::new(),
                keyless: false,
                adopted: None,
                recur_note: None,
            },
        );
        mode.recent.push(label.to_string());
    }

    #[test]
    fn keyless_prefix_lookup_never_adopts_a_named_identity() {
        let mut mode = mode(PoolAddressing::Shared, 2);
        remember_sealed(&mut mode, "named", UnitHash([1; 16]));
        mode.convs.get_mut("named").unwrap().tokens = vec![7; grid_tokens()];
        let prompt = vec![7; grid_tokens() + 32];
        assert_eq!(mode.label_for_prefix(&prompt), None);

        remember_sealed(&mut mode, "keyless-source", UnitHash([2; 16]));
        let source = mode.convs.get_mut("keyless-source").unwrap();
        source.tokens = vec![7; grid_tokens()];
        source.keyless = true;
        assert_eq!(
            mode.label_for_prefix(&prompt).as_deref(),
            Some("keyless-source")
        );
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        std::env::temp_dir()
            .join(format!("imparo-pool-{name}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn host_read_failure_keeps_host_index_and_conversation_unchanged() {
        let mut mode = mode(PoolAddressing::ExplicitHostTransfers, 1);
        let (unit, hash) = seed(&mut mode, "idle", 3);
        remember_sealed(&mut mode, "idle", hash);
        let mover = FakeMover::default();
        mode.demote_units_with(&mover, [unit]).unwrap();
        mover.set_fail_read(true);
        let dir = scratch("host-read-fail");
        let root = crate::ConfigRoot::new(b"model", (1, 1), b"geometry");
        let store = Store::open(&dir, &root).unwrap();
        let disk = crate::disk::DiskQueue::new(store.clone());

        assert!(
            mode.evict_oldest_host_to_disk_with(
                &mover,
                "incoming",
                Some(&store),
                Some(&disk),
            )
            .is_err()
        );
        assert!(matches!(
            mode.resident.residency(unit),
            Some(UnitResidency::Host(_))
        ));
        assert!(mode.convs.contains_key("idle"));
        assert_eq!(mode.resident.pool.unit_hash(unit), Some(hash));
        let state = mover.0.borrow();
        assert!(state.frees.is_empty());
        assert_eq!(state.allocations.len(), 1);
        drop(state);
        drop(disk);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn durable_ack_failure_keeps_host_index_and_conversation_unchanged() {
        let mut mode = mode(PoolAddressing::ExplicitHostTransfers, 1);
        let (unit, hash) = seed(&mut mode, "idle", 5);
        remember_sealed(&mut mode, "idle", hash);
        let mover = FakeMover::default();
        mode.demote_units_with(&mover, [unit]).unwrap();
        let dir = scratch("durable-ack-fail");
        let root = crate::ConfigRoot::new(b"model", (1, 1), b"geometry");
        let store = Store::open(&dir, &root).unwrap();
        let disk = crate::disk::DiskQueue::disconnected_for_test();

        assert!(
            mode.evict_oldest_host_to_disk_with(
                &mover,
                "incoming",
                Some(&store),
                Some(&disk),
            )
            .is_err()
        );
        assert!(matches!(
            mode.resident.residency(unit),
            Some(UnitResidency::Host(_))
        ));
        assert!(mode.convs.contains_key("idle"));
        assert_eq!(mode.resident.pool.unit_hash(unit), Some(hash));
        assert!(!store.has_unit(&hash));
        let state = mover.0.borrow();
        assert!(state.frees.is_empty());
        assert_eq!(state.allocations.len(), 1);
        drop(state);
        drop(disk);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn host_to_disk_writes_canonical_blob_before_same_mover_frees_handle() {
        let mut mode = mode(PoolAddressing::ExplicitHostTransfers, 1);
        let (unit, hash) = seed(&mut mode, "idle", 4);
        remember_sealed(&mut mode, "idle", hash);
        let mover = FakeMover::default();
        mode.demote_units_with(&mover, [unit]).unwrap();
        let expected = match mode.resident.residency(unit) {
            Some(UnitResidency::Host(host)) => {
                mode.host_unit_canonical_with(&mover, host).unwrap()
            }
            _ => panic!("unit should be Host resident"),
        };
        let raw = mover
            .0
            .borrow()
            .allocations
            .values()
            .next()
            .expect("Host allocation")
            .clone();
        let dir = scratch("host-durable");
        let root = crate::ConfigRoot::new(b"model", (1, 1), b"geometry");
        let store = Store::open(&dir, &root).unwrap();
        let disk = crate::disk::DiskQueue::new(store.clone());

        assert!(
            mode.evict_oldest_host_to_disk_with(
                &mover,
                "incoming",
                Some(&store),
                Some(&disk),
            )
            .unwrap()
        );
        let stored = store.get_unit(&hash).expect("durable canonical unit");
        assert_eq!(stored, expected);
        let checkpoint = crate::state::delta_blob(&crate::state::anchor_link(
            &KvState {
                boundary: grid_tokens(),
                full: Vec::new(),
                window: Vec::new(),
                recurrent: Vec::new(),
            },
            grid_tokens(),
        ));
        let decoded = crate::state::state_from_blobs(&[stored], &checkpoint).unwrap();
        assert_eq!(decoded.full.len(), 1);
        let layer = &decoded.full[0];
        assert_eq!(
            (layer.layer, layer.base_pos, layer.positions),
            (3, 0, grid_tokens())
        );
        let k_len = grid_tokens() * 3;
        assert_eq!(layer.k, raw[..k_len]);
        assert_eq!(layer.v, raw[k_len..]);
        assert!(mode.resident.residency(unit).is_none());
        assert!(!mode.convs.contains_key("idle"));
        let state = mover.0.borrow();
        assert_eq!(state.frees.len(), 1);
        assert!(state.allocations.is_empty());
        drop(state);
        drop(disk);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn shared_host_pressure_drops_refs_until_last_copy_is_durable() {
        let mut mode = mode(PoolAddressing::ExplicitHostTransfers, 1);
        let (unit, hash) = seed(&mut mode, "owner", 8);
        remember_sealed(&mut mode, "owner", hash);
        let mover = FakeMover::default();
        mode.probe_with(&mover, &cid("reader"), &[hash])
            .expect("reader adopts shared unit");
        remember_sealed(&mut mode, "reader", hash);
        mode.demote_units_with(&mover, [unit]).unwrap();
        assert_eq!(mode.resident.pool.refcount(unit), 2);

        let dir = scratch("shared-host-pressure");
        let root = crate::ConfigRoot::new(b"model", (1, 1), b"geometry");
        let store = Store::open(&dir, &root).unwrap();
        let disk = crate::disk::DiskQueue::new(store.clone());

        assert!(
            mode.evict_oldest_host_to_disk_with(
                &mover,
                "incoming",
                Some(&store),
                Some(&disk),
            )
            .unwrap()
        );
        assert!(!mode.convs.contains_key("owner"));
        assert!(mode.convs.contains_key("reader"));
        assert_eq!(mode.resident.pool.refcount(unit), 1);
        assert!(matches!(
            mode.resident.residency(unit),
            Some(UnitResidency::Host(_))
        ));
        assert!(!store.has_unit(&hash));
        {
            let state = mover.0.borrow();
            assert!(state.frees.is_empty());
            assert_eq!(state.allocations.len(), 1);
        }

        assert!(
            mode.evict_oldest_host_to_disk_with(
                &mover,
                "incoming",
                Some(&store),
                Some(&disk),
            )
            .unwrap()
        );
        assert!(!mode.convs.contains_key("reader"));
        assert!(mode.resident.residency(unit).is_none());
        assert!(store.has_unit(&hash));
        {
            let state = mover.0.borrow();
            assert_eq!(state.frees.len(), 1);
            assert!(state.allocations.is_empty());
        }

        drop(disk);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn same_label_replacement_releases_old_tail_before_new_allocation() {
        let mut strides = BTreeMap::new();
        strides.insert(3, (1, 1));
        let mut mode = PoolMode::new(
            PoolAddressing::Shared,
            None,
            vec![3],
            UNIT_BLOCKS as u32,
            0,
            strides,
            crate::ConfigRoot::new(b"model", (1, 1), b"geometry"),
        );
        let old = mode.resident.alloc_unit().expect("old tail fills capacity");
        mode.drain_lifecycle();
        mode.convs.insert(
            "same".to_string(),
            ConvState {
                tokens: vec![1],
                sealed_units: 0,
                tail: vec![old.clone()],
                hashes: Vec::new(),
                pending: Vec::new(),
                spilled_units: 0,
                committed_at: 0,
                cuts: Vec::new(),
                disk_ckpts: Vec::new(),
                keyless: false,
                adopted: None,
                recur_note: None,
            },
        );

        mode.release_tail_for_replacement("same");
        assert!(mode.convs["same"].tail.is_empty());
        let fresh = mode
            .alloc_units(1, "same", None, None)
            .expect("released tail must fit");
        assert_eq!(fresh, vec![old]);
    }

    /// `release_idle` names a tenant in its signature and never reads it. Every
    /// model-facing method panics, so the day one of them starts being called this
    /// stops compiling into a passing test and says which one.
    struct NoTenant;

    impl crate::tenant::PoolTenant for NoTenant {
        fn recurrent_elems(&self) -> usize {
            0
        }
        fn backend(&self) -> Option<&'static dyn Backend> {
            None
        }
        fn kv_runtime(&self) -> &crate::state::KvRuntime {
            unimplemented!("release_idle must not read the tenant's runtime")
        }
        fn kv_state_geometry(&self) -> Vec<crate::LayerStateGeom> {
            unimplemented!("release_idle must not read the tenant's geometry")
        }
        fn kv_apply_tables(&self, _tables: &BTreeMap<u32, Vec<u32>>) {
            unimplemented!("release_idle must not install block tables")
        }
        fn kv_apply_region(&self, _region: usize) {
            unimplemented!("release_idle must not move the ring region")
        }
        fn kv_recurrent_note(&self) -> Option<(usize, Vec<u8>)> {
            unimplemented!("release_idle must not read recurrent state")
        }
        fn kv_capture_windows(&self, _boundary: usize) -> crate::KvState {
            unimplemented!("release_idle must not capture")
        }
        fn kv_capture_delta(&self, _boundary: usize, _from: usize) -> crate::KvDelta {
            unimplemented!("release_idle must not capture")
        }
        fn kv_assemble_chain(
            &self,
            _chain: &[&crate::KvDelta],
        ) -> Option<crate::KvState> {
            unimplemented!("release_idle must not assemble")
        }
        fn kv_resume(&mut self, _state: &crate::KvState) -> Result<(), String> {
            unimplemented!("release_idle must not resume")
        }
    }

    /// THE CUT LIST MUST STAY ASCENDING. `extents_below` is a `partition_point` and
    /// `Cut::for_stream` builds extents that have to abut, so an out-of-order entry is
    /// not a cosmetic problem: it makes the manifest name a cut whose extents stop
    /// short, and the reader refuses the checkpoint that counts its tail from there
    /// ("the extents reach 1664, the checkpoint counts its tail from 1792").
    ///
    /// A BRANCH POINT is the cut that arrives out of order. It sits where a user
    /// message starts, which is below where the previous request ended, so it is
    /// recorded after a higher cut is already in the list.
    #[test]
    fn a_branch_point_lands_in_order_below_a_cut_already_recorded() {
        let g = grid_tokens();
        let mut cuts = vec![24 * g];
        record_cut(&mut cuts, 22 * g); // the branch: BELOW what is already there
        record_cut(&mut cuts, 27 * g); // this request's end
        assert_eq!(&cuts, &vec![22 * g, 24 * g, 27 * g]);
        assert!(cuts.windows(2).all(|w| w[0] < w[1]), "strictly ascending");

        // Idempotent: re-recording a boundary already cut is the common case, because
        // a re-committed branch point names the same position.
        record_cut(&mut cuts, 24 * g);
        assert_eq!(&cuts, &vec![22 * g, 24 * g, 27 * g], "no duplicate");
        // 0 is "no cut", not a boundary at the start of the stream.
        record_cut(&mut cuts, 0);
        assert_eq!(&cuts, &vec![22 * g, 24 * g, 27 * g], "zero is not a cut");
    }

    /// `end` records the request's own end, and it is the same rule the resume path
    /// uses -- so this says nothing about WHERE a resume point lands, only that the
    /// list `end` leaves behind is ordered and holds it.
    #[test]
    fn end_records_this_requests_own_cut() {
        let g = grid_tokens();
        let mut strides = BTreeMap::new();
        strides.insert(3, (1, 1));
        let mut mode = PoolMode::new(
            PoolAddressing::Shared,
            None,
            vec![3],
            UNIT_BLOCKS as u32,
            0,
            strides,
            crate::ConfigRoot::new(b"model", (1, 1), b"geometry"),
        );
        let units = 28;
        let hashes: Vec<UnitHash> =
            (0..units).map(|i| UnitHash([i as u8; 16])).collect();
        mode.convs.insert(
            "c".to_string(),
            ConvState {
                tokens: vec![7; units * g],
                sealed_units: units,
                tail: Vec::new(),
                hashes: hashes.clone(),
                pending: Vec::new(),
                spilled_units: 0,
                committed_at: 0,
                // A branch point recorded earlier in this turn.
                cuts: vec![22 * g],
                disk_ckpts: Vec::new(),
                keyless: false,
                adopted: None,
                recur_note: None,
            },
        );
        mode.end("c", vec![7; units * g], &hashes).unwrap();
        let own_end = crate::identity::resume_point(units * g, units * g);
        let cuts = &mode.convs["c"].cuts;
        assert_eq!(
            cuts,
            &vec![22 * g, own_end],
            "the branch, then this request's end"
        );
        assert!(cuts.windows(2).all(|w| w[0] < w[1]), "strictly ascending");
    }

    fn conv_holding(hash: UnitHash) -> ConvState {
        ConvState {
            tokens: vec![1],
            sealed_units: 1,
            tail: Vec::new(),
            hashes: vec![hash],
            pending: Vec::new(),
            spilled_units: 0,
            committed_at: 0,
            cuts: Vec::new(),
            disk_ckpts: Vec::new(),
            keyless: false,
            adopted: None,
            recur_note: None,
        }
    }

    /// The contract `drain_lifecycle` states: one allocator transition, one backend
    /// advice. This is the gate that was missing -- `advise_placement` returns on its
    /// first line without a backend, so nothing here could see a second advice, and a
    /// direct call beside the queue survived review.
    #[test]
    fn releasing_a_conversation_advises_each_placement_exactly_once() {
        let mut mode = mode(PoolAddressing::Shared, 4);
        let (_unit, hash) = seed(&mut mode, "old", 1);
        mode.convs.insert("old".to_string(), conv_holding(hash));
        mode.recent.push("old".to_string());
        // Retain nothing beyond the incoming conversation, so "old" is dropped.
        mode.resident_budget = 0;

        let before = mode.advice_applied;
        mode.release_idle(&NoTenant, "new", &[], None, None)
            .expect("release must succeed");

        assert_eq!(
            mode.advice_applied - before,
            1,
            "one released placement, one advice"
        );
        assert!(
            mode.resident.drain_lifecycle().is_empty(),
            "release_idle must drain the events it creates, not leave them to be \
             applied by whatever operation happens to run next"
        );
    }
}

fn clone_windows(s: &KvState) -> KvState {
    KvState {
        boundary: s.boundary,
        recurrent: s.recurrent.clone(),
        full: Vec::new(),
        window: s
            .window
            .iter()
            .map(|l| crate::KvLayerState {
                layer: l.layer,
                base_pos: l.base_pos,
                positions: l.positions,
                k: l.k.clone(),
                v: l.v.clone(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::{cap_ckpts, cuts_for_switch_in};
    use crate::store::{Ckpt, Cut};
    use crate::{Manifest, UnitHash};

    fn ck(boundary: u64, from: u64, turn: bool) -> Ckpt {
        Ckpt {
            boundary,
            from,
            blob: crate::UnitHash([0; 16]),
            tail: Vec::new(),
            turn,
        }
    }

    /// The windowed shape, which is the one that chains. `win` is how far a chain must
    /// reach; a test that is not about ancestors passes 0.
    fn deltas(win: usize) -> crate::CheckpointShape {
        crate::CheckpointShape::WindowDeltas { win_max: win }
    }

    fn ends(kept: &[Ckpt]) -> Vec<u64> {
        kept.iter().map(|c| c.boundary).collect()
    }

    #[test]
    fn resident_switch_in_keeps_prior_disk_cuts() {
        let prior = vec![1280, 1344];
        assert_eq!(cuts_for_switch_in(&prior, None), prior);

        let manifest = Manifest {
            boundary: 640,
            cuts: vec![Cut {
                end: 640,
                unit: UnitHash([1; 16]),
            }],
            ckpts: Vec::new(),
            keyless: false,
        };
        assert_eq!(
            cuts_for_switch_in(&prior, Some(&manifest)),
            vec![640],
            "a real disk restore must take its cuts from the manifest"
        );
    }

    /// The rule the disk layout turns on: a user turn keeps ONE state, at the cut a
    /// later prompt attaches to, and its tool-call steps replace each other. Without it
    /// a turn with many tool calls spends the whole cap and the older turns -- the
    /// rewind targets the cap exists for -- are evicted by a single chatty turn.
    #[test]
    fn a_turn_keeps_one_state_at_its_branch_point() {
        // turn A starts at 100 and takes two tool calls; turn B starts at 700 and is
        // mid-flight, its newest step at 900.
        let mut kept = vec![
            ck(100, 0, true),
            ck(300, 0, false),
            ck(500, 0, false),
            ck(700, 0, true),
            ck(900, 0, false),
        ];
        cap_ckpts(&mut kept, 20, deltas(0));
        // 100 and 700 are branch points: where a user message starts, and equally where
        // everything before it ended. So 700 already carries all of turn A, and A's own
        // steps (300, 500) are superseded by it rather than sealed. 900 is the tip.
        assert_eq!(
            ends(&kept),
            vec![100, 700, 900],
            "a branch point per turn, plus the live tip"
        );
    }

    #[test]
    fn the_cap_counts_turns_not_steps() {
        // Three turns, two steps each. A cap of 2 keeps the newest TWO TURNS -- six
        // records, not six slots -- and 100 is the turn the cap evicts.
        let mut kept = vec![
            ck(100, 0, true),
            ck(200, 0, false),
            ck(300, 0, true),
            ck(400, 0, false),
            ck(500, 0, true),
            ck(600, 0, false),
        ];
        cap_ckpts(&mut kept, 2, deltas(0));
        assert_eq!(ends(&kept), vec![300, 500, 600], "two turns, plus the tip");
    }

    /// A link covers [from, boundary), so a survivor that cannot reach back a whole
    /// window on its own keeps the ancestors that carry it there.
    #[test]
    fn ancestors_a_survivor_still_needs_are_kept() {
        let mut kept = vec![ck(100, 0, true), ck(200, 100, true), ck(300, 200, true)];
        cap_ckpts(&mut kept, 1, deltas(250));
        assert_eq!(
            ends(&kept),
            vec![100, 200, 300],
            "300 owns only [200,300); reaching back 250 needs both ancestors"
        );
    }

    #[test]
    fn a_lone_step_survives_with_no_turn_marker_anywhere() {
        let mut kept = vec![ck(640, 0, false)];
        cap_ckpts(&mut kept, 20, deltas(0));
        assert_eq!(ends(&kept), vec![640]);
    }

    /// BOTH SHAPES KEEP THE SAME BOUNDARIES. A snapshot carries the whole state and a
    /// delta carries a span, but the retention rule does not read that: one state per
    /// turn either way. What the shape decides is whether the ANCESTORS of a kept
    /// boundary are kept with it -- a delta needs them to cover a window, a snapshot is
    /// complete on its own and needs none.
    #[test]
    fn the_shape_changes_the_ancestors_kept_not_the_turns() {
        let steps = || {
            vec![
                ck(640, 0, true),
                ck(704, 640, false),
                ck(768, 704, true),
                ck(832, 768, false),
            ]
        };
        // Two turns: {640, 704} and {768, 832}. Branch points 640 and 768, tip 832.
        let mut snap = steps();
        cap_ckpts(&mut snap, 20, crate::CheckpointShape::Snapshots);
        assert_eq!(ends(&snap), vec![640, 768, 832]);
        let mut delta = steps();
        cap_ckpts(&mut delta, 20, deltas(0));
        assert_eq!(ends(&delta), ends(&snap), "same turns, same boundaries");

        // Now give the deltas a window to cover. Each link spans 64 tokens, so 832
        // reaching back 192 needs 768 and 704 under it -- ancestors a snapshot would
        // never keep, because a snapshot at 832 already describes 832 by itself.
        let mut delta = steps();
        cap_ckpts(&mut delta, 20, deltas(192));
        assert_eq!(ends(&delta), vec![640, 704, 768, 832]);
        let mut snap = steps();
        cap_ckpts(&mut snap, 20, crate::CheckpointShape::Snapshots);
        assert_eq!(ends(&snap), vec![640, 768, 832], "no window, no ancestors");
    }

    /// The shape is read off the layer geometry, and a hybrid counts as windowed: its
    /// window rows still have to accumulate, and the recurrent blob rides in each link.
    #[test]
    fn the_shape_follows_the_windowed_layers() {
        use crate::{CheckpointShape, LayerStateGeom, StateKind};
        let geom = |kinds: &[StateKind]| -> Vec<LayerStateGeom> {
            kinds
                .iter()
                .enumerate()
                .map(|(i, &kind)| LayerStateGeom {
                    layer: i as u32,
                    kind,
                    k_stride: 2,
                    v_stride: 2,
                })
                .collect()
        };
        let win = StateKind::Window {
            window: 512,
            ring: 1024,
        };
        assert_eq!(
            CheckpointShape::of(&geom(&[StateKind::Full, win])),
            CheckpointShape::WindowDeltas { win_max: 512 },
        );
        // LFM2: full-attention layers only, its recurrent state outside the KV geometry.
        assert_eq!(
            CheckpointShape::of(&geom(&[StateKind::Full])),
            CheckpointShape::Snapshots,
        );
        assert_eq!(CheckpointShape::of(&[]), CheckpointShape::Snapshots);
        // A snapshot needs no ancestors, so its chain floor is the boundary itself.
        assert!(!CheckpointShape::Snapshots.chains());
        assert_eq!(CheckpointShape::Snapshots.chain_floor(1280), None);
        assert_eq!(
            CheckpointShape::WindowDeltas { win_max: 512 }.chain_floor(1280),
            Some(768)
        );
    }

    /// MANY steps and no turn marker collapse to one, and that is the rule rather than
    /// a defect -- it was mistaken for one during the 2026-08-26 review.
    ///
    /// `turn` is false only on a STEP -- the record `end` makes at the request's cut;
    /// `note_ckpt` and `note_branch` both set it, and a restored manifest's boundary is
    /// treated as a turn. So an
    /// all-false set means every record belongs to ONE turn, and one turn is meant to
    /// keep one state -- the cap is a budget of turns, not of requests. A chatty turn
    /// with ten tool calls must not spend ten of twenty slots.
    #[test]
    fn steps_within_one_turn_collapse_to_the_newest_however_many_there_are() {
        let mut kept = vec![
            ck(640, 0, false),
            ck(704, 640, false),
            ck(768, 704, false),
            ck(832, 768, false),
        ];
        cap_ckpts(&mut kept, 20, deltas(0));
        assert_eq!(
            ends(&kept),
            vec![832],
            "no turn marker means one turn, and one turn keeps one state"
        );
        // A turn marker splits them, and then the cap really is a budget of turns.
        let mut kept = vec![
            ck(640, 0, false),
            ck(704, 640, true),
            ck(768, 704, false),
            ck(832, 768, true),
        ];
        cap_ckpts(&mut kept, 20, deltas(0));
        // `turn` marks where a turn STARTS, so these are the branch points 704 and 832.
        // 640 is a step of the turn that ENDS at 704, and 704 already carries it.
        assert_eq!(
            ends(&kept),
            vec![704, 832],
            "one per turn, at the branch point"
        );
    }
}
