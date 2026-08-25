//! Pool-mode orchestration: resident multi-conversation KV over imparo-kv's
//! ResidentKv, with the disk tier feeding and fed through the same block tables.
//!
//! The one-copy discipline end to end: full-attention KV is written ONCE by
//! prefill into pooled blocks and never moves; sharing is table aliasing;
//! switching restores only the bounded windowed state; disk restore writes
//! straight into freshly allocated blocks (table-aware -- a linear restore would
//! scatter under paging); disk spill reads unit bytes from their blocks.

use imparo_model::kv::KvPoolMember;
use std::collections::BTreeMap;
use std::path::PathBuf;

use imparo_kv::identity::UnitHash;
use imparo_kv::resident::{UNIT_BLOCKS, UnitPlacement};
use imparo_kv::state::KvDelta;
use imparo_kv::{ConversationId, KvState, Manifest, ResidentKv, Store};
// The kv_* pool methods are defaults on this trait, not inherent methods.

const UNIT_TOKENS: usize = imparo_kv::UNIT_TOKENS;

/// A checkpoint's identity: the hash of the last WHOLE content unit at or below
/// it, plus its exact position.
///
/// Two parts because a turn boundary does not land on the 256-token content grid.
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
}

pub struct ConvState {
    pub tokens: Vec<u32>,
    pub sealed_units: usize,
    /// Pre-allocated placements for positions beyond the sealed boundary,
    /// consumed front-first as units seal.
    pub tail: Vec<UnitPlacement>,
    /// The sealed unit-hash chain (what a later spill writes as the manifest).
    hashes: Vec<UnitHash>,
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
    disk_ckpts: Vec<imparo_kv::store::Ckpt>,
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
    /// Which conversation each windowed REGION currently holds. A windowed layer keeps
    /// one ring per region, so a conversation whose region still names it can be resumed
    /// without restoring its window at all -- the bytes never left. One region is the
    /// single-ring layout, where only the active conversation qualifies.
    regions: Vec<Option<String>>,
}

fn cid(label: &str) -> ConversationId {
    ConversationId(label.to_string())
}

impl PoolMode {
    #[must_use]
    pub fn new(
        layers: Vec<u32>,
        capacity_blocks: u32,
        slack: usize,
        strides: BTreeMap<u32, (usize, usize)>,
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
            strides,
            regions: vec![None; imparo_model::kv::window_regions()],
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
            let h = label
                .bytes()
                .fold(0xcbf2_9ce4_8422_2325_u64, |a, b| {
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
        self.regions.len() > 1 && self.regions.iter().any(|h| h.as_deref() == Some(label))
    }

    /// Page-advise one placement's blocks (release or reuse).
    fn advise_placement(&self, pl: &UnitPlacement, free: bool) {
        let Some(be) = imparo_model::backend::active() else {
            return;
        };
        for (&layer, blocks) in pl {
            let Some(&(ks, vs)) = self.strides.get(&layer) else {
                continue;
            };
            for &b in blocks {
                let (ko, vo) =
                    (u64::from(b) * 64 * ks as u64, u64::from(b) * 64 * vs as u64);
                if free {
                    be.kv_advise_free(layer, false, ko, 64 * ks as u64);
                    be.kv_advise_free(layer, true, vo, 64 * vs as u64);
                } else {
                    be.kv_advise_reuse(layer, false, ko, 64 * ks as u64);
                    be.kv_advise_reuse(layer, true, vo, 64 * vs as u64);
                }
            }
        }
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
    ) {
        let key = (tip, delta.boundary);
        self.ckpt_order.retain(|h| *h != key);
        self.ckpt_order.push(key);
        self.window_ckpts.insert(key, CkptEntry { delta, prev, tokens });
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
    ///   rows  [u*256, B)  the stored tokens must be these tokens    literal comparison
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
            let u = b / UNIT_TOKENS;
            b <= upto
                && u >= 1
                && u <= max_units
                && hashes.get(u - 1) == Some(&tip)
                && ids.len() >= b
                && ids[u * UNIT_TOKENS..b] == tok[..]
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
    fn capture_ckpt(&mut self, model: &dyn KvPoolMember, label: &str, pk: PendingCkpt) {
        let mut delta = model.kv_capture_delta(pk.boundary, pk.from);
        let unit_end = pk.boundary / UNIT_TOKENS * UNIT_TOKENS;
        if pk.boundary > unit_end {
            let tail = self.convs.get(label).map_or_else(Vec::new, |c| c.tail.clone());
            let geom = model.kv_state_geometry();
            if let Some(be) = imparo_model::backend::active() {
                delta.tail =
                    self.read_rows(&cid(label), &tail, unit_end, pk.boundary, &geom, be);
                if delta.tail.is_empty() {
                    // rows unreachable: no checkpoint beats a wrong one
                    if imparo_model::log_on() {
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
            if imparo_model::log_on() {
                eprintln!(
                    "[imparo] kv ckpt {} dropped: no recurrent state",
                    pk.boundary
                );
            }
            return;
        }
        self.remember_delta(pk.tip, pk.prev, delta, pk.tokens);
    }

    /// The link at `key` and every ancestor its window still needs, newest first.
    ///
    /// A link owns `[from, boundary)`. Whoever restores at that boundary reads
    /// `[boundary - window, boundary)`, which no single link covers -- the ancestors are
    /// the only copy of the rows below `from`. Naming just the newest link was enough
    /// while the disk wrote whole windows and is not now.
    fn ckpt_chain(&self, key: CkptKey, win_max: usize) -> Vec<(KvDelta, Vec<u32>)> {
        let mut out = Vec::new();
        let floor = key.1.saturating_sub(win_max);
        let mut need = key.1;
        let mut cur = Some(key);
        while let Some(k) = cur {
            let Some(e) = self.window_ckpts.get(&k) else {
                break;
            };
            out.push((e.delta.clone(), e.tokens.clone()));
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
    fn try_assemble(&self, model: &dyn KvPoolMember, key: CkptKey) -> Option<KvState> {
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

    fn alloc_units(
        &mut self,
        n: usize,
        keep: &str,
    ) -> Result<Vec<UnitPlacement>, String> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            loop {
                if let Some(pl) = self.resident.alloc_unit() {
                    // recommit possibly advise-freed pages BEFORE any GPU write
                    self.advise_placement(&pl, false);
                    out.push(pl);
                    break;
                }
                if !self.resident.evict_unreferenced().is_empty() {
                    continue;
                }
                if self.resident.drop_lru_conversation(&cid(keep)) {
                    continue;
                }
                return Err("kv pool exhausted: request exceeds device capacity".into());
            }
        }
        Ok(out)
    }

    fn apply_tables(&self, model: &dyn KvPoolMember, label: &str, tail: &[UnitPlacement]) {
        let mut tables = BTreeMap::new();
        for &layer in &self.layers {
            tables.insert(layer, self.resident.table_for(&cid(label), layer, tail));
        }
        model.kv_apply_tables(&tables);
    }

    /// Prepare for a request: adopt what is resident, restore what disk holds,
    /// allocate the tail, apply tables and window state. Returns the resume
    /// position (0 = cold). `upper` bounds prompt+generation for tail sizing.
    ///
    /// # Errors
    /// Returns an error when the pool cannot fit the request even after eviction.
    #[allow(clippy::too_many_lines)]
    pub fn begin(
        &mut self,
        model: &mut dyn KvPoolMember,
        store: Option<&Store>,
        disk: Option<&crate::disk::DiskQueue>,
        label: &str,
        ids: &[u32],
        hashes: &[UnitHash],
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
        // A different conversation is coming in: the outgoing one's pending
        // checkpoints are captured from the still-intact rings and its state is
        // written through to disk NOW -- the only moment bytes ever move for it.
        // Then residency beyond the keep-set is released.
        if let Some(out) = self.active.clone() {
            if out != label {
                self.switch_out(model, store, disk, &out)?;
                self.release_idle(label);
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
                    let resume = imparo_model::kv::resume_point(c.tokens.len(), ids.len());
                    let need_units = upper
                        .div_ceil(UNIT_TOKENS)
                        .saturating_sub(c.sealed_units + self.convs[label].tail.len());
                    if need_units > 0 {
                        let more = self.alloc_units(need_units, label)?;
                        self.convs.get_mut(label).unwrap().tail.extend(more);
                    }
                    let tail = self.convs[label].tail.clone();
                    self.apply_tables(model, label, &tail);
                    let _ = self.resident.probe(&cid(label), hashes);
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
        let hit = self.resident.probe(&cid(label), hashes);

        // Deepest checkpoint this prompt reproduces AND whose chain still assembles.
        // Its boundary is a turn start, so it sits anywhere -- `k` is where the
        // POOLED part of it ends, and everything from there to `k_at` rides in the
        // checkpoint's own tail rows.
        let mut k_at = 0;
        let mut k_state: Option<KvState> = None;
        let mut k_tail: Vec<imparo_kv::KvLayerState> = Vec::new();
        for key in self.matching_ckpts(None, hashes, ids, usize::MAX, hit) {
            if let Some(st) = self.try_assemble(model, key) {
                k_at = key.1;
                k_tail.clone_from(&self.window_ckpts[&key].delta.tail);
                k_state = Some(st);
                break;
            }
        }
        let k = k_at / UNIT_TOKENS;
        if imparo_model::log_on() {
            eprintln!(
                "[imparo] kv begin {label}: prompt_units={} resident_hit={hit} ckpts={:?} chose={k_at}",
                hashes.len(),
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
            if st.best_match(hashes).is_none() && imparo_model::log_on() {
                // How far the stored chain and this prompt agree. A partial match means
                // the prompt DIVERGED from what was stored -- the re-rendered text of an
                // earlier turn does not re-tokenise to the stream that produced it --
                // rather than that nothing related is on disk.
                let best = st
                    .prefix_matches(hashes)
                    .into_iter()
                    .map(|(_, m, n)| (n, m.hashes.len()))
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
                (usize, imparo_kv::Manifest, Vec<Vec<u8>>, Vec<Vec<u8>>, PathBuf);
            let mut best: Option<Candidate> = None;
            let win_max = win_max_of(&model.kv_state_geometry());
            for (mpath, m, agree) in st.prefix_matches(hashes) {
                // Old stores name one boundary in `boundary`/`tail`; newer ones list every
                // kept boundary. Read both, so a store written before this still restores.
                let legacy = imparo_kv::store::Ckpt {
                    boundary: m.boundary,
                    // A store written before links: that blob owns its whole window.
                    from: 0,
                    blob: imparo_kv::UnitHash([0; 16]),
                    tail: m.tail.clone(),
                };
                let candidates: Vec<&imparo_kv::store::Ckpt> = if m.ckpts.is_empty() {
                    vec![&legacy]
                } else {
                    m.ckpts.iter().collect()
                };
                for c in candidates {
                    let b = c.boundary as usize;
                    let units = b / UNIT_TOKENS;
                    let unit_end = units * UNIT_TOKENS;
                    if units > agree
                        || units > m.hashes.len()
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
                        let floor = b.saturating_sub(win_max);
                        let mut desc: Vec<&imparo_kv::store::Ckpt> =
                            m.ckpts.iter().filter(|o| o.boundary as usize <= b).collect();
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
                    let Some(u) = m.hashes[..units]
                        .iter()
                        .map(|h| st.get_unit(h))
                        .collect::<Option<Vec<Vec<u8>>>>()
                    else {
                        continue;
                    };
                    // The link and everything below it, newest first: the assemble walks
                    // down until the window is covered. A store written before links has
                    // one whole-window blob, which reads back as an anchor.
                    let mut chain: Vec<Vec<u8>> = Vec::new();
                    let load = |h: &imparo_kv::UnitHash| st.get_checkpoint(h);
                    if m.ckpts.is_empty() {
                        let Some(one) = st.checkpoint_for(&mpath) else { continue };
                        chain.push(one);
                    } else {
                        let Some(target) = load(&c.blob) else { continue };
                        chain.push(target);
                        let mut older: Vec<&imparo_kv::store::Ckpt> =
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
                    }
                    let chosen = imparo_kv::Manifest {
                        boundary: c.boundary,
                        hashes: m.hashes[..units].to_vec(),
                        tail: c.tail.clone(),
                        ckpts: m.ckpts.clone(),
                        keyless: m.keyless,
                    };
                    best = Some((b, chosen, u, chain, mpath.clone()));
                }
            }
            if let Some((b, m, u, chain, mpath)) = best {
                if imparo_model::log_on() {
                    eprintln!("[imparo] kv disk: restoring at boundary {b}");
                }
                // Absent the fork header this IS that conversation, being rewound -- so
                // the file it was stored under is replaced rather than left behind. The
                // caller sets `new_conversation` when the client asked for a fork, and
                // then nothing is adopted for replacement.
                adopted_from = Some((mpath, m.keyless));
                from_disk = Some((m, u, chain));
            } else if imparo_model::log_on() {
                // WHY none of them fit, not just that none did. Each boundary is refused
                // for exactly one of four reasons and they mean different things: too
                // deep for what we agree on, past the prompt, already beaten by a
                // resident checkpoint, or its sub-unit tokens are somebody else's.
                for (_, m, agree) in st.prefix_matches(hashes) {
                    for c in &m.ckpts {
                        let b = c.boundary as usize;
                        let units = b / UNIT_TOKENS;
                        let ue = units * UNIT_TOKENS;
                        let why = if units > agree || units > m.hashes.len() {
                            "needs more units than we agree on"
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
                            m.hashes.len()
                        );
                    }
                }
                let best_n = st
                    .prefix_matches(hashes)
                    .into_iter()
                    .map(|(_, m, n)| (n, m.hashes.len(), m.ckpts.len()))
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
        let mut restored_ckpts: Vec<imparo_kv::store::Ckpt> = Vec::new();
        // (tip, link, sub-unit tail tokens) for every link a disk restore assembled from,
        // so the resident chain can be rebuilt from it -- see the comment at the fill.
        let mut restored_links: Vec<(UnitHash, KvDelta, Vec<u32>)> = Vec::new();
        let prior_disk_ckpts = self
            .convs
            .get(label)
            .map(|c| c.disk_ckpts.clone())
            .unwrap_or_default();
        // The conversation keeps only the units at or below its resume boundary;
        // everything beyond is re-filled into fresh tail blocks (sealed shared
        // units must never be rewritten).
        // (resume position, windowed+recurrent state, full rows above the last unit)
        let (resume, restored_windows, tail_rows): (
            usize,
            Option<KvState>,
            Vec<imparo_kv::KvLayerState>,
        ) = if let Some((manifest, unit_bytes, chain_blobs)) = from_disk {
                let n = manifest.hashes.len();
                let boundary = manifest.boundary as usize;
                restored_ckpts.clone_from(&manifest.ckpts);
                // adopt what is already resident among the manifest's units; the
                // rest gets fresh blocks and a table-aware write-in
                let resident_hit = self.resident.probe(&cid(label), &manifest.hashes);
                let missing = n - resident_hit;
                let fresh = self.alloc_units(missing, label)?;
                let links: Vec<imparo_kv::KvDelta> = chain_blobs
                    .iter()
                    .map(|b| imparo_kv::state::delta_from_blob(b))
                    .collect::<Result<_, _>>()?;
                let refs: Vec<&imparo_kv::KvDelta> = links.iter().collect();
                // The links this restore just parsed ARE the resident chain, and dropping
                // them made the next commit re-capture a whole window: `matching_ckpts`
                // found nothing, so the fallback wrote an anchor (21.0 MiB on E4B f16)
                // where a link is 3.5 MiB -- one whole window per restart per
                // conversation. Rebuilt below, shallowest first, so each `prev` key
                // exists before the link that names it.
                restored_links = links
                    .iter()
                    .filter_map(|d| {
                        let u = d.boundary / UNIT_TOKENS;
                        let tip = *manifest.hashes.get(u.checked_sub(1)?)?;
                        let tail = manifest
                            .ckpts
                            .iter()
                            .find(|c| c.boundary as usize == d.boundary)
                            .map(|c| c.tail.clone())?;
                        Some((tip, d.clone(), tail))
                    })
                    .collect();
                restored_links.sort_by_key(|(_, d, _)| d.boundary);
                // The BOUNDED geometry, the same view the resident chain assembles
                // against: a delta covers a window, not a whole context.
                let wgeom = imparo_model::kv::bounded_geometry(&model.kv_state_geometry());
                let state =
                    imparo_kv::state::state_from_units_and_chain(&unit_bytes, &wgeom, &refs)?;
                // write the missing units' bytes into their blocks
                let be = imparo_model::backend::active().ok_or("no backend")?;
                for (i, pl) in fresh.iter().enumerate() {
                    let u = resident_hit + i;
                    for ls in &state.full {
                        let stride = ls.k.len() / ls.positions;
                        let vstride = ls.v.len() / ls.positions;
                        let Some(blocks) = pl.get(&ls.layer) else {
                            continue;
                        };
                        for (bi, &blk) in blocks.iter().enumerate() {
                            let pos0 = u * UNIT_TOKENS + bi * 64;
                            be.write_kv_bytes(
                                ls.layer,
                                false,
                                (blk as usize * 64 * stride) as u64,
                                &ls.k[pos0 * stride..(pos0 + 64) * stride],
                            );
                            be.write_kv_bytes(
                                ls.layer,
                                true,
                                (blk as usize * 64 * vstride) as u64,
                                &ls.v[pos0 * vstride..(pos0 + 64) * vstride],
                            );
                        }
                    }
                }
                // seal the written units under their manifest hashes
                for (i, pl) in fresh.into_iter().enumerate() {
                    let u = resident_hit + i;
                    let _ =
                        self.resident
                            .seal_unit(&cid(label), manifest.hashes[u], &pl);
                }
                // Rows above the last whole unit: the checkpoint carried them, and
                // they go into this conversation's OWN tail blocks below -- there is
                // no unit for them to be part of.
                let lo = n * UNIT_TOKENS;
                let leftover: Vec<imparo_kv::KvLayerState> = state
                    .full
                    .iter()
                    .filter(|_| boundary > lo)
                    .map(|ls| {
                        let (ks, vs) =
                            (ls.k.len() / ls.positions, ls.v.len() / ls.positions);
                        imparo_kv::KvLayerState {
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
                let tip = manifest.hashes[n - 1];
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
                    manifest.tail.clone(),
                );
                (boundary, Some(windows), leftover)
            } else if k_at > 0 {
                // truncate the adopted list to the checkpointed boundary
                let _ = self.resident.probe(&cid(label), &hashes[..k]);
                (k_at, k_state, k_tail)
            } else {
                let _ = self.resident.probe(&cid(label), &[]);
                (0, None, Vec::new())
            };

        let resume_units = resume / UNIT_TOKENS;
        // tail for everything beyond the sealed resume boundary
        let need_units = upper.div_ceil(UNIT_TOKENS) - resume_units;
        let tail = self.alloc_units(need_units, label)?;
        // The checkpoint's own rows for [resume_units * 256, resume) go into the
        // FIRST tail unit -- this conversation's blocks, not the donor's.
        if !tail_rows.is_empty() {
            if let Some(be) = imparo_model::backend::active() {
                Self::write_rows(
                    &tail_rows,
                    resume_units * UNIT_TOKENS,
                    &tail,
                    &model.kv_state_geometry(),
                    be,
                );
            }
        }
        self.apply_tables(model, label, &tail);
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
            if imparo_model::log_on() {
                eprintln!(
                    "[imparo] kv region {region}: {label} keeps its window (tip {tip_before}, resume {resume})"
                );
            }
        }
        model.kv_resume(&state)?;
        let prior_note = model.kv_recurrent_note();
        self.convs.insert(
            label.to_string(),
            ConvState {
                tokens: Vec::new(),
                sealed_units: resume_units,
                tail,
                hashes: hashes[..resume_units].to_vec(),
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
        // The restored chain becomes the resident one. Shallowest first: `prev` names the
        // link below, and remember_delta's LRU must see that key already present or it
        // would treat the ancestor as evictable.
        let mut prev: Option<CkptKey> = None;
        for (tip, delta, tokens) in std::mem::take(&mut restored_links) {
            let key = (tip, delta.boundary);
            self.remember_delta(tip, prev, delta, tokens);
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
        let sealed_target = final_tokens.len() / UNIT_TOKENS;
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
        // Superseded records die uncaptured; what stays must remain capturable
        // from the rings (filled - boundary <= slack), which the eager-capture
        // rule at the branch site guarantees for everything recorded this turn.
        let slack = self.slack;
        c.pending.retain(|pk| filled - pk.boundary <= slack);
        for (h, pl) in &sealed_placements {
            let _ = self.resident.seal_unit(&cid(label), *h, pl);
        }

        // The turn's own resume point, at the last unit it filled. Unit-aligned by
        // construction, so it carries no tail rows: this one is not a turn
        // boundary, it is "as far as this conversation got".
        let boundary = sealed_target * UNIT_TOKENS;
        if boundary > 0 {
            let tip = hashes_full[sealed_target - 1];
            let tokens = self.convs.get(label).map_or_else(Vec::new, |c| c.tokens.clone());
            let prev = self.prev_ckpt_at(label, hashes_full, &tokens, boundary - 1);
            if let Some(c) = self.convs.get_mut(label) {
                if !c.pending.iter().any(|pk| pk.boundary == boundary) {
                    c.pending.push(PendingCkpt {
                        tip,
                        prev,
                        boundary,
                        from: prev.map_or(0, |(_, b)| b),
                        tokens: Vec::new(),
                    });
                }
            }
        }
        self.active = Some(label.to_string());
        Ok(())
    }

    /// Capture `label`'s pending checkpoints from the rings (valid: nothing has
    /// run since this conversation's last forward).
    fn materialize_pendings(&mut self, model: &dyn KvPoolMember, label: &str) {
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
    pub fn switch_out(
        &mut self,
        model: &dyn KvPoolMember,
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
    pub fn commit_to_disk(
        &mut self,
        model: &dyn KvPoolMember,
        store: Option<&Store>,
        disk: Option<&crate::disk::DiskQueue>,
        label: &str,
    ) -> Result<bool, String> {
        // Pendings first: a recorded-but-uncaptured branch point has no bytes, and the
        // rings still hold what it needs at this instant. This is also what makes the
        // per-turn write-through and the switch-out path do the same thing.
        self.materialize_pendings(model, label);
        let Some(c) = self.convs.get(label) else {
            return Ok(false);
        };
        let (sealed, spilled) = (c.sealed_units, c.spilled_units);
        let hashes = c.hashes.clone();
        let tokens = c.tokens.clone();
        if sealed == 0 || hashes.len() < sealed {
            return Ok(false);
        }
        // Nothing has happened since the last write: same units, same tip. Re-committing
        // is 21 MiB of checkpoint (measured, E4B) for no new durability, and switch-out
        // and shutdown both land here right after a turn end did the work.
        if spilled == sealed && c.committed_at == model.kv_runtime().filled {
            return Ok(false);
        }
        if let (Some(st), Some(dq)) = (store, disk) {
            let t_write = std::time::Instant::now();
            let be = imparo_model::backend::active().ok_or("no backend")?;
            let geom = model.kv_state_geometry();
            for (u, &h) in hashes.iter().enumerate().take(sealed) {
                // Already durable, or already handed to the writer: reading it off
                // the device again would be 4 MiB of memcpy for a file that exists.
                if dq.holds(st, &h) {
                    continue;
                }
                let Some(bytes) = self.read_unit_canonical(&cid(label), u, &geom, be)
                else {
                    continue;
                };
                dq.put_unit(h, bytes);
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
            let ck_at = model.kv_runtime().filled.max(sealed * UNIT_TOKENS);
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
            let win_max = win_max_of(&geom);
            let chain: Vec<(imparo_kv::KvDelta, Vec<u32>)> = if let Some(key) = recorded {
                self.ckpt_chain(key, win_max)
            } else {
                let unit_end = ck_at / UNIT_TOKENS * UNIT_TOKENS;
                let units = (unit_end / UNIT_TOKENS).min(sealed);
                let mut st = model.kv_capture_windows(ck_at);
                let toks = tokens
                    .get(units * UNIT_TOKENS..ck_at)
                    .map(<[u32]>::to_vec)
                    .unwrap_or_default();
                if ck_at > units * UNIT_TOKENS {
                    let tail_pl = c.tail.clone();
                    st.full = self.read_rows(
                        &cid(label),
                        &tail_pl,
                        units * UNIT_TOKENS,
                        ck_at,
                        &geom,
                        be,
                    );
                    if st.full.is_empty() || toks.len() != ck_at - units * UNIT_TOKENS {
                        // Cannot describe the part above the units: fall back to the
                        // aligned point rather than claim more than we carry.
                        st = model.kv_capture_windows(units * UNIT_TOKENS);
                    }
                }
                // No recorded link: this capture owns its whole window, so it is an
                // ANCHOR (`from: 0`) and needs no predecessor.
                let d = imparo_kv::KvDelta {
                    boundary: st.boundary,
                    from: 0,
                    window: st.window,
                    tail: imparo_kv::state::tail_rows(&st.full, st.boundary),
                    recurrent: st.recurrent,
                };
                vec![(d, toks)]
            };
            let Some((link, tail_tokens)) = chain.first().cloned() else {
                return Ok(false);
            };
            let boundary = link.boundary;
            let units = boundary / UNIT_TOKENS;
            if units > sealed {
                return Ok(false);
            }
            let tail = if boundary > units * UNIT_TOKENS {
                tail_tokens
            } else {
                Vec::new()
            };
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
            let mut kept: Vec<imparo_kv::store::Ckpt> = self
                .convs
                .get(label)
                .map(|c| c.disk_ckpts.clone())
                .unwrap_or_default();
            let mut blobs: Vec<(imparo_kv::UnitHash, Vec<u8>)> = Vec::new();
            let mut ck_bytes = 0usize;
            // Walk the chain newest first and stop as soon as the manifest covers a whole
            // window back from this boundary. An entry ALREADY on disk that reaches at
            // least as deep wins: it is written, and replacing an anchor with a delta at
            // the same boundary buys nothing and adds a dependency. That is what makes a
            // re-committed boundary cost zero bytes.
            let floor = boundary.saturating_sub(win_max);
            let mut need = boundary;
            for (i, (d, toks)) in chain.iter().enumerate() {
                // The newest link is always named -- it IS the resume point, and it
                // carries the recurrent snapshot and the sub-unit tail. Only its
                // ancestors are there to cover a window, and a model with no windowed
                // layer has none to cover (`win_max` 0, `floor` == boundary).
                if i > 0 && need <= floor {
                    break;
                }
                let b = d.boundary;
                let u = b / UNIT_TOKENS;
                let t = if b > u * UNIT_TOKENS { toks.clone() } else { Vec::new() };
                if t.len() != b - u * UNIT_TOKENS {
                    continue; // cannot describe the sub-unit part: do not name it
                }
                if let Some(old) = kept.iter().find(|c| c.boundary == b as u64) {
                    if old.from <= d.from as u64 && dq.holds_checkpoint(st, &old.blob) {
                        need = need.min(old.from as usize);
                        continue;
                    }
                }
                let bytes = imparo_kv::state::delta_blob(d);
                let a = imparo_kv::store::blob_hash(&bytes);
                kept.retain(|c| c.boundary != b as u64);
                kept.push(imparo_kv::store::Ckpt {
                    boundary: b as u64,
                    from: d.from as u64,
                    blob: a,
                    tail: t,
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
                let ue = b / UNIT_TOKENS * UNIT_TOKENS;
                b <= tokens.len()
                    && c.tail.len() == b - ue
                    && tokens.get(ue..b).is_some_and(|t| t == c.tail)
            });
            kept.sort_by_key(|c| c.boundary);
            cap_ckpts(&mut kept, self.disk_ckpt_cap, win_max);
            let keyless = self.convs.get(label).is_some_and(|c| c.keyless);
            let m = Manifest {
                boundary: boundary as u64,
                hashes: hashes[..units].to_vec(),
                tail,
                ckpts: kept.clone(),
                keyless,
            };
            let blobs_written = blobs.len();
            dq.commit(label, m, blobs);
            // SUPERSEDE. A keyless conversation is renamed by its own growth and again by
            // every restart, so the manifest it restored from can never be asked for
            // again -- and it holds its checkpoint blobs against the sweep until the
            // store hits its cap. Drop it once THIS commit names every boundary it did,
            // and only then: until that holds it is still the only way back to those
            // turns. Queued behind the commit, so a crash in between leaves the old one
            // standing rather than neither.
            let superseded = self.convs.get(label).and_then(|c| c.adopted.clone());
            if let Some(p) = superseded {
                dq.drop_manifests(vec![p]);
                if let Some(c) = self.convs.get_mut(label) {
                    c.adopted = None;
                }
                if imparo_model::log_on() {
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
            if imparo_model::log_on() {
                eprintln!(
                    "[imparo] kv write-through: {} unit(s) + {:.1} MiB checkpoint [{}, {boundary}) in {} blob(s), {:.1} ms",
                    sealed - spilled,
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
        self.strides
            .values()
            .map(|&(k, v)| UNIT_TOKENS * (k + v))
            .sum()
    }

    /// Drop device residency for conversations outside the keep-set (the
    /// incoming one, always, plus the most recent others whose units fit the
    /// resident-bytes budget) and hand the freed block pages back to the OS.
    /// Everything dropped was spilled at its own switch-out, so the disk
    /// restore path serves any return visit.
    fn release_idle(&mut self, incoming: &str) {
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
            return;
        }
        for l in &drop {
            self.resident.pool.forget(&cid(l));
            self.convs.remove(l);
            self.recent.retain(|r| r != l);
            if self.active.as_deref() == Some(l.as_str()) {
                self.active = None;
            }
        }
        let freed = self.resident.evict_unreferenced();
        for pl in &freed {
            self.advise_placement(pl, true);
        }
        // Checkpoints reachable from kept conversations stay; the rest are on
        // disk (their conv's spill) or re-capturable, so their RAM goes too.
        let mut reach: std::collections::BTreeSet<CkptKey> =
            std::collections::BTreeSet::new();
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
        geom: &[imparo_kv::LayerStateGeom],
        be: &dyn imparo_backend::Backend,
    ) -> Vec<imparo_kv::KvLayerState> {
        let mut out = Vec::new();
        if to <= from {
            return out;
        }
        for g in geom {
            if !matches!(g.kind, imparo_kv::StateKind::Full) {
                continue;
            }
            let table = self.resident.table_for(conv, g.layer, tail);
            if table.len() * 64 < to {
                return Vec::new(); // placement missing: no tail rows, no checkpoint
            }
            let n = to - from;
            let mut k = vec![0u8; n * g.k_stride];
            let mut v = vec![0u8; n * g.v_stride];
            let mut p = from;
            while p < to {
                let run = (64 - p % 64).min(to - p);
                let blk = table[p / 64] as usize;
                let off = blk * 64 + p % 64;
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
            out.push(imparo_kv::KvLayerState {
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
        rows: &[imparo_kv::KvLayerState],
        base: usize,
        tail: &[UnitPlacement],
        geom: &[imparo_kv::LayerStateGeom],
        be: &dyn imparo_backend::Backend,
    ) {
        for ls in rows {
            let Some(g) = geom.iter().find(|g| g.layer == ls.layer) else {
                continue;
            };
            let mut i = 0;
            while i < ls.positions {
                let p = ls.base_pos + i;
                let run = (64 - p % 64).min(ls.positions - i);
                let unit = (p - base) / UNIT_TOKENS;
                let Some(blocks) = tail.get(unit).and_then(|pl| pl.get(&ls.layer)) else {
                    return;
                };
                let blk = blocks[(p - base) % UNIT_TOKENS / 64] as usize;
                let off = blk * 64 + p % 64;
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

    fn read_unit_canonical(
        &self,
        conv: &ConversationId,
        unit_index: usize,
        geom: &[imparo_kv::LayerStateGeom],
        be: &dyn imparo_backend::Backend,
    ) -> Option<Vec<u8>> {
        let full: Vec<&imparo_kv::LayerStateGeom> = geom
            .iter()
            .filter(|g| matches!(g.kind, imparo_kv::StateKind::Full))
            .collect();
        // Write-through of ONE unit's full-attention rows; nothing here is a
        // conversation checkpoint, so it carries no recurrent state.
        let mut state = KvState {
            boundary: UNIT_TOKENS,
            full: Vec::new(),
            window: Vec::new(),
            recurrent: Vec::new(),
        };
        for g in full {
            let table = self.resident.table_for(conv, g.layer, &[]);
            if table.len() < (unit_index + 1) * UNIT_BLOCKS {
                return None; // placement missing (evicted mid-flight): skip write-through
            }
            let blocks =
                &table[unit_index * UNIT_BLOCKS..(unit_index + 1) * UNIT_BLOCKS];
            let mut k = vec![0u8; UNIT_TOKENS * g.k_stride];
            let mut v = vec![0u8; UNIT_TOKENS * g.v_stride];
            for (bi, &blk) in blocks.iter().enumerate() {
                be.read_kv_bytes(
                    g.layer,
                    false,
                    (blk as usize * 64 * g.k_stride) as u64,
                    &mut k[bi * 64 * g.k_stride..(bi + 1) * 64 * g.k_stride],
                );
                be.read_kv_bytes(
                    g.layer,
                    true,
                    (blk as usize * 64 * g.v_stride) as u64,
                    &mut v[bi * 64 * g.v_stride..(bi + 1) * 64 * g.v_stride],
                );
            }
            state.full.push(imparo_kv::KvLayerState {
                layer: g.layer,
                base_pos: 0,
                positions: UNIT_TOKENS,
                k,
                v,
            });
        }
        imparo_kv::state::unit_blobs(&state).into_iter().next()
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
    pub fn note_ckpt(
        &mut self,
        model: &dyn KvPoolMember,
        label: &str,
        tip: UnitHash,
        prev: Option<CkptKey>,
        boundary: usize,
        tokens: Vec<u32>,
    ) {
        self.capture_ckpt(
            model,
            label,
            PendingCkpt {
                tip,
                prev,
                boundary,
                from: prev.map_or(0, |(_, b)| b),
                tokens,
            },
        );
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
        if let Some(c) = self.convs.get_mut(label) {
            if !c.pending.iter().any(|pk| pk.boundary == boundary) {
                c.pending.push(PendingCkpt {
                    tip,
                    prev,
                    boundary,
                    from: prev.map_or(0, |(_, b)| b),
                    tokens,
                });
            }
        }
    }

    /// Rewind this conversation to the deepest recorded boundary at or below `at`,
    /// restoring the state there rather than just moving the fill mark.
    ///
    /// `kv_set_filled` is enough for a model whose whole state is the attention cache: a
    /// row above the mark is simply overwritten. It is NOT enough for a recurrent model.
    /// A convolution state is a running summary of the last few inputs with no inverse,
    /// so there is nothing to un-apply -- the only way back to position `p` is the
    /// snapshot taken at `p`. That is what the checkpoints already hold, and what the
    /// request path already uses to resume; this is the same move, asked for from turn
    /// close.
    ///
    /// Returns the boundary it landed on, or None when no recorded checkpoint at or
    /// below `at` is one this conversation's tokens prove -- in which case the caller
    /// must leave the state alone rather than half-rewind it.
    pub fn rewind_to_checkpoint(
        &mut self,
        model: &mut dyn KvPoolMember,
        label: &str,
        hashes: &[UnitHash],
        ids: &[u32],
        at: usize,
    ) -> Option<usize> {
        let sealed = self.convs.get(label)?.sealed_units;
        let key = self
            .matching_ckpts(Some(label), hashes, ids, at, sealed)
            .into_iter()
            .find(|k| self.try_assemble(model, *k).is_some())?;
        let st = self.try_assemble(model, key)?;
        model.kv_resume(&st).ok()?;
        Some(st.boundary)
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
            .filter(|(_, c)| !c.tokens.is_empty())
            .map(|(l, c)| {
                (l, c.tokens.iter().zip(ids).take_while(|(a, b)| a == b).count())
            })
            // A prompt that shares only a preamble with a resident conversation is not
            // that conversation: below one whole unit there is nothing to stand on and
            // nothing to adopt either, so the content name is the honest answer.
            .filter(|(_, n)| *n >= UNIT_TOKENS)
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
    pub fn digest(&self, model: &dyn KvPoolMember, label: &str, positions: usize) {
        if !std::env::var("IMPARO_KV_DIGEST").is_ok_and(|v| v == "1") {
            return;
        }
        let Some(be) = imparo_model::backend::active() else {
            return;
        };
        let geom = model.kv_state_geometry();
        let tail = self.convs.get(label).map_or_else(Vec::new, |c| c.tail.clone());
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
            let imparo_kv::StateKind::Window { window, ring } = g.kind else {
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
    pub fn forget(&mut self, labels: &[String]) {
        for l in labels {
            self.resident.pool.forget(&cid(l));
            self.convs.remove(l);
            if self.active.as_deref() == Some(l.as_str()) {
                self.active = None;
            }
        }
        let _ = self.resident.evict_unreferenced();
    }
}

/// Whether a captured state actually carries what this model's recurrent layers
/// need. False means the capture could not describe that position.
/// The widest window any layer keeps: how far back a checkpoint chain has to reach for
/// a restore at its newest boundary to see a whole window.
fn win_max_of(geom: &[imparo_kv::LayerStateGeom]) -> usize {
    geom.iter()
        .filter_map(|g| match g.kind {
            imparo_kv::StateKind::Window { window, .. } => Some(window),
            imparo_kv::StateKind::Full => None,
        })
        .max()
        .unwrap_or(0)
}

/// Bound how many boundaries a manifest names, without orphaning one that is still
/// named.
///
/// A link covers `[from, boundary)`, so dropping the oldest entry can leave a NEWER
/// boundary unable to reach back a whole window -- it would still be named, and the
/// restore would have to discover the hole. Keep the newest `cap` boundaries as roots
/// and every ancestor their chains still need; drop the rest.
fn cap_ckpts(kept: &mut Vec<imparo_kv::store::Ckpt>, cap: usize, win_max: usize) {
    if kept.len() <= cap {
        return;
    }
    let roots: Vec<u64> = kept.iter().rev().take(cap).map(|c| c.boundary).collect();
    let mut keep: std::collections::BTreeSet<u64> = roots.iter().copied().collect();
    for r in &roots {
        let floor = r.saturating_sub(win_max as u64);
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

fn recurrent_ok(model: &dyn KvPoolMember, recurrent: &[u8]) -> bool {
    model.plan().recurrent_elems() == 0 || !recurrent.is_empty()
}

fn clone_windows(s: &KvState) -> KvState {
    KvState {
        boundary: s.boundary,
        recurrent: s.recurrent.clone(),
        full: Vec::new(),
        window: s
            .window
            .iter()
            .map(|l| imparo_kv::KvLayerState {
                layer: l.layer,
                base_pos: l.base_pos,
                positions: l.positions,
                k: l.k.clone(),
                v: l.v.clone(),
            })
            .collect(),
    }
}
