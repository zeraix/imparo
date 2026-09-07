//! The COMMON spill/restore engine: capture canonical KV state through the Backend
//! trait and put it back, driven entirely by a per-layer geometry descriptor the
//! model's plan declares. No model knowledge lives here — a new model contributes
//! only its descriptor (docs/unified-kv-pool.md: "the pool's contract with the plan
//! is: per layer, a state KIND and its geometry").
//!
//! Canonical form is position-major per layer: rings are unwrapped at capture and
//! rewrapped at restore, so nothing ring-shaped or tuned reaches the bytes.

use imparo_backend::{Backend, BufId};

/// How one layer's growing-or-bounded state behaves. The three kinds and their
/// rules come from the design's state-kinds table; recurrent/conv arrives with the
/// first model that carries it (a whole-state snapshot, same descriptor shape).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateKind {
    /// Grows with context; canonical = linear positions [0, boundary).
    Full,
    /// Ring-buffered sliding window; canonical = the window tail, unwrapped.
    Window { window: usize, ring: usize },
}

/// How a model's UNPOOLED state is checkpointed. The two rows of the design's
/// state-kinds table (docs/unified-kv-pool.md, "State kinds"), named:
///
/// ```text
///   SWA window       checkpointed as DELTAS -- min(gap, window) rows -- chained,
///                    re-anchored when a gap exceeds the window
///   recurrent / GDN  checkpoint is a whole-state SNAPSHOT; advances only by replay
/// ```
///
/// A MODEL, NOT A LAYER, because a checkpoint covers every layer at once and
/// `KvState` carries both a `window` and a `recurrent` section. A hybrid that has
/// windowed layers AND recurrent state is `WindowDeltas`: the window rows still have
/// to accumulate, and the recurrent blob rides in each link.
///
/// This exists because the alternative -- asking `win_max == 0` at each decision --
/// was tried and is how five separate policies came to read "there is no window" as
/// "there is nothing to do" rather than "there is a different thing to do": the
/// ancestor walk, the step collapse, a checkpoint's predecessor, the spill boundary's
/// grid, and the persistence stop. One name, asked five times, cannot drift.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointShape {
    /// The widest window any layer keeps; how far back a chain must reach.
    WindowDeltas { win_max: usize },
    /// Every checkpoint is complete on its own.
    Snapshots,
}

impl CheckpointShape {
    /// Read the shape off the layer geometry: windowed layers mean deltas.
    #[must_use]
    pub fn of(geom: &[LayerStateGeom]) -> Self {
        match geom
            .iter()
            .filter_map(|g| match g.kind {
                StateKind::Window { window, .. } => Some(window),
                StateKind::Full => None,
            })
            .max()
        {
            Some(win_max) => Self::WindowDeltas { win_max },
            None => Self::Snapshots,
        }
    }

    /// How far back a chain must reach from `boundary`, or `None` when checkpoints do
    /// not chain and there is no walk to do.
    ///
    /// `None` rather than `boundary`, so a caller branches instead of running the
    /// window's arithmetic on a value chosen to neutralise it. A snapshot path that
    /// reads as "the window walk, disabled" is how a walk that should not have run at
    /// all kept running: `floor == boundary` silently stopped it after one link.
    #[must_use]
    pub fn chain_floor(self, boundary: usize) -> Option<usize> {
        match self {
            Self::WindowDeltas { win_max } => Some(boundary.saturating_sub(win_max)),
            Self::Snapshots => None,
        }
    }

    /// Whether a checkpoint may name a predecessor at all.
    #[must_use]
    pub fn chains(self) -> bool {
        matches!(self, Self::WindowDeltas { .. })
    }
}

/// One layer's capture geometry, derived by the model from its plan + KV types.
#[derive(Clone, Copy, Debug)]
pub struct LayerStateGeom {
    pub layer: u32,
    pub kind: StateKind,
    /// Bytes per position in the K cache (quant-type row stride).
    pub k_stride: usize,
    /// Bytes per position in the V cache.
    pub v_stride: usize,
}

/// One layer's captured state: canonical position-major K and V bytes.
#[derive(Clone)]
pub struct KvLayerState {
    pub layer: u32,
    /// First position these bytes cover.
    pub base_pos: usize,
    /// Positions covered.
    pub positions: usize,
    pub k: Vec<u8>,
    pub v: Vec<u8>,
}

/// The three numbers a pool member tracks, together because they are one fact in three
/// parts: what the cache can ever hold, what it has allocated, and what it holds now.
///
/// They used to be two fields inside KvCache -- which otherwise holds the CPU backend's
/// actual cache arrays -- and one field beside it. That is why exposing three numbers took
/// four accessors: the CPU arrays and the device counters are different things and had been
/// merged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvRuntime {
    /// The context bound: the cache can never hold more.
    pub capacity: usize,
    /// Slots currently allocated on the device.
    pub slots: usize,
    /// Positions currently held.
    pub filled: usize,
}

/// The bounded set, which is what gets captured rather than pooled.
#[must_use]
pub fn bounded_geometry(geom: &[LayerStateGeom]) -> Vec<LayerStateGeom> {
    geom.iter()
        .filter(|g| !matches!(g.kind, StateKind::Full))
        .copied()
        .collect()
}

/// The tightest ring slack across windowed layers: a boundary is capturable from the rings
/// while `filled - boundary <= slack`. The pool uses it to decide whether a checkpoint can
/// be DEFERRED to switch-out or has to be taken eagerly mid-prefill.
#[must_use]
pub fn window_slack(geom: &[LayerStateGeom]) -> usize {
    geom.iter()
        .filter_map(|g| match g.kind {
            StateKind::Window { window, ring } => Some(ring - window),
            StateKind::Full => None,
        })
        .min()
        .unwrap_or(usize::MAX)
}

/// The pooled set: layers whose state grows with context.
#[must_use]
pub fn full_layers(geom: &[LayerStateGeom]) -> Vec<u32> {
    geom.iter()
        .filter(|g| matches!(g.kind, StateKind::Full))
        .map(|g| g.layer)
        .collect()
}

/// Captured cache state at a unit-aligned boundary.
pub struct KvState {
    /// The resume position; the tail beyond it is reprocessed from tokens.
    pub boundary: usize,
    /// Full-attention layers: positions [0, boundary).
    pub full: Vec<KvLayerState>,
    /// Windowed layers: the window tail [boundary - window, boundary).
    pub window: Vec<KvLayerState>,
    /// The whole per-conversation RECURRENT buffer at `boundary`, little-endian f32.
    /// Empty for a model with no recurrent layers.
    ///
    /// A WHOLE-BUFFER snapshot, not position-indexed: a convolution history is
    /// `n_embd * (l_cache - 1)` values whatever the context length, so there is nothing
    /// to slice and no ring to unwrap. It is here for the same reason the windowed
    /// layers are -- neither can be rebuilt by recompute or by truncation, so both are
    /// checkpointed (docs/unified-kv-pool.md, the state-kinds table).
    pub recurrent: Vec<u8>,
}

/// Reads a device recurrent buffer as little-endian f32 bytes.
///
/// `from` is `BufId::Recur` for the live state and `BufId::RecurSnap` for a snapshot the
/// batch took at a boundary inside itself.
///
/// `elems` is 0 for a model without one, and then this is empty and every path below is
/// a no-op -- which is why gemma4 sees no change at all.
#[must_use]
pub fn capture_recurrent(be: &dyn Backend, elems: usize, from: BufId) -> Vec<u8> {
    if elems == 0 {
        return Vec::new();
    }
    let mut f = vec![0.0_f32; elems];
    be.read(from, 0, &mut f);
    let mut out = Vec::with_capacity(elems * 4);
    for v in f {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Writes a captured recurrent state back.
///
/// A blob whose length disagrees with the buffer is REFUSED rather than truncated: it
/// means the checkpoint came from a different model, and half a state restored is a
/// conversation that continues with plausible wrong numbers.
pub fn restore_recurrent(
    be: &dyn Backend,
    elems: usize,
    blob: &[u8],
) -> Result<(), String> {
    if elems == 0 && blob.is_empty() {
        return Ok(());
    }
    if blob.len() != elems * 4 {
        return Err(format!(
            "recurrent state: checkpoint has {} bytes, this model needs {}",
            blob.len(),
            elems * 4
        ));
    }
    let f: Vec<f32> = blob
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    be.write(BufId::Recur, 0, &f);
    Ok(())
}

/// Captures canonical state at `boundary` (device idle; the caller aligned the
/// boundary to the unit grid and its ring slack).
#[must_use]
pub fn capture(
    be: &dyn Backend,
    geom: &[LayerStateGeom],
    boundary: usize,
    recurrent: Vec<u8>,
) -> KvState {
    let mut full = Vec::new();
    let mut window = Vec::new();
    for g in geom {
        match g.kind {
            StateKind::Full => {
                let mut k = vec![0u8; boundary * g.k_stride];
                let mut v = vec![0u8; boundary * g.v_stride];
                be.read_kv_bytes(g.layer, false, 0, &mut k);
                be.read_kv_bytes(g.layer, true, 0, &mut v);
                full.push(KvLayerState {
                    layer: g.layer,
                    base_pos: 0,
                    positions: boundary,
                    k,
                    v,
                });
            }
            StateKind::Window { window: win, ring } => {
                let base = boundary.saturating_sub(win);
                let n = boundary - base;
                let mut k = vec![0u8; n * g.k_stride];
                let mut v = vec![0u8; n * g.v_stride];
                let mask = ring - 1;
                for (i, p) in (base..boundary).enumerate() {
                    let slot = p & mask;
                    be.read_kv_bytes(
                        g.layer,
                        false,
                        (slot * g.k_stride) as u64,
                        &mut k[i * g.k_stride..(i + 1) * g.k_stride],
                    );
                    be.read_kv_bytes(
                        g.layer,
                        true,
                        (slot * g.v_stride) as u64,
                        &mut v[i * g.v_stride..(i + 1) * g.v_stride],
                    );
                }
                window.push(KvLayerState {
                    layer: g.layer,
                    base_pos: base,
                    positions: n,
                    k,
                    v,
                });
            }
        }
    }
    KvState {
        boundary,
        full,
        window,
        recurrent,
    }
}

/// A windowed-state DELTA: per windowed layer, the rows `[max(from, boundary -
/// window), boundary)`. Cost is `min(gap, window)` positions (design: "Checkpoint
/// size"); a delta whose gap covers the whole window is self-sufficient (an
/// anchor). `full` layers are never delta'd here — their bytes live in pooled
/// units already.
#[derive(Clone)]
pub struct KvDelta {
    pub boundary: usize,
    /// Predecessor checkpoint boundary this delta chains from (rows below it are
    /// the predecessor's job). `0` makes every layer an anchor.
    ///
    /// Must be at most the predecessor's boundary. An OVERLAP is harmless (the
    /// newer link simply re-states rows the older one also has); a GAP is fatal,
    /// because `assemble_chain` walks down to `boundary - window` and stops at the
    /// first position no link covers.
    pub from: usize,
    pub window: Vec<KvLayerState>,
    /// Full-attention rows for `[floor_unit(boundary), boundary)` -- the part of
    /// the prefix that lies ABOVE the last whole content unit. Empty whenever the
    /// boundary is unit-aligned, which is every checkpoint the engine takes on its
    /// own.
    ///
    /// WHY A CHECKPOINT CARRIES ANY FULL-ATTENTION ROWS AT ALL, when the whole point
    /// of the pool is that full KV is written once and never copied: a turn boundary
    /// lands wherever the user's message starts, and extents are cut where REQUESTS
    /// ended. Extents stay the unit of SHARING; these rows are whatever lies above the
    /// last one:
    ///
    /// ```text
    ///   extents [0, ue)   pooled, shared by hash, never rewritten
    ///   rows [ue, B)      carried here, written into the borrower's OWN tail blocks
    ///                     when it resumes
    /// ```
    ///
    /// `ue` is the last cut at or below `B`, which is why `UEND` is recorded in the
    /// blob: no constant divides it. Under the shipped layout a checkpoint lands ON a
    /// cut and this is empty -- which is the point of cutting where the request ends.
    /// The alternative is rounding `B` down to `ue`, which reprocesses those positions
    /// on every restore, and for a recurrent model they must go back through the mixer.
    pub tail: Vec<KvLayerState>,
    /// The whole recurrent buffer AT `boundary`, little-endian f32; empty for a model
    /// with none.
    ///
    /// NOT a delta of anything, and deliberately not chained: a convolution history is
    /// one buffer at one position, so the newest link carries the whole of it and the
    /// older links' copies are never read. It is small -- 0.34 MiB for LFM2.5 against
    /// the windowed rows beside it -- and the alternative is a chain that assembles a
    /// state the pool must then REFUSE for want of it.
    pub recurrent: Vec<u8>,
}

/// Captures a windowed-layer delta at `boundary`, chained from a previous
/// checkpoint at `from` (device idle, boundary unit-aligned, `from <= boundary`).
#[must_use]
pub fn capture_delta(
    be: &dyn Backend,
    geom: &[LayerStateGeom],
    recurrent: Vec<u8>,
    boundary: usize,
    from: usize,
) -> KvDelta {
    let mut window = Vec::new();
    for g in geom {
        let StateKind::Window { window: win, ring } = g.kind else {
            continue;
        };
        let base = boundary.saturating_sub(win).max(from);
        let n = boundary - base;
        let mut k = vec![0u8; n * g.k_stride];
        let mut v = vec![0u8; n * g.v_stride];
        let mask = ring - 1;
        for (i, p) in (base..boundary).enumerate() {
            let slot = p & mask;
            be.read_kv_bytes(
                g.layer,
                false,
                (slot * g.k_stride) as u64,
                &mut k[i * g.k_stride..(i + 1) * g.k_stride],
            );
            be.read_kv_bytes(
                g.layer,
                true,
                (slot * g.v_stride) as u64,
                &mut v[i * g.v_stride..(i + 1) * g.v_stride],
            );
        }
        window.push(KvLayerState {
            layer: g.layer,
            base_pos: base,
            positions: n,
            k,
            v,
        });
    }
    KvDelta {
        boundary,
        from,
        window,
        // Filled by the caller that owns the block tables: these rows live in
        // POOLED blocks, and `read_kv_bytes` addresses a layer's buffer
        // physically, so only the pool can say where position p actually is.
        tail: Vec::new(),
        recurrent,
    }
}

/// Assembles the full windowed state at the NEWEST delta's boundary from a chain
/// ordered newest-first (each entry's `from` equals the next entry's `boundary`).
/// Returns None when the chain does not cover some layer's window — the caller
/// treats that as "no checkpoint" (an evicted link).
#[must_use]
pub fn assemble_chain(geom: &[LayerStateGeom], chain: &[&KvDelta]) -> Option<KvState> {
    let newest = chain.first()?;
    let boundary = newest.boundary;
    let mut window = Vec::new();
    for g in geom {
        let StateKind::Window { window: win, .. } = g.kind else {
            continue;
        };
        let base = boundary.saturating_sub(win);
        let n = boundary - base;
        let mut k = vec![0u8; n * g.k_stride];
        let mut v = vec![0u8; n * g.v_stride];
        // Walk newest -> older; each delta contributes its own slice. `need`
        // descends to `base`; a gap (missing link) leaves it uncovered.
        let mut need = boundary;
        for d in chain {
            if need <= base {
                break;
            }
            let ls = d.window.iter().find(|l| l.layer == g.layer)?;
            if ls.base_pos + ls.positions < need {
                return None; // chain out of order or truncated capture
            }
            let lo = ls.base_pos.max(base);
            for p in lo..need {
                let src = p - ls.base_pos;
                let dst = p - base;
                k[dst * g.k_stride..(dst + 1) * g.k_stride]
                    .copy_from_slice(&ls.k[src * g.k_stride..(src + 1) * g.k_stride]);
                v[dst * g.v_stride..(dst + 1) * g.v_stride]
                    .copy_from_slice(&ls.v[src * g.v_stride..(src + 1) * g.v_stride]);
            }
            need = lo;
        }
        if need > base {
            return None;
        }
        window.push(KvLayerState {
            layer: g.layer,
            base_pos: base,
            positions: n,
            k,
            v,
        });
    }
    Some(KvState {
        boundary,
        full: Vec::new(),
        window,
        // From the NEWEST link only. A convolution history is one buffer at one
        // position, so it is not chained -- the older links' copies describe earlier
        // positions and are never read.
        recurrent: newest.recurrent.clone(),
    })
}

/// Writes a captured state back (device idle; caller has grown the cache to
/// `state.boundary` first and sets its own filled bookkeeping after).
pub fn restore(
    be: &dyn Backend,
    geom: &[LayerStateGeom],
    state: &KvState,
    recurrent_elems: usize,
) -> Result<(), String> {
    restore_recurrent(be, recurrent_elems, &state.recurrent)?;
    let find = |layer: u32| geom.iter().find(|g| g.layer == layer);
    for ls in &state.full {
        let Some(g) = find(ls.layer) else { continue };
        be.write_kv_bytes(ls.layer, false, (ls.base_pos * g.k_stride) as u64, &ls.k);
        be.write_kv_bytes(ls.layer, true, (ls.base_pos * g.v_stride) as u64, &ls.v);
    }
    for ls in &state.window {
        let Some(g) = find(ls.layer) else { continue };
        let StateKind::Window { ring, .. } = g.kind else {
            continue;
        };
        let mask = ring - 1;
        for i in 0..ls.positions {
            let slot = (ls.base_pos + i) & mask;
            be.write_kv_bytes(
                ls.layer,
                false,
                (slot * g.k_stride) as u64,
                &ls.k[i * g.k_stride..(i + 1) * g.k_stride],
            );
            be.write_kv_bytes(
                ls.layer,
                true,
                (slot * g.v_stride) as u64,
                &ls.v[i * g.v_stride..(i + 1) * g.v_stride],
            );
        }
    }
    Ok(())
}

// --- canonical blob serialization (unit blobs + checkpoint blob) ---

const UNIT_MAGIC: u32 = 0x494B_5655; // "IKVU"
const CKPT_MAGIC: u32 = 0x494B_5643; // "IKVC"

// THE FRAME, and why it is not a list of fields in a fixed order.
//
// ```text
//   u32 magic          "IKVU" / "IKVC"
//   u32 format         this number
//   repeated:
//     u32 tag          four ASCII bytes naming the section
//     u64 len
//     u8[len]          the section's own encoding
// ```
//
// A reader takes the sections it knows, IGNORES the ones it does not, and fails only
// when a section it requires is absent -- naming which. That makes the format number
// mean one thing: **the meaning of an existing section changed**. Adding a section
// never bumps it, and adding one is what actually happens.
//
// Written after bumping a positional version twice in one day, for two ADDITIONS: the
// recurrent state, then the sub-unit tail rows. Each bump invalidated every file in
// every store, for fields an older reader could simply have skipped.
//
// What is queued behind this, as the worked example: media. When a vision model lands,
// a checkpoint gains a `MEDI` section describing the images its positions cover, and
// the manifest gains their content ids -- because image placeholder tokens all carry
// the SAME token id, so two different pictures at the same span hash identically and
// would dedup onto each other's KV. Under this frame that is one new tag and no
// version change; under the old one it was a third bump.
const FORMAT: u32 = 2;

/// One section: `tag`, length, payload.
fn push_section(out: &mut Vec<u8>, tag: [u8; 4], payload: &[u8]) {
    out.extend_from_slice(&tag);
    push_u64(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

/// Splits a blob into its sections, checking the magic and refusing a format from the
/// future (which this build cannot know how to read).
fn sections<'a>(
    blob: &'a [u8],
    magic: u32,
    what: &str,
) -> Result<std::collections::BTreeMap<[u8; 4], &'a [u8]>, String> {
    let mut r = Reader(blob);
    if r.u32()? != magic {
        return Err(format!("{what}: bad magic"));
    }
    let got = r.u32()?;
    if got == 0 || got > FORMAT {
        return Err(format!(
            "{what}: format {got}, this build reads 1..={FORMAT}"
        ));
    }
    let mut out = std::collections::BTreeMap::new();
    while !r.0.is_empty() {
        let tag: [u8; 4] = r.bytes(4)?.try_into().map_err(|_| "short tag")?;
        let len = r.u64()? as usize;
        out.insert(tag, r.bytes(len)?);
    }
    Ok(out)
}

/// A section this build needs, by name, so a missing one says which.
fn need<'a>(
    sec: &std::collections::BTreeMap<[u8; 4], &'a [u8]>,
    tag: [u8; 4],
    what: &str,
) -> Result<&'a [u8], String> {
    sec.get(&tag).copied().ok_or_else(|| {
        format!(
            "{what}: no {} section",
            std::str::from_utf8(&tag).unwrap_or("????")
        )
    })
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32, String> {
        let (a, b) = self.0.split_at_checked(4).ok_or("short read")?;
        self.0 = b;
        Ok(u32::from_le_bytes(a.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let (a, b) = self.0.split_at_checked(8).ok_or("short read")?;
        self.0 = b;
        Ok(u64::from_le_bytes(a.try_into().unwrap()))
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        let (a, b) = self.0.split_at_checked(n).ok_or("short read")?;
        self.0 = b;
        Ok(a)
    }
}

/// Slices a captured state's full-attention layers into one blob per extent.
///
/// `cuts` are absolute end positions, ascending, the last at most `state.boundary`.
/// Extent `i` covers `[cuts[i - 1], cuts[i])`, counting from 0 for the first. Each
/// blob names its own extent, so the reader does not have to know the cut to put
/// the rows back where they came from -- which is the point: the cut is where the
/// requests ended, not a constant.
#[must_use]
pub fn unit_blobs_at(state: &KvState, cuts: &[usize]) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(cuts.len());
    // Where these rows actually begin. A write-through of ONE extent holds only that
    // extent, so assuming 0 makes every blob claim to start at the beginning of the
    // conversation -- which reads back as extents that all overlap.
    // From the first layer, and every layer is then sliced by it. A capture puts all
    // full layers at the same base, so this holds -- but it is an assumption the type
    // does not carry, and a layer that disagreed would be sliced at another layer's
    // offset and write the wrong rows under the right hash. Cheap to check, silent to
    // get wrong.
    debug_assert!(
        state
            .full
            .windows(2)
            .all(|w| w[0].base_pos == w[1].base_pos),
        "full layers disagree about where their rows start"
    );
    let mut start = state.full.first().map_or(0, |ls| ls.base_pos);
    for &end in cuts {
        if end <= start || end > state.boundary {
            break;
        }
        let mut full = Vec::new();
        push_u32(&mut full, state.full.len() as u32);
        for ls in &state.full {
            let ks = ls.k.len() / ls.positions;
            let vs = ls.v.len() / ls.positions;
            let lo = start - ls.base_pos;
            let hi = end - ls.base_pos;
            push_u32(&mut full, ls.layer);
            push_u64(&mut full, ks as u64);
            push_u64(&mut full, vs as u64);
            full.extend_from_slice(&ls.k[lo * ks..hi * ks]);
            full.extend_from_slice(&ls.v[lo * vs..hi * vs]);
        }
        let mut blob = Vec::new();
        push_u32(&mut blob, UNIT_MAGIC);
        push_u32(&mut blob, FORMAT);
        let mut extent = Vec::new();
        push_u64(&mut extent, start as u64);
        push_u64(&mut extent, end as u64);
        push_section(&mut blob, *b"EXTN", &extent);
        push_section(&mut blob, *b"FULL", &full);
        out.push(blob);
        start = end;
    }
    out
}

/// `unit_blobs_at` on the resident tiling. TEST ONLY.
///
/// Nothing on the disk side cuts blobs on that tiling: every writer passes the
/// extents its requests produced. Kept because the round-trip tests want a whole
/// state sliced somehow and do not care where.
#[cfg(test)]
#[must_use]
pub fn unit_blobs(state: &KvState) -> Vec<Vec<u8>> {
    unit_blobs_at(state, &crate::identity::resident_bounds(state.boundary))
}

/// One link of the chain, as the disk holds it: the window rows this delta OWNS --
/// `[from, boundary)` -- rather than the whole window.
///
/// The resident chain has always been deltas (`min(gap, window)` rows per turn); the
/// disk wrote a full window per boundary, which is the same state said the expensive
/// way. Writing the link instead makes a turn's checkpoint cost a turn's slice, and the
/// reader walks `from` down to `boundary - window` exactly as `assemble_chain` does in
/// memory. The recurrent half is not a delta -- it is a whole state, because a
/// convolution has no per-position history to slice.
#[must_use]
pub fn delta_blob(delta: &KvDelta) -> Vec<u8> {
    let mut out = Vec::new();
    push_u32(&mut out, CKPT_MAGIC);
    push_u32(&mut out, FORMAT);
    push_section(&mut out, *b"BNDY", &(delta.boundary as u64).to_le_bytes());
    push_section(&mut out, *b"FROM", &(delta.from as u64).to_le_bytes());
    let mut win = Vec::new();
    push_layers(&mut win, &delta.window);
    push_section(&mut out, *b"WNDW", &win);
    push_section(&mut out, *b"RECU", &delta.recurrent);
    let mut tail = Vec::new();
    push_layers(&mut tail, &delta.tail);
    push_section(&mut out, *b"TAIL", &tail);
    out
}

/// Read one link back.
///
/// # Errors
/// When the blob is not a checkpoint, or a section is missing or malformed.
pub fn delta_from_blob(blob: &[u8]) -> Result<KvDelta, String> {
    let sec = sections(blob, CKPT_MAGIC, "checkpoint")?;
    let at = |name: &[u8; 4]| -> Result<usize, String> {
        Ok(u64::from_le_bytes(
            need(&sec, *name, "checkpoint")?
                .try_into()
                .map_err(|_| "checkpoint: position field is not 8 bytes")?,
        ) as usize)
    };
    let boundary = at(b"BNDY")?;
    // A blob written before links carried one covers the whole window on its own, which
    // is what an anchor is. So this reads both shapes and the caller needs to know only
    // about chains.
    let from = at(b"FROM").unwrap_or(0);
    let window = read_layers(&mut Reader(need(&sec, *b"WNDW", "checkpoint")?))?;
    let tail = read_layers(&mut Reader(need(&sec, *b"TAIL", "checkpoint")?))?;
    let recurrent = need(&sec, *b"RECU", "checkpoint")
        .map(<[u8]>::to_vec)
        .unwrap_or_default();
    Ok(KvDelta {
        boundary,
        from,
        window,
        tail,
        recurrent,
    })
}

/// A whole captured state, expressed as the ONE blob format: an anchor link.
///
/// There used to be a second format here -- `checkpoint_blob`, a whole-window blob with
/// a `UEND` section -- written by `--spill` and the server's non-pool commit, while the
/// pool wrote chain links. Both landed in `ckpt/<hash>` and both were named by
/// `Manifest::ckpts[].blob`, so a reader was handed one and could not ask which it got.
/// The chain IS the format now, and a whole state is simply the degenerate chain: one
/// link, `from = 0`, covering its whole window.
///
/// `UEND` is not needed and is not written. The reader derives the cut from the tail's
/// own `base_pos`, which `tail_rows` sets to exactly `unit_end`; an empty tail means the
/// boundary IS the cut, which is the common case under the shipped layout.
#[must_use]
pub fn anchor_link(state: &KvState, unit_end: usize) -> KvDelta {
    KvDelta {
        boundary: state.boundary,
        // An anchor owes nothing to an older link.
        from: 0,
        window: state.window.clone(),
        // Only the part ABOVE the last extent. `state.full` may describe the whole
        // prefix (that is what a non-pool capture produces) and the extents already
        // carry everything below `unit_end`; writing it twice would put the same rows
        // in the blob and in the extent files.
        tail: tail_rows(&state.full, state.boundary, unit_end),
        recurrent: state.recurrent.clone(),
    }
}

/// The rows of `full` between `unit_end` and `boundary` -- what the units below do
/// not already carry.
///
/// `unit_end` is passed in rather than derived: the cut is where a stream's requests
/// ended, so no constant divides it. Under the disk layout it EQUALS `boundary` and
/// this returns nothing, which is the point of cutting where the checkpoint lands.
#[must_use]
pub fn tail_rows(
    full: &[KvLayerState],
    boundary: usize,
    unit_end: usize,
) -> Vec<KvLayerState> {
    full.iter()
        .filter_map(|ls| {
            let lo = unit_end.max(ls.base_pos);
            let hi = boundary.min(ls.base_pos + ls.positions);
            if hi <= lo {
                return None;
            }
            let ks = ls.k.len() / ls.positions;
            let vs = ls.v.len() / ls.positions;
            let (i, n) = (lo - ls.base_pos, hi - lo);
            Some(KvLayerState {
                layer: ls.layer,
                base_pos: lo,
                positions: n,
                k: ls.k[i * ks..(i + n) * ks].to_vec(),
                v: ls.v[i * vs..(i + n) * vs].to_vec(),
            })
        })
        .collect()
}

fn push_layers(out: &mut Vec<u8>, layers: &[KvLayerState]) {
    push_u32(out, layers.len() as u32);
    for ls in layers {
        push_u32(out, ls.layer);
        push_u64(out, ls.base_pos as u64);
        push_u64(out, ls.positions as u64);
        push_u64(out, ls.k.len() as u64);
        push_u64(out, ls.v.len() as u64);
        out.extend_from_slice(&ls.k);
        out.extend_from_slice(&ls.v);
    }
}

fn read_layers(r: &mut Reader<'_>) -> Result<Vec<KvLayerState>, String> {
    let n = r.u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = r.u32()?;
        let base_pos = r.u64()? as usize;
        let positions = r.u64()? as usize;
        let klen = r.u64()? as usize;
        let vlen = r.u64()? as usize;
        let k = r.bytes(klen)?.to_vec();
        let v = r.bytes(vlen)?.to_vec();
        out.push(KvLayerState {
            layer,
            base_pos,
            positions,
            k,
            v,
        });
    }
    Ok(out)
}

/// The same state, built from a CHAIN of links rather than one whole-window blob.
///
/// The disk holds what the resident pool holds: one link per turn boundary, each owning
/// `[from, boundary)`. `chain` is newest first; `assemble_chain` walks it down until the
/// window is covered and answers None if a link is missing, so a caller can fall back to
/// a shallower boundary rather than restore a hole.
///
/// # Errors
/// When the units disagree with the boundary, or the chain does not cover the window.
pub fn state_from_units_and_chain(
    units: &[Vec<u8>],
    geom: &[LayerStateGeom],
    chain: &[&KvDelta],
) -> Result<KvState, String> {
    let newest = chain.first().ok_or("checkpoint chain is empty")?;
    let boundary = newest.boundary;
    // Where this link counts its tail from, which is where the extents must reach.
    // A tail names its own start; an EMPTY tail means the cut IS the boundary, which
    // is what the disk layout produces once extents end where checkpoints land.
    let unit_end = newest.tail.first().map_or(boundary, |ls| ls.base_pos);
    let full = full_from_units(units, unit_end)?;
    let mut st = assemble_chain(geom, chain)
        .ok_or("checkpoint chain does not reach back a whole window")?;
    st.full = full;
    append_tail(&mut st.full, &newest.tail)?;
    Ok(st)
}

/// Extents -> one contiguous run per layer covering `[0, last cut)`.
///
/// Each blob names the extent it holds, so this checks they abut from 0 rather than
/// multiplying a count by a constant. Returns the rows and the position they reach.
fn full_from_units(
    units: &[Vec<u8>],
    unit_end: usize,
) -> Result<Vec<KvLayerState>, String> {
    let mut full: Vec<KvLayerState> = Vec::new();
    let mut at = 0_usize;
    for (u, blob) in units.iter().enumerate() {
        let what = format!("unit {u}");
        let usec = sections(blob, UNIT_MAGIC, &what)?;
        let mut er = Reader(need(&usec, *b"EXTN", &what)?);
        let (from, to) = (er.u64()? as usize, er.u64()? as usize);
        if from != at {
            return Err(format!(
                "unit {u} covers [{from}, {to}) but the extents before it reach {at}"
            ));
        }
        let rows = to.checked_sub(from).filter(|r| *r > 0).ok_or_else(|| {
            format!("unit {u} covers [{from}, {to}), which is not a forward extent")
        })?;
        let mut r = Reader(need(&usec, *b"FULL", &what)?);
        let n_layers = r.u32()? as usize;
        if u == 0 {
            full = Vec::with_capacity(n_layers);
        } else if n_layers != full.len() {
            return Err(format!("unit {u}: layer count changed"));
        }
        for li in 0..n_layers {
            let layer = r.u32()?;
            let ks = r.u64()? as usize;
            let vs = r.u64()? as usize;
            let k = r.bytes(rows * ks)?.to_vec();
            let v = r.bytes(rows * vs)?.to_vec();
            if u == 0 {
                full.push(KvLayerState {
                    layer,
                    base_pos: 0,
                    positions: 0,
                    k: Vec::new(),
                    v: Vec::new(),
                });
            } else if full[li].layer != layer {
                return Err(format!("unit {u}: layer order changed"));
            }
            full[li].k.extend_from_slice(&k);
            full[li].v.extend_from_slice(&v);
            full[li].positions += rows;
        }
        at = to;
    }
    if at != unit_end {
        return Err(format!(
            "the extents reach {at}, the checkpoint counts its tail from {unit_end}"
        ));
    }
    Ok(full)
}

/// Appends the sub-unit rows so each layer is ONE run covering `[0, boundary)`: the
/// restore slices by absolute position, and a run with a hole restores garbage above
/// the units.
fn append_tail(full: &mut [KvLayerState], tail: &[KvLayerState]) -> Result<(), String> {
    for ls in tail {
        let Some(dst) = full.iter_mut().find(|d| d.layer == ls.layer) else {
            return Err(format!(
                "checkpoint tail names layer {}, the units do not",
                ls.layer
            ));
        };
        if dst.base_pos + dst.positions != ls.base_pos {
            return Err(format!(
                "checkpoint tail for layer {} starts at {}, the units end at {}",
                ls.layer,
                ls.base_pos,
                dst.base_pos + dst.positions
            ));
        }
        dst.k.extend_from_slice(&ls.k);
        dst.v.extend_from_slice(&ls.v);
        dst.positions += ls.positions;
    }
    Ok(())
}

/// Rebuilds a restorable state from extent blobs plus a CHAIN of link blobs.
///
/// `links` newest first, as `Manifest::ckpts` names them descending. One link with
/// `from == 0` is an anchor and covers its window alone; that is what `--spill` and the
/// server's non-pool commit write, and it is the same format the pool writes per turn.
/// Blobs written before links carried a `FROM` are still read: `delta_from_blob`
/// defaults it to 0, which is what an anchor means.
///
/// # Errors
/// Refuses malformed blobs, version mismatches, and geometry that disagrees
/// between units.
pub fn state_from_blobs(
    units: &[Vec<u8>],
    checkpoint: &[u8],
) -> Result<KvState, String> {
    let sec = sections(checkpoint, CKPT_MAGIC, "checkpoint")?;
    let boundary = u64::from_le_bytes(
        need(&sec, *b"BNDY", "checkpoint")?
            .try_into()
            .map_err(|_| "checkpoint: BNDY is not 8 bytes")?,
    ) as usize;
    // The units cover the prefix up to the last GRID step; the rest -- the distance
    // from that step to the boundary, under grid_tokens() positions -- rides in the
    // checkpoint's own tail rows. (Said 256 here, from the retired unit grid.)
    let window = read_layers(&mut Reader(need(&sec, *b"WNDW", "checkpoint")?))?;
    let recurrent = need(&sec, *b"RECU", "checkpoint")?.to_vec();
    let tail = read_layers(&mut Reader(need(&sec, *b"TAIL", "checkpoint")?))?;
    let unit_end = tail.first().map_or(boundary, |layer| layer.base_pos);
    let mut full = full_from_units(units, unit_end)?;
    append_tail(&mut full, &tail)?;
    Ok(KvState {
        boundary,
        full,
        window,
        recurrent,
    })
}

/// Refuses malformed blobs, a chain that does not reach back a whole window, and
/// extents that disagree with the cut the tail names.
pub fn state_from_links(
    geom: &[LayerStateGeom],
    units: &[Vec<u8>],
    links: &[Vec<u8>],
) -> Result<KvState, String> {
    let parsed: Vec<KvDelta> = links
        .iter()
        .map(|b| delta_from_blob(b))
        .collect::<Result<_, _>>()?;
    let newest = parsed.first().ok_or("restore: no checkpoint links")?;
    // The cut comes from the tail's own `base_pos` -- `tail_rows` sets it to exactly
    // `unit_end` -- so no `UEND` section is needed. An empty tail means the boundary IS
    // the cut, which is what cutting extents at request ends buys.
    let unit_end = newest.tail.first().map_or(newest.boundary, |l| l.base_pos);
    let refs: Vec<&KvDelta> = parsed.iter().collect();
    let mut st = assemble_chain(geom, &refs)
        .ok_or("restore: the link chain does not cover a whole window")?;
    st.full = full_from_units(units, unit_end)?;
    append_tail(&mut st.full, &newest.tail)?;
    Ok(st)
}

#[cfg(test)]
mod tests {
    /// ONE BLOB FORMAT. A whole state and a chain link are the same bytes.
    ///
    /// They were two: `checkpoint_blob` (whole window, with `UEND`) from `--spill` and
    /// the server's non-pool commit, and `delta_blob` (a link) from the pool -- both in
    /// `ckpt/<hash>`, both named by `Manifest::ckpts[].blob`, so a reader was handed one
    /// and could not ask which it got. A whole state is now the degenerate chain: one
    /// link with `from = 0`.
    #[test]
    fn a_whole_state_is_a_link_with_no_ancestor() {
        let s = state(2 * grid_tokens());
        let a = anchor_link(&s, s.boundary);
        assert_eq!(a.from, 0, "an anchor owes nothing to an older link");
        assert_eq!(a.boundary, s.boundary);
        // Written and read by the SAME pair the pool uses.
        let blob = delta_blob(&a);
        let back = delta_from_blob(&blob).expect("an anchor reads as a link");
        assert_eq!((back.boundary, back.from), (s.boundary, 0));
        assert_eq!(
            back.recurrent, s.recurrent,
            "the recurrent half must survive"
        );
        // And a real chain link goes through the very same reader.
        let d = delta(1, grid_tokens(), 2 * grid_tokens(), 128);
        assert_eq!(
            delta_from_blob(&delta_blob(&d)).unwrap().from,
            grid_tokens()
        );
    }

    use super::*;
    use crate::identity::grid_tokens;

    /// The one blob format, written from a whole captured state.
    fn ck_blob(s: &KvState, unit_end: usize) -> Vec<u8> {
        delta_blob(&anchor_link(s, unit_end))
    }

    /// A state's own capture geometry, so a test can read back what it wrote without a
    /// model. `assemble_chain` only walks Window layers; full ones ride in the extents.
    fn geom_of(s: &KvState) -> Vec<LayerStateGeom> {
        s.window
            .iter()
            .map(|l| LayerStateGeom {
                layer: l.layer,
                kind: StateKind::Window {
                    window: l.positions,
                    ring: l.positions.next_power_of_two(),
                },
                k_stride: l.k.len() / l.positions,
                v_stride: l.v.len() / l.positions,
            })
            .collect()
    }

    /// WHAT THE OLD READER COULD NOT DO: restore from a CHAIN, which is what a store
    /// the pool wrote actually holds.
    ///
    /// `state_from_blobs` wanted one whole-window blob with a `UEND` section. Handed a
    /// link it failed on the missing section -- so `imparo-forward --restore` could not
    /// read a pool-written store at all. One format means the chain reader is the only
    /// reader, and the extents still come from the extent blobs.
    #[test]
    fn a_multi_link_chain_and_its_extents_rebuild_the_whole_state() {
        let b = 2 * grid_tokens();
        let s = state(b);
        let units = unit_blobs(&s);
        let geom = vec![LayerStateGeom {
            layer: 1,
            kind: StateKind::Window {
                window: b,
                ring: b.next_power_of_two(),
            },
            k_stride: 4,
            v_stride: 4,
        }];
        // Newest owns [b/2, b); its ancestor owns [0, b/2). Neither covers the window
        // alone, which is the whole point -- an anchor would hide the defect.
        let newest = delta(1, b / 2, b, b / 2);
        let older = delta(1, 0, b / 2, b / 2);
        let links = vec![delta_blob(&newest), delta_blob(&older)];
        let back = state_from_links(&geom, &units, &links)
            .expect("a chain plus its extents must rebuild the state");
        assert_eq!(back.boundary, b);
        let w = &back.window[0];
        assert_eq!(
            (w.base_pos, w.positions),
            (0, b),
            "the window must be whole"
        );
        // Row value is its absolute position, so a mis-ordered assembly shows here.
        for p in 0..b {
            assert_eq!(
                w.k[p * 4],
                (p % 256) as u8,
                "position {p} came from the wrong link"
            );
        }
        // And the full-attention rows came from the extents, not from the links.
        assert_eq!(back.full.len(), s.full.len());
        assert_eq!(back.full[0].positions, b);
    }

    fn read_back(s: &KvState, units: &[Vec<u8>], ck: &[u8]) -> Result<KvState, String> {
        state_from_links(&geom_of(s), units, &[ck.to_vec()])
    }

    fn state(boundary: usize) -> KvState {
        let mk =
            |layer: u32, base: usize, n: usize, ks: usize, vs: usize| KvLayerState {
                layer,
                base_pos: base,
                positions: n,
                k: (0..n * ks).map(|i| (i % 251) as u8).collect(),
                v: (0..n * vs).map(|i| (i % 241) as u8).collect(),
            };
        KvState {
            boundary,
            full: vec![mk(3, 0, boundary, 64, 64), mk(9, 0, boundary, 64, 64)],
            window: vec![mk(1, boundary.saturating_sub(128), 128, 32, 32)],
            // Non-empty on purpose: the round trip must carry it, and a zero-length
            // blob would pass whether or not the format writes it at all.
            recurrent: (0..64_u32).flat_map(u32::to_le_bytes).collect(),
        }
    }

    #[test]
    fn round_trip() {
        let s = state(2 * grid_tokens());
        let units = unit_blobs(&s);
        assert_eq!(units.len(), 2);
        let ck = ck_blob(&s, s.boundary / grid_tokens() * grid_tokens());
        let back = read_back(&s, &units, &ck).unwrap();
        assert_eq!(back.boundary, s.boundary);
        for (a, b) in s.full.iter().zip(&back.full) {
            assert_eq!((a.layer, &a.k, &a.v), (b.layer, &b.k, &b.v));
        }
        for (a, b) in s.window.iter().zip(&back.window) {
            assert_eq!(
                (a.layer, a.base_pos, a.positions, &a.k, &a.v),
                (b.layer, b.base_pos, b.positions, &b.k, &b.v)
            );
        }
    }

    /// A checkpoint whose boundary is NOT on the unit grid: the units carry
    /// [0, 512) and the blob's tail carries [512, 600), and what comes back is one
    /// contiguous run of 600 positions per layer.
    #[test]
    fn blob_round_trip_off_the_grid() {
        let s = state(600);
        let units = unit_blobs(&s);
        assert_eq!(units.len(), 9, "600 tokens is nine whole 64-blocks plus 24");
        let back = read_back(
            &s,
            &units,
            &ck_blob(&s, s.boundary / grid_tokens() * grid_tokens()),
        )
        .unwrap();
        assert_eq!(back.boundary, 600);
        for (a, b) in s.full.iter().zip(&back.full) {
            assert_eq!(b.positions, 600);
            assert_eq!((a.layer, &a.k, &a.v), (b.layer, &b.k, &b.v));
        }
        assert_eq!(back.recurrent, s.recurrent);
    }

    /// The units alone are not enough: dropping the tail rows must be REFUSED, not
    /// restored as 512 positions labelled 600.
    #[test]
    fn a_v2_shaped_blob_is_refused() {
        let s = state(600);
        let units = unit_blobs(&s);
        let mut ck = ck_blob(&s, s.boundary / grid_tokens() * grid_tokens());
        // v2 wrote no tail; simulate one by truncating the layer count to zero.
        let n = ck.len();
        ck.truncate(n - 4);
        assert!(read_back(&s, &units, &ck).is_err());
    }

    fn delta(layer: u32, from: usize, boundary: usize, win: usize) -> KvDelta {
        let base = boundary.saturating_sub(win).max(from);
        let n = boundary - base;
        KvDelta {
            boundary,
            from,
            window: vec![KvLayerState {
                layer,
                base_pos: base,
                positions: n,
                // row value = its absolute position, so assembly order shows
                k: (base..boundary)
                    .flat_map(|p| [(p % 256) as u8; 4])
                    .collect(),
                v: (base..boundary)
                    .flat_map(|p| [(p % 256) as u8; 4])
                    .collect(),
            }],
            tail: Vec::new(),
            recurrent: Vec::new(),
        }
    }

    const GEOM: [LayerStateGeom; 1] = [LayerStateGeom {
        layer: 1,
        kind: StateKind::Window {
            window: 128,
            ring: 256,
        },
        k_stride: 4,
        v_stride: 4,
    }];

    /// Turn boundaries do not land on the unit grid, and the chain must not care:
    /// each link still starts exactly where the previous one ended.
    #[test]
    fn chain_assembles_across_unaligned_boundaries() {
        let a = delta(1, 0, 600, 128);
        let d1 = delta(1, 600, 673, 128);
        let d2 = delta(1, 673, 741, 128);
        let s = assemble_chain(&GEOM, &[&d2, &d1, &a]).unwrap();
        assert_eq!(s.boundary, 741);
        let ls = &s.window[0];
        assert_eq!((ls.base_pos, ls.positions), (741 - 128, 128));
        for (i, p) in (741 - 128..741).enumerate() {
            assert_eq!(
                ls.k[i * 4],
                (p % 256) as u8,
                "position {p} came from the wrong link"
            );
        }
    }

    /// A GAP is what breaks a chain -- an OVERLAP does not. `from` behind the
    /// predecessor's boundary re-states rows, which is harmless; ahead of it
    /// leaves positions nothing covers.
    #[test]
    fn an_overlapping_link_still_assembles() {
        let a = delta(1, 0, 600, 128);
        let d1 = delta(1, 550, 673, 128); // from < a.boundary: overlap
        let s = assemble_chain(&GEOM, &[&d1, &a]).unwrap();
        assert_eq!(s.window[0].positions, 128);
        // from AHEAD of the predecessor's boundary: nothing covers [600, 620).
        let gap = delta(1, 620, 673, 128);
        assert!(assemble_chain(&GEOM, &[&gap, &a]).is_none());
    }

    #[test]
    fn chain_assembles_to_the_window() {
        // anchor at 256 (gap 256 >= window), delta 256->320, delta 320->384
        let a = delta(1, 0, 256, 128);
        let d1 = delta(1, 256, 320, 128);
        let d2 = delta(1, 320, 384, 128);
        let s = assemble_chain(&GEOM, &[&d2, &d1, &a]).unwrap();
        assert_eq!(s.boundary, 384);
        let ls = &s.window[0];
        assert_eq!((ls.base_pos, ls.positions), (256, 128));
        // every row carries its absolute position
        for (i, p) in (256..384).enumerate() {
            assert_eq!(ls.k[i * 4], (p % 256) as u8);
        }
    }

    #[test]
    fn anchor_alone_serves() {
        let a = delta(1, 0, 256, 128);
        let s = assemble_chain(&GEOM, &[&a]).unwrap();
        assert_eq!((s.window[0].base_pos, s.window[0].positions), (128, 128));
    }

    #[test]
    fn broken_chain_returns_none() {
        // newest needs rows down to 256; skipping d1 leaves [256, 320) uncovered
        let a = delta(1, 0, 256, 128);
        let d2 = delta(1, 320, 384, 128);
        assert!(assemble_chain(&GEOM, &[&d2, &a]).is_none());
    }

    /// The pool writes ONE extent at a time, from a state that holds only that
    /// extent's rows. A blob built that way must name where those rows really are:
    /// assuming 0 made every extent claim the start of the conversation, and the
    /// reader then saw extents that all overlapped. Every disk restore failed and no
    /// unit test noticed, because they all wrote the whole state at once.
    #[test]
    fn an_extent_written_on_its_own_names_where_its_rows_are() {
        let whole = state(2 * grid_tokens());
        let together = unit_blobs_at(&whole, &[grid_tokens(), 2 * grid_tokens()]);
        assert_eq!(together.len(), 2);

        // The same second extent, built alone the way the pool builds it.
        let ks = whole.full[0].k.len() / whole.full[0].positions;
        let vs = whole.full[0].v.len() / whole.full[0].positions;
        let alone = KvState {
            boundary: 2 * grid_tokens(),
            full: whole
                .full
                .iter()
                .map(|ls| KvLayerState {
                    layer: ls.layer,
                    base_pos: grid_tokens(),
                    positions: grid_tokens(),
                    k: ls.k[grid_tokens() * ks..].to_vec(),
                    v: ls.v[grid_tokens() * vs..].to_vec(),
                })
                .collect(),
            window: Vec::new(),
            recurrent: Vec::new(),
        };
        let one = unit_blobs_at(&alone, &[2 * grid_tokens()]);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0], together[1], "a lone extent must equal its slice");

        // ... and the pair still assembles, which is what the restore does.
        let ck = ck_blob(&whole, 2 * grid_tokens());
        let back =
            read_back(&alone, &[together[0].clone(), one[0].clone()], &ck).unwrap();
        assert_eq!(back.full[0].positions, 2 * grid_tokens());
        assert_eq!(back.full[0].k, whole.full[0].k);
    }

    #[test]
    fn extents_that_do_not_abut_are_refused() {
        let whole = state(2 * grid_tokens());
        let b = unit_blobs_at(&whole, &[grid_tokens(), 2 * grid_tokens()]);
        let ck = ck_blob(&whole, 2 * grid_tokens());
        // second extent first: [256,512) cannot follow position 0
        let Err(err) = read_back(&whole, &[b[1].clone(), b[0].clone()], &ck) else {
            panic!("extents out of order were accepted");
        };
        assert!(err.contains("reach 0"), "{err}");
    }

    #[test]
    fn refuses_mismatches() {
        let s = state(grid_tokens());
        let units = unit_blobs(&s);
        let ck = ck_blob(&s, s.boundary / grid_tokens() * grid_tokens());
        assert!(read_back(&s, &units[..0], &ck).is_err());
        let mut bad = units.clone();
        bad[0][0] ^= 1; // the magic
        assert!(read_back(&s, &bad, &ck).is_err());
    }

    /// The property the frame exists for: a section this build does not know is
    /// SKIPPED, not fatal. Without it, every addition is a format bump and every
    /// bump invalidates every store.
    #[test]
    fn an_unknown_section_is_ignored() {
        let s = state(grid_tokens());
        let mut ck = ck_blob(&s, s.boundary / grid_tokens() * grid_tokens());
        // What a future writer would append -- media descriptors, say.
        super::push_section(&mut ck, *b"MEDI", b"an image content id lives here");
        let back = read_back(&s, &unit_blobs(&s), &ck).unwrap();
        assert_eq!(back.boundary, s.boundary);
        assert_eq!(back.recurrent, s.recurrent);
    }

    /// A format from the future is refused rather than read as this one.
    #[test]
    fn a_newer_format_is_refused() {
        let s = state(grid_tokens());
        let mut ck = ck_blob(&s, s.boundary / grid_tokens() * grid_tokens());
        ck[4..8].copy_from_slice(&(super::FORMAT + 1).to_le_bytes());
        let Err(err) = read_back(&s, &unit_blobs(&s), &ck) else {
            panic!("a newer format was accepted")
        };
        assert!(err.contains("this build reads"), "{err}");
    }

    /// A required section that is absent says WHICH, because "short read" is not a
    /// diagnosis.
    #[test]
    fn a_missing_section_names_itself() {
        let s = state(grid_tokens());
        let units = unit_blobs(&s);
        // A checkpoint carrying only its boundary.
        let mut ck = Vec::new();
        super::push_u32(&mut ck, super::CKPT_MAGIC);
        super::push_u32(&mut ck, super::FORMAT);
        super::push_section(&mut ck, *b"BNDY", &(grid_tokens() as u64).to_le_bytes());
        let Err(err) = read_back(&s, &units, &ck) else {
            panic!("a checkpoint with no window section was accepted")
        };
        assert!(err.contains("WNDW"), "{err}");
    }
}
