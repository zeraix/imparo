//! GPU workflow machinery shared by model modules.
//!
//! Everything here is about the DEVICE PATH rather than about any architecture: which
//! backend is live, how big a score tile may be, and how to read a buffer back for a probe.
//! It lived in gemma4/workflow_gpu.rs while gemma4 was the only model, and the second model
//! is what makes that expensive -- LFM2 would copy every one of these verbatim, and a copy
//! has to be kept in step by hand forever.
//!
//! What is NOT here: anything that reads a plan. Buffer lists, layer geometry and the
//! forward graph belong to the model. `had_nrot` stayed with gemma4 for that reason -- it
//! reads gemma4's own rotation knobs.

use imparo_backend::{Backend, BufId};

/// The active backend, selected once. TODAY this is Metal; the imparo-cuda skeleton
/// registers here behind its feature -- adding a backend touches its crate plus this
/// one selection point, nothing else (the #16 multi-developer rule).
pub(crate) fn be() -> &'static dyn Backend {
    // Only reached when routing already confirmed a backend is active.
    crate::backend::active().expect("gpu workflow entered with no active backend")
}

/// Close and reset a row submission even when an encoder rejects a later layer.
/// The encoder opens the forward but does not call `end`; queued work and pending
/// device errors are drained before the scheduler may select another conversation.
pub(crate) fn submit_decode_rows(
    backend: &dyn Backend,
    route: imparo_backend::RowRoute,
    encode: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    if !backend.set_decode_rows(Some(route)) {
        return Err("the backend has no decode rows".into());
    }
    let encoded = encode();
    let ended = backend.end().map_err(|rc| format!("co-batched step failed rc={rc}"));
    let reset = backend.set_decode_rows(None);
    encoded?;
    ended?;
    if !reset {
        return Err("the backend could not close decode rows".into());
    }
    Ok(())
}

/// Checkpoints switch conversation bindings only after the row forward is closed.
/// Reuse the backend's ordinary post-forward copy so CUDA never changes slots inside
/// a live forward/graph. Always restore the host workflow's selected slot on error.
pub(crate) fn snapshot_decode_rows(
    backend: &dyn Backend,
    rows: &[crate::DecodeRow],
    selected: u32,
    recurrent_elems: u32,
) -> Result<(), String> {
    if !rows.iter().any(|r| r.snap) {
        return Ok(());
    }
    let copied = (|| {
        for row in rows.iter().filter(|r| r.snap) {
            let offset = row.plane_out.checked_mul(recurrent_elems)
                .ok_or("co-batched snapshot plane overflow")?;
            if !backend.select_slot(row.slot) {
                return Err(format!("co-batched snapshot: slot {} not selected", row.slot));
            }
            backend.copy_range_after_forward(
                BufId::RecurSnap, 0, BufId::Recur, offset, recurrent_elems,
            ).map_err(|rc| format!("co-batched snapshot: slot {} copy failed rc={rc}", row.slot))?;
        }
        Ok(())
    })();
    let restored = backend.select_slot(selected);
    if !restored {
        return Err(format!("co-batched snapshot: could not restore slot {selected}; copy result: {copied:?}"));
    }
    copied
}

/// Hard cap on attention scores held in threadgroup memory (32 KiB). The
/// attention dispatch slices any longer span (scores-aware slices), so this is
/// a tile size, not a context limit. IMPARO_MAX_SCORES shrinks it (test
/// instrument: exercises the many-slice path at gate-sized prompts).
pub(crate) fn max_scores() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("IMPARO_MAX_SCORES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192)
            .clamp(64, 8192)
    })
}

/// Threadgroup floats actually needed for a step's attention scores.
///
/// Always requesting the 32 KiB cap throttled occupancy: the kernel measured 18.96 us even
/// with a single position to attend to.
pub(crate) fn scores_needed(start_pos: u32, n_tok: u32, window: u32) -> u32 {
    let span = start_pos + n_tok;
    let needed = if window > 0 { span.min(window) } else { span };
    needed.clamp(1, max_scores())
}

/// Diagnostic per-layer quantization masks (IMPARO_KVQ_MASK_K/_V); see the metal side.
pub(crate) fn kvq_mask_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("IMPARO_KVQ_MASK_K").is_ok()
            || std::env::var("IMPARO_KVQ_MASK_V").is_ok()
    })
}

/// Which layer `gprobe` reports on.
///
/// The host side has IMPARO_PROBE_LAYER; without the same control here the two can only
/// be diffed at layer 0, which is where an error that ACCUMULATES is least visible.
///
/// Read ONCE and cached: this gate sits at ~17 probe sites inside the layer loop, so at
/// decode it ran ~600 times a token, and `env::var` takes a process-global lock and scans
/// `environ` on every call -- about 0.6 ms of the ~27 ms step, for a probe that is off.
pub(crate) fn gpu_probe_layer() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        // No probe means NO layer is special. Returning 0 here used to disable fused
        // semantic operations on layer 0 even though every gprobe call immediately
        // returned; E4B therefore ran the sidecar on 41/42 layers in ordinary builds.
        if std::env::var_os("IMPARO_GPU_PROBE").is_none() {
            return usize::MAX;
        }
        std::env::var("IMPARO_GPU_PROBE_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// IMPARO_GPU_PROBE_ROW=last: the probes read the CHUNK'S LAST ROW instead of its first.
/// Row 0 of a chunk is a different position for every chunk width, so two runs at different
/// batch widths can only be compared at the last row of their final chunk, which is the
/// same position in both. Read once, like the layer above.
pub(crate) fn gpu_probe_last_row() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("IMPARO_GPU_PROBE_ROW").is_ok_and(|v| v == "last"))
}

/// Like `gprobe`, but the buffer holds HALVES: reads raw bits and decodes. `n` halves,
/// `off_halves` must be even. `m::read` copies float-sized words, so a plain gprobe on a
/// half buffer prints reinterpreted garbage -- which cost this session a wrong theory.
/// Kept though currently uncalled: it is the only correct way to inspect a half mirror,
/// and the next half-staging investigation will need it on day one.
#[allow(dead_code)]
pub(crate) fn gprobe_half(name: &str, buf: BufId, off_halves: u64, n: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("IMPARO_GPU_PROBE").is_ok()) {
        return;
    }
    let _ = be().end();
    let mut w = vec![0.0_f32; n / 2];
    be().read(buf, off_halves / 2, &mut w);
    let dec = |h: u16| -> f32 {
        let s = if h >> 15 == 0 { 1.0_f32 } else { -1.0_f32 };
        let e = i32::from((h >> 10) & 0x1F);
        let m_ = f32::from(h & 0x3FF);
        if e == 0 {
            s * m_ * (2.0_f32).powi(-24)
        } else if e == 31 {
            f32::NAN
        } else {
            s * (1.0 + m_ / 1024.0) * (2.0_f32).powi(e - 15)
        }
    };
    let mut v = Vec::with_capacity(n);
    for f in &w {
        let b = f.to_bits();
        v.push(dec((b & 0xFFFF) as u16));
        v.push(dec((b >> 16) as u16));
    }
    let head: Vec<String> = v.iter().take(8).map(|x| format!("{x:.5}")).collect();
    eprintln!("[gpu] {name:<20} halves={n} [{}]", head.join(", "));
    be().begin();
}

/// Syncs, reads, and reports a GPU buffer without modifying it or writing a file.
///
/// Gated on IMPARO_GPU_PROBE; it slices the command buffer, so it is a debugging tool and
/// not something to leave on.
/// A probe over a buffer of u32s: index arrays, which `gprobe` cannot read.
///
/// A permutation's entries are small integers, and their bit patterns as f32 are denormals
/// -- every one of them prints as 0.00000 and an rms of zero. Reading `inv` that way once
/// looked exactly like a kernel that had written nothing.
pub(crate) fn gprobe_u32(name: &str, buf: BufId, off: u64, n: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("IMPARO_GPU_PROBE").is_ok()) {
        return;
    }
    let _ = be().end();
    let mut v = vec![0.0_f32; n];
    be().read(buf, off, &mut v);
    let ids: Vec<String> = v.iter().map(|x| x.to_bits().to_string()).collect();
    eprintln!("[gpu] {name}: n={n} [{}]", ids.join(", "));
}

pub(crate) fn gprobe(name: &str, buf: BufId, off: u64, n: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static VALUES: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("IMPARO_GPU_PROBE").is_ok()) {
        return;
    }
    let _ = be().end();
    let mut v = vec![0.0_f32; n];
    be().read(buf, off, &mut v);
    let finite = v.iter().filter(|x| x.is_finite()).count();
    if finite != n {
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            eprintln!("[gpu] {name}: first non-finite at index {i} value {}", v[i]);
        }
    }
    // Exact fingerprint of the WHOLE buffer: an order-sensitive fold of the raw bits.
    // The rms-and-five-values print hides sub-5th-decimal drift; two runs of the same
    // input must produce the same checksum on every probe, and the first probe where they
    // do not names the nondeterministic op (task #5).
    let ck = v.iter().fold(0u64, |a, x| {
        a.wrapping_mul(0x0000_0100_0000_01B3)
            .wrapping_add(u64::from(x.to_bits()))
    });
    eprintln!("[gpu] {name}: ck={ck:016x}");
    // IMPARO_GPU_PROBE_VALUES=<name> prints every value of that probe as its f32 bits in
    // hex, in buffer order, so another engine's activations at the same point can be
    // compared value by value (dev_harness/layer_agree.py). The line below shows five.
    // It goes to stderr like every probe line: a probe writes no file.
    let values = VALUES.get_or_init(|| std::env::var("IMPARO_GPU_PROBE_VALUES").ok());
    if values.as_deref() == Some(name) {
        let bits: Vec<String> =
            v.iter().map(|x| format!("{:08x}", x.to_bits())).collect();
        eprintln!(
            "[gpu] values {name} L{} n={n}: {}",
            gpu_probe_layer(),
            bits.join(" ")
        );
    }
    let rms = (v
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| x * x)
        .sum::<f32>()
        / n as f32)
        .sqrt();
    // Keep the fold order identical to llama.cpp's eval callback. This makes a
    // compact probe useful across backends even when dumping a multi-million-
    // element activation would be too expensive.
    let sum = v.iter().copied().fold(0.0_f32, |acc, value| acc + value);
    let head: Vec<String> = v.iter().take(5).map(|x| format!("{x:.5}")).collect();
    eprintln!(
        "[gpu] {name:<20} n={n:<7} finite={finite:<7} rms={rms:.5} sum={sum:.6} [{}]",
        head.join(", ")
    );
    be().begin();
}

// ---- ACTIVATION BUFFERS ------------------------------------------------------------
//
// Sizing, aliasing and placement of a batch's activation buffers. WHICH buffers a model
// needs and WHICH of them are never live at the same time are facts about that model's
// graph; the rounding, the arena arithmetic and the placement calls are not. So a model
// declares its buffers and the mechanism below places them.
//
// What is deliberately NOT generic: a placement INSIDE another buffer's bytes. Whether a
// buffer is dead at the moment another wants its pages is a statement about one graph, and
// forcing it into a shared type is how the condition guarding it gets dropped. Models do
// that in `place_within`, with the layout handed back to them.

use std::collections::BTreeMap;

/// Where one activation buffer lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Its own allocation.
    Dedicated,
    /// Placed in an arena GROUP. Members of a group are laid out one after another;
    /// groups share the same bytes, because a group is a set of buffers that IS live
    /// while the others are not.
    Group(u8),
    /// Placed INSIDE another buffer's region, at the `slot`-th page-rounded chunk of its
    /// own size. Costs zero memory: it aliases pages the host buffer already holds.
    ///
    /// Skipped silently when the host was not arena-placed, or when the chunk would not
    /// fit inside it. Skipping is not a failure -- the alias is an optimisation, and the
    /// backend's slower path is still correct.
    ///
    /// This was a callback, `Architecture::place_within`, which every architecture with a
    /// half-staging buffer would have written for itself. It is a fact about a buffer, so
    /// it belongs in the buffer list next to the buffer -- where a reader looks for it.
    Within {
        /// The buffer whose pages this one aliases.
        host: BufId,
        /// Which chunk of the host's region, counting in this buffer's own page-rounded
        /// size. Two half-precision mirrors of the same activation are slots 0 and 1.
        slot: u8,
    },
}

/// One activation buffer a workflow needs for a batch.
#[derive(Clone, Copy, Debug)]
pub struct BufferRequirement {
    pub id: BufId,
    pub bytes: u64,
    pub placement: Placement,
}

/// Backend-owned half scratch used while a quantized KV cache is consumed.
///
/// This is a GPU execution contract, not a Gemma-specific buffer: every model
/// workflow that calls `Backend::attention` with quantized K or V must declare the
/// corresponding slot.  Centralizing the capacity formula prevents a new workflow
/// from compiling successfully and then launching a dequant kernel at a null buffer.
pub fn kv_dequant_scratch_requirements(
    plan: &crate::ModelPlan,
    capacity: usize,
    k_needed: bool,
    v_needed: bool,
) -> Vec<BufferRequirement> {
    if !k_needed && !v_needed {
        return Vec::new();
    }
    let head_max = plan
        .layers
        .iter()
        .map(|layer| layer.attention.head_dim() as usize)
        .max()
        .unwrap_or(0);
    let bytes = (capacity.max(crate::kv::KV_FIRST_SLOTS)
        * plan.config.n_kv_heads as usize
        * head_max
        * std::mem::size_of::<u16>()) as u64;
    let mut requirements = Vec::with_capacity(2);
    if k_needed {
        requirements.push(BufferRequirement {
            id: BufId::Kdq,
            bytes,
            placement: Placement::Dedicated,
        });
    }
    if v_needed {
        requirements.push(BufferRequirement {
            id: BufId::Vdq,
            bytes,
            placement: Placement::Dedicated,
        });
    }
    requirements
}

/// Quantized-KV mirrors retain physical page addressing, just like their cache.
/// A multi-conversation pool can exceed one logical context; resizing activations
/// must retain that physical extent without changing model/tuner geometry.
fn fit_paged_kv_scratch(
    requirements: &mut [BufferRequirement],
    logical_capacity: usize,
    physical_rows: usize,
) -> Result<(), String> {
    let logical_rows = logical_capacity.max(crate::kv::KV_FIRST_SLOTS) as u64;
    let rows = (physical_rows as u64).max(logical_rows);
    for req in requirements {
        if !matches!(req.id, BufId::Kdq | BufId::Vdq) {
            continue;
        }
        // These requirements originate from kv_dequant_scratch_requirements.
        if req.bytes % logical_rows != 0 {
            return Err("KV scratch does not have a whole-row layout".into());
        }
        req.bytes = (req.bytes / logical_rows).checked_mul(rows)
            .ok_or("physical KV scratch bytes overflow")?;
    }
    Ok(())
}

/// Backend-owned half-precision mirrors of the activation.
///
/// This is a GPU execution contract, not a Gemma-specific buffer.  The Metal backend's
/// activation PRODUCERS -- `rms_norm` for CUR, `attention` for ATTN, the residual `add`
/// for X, and the fused up-projection epilogue for G -- write a half copy of their output
/// into these slots, and the weight matmuls
/// then read that copy instead of converting f32 to half again in every threadgroup that
/// re-reads the same row.  Nothing about the arithmetic changes: both paths multiply as
/// `simdgroup_half8x8` into an f32 accumulator, so the mirror moves only WHERE the one
/// f32 -> f16 rounding happens.  Without the slots the backend converts inline, which is
/// correct and slower.
///
/// Every gate on that path reads `bufs[Xh] != nil`, so a workflow that does not declare
/// these compiles, runs, produces right answers, and silently never takes the faster path
/// -- with nothing in any log to say so.  LFM2 shipped that way: one missing declaration
/// disabled the producer writes and both matmul readers at once.
///
/// Two slots rather than one: the fused up epilogue writes G's mirror while the SAME
/// dispatch is still reading CUR's mirror out of the first slot, so they cannot share
/// bytes.  Both ALIAS `host`'s pages, which is why `host` must be a buffer that is dead
/// during prefill -- `U` on a SwiGLU feed-forward, because the fused epilogue writes G and
/// leaves U untouched.  That is a REQUIREMENT, not a nicety: with the epilogue unfused the
/// up projection writes U while the mirror of its own input still lives in U's pages, and
/// the dispatch reads and writes the same bytes.
///
/// A dead host is necessary and NOT sufficient: overlapping arena groups all start at
/// offset zero, so another group's live buffers reach into the host's bytes whenever that
/// group is larger.  `place_buffers` checks it and allocates the slot dedicated when the
/// alias would be unsafe, so this list does not have to know the arena's shape.  A slot
/// that does not fit inside `host` at all is skipped, and the backend keeps converting
/// inline.
///
/// `widest_elems` is the largest per-token activation any matmul stages through the
/// mirror, which is `n_ff` on a SwiGLU feed-forward -- wider than the model itself.
pub fn half_activation_mirror_requirements(
    batch: usize,
    widest_elems: usize,
    host: BufId,
) -> Vec<BufferRequirement> {
    // PADDED to a whole token tile, not sized at `batch`. A prefill GEMM walks whole
    // tiles: the conversion pass fills `ceil(n_tok/tile)*tile` rows, and a kernel reading
    // the operand straight from the mirror loads whole 8-row fragments off the end of the
    // last one. At 455 tokens and n_in 10752 the conversion wrote 537 KB past a mirror
    // sized for 455 rows, into whatever the arena had packed after it.
    let rows = batch.next_multiple_of(imparo_backend::MAX_GEMM_TOKEN_TILE);
    let bytes = (rows * widest_elems * std::mem::size_of::<u16>()) as u64;
    [BufId::Xh, BufId::Xh2]
        .into_iter()
        .enumerate()
        .map(|(slot, id)| BufferRequirement {
            id,
            bytes,
            placement: Placement::Within {
                host,
                slot: slot as u8,
            },
        })
        .collect()
}

/// Where each arena-placed buffer ended up, for a model that needs to place something
/// inside one of them.
pub struct Layout {
    regions: BTreeMap<u32, (u64, u64)>,
    bytes: u64,
}

impl Layout {
    /// Byte offset and length of an arena-placed buffer, or None if it was dedicated.
    pub fn region(&self, id: BufId) -> Option<(u64, u64)> {
        self.regions.get(&(id as u32)).copied()
    }

    /// Bytes this layout actually allocated: the arena plus every dedicated buffer.
    ///
    /// Reported by the thing that did the allocating. The footprint line used to
    /// hand-recompute it from the model's dimensions -- a second copy of
    /// `buffer_requirements` that hardcoded gemma4's widest head_dim, so for any other
    /// architecture it would have printed a confident wrong number.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.bytes
    }
}

/// A prefill batch is rounded up to a whole GEMM token tile.
///
/// The prefill GEMM reads activations with simdgroup_load, which takes whole 8-row tiles
/// and cannot mask, so rows past the token count are READ and must be inside the buffer.
/// Sizing a batch of 11 for exactly 11 rows is what made an 11-token prompt fault the GPU.
///
/// Being inside the buffer is not enough on its own: their CONTENT has to be defined.
/// Arena buffers came from posix_memalign, so those rows were heap garbage that differed
/// every run, and identical runs disagreed (batch 512 with a 9-token remainder gave
/// 19.1566, 19.1567, 19.1548). `imparo_metal_arena` now zeroes.
///
/// Decode is exempt: at one token the GEMM is not used, and a floor there would undo the
/// whole point of sizing buffers to the batch.
///
/// 128 is the widest RT_TOKENS in the backend's shape table -- a BACKEND fact this layer
/// currently hardcodes. It belongs behind a Backend accessor; until then it is here rather
/// than in each model, so there is one copy to move.
pub const GEMM_TOKEN_TILE: usize = 128;

#[must_use]
pub fn batch_floor(b_req: usize) -> usize {
    if b_req > 1 {
        b_req.div_ceil(GEMM_TOKEN_TILE) * GEMM_TOKEN_TILE
    } else {
        1
    }
}

/// One decision of `plan_layout` for one buffer, in the order `apply_layout` carries them out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Its own allocation.
    Alloc,
    /// A region of the arena: a group member laid out after its predecessors.
    Place { offset: u64 },
    /// An alias placed inside its host's region, at `host_off + skip`.
    AliasPlace {
        host: BufId,
        slot: u8,
        host_off: u64,
        skip: u64,
    },
    /// An alias promoted to its own allocation: another group's bytes reach into the host.
    AliasDedicated {
        host: BufId,
        slot: u8,
        host_off: u64,
        skip: u64,
        others_reach: u64,
    },
    /// An alias that does not fit inside its host: nothing allocated, the backend's slower
    /// path is still correct.
    AliasSkipped {
        host: BufId,
        slot: u8,
        skip: u64,
        host_bytes: u64,
    },
    /// An alias whose host is not in the arena: nothing to alias into, nothing allocated.
    AliasNoHost,
}

/// One buffer of a `LayoutPlan`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub id: BufId,
    pub bytes: u64,
    pub action: Action,
}

/// The layout as decided, before anything is allocated.
///
/// ONE DERIVATION (task #194). The fit's activation reserve and the allocator used to compute
/// the same number separately: `layout_bytes` summed the groups and the dedicated buffers and
/// skipped every `Placement::Within`, while `place_buffers` promoted an alias that another
/// group reaches into to a Dedicated allocation. On Qwen3.8-27B that left the reservation
/// 34.2 MiB short of what the run then allocated -- and the fit promotes weights into the room
/// it thinks it has, so the tier was oversubscribed by exactly that. Both now read this plan.
#[derive(Clone, Debug)]
pub struct LayoutPlan {
    /// Bytes of the arena: the largest group when groups overlap, their sum when they do not.
    pub arena: u64,
    /// The arena plus every allocation, page-rounded the way the backend rounds.
    pub bytes: u64,
    /// Arena regions by buffer id: offset and length.
    pub regions: BTreeMap<u32, (u64, u64)>,
    pub steps: Vec<Step>,
}

/// Decide the layout of `reqs` without touching the backend.
///
/// Groups share bytes: the arena is sized to the LARGEST group and every group starts at
/// offset zero; with `overlap` off they are laid end to end instead (IMPARO_NO_ARENA_OVERLAP),
/// which is the A/B for whether an aliasing bug is an aliasing bug. `page` is the backend's
/// page rounding, passed in so the plan can be checked without a device.
#[must_use]
pub fn plan_layout_with(
    reqs: &[BufferRequirement],
    overlap: bool,
    page: &dyn Fn(u64) -> u64,
) -> LayoutPlan {
    let mut group_bytes: BTreeMap<u8, u64> = BTreeMap::new();
    for r in reqs {
        if let Placement::Group(g) = r.placement {
            *group_bytes.entry(g).or_insert(0) += page(r.bytes);
        }
    }
    let arena = if overlap {
        group_bytes.values().copied().max().unwrap_or(0)
    } else {
        group_bytes.values().sum()
    };

    // Where each group starts: zero when they share, cumulative when they do not.
    let mut group_start: BTreeMap<u8, u64> = BTreeMap::new();
    let mut running = 0_u64;
    for (&g, &bytes) in &group_bytes {
        group_start.insert(g, if overlap { 0 } else { running });
        running += bytes;
    }

    let mut at: BTreeMap<u8, u64> = group_start;
    let mut regions = BTreeMap::new();
    let mut steps = Vec::with_capacity(reqs.len());
    let mut bytes = arena;
    for r in reqs {
        match r.placement {
            Placement::Dedicated => {
                bytes += page(r.bytes);
                steps.push(Step {
                    id: r.id,
                    bytes: r.bytes,
                    action: Action::Alloc,
                });
            }
            Placement::Group(g) => {
                let off = *at.get(&g).unwrap_or(&0);
                regions.insert(r.id as u32, (off, r.bytes));
                at.insert(g, off + page(r.bytes));
                steps.push(Step {
                    id: r.id,
                    bytes: r.bytes,
                    action: Action::Place { offset: off },
                });
            }
            // Decided below: an alias needs the region it lands in to exist already.
            Placement::Within { .. } => {}
        }
    }
    // WHICH GROUP EACH BUFFER IS IN, so an alias can ask what else reaches its bytes.
    let group_of: BTreeMap<u32, u8> = reqs
        .iter()
        .filter_map(|r| match r.placement {
            Placement::Group(g) => Some((r.id as u32, g)),
            _ => None,
        })
        .collect();
    for r in reqs {
        let Placement::Within { host, slot } = r.placement else {
            continue;
        };
        let Some((host_off, host_bytes)) = regions.get(&(host as u32)).copied() else {
            // Host was dedicated or forced out; nothing to alias into.
            steps.push(Step {
                id: r.id,
                bytes: r.bytes,
                action: Action::AliasNoHost,
            });
            continue;
        };
        let skip = page(r.bytes) * u64::from(slot);
        // AN ALIAS IS ONLY SAFE BEYOND EVERY OTHER GROUP'S END. Overlapping groups all
        // start at offset zero, so bytes inside one group's region are ALSO written by
        // any other group whose own region reaches that far -- being inside a host that
        // is dead is necessary and not sufficient.
        //
        // gemma4 and LFM2 satisfy this by accident: their feed-forward group is the
        // largest, so nothing reaches into it. Qwen3.8's mixer group is more than twice
        // the feed-forward group, and the half-activation mirrors landed in bytes its
        // packed projection was writing -- the GEMM read its own operand as it was being
        // overwritten and the logits came back NaN at every prompt over 64 tokens.
        //
        // Dedicated rather than skipped: the mirror family is worth its bytes (17 MB per
        // slot here), and skipping would silently cost the whole fast path instead.
        let others_reach = if overlap {
            group_bytes
                .iter()
                .filter(|(g, _)| group_of.get(&(host as u32)) != Some(g))
                .map(|(_, &b)| b)
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        if host_off + skip < others_reach {
            bytes += page(r.bytes);
            steps.push(Step {
                id: r.id,
                bytes: r.bytes,
                action: Action::AliasDedicated {
                    host,
                    slot,
                    host_off,
                    skip,
                    others_reach,
                },
            });
            continue;
        }
        if skip + r.bytes > host_bytes {
            steps.push(Step {
                id: r.id,
                bytes: r.bytes,
                action: Action::AliasSkipped {
                    host,
                    slot,
                    skip,
                    host_bytes,
                },
            });
            continue;
        }
        regions.insert(r.id as u32, (host_off + skip, r.bytes));
        steps.push(Step {
            id: r.id,
            bytes: r.bytes,
            action: Action::AliasPlace {
                host,
                slot,
                host_off,
                skip,
            },
        });
    }
    LayoutPlan {
        arena,
        bytes,
        regions,
        steps,
    }
}

/// `plan_layout_with` under the engine's own rules: the backend's page rounding and the
/// IMPARO_NO_ARENA_OVERLAP switch.
#[must_use]
pub fn plan_layout(reqs: &[BufferRequirement]) -> LayoutPlan {
    let overlap = std::env::var("IMPARO_NO_ARENA_OVERLAP").is_err();
    plan_layout_with(reqs, overlap, &|b| be().page_round(b))
}

/// The bytes `place_buffers` would allocate for `reqs`, without allocating -- the plan's own
/// count, so it includes an alias promoted to Dedicated. The activation term of the fast-tier
/// reserve (docs/memory-tiers-and-fit.md section 2) is this number at the largest batch.
#[must_use]
pub fn layout_bytes(reqs: &[BufferRequirement]) -> u64 {
    plan_layout(reqs).bytes
}

/// Carry a plan out: size the arena, place the grouped buffers and the aliases, allocate the
/// dedicated ones, and hand back where everything landed. Every alias decision is logged
/// here under IMPARO_LOG, because a skipped alias is invisible in the output -- the backend's
/// slower path is still correct -- and a performance cliff with no explanation is worse than
/// a log line. Absence of evidence is not evidence.
///
/// # Errors
/// When the backend cannot allocate.
pub fn apply_layout(plan: LayoutPlan) -> Result<Layout, String> {
    be().arena(plan.arena).map_err(|rc| {
        format!("metal arena rc={rc} (asked for {} bytes)", plan.arena)
    })?;
    for s in &plan.steps {
        match s.action {
            Action::Alloc => {
                be().alloc(s.id, s.bytes)
                    .map_err(|rc| format!("metal alloc {:?} failed rc={rc}", s.id))?;
            }
            Action::Place { offset } => {
                be().place(s.id, offset, s.bytes)
                    .map_err(|rc| format!("metal place {:?} rc={rc}", s.id))?;
            }
            Action::AliasDedicated {
                host,
                slot,
                host_off,
                skip,
                others_reach,
            } => {
                be().alloc(s.id, s.bytes)
                    .map_err(|rc| format!("metal alloc {:?} failed rc={rc}", s.id))?;
                if crate::log_on() {
                    eprintln!(
                        "[imparo] alias {:?} slot {slot} DEDICATED: another group reaches \
                         {others_reach} bytes, past {host_off}+{skip} inside {host:?}",
                        s.id
                    );
                }
            }
            Action::AliasSkipped {
                host,
                slot,
                skip,
                host_bytes,
            } => {
                if crate::log_on() {
                    eprintln!(
                        "[imparo] alias {:?} slot {slot} SKIPPED: {} + {} > {host_bytes} \
                         inside {host:?}",
                        s.id, skip, s.bytes
                    );
                }
            }
            Action::AliasPlace {
                host,
                slot,
                host_off,
                skip,
            } => {
                be().place(s.id, host_off + skip, s.bytes)
                    .map_err(|rc| format!("metal place {:?} rc={rc}", s.id))?;
                if crate::log_on() {
                    eprintln!(
                        "[imparo] alias {:?} slot {slot} at {}+{skip} ({} bytes) inside {host:?}",
                        s.id, host_off, s.bytes
                    );
                }
            }
            Action::AliasNoHost => {}
        }
    }
    Ok(Layout {
        regions: plan.regions,
        bytes: plan.bytes,
    })
}

/// Plan and carry out in one call: what every workflow's activation allocation uses.
///
/// # Errors
/// When the backend cannot allocate.
pub fn place_buffers(reqs: &[BufferRequirement]) -> Result<Layout, String> {
    apply_layout(plan_layout(reqs))
}

// ---------------------------------------------------------------------------------------
// device setup, written once for every architecture
// ---------------------------------------------------------------------------------------
//
// None of this reads a model. It reads the plan, the shared state and the architecture's
// declared buffer list -- so a second model that wrote its own copy would be writing the
// same 164 lines with different names. The architecture supplies exactly two things: WHAT
// buffers it needs (`Architecture::buffer_requirements`) and any buffer that ALIASES
// inside the layout once it exists (`Architecture::place_within`).

use crate::{Architecture, Workflow};

/// Footprint probe around the FIRST occurrence of each labelled point only.
///
/// The startup probes stop at 45.8 MiB and the post-request probe reads 222 MiB with
/// almost no KV or activations allocated, so ~130 MiB appears somewhere in between and
/// nothing so far says where. Once per label, because these sit on the batch path.
pub fn probe_first(label: &str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SEEN: [AtomicBool; 4] = [
        AtomicBool::new(false),
        AtomicBool::new(false),
        AtomicBool::new(false),
        AtomicBool::new(false),
    ];
    let slot = match label {
        "gpu batch entry" => 0,
        "gpu after writes" => 1,
        "gpu after submit" => 2,
        _ => 3,
    };
    if !SEEN[slot].swap(true, Ordering::Relaxed) {
        crate::host::log_footprint(label);
    }
}

/// Prefill fuses the activation into the up projection's write-back; decode does not --
/// one token is one row, and the masked write-back the fusion forces costs more than the
/// pass it saves.
pub(crate) fn should_fuse_epilogue(
    n_tok: u32,
    backend_supports_activation: bool,
) -> bool {
    fuse_epilogue_enabled() && backend_supports_activation && n_tok > 1
}

/// IMPARO_FUSE_EPILOGUE=0 issues the two projections and a separate `act_mul` instead,
/// which is what llama.cpp does (two `mul_mat`s and a GLU op).
///
/// The two are not the trade they look like. Fusing moves FEWER bytes -- the up
/// projection reads the gate output and writes once, where the split path writes the up
/// output, then reads both and writes again -- but on the Metal Q8 GEMM it forces the
/// MASKED write-back for every one of those dispatches, because the predicate-free route
/// requires `epilogue == 0`. That path spills the accumulators through threadgroup
/// memory, barriers, and finishes with a SCALAR per-element read-modify-write where the
/// unfused route ends in a vector `simdgroup_store`. Which side wins is a measurement,
/// and it is one knob for every architecture rather than one per directory.
pub(crate) fn fuse_epilogue_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_FUSE_EPILOGUE").map_or(true, |v| v != "0"))
}

/// Layers per command buffer for this forward: the tuner's seat, LOWERED so that no one
/// buffer holds the device longer than the stall budget.
///
/// A COMMAND BUFFER'S DURATION IS OTHERWISE UNBOUNDED IN MODEL SIZE. Every
/// architecture encodes a whole prefill chunk into one buffer when the seat is 0, and
/// nothing in that decision scales with the model: LFM2 and gemma4 E4B land around
/// 450-500 ms, and Qwen3.8-27B lands at 5.0 s on the same 512-token chunk. A buffer
/// that long leaves the compositor no gap at all, which is the failure this bound
/// exists for -- a 2026-09-09 run held one for over fifteen minutes and took the
/// machine with it.
///
/// THE BOUND IS A TIME PER LAYER AT A WIDTH, NOT A LAYER COUNT. A buffer costs
/// (layers in it) x (this model's GPU seconds per layer at this chunk width), and the
/// second factor is what is unknown. It is measured when a prefill region ends
/// (`prefill_region_ended`, from the region's longest buffer) and kept as the LARGEST
/// per-layer time seen at the WIDEST width seen; a query at a narrower width takes that
/// time as it is (a narrower chunk never costs a layer more than a wider one) and a
/// wider query scales it by the width ratio (the compute-bound upper bound). So the
/// bound only ever tightens for a given width, and a 2-token tail -- whose layer costs
/// its weight read, not its two tokens -- cannot masquerade as a rate and shrink every
/// later 512-token buffer to one layer.
///
/// Until 2026-09-11 the bound read the LAST region's longest buffer at the start of the
/// next forward and lowered a layer count. That left the first region of every process
/// unbounded (4.2-5.0 s in one buffer on the 27B), and the first chunk of every later
/// request too, because the region before a prefill is a decode step whose buffers are
/// milliseconds wide -- a single-chunk request was never bounded at all. Before its
/// first measurement a process now splits the region into eight buffers: the first
/// region's cost is unknown, the largest model this tree has run holds the device 5 s
/// in one buffer, and eight boundaries cost about 2 ms on the smallest models'
/// half-second regions, once per process. From the second region on the measurement
/// decides. Regions a workflow ends without reporting (an early return) are folded in
/// at the next call, one forward late.
///
/// THE SEAT STILL DECIDES SPEED; this only ever lowers it. `flush_layers` is swept by
/// imparo-tune and stays the answer to "how many layers per buffer is fastest"; the bound
/// is a separate question -- "how long may one buffer hold the device" -- and the two are
/// kept apart on purpose. On the small models the measured time puts the cap above their
/// layer count, so their seat (one buffer) is what runs after the first region.
///
/// The budget is a POLICY, not a derivation, and is stated as one: 1000 ms by default,
/// `IMPARO_CB_STALL_MS` to change it. Two things pin it rather than taste. It sits
/// under the mega failsafe's ~1.2 s spin cap, so a prefill buffer can no longer outlive
/// the mechanism that catches a wedged decode. And it is above both shipped small
/// models' whole-prefill buffers, measured, so neither of them moves -- a bound that
/// silently reshaped LFM2 or E4B would be paying for the 27B with their speed. No
/// probe was run to find the point where a compositor actually starves: deliberately
/// holding the GPU to find it is how this machine was hurt before.
///
/// Decode (`b == 1`) is returned untouched. A decode buffer is one token, milliseconds
/// wide, and was never the thing holding the device.
pub fn flush_layers_bounded(b: u32, layers: usize) -> usize {
    use std::sync::atomic::Ordering;
    // A region the workflow ended without reporting is folded in here, one forward late.
    prefill_region_ended();
    let decode = b <= 1;
    let seat = be().flush_layers(decode) as usize;
    if decode || layers == 0 {
        return seat;
    }
    // A seat of 0, or one wider than the model, means one buffer for the whole forward.
    let want_by_seat = if seat == 0 || seat > layers {
        layers
    } else {
        seat
    };
    let per_layer_s = prefill_layer_seconds_at(b);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    let cap = if per_layer_s > 0.0 {
        ((stall_budget_s() / per_layer_s).floor() as usize).max(1)
    } else {
        layers.div_ceil(FIRST_REGION_BUFFERS).max(1)
    };
    let out = want_by_seat.min(cap);
    LAST_PREFILL_LPB.store(out.min(layers), Ordering::Relaxed);
    LAST_PREFILL_B.store(b, Ordering::Relaxed);
    // The call sites read 0 as "one buffer for the whole forward"; say that when the
    // answer is the whole model, so an unbounded model keeps the seat it was tuned with.
    if out >= layers { seat } else { out }
}

/// Buffers the first prefill region of a process is split into, before any time has
/// been measured. See `flush_layers_bounded`.
const FIRST_REGION_BUFFERS: usize = 8;

/// The reference measurement: the widest prefill chunk this process has run (tokens)
/// and the largest GPU seconds one layer took in any region (f64 bits). 0 until the
/// first region ends.
static PREFILL_REF_B: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
static PREFILL_REF_LAYER_S: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// The geometry the last prefill forward was given -- layers per buffer and chunk width
/// -- so its region can be read back when it ends. 0 layers = nothing pending.
static LAST_PREFILL_LPB: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static LAST_PREFILL_B: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// The GPU seconds one layer is expected to take in a prefill chunk of `b` tokens: the
/// reference time as it is up to the reference width, scaled by the width ratio above
/// it. 0 while nothing has been measured.
fn prefill_layer_seconds_at(b: u32) -> f64 {
    use std::sync::atomic::Ordering;
    let ref_b = PREFILL_REF_B.load(Ordering::Relaxed);
    if ref_b == 0 {
        return 0.0;
    }
    let t = f64::from_bits(PREFILL_REF_LAYER_S.load(Ordering::Relaxed));
    if b <= ref_b {
        t
    } else {
        t * f64::from(b) / f64::from(ref_b)
    }
}

/// A prefill region has ended (after `be().end()`): its longest command buffer over the
/// layers that buffer held is one layer's time at the region's width, and it joins the
/// reference (widest width, largest time). A decode region, or one no prefill forward
/// announced, leaves the reference alone. Every change of the reference is logged, so a
/// later, tighter cap stays visible in raw run logs.
pub fn prefill_region_ended() {
    use std::sync::atomic::Ordering;
    let lpb = LAST_PREFILL_LPB.swap(0, Ordering::Relaxed);
    let b = LAST_PREFILL_B.load(Ordering::Relaxed);
    if lpb == 0 || b == 0 {
        return;
    }
    let longest = be().longest_cb_gpu_seconds();
    if longest <= 0.0 {
        return;
    }
    #[allow(clippy::cast_precision_loss)]
    let layer_s = longest / lpb as f64;
    let old_b = PREFILL_REF_B.load(Ordering::Relaxed);
    let old_t = f64::from_bits(PREFILL_REF_LAYER_S.load(Ordering::Relaxed));
    let new_b = old_b.max(b);
    let new_t = old_t.max(layer_s);
    if new_b == old_b && new_t <= old_t {
        return;
    }
    PREFILL_REF_B.store(new_b, Ordering::Relaxed);
    PREFILL_REF_LAYER_S.store(new_t.to_bits(), Ordering::Relaxed);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    let cap_at_ref =
        ((stall_budget_s() / prefill_layer_seconds_at(new_b)).floor() as usize).max(1);
    eprintln!(
        "[imparo] prefill buffer bound: {lpb} layers x {b} tokens held the GPU {:.0} ms in \
         one buffer ({:.1} ms per layer); reference now {:.1} ms per layer at {new_b} \
         tokens, so a buffer there may hold {cap_at_ref} layers under the {:.0} ms budget \
         (IMPARO_CB_STALL_MS)",
        longest * 1e3,
        layer_s * 1e3,
        new_t * 1e3,
        stall_budget_s() * 1e3
    );
}

/// How long one command buffer may hold the device. See `flush_layers_bounded`.
pub fn stall_budget_s() -> f64 {
    static MS: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("IMPARO_CB_STALL_MS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or(1000.0)
            * 1e-3
    })
}

impl<A: Architecture> Workflow<A> {
    /// This model's per-layer KV byte budget at `positions`, from the shared geometry.
    #[must_use]
    pub fn kv_bytes_for(&self, positions: usize) -> Vec<u64> {
        crate::kv::kv_bytes_for(
            &self.plan,
            positions,
            self.state.kv_rt.capacity,
            self.state.kv_ring_batch,
        )
    }

    /// Explicit logical slots and independent K/V strides matching `kv_bytes_for`.
    #[must_use]
    pub fn kv_layout_for(&self, positions: usize) -> Vec<imparo_backend::KvLayout> {
        crate::kv::kv_layout_for(
            &self.plan,
            positions,
            self.state.kv_rt.capacity,
            self.state.kv_ring_batch,
        )
    }

    /// Reports what the backend holds for each KV layer, grouped.
    ///
    /// Grouped rather than totalled because the total does not say WHICH layers own it,
    /// and at ctx 8192 this engine held 62 MiB more than the fork for the same model.
    fn log_kv_groups(&self, kv_bytes: &[u64]) {
        if !crate::log_on() {
            return;
        }
        let mut groups: std::collections::BTreeMap<(u32, usize), (usize, u64)> =
            std::collections::BTreeMap::new();
        for layer in &self.plan.layers {
            if layer.kv_source != crate::KvSource::Own
                || !layer.attention.is_attention()
            {
                continue;
            }
            let hd = layer.attention.head_dim();
            let slots = match crate::kv::ring_slots(
                layer.attention,
                self.state.kv_ring_batch,
            ) {
                0 => self.state.kv_rt.capacity,
                r => r.min(self.state.kv_rt.capacity),
            };
            let e = groups.entry((hd, slots)).or_insert((0, 0));
            e.0 += 1;
            e.1 += kv_bytes[layer.index as usize] * 2; // K and V
        }
        for ((hd, slots), (n, bytes)) in &groups {
            eprintln!(
                "[imparo] kv: {n:>2} layers head_dim={hd:<4} slots={slots:<6} {:>8.1} MiB",
                *bytes as f64 / (1 << 20) as f64
            );
        }
        if let Some(d) = &self.plan.drafter {
            let first = self.plan.config.n_layers as usize;
            let bytes = kv_bytes[first..].iter().sum::<u64>() * 2; // K and V
            eprintln!(
                "[imparo] kv: {:>2} drafter layers head_dim={:<4} slots={:<6} {:>8.1} MiB, \
                 {:.1} KiB per token",
                d.layers,
                d.head_dim,
                self.state.kv_rt.capacity,
                bytes as f64 / (1 << 20) as f64,
                crate::kv::drafter_kv_bytes_per_token(&self.plan) as f64 / 1024.0
            );
        }
    }

    /// Allocates the activation buffers for a batch of `b_req` tokens.
    ///
    /// Re-run whenever the batch width changes, and it SHRINKS as well as grows: a prefill
    /// buffer sized for the whole prompt would otherwise be held for the context's life,
    /// but a decode step is one token wide. On unified memory that is not just waste --
    /// memory the engine does not hold is page cache the weights stream through, so
    /// returning it can make decode faster as well as smaller.
    ///
    /// KV is deliberately untouched: it holds conversation state and must survive a width
    /// change.
    ///
    /// # Errors
    /// When the backend cannot allocate.
    /// Returns the bytes it allocated, so a caller that wants to report the footprint
    /// does not run the placement a second time to find out.
    // Capacity belongs to allocation, not mathematical batch width or output demand.
    // The opt-in observer lease retains exactly the already required M3 layout:
    // batch_floor(3) is unchanged, and prefill/capture-off retain normal shrinking.
    fn activation_layout_request(&self, rows: usize) -> (usize, crate::OutputDemand) {
        let rows = rows.max(self.state.activation_floor_rows);
        if (1..=3).contains(&rows)
            && self
                .state
                .layer_outputs
                .as_ref()
                .is_some_and(crate::layer_outputs::LayerOutputCapture::active)
            && (crate::e4b_retained_decode_policy_enabled()
                || std::env::var("IMPARO_LAB_SPEC_ACTIVATION_CAP").as_deref()
                    == Ok("3"))
        {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[imparo] speculative activation capacity=3 tile_rows={} logits_rows=3",
                    batch_floor(3)
                );
            }
            (3, crate::OutputDemand::AllTokens)
        } else {
            (rows, self.state.output_demand)
        }
    }

    pub fn gpu_alloc_activations(&mut self, b_req: usize) -> Result<u64, String> {
        let (layout_rows, layout_demand) = self.activation_layout_request(b_req);
        let b = batch_floor(layout_rows);
        let mut reqs = A::buffer_requirements_for_output(
            &self.plan,
            b,
            self.state.kv_rt.capacity,
            layout_demand,
            layout_rows,
        )?;
        let physical_rows = self.state.kv_commit.pool_capacity
            .checked_mul(imparo_kv::page_cells())
            .ok_or("physical KV scratch rows overflow")?;
        fit_paged_kv_scratch(&mut reqs, self.state.kv_rt.capacity, physical_rows)?;
        let layout = place_buffers(&reqs)?;
        self.state.gpu_batch = layout_rows;
        self.state.gpu_all_logits = layout_demand.requires_all_positions();
        Ok(layout.total_bytes())
    }

    /// Re-sizes the activation buffers when the batch width changes, and only then.
    ///
    /// # Errors
    /// When the backend cannot allocate.
    pub fn gpu_fit_batch(&mut self, b: usize) -> Result<(), String> {
        let (layout_rows, layout_demand) = self.activation_layout_request(b);
        if layout_rows == self.state.gpu_batch
            && self.state.gpu_all_logits == layout_demand.requires_all_positions()
        {
            return Ok(());
        }
        let was = self.state.gpu_batch;
        self.gpu_alloc_activations(b)?;

        if crate::log_on() {
            eprintln!("[imparo] activation width {was} -> {b}");
            crate::host::log_footprint(&format!("resize to {b}"));
        }
        Ok(())
    }

    /// Brings the device path up if it is not already.
    ///
    /// Entered from the forward AND from the pool -- `kv_restore` and `kv_prepare_pool`
    /// both arrive without a forward, so either can be the first thing to touch the
    /// device. gemma4's forward used to carry a second copy of this inline.
    ///
    /// # Errors
    /// When the architecture has no device forward, or preparation fails.
    pub fn ensure_gpu_ready(&mut self) -> Result<(), String> {
        if self.state.gpu_ready {
            return Ok(());
        }
        crate::host::log_footprint("before gpu prepare");
        let ring_batch = crate::prefill_batch();
        let initial_batch = self.state.initial_gpu_batch.take().unwrap_or(ring_batch);
        self.gpu_prepare_initial(ring_batch, initial_batch)?;
        self.state.kv_rt.slots = crate::kv::KV_FIRST_SLOTS;
        self.state.gpu_ready = true;
        Ok(())
    }

    /// Allocates activation and KV buffers for a batch of `max_batch` tokens.
    ///
    /// # Errors
    /// When the architecture has no device forward, or the backend cannot allocate.
    pub fn gpu_prepare(&mut self, max_batch: usize) -> Result<(), String> {
        self.gpu_prepare_initial(max_batch, max_batch)
    }

    fn gpu_prepare_initial(
        &mut self,
        max_batch: usize,
        initial_batch: usize,
    ) -> Result<(), String> {
        if initial_batch == 0 || initial_batch > max_batch {
            return Err("initial activation width exceeds the prefill bound".into());
        }
        if !A::DEVICE {
            return Err(crate::no_device_workflow(&self.plan));
        }
        // Set BEFORE any kv_bytes_for call: it sizes the windowed rings, and a ring cannot
        // be resized later without invalidating every position already mapped into it.
        self.state.kv_ring_batch = max_batch;
        let act_bytes = self.gpu_alloc_activations(initial_batch)?;

        // Per-conversation recurrent state: allocated ONCE, from the plan, and never
        // resized -- it is constant in context. Outside `buffer_requirements` on purpose:
        // that list is re-placed on every batch-width change, and this buffer holds
        // conversation state that must survive one.
        // The persistent kernel's scratch, sized once from the widest FFN, before any region
        // runs: a regrow inside a region would swap the buffer that carries its error word.
        let n_mid = self
            .plan
            .layers
            .iter()
            .map(|l| l.ffn.max_hidden())
            .max()
            .unwrap_or(self.plan.config.n_ff)
            .max(self.plan.config.n_ff);
        let attn_hd = self
            .plan
            .layers
            .iter()
            .map(|l| l.attention.head_dim())
            .max()
            .unwrap_or(0);
        be().mega_reserve(n_mid, self.plan.config.n_heads, attn_hd)
            .map_err(|rc| format!("metal mega scratch reserve rc={rc}"))?;
        let recur = self.plan.recurrent_elems();
        if recur > 0 {
            // Use the same negotiated plane count as the decode cursors. Backends
            // with rolling state keep the pre-step plane for rollback; a fixed-state
            // graph backend executes in place in its single admitted plane.
            be().alloc(
                BufId::Recur,
                u64::from(recur) * 4 * u64::from(self.state.recur_planes),
            )
            .map_err(|rc| format!("metal alloc recurrent state rc={rc}"))?;
            // The snapshot twin, same size: where the state at a boundary INSIDE a batch
            // is written. See `arm_recurrent_snapshot`.
            be().alloc(BufId::RecurSnap, u64::from(recur) * 4)
                .map_err(|rc| format!("metal alloc recurrent snapshot rc={rc}"))?;
            self.state.gpu_recur_snap_slots = 1;
            self.zero_recurrent();
            if crate::log_on() {
                eprintln!(
                    "[imparo] recurrent state {:.2} MiB per conversation",
                    f64::from(recur) * 4.0 / (1 << 20) as f64
                );
            }
        }

        let kv_bytes = self.kv_bytes_for(crate::kv::KV_FIRST_SLOTS);
        let kv_layout = self.kv_layout_for(crate::kv::KV_FIRST_SLOTS);
        // Room to grow in place: the pooled layers reach the pool's whole device tier, every
        // other layer the whole context, without moving a row.
        let tier = if be().kv_commits_on_demand() {
            crate::placement::kv_tier()
        } else {
            None
        };
        let pool_blocks = crate::kv::pool_capacity_blocks(
            &self.plan,
            self.state.kv_rt.capacity,
            max_batch,
            tier,
            be().kv_max_view_bytes(),
        );
        self.state.kv_commit.pool_capacity = pool_blocks;
        let kv_reserve = crate::kv::kv_reserve_for_pool(
            &self.plan,
            self.state.kv_rt.capacity,
            max_batch,
            pool_blocks,
        );
        self.log_kv_groups(&kv_bytes);
        crate::host::log_footprint("gpu activations");
        be().alloc_kv_reserved(&kv_bytes, &kv_reserve, &kv_layout)
            .map_err(|rc| format!("metal kv alloc failed rc={rc}"))?;
        crate::host::log_footprint("gpu kv");

        // Persistent transformed weights are admitted only after higher-priority
        // activation and KV allocations. Architectures without such a cache inherit a
        // no-op, and each backend retains fail-closed authority over memory fit.
        A::device_prepare(self)?;

        // Account for what the engine ACTUALLY allocates, from the layout that did it.
        // Footprint once read 452 MiB against a hand estimate of ~138, and guessing at
        // the difference is how you optimise the wrong thing.
        // The recurrent state was missing from this line, and on Qwen3.8-27B that is not a
        // rounding error: 149.62 MiB of state became 598.5 MiB of allocation (three planes
        // so a failed step is undone by not advancing an index, plus one boundary
        // snapshot), which read as 600 MiB of unexplained growth between two footprint
        // stages. A line that omits an allocation invites exactly the guessing the comment
        // above forbids, so it names every one.
        let kv_total: u64 = kv_bytes.iter().sum::<u64>() * 2; // K and V
        let recur = u64::from(self.plan.recurrent_elems()) * 4;
        let planes = self.state.recur_planes;
        let recur_total = recur * u64::from(planes) + recur;
        let mib = |b: u64| b as f64 / (1 << 20) as f64;
        eprintln!(
            "[imparo] gpu alloc: kv={:.1} MiB activations={:.1} MiB recurrent={:.1} MiB              ({} planes + snapshot of {:.1}) (max_batch={max_batch}, initial_batch={initial_batch})",
            mib(kv_total),
            act_bytes as f64 / (1 << 20) as f64,
            mib(recur_total),
            planes,
            mib(recur)
        );
        // LAST, because it is the last thing load owes the device: every allocation above
        // has joined whatever the backend keeps resident, so wiring them now is wiring
        // them once. See `Backend::wire_weights` for why it is not left to request one.
        be().wire_weights(stall_budget_s());
        Ok(())
    }

    /// Makes sure every layer's device cache can hold `positions`, keeping the contents.
    ///
    /// # Errors
    /// When the backend cannot allocate.
    pub fn kv_fit(&mut self, positions: usize) -> Result<(), String> {
        if !be().supports_kv_incremental_commit() {
            if positions <= self.state.kv_rt.slots {
                return Ok(());
            }
            let want = crate::kv::kv_round(positions).min(self.state.kv_rt.capacity);
            if want <= self.state.kv_rt.slots {
                return Ok(());
            }
            let bytes = self.kv_bytes_for(want);
            let layout = self.kv_layout_for(want);
            be().grow_kv_layout(&bytes, &layout)
                .map_err(|rc| format!("metal kv grow failed rc={rc}"))?;
            if crate::log_on() {
                eprintln!("[imparo] kv slots {} -> {want}", self.state.kv_rt.slots);
            }
            self.state.kv_rt.slots = want;
            return Ok(());
        }
        let cap = self.state.kv_rt.capacity;
        if positions > self.state.kv_rt.slots {
            // Asking for exactly what a preparation covers lets the backend adopt it with no
            // wait; past it, the next chunk boundary.
            let prepared = self.state.kv_commit.prefetched;
            let want = if prepared >= positions {
                prepared
            } else {
                crate::kv::kv_commit_rows(positions, cap)
            };
            let bytes = self.kv_fit_bytes(want);
            let layout = self.kv_layout_for(want);
            be().grow_kv_layout(&bytes, &layout)
                .map_err(|rc| format!("metal kv grow failed rc={rc}"))?;
            if crate::log_on() {
                eprintln!("[imparo] kv slots {} -> {want}", self.state.kv_rt.slots);
            }
            self.state.kv_rt.slots = want;
            self.state.kv_commit.prefetched = 0;
        }
        // Within a chunk of the end: get the next chunk ready off the step's path.
        let slots = self.state.kv_rt.slots;
        let step = crate::prefill_batch();
        if slots < cap
            && slots - positions.min(slots) < step
            && self.state.kv_commit.prefetched <= slots
        {
            let next = (slots + step).min(cap);
            be().kv_prefetch(&self.kv_fit_bytes(next));
            self.state.kv_commit.prefetched = next;
        }
        Ok(())
    }

    /// Per-layer bytes for `positions`, leaving out the layers the pool places: their
    /// storage follows its blocks (`KvPoolMember::kv_commit_blocks`), not a position count.
    fn kv_fit_bytes(&self, positions: usize) -> Vec<u64> {
        let mut bytes = self.kv_bytes_for(positions);
        if self.state.kv_commit.pooled {
            for g in crate::kv::state_geometry(&self.plan, self.state.kv_ring_batch) {
                if matches!(g.kind, imparo_kv::StateKind::Full) {
                    bytes[g.layer as usize] = 0;
                }
            }
        }
        bytes
    }

    /// Clears the device recurrent state: the device twin of `RecurrentState::reset`.
    ///
    /// A conversation that resumed on a dirty state gives the right shape and the wrong
    /// numbers.
    ///
    /// THE BACKEND FILLS; THE HOST DOES NOT BUILD THE ZEROS. This used to allocate a host
    /// `Vec<f32>` of `recurrent_elems * planes`, fault every page of it in, and
    /// memcpy it across. On Qwen3.8-27B that is a gigabyte of state, and it cost 3374 ms
    /// on the first conversation and 65 ms on every one after -- all of it to produce a
    /// source operand whose value is known. `Backend::zero` fills the buffer in place.
    pub fn zero_recurrent(&mut self) {
        let n = self.plan.recurrent_elems() as u64;
        if n > 0 {
            be().zero(BufId::Recur, 0, n * u64::from(self.state.recur_planes));
            // The snapshot twin too. A layer kind writes only the part of the state it
            // owns -- a short convolution writes its history and nothing else -- so any
            // region no snapshot dispatch covers would be read back as whatever the
            // allocation happened to contain. LFM2 has no such region (s_elems = 0); a
            // gated-delta-rule model would.
            be().zero(BufId::RecurSnap, 0, n);
            // A note that lived only in that buffer is gone with it.
            if self.state.recur_ckpt_on_device {
                self.state.recur_ckpt_on_device = false;
                self.state.recur_ckpt.clear();
            }
        }
    }

    /// Tells the batch about to run whether a checkpoint boundary falls inside it, and
    /// where.
    ///
    /// A checkpoint for a boundary cannot be read off the device afterwards -- the live
    /// buffer holds only "now". The batch therefore writes it aside as it passes, and
    /// `state.recur_snap` is how the model's graph is told to: `Some(k)` means "the
    /// boundary is k tokens into this batch". Every state-holding kernel then writes
    /// its part of the state as of row k into `BufId::RecurSnap`: the short convolution
    /// through an extra state-shaped dispatch, the delta rule in passing (its matrix
    /// lives in registers for the batch). A snapshot is the WHOLE recurrent state or it
    /// is not one: a checkpoint carrying the conv history alone restored Qwen3.8-27B
    /// with an all-zero matrix, and the adopting conversation answered differently from
    /// its first token.
    ///
    /// ```text
    /// chunk grid 512, unit grid 256, an 800-token prefill:
    ///   cut on the boundary   [0,512) [512,768) [768,800)   3 batches, snapshot is "now"
    ///   snapshot in the batch [0,512) [512,800)             2 batches, k = 256 in the 2nd
    /// ```
    ///
    /// The cut is what this replaces: it measured +29 ms on that 800-token prefill.
    ///
    /// Only the LAST boundary a call reaches is ever checkpointed -- checkpoints go at
    /// branch points and turn ends, not on a grid, and everything below is already the
    /// previous call's copy. So `last` is that boundary and the window is `(at, at + n]`.
    ///
    /// A no-op, leaving None, for a model with no recurrent layers or a batch that
    /// reaches no boundary.
    pub fn arm_recurrent_snapshot(&mut self, at: usize, n: usize, last: usize) {
        // A row-layout batch (a tree) holds no boundary: its rows are not a prefix of the answer.
        self.state.recur_snap = (self.plan.recurrent_elems() > 0
            && self.state.row_layout == 0
            && last > at
            && last <= at + n
            && last % imparo_kv::grid_tokens() == 0)
            .then(|| u32::try_from(last - at).expect("chunk fits u32"));
    }

    /// Notes what the armed batch wrote aside, and disarms.
    ///
    /// Called after the batch whatever happened, so an armed flag never survives into a
    /// batch that was not told about it -- a stale `Some(k)` would have the next graph
    /// snapshot a boundary that is not there.
    ///
    /// A synchronous decode leaves the bytes on the device (`recur_ckpt_on_device`) for
    /// the checkpoint or switch that asks; an interleaved one reads them back here,
    /// because a queued step it later discards may overwrite the buffer.
    pub fn take_recurrent_snapshot(&mut self, at: usize) {
        let Some(k) = self.state.recur_snap.take() else {
            return;
        };
        let n = self.plan.recurrent_elems() as usize;
        self.state.recur_ckpt_at = at + k as usize;
        if !self.plan.decode_interleave {
            self.state.recur_ckpt.clear();
            self.state.recur_ckpt_on_device = true;
            return;
        }
        let t = std::time::Instant::now();
        self.state.recur_ckpt = crate::kv::read_recurrent(n, BufId::RecurSnap, 0);
        self.state.recur_ckpt_on_device = false;
        // Under IMPARO_PROF, so the readback is attributed: it runs at every grid boundary
        // a decode crosses (every 64 tokens) and blocks the host until the bytes are in.
        if std::env::var("IMPARO_PROF").is_ok_and(|v| v == "1") {
            eprintln!(
                "[prof] recurrent snapshot readback {:.1} MiB in {:.1} ms at boundary {}",
                n as f64 * 4.0 / (1u64 << 20) as f64,
                t.elapsed().as_secs_f64() * 1e3,
                self.state.recur_ckpt_at
            );
        }
    }

    /// Reads back a summary of the device KV, for the drift instruments.
    pub fn kv_scan(&self, positions: usize) {
        crate::kv::scan(be(), &self.plan, positions);
    }
}

/// Split a Prefill chunk so only its final 64-or-more aligned rows continue
/// through work that contributes to the requested logits.
pub(crate) fn tail_split(b: u32, align: u32) -> Option<(u32, u32)> {
    tail_split_min_rows(b, align, 64)
}

/// Variant for a backend-selected row-local tail. The workflow may use this
/// only after its final state write, when discarded rows cannot affect KV or
/// recurrent state.
pub(crate) fn tail_split_min_rows(
    b: u32,
    align: u32,
    min_rows: u32,
) -> Option<(u32, u32)> {
    let align = align.max(1);
    let min_rows = min_rows.max(1);
    let r0 = (b.checked_sub(min_rows)? / align) * align;
    let rows = b - r0;
    (rows >= min_rows).then_some((r0, rows))
}

/// Row alignment required by the current attention query tiles. This remains a
/// probe rather than a tuned kernel choice.
pub(crate) fn tail_align() -> u32 {
    std::env::var("IMPARO_TAIL_ALIGN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16)
}

/// `IMPARO_LAYER_SKIP_LOG=1`: print, per prefill chunk, how many layers ran (#127).
/// A probe, not a knob: it changes nothing but stderr.
pub(crate) fn layer_skip_log() -> bool {
    std::env::var("IMPARO_LAYER_SKIP_LOG").is_ok_and(|v| v == "1")
}

/// IMPARO_COBATCH_TRACE=1: a hash of every row of `buf` at this point of the co-batched step,
/// on stderr. Ends and restarts the command buffer; off, it does nothing at all.
pub(crate) fn trace_rows(tag: &str, layer: usize, buf: BufId, rows: u32, width: u32) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("IMPARO_COBATCH_TRACE").is_some()) {
        return;
    }
    let _ = be().end();
    let mut v = vec![0.0_f32; (rows * width) as usize];
    be().read(buf, 0, &mut v);
    for (r, row) in v.chunks(width as usize).enumerate() {
        let h = row.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, x| {
            (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3)
        });
        eprintln!("[cobatch] {tag} layer={layer} row={r} hash={h:016x}");
    }
    be().begin();
}

/// The convolution window of every recurrent layer: `r_elems / n_embd` inputs of `n_embd`
/// values, which is what a tree verify must keep per row so a commit can rebuild the
/// accepted path's windows.
///
/// Shared rather than per model: LFM2 and LFM2-MoE hold the same short convolution, and a
/// second copy of this rule is a second place for it to drift.
#[must_use]
pub fn conv_windows(plan: &crate::ModelPlan) -> Vec<crate::ConvWindow> {
    let width = plan.config.n_embd;
    plan.layers
        .iter()
        .enumerate()
        .filter_map(|(layer, l)| match l.attention {
            crate::Attention::Recurrent { r_elems, .. }
                if width != 0 && r_elems != 0 && r_elems % width == 0 =>
            {
                Some(crate::ConvWindow {
                    layer: u32::try_from(layer).ok()?,
                    width,
                    history: r_elems / width,
                })
            }
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod plan_tests {
    use super::*;

    const PAGE: u64 = 16384;
    fn page(b: u64) -> u64 {
        b.div_ceil(PAGE) * PAGE
    }
    fn req(id: BufId, bytes: u64, placement: Placement) -> BufferRequirement {
        BufferRequirement {
            id,
            bytes,
            placement,
        }
    }

    #[test]
    fn paged_kv_scratch_preserves_pool_extent_across_batch_resizes() {
        for batch in [512_u64, 2, 1] {
            let mut reqs = vec![
                req(BufId::Q, batch * 4096 * 4, Placement::Dedicated),
                req(BufId::Kdq, 6656 * 1024 * 2, Placement::Dedicated),
                req(BufId::Vdq, 6656 * 1024 * 2, Placement::Dedicated),
            ];
            fit_paged_kv_scratch(&mut reqs, 6656, 13312).unwrap();
            assert_eq!(reqs[0].bytes, batch * 4096 * 4);
            assert_eq!(reqs[1].bytes, 13312 * 1024 * 2);
            assert_eq!(reqs[2].bytes, 13312 * 1024 * 2);
        }
    }

    #[test]
    fn paged_kv_scratch_retains_floor_and_absent_mirrors() {
        let floor = crate::kv::KV_FIRST_SLOTS;
        let mut reqs = vec![req(BufId::Kdq, floor as u64 * 2048, Placement::Dedicated)];
        fit_paged_kv_scratch(&mut reqs, 64, 128).unwrap();
        assert_eq!(reqs[0].bytes, floor as u64 * 2048);
        fit_paged_kv_scratch(&mut [], 6656, 13312).unwrap();
    }

    #[test]
    fn paged_kv_scratch_rejects_invalid_row_layout_and_overflow() {
        let mut bad = [req(BufId::Kdq, 6657, Placement::Dedicated)];
        assert!(fit_paged_kv_scratch(&mut bad, 6656, 13312).is_err());
        let mut large = [req(BufId::Vdq, 6656 * 2048, Placement::Dedicated)];
        assert!(fit_paged_kv_scratch(&mut large, 6656, usize::MAX).is_err());
    }

    /// The #194 defect: an alias another group reaches into is promoted to its own
    /// allocation by the allocator, and the reserve has to count it. The mixer group (1) is
    /// larger than the host's group (0), so the alias at offset 0 of the host sits inside
    /// bytes group 1 also writes.
    #[test]
    fn a_promoted_alias_is_counted_in_the_plan_bytes() {
        let reqs = [
            req(BufId::X, 1 << 20, Placement::Group(0)),
            req(BufId::Cur, 3 << 20, Placement::Group(1)),
            req(
                BufId::Q,
                512 << 10,
                Placement::Within {
                    host: BufId::X,
                    slot: 0,
                },
            ),
        ];
        let plan = plan_layout_with(&reqs, true, &page);
        assert_eq!(plan.arena, 3 << 20, "the arena is the largest group");
        assert_eq!(
            plan.bytes,
            (3 << 20) + page(512 << 10),
            "the promoted alias is allocated"
        );
        assert!(matches!(
            plan.steps.iter().find(|s| s.id == BufId::Q).unwrap().action,
            Action::AliasDedicated { host: BufId::X, slot: 0, host_off: 0, skip: 0, others_reach } if others_reach == 3 << 20
        ));
        assert!(
            !plan.regions.contains_key(&(BufId::Q as u32)),
            "a dedicated alias has no arena region"
        );
    }

    /// Hosted in the largest group AND past every other group's end (group 0 reaches 1 MiB,
    /// slot 2 of a 512 KiB alias starts at 1 MiB), the alias costs nothing and lands inside
    /// its host. Slot 1 would start at 512 KiB, inside group 0's reach, and be promoted.
    #[test]
    fn an_alias_past_every_other_groups_reach_is_free() {
        let reqs = [
            req(BufId::X, 1 << 20, Placement::Group(0)),
            req(BufId::Cur, 3 << 20, Placement::Group(1)),
            req(
                BufId::Q,
                512 << 10,
                Placement::Within {
                    host: BufId::Cur,
                    slot: 2,
                },
            ),
        ];
        let plan = plan_layout_with(&reqs, true, &page);
        assert_eq!(plan.bytes, 3 << 20);
        assert_eq!(
            plan.regions.get(&(BufId::Q as u32)).copied(),
            Some((1 << 20, 512 << 10))
        );
        assert!(matches!(
            plan.steps.iter().find(|s| s.id == BufId::Q).unwrap().action,
            Action::AliasPlace { host: BufId::Cur, slot: 2, host_off: 0, skip } if skip == 1 << 20
        ));
        let inside = [
            req(BufId::X, 1 << 20, Placement::Group(0)),
            req(BufId::Cur, 3 << 20, Placement::Group(1)),
            req(
                BufId::Q,
                512 << 10,
                Placement::Within {
                    host: BufId::Cur,
                    slot: 1,
                },
            ),
        ];
        let plan = plan_layout_with(&inside, true, &page);
        assert_eq!(plan.bytes, (3 << 20) + (512 << 10));
        assert!(matches!(
            plan.steps.iter().find(|s| s.id == BufId::Q).unwrap().action,
            Action::AliasDedicated { others_reach, .. } if others_reach == 1 << 20
        ));
    }

    /// An alias that does not fit its host is skipped, allocates nothing, and one whose host
    /// is dedicated has nothing to alias into; a dedicated buffer is page-rounded into the sum.
    #[test]
    fn skipped_and_hostless_aliases_cost_nothing_and_dedicated_buffers_are_page_rounded()
     {
        let reqs = [
            req(BufId::X, 1 << 20, Placement::Group(0)),
            req(BufId::Cur, 100, Placement::Dedicated),
            req(
                BufId::Q,
                2 << 20,
                Placement::Within {
                    host: BufId::X,
                    slot: 0,
                },
            ),
            req(
                BufId::K,
                4096,
                Placement::Within {
                    host: BufId::Cur,
                    slot: 0,
                },
            ),
        ];
        let plan = plan_layout_with(&reqs, true, &page);
        assert_eq!(plan.bytes, (1 << 20) + PAGE);
        assert!(matches!(
            plan.steps.iter().find(|s| s.id == BufId::Q).unwrap().action,
            Action::AliasSkipped { host: BufId::X, .. }
        ));
        assert_eq!(
            plan.steps.iter().find(|s| s.id == BufId::K).unwrap().action,
            Action::AliasNoHost
        );
    }

    /// With overlap off the groups are laid end to end and nothing reaches into anything.
    #[test]
    fn without_overlap_groups_are_end_to_end_and_no_alias_is_promoted() {
        let reqs = [
            req(BufId::X, 1 << 20, Placement::Group(0)),
            req(BufId::Cur, 3 << 20, Placement::Group(1)),
            req(
                BufId::Q,
                512 << 10,
                Placement::Within {
                    host: BufId::X,
                    slot: 0,
                },
            ),
        ];
        let plan = plan_layout_with(&reqs, false, &page);
        assert_eq!(plan.arena, 4 << 20);
        assert_eq!(plan.bytes, 4 << 20);
        assert_eq!(
            plan.regions.get(&(BufId::Cur as u32)).copied(),
            Some((1 << 20, 3 << 20))
        );
        assert!(matches!(
            plan.steps.iter().find(|s| s.id == BufId::Q).unwrap().action,
            Action::AliasPlace { .. }
        ));
    }
}


#[cfg(all(test, any(feature = "cuda", feature = "cuda-dynamic")))]
mod row_submission_gpu_tests {
    use super::*;
    #[test]
    #[ignore = "requires an idle CUDA device and owns process-global state"]
    fn gpu_decode_rows_encoder_error_closes() {
        let backend=imparo_cuda::CudaBackend;
        let weights=Box::leak(vec![1.0_f32;1024].into_boxed_slice());
        unsafe { backend.init_weights(weights.as_ptr().cast(),4096) }.unwrap();
        for id in [BufId::Q,BufId::Attn,BufId::Recur,BufId::RecurSnap] {backend.alloc(id,1024).unwrap();}
        assert!(backend.supports_argmax_rows());
        backend.alloc(BufId::Logits,32).unwrap();backend.alloc(BufId::Tmp,8).unwrap();
        backend.write(BufId::Logits,0,&[0.0,3.0,1.0,2.0,8.0,4.0,2.0,0.0]);
        backend.argmax_rows(BufId::Logits,BufId::Tmp,4,2);backend.end().unwrap();
        let mut picks=[0.0_f32;2];backend.read(BufId::Tmp,0,&mut picks);
        assert_eq!(picks.map(f32::to_bits),[1,0]);
        backend.end().unwrap();assert!(backend.set_slots(2,&[]));
        let route=imparo_backend::RowRoute::Fast;
        let failed=submit_decode_rows(&backend,route,|| {
            backend.begin_forward(false);
            Err("injected encoder rejection".into())
        });
        assert!(failed.unwrap_err().contains("injected encoder rejection"));
        assert!(backend.select_slot(1),"encoder rejection left the forward open");
        assert!(!backend.head_norm_rope_at(BufId::Q,0,32,1e-6,1,&[0],32,10000.0,None),"row mode leaked");
        let failed=submit_decode_rows(&backend,route,|| {
            backend.begin_forward(false);
            // Invalid shape is rejected before any kernel launch; exercise pending-error drain.
            backend.causal_conv(imparo_backend::ConvForm::PlainSilu,BufId::Q,0,BufId::Recur,0,0,BufId::Attn,32,1,1);
            Ok(())
        });
        assert!(failed.unwrap_err().contains("co-batched step failed"));
        assert!(backend.select_slot(0),"native rejection left the forward open");
        submit_decode_rows(&backend,route,|| {backend.begin_forward(false);Ok(())}).unwrap();
        assert!(backend.select_slot(1),"next submission did not recover");
        // A two-row checkpoint crosses a real slot boundary after end(), and copies
        // the committed plane rather than the old plane into each slot's snapshot.
        for slot in [0,1] {
            assert!(backend.select_slot(slot));
            let values:Vec<f32>=(0..8).map(|i| (slot*100+i) as f32).collect();
            backend.write(BufId::Recur,0,&values);backend.end().unwrap();
        }
        assert!(backend.select_slot(0));
        submit_decode_rows(&backend,route,|| {backend.begin_forward(false);Ok(())}).unwrap();
        let rows:Vec<_>=[0,1].into_iter().map(|slot| crate::DecodeRow {
            slot,token:0,pos:63,plane_in:0,plane_out:1,snap:true,
        }).collect();
        snapshot_decode_rows(&backend,&rows,0,4).unwrap();
        let mut live=[0.0;4];backend.read(BufId::Recur,0,&mut live);
        assert_eq!(live,[0.0,1.0,2.0,3.0],"selected slot was not restored");
        for slot in [0,1] {
            assert!(backend.select_slot(slot));
            let mut got=[0.0;4];backend.read(BufId::RecurSnap,0,&mut got);
            assert_eq!(got,std::array::from_fn(|i| (slot*100+i as u32+4) as f32));
        }

    }
}
