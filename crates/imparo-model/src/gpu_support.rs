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
pub(crate) fn gprobe(name: &str, buf: BufId, off: u64, n: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
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
/// bytes.  Both ALIAS `host`'s pages and cost zero memory, which is why `host` must be a
/// buffer that is dead during prefill -- `U` on a SwiGLU feed-forward, because the fused
/// epilogue writes G and leaves U untouched.  That is a REQUIREMENT, not a nicety: with
/// the epilogue unfused the up projection writes U while the mirror of its own input
/// still lives in U's pages, and the dispatch reads and writes the same bytes.  A slot
/// that does not fit inside `host` is skipped, and the backend keeps converting inline.
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

/// Size the arena, place the grouped buffers, allocate the dedicated ones, and hand back
/// where everything landed.
///
/// Groups share bytes: the arena is sized to the LARGEST group, and every group starts at
/// offset zero. IMPARO_NO_ARENA_OVERLAP lays them end to end instead, which is the A/B for
/// whether an aliasing bug is an aliasing bug.
/// The bytes `place_buffers` would allocate for `reqs`, without allocating: the arena (the
/// largest group when groups overlap, their sum when they do not) plus every dedicated
/// buffer, page-rounded the way the backend rounds. The activation term of the fast-tier
/// reserve (docs/memory-tiers-and-fit.md section 2) is this number at the largest batch.
#[must_use]
pub fn layout_bytes(reqs: &[BufferRequirement]) -> u64 {
    let overlap = std::env::var("IMPARO_NO_ARENA_OVERLAP").is_err();
    let mut group_bytes: BTreeMap<u8, u64> = BTreeMap::new();
    let mut dedicated = 0_u64;
    for r in reqs {
        match r.placement {
            Placement::Group(g) => {
                *group_bytes.entry(g).or_insert(0) += be().page_round(r.bytes);
            }
            Placement::Dedicated => dedicated += be().page_round(r.bytes),
            Placement::Within { .. } => {}
        }
    }
    let arena = if overlap {
        group_bytes.values().copied().max().unwrap_or(0)
    } else {
        group_bytes.values().sum()
    };
    arena + dedicated
}

pub fn place_buffers(reqs: &[BufferRequirement]) -> Result<Layout, String> {
    let overlap = std::env::var("IMPARO_NO_ARENA_OVERLAP").is_err();
    let mut group_bytes: BTreeMap<u8, u64> = BTreeMap::new();
    for r in reqs {
        if let Placement::Group(g) = r.placement {
            *group_bytes.entry(g).or_insert(0) += be().page_round(r.bytes);
        }
    }
    let arena = if overlap {
        group_bytes.values().copied().max().unwrap_or(0)
    } else {
        group_bytes.values().sum()
    };
    be().arena(arena)
        .map_err(|rc| format!("metal arena rc={rc} (asked for {arena} bytes)"))?;

    // Where each group starts: zero when they share, cumulative when they do not.
    let mut group_start: BTreeMap<u8, u64> = BTreeMap::new();
    let mut running = 0_u64;
    for (&g, &bytes) in &group_bytes {
        group_start.insert(g, if overlap { 0 } else { running });
        running += bytes;
    }

    let mut at: BTreeMap<u8, u64> = group_start;
    let mut regions = BTreeMap::new();
    let mut bytes = arena;
    for r in reqs {
        match r.placement {
            Placement::Dedicated => {
                bytes += be().page_round(r.bytes);
                be().alloc(r.id, r.bytes)
                    .map_err(|rc| format!("metal alloc {:?} failed rc={rc}", r.id))?;
            }
            Placement::Group(g) => {
                let off = *at.get(&g).unwrap_or(&0);
                be().place(r.id, off, r.bytes)
                    .map_err(|rc| format!("metal place {:?} rc={rc}", r.id))?;
                regions.insert(r.id as u32, (off, r.bytes));
                at.insert(g, off + be().page_round(r.bytes));
            }
            // Handled below: an alias needs the region it lands in to exist already.
            Placement::Within { .. } => {}
        }
    }
    for r in reqs {
        let Placement::Within { host, slot } = r.placement else {
            continue;
        };
        let Some((host_off, host_bytes)) = regions.get(&(host as u32)).copied() else {
            continue; // host was dedicated or forced out; nothing to alias into
        };
        let skip = be().page_round(r.bytes) * u64::from(slot);
        // A SKIPPED alias is invisible in the output -- the backend's slower path is
        // still correct -- so it has to be visible in the log, or a performance cliff
        // has no explanation. Absence of evidence is not evidence.
        if skip + r.bytes > host_bytes {
            if crate::log_on() {
                eprintln!(
                    "[imparo] alias {:?} slot {slot} SKIPPED: {} + {} > {host_bytes} \
                     inside {host:?}",
                    r.id, skip, r.bytes
                );
            }
            continue;
        }
        be().place(r.id, host_off + skip, r.bytes)
            .map_err(|rc| format!("metal place {:?} rc={rc}", r.id))?;
        if crate::log_on() {
            eprintln!(
                "[imparo] alias {:?} slot {slot} at {}+{skip} ({} bytes) inside {host:?}",
                r.id, host_off, r.bytes
            );
        }
        regions.insert(r.id as u32, (host_off + skip, r.bytes));
    }
    Ok(Layout { regions, bytes })
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
    pub fn gpu_alloc_activations(&mut self, b_req: usize) -> Result<u64, String> {
        let b = batch_floor(b_req);
        let reqs = A::buffer_requirements(&self.plan, b, self.state.kv_rt.capacity);
        let layout = place_buffers(&reqs)?;
        self.state.gpu_batch = b_req;
        Ok(layout.total_bytes())
    }

    /// Re-sizes the activation buffers when the batch width changes, and only then.
    ///
    /// # Errors
    /// When the backend cannot allocate.
    pub fn gpu_fit_batch(&mut self, b: usize) -> Result<(), String> {
        if b == self.state.gpu_batch {
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
        self.gpu_prepare(crate::prefill_batch())?;
        self.state.kv_rt.slots = crate::kv::KV_FIRST_SLOTS;
        self.state.gpu_ready = true;
        Ok(())
    }

    /// Allocates activation and KV buffers for a batch of `max_batch` tokens.
    ///
    /// # Errors
    /// When the architecture has no device forward, or the backend cannot allocate.
    pub fn gpu_prepare(&mut self, max_batch: usize) -> Result<(), String> {
        if !A::DEVICE {
            return Err(crate::no_device_workflow(&self.plan));
        }
        // Set BEFORE any kv_bytes_for call: it sizes the windowed rings, and a ring cannot
        // be resized later without invalidating every position already mapped into it.
        self.state.kv_ring_batch = max_batch;
        let act_bytes = self.gpu_alloc_activations(max_batch)?;

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
            be().alloc(BufId::Recur, u64::from(recur) * 4)
                .map_err(|rc| format!("metal alloc recurrent state rc={rc}"))?;
            // The snapshot twin, same size: where the state at a boundary INSIDE a batch
            // is written. See `arm_recurrent_snapshot`.
            be().alloc(BufId::RecurSnap, u64::from(recur) * 4)
                .map_err(|rc| format!("metal alloc recurrent snapshot rc={rc}"))?;
            // The pre-step copies a decode step is rolled back from: one per pipe slot,
            // since two steps can be in flight and a failed one is rolled back to the
            // state before IT, not before the one queued behind it.
            be().alloc(BufId::RecurPrev, u64::from(recur) * 4 * 2)
                .map_err(|rc| format!("metal alloc recurrent rollback rc={rc}"))?;
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
        self.log_kv_groups(&kv_bytes);
        crate::host::log_footprint("gpu activations");
        be().alloc_kv_layout(&kv_bytes, &kv_layout)
            .map_err(|rc| format!("metal kv alloc failed rc={rc}"))?;
        crate::host::log_footprint("gpu kv");

        // Persistent transformed weights are admitted only after higher-priority
        // activation and KV allocations. Architectures without such a cache inherit a
        // no-op, and each backend retains fail-closed authority over memory fit.
        A::device_prepare(self)?;

        // Account for what the engine ACTUALLY allocates, from the layout that did it.
        // Footprint once read 452 MiB against a hand estimate of ~138, and guessing at
        // the difference is how you optimise the wrong thing.
        let kv_total: u64 = kv_bytes.iter().sum::<u64>() * 2; // K and V
        eprintln!(
            "[imparo] gpu alloc: kv={:.1} MiB activations={:.1} MiB (max_batch={max_batch})",
            kv_total as f64 / (1 << 20) as f64,
            act_bytes as f64 / (1 << 20) as f64
        );
        Ok(())
    }

    /// Makes sure every layer's device cache can hold `positions`, keeping the contents.
    ///
    /// # Errors
    /// When the backend cannot allocate.
    pub fn kv_fit(&mut self, positions: usize) -> Result<(), String> {
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
        Ok(())
    }

    /// Clears the device recurrent state: the device twin of `RecurrentState::reset`.
    ///
    /// A conversation that resumed on a dirty state gives the right shape and the wrong
    /// numbers. Allocating the zeros here rather than keeping them resident because this
    /// runs once per conversation, not once per token.
    pub fn zero_recurrent(&self) {
        let n = self.plan.recurrent_elems() as usize;
        if n > 0 {
            let zeros = vec![0.0_f32; n];
            be().write(BufId::Recur, 0, &zeros);
            // The snapshot twin too. A layer kind writes only the part of the state it
            // owns -- a short convolution writes its history and nothing else -- so any
            // region no snapshot dispatch covers would be read back as whatever the
            // allocation happened to contain. LFM2 has no such region (s_elems = 0); a
            // gated-delta-rule model would.
            be().write(BufId::RecurSnap, 0, &zeros);
        }
    }

    /// Tells the batch about to run whether a checkpoint boundary falls inside it, and
    /// where.
    ///
    /// A checkpoint for a boundary cannot be read off the device afterwards -- the live
    /// buffer holds only "now". The batch therefore writes it aside as it passes, and
    /// `state.recur_snap` is how the model's graph is told to: `Some(k)` means "the
    /// boundary is k tokens into this batch", and the graph dispatches one extra
    /// state-shaped kernel into `BufId::RecurSnap`.
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
        self.state.recur_snap = (self.plan.recurrent_elems() > 0
            && last > at
            && last <= at + n
            && last % imparo_kv::grid_tokens() == 0)
            .then(|| u32::try_from(last - at).expect("chunk fits u32"));
    }

    /// Reads back what the armed batch wrote aside, and disarms.
    ///
    /// Called after the batch whatever happened, so an armed flag never survives into a
    /// batch that was not told about it -- a stale `Some(k)` would have the next graph
    /// snapshot a boundary that is not there.
    pub fn take_recurrent_snapshot(&mut self, at: usize) {
        let Some(k) = self.state.recur_snap.take() else {
            return;
        };
        let n = self.plan.recurrent_elems() as usize;
        self.state.recur_ckpt = crate::kv::read_recurrent(n, BufId::RecurSnap);
        self.state.recur_ckpt_at = at + k as usize;
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
