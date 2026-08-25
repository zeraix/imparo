//! The backend seam (task #16). llama.cpp's LAYERING without ggml's graph engine:
//! a trait shaped by the REAL op surface the gemma4 workflow uses -- derived from
//! gemma4_metal.rs's call list, not idealized -- plus named buffer slots each backend
//! maps to its own allocations. Model workflows are hand-scheduled functions calling
//! this trait directly; there is no graph IR, no scheduler, no fusion pass (user
//! decision, 2026-08-20: the direct style beat the fork's graph machinery on host
//! overhead, and simplicity is the point of the handoff).
//!
//! Registry/buffer-slot ideas are borrowed from ggml-backend.h as ideas only.
//!
//! Tuned per-backend geometry (the set_* knob family) is deliberately NOT on the
//! trait: knobs are a backend's private business, applied from the shared
//! imparo-host store by the backend's own apply_host_config.

/// Named activation/scratch slots. Each backend maps a slot to its own allocation;
/// the ids are the cross-backend vocabulary the workflows speak.
///
/// The discriminants are today's wire values (imparo-metal's `buf` module), so the
/// Metal implementation is a pass-through.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum BufId {
    X = 0,
    Cur = 1,
    Q = 2,
    K = 3,
    V = 4,
    Attn = 5,
    O = 6,
    G = 7,
    U = 8,
    /// MODEL-PRIVATE SLOTS. A buffer only one architecture has does not belong in a
    /// shared enum under that architecture's name: three of these were Gate, Back and
    /// PerLayer, which exist solely for gemma4's per-layer embeddings, while LFM2 had no
    /// way to name its ShortConv projection at all. A model gives its own slots names
    /// (`const GATE: BufId = BufId::Model0;`) and the backend only ever sees an index.
    Model0 = 9,
    Model1 = 10,
    Model2 = 11,
    Logits = 12,
    Tokens = 13,
    Tmp = 14,
    AttnPart = 15,
    Xh = 16,
    Kdq = 17,
    Vdq = 18,
    Xh2 = 19,
    // Appended rather than interleaved: every index above is a wire value the backends
    // already use. Five more because a linear-attention block needs more scratch than a
    // transformer one -- Qwen3.5's gated delta rule carries a conv state, a state matrix
    // and its own intermediates.
    Model3 = 20,
    Model4 = 21,
    Model5 = 22,
    Model6 = 23,
    Model7 = 24,
    /// Per-conversation RECURRENT state: a short convolution's history, a linear
    /// attention's state matrix. Shared rather than model-private because every
    /// recurrent architecture needs exactly one of these and they all need it for the
    /// same reason -- and because it is the one activation-side buffer that must SURVIVE
    /// a batch-width change, which the machinery has to know without asking the model.
    ///
    /// NOT a KV-pool tenant: the pool exists for state that grows with context, and this
    /// is `n_embd * (l_cache - 1)` floats however long the conversation runs.
    Recur = 25,
    /// Where a recurrent snapshot is written when the boundary it describes falls INSIDE
    /// a batch. Same shape as `Recur` and shared for the same reason.
    ///
    /// Why a second buffer rather than stopping the batch on the boundary: chunks are cut
    /// on the prefill grid and pool units are 256 tokens, so a boundary routinely lands
    /// mid-chunk. Cutting the chunk to stand on it measured +29 ms on an 800-token
    /// prefill. The state at that boundary does not need standing on -- it is the tail of
    /// values the batch already computed -- so it is written aside by one small dispatch
    /// while the batch runs to its natural end.
    RecurSnap = 26,
}

impl BufId {
    /// Entries a backend's buffer table must hold.
    ///
    /// THIS is the definition; a backend's own size is a CAPACITY, not a second opinion.
    /// Adding a model-private slot is one edit here, and it stays one edit as long as the
    /// backends have room -- which is why they are sized well above this rather than
    /// exactly at it. `check_buf_table` turns "the backend is smaller than the enum" from
    /// an out-of-bounds index into a message at init.
    ///
    /// Three declarations of one number is how NO_WEIGHT ended up with two different
    /// values in this codebase, so there is one declaration and a check.
    pub const COUNT: usize = 27;
}

/// Fail loudly when a backend's buffer table cannot hold every `BufId`.
///
/// # Errors
/// Returns a message naming both sizes.
pub fn check_buf_table(backend: &str, capacity: usize) -> Result<(), String> {
    if capacity < BufId::COUNT {
        return Err(format!(
            "{backend} buffer table holds {capacity} entries but BufId needs {};              the backend would index out of bounds",
            BufId::COUNT
        ));
    }
    Ok(())
}

/// The weight-type -> kernel table index (see imparo-cpu's WeightKind, whose
/// discriminants match this wire).
pub type WeightKindWire = u32;

/// Sentinel w_off meaning "no weight vector" for rms_norm (normalize only).
///
/// THE definition -- backends must compare against this and nothing else. It was
/// 0xFFFF_FFFF, and two other places disagreed with it: imparo-metal declared its own copy
/// and the CUDA backend tested `w_off != u64::MAX`, so a no-weight norm on CUDA read a
/// weight vector at offset 4294967295. All-ones in 64 bits cannot be a real tensor offset,
/// which is what the Metal kernel's IMPARO_NO_WEIGHT now is too.
pub const NO_WEIGHT: u64 = u64::MAX;

/// One execution backend. Method-for-method the surface `gemma4_metal.rs` actually
/// calls; buffer arguments are `BufId`, weights are addressed by byte offset into
/// the backend's mapped weight blob.
///
/// KV ownership (alloc_kv/grow_kv/kv_store/kv_dequant/read_kv_bytes) sits behind
/// this trait ON PURPOSE: the unified-kv-pool + disk tier implements these same
/// calls without touching any workflow.
#[allow(clippy::too_many_arguments)]
/// The gated activation a matmul kernel fuses into its write-back.
///
/// The wire values are what the kernels compare against, so they are FIXED; `EPI_NONE`,
/// `EPI_GELU` and `EPI_SILU` in imparo.metal must agree with them.
///
/// This was a `bool`. "Epilogue on" meant GELU, because gemma4 was the only model, and a
/// SwiGLU model that turned it on would have got GELU and plausible wrong numbers -- the
/// knob table carried a comment warning not to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Epilogue {
    /// Write the product; no activation.
    None = 0,
    /// `y = gelu(y) * product`. gemma4's feed-forward.
    Gelu = 1,
    /// `y = silu(y) * product`. SwiGLU -- LFM2's, and most other architectures'.
    Silu = 2,
}

pub trait Backend {
    // --- session ---
    fn begin(&self);
    /// Commit what is encoded and return WITHOUT waiting, so the GPU runs it while the
    /// CPU encodes what follows. A read after this sees whatever was there before.
    fn flush(&self);
    /// Commit and WAIT. The only call after which a `read` is meaningful.
    ///
    /// # Errors
    /// Returns the command buffer's error code.
    fn end(&self) -> Result<(), i32>;
    /// GPU microseconds of the last `begin`/`end` region, or 0.0 if this backend cannot
    /// report it. Timing a candidate with a wall clock also charges submission latency;
    /// where the device can report its own busy time, the tuner uses that instead.
    fn last_gpu_us(&self) -> f64 {
        0.0
    }
    /// Queried and measured device limits, for the derivations. Default is all zero --
    /// "nothing established" -- so a backend that has not implemented it cannot have a
    /// derivation silently compute against a made-up number.
    fn device_profile(&self) -> DeviceProfile {
        DeviceProfile::default()
    }
    /// TFLOPS holding the `idx`-th candidate accumulator count live. Sweeping it finds
    /// the spill cliff. 0.0 where a backend has no such probe.
    fn spill_rate(&self, _idx: u32, _tgs: u32, _tpg: u32, _iters: u32) -> f64 {
        0.0
    }
    /// Streaming read GB/s at a given working set. Sweeping it finds the cache knee.
    fn bw_read(&self, _bytes: u64, _reps: u32, _tgs: u32, _tpg: u32) -> f64 {
        0.0
    }
    /// TFLOPS with the prefill attention score loop's own operand mix. This is what the
    /// score phase's measured rate should be compared against. 0.0 where a backend has no
    /// such probe.
    fn scoremix_rate(&self, _tgs: u32, _sgs: u32, _iters: u32, _stride: u32, _kspan: u32) -> f64 {
        0.0
    }
    /// Microseconds to SUBMIT an empty command buffer without waiting (or the backend's
    /// equivalent submission unit). 0.0 where a backend has no such probe.
    fn commit_overhead(&self, _n: u32) -> f64 {
        0.0
    }
    /// Microseconds per submit AND WAIT -- a different, much larger quantity.
    fn sync_overhead(&self, _n: u32) -> f64 {
        0.0
    }
    /// The cache types this backend is ACTUALLY configured with, as a short tag.
    ///
    /// A self-check, not a convenience: the tuner keys its stored config by the cache type
    /// it believes it is tuning for, and that belief comes from a flag. If the flag did not
    /// reach the backend, every per-cache-type answer it records is mislabelled. Asking the
    /// backend what it actually holds is the only way to catch that.
    fn kv_tag(&self) -> String {
        "f16".to_string()
    }
    /// Microseconds the host spends ENCODING one dispatch, with no execution in it.
    fn encode_cost(&self, _n: u32) -> f64 {
        0.0
    }
    /// Command-buffer length policy: layers per flush at decode / prefill.
    fn flush_layers(&self, decode: bool) -> u32;

    // --- buffers and arena ---
    fn alloc(&self, id: BufId, bytes: u64) -> Result<(), i32>;
    fn arena(&self, bytes: u64) -> Result<(), i32>;
    fn place(&self, id: BufId, offset: u64, bytes: u64) -> Result<(), i32>;
    fn page_round(&self, n: u64) -> u64;
    fn alloc_kv(&self, bytes: &[u64]) -> Result<(), i32>;
    fn grow_kv(&self, bytes: &[u64]) -> Result<(), i32>;
    fn write(&self, id: BufId, off: u64, src: &[f32]);
    fn write_u32(&self, id: BufId, off: u64, src: &[u32]);
    fn read(&self, id: BufId, off: u64, dst: &mut [f32]);
    fn read_kv_bytes(&self, layer: u32, is_v: bool, off: u64, dst: &mut [u8]);
    /// The KV pool's restore path: raw bytes back into a layer's cache (device idle).
    fn write_kv_bytes(&self, layer: u32, is_v: bool, off: u64, src: &[u8]);
    /// The pool's placement decision for one layer: entry i maps positions
    /// [i*64, i*64+64) to physical block entries[i]. Capability-gated on
    /// `pool_caps().paged_reads` growing true per backend.
    fn set_kv_page_table(&self, layer: u32, entries: &[u32]);
    /// Which REGION of a windowed layer's cache the resident conversation owns, as a byte
    /// offset into that layer's K and V caches.
    ///
    /// A windowed layer holds one ring per conversation that may be live at once. Pointing
    /// the layer at a conversation's own ring keeps every kernel addressing 0..ring-1, so
    /// the ring rule, the wrap test and the quantized dequant scratch are untouched by
    /// regions -- and a conversation's window stops being clobbered by whoever ran last.
    /// Pooled (full-attention) layers place through the block table and stay at 0.
    /// Default: no-op, for a backend that has not grown regions yet.
    fn set_kv_region(&self, _layer: u32, _k_off: u64, _v_off: u64) {}
    /// Tell the OS the pages backing a freed KV range are discardable
    /// (MADV_FREE-style). The pool calls this when residency is dropped so the
    /// footprint actually falls; content is gone, the buffer stays valid.
    /// Default: no-op (a backend without page-level release just keeps the pages).
    fn kv_advise_free(&self, _layer: u32, _is_v: bool, _off: u64, _len: u64) {}
    /// Recommit a previously advise-freed range (call before the block is
    /// written again). Default: no-op.
    fn kv_advise_reuse(&self, _layer: u32, _is_v: bool, _off: u64, _len: u64) {}

    // --- compute ---
    fn matmat(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        n_in: u32,
        n_out: u32,
        src: BufId,
        dst: BufId,
        n_tok: u32,
    );
    fn matmat_from(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        n_in: u32,
        n_out: u32,
        src: BufId,
        dst: BufId,
        n_tok: u32,
        src_row: u32,
    );
    /// Which gated activation the matmul kernels fuse into their write-back, or None.
    ///
    /// Sticky: it applies to every `matmat`/`matvec` until set again, so a caller sets it,
    /// dispatches, and sets it back.
    fn set_epilogue(&self, epi: Epilogue);

    /// LFM2's gated short convolution, for `n_tok` tokens of `width` channels.
    ///
    /// `bcx` is the input projection, token-major with three chunks of `width` per token
    /// (b, c, x in that order). `conv_w` is at `w_off`, channel-major with the tap
    /// fastest. `state` holds `kernel - 1` past values per channel, oldest first, at
    /// element offset `state_off`, and is ADVANCED in place.
    ///
    /// The state advance is a second dispatch, not part of this one: the new state is the
    /// tail of the same sequence the outputs read, so a single dispatch would have
    /// threads overwriting slots other threads still need, and a kernel cannot barrier
    /// its whole grid.
    fn shortconv(
        &self,
        bcx: BufId,
        w_off: u64,
        state: BufId,
        state_off: u32,
        out: BufId,
        width: u32,
        kernel: u32,
        n_tok: u32,
    );
    /// The short-convolution state as of `n_tok` tokens into this batch, written to
    /// `snap` and leaving `state` alone.
    ///
    /// This is what lets a batch run past a checkpoint boundary instead of being cut to
    /// end on one. The state after `n_tok` tokens is just the last `kernel - 1` values of
    /// `b * x` ending there, and `bcx` already holds them, so the mid-batch state is
    /// COMPUTED rather than stood on.
    ///
    /// Must be dispatched BEFORE `shortconv` advances `state`: with `n_tok` shorter than
    /// the history, part of the answer is the pre-batch state.
    fn shortconv_snapshot(
        &self,
        bcx: BufId,
        state: BufId,
        state_off: u32,
        snap: BufId,
        snap_off: u32,
        width: u32,
        kernel: u32,
        n_tok: u32,
    );
    /// One embedding-table row, dequantised and scaled into `dst`.
    ///
    /// `wkind` is the same weight-type table index `matmat` takes. It used to be absent,
    /// which made the op Q4_0 by assumption: a table with another layout would have been
    /// unpacked by the Q4_0 reader and produced plausible garbage.
    fn row(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        width: u32,
        index: u32,
        scale: f32,
        dst: BufId,
        dst_off: u32,
    );
    /// Every embedding row of a batch in ONE dispatch: `idx` holds `n_rows` u32 table
    /// indices, written by `write_u32`. Returns false when the backend has no kernel for
    /// this kind or cannot vectorise this width, in which case the caller must fall back
    /// to `row` per token -- which is what the prefill path did for every token before
    /// this existed. Default: no backend support.
    fn gather_rows(
        &self,
        _wkind: WeightKindWire,
        _w_off: u64,
        _width: u32,
        _table_rows: u32,
        _scale: f32,
        _dst: BufId,
        _dst_off: u32,
        _idx: BufId,
        _n_rows: u32,
    ) -> bool {
        false
    }
    fn rms_norm(
        &self,
        buf: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    );
    fn rms_norm_from(
        &self,
        buf: BufId,
        src: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    );
    fn rope(
        &self,
        buf: BufId,
        n_rot: u32,
        base: f32,
        head_dim: u32,
        n_heads: u32,
        start_pos: u32,
        n_tok: u32,
        freqs: Option<&[f32]>,
    );
    fn hadamard(&self, buf: BufId, n: u32, nrot: u32);
    fn kv_store(
        &self,
        src: BufId,
        layer: u32,
        width: u32,
        start_pos: u32,
        n_tok: u32,
        is_v: bool,
        ring: u32,
    );
    fn attention(
        &self,
        kv_layer: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv: u32,
        kv_width: u32,
        start_pos: u32,
        window: u32,
        n_tok: u32,
        max_scores: u32,
        ring: u32,
    );
    /// Which activation every epilogue applies, for this process.
    ///
    /// SPECIALISES the kernels, so it must be called before `init_weights`; afterwards it
    /// is a no-op. One value per process because that is what a model is -- gemma4 is
    /// GELU in every layer, LFM2 is SiLU in every layer.
    ///
    /// It is not a per-dispatch argument because making it one MEASURABLY changed the
    /// arithmetic: with the value still GELU, a runtime branch in the epilogue moved
    /// gemma4's n=16 logits from 25.582184 to 25.582018, reproducibly.
    fn set_activation(&self, act: Epilogue);
    /// `a = act(a)`, elementwise.
    fn act(&self, a: BufId, n: u32);
    /// `a = act(a) * b`, elementwise -- the DECODE form of the fused epilogue.
    ///
    /// Prefill folds this into the up projection's write-back. At one token that measured
    /// worse (38.2-38.6 -> 37.9-38.2 tok/s): a read-modify-write per output row inside
    /// the GEMV, against a wide vectorised pass here.
    fn act_mul(&self, a: BufId, b: BufId, n: u32);
    fn add(&self, a: BufId, b: BufId, n: u32);
    fn add_scale(&self, a: BufId, b: BufId, k: f32, n: u32);
    fn scale(&self, a: BufId, k: f32, n: u32);
    fn copy(&self, dst: BufId, src: BufId, n: u32);
    fn mul_strided(
        &self,
        a: BufId,
        b: BufId,
        n: u32,
        b_off: u32,
        b_stride: u32,
        a_stride: u32,
        n_tok: u32,
    );
    fn softcap(&self, a: BufId, cap: f32, n: u32);
    fn argmax(&self, src: BufId, dst: BufId, n: u32);
    fn ple_gather_combine(
        &self,
        proj: BufId,
        tokens_buf: BufId,
        w_offset: u64,
        width: u32,
        emb_scale: f32,
        comb_scale: f32,
        n_tok: u32,
    );

    // --- weights + config + identity ---
    /// Make the mapped weight blob available to the backend (Metal: share the mmap;
    /// CUDA: upload). Called once at load by the composition root, so imparo-cpu need
    /// not know any backend exists. `base`/`len` describe the CPU-visible mapping.
    ///
    /// # Safety
    /// `base` must point to `len` readable bytes that outlive every kernel.
    unsafe fn init_weights(&self, base: *const u8, len: u64) -> Result<(), i32>;

    /// Configure KV cache storage types (f16=1, q4_0=2, q8_0=8) before build.
    fn set_kv_types(&self, k: u32, v: u32);

    /// GPU profiling snapshot since the last read (empty when unsupported).
    fn prof_stats(&self) -> ProfStats;
    fn prof_enable(&self, on: bool);

    /// Bytes the backend currently has allocated on the device.
    fn allocated_bytes(&self) -> u64;

    /// Short device tag contributed to the host fingerprint (the #14 seam).
    fn device_tag(&self) -> String;

    /// The KV pool's capability descriptor for this backend.
    fn pool_caps(&self) -> PoolCaps;
}

/// One storage tier a backend can place KV state in, nearest-compute first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    /// Discrete device memory (CUDA/ROCm): explicit allocations, no lazy commit.
    Device,
    /// Host RAM as a real intermediate (discrete platforms only).
    Host,
    /// One address space for CPU and GPU (Apple Silicon, integrated).
    Unified,
    /// Durable storage; never a backing store for live decode.
    Disk,
}

/// The pool's capability descriptor (docs/unified-kv-pool.md "Backend capability
/// descriptor").
///
/// What is ENFORCED, and what is only declared, because the difference has already
/// cost a wrong answer once:
///
/// ```text
/// paged_reads      enforced. The server refuses to run the pool without it: a block
///                  table installed for a kernel that ignores it reads somebody else's
///                  rows, and nothing would fail.
/// block_cells      enforced by the engine gate, not by code -- it is the byte-identity
///                  resume grid, proven per backend at every split point.
/// shared_address   DECLARED ONLY. Nothing reads it yet. It exists so the first mover
///                  between tiers consults it instead of assuming a unified address
///                  space, which is what a Metal-shaped pool would do on a discrete GPU.
/// tiers            DECLARED ONLY, same reason. Tier::Host has no mover today.
/// ```
///
/// The intent behind the last two is that a "copy" between CPU and GPU is a change of
/// reader on a shared address space and a PCIe transfer on a discrete one; a tier list
/// that omits Host on Metal is how the wasteful case is meant to become unrepresentable.
/// It is not unrepresentable yet -- there is simply no code that moves between tiers.
#[derive(Clone, Copy, Debug)]
pub struct PoolCaps {
    /// Smallest block the attention kernel can index independently, derived from
    /// that kernel's tile geometry. Must divide the pool's unit size.
    pub block_cells: u32,
    /// Whether attention accepts a block table; without it the pool uses the
    /// gather-into-scratch or contiguous fallback.
    pub paged_reads: bool,
    /// Whether device and host address the same bytes.
    pub shared_address: bool,
    /// Ordered, nearest compute first. Disk last everywhere.
    pub tiers: &'static [Tier],
}

/// A backend-agnostic profiling snapshot (the wrapper in the model layer prints it
/// without any per-backend cfg).
#[derive(Clone, Debug, Default)]
pub struct ProfStats {
    pub gpu_s: f64,
    pub wall_s: f64,
    pub cbs: u64,
    pub dispatches: u64,
    /// (category name, ticks, calls) since the last read.
    pub categories: Vec<(String, f64, u64)>,
}

/// The four-category knob taxonomy (see imparo-tune's knobs.rs for the doctrine):
/// only `Benched` knobs are swept; the others are computed or profiled once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnobCategory {
    ModelShape,
    DeviceProfile,
    Arithmetic,
    Benched,
}


/// The stage-1 workload a Micro knob is judged on, spoken in Backend-trait ops so the
/// shared tuner builds it for any backend from the model's shapes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Workload {
    /// The decode-step matmuls at n_tok=1, at the model's real weight offsets, summed.
    DecodeMix,
    /// A narrow batch through the layer's big matmuls (the MTP verification shape).
    NarrowMix(u32),
    /// Single-query attention against a long context.
    AttentionDecode,
    /// Single-query attention against a context DEEP enough to reach the streaming
    /// decode kernel.
    ///
    /// AttentionDecode runs at position 512, and the streaming path only engages at
    /// `n_pos >= attn_stream_min_pos`, which is thousands. So every knob that governs
    /// the streaming kernel -- head sharing, slice count, the threadgroup floor -- was
    /// being ranked on a workload that never reached it, and their candidates duly
    /// landed within noise of each other. Same defect as a prefill tile knob declared on
    /// DecodeMix: the workload has to exercise the thing the knob selects.
    AttentionDecodeDeep,
    /// Batched prefill attention against a SHALLOW context -- a cold first chunk.
    AttentionPrefill,
    /// A full prefill chunk against a DEEP context: the shape a 16k prefill spends
    /// almost all of its attention time in, since the scan is the context length and a
    /// 16k scan is 32x a 512 one. Judging a prefill attention shape on the shallow
    /// workload ranks it on 3% of the work it will actually do.
    AttentionPrefillDeep,
    /// ONE DECODE STEP'S ATTENTION, across every layer at ITS OWN geometry and span.
    ///
    /// Every other attention workload here times a single dispatch, and for a model with
    /// one attention geometry that is the same thing. For a model with two it is not:
    /// gemma4 E4B runs 7 full-attention layers at 512 dims among 35 windowed ones at 256
    /// whose spans never exceed their window, so a step's attention is a MIX and a deep
    /// dispatch is 7/42 of it.
    ///
    /// The gap that produced this: the streaming-decode boundary measured on the isolated
    /// deep dispatch reported the score-tile path ahead at 16k, while the engine at 17k
    /// measured streaming 7.3% CHEAPER for exactly those layers (differenced against a run
    /// with IMPARO_SKIP_ATTN=2). Opposite signs, so the isolated dispatch is not what the
    /// engine does.
    DecodeAttentionStep,
    /// A SHORT prefill tail against a DEEP context -- what a prefix-matched turn runs.
    ///
    /// This is the agentic pattern, not an edge case: a continuation re-sends the whole
    /// conversation, the cache matches almost all of it, and only the tail is prefilled
    /// (STATUS.md records reused=640 of 721). The shape is neither of the other two --
    /// too few tokens for the wide prefill tile, too many for the decode path -- and
    /// until now no workload measured it, so no knob was ever chosen for the turn shape
    /// the engine most often sees.
    AttentionPrefillReuse,
    /// The layer's two widest matmuls at the PREFILL tile: n_embd->n_ff and n_ff->n_embd
    /// at 512 tokens. This is the shape the GEMM tile knobs actually govern. Judging them
    /// on DecodeMix instead -- n_tok=1 matvecs -- is measuring a different kernel path,
    /// and is why a micro-bench once ranked a tile 1.6% faster that was 11% slower in a
    /// real prefill. That was read as "micro-benchmarks cannot rank GEMM tiles, move the
    /// knob end-to-end"; it actually meant the workload was wrong.
    PrefillGemm,
}

/// How a Benched knob's value is found.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SweepKind {
    /// Not swept at all -- see `KnobDecl::derive`. Present so a derived knob can sit in
    /// the registry beside the benched ones instead of hiding in backend init.
    Derived,
    /// Interleaved sweep over `values` with a per-axis noise floor.
    Values,
    /// A routing boundary: at each rung `n` of `ladder`, a matmat of `n` tokens is timed
    /// with the knob forced to `hi` (its kernel covers n) against `lo` (it does not).
    /// The pick is the last width where the hi-side kernel keeps winning; scans run three
    /// times and must agree within one rung, else the compiled default stands.
    Crossing {
        ladder: &'static [u32],
        hi: u32,
        lo: u32,
    },
    /// A routing boundary measured on CONTEXT SPANS rather than token counts: at each
    /// rung the two decode-attention kernels race a single-query dispatch at that span,
    /// and the pick is the FIRST span where the `hi` side wins and keeps winning (the
    /// knob is a `>= threshold` test, so the pick is the rung itself).
    ///
    /// Separate from `Crossing` because the direction is inverted (the hi side loses
    /// while shallow and wins deep) and because the timed op is an attention dispatch,
    /// which has to rotate KV layers -- reps against one layer are served from the SLC
    /// and stop measuring memory at all.
    SpanCrossing {
        ladder: &'static [u32],
        hi: u32,
        lo: u32,
    },
}

/// One backend-owned performance knob. The shared tuner machinery consumes these;
/// a backend developer adds knobs HERE (their crate), never in shared code. REGISTRY
/// ORDER IS SWEEP ORDER: a knob whose measurement depends on another's pick (nb8_max's
/// crossing runs against the nb8_shape winner) is declared after it.
/// There is no `stage` field and no end-to-end search. Every knob here is chosen by math
/// or by a per-kernel micro-bench; a knob that can only be judged by running the whole
/// engine does not belong in this registry. Command-buffer length is the one property
/// with no per-kernel proxy by construction, and it is DERIVED from measured encode cost
/// and buffer turnaround rather than swept -- see the note above KNOBS in the backend.
/// What the tuner knows about the model in front of it, for deciding whether a knob
/// APPLIES at all. Shapes come from the GGUF header, so this costs no tensor mapping.
///
/// Not every knob is meaningful for every model. A knob selecting a MoE routing path has
/// nothing to say about a dense model; one that picks between head-dim-512 attention
/// kernels is irrelevant to a model with no such layers. Offering an inapplicable knob is
/// worse than useless: the tuner spends measurements on a value that changes nothing, and
/// then RECORDS a pick for it, which reads as a decision when nothing was decided.
/// What the tuner knows about the DEVICE. Queried where the API exposes it, measured
/// where it does not; a zero means "not established on this host yet", and a derivation
/// that needs one must say so rather than invent a number.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceProfile {
    /// Threadgroup memory a single threadgroup may declare. QUERIED.
    pub threadgroup_bytes: u64,
    /// Threads a single threadgroup may hold. QUERIED.
    pub max_threads: u32,
    /// Working set past which reads fall from cache rate to DRAM rate. MEASURED by
    /// sweeping the set and finding the knee; 0 until that probe runs at init.
    pub cache_knee_bytes: u64,
    /// Streaming read rate past the knee, MB/s. MEASURED; 0 until probed.
    pub dram_read_mbs: u32,
    /// Accumulator fragments one THREAD may hold before the compiler spills them to
    /// device memory. MEASURED by sweeping the count and finding the cliff; 0 until
    /// probed. Per thread and flat -- it does NOT divide among a threadgroup's threads,
    /// and assuming it did produced a bound that rejected a legal shape.
    pub max_accumulators: u32,
    /// The prefill attention score loop's OWN matrix-op ceiling, GFLOPS. MEASURED with
    /// that loop's operand mix -- staged Q re-read from threadgroup memory, K streamed
    /// from device -- not with a generic matrix peak. 0 until probed.
    ///
    /// Why it is a separate number: the score phase was judged against a peak measured on
    /// a DIFFERENT access pattern, which made it look like it was leaving most of the
    /// machine unused -- 37% of 7.3 TFLOPS. Measured with its own mix the ceiling is about
    /// 4.1, so it sits near 65%. A ceiling measured on the wrong access pattern is the
    /// ceiling of a kernel that does not exist.
    pub attn_score_ceiling_gflops: u32,
    /// THREADS a memory-bound dispatch needs in flight before the machine stops scaling.
    /// MEASURED; 0 until probed.
    ///
    /// Separate from `fill_threadgroups` because they are different quantities: an
    /// ALU-dense kernel saturates when the arithmetic units are busy, a latency-bound one
    /// not until enough loads are in flight to hide the memory. `attn_min_tgs` gates a
    /// decode attention dispatch, which is the latter.
    ///
    /// IN THREADS, NOT THREADGROUPS, and that is the whole portability of it. A
    /// threadgroup count only means something alongside the threads-per-threadgroup it was
    /// measured at -- the probe runs 256, the decode attention kernel runs `attn_threads`,
    /// and a consumer that took the count as-is would silently import the probe's choice.
    /// Threads divide out cleanly: a consumer divides by its OWN threadgroup size.
    pub fill_threads_membound: u32,
    /// Nanoseconds to SUBMIT an empty command buffer, not waited on -- what a mid-graph
    /// flush costs. MEASURED; 0 until probed.
    pub commit_overhead_ns: u32,
    /// Nanoseconds the host spends encoding one dispatch. MEASURED; 0 until probed.
    ///
    /// Together with `commit_overhead_ns` and the measured cost of a layer's work, these
    /// are what decide how many layers belong in one command buffer -- a trade between
    /// paying the fixed submission cost too often and letting the GPU idle while the host
    /// is still encoding. Neither cost is reported by any API, and a literal in their
    /// place is one machine's balance written down.
    pub encode_cost_ns: u32,
    /// GPU nanoseconds for ONE layer's decode matmuls, at THIS model's dimensions.
    /// MEASURED (the DecodeMix workload); 0 until probed.
    ///
    /// The one entry here that is not pure device ground truth, and it is deliberate: the
    /// trade it feeds -- how many layers belong in a command buffer -- is a model x device
    /// question, not a device one. A 2-billion-parameter layer and a 70-billion one cover
    /// the same fixed submission cost with very different amounts of work. Naming it for
    /// what it is beats pretending it belongs to the device.
    pub layer_work_ns: u32,
    /// GPU nanoseconds for ONE layer's PREFILL matmuls, at this model's dimensions and a
    /// full chunk. MEASURED (the PrefillGemm workload, scaled back up from its slice);
    /// 0 until probed.
    ///
    /// The term that separates the two flush knobs. A prefill layer carries hundreds of
    /// times the GPU work of a decode layer while costing the host the SAME encode, so the
    /// encode lag that justifies flushing during decode is noise during prefill.
    pub layer_work_prefill_ns: u32,

    /// Threadgroups needed before the GPU stops scaling -- the point at which adding more
    /// stops buying parallelism and starts queueing. MEASURED; 0 until probed.
    ///
    /// This is what a dispatch has to reach to be worth dispatching whole rather than
    /// sliced, and it is a pure DEVICE property: it follows core count, and a literal in
    /// its place is one machine's core count written down.
    pub fill_threadgroups: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelFacts {
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_head: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    /// Largest head dim across layers -- architectures mix geometries (gemma4 E4B runs
    /// 512 on its full-attention layers and 256 on its windowed ones).
    pub deep_head_dim: u32,
    /// 0 for a dense model. Non-zero enables the MoE knobs and nothing else.
    pub n_experts: u32,
    /// Blocks in the model. From the GGUF header.
    pub n_layers: u32,
    /// Compute dispatches ONE layer encodes in a decode step. Counted from the model's
    /// forward code, not measured -- the tuner never runs the graph, and the engine's
    /// dispatch counter reads 0 unless profiling is on.
    ///
    /// It is here because it is the work term in the command-buffer trade: what a flush
    /// costs the host is a submission, and what it buys is that the GPU can start on the
    /// layers already encoded.
    pub layer_dispatches: u32,
    /// Weight-kind wire values present on this model's layer projections, as a bitmask
    /// over `WeightKind` discriminants (bit 0 = F32, bit 1 = Q4_0, bit 2 = Q8_0).
    ///
    /// A knob that governs one quant's kernels must not be swept on a model that has no
    /// tensor of that quant: the workload would dispatch a different kernel family and
    /// rank the candidates on noise. `applies` reads this.
    pub weight_kinds: u32,
}

pub struct KnobDecl {
    pub name: &'static str,
    /// A SECOND regime the winner must not be bad in, or `None` to skip the check.
    ///
    /// A REGIME is a runtime state the engine works in -- a deep prefill chunk, a
    /// prefix-matched tail, a decode step. A VARIANT is a per-regime VALUE: one knob, or
    /// one tuple, answering differently in different regimes. A flip here is the signal
    /// that this knob needs variants rather than one value.
    ///
    /// A knob is swept on the regime where its kernel spends the most time, which is the
    /// right thing to optimise -- but the engine runs other regimes too, and one value
    /// has to serve all of them. This re-measures the winner against the runner-up on a
    /// second workload and says so if the order FLIPS. A flip is the signal that no
    /// single value serves both and the knob wants a per-regime value.
    ///
    /// The prefill attention shape is swept on the deep chunk (256 threadgroups) and
    /// cross-checked on the prefix-matched tail (32 threadgroups, under the ~72 that
    /// fill this GPU) -- very different occupancy, same kernel. Today they agree; the
    /// point of the check is to notice when they stop.
    pub cross_check: Option<Workload>,
    /// Whether this knob applies to the model under test. `None` means always.
    ///
    /// Checked BEFORE the screen, so an inapplicable knob costs no measurement and gets
    /// no recorded pick. The tuner says which knobs it skipped and why, because a knob
    /// silently missing from a config file is indistinguishable from one that was
    /// measured and left at its default.
    pub applies: Option<fn(&ModelFacts) -> bool>,
    /// Whether a VALUE can run at all here, or `None` if every declared value can.
    ///
    /// A candidate that provably cannot run should never reach a dispatch. Today such a
    /// value is rejected by MEASURING it: the screen catches it because it reads 292x off
    /// the pace, which costs a dispatch of a register-spilling kernel saturating the
    /// memory bus -- the thing that stalled a machine for thirty seconds. Legality is
    /// arithmetic; it belongs before the screen, not in it.
    ///
    /// The predicate should ASK THE BACKEND rather than restate its rule. The backend
    /// owns the shape tables and the register model; a copy of that arithmetic in the
    /// registry is a second source of truth that will drift, which is the failure this
    /// codebase keeps finding.
    pub legal: Option<fn(u32, &ModelFacts, &DeviceProfile) -> bool>,
    /// Whether moving this knob can change output BITS.
    ///
    /// The tuner ranks on TIME. A knob that also moves the numerics can therefore trade
    /// accuracy for speed with nothing noticing -- staging Q as half is worth real time
    /// and moved the q4 agreement from 0.301 to 1.025 against a 1.0 tolerance, and that
    /// trade was caught by a human, not by the tuner.
    ///
    /// Declaring it does not decide it. The tuner takes the choice as an argument:
    /// bit-affecting knobs are held at their incumbent unless the caller opts in, and
    /// opting in prints what it obliges -- regenerate the pins, re-check agreement.
    pub bit_affecting: bool,
    /// COMPUTED from ground truth instead of measured, when set.
    ///
    /// This is the "math decides" half of the design given a mechanism. A knob with a
    /// derivation is never swept: its value is a function of the model's shape and the
    /// device's limits, and searching for something you can compute is how a literal
    /// ends up frozen at whatever the author's machine happened to be.
    ///
    /// A derived value is REPORTED in the stored config and never APPLIED from it. The
    /// config records what this host computed; another host must compute its own, and
    /// restoring a number derived elsewhere is exactly the bug deriving it prevents.
    pub derive: Option<fn(&ModelFacts, &DeviceProfile) -> u32>,
    /// CANDIDATES computed from ground truth, when set, instead of `values`.
    ///
    /// The middle case between "math decides" and "the bench searches". Some knobs are
    /// not derivable -- their optimum is a trade the device settles, so it must be
    /// measured -- but the RANGE worth measuring still follows from a measured quantity,
    /// and a hand-typed list is one machine's range written down.
    ///
    /// `attn_min_tgs` is the example. It targets a threadgroup count for the sliced decode
    /// attention dispatch, so the memory-bound fill point is what sets its scale; but more
    /// slices also means more partials to combine, and where that trade lands is not
    /// something arithmetic answers. So the fill point picks the candidates and the bench
    /// picks among them.
    pub candidates: Option<fn(&ModelFacts, &DeviceProfile) -> Vec<u32>>,
    /// Knobs that must be SETTLED before this one is measured.
    ///
    /// This axis was rejected once, on the grounds that every real instance was a boundary
    /// inside the tuple whose regimes it defines -- which the tuple already orders. Then a
    /// measurement produced an instance that is neither, and across tuples:
    ///
    ///   attn_min_tgs = 72 (untuned)   the span scan finds streaming FASTER at every rung,
    ///                                 crossing at 2048
    ///   attn_min_tgs = 48 (tuned)     the same scan finds it SLOWER at every rung, no
    ///                                 crossing at all
    ///
    /// attn_min_tgs sets how the score-tile path slices, and the score-tile path is the
    /// side attn_stream_min_pos races against. Settle it first and the boundary moves.
    ///
    /// The registry order already satisfies this -- by accident of where the declarations
    /// sit in the file, which is exactly the fragility that motivated the axis. Declaring
    /// it does not reorder anything; it lets the tuner CHECK that the order holds, and say
    /// so if an edit breaks it.
    pub after: &'static [&'static str],
    /// The TUPLE this knob belongs to, or `None` for one that stands alone.
    ///
    /// A tuple is several knobs that collapse into ONE virtual knob: they sit at the same
    /// level and meet inside a single selection, so no member has a best value on its
    /// own, and what the tuner picks is the TUPLE -- that composite IS the value.
    ///
    /// The default search is coordinate-wise: hold everything, move one knob, keep it if
    /// it wins. That is only valid when the knobs are independent, and several here are
    /// not -- they meet inside one selection and their best values are joint:
    ///
    ///   lanes x nr0        one pipeline table, indexed by BOTH: p_q4mm_lanes[lanes][nr0]
    ///   nb8_shape x nb8_max   which narrow tile, and the batch size at which it engages;
    ///                      the right boundary depends on which tile sits behind it
    ///   the prefill attention shape   NSG, BLK and PT are each forced by a different
    ///                      limit (accumulator registers, unit count reaching NSG,
    ///                      threadgroup bytes), so no single move reaches the optimum --
    ///                      about ten attempts across two sessions failed exactly here
    ///
    /// A coordinate sweep over a coupled group reports "nothing beat the incumbent" and
    /// is believed, because every single step really is worse. Only the joint move wins.
    pub tuple: Option<&'static str>,
    pub category: KnobCategory,
    /// Candidate values for Values-swept knobs; empty for Crossing knobs.
    pub values: &'static [u32],
    pub apply: fn(u32),
    pub current: fn() -> u32,
    /// Candidates must pass the tiny-size screen before any full-size sweep -- the
    /// machine-safety rule: a register-spilling candidate reads ~35-80x there and
    /// would otherwise freeze the host with multi-second command buffers.
    pub screened: bool,
    pub sweep: SweepKind,
    /// Judged on this workload (Micro stage; EndToEnd knobs are measured on the model).
    /// The regime this knob is measured in.
    ///
    /// READ ONLY for `SweepKind::Values`. A Crossing or SpanCrossing knob drives its own
    /// ladder -- it is looking for where two paths cross, which is a different
    /// measurement from timing one -- and a Derived knob is not measured at all. The
    /// field is inert for those, and `--explain` says "own ladder" rather than repeating
    /// a value nothing reads.
    pub workload: Workload,
}

/// The registry half of a backend: its knobs and its OWN search-space version.
/// Versions are per backend so a CUDA space bump never invalidates a Metal config;
/// the stored-config fingerprint carries device_tag + this version.
pub trait BackendKnobs {
    fn knob_registry(&self) -> &'static [KnobDecl];
    fn space_version(&self) -> u32;
}
