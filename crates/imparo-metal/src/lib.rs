#![doc = "Metal backend: GPU kernels and GPU-resident activation buffers."]
//! Kernels are named for operation and quantisation, never for a model. Weights are shared
//! with the GPU without copying; activations stay in GPU buffers across an entire layer so
//! there is one synchronisation per forward pass rather than one per matmul.

#![cfg(target_os = "macos")]

pub use imparo_host as hostconfig;
pub mod backend_impl;
pub use backend_impl::MetalBackend;

/// Identifiers for the persistent activation buffers.
pub mod buf {
    pub const X: u32 = 0;
    pub const CUR: u32 = 1;
    pub const Q: u32 = 2;
    pub const K: u32 = 3;
    pub const V: u32 = 4;
    pub const ATTN: u32 = 5;
    pub const O: u32 = 6;
    pub const G: u32 = 7;
    pub const U: u32 = 8;
    pub const GATE: u32 = 9;
    pub const BACK: u32 = 10;
    pub const PER_LAYER: u32 = 11;
    pub const LOGITS: u32 = 12;
    pub const TOKENS: u32 = 13;
    pub const TMP: u32 = 14;
    /// Per-slice partials for split-KV decode attention.
    pub const ATTN_PART: u32 = 15;
    /// Half copy of a prefill matmul's activation input.
    pub const XH: u32 = 16;
    /// Half scratch a quantized KV cache is dequantised into for prefill attention.
    pub const KDQ: u32 = 17;
    pub const VDQ: u32 = 18;
    /// Second half mirror (the fused up epilogue writes G's half copy here while XH
    /// still holds CUR's); aliases the free half of U's arena region.
    pub const XH2: u32 = 19;
    pub const COUNT: u32 = 20;
}

unsafe extern "C" {
    fn imparo_metal_init(base: *const core::ffi::c_void, len: u64) -> i32;
    fn imparo_metal_tune(sgs: u32, rows: u32);
    fn imparo_metal_set_attn_threads(n: u32);
    fn imparo_metal_set_attn_threads_prefill(n: u32);
    fn imparo_metal_set_lanes(lanes: u32);
    fn imparo_metal_set_nr0(nr0: u32);
    fn imparo_metal_set_skip_mma(on: u32);
    fn imparo_metal_set_half_a(on: u32);
    fn imparo_metal_set_kv_region(layer: u32, k_off: u64, v_off: u64);
    fn imparo_metal_set_kv_types(k: u32, v: u32);
    fn imparo_metal_kv_types(k: *mut u32, v: *mut u32);
    fn imparo_metal_read_kv(layer: u32, is_v: u32, off: u64, dst: *mut u8, n: u64);
    fn imparo_metal_write_kv(layer: u32, is_v: u32, off: u64, src: *const u8, n: u64);
    fn imparo_metal_set_kv_pages(layer: u32, e: *const u32, n: u32);
    fn imparo_metal_set_concurrent(on: u32);
    fn imparo_metal_argmax(src: u32, dst: u32, n: u32);
    fn imparo_metal_set_skip_attn(on: u32);
    fn imparo_metal_set_skip_cat(cat: u32);
    fn imparo_metal_set_epilogue(kind: u32);
    fn imparo_metal_set_epilogue_act(kind: u32);
    fn imparo_metal_mma_peak(tgs: u32, sgs: u32, iters: u32) -> f64;
    fn imparo_metal_mma_loaded(tgs: u32, sgs: u32, iters: u32) -> f64;
    fn imparo_metal_mma_device_a(tgs: u32, sgs: u32, iters: u32, stride: u32) -> f64;
    fn imparo_metal_scoremix_rate(tgs: u32, sgs: u32, iters: u32, stride: u32, kspan: u32) -> f64;
    fn imparo_metal_commit_overhead(n: u32) -> f64;
    fn imparo_metal_sync_overhead(n: u32) -> f64;
    fn imparo_metal_encode_cost(n: u32) -> f64;
    fn imparo_metal_mma_loaded_smem(tgs: u32, sgs: u32, iters: u32, smem: u32) -> f64;
    fn imparo_metal_set_qtile(on: u32);
    fn imparo_metal_set_rt(on: u32);
    fn imparo_metal_set_rt_all(on: u32);
    fn imparo_metal_set_shrink(on: u32);
    fn imparo_metal_set_attn_min_tgs(n: u32);
    fn imparo_metal_set_attn_stream_slices(n: u32);
    fn imparo_metal_attn_stream_hq_current() -> u32;
    fn imparo_metal_attn_stream_slices_current() -> u32;
    fn imparo_metal_set_attn_stream_hq(n: u32);
    fn imparo_metal_attn_min_tgs_current() -> u32;
    fn imparo_metal_set_rt_shape_pre(i: u32);
    fn imparo_metal_last_gpu_us() -> f64;
    fn imparo_metal_threadgroup_bytes() -> u64;
    fn imparo_metal_max_threads_tg() -> u32;
    fn imparo_metal_rt_shape_legal(i: u32) -> u32;
    fn imparo_metal_set_measured_max_acc(v: u32);
    fn imparo_metal_spill_rate(idx: u32, tgs: u32, tpg: u32, iters: u32) -> f64;
    fn imparo_metal_pt_512x() -> u32;
    fn imparo_metal_pt_256x() -> u32;
    fn imparo_metal_set_qcomb_blk(v: u32);
    fn imparo_metal_qcomb_blk_current() -> u32;
    fn imparo_metal_ple_gather_combine(
        proj: u32,
        tokens_buf: u32,
        w_offset: u64,
        width: u32,
        emb_scale: f32,
        comb_scale: f32,
        n_tok: u32,
    );
    fn imparo_metal_rt_shapes() -> u32;
    fn imparo_metal_buf_count() -> u32;
    fn imparo_metal_set_q8_decode_sgs(sgs: u32);
    fn imparo_metal_q8_decode_sgs() -> u32;
    fn imparo_metal_set_q8_decode_rows(rows: u32);
    fn imparo_metal_q8_decode_rows() -> u32;
    fn imparo_metal_set_q8_batch_sgs(sgs: u32);
    fn imparo_metal_q8_batch_sgs() -> u32;
    fn imparo_metal_set_q8_token_tile(tile: u32);
    fn imparo_metal_q8_token_tile() -> u32;
    fn imparo_metal_set_q8_gemm_shape(shape: u32);
    fn imparo_metal_q8_gemm_shape() -> u32;
    fn imparo_metal_set_q8_gemm_large_shape(shape: u32);
    fn imparo_metal_q8_gemm_large_shape() -> u32;
    fn imparo_metal_set_q8_gemm_large_min_tok(n: u32);
    fn imparo_metal_q8_gemm_large_min_tok() -> u32;
    fn imparo_metal_set_q8_full_tiles(on: u32);
    fn imparo_metal_q8_full_tiles() -> u32;
    fn imparo_metal_set_q8_gemv_max_tok(n: u32);
    fn imparo_metal_q8_gemv_max_tok() -> u32;
    fn imparo_metal_set_q8_all(on: u32);
    fn imparo_metal_q8_gemm_shapes() -> u32;
    fn imparo_metal_set_attn_live_mask(on: u32);
    fn imparo_metal_attn_live_mask() -> u32;
    fn imparo_metal_rt_shape_current() -> u32;
    fn imparo_metal_lanes_current() -> u32;
    fn imparo_metal_nr0_current() -> u32;
    fn imparo_metal_set_nr0_all(on: u32);
    fn imparo_metal_set_qcomb(on: u32);
    fn imparo_metal_set_kvq_mask(mk: u32, mv: u32, ty: u32);
    fn imparo_metal_set_kvq_rt(ty: u32, lo: u32, hi: u32);
    fn imparo_metal_set_nb8_max(n: u32);
    fn imparo_metal_set_gemv_max_tok(n: u32);
    fn imparo_metal_gemv_max_tok_current() -> u32;
    fn imparo_metal_set_nb8_shape(v: u32);
    fn imparo_metal_nb8_max_current() -> u32;
    fn imparo_metal_nb8_shape_current() -> u32;
    fn imparo_metal_hadamard(buf: u32, n: u32, nrot: u32);
    fn imparo_metal_set_attn_short(on: u32);
    fn imparo_metal_set_attn_stream(on: u32);
    fn imparo_metal_set_attn_stream_min_pos(v: u32);
    fn imparo_metal_attn_stream_min_pos_current() -> u32;
    fn imparo_metal_attn_stream_current() -> u32;
    fn imparo_metal_set_attn_blk(v: u32);
    fn imparo_metal_attn_blk_current() -> u32;
    fn imparo_metal_set_attn_stage(v: u32);
    fn imparo_metal_set_attn_gqa(on: u32);
    fn imparo_metal_sgs_current() -> u32;
    fn imparo_metal_attn_threads_current() -> u32;
    fn imparo_metal_attn_threads_pf_current() -> u32;
    fn imparo_metal_set_rt_shape(i: u32) -> i32;
    fn imparo_metal_prof_enable(on: u32);
    fn imparo_metal_prof_cats(ticks: *mut f64, calls: *mut u64, n: *mut u32);
    fn imparo_metal_prof_cat_name(i: u32) -> *const std::os::raw::c_char;
    fn imparo_metal_prof_read(
        gpu_s: *mut f64,
        wall_s: *mut f64,
        cbs: *mut u64,
        disp: *mut u64,
    );
    fn imparo_metal_alloc(id: u32, bytes: u64) -> i32;
    fn imparo_metal_alloc_kv(n_layers: u32, bytes: *const u64) -> i32;
    fn imparo_metal_arena(bytes: u64) -> i32;
    fn imparo_metal_place(id: u32, offset: u64, bytes: u64) -> i32;
    fn imparo_metal_page_round(n: u64) -> u64;
    fn imparo_metal_grow_kv(n_layers: u32, bytes: *const u64) -> i32;
    fn imparo_metal_allocated_bytes() -> u64;
    fn imparo_metal_bw_read(bytes: u64, reps: u32, tgs: u32, tpg: u32) -> f64;
    fn imparo_metal_gemv_probe(
        n_in: u32,
        n_out: u32,
        lanes: u32,
        sgs: u32,
        mode: u32,
        iters: u32,
        split: u32,
    ) -> f64;
    fn imparo_metal_begin();
    fn imparo_metal_flush();
    fn imparo_metal_end() -> i32;
    fn imparo_metal_kv_advise_free(layer: u32, is_v: u32, off: u64, len: u64);
    fn imparo_metal_kv_advise_reuse(layer: u32, is_v: u32, off: u64, len: u64);
    fn imparo_metal_write(id: u32, off: u64, src: *const f32, n: u64);
    fn imparo_metal_read(id: u32, off: u64, dst: *mut f32, n: u64);
    fn imparo_metal_matmat(
        is_q4: u32,
        w_off: u64,
        n_in: u32,
        n_out: u32,
        src: u32,
        dst: u32,
        n_tok: u32,
        src_row: u32,
    );
    fn imparo_metal_row(
        wkind: u32,
        w_off: u64,
        width: u32,
        index: u32,
        scale: f32,
        dst: u32,
        dst_off: u32,
    );
    fn imparo_metal_gather_rows(
        wkind: u32,
        w_off: u64,
        width: u32,
        table_rows: u32,
        scale: f32,
        dst: u32,
        dst_off: u32,
        idx_buf: u32,
        n_rows: u32,
    ) -> i32;
    fn imparo_metal_rms_norm_add(
        buf: u32,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
        add_buf: u32,
    );
    fn imparo_metal_rms_norm(
        buf: u32,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    );
    fn imparo_metal_rms_norm_from(
        buf: u32,
        src_buf: u32,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    );
    fn imparo_metal_rope(
        buf: u32,
        n_rot: u32,
        base: f32,
        head_dim: u32,
        n_heads: u32,
        start_pos: u32,
        n_tok: u32,
        freqs: *const f32,
        n_freqs: u32,
    );
    fn imparo_metal_kv_store(
        src: u32,
        layer: u32,
        width: u32,
        start_pos: u32,
        n_tok: u32,
        is_v: u32,
        ring: u32,
    );
    fn imparo_metal_attention(
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
    fn imparo_metal_act_mul(a: u32, b: u32, n: u32);
    fn imparo_metal_act(a: u32, n: u32);
    fn imparo_metal_shortconv(
        bcx: u32,
        w_off: u64,
        state: u32,
        state_off: u32,
        out: u32,
        width: u32,
        kern: u32,
        n_tok: u32,
    );
    fn imparo_metal_shortconv_snapshot(
        bcx: u32,
        state: u32,
        state_off: u32,
        snap: u32,
        snap_off: u32,
        width: u32,
        kern: u32,
        n_tok: u32,
    );
    fn imparo_metal_add(a: u32, b: u32, n: u32);
    fn imparo_metal_add_scale(a: u32, b: u32, k: f32, n: u32);
    fn imparo_metal_mul_strided(
        a: u32,
        b: u32,
        n: u32,
        b_off: u32,
        b_stride: u32,
        a_stride: u32,
        n_tok: u32,
    );
    fn imparo_metal_scale(a: u32, k: f32, n: u32);
    fn imparo_metal_copy(dst: u32, src: u32, n: u32);
    fn imparo_metal_softcap(a: u32, cap: f32, n: u32);
    fn imparo_metal_ple_combine(
        proj: u32,
        emb: u32,
        width: u32,
        emb_scale: f32,
        comb_scale: f32,
        n_tok: u32,
    );
}

/// Sentinel meaning "no learned weight" for [`rms_norm`].
///
/// RE-EXPORTED, not redeclared. This crate used to carry its own copy, and a second copy
/// of a wire constant is a second chance to disagree with it -- which is exactly what
/// happened: the shared one said 0xFFFF_FFFF, CUDA tested u64::MAX. IMPARO_NO_WEIGHT in
/// imparo.metal is the kernel-side spelling of this same value.
pub use imparo_backend::NO_WEIGHT;

/// # Errors
///
/// Returns a non-zero code when no Metal device is available or the library fails to build.
///
/// # Safety
///
/// `base` must point to a live read-only mapping of at least `len` bytes.
/// Sets the matmul dispatch geometry: simdgroups per threadgroup, output rows per
/// simdgroup. Swept empirically; the best values are hardware dependent.
pub fn tune(sgs: u32, rows: u32) {
    unsafe { imparo_metal_tune(sgs, rows) }
}

/// Enables the staged prefill kernel (one lane per token). Off by default: it lost to the
/// simpler token-tile kernel when measured.
/// Attention threadgroup width (multiple of 32).
pub fn set_attn_threads(n: u32) {
    unsafe { imparo_metal_set_attn_threads(n) }
}

/// Lanes cooperating on one output row in the decode matmul: 4, 8, 16 or 32.
pub fn set_lanes(lanes: u32) {
    unsafe { imparo_metal_set_lanes(lanes) }
}
/// Output rows per thread in the decode GEMV. Rounded down to a power of two, max 8.
pub fn set_nr0(nr0: u32) {
    unsafe { imparo_metal_set_nr0(nr0) }
}

/// Diagnostic: run the prefill kernel's staging without the matrix multiplies, to
/// attribute its cost between the two.
pub fn set_skip_mma(on: bool) {
    unsafe { imparo_metal_set_skip_mma(u32::from(on)) }
}

/// Select the register-tiled prefill GEMM.
pub fn set_rt(on: bool) {
    unsafe { imparo_metal_set_rt(u32::from(on)) }
}

/// Attention threadgroup count below which the KV range is split across threadgroups.
pub fn set_attn_min_tgs(n: u32) {
    unsafe { imparo_metal_set_attn_min_tgs(n) }
}
/// Register-blocking depth in the attention matrix phases.
/// Query heads per threadgroup in the stream kernel (1 or 2). GQA row sharing:
/// at 2, one K/V row read serves two heads.
/// Force the query-tiled prefill kernel instead of the combined one (probes:
/// `init` does not read IMPARO_QCOMB).
pub fn set_qcomb(on: bool) {
    unsafe { imparo_metal_set_qcomb(u32::from(on)) }
}

/// How far through the attention kernel to run: 1 scores, 2 adds softmax, 3 full.
/// Wrong results by construction -- it exists to attribute the phases.
pub fn set_attn_stage(v: u32) {
    unsafe { imparo_metal_set_attn_stage(v) }
}

pub fn set_attn_stream_hq(v: u32) {
    unsafe { imparo_metal_set_attn_stream_hq(v) }
}

/// Slice-count cap for the stream kernel.
#[must_use]
pub fn attn_stream_hq_current() -> u32 {
    unsafe { imparo_metal_attn_stream_hq_current() }
}

#[must_use]
pub fn attn_stream_slices_current() -> u32 {
    unsafe { imparo_metal_attn_stream_slices_current() }
}

pub fn set_attn_stream_slices(v: u32) {
    unsafe { imparo_metal_set_attn_stream_slices(v) }
}

pub fn set_attn_stream_min_pos(v: u32) {
    unsafe { imparo_metal_set_attn_stream_min_pos(v) }
}
pub fn attn_stream_min_pos_current() -> u32 {
    unsafe { imparo_metal_attn_stream_min_pos_current() }
}
pub fn set_attn_stream(v: u32) {
    unsafe { imparo_metal_set_attn_stream(v) }
}
pub fn attn_stream_current() -> u32 {
    unsafe { imparo_metal_attn_stream_current() }
}
pub fn set_attn_blk(v: u32) {
    unsafe { imparo_metal_set_attn_blk(v) }
}
#[must_use]
pub fn attn_blk_current() -> u32 {
    unsafe { imparo_metal_attn_blk_current() }
}
/// Threadgroups it takes to fill this GPU, which sets the decode slice count.
#[must_use]
pub fn attn_min_tgs_current() -> u32 {
    unsafe { imparo_metal_attn_min_tgs_current() }
}

/// Fold the gated activation into the next prefill matmul's write-back: it computes
/// `dst = act(dst) * result` instead of `dst = result`, replacing a separate pass.
/// Applies to ONE matmul; clear it afterwards.
pub fn set_epilogue(kind: u32) {
    unsafe { imparo_metal_set_epilogue(kind) }
}

/// Which activation the fused epilogue applies, for this process.
///
/// Read at pipeline BUILD, so it must be set before `init`. Setting it afterwards is a
/// no-op and would silently leave GELU.
pub fn set_epilogue_act(kind: u32) {
    unsafe { imparo_metal_set_epilogue_act(kind) }
}

/// Attention threadgroup width used during prefill (decode has its own).
pub fn set_attn_threads_prefill(n: u32) {
    unsafe { imparo_metal_set_attn_threads_prefill(n) }
}

/// Register-tile shape, chosen BEFORE init so only that pipeline is built.
pub fn set_rt_shape_pre(i: u32) {
    unsafe { imparo_metal_set_rt_shape_pre(i) }
}

/// TFLOPS from back-to-back simdgroup multiply-accumulates with operands in registers.
pub fn mma_peak(tgs: u32, sgs: u32, iters: u32) -> f64 {
    unsafe { imparo_metal_mma_peak(tgs, sgs, iters) }
}

/// Same, but reloading operands from threadgroup memory at the GEMM's 6:8 load ratio.
/// Same loop with the A operands read from device memory at `stride`, as the GEMM does.
pub fn mma_device_a(tgs: u32, sgs: u32, iters: u32, stride: u32) -> f64 {
    unsafe { imparo_metal_mma_device_a(tgs, sgs, iters, stride) }
}

/// TFLOPS with the prefill attention score loop's OWN operand mix: staged Q re-read from
/// threadgroup memory, K streamed from device, four multiply-accumulates per four loads.
///
/// This is the number the score phase should be judged against. A generic matrix peak is
/// measured on a different access pattern, so the fraction-of-peak it yields describes a
/// kernel that does not exist.
pub fn scoremix_rate(tgs: u32, sgs: u32, iters: u32, stride: u32, kspan: u32) -> f64 {
    unsafe { imparo_metal_scoremix_rate(tgs, sgs, iters, stride, kspan) }
}

/// Microseconds to SUBMIT an empty command buffer -- created and committed, not waited on.
/// This is what a mid-graph flush costs, because `flush` commits and returns.
#[must_use]
pub fn commit_overhead(n: u32) -> f64 {
    unsafe { imparo_metal_commit_overhead(n) }
}

/// Microseconds per commit AND WAIT -- what the end of a graph pays, where the host must
/// see the result. A different quantity from `commit_overhead`, and much larger.
#[must_use]
pub fn sync_overhead(n: u32) -> f64 {
    unsafe { imparo_metal_sync_overhead(n) }
}

/// Microseconds the CPU spends encoding ONE dispatch, with no execution and no commit in
/// it. The other half of the `flush_layers` trade.
#[must_use]
pub fn encode_cost(n: u32) -> f64 {
    unsafe { imparo_metal_encode_cost(n) }
}

pub fn mma_loaded(tgs: u32, sgs: u32, iters: u32) -> f64 {
    unsafe { imparo_metal_mma_loaded(tgs, sgs, iters) }
}

/// Same, with the threadgroup allocation padded, to isolate what occupancy costs.
pub fn mma_loaded_smem(tgs: u32, sgs: u32, iters: u32, smem: u32) -> f64 {
    unsafe { imparo_metal_mma_loaded_smem(tgs, sgs, iters, smem) }
}

/// How many register-tile shapes this build offers.
pub fn rt_shape_count() -> u32 {
    unsafe { imparo_metal_rt_shapes() }
}

// ---- Q8_0 knobs --------------------------------------------------------------------
// Setters clamp in the bridge, so an illegal value leaves the current one standing
// rather than selecting a pipeline that was never built.
macro_rules! q8_knob {
    ($set:ident, $get:ident, $c_set:ident, $c_get:ident) => {
        pub fn $set(v: u32) {
            unsafe { $c_set(v) }
        }
        #[must_use]
        pub fn $get() -> u32 {
            unsafe { $c_get() }
        }
    };
}
q8_knob!(
    set_q8_decode_sgs,
    q8_decode_sgs,
    imparo_metal_set_q8_decode_sgs,
    imparo_metal_q8_decode_sgs
);
q8_knob!(
    set_q8_decode_rows,
    q8_decode_rows,
    imparo_metal_set_q8_decode_rows,
    imparo_metal_q8_decode_rows
);
q8_knob!(
    set_q8_batch_sgs,
    q8_batch_sgs,
    imparo_metal_set_q8_batch_sgs,
    imparo_metal_q8_batch_sgs
);
q8_knob!(
    set_q8_token_tile,
    q8_token_tile,
    imparo_metal_set_q8_token_tile,
    imparo_metal_q8_token_tile
);
q8_knob!(
    set_q8_gemm_shape,
    q8_gemm_shape,
    imparo_metal_set_q8_gemm_shape,
    imparo_metal_q8_gemm_shape
);
q8_knob!(
    set_q8_gemm_large_shape,
    q8_gemm_large_shape,
    imparo_metal_set_q8_gemm_large_shape,
    imparo_metal_q8_gemm_large_shape
);
q8_knob!(
    set_q8_gemm_large_min_tok,
    q8_gemm_large_min_tok,
    imparo_metal_set_q8_gemm_large_min_tok,
    imparo_metal_q8_gemm_large_min_tok
);
q8_knob!(
    set_q8_full_tiles,
    q8_full_tiles,
    imparo_metal_set_q8_full_tiles,
    imparo_metal_q8_full_tiles
);
q8_knob!(
    set_q8_gemv_max_tok,
    q8_gemv_max_tok,
    imparo_metal_set_q8_gemv_max_tok,
    imparo_metal_q8_gemv_max_tok
);
/// Skip the prefill mask loop on provably-live position blocks. Must be set BEFORE
/// init: it selects which prefill attention pipelines are compiled.
pub fn set_attn_live_mask(on: u32) {
    unsafe { imparo_metal_set_attn_live_mask(on) }
}
#[must_use]
pub fn attn_live_mask() -> u32 {
    unsafe { imparo_metal_attn_live_mask() }
}
/// Build every Q8 GEMM candidate instead of only the selected shapes. Sweeping only:
/// a compute pipeline is GPU-resident code and these kernels are fully unrolled.
pub fn set_q8_all(on: u32) {
    unsafe { imparo_metal_set_q8_all(on) }
}
#[must_use]
pub fn q8_gemm_shapes() -> u32 {
    unsafe { imparo_metal_q8_gemm_shapes() }
}

/// The shape currently selected, so a sweep that finds no clear winner can keep it.
#[must_use]
pub fn rt_shape_current() -> u32 {
    unsafe { imparo_metal_rt_shape_current() }
}

/// The values currently in effect. A sweep seeds its incumbent with these, so a knob whose
/// candidates all sit inside the noise floor keeps what was configured instead of handing
/// the seat to whichever value happened to be measured first.
#[must_use]
pub fn lanes_current() -> u32 {
    unsafe { imparo_metal_lanes_current() }
}
#[must_use]
pub fn nr0_current() -> u32 {
    unsafe { imparo_metal_nr0_current() }
}
/// Narrow-N GEMM boundary: batches of 2..n tokens take the 64x8 tile (0 disables).
pub fn set_nb8_max(n: u32) {
    unsafe { imparo_metal_set_nb8_max(n) }
}
pub fn set_gemv_max_tok(n: u32) {
    unsafe { imparo_metal_set_gemv_max_tok(n) }
}
/// Single-knob flush setters, so the registry's one-value apply hook fits the
/// two-value engine call (the other side keeps its current value).
pub fn set_flush_layers_decode(v: u32) {
    set_flush_layers(v, flush_layers(false));
}
pub fn set_flush_layers_prefill_only(v: u32) {
    set_flush_layers(flush_layers(true), v);
}
#[must_use]
pub fn gemv_max_tok_current() -> u32 {
    unsafe { imparo_metal_gemv_max_tok_current() }
}
/// Narrow-N tile variant: 0 = two simdgroups / 64 threads, 1 = one / 32.
pub fn set_nb8_shape(v: u32) {
    unsafe { imparo_metal_set_nb8_shape(v) }
}
#[must_use]
pub fn nb8_max_current() -> u32 {
    unsafe { imparo_metal_nb8_max_current() }
}
#[must_use]
pub fn nb8_shape_current() -> u32 {
    unsafe { imparo_metal_nb8_shape_current() }
}
#[must_use]
pub fn sgs_current() -> u32 {
    unsafe { imparo_metal_sgs_current() }
}
#[must_use]
pub fn attn_threads_current() -> u32 {
    unsafe { imparo_metal_attn_threads_current() }
}
#[must_use]
pub fn attn_threads_pf_current() -> u32 {
    unsafe { imparo_metal_attn_threads_pf_current() }
}

/// Select a register-tile shape. Fails if the index has no pipeline on this device --
/// a shape can exceed the device's threadgroup memory or thread limit, and that is a
/// property of the host, which is the whole reason the choice is measured here.
pub fn set_rt_shape(i: u32) -> Result<(), String> {
    if unsafe { imparo_metal_set_rt_shape(i) } == 0 {
        Ok(())
    } else {
        Err(format!(
            "register-tile shape {i} unavailable on this device"
        ))
    }
}

/// Submission profile since the last read: GPU-busy seconds, wall seconds spent inside
/// commit/wait, command buffers, and dispatches. Reading resets the counters.
pub struct Prof {
    pub gpu_s: f64,
    pub wall_s: f64,
    pub cbs: u64,
    pub dispatches: u64,
}

pub fn prof_enable(on: bool) {
    unsafe { imparo_metal_prof_enable(u32::from(on)) }
}

/// Per-kernel-category GPU time since the last read, as (name, ticks, calls).
///
/// Ticks are the hardware timestamp counter's own unit. They are reported as shares of the
/// total rather than converted to seconds: the share is what decides where to work, and a
/// tick-to-nanosecond factor that varies by device would be one more thing to get wrong.
pub fn prof_categories() -> Vec<(String, f64, u64)> {
    let (mut ticks, mut calls, mut n) = ([0.0f64; 32], [0u64; 32], 0u32);
    unsafe {
        imparo_metal_prof_cats(ticks.as_mut_ptr(), calls.as_mut_ptr(), &raw mut n);
    };
    (0..n.min(32) as usize)
        .map(|i| {
            let name = unsafe {
                std::ffi::CStr::from_ptr(imparo_metal_prof_cat_name(i as u32))
            };
            (name.to_string_lossy().into_owned(), ticks[i], calls[i])
        })
        .filter(|(_, t, calls)| *t > 0.0 || *calls > 0)
        .collect()
}

pub fn prof_read() -> Prof {
    let (mut gpu_s, mut wall_s, mut cbs, mut dispatches) = (0.0, 0.0, 0u64, 0u64);
    unsafe {
        imparo_metal_prof_read(
            &raw mut gpu_s,
            &raw mut wall_s,
            &raw mut cbs,
            &raw mut dispatches,
        );
    };
    Prof {
        gpu_s,
        wall_s,
        cbs,
        dispatches,
    }
}

/// # Errors
///
/// Returns a non-zero code when no Metal device is available.
///
/// # Safety
///
/// `base` must point to a live read-only mapping of at least `len` bytes.
pub unsafe fn init_tuned(base: *const u8, len: u64) -> Result<(), i32> {
    // The buffer table must be able to hold every BufId. Checked rather than assumed:
    // the two numbers live in different languages and an undersized table indexes out of
    // bounds instead of failing.
    if let Err(e) =
        imparo_backend::check_buf_table("metal", unsafe { imparo_metal_buf_count() } as usize)
    {
        eprintln!("[imparo] {e}");
        return Err(1);
    }
    // measured host configuration first, explicit env overrides after
    // A tuner must not seed itself from a previous tuning: the result would depend on what
    // happened to be stored, so running it twice could give two answers. It sets
    // IMPARO_NO_HOSTCONFIG and starts from the compiled defaults every time.
    let skip_cfg = std::env::var("IMPARO_NO_HOSTCONFIG").is_ok_and(|v| v == "1");
    if let Some(batch) = if skip_cfg {
        None
    } else {
        apply_host_config(len)
    } {
        if std::env::var("IMPARO_BATCH").is_err() {
            unsafe { std::env::set_var("IMPARO_BATCH", batch.to_string()) };
        }
    }
    if let Ok(v) = std::env::var("IMPARO_SGS") {
        if let Ok(n) = v.parse::<u32>() {
            tune(n, 0);
        }
    }
    if let Ok(v) = std::env::var("IMPARO_ROWS") {
        if let Ok(n) = v.parse::<u32>() {
            tune(0, n);
        }
    }
    // #5 REPRODUCER ONLY: raising min_tok routes multi-token batches to the GEMV,
    // the one geometry that carries the visibility wobble (any multi-token size can
    // fire; see metal_visibility_repro/README.md). Shipping keeps 2, and with the
    // nb8 tile covering 2..15 no multi-token batch reaches the GEMV by default.

    // IMPARO_ATTN_HQ=2: GQA row sharing in the stream kernel (two query heads
    // per threadgroup), halving the unique KV bytes the cache must serve.
    if let Ok(v) = std::env::var("IMPARO_ATTN_HQ") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_attn_stream_hq(n) }
        }
    }
    if let Ok(v) = std::env::var("IMPARO_ATTN_STREAM_SLICES") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_attn_stream_slices(n) }
        }
    }
    if let Ok(v) = std::env::var("IMPARO_ATTN_MIN_TGS") {
        if let Ok(n) = v.parse::<u32>() {
            set_attn_min_tgs(n);
        }
    }
    if let Ok(v) = std::env::var("IMPARO_ATTN_THREADS_PF") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_attn_threads_prefill(n) }
        }
    }
    if let Ok(v) = std::env::var("IMPARO_ATTN_THREADS") {
        if let Ok(n) = v.parse::<u32>() {
            set_attn_threads(n);
        }
    }
    // bit0 = skip multiplies, bit1 = skip dequantisation
    if let Ok(v) = std::env::var("IMPARO_SKIP_MMA") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_skip_mma(n) }
        }
    }
    // IMPARO_HALF_A: stage prefill-GEMM activations as half (llama.cpp's own mul_mm
    // staging). Gated by the fork-agreement harness, not the old exact-to-self gate.
    if let Ok(v) = std::env::var("IMPARO_HALF_A") {
        unsafe { imparo_metal_set_half_a(u32::from(v != "0")) };
    }
    // KV cache types from the environment (probe binaries); the server configures them
    // through `set_kv_types` before engine construction instead. Only applied when a
    // variable is present, so a prior explicit configuration is never clobbered.
    if std::env::var("IMPARO_CTK").is_ok() || std::env::var("IMPARO_CTV").is_ok() {
        let t = |var: &str| match std::env::var(var).as_deref() {
            Ok("q4_0") => 2,
            Ok("q8_0") => 8,
            _ => 1,
        };
        unsafe { imparo_metal_set_kv_types(t("IMPARO_CTK"), t("IMPARO_CTV")) };
    }
    // Concurrent dispatch with per-site read/write hazard tracking. DEFAULT ON
    // (serial encoding measured +1.4 ms/token of drain bubbles at 16k decode);
    // IMPARO_CONCURRENT=0 reverts to serial. Bit-identical output either way --
    // it changes overlap, not arithmetic.
    if let Ok(v) = std::env::var("IMPARO_CONCURRENT") {
        unsafe { imparo_metal_set_concurrent(u32::from(v != "0")) };
    }
    if let Ok(v) = std::env::var("IMPARO_QTILE") {
        unsafe { imparo_metal_set_qtile(u32::from(v != "0")) };
    }
    if let Ok(v) = std::env::var("IMPARO_SKIP_CAT") {
        // names must match PROF_CAT_NAME order in the backend
        let cats = [
            "matmat_prefill",
            "matmat_decode",
            "row",
            "rms_norm",
            "rope",
            "kv_store",
            "attention",
            "elementwise",
            "mul_strided",
            "ple_combine",
        ];
        if let Some(i) = cats.iter().position(|c| *c == v) {
            unsafe { imparo_metal_set_skip_cat(i as u32) };
        }
    }
    // 1 = skip all attention, 2 = skip only hd-512 layers, 3 = skip only hd-256.
    if let Ok(v) = std::env::var("IMPARO_SKIP_ATTN") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_skip_attn(n) }
        }
    }
    if let Ok(v) = std::env::var("IMPARO_RT") {
        set_rt(v != "0");
    }
    if let Ok(v) = std::env::var("IMPARO_SHRINK") {
        unsafe { imparo_metal_set_shrink(u32::from(v != "0")) };
    }
    // Shape and sweep mode must be chosen BEFORE init, which is where pipelines are built.
    if let Ok(v) = std::env::var("IMPARO_RT_SHAPE") {
        if let Ok(i) = v.parse::<u32>() {
            unsafe { imparo_metal_set_rt_shape_pre(i) }
        }
    }
    // Pre-init like IMPARO_RT_SHAPE: it decides which pipelines get compiled.
    if let Ok(v) = std::env::var("IMPARO_ATTN_LIVE_MASK") {
        set_attn_live_mask(u32::from(v != "0"));
    }
    if std::env::var("IMPARO_RT_ALL").is_ok_and(|v| v == "1") {
        unsafe { imparo_metal_set_rt_all(1) };
    }
    // Builds the nr0 variants WITHOUT selecting one. IMPARO_NR0 selects; these are separate
    // questions and sharing one variable made the tuner measure at nr0=8 throughout.
    if std::env::var("IMPARO_NR0_ALL").is_ok_and(|v| v == "1") {
        unsafe { imparo_metal_set_nr0_all(1) };
    }
    // IMPARO_QCOMB=1: the combined prefill-attention rewrite (staged float Q, register
    // accumulator). Experiment, default off; bit-identical output is a design input.
    if let Ok(v) = std::env::var("IMPARO_QCOMB") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_qcomb(n) }
        }
    }
    // Narrow-N GEMM boundary: batches of 2..N tokens take the 64x8 tile.
    // IMPARO_NB8=0 disables (A/B back to the wide tile); any other value sets the
    // boundary directly. The tuned value comes from hostconfig (v10).
    if let Ok(v) = std::env::var("IMPARO_NB8") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_nb8_max(n) }
        }
    }
    if let Ok(v) = std::env::var("IMPARO_NB8_SHAPE") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_nb8_shape(n) }
        }
    }
    // GEMV boundary: batches of 2..N tokens take the GEMV. A/B override; the tuned
    // value comes from hostconfig (v11).
    if let Ok(v) = std::env::var("IMPARO_GEMV_MAX") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_gemv_max_tok(n) }
        }
    }
    // Diagnostic: quantize only the masked layers' K/V (hex bitmasks over layer ids);
    // the caches must be f16-sized (leave IMPARO_CTK/CTV unset) and prefill-only.
    {
        let mask = |var: &str| {
            std::env::var(var)
                .ok()
                .and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok())
                .unwrap_or(0)
        };
        let (mk, mv) = (mask("IMPARO_KVQ_MASK_K"), mask("IMPARO_KVQ_MASK_V"));
        if mk != 0 || mv != 0 {
            let ty = match std::env::var("IMPARO_KVQ_TYPE").as_deref() {
                Ok("q8_0") => 8,
                _ => 2,
            };
            unsafe { imparo_metal_set_kvq_mask(mk, mv, ty) };
        }
        // Roundtrip diagnostic: quantize+dequantize in the store, f16 cache layout.
        if let Ok(v) = std::env::var("IMPARO_KVQ_RT") {
            let ty = if v == "q8_0" { 8 } else { 2 };
            let n = |var: &str, dflt: u32| {
                std::env::var(var)
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(dflt)
            };
            unsafe {
                imparo_metal_set_kvq_rt(
                    ty,
                    n("IMPARO_KVQ_RT_LO", 0),
                    n("IMPARO_KVQ_RT_HI", u32::MAX),
                );
            };
        }
    }
    if let Ok(v) = std::env::var("IMPARO_ATTN_SHORT") {
        unsafe { imparo_metal_set_attn_short(u32::from(v != "0")) };
    }
    if let Ok(v) = std::env::var("IMPARO_ATTN_BLK") {
        if let Ok(n) = v.parse::<u32>() {
            set_attn_blk(n);
        }
    }
    if let Ok(v) = std::env::var("IMPARO_ATTN_STAGE") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_attn_stage(n) }
        }
    }
    // The VALUE is the group size, not a flag: 0 off, 1 "use the whole GQA share",
    // 2/4/8 an explicit group. Grouping divides KV traffic by the group and multiplies
    // accumulators per thread by it, and the kernel is bound by the second, so the
    // largest group that fits is not automatically the fastest. This parsed `v != "0"`,
    // which collapsed every group size to 1 and made the smaller ones unreachable.
    if let Ok(v) = std::env::var("IMPARO_ATTN_GQA") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_attn_gqa(n) };
        }
    }
    if let Ok(v) = std::env::var("IMPARO_LANES") {
        if let Ok(n) = v.parse::<u32>() {
            set_lanes(n);
        }
    }
    if let Ok(v) = std::env::var("IMPARO_ATTN_STREAM_MIN_POS") {
        if let Ok(n) = v.parse::<u32>() {
            set_attn_stream_min_pos(n);
        }
    }
    if let Ok(v) = std::env::var("IMPARO_ATTN_STREAM") {
        if let Ok(n) = v.parse::<u32>() {
            set_attn_stream(n);
        }
    }
    if let Ok(v) = std::env::var("IMPARO_FLUSH_LAYERS") {
        if let Ok(n) = v.parse::<u32>() {
            set_flush_layers(n, flush_layers(false));
        }
    }
    if let Ok(v) = std::env::var("IMPARO_FLUSH_LAYERS_PREFILL") {
        if let Ok(n) = v.parse::<u32>() {
            set_flush_layers(flush_layers(true), n);
        }
    }
    if let Ok(v) = std::env::var("IMPARO_NR0") {
        if let Ok(n) = v.parse::<u32>() {
            set_nr0(n);
        }
    }
    unsafe { init(base, len) }?;

    // These two need a live device: one selects among pipelines, the other allocates a
    // counter buffer. Applied before init they could only fail, and silently -- the shape
    // request would report "unavailable" for every index because none existed yet.
    if std::env::var("IMPARO_PROF").is_ok_and(|v| v == "1") {
        prof_enable(true);
    }
    Ok(())
}

/// # Safety
/// `base` must point to `len` readable bytes that outlive every kernel: Metal wraps
/// the mapping without copying.
pub unsafe fn init(base: *const u8, len: u64) -> Result<(), i32> {
    let rc = unsafe { imparo_metal_init(base.cast(), len) };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// # Errors
///
/// Returns a non-zero code when allocation fails.
pub fn alloc(id: u32, bytes: u64) -> Result<(), i32> {
    let rc = unsafe { imparo_metal_alloc(id, bytes) };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// # Errors
///
/// Returns a non-zero code when allocation fails.
/// Bytes Metal reports as allocated by this process's device. Compared against
/// phys_footprint it separates memory that is in our buffers from memory the
/// driver holds privately.
#[must_use]
pub fn allocated_bytes() -> u64 {
    unsafe { imparo_metal_allocated_bytes() }
}

/// GPU microseconds of the last begin/end region, from the command buffer's own
/// timestamps -- no submission latency included.
/// BLK for the QT-8 qcomb prefill attention kernels (q4/q8 prefill). 4 leaves half the
/// simdgroups idle at PT 128 / NSG 8; 2 fills them at a worse load-to-MAC ratio. Which
/// trade wins is for the device to say.
pub fn set_qcomb_blk(v: u32) {
    unsafe { imparo_metal_set_qcomb_blk(v) };
}

#[must_use]
pub fn qcomb_blk_current() -> u32 {
    unsafe { imparo_metal_qcomb_blk_current() }
}

/// Hand the backend the measured accumulator cliff, before the library compiles.
pub fn set_measured_max_acc(v: u32) {
    unsafe { imparo_metal_set_measured_max_acc(v) }
}

/// Live-accumulator counts the spill probe sweeps, in order.
pub const SPILL_NACC: [u32; 8] = [4, 8, 12, 16, 24, 32, 48, 64];

/// TFLOPS holding `SPILL_NACC[idx]` accumulators live. The rate holds while they fit in
/// registers and collapses once the compiler spills them, and that cliff IS the register
/// budget -- a limit no Metal API reports.
#[must_use]
pub fn spill_rate(idx: u32, tgs: u32, tpg: u32, iters: u32) -> f64 {
    unsafe { imparo_metal_spill_rate(idx, tgs, tpg, iters) }
}

/// Whether a register-tile shape can run at all: its threads x accumulators against the
/// register budget. The BACKEND owns this rule -- the registry asks rather than restates.
#[must_use]
pub fn rt_shape_legal(i: u32) -> bool {
    unsafe { imparo_metal_rt_shape_legal(i) != 0 }
}

/// Threadgroup memory a single threadgroup may declare, from the device.
#[must_use]
pub fn threadgroup_bytes() -> u64 {
    unsafe { imparo_metal_threadgroup_bytes() }
}

#[must_use]
pub fn max_threads_per_threadgroup() -> u32 {
    unsafe { imparo_metal_max_threads_tg() }
}

/// Position tile width derived at init from this device's threadgroup limit.
#[must_use]
pub fn pt_512x() -> u32 {
    unsafe { imparo_metal_pt_512x() }
}

#[must_use]
pub fn pt_256x() -> u32 {
    unsafe { imparo_metal_pt_256x() }
}

#[must_use]
pub fn last_gpu_us() -> f64 {
    unsafe { imparo_metal_last_gpu_us() }
}

/// Streaming-read bandwidth in GB/s: uint4 loads only, no arithmetic. The reference
/// a memory-bound kernel should be judged against.
#[must_use]
pub fn bw_read(bytes: u64, reps: u32, tgs: u32, tpg: u32) -> f64 {
    unsafe { imparo_metal_bw_read(bytes, reps, tgs, tpg) }
}

/// GEMV read bandwidth with the arithmetic removed. `mode` 0 is the real kernel's
/// 18-byte-stride addressing, 1 is contiguous uint4 over the same payload bytes.
#[must_use]
pub fn gemv_probe(
    n_in: u32,
    n_out: u32,
    lanes: u32,
    sgs: u32,
    mode: u32,
    iters: u32,
    split: u32,
) -> f64 {
    unsafe { imparo_metal_gemv_probe(n_in, n_out, lanes, sgs, mode, iters, split) }
}

/// Grow KV buffers to at least `bytes` per layer, preserving their contents.
///
/// # Errors
/// Returns a non-zero code when a Metal allocation fails.
pub fn grow_kv(bytes: &[u64]) -> Result<(), i32> {
    let rc = unsafe {
        imparo_metal_grow_kv(u32::try_from(bytes.len()).unwrap_or(0), bytes.as_ptr())
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Size the shared activation arena. Buffers already placed in it are dropped.
///
/// # Errors
/// Returns a non-zero code when the host allocation fails.
pub fn arena(bytes: u64) -> Result<(), i32> {
    let rc = unsafe { imparo_metal_arena(bytes) };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Place a buffer at `offset` in the arena. Two groups whose lifetimes do not overlap are
/// both laid out from 0, so they share the same bytes.
///
/// # Errors
/// Returns a non-zero code when the range does not fit or the buffer cannot be made.
pub fn place(id: u32, offset: u64, bytes: u64) -> Result<(), i32> {
    let rc = unsafe { imparo_metal_place(id, offset, bytes) };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Round up to the host page size, which is what the arena lays out in.
#[must_use]
pub fn page_round(n: u64) -> u64 {
    unsafe { imparo_metal_page_round(n) }
}

pub fn alloc_kv(bytes: &[u64]) -> Result<(), i32> {
    let rc = unsafe {
        imparo_metal_alloc_kv(u32::try_from(bytes.len()).unwrap_or(0), bytes.as_ptr())
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

pub fn begin() {
    unsafe { imparo_metal_begin() }
}
/// Commit the work encoded so far and start a new command buffer without waiting, so the
/// GPU runs it while the CPU encodes what follows.
pub fn flush() {
    unsafe { imparo_metal_flush() }
}

/// How many layers to encode before committing, at decode and at prefill. 0 disables the
/// split for that phase.
///
/// Lives here rather than being read from the environment per forward pass: the call site
/// runs once per layer per token, and it is a tuned value like `lanes` or `sgs` -- the
/// balance it strikes between encode/execute overlap and live command buffers is a
/// property of the machine, so the tuner has to be able to set it.
static FLUSH_DECODE: core::sync::atomic::AtomicU32 =
    core::sync::atomic::AtomicU32::new(7);
static FLUSH_PREFILL: core::sync::atomic::AtomicU32 =
    core::sync::atomic::AtomicU32::new(0);

pub fn set_flush_layers(decode: u32, prefill: u32) {
    FLUSH_DECODE.store(decode, core::sync::atomic::Ordering::Relaxed);
    FLUSH_PREFILL.store(prefill, core::sync::atomic::Ordering::Relaxed);
}

#[must_use]
pub fn flush_layers(decode: bool) -> u32 {
    let a = if decode {
        &FLUSH_DECODE
    } else {
        &FLUSH_PREFILL
    };
    a.load(core::sync::atomic::Ordering::Relaxed)
}

/// # Errors
///
/// Returns a non-zero code when the command buffer reports an error.
pub fn end() -> Result<(), i32> {
    let rc = unsafe { imparo_metal_end() };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

pub fn kv_advise_free(layer: u32, is_v: bool, off: u64, len: u64) {
    unsafe { imparo_metal_kv_advise_free(layer, u32::from(is_v), off, len) }
}

pub fn kv_advise_reuse(layer: u32, is_v: bool, off: u64, len: u64) {
    unsafe { imparo_metal_kv_advise_reuse(layer, u32::from(is_v), off, len) }
}

pub fn write(id: u32, off: u64, src: &[f32]) {
    unsafe { imparo_metal_write(id, off, src.as_ptr(), src.len() as u64) }
}
pub fn read(id: u32, off: u64, dst: &mut [f32]) {
    unsafe { imparo_metal_read(id, off, dst.as_mut_ptr(), dst.len() as u64) }
}
/// `wkind` is the weight-type -> kernel table index (imparo_gguf::weights::WeightKind
/// as u32: 0 = F32, 1 = Q4_0). Load-time validation guarantees no other value arrives.
pub fn matmat(
    wkind: u32,
    w_off: u64,
    n_in: u32,
    n_out: u32,
    src: u32,
    dst: u32,
    n_tok: u32,
) {
    unsafe { imparo_metal_matmat(wkind, w_off, n_in, n_out, src, dst, n_tok, 0) }
}

/// Same, but reading input rows starting at `src_row` instead of 0.
pub fn matmat_from(
    wkind: u32,
    w_off: u64,
    n_in: u32,
    n_out: u32,
    src: u32,
    dst: u32,
    n_tok: u32,
    src_row: u32,
) {
    unsafe { imparo_metal_matmat(wkind, w_off, n_in, n_out, src, dst, n_tok, src_row) }
}
pub fn row(
    wkind: u32,
    w_off: u64,
    width: u32,
    index: u32,
    scale: f32,
    dst: u32,
    dst_off: u32,
) {
    unsafe { imparo_metal_row(wkind, w_off, width, index, scale, dst, dst_off) }
}
/// One dispatch for a whole batch of embedding rows. `false` when the backend refused
/// (an unsupported kind or a width it cannot vectorise), so the caller keeps `row`.
#[must_use]
pub fn gather_rows(
    wkind: u32,
    w_off: u64,
    width: u32,
    table_rows: u32,
    scale: f32,
    dst: u32,
    dst_off: u32,
    idx_buf: u32,
    n_rows: u32,
) -> bool {
    unsafe {
        imparo_metal_gather_rows(
            wkind, w_off, width, table_rows, scale, dst, dst_off, idx_buf, n_rows,
        ) == 0
    }
}
pub fn rms_norm(
    buf: u32,
    w_off: u64,
    width: u32,
    eps: f32,
    n_row: u32,
    row_stride: u32,
    base_off: u32,
) {
    unsafe {
        imparo_metal_rms_norm(buf, w_off, width, eps, n_row, row_stride, base_off);
    }
}

/// Normalise `src` into `buf`. Same as [`rms_norm`] when they are the same buffer.
///
/// Two call sites per layer used to `copy` the row and then normalise it in place -- 84
/// dispatches per decoded token, each moving 10 KB, in a kernel that is almost all
/// dispatch latency.
pub fn rms_norm_from(
    buf: u32,
    src: u32,
    w_off: u64,
    width: u32,
    eps: f32,
    n_row: u32,
    row_stride: u32,
    base_off: u32,
) {
    unsafe {
        imparo_metal_rms_norm_from(
            buf, src, w_off, width, eps, n_row, row_stride, base_off,
        );
    }
}

/// rms_norm with a residual add folded into its scale pass.
///
/// Every rms_norm in the forward pass is followed by an elementwise add of another buffer.
/// Fusing them removes a dispatch, a read of the row and a write of it, three times per
/// layer.
pub fn rms_norm_add(
    buf: u32,
    w_off: u64,
    width: u32,
    eps: f32,
    n_row: u32,
    row_stride: u32,
    base_off: u32,
    add_buf: u32,
) {
    unsafe {
        imparo_metal_rms_norm_add(
            buf, w_off, width, eps, n_row, row_stride, base_off, add_buf,
        );
    }
}
/// NEOX rope. `freqs` is `rope_freqs.weight`, which divides the inverse frequency and which
/// this model carries on full-attention layers only; `None` applies none.
pub fn rope(
    buf: u32,
    n_rot: u32,
    base: f32,
    head_dim: u32,
    n_heads: u32,
    start_pos: u32,
    n_tok: u32,
    freqs: Option<&[f32]>,
) {
    unsafe {
        imparo_metal_rope(
            buf,
            n_rot,
            base,
            head_dim,
            n_heads,
            start_pos,
            n_tok,
            freqs.map_or(std::ptr::null(), <[f32]>::as_ptr),
            freqs.map_or(0, |f| u32::try_from(f.len()).unwrap_or(0)),
        );
    }
}
/// Encodes a greedy argmax over `n` floats of `src`; `dst` receives ONE u32 (the index,
/// smallest on ties) at offset 0, to be read back with [`read`] and `f32::to_bits`.
pub fn argmax(src: u32, dst: u32, n: u32) {
    unsafe { imparo_metal_argmax(src, dst, n) }
}
/// Sets the KV cache storage types (GGML ids: 1 f16, 2 q4_0, 8 q8_0).
/// The configured cache types as a short tag -- the same string the tuner keys its stored
/// config by, so a tuning measured on one cache can never be applied to another.
///
/// Encoding is this crate's own (1 = f16, 2 = q4_0, 8 = q8_0), so the mapping lives here
/// rather than in the store, which has no business knowing it.
#[must_use]
pub fn kv_tag() -> String {
    let (mut k, mut v) = (1u32, 1u32);
    unsafe { imparo_metal_kv_types(&raw mut k, &raw mut v) };
    let name = |t: u32| match t {
        2 => "q4_0",
        8 => "q8_0",
        _ => "f16",
    };
    if k == v {
        name(k).to_string()
    } else {
        format!("{}-{}", name(k), name(v))
    }
}

pub fn set_kv_types(k: u32, v: u32) {
    unsafe { imparo_metal_set_kv_types(k, v) }
}

/// Restore raw bytes into a KV cache buffer (call after the GPU is idle) -- the KV
/// pool's restore path.
pub fn write_kv_bytes(layer: u32, is_v: bool, off: u64, src: &[u8]) {
    unsafe {
        imparo_metal_write_kv(
            layer,
            u32::from(is_v),
            off,
            src.as_ptr(),
            src.len() as u64,
        );
    }
}

/// The pool assigns physical blocks: replace a layer's 64-cell page table.
pub fn set_kv_page_table(layer: u32, entries: &[u32]) {
    unsafe { imparo_metal_set_kv_pages(layer, entries.as_ptr(), entries.len() as u32) };
}

/// The pool points a windowed layer at the live conversation's own ring.
pub fn set_kv_region(layer: u32, k_off: u64, v_off: u64) {
    unsafe { imparo_metal_set_kv_region(layer, k_off, v_off) };
}

/// Debug: raw bytes from a KV cache buffer (call after the GPU is idle).
pub fn read_kv_bytes(layer: u32, is_v: bool, off: u64, dst: &mut [u8]) {
    unsafe {
        imparo_metal_read_kv(
            layer,
            u32::from(is_v),
            off,
            dst.as_mut_ptr(),
            dst.len() as u64,
        );
    }
}

/// Dequantises one layer's K or V cache into the prefill scratch; no-op under f16.
/// In-place blockwise Hadamard rotation (64-value blocks); the quantized-KV rotation.
pub fn hadamard(buf: u32, n: u32, nrot: u32) {
    unsafe { imparo_metal_hadamard(buf, n, nrot) }
}

pub fn kv_store(
    src: u32,
    layer: u32,
    width: u32,
    start_pos: u32,
    n_tok: u32,
    is_v: bool,
    ring: u32,
) {
    unsafe {
        imparo_metal_kv_store(
            src,
            layer,
            width,
            start_pos,
            n_tok,
            u32::from(is_v),
            ring,
        );
    }
}
#[allow(clippy::too_many_arguments)]
pub fn attention(
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
) {
    unsafe {
        imparo_metal_attention(
            kv_layer, head_dim, n_heads, n_kv, kv_width, start_pos, window, n_tok,
            max_scores, ring,
        );
    }
}
pub fn act_mul(a: u32, b: u32, n: u32) {
    unsafe { imparo_metal_act_mul(a, b, n) }
}
pub fn act(a: u32, n: u32) {
    unsafe { imparo_metal_act(a, n) }
}

/// LFM2's gated short convolution; see `Backend::shortconv`.
#[allow(clippy::too_many_arguments)]
pub fn shortconv(
    bcx: u32,
    w_off: u64,
    state: u32,
    state_off: u32,
    out: u32,
    width: u32,
    kern: u32,
    n_tok: u32,
) {
    unsafe { imparo_metal_shortconv(bcx, w_off, state, state_off, out, width, kern, n_tok) }
}

/// The short-conv state at a boundary inside the batch; see `Backend::shortconv_snapshot`.
#[allow(clippy::too_many_arguments)]
pub fn shortconv_snapshot(
    bcx: u32,
    state: u32,
    state_off: u32,
    snap: u32,
    snap_off: u32,
    width: u32,
    kern: u32,
    n_tok: u32,
) {
    unsafe {
        imparo_metal_shortconv_snapshot(
            bcx, state, state_off, snap, snap_off, width, kern, n_tok,
        );
    }
}
pub fn add(a: u32, b: u32, n: u32) {
    unsafe { imparo_metal_add(a, b, n) }
}
pub fn add_scale(a: u32, b: u32, k: f32, n: u32) {
    unsafe { imparo_metal_add_scale(a, b, k, n) }
}
pub fn mul_strided(
    a: u32,
    b: u32,
    n: u32,
    b_off: u32,
    b_stride: u32,
    a_stride: u32,
    n_tok: u32,
) {
    unsafe { imparo_metal_mul_strided(a, b, n, b_off, b_stride, a_stride, n_tok) }
}
pub fn scale(a: u32, k: f32, n: u32) {
    unsafe { imparo_metal_scale(a, k, n) }
}
pub fn copy(dst: u32, src: u32, n: u32) {
    unsafe { imparo_metal_copy(dst, src, n) }
}
pub fn softcap(a: u32, cap: f32, n: u32) {
    unsafe { imparo_metal_softcap(a, cap, n) }
}
pub fn ple_combine(
    proj: u32,
    emb: u32,
    width: u32,
    emb_scale: f32,
    comb_scale: f32,
    n_tok: u32,
) {
    unsafe { imparo_metal_ple_combine(proj, emb, width, emb_scale, comb_scale, n_tok) }
}

/// Upload token ids to a device buffer.
///
/// The transfer primitive is typed `f32` because every other buffer holds floats; token
/// ids are the one exception, and reinterpreting them here keeps that special case in the
/// backend crate rather than making every caller cast.
pub fn write_u32(id: u32, off: u64, src: &[u32]) {
    let as_f32 =
        unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<f32>(), src.len()) };
    write(id, off, as_f32);
}

/// Gather each token's per-layer embedding row from the Q4_0 table and fold it into
/// `proj` in one dispatch, instead of one row dispatch per token into a staging buffer.
pub fn ple_gather_combine(
    proj: u32,
    tokens_buf: u32,
    w_offset: u64,
    width: u32,
    emb_scale: f32,
    comb_scale: f32,
    n_tok: u32,
) {
    unsafe {
        imparo_metal_ple_gather_combine(
            proj,
            tokens_buf,
            // WAS `as u32`, which truncated silently at the FFI boundary before the
            // kernel ever saw it.
            w_offset,
            width,
            emb_scale,
            comb_scale,
            n_tok,
        );
    }
}

#[must_use]
pub fn available() -> bool {
    true
}

/// Applies a stored configuration when one matches this host. Returns the prefill batch.
///
/// A missing or mismatched file is not an error: the engine runs on defaults and says so,
/// because the only cost of being wrong here is leaving performance unclaimed.
/// `model_bytes` is the size of the weights file this process is about to run.
///
/// The stored configuration is measured END TO END on one model: `lanes` and `rt_shape`
/// are chosen against that model's tensor shapes, `batch` against its layer count. The
/// host fingerprint says nothing about which model that was, so tuning on one model and
/// then running another silently applies the first one's geometry. Comparing sizes is
/// coarse -- two different files could match -- and that is acceptable here for the reason
/// the design doc gives: being wrong about this key costs a re-benchmark, never a wrong
/// answer.
#[must_use]
pub fn apply_host_config(model_bytes: u64) -> Option<usize> {
    use imparo_backend::BackendKnobs as _;
    // The registry is the one enumeration: every stored key is applied through its
    // declared hook, so a knob the tuner writes cannot be silently dropped here --
    // this function once hardcoded the list and lost first `lanes`, then
    // `gemv_max_tok`, and read space v10 after the space moved to v11.
    let space = MetalBackend.space_version();
    let c = imparo_host::read_for_device(model_bytes, false, ("metal", space, &kv_tag()))?;
    // MEASURED ground truth first, because shape values are DERIVED from it and the
    // derivation runs when the kernel library is compiled -- which is after this and
    // before anything else. A missing profile leaves the compiled fallback in place.
    for (k, v) in &c.device {
        if k == "device_max_accumulators" {
            set_measured_max_acc(u32::try_from(*v).unwrap_or(0));
        }
    }
    let reg = MetalBackend.knob_registry();
    for (k, v) in &c.knobs {
        match reg.iter().find(|d| d.name == k.as_str()) {
            Some(d) => (d.apply)(*v),
            None => eprintln!(
                "[imparo] host config key '{k}' unknown to this build; \
                               ignored"
            ),
        }
    }
    eprintln!(
        "[imparo] host config loaded from {}",
        imparo_host::path_for(&imparo_host::fingerprint_for("metal", space, &kv_tag())).display()
    );
    c.batch
}
