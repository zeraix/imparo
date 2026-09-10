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

/// THE BRICK'S GATE, test support. Runs the MSL decode brick (`tm_sub32`) over one
/// tensor row's SCALE RUN and PAYLOAD RUN -- the two byte runs the tile-major layout
/// concatenates -- and returns `n_elems` floats.
///
/// It takes the runs rather than blocks because that is exactly the brick's contract, so
/// the decode is gated with no addressing mixed in; it is also the only shape that can
/// express IQ3_S, whose scales are TWO spans of the source block. `tests/brick_rows.rs`
/// splits by `TmRule`'s spans -- the same table the repack moves bytes with -- and diffs
/// the result against `imparo_cpu::quants::row_codec`, which is itself pinned bit-exact
/// against llama.cpp's own dequantiser.
///
/// # Errors
/// The backend's return code when the device, the pipeline or the dispatch fails.
#[doc(hidden)]
pub fn decode_probe_for_tests(
    wfmt: u32,
    scales: &[u8],
    payload: &[u8],
    n_elems: usize,
) -> Result<Vec<f32>, i32> {
    // The probe needs the device and the compiled library, which `imparo_metal_init`
    // builds; the process runs one model, so this init is the test's model.
    static ONCE: std::sync::Once = std::sync::Once::new();
    static DUMMY: [u8; 64] = [0; 64];
    ONCE.call_once(|| {
        let rc = unsafe { imparo_metal_init(DUMMY.as_ptr().cast(), DUMMY.len() as u64) };
        assert!(rc == 0 || rc == 4, "metal init for the brick gate: rc={rc}");
    });
    let mut out = vec![0.0_f32; n_elems];
    let rc = unsafe {
        imparo_metal_decode_probe(
            wfmt,
            scales.as_ptr().cast(),
            scales.len() as u64,
            payload.as_ptr().cast(),
            payload.len() as u64,
            u32::try_from(n_elems).map_err(|_| -1_i32)?,
            out.as_mut_ptr(),
        )
    };
    if rc == 0 { Ok(out) } else { Err(rc) }
}

unsafe extern "C" {
    fn imparo_metal_set_qcomb_nsg(n: u32);
    fn imparo_metal_set_attn_fa(v: u32);
    fn imparo_metal_set_fa_nsg(v: u32);
    fn imparo_metal_fa_nsg() -> u32;
    fn imparo_metal_fa_has_hd(hd: u32) -> u32;
    fn imparo_metal_fa_nsg_mask(hd: u32) -> u32;
    fn imparo_metal_init(base: *const core::ffi::c_void, len: u64) -> i32;
    /// THE BRICK'S GATE (test-only). Decodes row-major blocks of `wfmt` through the same
    /// `tm_sub32` the GEMM, the GEMV and the gather call, so `tests/brick_rows.rs` can diff
    /// the MSL transcription against the CPU row codec. Not on any serving path.
    fn imparo_metal_decode_probe(
        wfmt: u32,
        scales: *const core::ffi::c_void,
        n_scale_bytes: u64,
        payload: *const core::ffi::c_void,
        n_pay_bytes: u64,
        n_elems: u32,
        out: *mut f32,
    ) -> i32;
    fn imparo_metal_working_set_budget() -> u64;
    fn imparo_metal_set_placement(segs: *const WSegWire, n: u32, budget: u64);
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
    fn imparo_metal_scoremix_rate(
        tgs: u32,
        sgs: u32,
        iters: u32,
        stride: u32,
        kspan: u32,
    ) -> f64;
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
    fn imparo_metal_stage_rows(
        off: u64,
        row_bytes: u32,
        ids: *const u32,
        n: u32,
        dst: u32,
    ) -> i32;
    fn imparo_metal_transform_weights(
        jobs: *const WXformWire,
        n: u32,
        applied: *mut u8,
    ) -> i32;
    fn imparo_metal_read_weight_bytes(off: u64, bytes: u64, out: *mut u8) -> i32;
    fn imparo_metal_set_weight_kind_types(pairs: *const u32, n: u32);
    fn imparo_metal_refused_dispatches() -> u64;
    fn imparo_metal_rt_route_kinds() -> u64;
    fn imparo_metal_ple_gather_combine_staged(
        proj: u32,
        rows: u32,
        width: u32,
        emb_scale: f32,
        comb_scale: f32,
        n_tok: u32,
    );
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
    fn imparo_metal_set_q8_tm_decode_sgs(sgs: u32);
    fn imparo_metal_q8_tm_decode_sgs() -> u32;
    fn imparo_metal_q8_decode_sgs() -> u32;
    fn imparo_metal_set_q8_decode_rows(rows: u32);
    fn imparo_metal_q8_decode_rows() -> u32;
    fn imparo_metal_set_q8_batch_sgs(sgs: u32);
    fn imparo_metal_q8_batch_sgs() -> u32;
    fn imparo_metal_set_q8_token_tile(tile: u32);
    fn imparo_metal_q8_token_tile() -> u32;
    fn imparo_metal_set_st_gemm_shape(shape: u32);
    fn imparo_metal_set_st_gemm_shape_pin(shape: u32);
    fn imparo_metal_set_q8_grid_token_x(on: u32);
    fn imparo_metal_q8_pick_shape(n_tok: u32, n_in: u32) -> u32;
    fn imparo_metal_q8_shape_tokens(shape: u32) -> u32;
    fn imparo_metal_set_q8_skip(bits: u32);
    fn imparo_metal_set_q8_typed_scale(on: u32);
    fn imparo_metal_set_q8_dev_a(on: u32);
    fn imparo_metal_set_q8_clamp_edge(on: u32);
    fn imparo_metal_set_q8_mma_fence(on: u32);
    fn imparo_metal_set_rt_mma_fence(on: u32);
    fn imparo_metal_set_attn_skip(bits: u32);
    fn imparo_metal_st_gemm_shape() -> u32;
    fn imparo_metal_set_st_gemm_large_shape(shape: u32);
    fn imparo_metal_st_gemm_large_shape() -> u32;
    fn imparo_metal_set_q8_full_tiles(on: u32);
    fn imparo_metal_q8_full_tiles() -> u32;
    fn imparo_metal_set_q8_gemv_max_tok(n: u32);
    fn imparo_metal_q8_gemv_max_tok() -> u32;
    fn imparo_metal_set_q8_all(on: u32);
    fn imparo_metal_st_gemm_shapes() -> u32;
    fn imparo_metal_set_q8_design(v: u32);
    fn imparo_metal_q8_design() -> u32;
    fn imparo_metal_q8_design_legal(v: u32) -> u32;
    fn imparo_metal_q8_designs() -> u32;
    fn imparo_metal_set_attn_live_mask(on: u32);
    fn imparo_metal_attn_live_mask() -> u32;
    fn imparo_metal_set_attention_head_dims(hds: *const u32, n: u32);
    fn imparo_metal_set_attention_kv_widths(ws: *const u32, n: u32);
    fn imparo_metal_set_qcomb_pt(v: u32);
    fn imparo_metal_qcomb_pt() -> u32;
    fn imparo_metal_qcomb_has_hd(hd: u32) -> u32;
    fn imparo_metal_qcomb_pt_limit(hd: u32) -> u32;
    fn imparo_metal_qcomb_nsg_mask(hd: u32) -> u32;
    fn imparo_metal_qcomb_nsg() -> u32;
    fn imparo_metal_lanes_min() -> u32;
    fn imparo_metal_lanes_max() -> u32;
    fn imparo_metal_set_qcomb_mask(m: u32);
    fn imparo_metal_qcomb_mask() -> u32;
    fn imparo_metal_qcomb_slot_count() -> u32;
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
    fn imparo_metal_kv_page_cells() -> u32;
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
    fn imparo_metal_set_attn_fd(on: u32);
    fn imparo_metal_set_attn_fd_chunk(v: u32);
    fn imparo_metal_attn_fd_chunk() -> u32;
    fn imparo_metal_set_attn_vec_max_keys(v: u32);
    fn imparo_metal_attn_vec_max_keys() -> u32;
    fn imparo_metal_set_mega_tgs(v: u32);
    fn imparo_metal_set_mega_nsg(v: u32);
    fn imparo_metal_mega_tgs_current() -> u32;
    fn imparo_metal_mega_nsg_current() -> u32;
    fn imparo_metal_mega_threads_limit() -> u32;
    fn imparo_metal_gpu_cores() -> u32;
    fn imparo_metal_mega_level() -> u32;
    fn imparo_metal_attn_fd_chunk_mask(hd: u32, share: u32) -> u32;
    fn imparo_metal_sgs_current() -> u32;
    fn imparo_metal_attn_threads_current() -> u32;
    fn imparo_metal_attn_threads_pf_current() -> u32;
    fn imparo_metal_set_rt_shape(i: u32) -> i32;
    fn imparo_metal_prof_enable(on: u32);
    fn imparo_metal_prof_barriers() -> u64;
    fn imparo_metal_prof_cats(secs: *mut f64, calls: *mut u64, n: *mut u32);
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
    fn imparo_metal_zero(id: u32, off: u64, n: u64);
    fn imparo_metal_wire_weights(stall_budget_s: f64);
    fn imparo_metal_set_weight_path(path: *const std::os::raw::c_char);
    fn imparo_metal_longest_cb_us() -> f64;
    fn imparo_metal_read(id: u32, off: u64, dst: *mut f32, n: u64);
    fn imparo_metal_end_async() -> i32;
    fn imparo_metal_wait_outstanding() -> i32;
    fn imparo_metal_mega_recover() -> i32;
    fn imparo_metal_mega_reserve(n_mid: u32, attn_heads: u32, attn_hd: u32) -> i32;
    fn imparo_metal_decode_pipelining() -> u32;
    fn imparo_metal_stages_rows() -> u32;
    fn imparo_metal_argmax_feed(
        src: u32,
        tokens: u32,
        pick: u32,
        pick_slot: u32,
        n: u32,
    );
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
    fn imparo_metal_mega_layer(e: *const MegaEntryFfi) -> bool;
    fn imparo_metal_mega_wfmt(wkind: u32) -> u32;
    fn imparo_metal_ffn_persistent(
        gate_off: u64,
        up_off: u64,
        down_off: u64,
        n_in: u32,
        n_mid: u32,
        n_out: u32,
        src: u32,
        gtmp: u32,
        utmp: u32,
        dst: u32,
    ) -> bool;
    fn imparo_metal_matmat_gated(
        gate_kind: u32,
        gate_off: u64,
        up_kind: u32,
        up_off: u64,
        n_in: u32,
        n_out: u32,
        src: u32,
        dst: u32,
        n_tok: u32,
    ) -> u32;
    fn imparo_metal_row(
        wkind: u32,
        w_off: u64,
        width: u32,
        index: u32,
        scale: f32,
        dst: u32,
        dst_off: u32,
    );
    fn imparo_metal_mega_program_end();
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
    fn imparo_metal_rms_norm_add_row(
        dst: u32,
        src: u32,
        add: u32,
        w1_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        out_scale: f32,
        dual: u32,
        w2_off: u64,
        out: u32,
    ) -> bool;
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
    fn imparo_metal_add_rms_norm(
        dst: u32,
        resid: u32,
        other: u32,
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
    fn imparo_metal_head_norm_rope(
        buf: u32,
        w_off: u64,
        head_dim: u32,
        eps: f32,
        n_heads: u32,
        start_pos: u32,
        n_tok: u32,
        n_rot: u32,
        base: f32,
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
        scale: f32,
    );
    fn imparo_metal_act_mul(a: u32, b: u32, n: u32);
    fn imparo_metal_act(a: u32, n: u32);
    fn imparo_metal_causal_conv(
        form: u32,
        src: u32,
        w_off: u64,
        state: u32,
        state_off: u32,
        state_out_off: u32,
        out: u32,
        width: u32,
        kern: u32,
        n_tok: u32,
    );
    fn imparo_metal_causal_conv_snapshot(
        form: u32,
        src: u32,
        state: u32,
        state_off: u32,
        snap: u32,
        snap_off: u32,
        width: u32,
        kern: u32,
        n_tok: u32,
    );
    fn imparo_metal_delta_net(
        qkv: u32,
        alpha: u32,
        beta: u32,
        a_off: u64,
        dt_off: u64,
        state: u32,
        state_off: u32,
        state_out_off: u32,
        out: u32,
        k_heads: u32,
        v_heads: u32,
        key_dim: u32,
        value_dim: u32,
        n_tok: u32,
        eps: f32,
        // The fused epilogue: IMPARO_NO_EPILOGUE in norm_w_off means "not fused".
        norm_w_off: u64,
        gate: u32,
    ) -> bool;
    fn imparo_metal_set_recurrent_dims(key_dim: u32, value_dim: u32);
    fn imparo_metal_supports_gated_delta() -> u32;
    fn imparo_metal_delta_net_fuses_epilogue() -> u32;
    fn imparo_metal_mul_sigmoid(
        a: u32,
        b: u32,
        n: u32,
        b_off: u32,
        b_stride: u32,
        a_stride: u32,
        n_row: u32,
    );
    fn imparo_metal_copy_strided(
        dst: u32,
        src: u32,
        width: u32,
        src_off: u32,
        src_stride: u32,
        n_row: u32,
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
    fn imparo_metal_copy_range(dst: u32, dst_off: u32, src: u32, src_off: u32, n: u32);
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
/// Commit the region without waiting; see `Backend::end_async`.
///
/// # Errors
/// The command buffer's error code.
pub fn end_async() -> Result<(), i32> {
    let rc = unsafe { imparo_metal_end_async() };
    if rc == 0 { Ok(()) } else { Err(rc) }
}
/// After a mega-kernel failure, once every outstanding region is retired and the failed
/// step's state is rolled back: reset the kernel's sync words and hold the route for a
/// backoff of regions (see `Backend::mega_recover`).
///
/// # Errors
/// 1 when a region is still outstanding (the engine must retire it first).
/// Reserves the mega-kernel's scratch once at load: the model's widest FFN row and the deep
/// attention body's partials (`attn_heads` query heads at the widest head `attn_hd`).
pub fn mega_reserve(n_mid: u32, attn_heads: u32, attn_hd: u32) -> Result<(), i32> {
    let rc = unsafe { imparo_metal_mega_reserve(n_mid, attn_heads, attn_hd) };
    if rc == 0 { Ok(()) } else { Err(rc) }
}
pub fn mega_recover() -> Result<(), i32> {
    let rc = unsafe { imparo_metal_mega_recover() };
    if rc == 0 { Ok(()) } else { Err(rc) }
}
/// Retire the oldest outstanding region; see `Backend::wait_outstanding`.
///
/// # Errors
/// The command buffer's error code.
pub fn wait_outstanding() -> Result<(), i32> {
    let rc = unsafe { imparo_metal_wait_outstanding() };
    if rc == 0 { Ok(()) } else { Err(rc) }
}
#[must_use]
pub fn decode_pipelining() -> bool {
    unsafe { imparo_metal_decode_pipelining() != 0 }
}
#[must_use]
pub fn stages_rows() -> bool {
    unsafe { imparo_metal_stages_rows() != 0 }
}
pub fn argmax_feed(src: u32, tokens: u32, pick: u32, pick_slot: u32, n: u32) {
    unsafe { imparo_metal_argmax_feed(src, tokens, pick, pick_slot, n) }
}
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
    set_q8_tm_decode_sgs,
    q8_tm_decode_sgs,
    imparo_metal_set_q8_tm_decode_sgs,
    imparo_metal_q8_tm_decode_sgs
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
    set_st_gemm_shape,
    st_gemm_shape,
    imparo_metal_set_st_gemm_shape,
    imparo_metal_st_gemm_shape
);
q8_knob!(
    set_st_gemm_large_shape,
    st_gemm_large_shape,
    imparo_metal_set_st_gemm_large_shape,
    imparo_metal_st_gemm_large_shape
);
q8_knob!(
    set_q8_full_tiles,
    q8_full_tiles,
    imparo_metal_set_q8_full_tiles,
    imparo_metal_q8_full_tiles
);
q8_knob!(
    set_q8_design,
    q8_design,
    imparo_metal_set_q8_design,
    imparo_metal_q8_design
);
/// Whether a `q8_design` value can run here: 0 (st_gemm) always; k >= 1 when rt shape
/// k - 1 passes the register model. Asked of the bridge, which owns both tables.
#[must_use]
pub fn q8_design_legal(v: u32) -> bool {
    unsafe { imparo_metal_q8_design_legal(v) != 0 }
}
/// Number of `q8_design` values: st_gemm plus one per RT_SHAPES row.
#[must_use]
pub fn q8_designs() -> u32 {
    unsafe { imparo_metal_q8_designs() }
}
q8_knob!(
    set_q8_gemv_max_tok,
    q8_gemv_max_tok,
    imparo_metal_set_q8_gemv_max_tok,
    imparo_metal_q8_gemv_max_tok
);
/// Which of THIS MODEL's head dims take qcomb: one bit per dim it uses, bit i for the
/// i-th. A clear bit falls through to qtile, whose head dim is dynamic.
///
/// A mask rather than a size threshold, because a threshold assumes qcomb-worthiness rises
/// with the dim and nothing measured that. The candidate set is derived from
/// `qcomb_slot_count`, so no ladder is written down.
///
/// BIT-AFFECTING: qtile and qcomb sum the same values in different orders. The default
/// preserves today's routing so the pins hold; it is not a measured rule -- qcomb at
/// head_dim 64 measured 3.7% FASTER than qtile.
/// The attention head dims this model uses. Compiled for, not looked up: the dim sizes a
/// register array, which the Metal compiler requires to be a constant expression. Must be
/// called BEFORE init -- the library bakes them in.
pub fn set_attention_head_dims(dims: &[u32]) {
    unsafe {
        imparo_metal_set_attention_head_dims(
            dims.as_ptr(),
            u32::try_from(dims.len()).unwrap_or(u32::MAX),
        );
    }
}

/// K/V row widths for the head-dim slots, same order as `set_attention_head_dims`; set
/// BEFORE init so the prefill attention library compiles the stride as a constant.
pub fn set_attention_kv_widths(widths: &[u32]) {
    unsafe {
        imparo_metal_set_attention_kv_widths(
            widths.as_ptr(),
            u32::try_from(widths.len()).expect("kv width count fits u32"),
        );
    }
}

/// The FA attention op's simdgroups per threadgroup (knob `attn_fa_nsg`). A template
/// parameter of the kernel, so it is injected at library compile: set BEFORE init.
pub fn set_fa_nsg(n: u32) {
    unsafe { imparo_metal_set_fa_nsg(n) }
}

/// What the backend would compile now: the override when set, else the default.
#[must_use]
pub fn fa_nsg() -> u32 {
    unsafe { imparo_metal_fa_nsg() }
}

/// The simdgroup counts LEGAL for the FA op at this head dim (the kernel's divisibility
/// rules and the device's thread cap), for the registry's candidates. Empty when the op
/// is not built for the dim.
pub fn fa_nsg_candidates(hd: u32) -> Vec<u32> {
    let mask = unsafe { imparo_metal_fa_nsg_mask(hd) };
    (0..6)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| 1u32 << i)
        .collect()
}

/// Keys per flash-decoding slice (knob `attn_fd_chunk`); a runtime value, set any time.
pub fn set_attn_fd_chunk(v: u32) {
    unsafe { imparo_metal_set_attn_fd_chunk(v) }
}

#[must_use]
pub fn attn_fd_chunk() -> u32 {
    unsafe { imparo_metal_attn_fd_chunk() }
}

/// The vector decode kernel's span limit in keys (knob `attn_vec_max_keys`): a single query
/// over at most this many keys takes the one-simdgroup-per-position kernel, which reads K/V
/// once per query head; longer spans take the routes that read K once per KV head.
pub fn set_attn_vec_max_keys(v: u32) {
    unsafe { imparo_metal_set_attn_vec_max_keys(v) }
}

#[must_use]
pub fn attn_vec_max_keys() -> u32 {
    unsafe { imparo_metal_attn_vec_max_keys() }
}

/// Mega (persistent) block grid: threadgroups (knob `mega_tgs`) and simdgroups per
/// threadgroup (knob `mega_nsg`). The host clamps both so that tgs * nsg * 32 stays within
/// `mega_threads_limit` -- every threadgroup of the grid must be resident at once.
pub fn set_mega_tgs(v: u32) {
    unsafe { imparo_metal_set_mega_tgs(v) }
}
pub fn set_mega_nsg(v: u32) {
    unsafe { imparo_metal_set_mega_nsg(v) }
}
#[must_use]
pub fn mega_tgs_current() -> u32 {
    unsafe { imparo_metal_mega_tgs_current() }
}
#[must_use]
pub fn mega_nsg_current() -> u32 {
    unsafe { imparo_metal_mega_nsg_current() }
}
/// gpu_cores * the mega pipeline's maxTotalThreadsPerThreadgroup: the co-residency bound
/// the grid must stay under. 0 before init or without the pipeline.
#[must_use]
pub fn mega_threads_limit() -> u32 {
    unsafe { imparo_metal_mega_threads_limit() }
}
/// The GPU core count read from the IORegistry; 0 when unreadable.
#[must_use]
pub fn gpu_cores() -> u32 {
    unsafe { imparo_metal_gpu_cores() }
}
/// IMPARO_MEGA_FFN level: 0 off, 1 the FFN block, 2 the FFN + PLE block.
#[must_use]
pub fn mega_level() -> u32 {
    unsafe { imparo_metal_mega_level() }
}

/// The chunks LEGAL for the flash-decoding route at this head dim and GQA share (the
/// slice's scores must fit the device's threadgroup memory). Empty where the route is not
/// built.
pub fn attn_fd_chunk_candidates(hd: u32, share: u32) -> Vec<u32> {
    let mask = unsafe { imparo_metal_attn_fd_chunk_mask(hd, share) };
    (0..5)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| 128u32 << i)
        .collect()
}

/// Whether the FA op serves this head dim (slot 0, hd <= 128).
pub fn fa_has_hd(hd: u32) -> bool {
    unsafe { imparo_metal_fa_has_hd(hd) != 0 }
}

pub fn qcomb_has_hd(hd: u32) -> bool {
    unsafe { imparo_metal_qcomb_has_hd(hd) != 0 }
}

/// The lanes the Q4 matmat pipeline table is actually built for, read from the table
/// rather than written down beside it.
#[must_use]
pub fn lanes_built() -> (u32, u32) {
    unsafe { (imparo_metal_lanes_min(), imparo_metal_lanes_max()) }
}

/// Set the qcomb simdgroup count; 0 restores the backend's own derivation.
pub fn set_qcomb_nsg(n: u32) {
    unsafe { imparo_metal_set_qcomb_nsg(n) }
}

/// What the backend would use now: the override when set, else what it derives.
pub fn qcomb_nsg() -> u32 {
    unsafe { imparo_metal_qcomb_nsg() }
}

/// The simdgroup counts that are LEGAL at this head dim, for the registry's candidates.
///
/// Derived by the backend from the same test its own derivation uses -- NDB whole, room
/// for a second threadgroup, accumulator budget -- and intersected across the compiled qt
/// variants, because one knob value is applied to both. Empty when the device profile has
/// not been measured, which is the caller's cue to leave the derivation alone.
pub fn qcomb_nsg_candidates(hd: u32) -> Vec<u32> {
    let mask = unsafe { imparo_metal_qcomb_nsg_mask(hd) };
    (0..6)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| 1u32 << i)
        .collect()
}

pub fn qcomb_pt_limit(hd: u32) -> u32 {
    unsafe { imparo_metal_qcomb_pt_limit(hd) }
}

/// The qcomb position tile. TUNED within a derived bound: `qcomb_derive_pt` says what fits
/// this device, and this picks inside it. The widest tile that fits is NOT the fastest --
/// on E4B q4_0 prefill the derived 480 ran 1.3% slower than 128.
pub fn set_qcomb_pt(v: u32) {
    unsafe { imparo_metal_set_qcomb_pt(v) }
}

/// Current qcomb position tile. See `set_qcomb_pt`.
#[must_use]
pub fn qcomb_pt() -> u32 {
    unsafe { imparo_metal_qcomb_pt() }
}

pub fn set_qcomb_mask(m: u32) {
    unsafe { imparo_metal_set_qcomb_mask(m) }
}

/// Current qcomb slot mask. See `set_qcomb_mask`.
#[must_use]
pub fn qcomb_mask() -> u32 {
    unsafe { imparo_metal_qcomb_mask() }
}

/// How many head dims this model uses; the registry derives its candidates from it.
#[must_use]
pub fn qcomb_slot_count() -> u32 {
    unsafe { imparo_metal_qcomb_slot_count() }
}

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
/// Which Q8 prefill shape a dispatch of `n_tok` tokens over `n_in` inputs would use,
/// with both of the pair's pipelines assumed present.
///
/// Exposed so the padding rule can be asserted WITHOUT a GPU. It used to be inline in the
/// encode path, and the only way to check it was to run the engine with IMPARO_Q8_LOG=1
/// and read which shape came out -- a test for arithmetic that needed a Metal device.
#[must_use]
pub fn q8_pick_shape(n_tok: u32, n_in: u32) -> u32 {
    unsafe { imparo_metal_q8_pick_shape(n_tok, n_in) }
}

/// The token tile one Q8 GEMM shape uses, read from the table rather than written beside
/// it. The mirror's padding has to cover the widest of these.
#[must_use]
pub fn q8_shape_tokens(shape: u32) -> u32 {
    unsafe { imparo_metal_q8_shape_tokens(shape) }
}

#[must_use]
pub fn st_gemm_shapes() -> u32 {
    unsafe { imparo_metal_st_gemm_shapes() }
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

/// Paged attention's page in KV cells, as the shader was COMPILED with it.
///
/// The engine declares `KV_PAGE_CELLS` once and hands it to the MSL compiler as a
/// preprocessor macro, so `pool_caps` reporting this cannot drift from what
/// `kv_slot` actually indexes. It was a literal 64 in both places, which is the
/// shape of failure that reads somebody else's rows and never errors.
#[must_use]
pub fn kv_page_cells() -> u32 {
    unsafe { imparo_metal_kv_page_cells() }
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

/// Per-kernel-category GPU time since the last read, as (name, SECONDS, calls).
///
/// Seconds, not ticks. The backend converts each region's counter ticks with the tick
/// period that region measured, because the period is only known once a region samples the
/// CPU and GPU clocks together. It used to hand back ticks and the caller printed shares,
/// which read as durations and were not: a category's share times WALL divides every
/// category by however much the concurrent ones overlapped, and it understated a serial
/// category by exactly that factor (2026-09-09; the qwen35 delta rule read 1.9 ms/token
/// that way against a real 4.0).
///
/// Concurrent categories each report their own full duration, so the column can sum to
/// more than the wall it ran in. That is overlap, not error -- and it is why a category's
/// figure is comparable with a skip-and-diff only when nothing overlaps it.
pub fn prof_categories() -> Vec<(String, f64, u64)> {
    let (mut secs, mut calls, mut n) = ([0.0f64; 32], [0u64; 32], 0u32);
    unsafe {
        imparo_metal_prof_cats(secs.as_mut_ptr(), calls.as_mut_ptr(), &raw mut n);
    };
    (0..n.min(32) as usize)
        .map(|i| {
            let name = unsafe {
                std::ffi::CStr::from_ptr(imparo_metal_prof_cat_name(i as u32))
            };
            (name.to_string_lossy().into_owned(), secs[i], calls[i])
        })
        .filter(|(_, t, calls)| *t > 0.0 || *calls > 0)
        .collect()
}

/// The profiler's category names, in the backend's own order.
///
/// `imparo_metal_prof_cat_name` returns an empty string past the last category, which is
/// how the count comes back without calling `prof_cats` (that one RESETS the counters).
fn profiler_categories() -> Vec<String> {
    // 32 is the ceiling `prof_categories` already sizes its buffers to; the loop stops at
    // the first empty name long before it, and a table that outgrew 32 would break that
    // function first.
    (0..32)
        .map(|i| unsafe {
            std::ffi::CStr::from_ptr(imparo_metal_prof_cat_name(i))
                .to_string_lossy()
                .into_owned()
        })
        .take_while(|name| !name.is_empty())
        .collect()
}

/// Buffer barriers `haz` emitted since profiling was enabled.
///
/// One per dispatch means the concurrent encoder buys nothing: every kernel's tail waits
/// for the next one's head. Counted rather than reasoned about, because reasoning about
/// which projections are independent has been wrong here before.
#[must_use]
pub fn prof_barriers() -> u64 {
    unsafe { imparo_metal_prof_barriers() }
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
/// One weight segment on the wire to the Metal backend: where a byte range of the mapping
/// lives (docs/memory-tiers-and-fit.md). Layout shared with `WSegWire` in imparo_metal.mm.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WSegWire {
    pub off: u64,
    pub bytes: u64,
    /// 0 = fast tier (wired), 1 = slow tier (pageable mapping), 2 = row-gathered table.
    pub tier: u32,
    pub layer: u32,
}

/// Hand the common runtime's placement to the backend BEFORE the weights are mapped:
/// `init` builds one buffer per segment from it. `budget` is the fast tier the placement
/// was computed against (0 = ask the device).
pub fn set_placement(segs: &[imparo_backend::WeightSegment], budget: u64) {
    let wire: Vec<WSegWire> = segs
        .iter()
        .map(|s| {
            let (tier, layer) = match s.tier {
                imparo_backend::WeightTier::Fast => (0, 0),
                imparo_backend::WeightTier::Slow { layer } => (1, layer),
                imparo_backend::WeightTier::HostStaged => (2, 0),
            };
            WSegWire {
                off: s.offset,
                bytes: s.bytes,
                tier,
                layer,
            }
        })
        .collect();
    unsafe { imparo_metal_set_placement(wire.as_ptr(), wire.len() as u32, budget) }
}

/// The fast tier's size in bytes (Metal's recommended working set), 0 when no device.
#[must_use]
pub fn working_set_budget() -> u64 {
    unsafe { imparo_metal_working_set_budget() }
}

pub unsafe fn init_tuned(base: *const u8, len: u64) -> Result<(), i32> {
    // The buffer table must be able to hold every BufId. Checked rather than assumed:
    // the two numbers live in different languages and an undersized table indexes out of
    // bounds instead of failing.
    if let Err(e) =
        imparo_backend::check_buf_table("metal", unsafe { imparo_metal_buf_count() }
            as usize)
    {
        eprintln!("[imparo] {e}");
        return Err(1);
    }
    // measured host configuration first, explicit env overrides after
    // The tuner applies it too: the stored value is the SEAT a candidate has to beat by
    // the noise floor, which is what makes repeated runs converge rather than disagree
    // (imparo-tune/src/main.rs, at `prepare`). IMPARO_NO_HOSTCONFIG=1 is for a run that
    // must start from compiled defaults on purpose.
    // Measured facts about the machine, before the tuned choices and before the library
    // compiles: every derived shape is computed from these.
    apply_device_profile();
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
    // IMPARO_ATTN_LIVE_MASK=0 forces the qcomb family to run its masking loop even where
    // the tile is provably all-live. PRE-INIT ONLY, and that is the whole point: it is
    // function constant 10, so the builder bakes it in and "exactly one pipeline set
    // exists either way". The tuner runs after init and cannot switch it, so this env is
    // the only way to A/B the fast path -- two processes, one pipeline set each.
    //
    // Not a registry knob: the registry declares what must be MEASURED by the sweep, and
    // a value the sweep cannot move would read as noise. A compiled default with an env
    // override is what knobs.rs prescribes for exactly this case.
    // IMPARO_QCOMB_MASK selects which of THIS model's head dims take qcomb, one bit per
    // dim (0 = none, all bits = all). It is how the route is A/B'd without running a full
    // --allow-bit-changes tune.
    // IMPARO_QCOMB_BLK selects the qcomb score-phase width (2 or 4). It is a registry
    // knob, but the env is how it is A/B'd without a full tune -- and it is how the fact
    // that it reached no dispatch at all was demonstrated.
    // IMPARO_QCOMB_PT overrides the prefill attention position tile. It sets the GLOBAL,
    // so it reaches qcomb and qtile alike; when it lived inside one dispatch's helper the
    // other path silently kept its default.
    if let Ok(v) = std::env::var("IMPARO_QCOMB_PT") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_qcomb_pt(n) }
        }
    }
    if let Ok(v) = std::env::var("IMPARO_QCOMB_BLK") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_qcomb_blk(n) }
        }
    }
    // IMPARO_QCOMB_NSG makes the simdgroup count MEASURABLE. qcomb_derive_nsg's loop
    // starts at 8 and only its upper bound was ever argued, so at head_dim 64 the legal
    // values 1, 2 and 4 were never tried -- and attn_threads_prefill, the lever one would
    // reach for, is inert at prefill under qcomb. Read here because the derivation feeds
    // the library's preprocessor macros, so it must be set before the library compiles.
    // The backend re-checks legality and refuses out loud rather than honouring silently.
    // The FA attention op is the default prefill attention at head dims <= 128 (see the
    // backend's g_attn_fa). IMPARO_ATTN_FA=0 routes those layers to qcomb instead -- the
    // route switch for A/Bs, a config and not a tuner knob.
    if let Ok(v) = std::env::var("IMPARO_ATTN_FA") {
        unsafe { imparo_metal_set_attn_fa(u32::from(v != "0")) };
    }
    if let Ok(v) = std::env::var("IMPARO_QCOMB_NSG") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_qcomb_nsg(n) }
        }
    }
    // The FA op's simdgroup count, overridable for A/Bs the way IMPARO_QCOMB_NSG is; the
    // tuner sets it through the knob's hook. Read before init: it is compile-time.
    if let Ok(v) = std::env::var("IMPARO_FA_NSG") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_fa_nsg(n) }
        }
    }
    if let Ok(v) = std::env::var("IMPARO_QCOMB_MASK") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_qcomb_mask(n) }
        }
    }
    if let Ok(v) = std::env::var("IMPARO_ATTN_LIVE_MASK") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_attn_live_mask(n) }
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
        // ASKED OF THE BACKEND, never retyped here. This list used to be a copy of
        // PROF_CAT_NAME kept in step by a comment, and the two are a silent-failure pair:
        // a category added on one side shifts every index on the other.
        let cats = profiler_categories();
        // NOT EVERY NAME IS HONOURED BY A DISPATCH. This list is the profiler's category
        // table; the skip is a separate mechanism, and only the sites that actually test
        // `g_skip_cat` can be skipped. "attention" is in the table and NO dispatch checks
        // it, so asking to skip attention silently skipped nothing and priced the stage at
        // ZERO -- which reads exactly like the answer "attention is free", on a model where
        // it is the only quadratic stage. Refuse instead of no-opping, and name the lever
        // that does work.
        if v == "attention" {
            eprintln!(
                "[imparo] IMPARO_SKIP_CAT=attention does nothing: no dispatch honours it. \
                 Use IMPARO_SKIP_ATTN=1 (all attention), 2 (hd-512), 3 (hd-256)."
            );
        } else {
            match cats.iter().position(|c| *c == v) {
                Some(i) => unsafe { imparo_metal_set_skip_cat(i as u32) },
                None => eprintln!(
                    "[imparo] IMPARO_SKIP_CAT={v}: not a category; known: {}",
                    cats.join(" ")
                ),
            }
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
    // THE SAME FOR THE Q8 FAMILY, which had the flag and no way to set it. `set_q8_all`
    // existed with an FFI binding and a wrapper and NO CALLER, so only the selected shape
    // was ever built -- and a sweep over the other nine set an index whose pipeline was
    // nil. The sweep printed a full ladder of plausible numbers, rising linearly with the
    // candidate INDEX (4051, 6237, 8455, 10542 ... 20577 us), which is not how a GEMM
    // shape behaves; the incumbent, measured first, could never be beaten. That is how
    // `st_gemm_shape=3` -- "the fork's shape" -- stayed unchallenged while it carried 82%
    // of an LFM2 prefill.
    if std::env::var("IMPARO_Q8_ALL").is_ok_and(|v| v == "1") {
        unsafe { imparo_metal_set_q8_all(1) };
    }
    // Pins the prefill GEMM shape for an end-to-end A/B, over the tuned value. Pre-init
    // like IMPARO_RT_SHAPE, because the index decides which pipelines are compiled --
    // and the pin also outranks `apply_host_config`, which runs later. The dispatch
    // prints the shape it used under IMPARO_Q8_LOG, which is how a pin is checked.
    if let Ok(v) = std::env::var("IMPARO_ST_GEMM_SHAPE") {
        if let Ok(i) = v.parse::<u32>() {
            unsafe { imparo_metal_set_st_gemm_shape_pin(i) }
        }
    }
    // Puts the TOKEN groups on the grid's x axis, the way llama.cpp's mul_mm dispatches.
    // Pre-init: it decides whether the token-major pipeline twins get compiled at all.
    if let Ok(v) = std::env::var("IMPARO_Q8_GRID_TOKEN_X") {
        unsafe { imparo_metal_set_q8_grid_token_x(u32::from(v != "0")) };
    }
    // Attribution probe for the prefill GEMM: 1 no multiply, 2 no staging, 4 no device
    // weight read, summed. Pre-init, because it is a function constant the pipelines are
    // compiled with. The logits it produces are wrong on purpose.
    // The edge-predicate-free entry point, A/B'd end to end. llama.cpp decides this per
    // dispatch from the shape (`bc_out = ne0 % 64 || ne1 % 32`); here it is one tuned
    // global, so an override is the only way to ask what it is worth.
    // The SECOND tile of the pair. Setting it to a different shape turns on the padding
    // rule in `q8_matmat`; leaving it equal to the first is the single-shape path.
    if let Ok(v) = std::env::var("IMPARO_ST_GEMM_LARGE_SHAPE") {
        if let Ok(i) = v.parse::<u32>() {
            set_st_gemm_large_shape(i);
        }
    }
    if let Ok(v) = std::env::var("IMPARO_Q8_FULL_TILES") {
        set_q8_full_tiles(u32::from(v != "0"));
    }
    // Reads a Q8_0 block scale as one `half` instead of rebuilding it from two bytes,
    // the way llama.cpp does. Pre-init: it is a function constant the pipelines carry.
    // Reads the activation operand straight from the f16 mirror, dropping the
    // threadgroup activation stage and with it 2 KB of the allocation. Pre-init: it
    // decides whether the unstaged pipelines are compiled.
    // 0 off, 1 no prefetch, 2 prefetch depth 2, 3 prefetch depth 5.
    if let Ok(v) = std::env::var("IMPARO_Q8_DEV_A") {
        if let Ok(n) = v.parse::<u32>() {
            unsafe { imparo_metal_set_q8_dev_a(n) };
        }
    }
    // Clamp the staging edge instead of branching on it, the way llama.cpp's mul_mm does.
    // Scheduling fences between the operand loads and the multiplies, as llama.cpp has.
    // The same fences asked of the Q4 rt_gemm, which E4B runs. Default off.
    // And for prefill attention, the deep leg's remaining growing term.
    // Phase probe for prefill attention: 1 no score MMA, 2 no softmax, 4 no P x V.
    // The logits it produces are wrong on purpose; only the time is read.
    if let Ok(v) = std::env::var("IMPARO_ATTN_SKIP") {
        if let Ok(bits) = v.parse::<u32>() {
            unsafe { imparo_metal_set_attn_skip(bits) };
        }
    }
    if let Ok(v) = std::env::var("IMPARO_Q8_MMA_FENCE") {
        unsafe { imparo_metal_set_q8_mma_fence(u32::from(v != "0")) };
    }
    if let Ok(v) = std::env::var("IMPARO_RT_MMA_FENCE") {
        unsafe { imparo_metal_set_rt_mma_fence(u32::from(v != "0")) };
    }
    if let Ok(v) = std::env::var("IMPARO_Q8_CLAMP_EDGE") {
        unsafe { imparo_metal_set_q8_clamp_edge(u32::from(v != "0")) };
    }
    if let Ok(v) = std::env::var("IMPARO_Q8_TYPED_SCALE") {
        unsafe { imparo_metal_set_q8_typed_scale(u32::from(v != "0")) };
    }
    if let Ok(v) = std::env::var("IMPARO_Q8_SKIP") {
        if let Ok(bits) = v.parse::<u32>() {
            unsafe { imparo_metal_set_q8_skip(bits) };
        }
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
    // IMPARO_ATTN_FD=0 refuses the flash-decoding route (hd <= 128, f16, GQA) for A/Bs.
    if let Ok(v) = std::env::var("IMPARO_ATTN_FD") {
        unsafe { imparo_metal_set_attn_fd(u32::from(v != "0")) };
    }
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
/// Fill `n` floats of buffer `id` with zero, in place: no host source to allocate,
/// fault in and read back.
pub fn zero(id: u32, off: u64, n: u64) {
    unsafe { imparo_metal_zero(id, off, n) }
}
/// The GGUF the mapping came from, so the repack can read converted tensors with the page
/// cache turned off. A path that will not convert to C (an interior NUL) is simply not set,
/// and the repack keeps reading through the mapping.
pub fn set_weight_path(path: &std::path::Path) {
    let Ok(c) = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()) else {
        return;
    };
    unsafe { imparo_metal_set_weight_path(c.as_ptr()) }
}
/// Make the weights GPU-resident now, so the first request does not wait for it.
pub fn wire_weights(stall_budget_s: f64) {
    unsafe { imparo_metal_wire_weights(stall_budget_s) }
}
/// GPU seconds of the LONGEST single command buffer of the last region.
pub fn longest_cb_gpu_seconds() -> f64 {
    unsafe { imparo_metal_longest_cb_us() * 1e-6 }
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

/// The gated pair as one dispatch: `dst = act(gate @ src) * (up @ src)`, `n_out` wide, with
/// the activation the library was compiled for. Returns false -- and has written nothing --
/// when this route cannot serve it (a decode width, unlike weight kinds, no half mirror,
/// or no pipeline for the tuned shape); the caller then issues the two projections.
#[allow(clippy::too_many_arguments)]
/// The mega entry's slot tables, emitted by build.rs from `mega_slots.rs` (task #158 step 2): the
/// same table the shader's slot constants and the bridge's enums come from.
pub mod mega_slots {
    include!(concat!(env!("OUT_DIR"), "/mega_slots_gen.rs"));
}
/// The roles a pointer slot of the mega entry can carry (mirrors the bridge's enum).
pub const MEGA_SLOT_NONE: u32 = 0;
/// A weight the kernel reads: `off` = its global offset, `id` = the offset-table slot that
/// receives the segment-local offset (`MEGA_NO` = none). Absent or slow-tier refuses the layer.
pub const MEGA_SLOT_WEIGHT: u32 = 1;
/// A weight the kernel reads only under a flag it also receives: absent = a dummy address.
pub const MEGA_SLOT_WEIGHT_OPT: u32 = 2;
/// An activation buffer: `id` = the buffer id, `off` = an element offset. R / W / RW are the
/// hazard roles the bridge forms its masks from.
pub const MEGA_SLOT_BUF_R: u32 = 3;
pub const MEGA_SLOT_BUF_W: u32 = 4;
pub const MEGA_SLOT_BUF_RW: u32 = 5;
/// This layer's K / V cache: the read view (attention over the cache) or the write view (this
/// token's row); the page table.
pub const MEGA_SLOT_KV_K_R: u32 = 6;
pub const MEGA_SLOT_KV_K_W: u32 = 7;
pub const MEGA_SLOT_KV_V_R: u32 = 8;
pub const MEGA_SLOT_KV_V_W: u32 = 9;
pub const MEGA_SLOT_KV_PT: u32 = 10;
/// One pointer slot of the mega entry as the bridge resolves it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MegaSlotFfi {
    pub role: u32,
    pub id: u32,
    pub off: u64,
}
/// One layer of any architecture for `imparo_metal_mega_layer` (task #158 step 2): the program's
/// words and floats, every pointer slot's role and source, and what the bridge decides from --
/// the pipeline family (`arch`: 0 gemma4, 1 LFM2), the IMPARO_MEGA_FFN level the entry needs,
/// the head dim (the pipeline slot; 0 = no attention body), whether the attention phase runs,
/// whether the entry writes the cache row, the cache layer and position, the scratch floats
/// the attention partials may use, the rope factor table. The bridge writes the derived header
/// words (grid, scratch, norm threads, attention split / body / heads per item) itself.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MegaEntryFfi {
    pub arch: u32,
    pub min_level: u32,
    pub head_dim: u32,
    pub attn_on: u32,
    pub kv_write: u32,
    pub kv_layer: u32,
    pub start_pos: u32,
    pub layer: u32,
    pub scratch_rows: u32,
    pub n_freqs: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub freqs: *const f32,
    pub slots: [MegaSlotFfi; mega_slots::MEGA_NP],
    pub u: [u32; mega_slots::MEGA_NU],
    pub f: [f32; mega_slots::MEGA_NF],
}
const _: () = assert!(
    std::mem::size_of::<MegaEntryFfi>()
        == 12 * 4
            + 8
            + mega_slots::MEGA_NP * 16
            + mega_slots::MEGA_NU * 4
            + mega_slots::MEGA_NF * 4
);
impl MegaEntryFfi {
    /// An empty entry of one pipeline family: every slot unused, every word zero.
    #[must_use]
    pub fn new(arch: u32) -> Self {
        Self {
            arch,
            min_level: 2,
            head_dim: 0,
            attn_on: 0,
            kv_write: 0,
            kv_layer: 0,
            start_pos: 0,
            layer: 0,
            scratch_rows: 0,
            n_freqs: 0,
            pad0: 0,
            pad1: 0,
            freqs: std::ptr::null(),
            slots: [MegaSlotFfi {
                role: MEGA_SLOT_NONE,
                id: 0,
                off: 0,
            }; mega_slots::MEGA_NP],
            u: [0; mega_slots::MEGA_NU],
            f: [0.0; mega_slots::MEGA_NF],
        }
    }
}
/// A weight slot: the segment-local offset lands in offset-table slot `off_slot`.
#[must_use]
pub fn mega_slot_weight(off: u64, off_slot: usize) -> MegaSlotFfi {
    MegaSlotFfi {
        role: MEGA_SLOT_WEIGHT,
        id: off_slot as u32,
        off,
    }
}
/// An activation buffer slot with a hazard role and an element offset.
#[must_use]
pub fn mega_slot_buf(role: u32, id: u32, off: u32) -> MegaSlotFfi {
    MegaSlotFfi {
        role,
        id,
        off: u64::from(off),
    }
}
/// A cache view or page-table slot (the layer is the entry's `kv_layer`).
#[must_use]
pub fn mega_slot_kv(role: u32) -> MegaSlotFfi {
    MegaSlotFfi {
        role,
        id: 0,
        off: 0,
    }
}
/// THE LEVEL THE QWEN35 PROGRAM NEEDS, and it is ABOVE the default (5) on purpose.
///
/// Measured 2026-09-08, 16-step decode at position 23 on UD-Q4_K_S, ms per token, two
/// rotated rounds:
///
/// ```text
///   dispatch path            127.16 / 123.76
///   mega tail, 18 x 16       140.81 / 140.22   +11.9%
///   mega tail, 18 x 32       153.09 / 151.58   +21.3%
/// ```
///
/// The trail is IDENTICAL in every run -- the phases are correct, they are not worth it yet.
/// ATTRIBUTED 2026-09-08: the gap was GRID IMBALANCE, not the kernel's arithmetic. Ablating
/// each phase prices the gated pair at 55.1 ms/token against the dispatch path's ~54 (+2%)
/// and the down projection at 36.0 against ~27 (+33%), and down is the phase whose 640 units
/// do not divide the 288-simdgroup grid: three waves run for 2.22 units of work. The grid is
/// derived from the phases' item counts now (18 x 18 here), which takes the route to 131.4
/// with the same step hash. See docs/grid-balance-and-occupancy.md. What is left against the
/// dispatch path is the gated pair's 4% residual waste and the norm and tail phases; the
/// fold's 3064 fewer dispatches only pay once the mixer arms join it. The pre-step state
/// copy that cost 2.4 ms a token here is gone -- the state is planes now (task #165).
/// `IMPARO_MEGA_FFN=6` runs it meanwhile.
pub const MEGA_LEVEL_QWEN35: u32 = 6;

/// The mega row brick's weight format for a wire kind, or 0 when the brick cannot read it
/// (no decode arm, or the tensor kept the row-major layout -- the mega phases read
/// tile-major only). An architecture whose weights differ per tensor asks this per tensor;
/// 0 means REFUSE the layer, never "read it as row-major".
#[must_use]
pub fn mega_wfmt(wkind: u32) -> u32 {
    unsafe { imparo_metal_mega_wfmt(wkind) }
}
/// One layer of any architecture as one persistent dispatch (or one entry of the program run).
/// Returns false, nothing encoded, when the bridge refuses the entry.
pub fn mega_layer(e: &MegaEntryFfi) -> bool {
    unsafe { imparo_metal_mega_layer(e) }
}
/// Level 5 of IMPARO_MEGA_FFN: the q/k/v rows, head norm + rope and the cache write join too.
#[must_use]
pub fn mega_qkv_wanted() -> bool {
    mega_level() >= 5
}
/// The mega program (task #153): encode the pending run of recorded layers (a no-op without one).
pub fn mega_program_end() {
    unsafe { imparo_metal_mega_program_end() }
}
/// Level 4 of IMPARO_MEGA_FFN: the decode attention (vec body) joins the mega block too.
#[must_use]
pub fn mega_attn_wanted() -> bool {
    mega_level() >= 4
}
/// Level 3 of IMPARO_MEGA_FFN: the o_proj rows and the sandwich norm join the mega block.
#[must_use]
pub fn mega_front_wanted() -> bool {
    mega_level() >= 3
}
/// The mega FFN block (gate|up -> act*mul -> down as one persistent dispatch); false when the
/// route is off (IMPARO_MEGA_FFN unset), unavailable, or disabled after a barrier timeout.
#[allow(clippy::too_many_arguments)]
pub fn ffn_persistent(
    gate_off: u64,
    up_off: u64,
    down_off: u64,
    n_in: u32,
    n_mid: u32,
    n_out: u32,
    src: u32,
    gtmp: u32,
    utmp: u32,
    dst: u32,
) -> bool {
    unsafe {
        imparo_metal_ffn_persistent(
            gate_off, up_off, down_off, n_in, n_mid, n_out, src, gtmp, utmp, dst,
        )
    }
}
pub fn matmat_gated(
    gate_kind: u32,
    gate_off: u64,
    up_kind: u32,
    up_off: u64,
    n_in: u32,
    n_out: u32,
    src: u32,
    dst: u32,
    n_tok: u32,
) -> bool {
    unsafe {
        imparo_metal_matmat_gated(
            gate_kind, gate_off, up_kind, up_off, n_in, n_out, src, dst, n_tok,
        ) != 0
    }
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
/// The gemma4 sandwich boundary as one dispatch: dst = (add + rms(src) * w1) * out_scale,
/// and with `dual`, out = rms(dst) * w2. Bit-identical to the separate dispatches (see the
/// kernel). Returns false when nothing was encoded.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_add_row(
    dst: u32,
    src: u32,
    add: u32,
    w1_off: u64,
    width: u32,
    eps: f32,
    n_row: u32,
    out_scale: f32,
    dual: bool,
    w2_off: u64,
    out: u32,
) -> bool {
    unsafe {
        imparo_metal_rms_norm_add_row(
            dst,
            src,
            add,
            w1_off,
            width,
            eps,
            n_row,
            out_scale,
            u32::from(dual),
            w2_off,
            out,
        )
    }
}

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
/// `dst = rms_norm(resid + other) * w` and `resid += other` in one dispatch (the pre-norm
/// residual order); see the native entry's note.
#[allow(clippy::too_many_arguments)]
pub fn add_rms_norm(
    dst: u32,
    resid: u32,
    other: u32,
    w_off: u64,
    width: u32,
    eps: f32,
    n_row: u32,
    row_stride: u32,
    base_off: u32,
) {
    unsafe {
        imparo_metal_add_rms_norm(
            dst, resid, other, w_off, width, eps, n_row, row_stride, base_off,
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
/// Per-head rms_norm then NEOX rope in one dispatch (gemma4's Q / K post-projection).
#[allow(clippy::too_many_arguments)]
pub fn head_norm_rope(
    buf: u32,
    w_off: u64,
    head_dim: u32,
    eps: f32,
    n_heads: u32,
    start_pos: u32,
    n_tok: u32,
    n_rot: u32,
    base: f32,
    freqs: Option<&[f32]>,
) {
    unsafe {
        imparo_metal_head_norm_rope(
            buf,
            w_off,
            head_dim,
            eps,
            n_heads,
            start_pos,
            n_tok,
            n_rot,
            base,
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
    attention_scaled(kv_layer, head_dim, n_heads, n_kv, kv_width, start_pos,
        window, n_tok, max_scores, ring, 1.0);
}

/// FA scales the scores after its half-Q dot product. Other routes retain prescaled Q.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_scaled(
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
    scale: f32,
) {
    unsafe {
        imparo_metal_attention(
            kv_layer, head_dim, n_heads, n_kv, kv_width, start_pos, window, n_tok,
            max_scores, ring, scale,
        );
    }
}
pub fn act_mul(a: u32, b: u32, n: u32) {
    unsafe { imparo_metal_act_mul(a, b, n) }
}
pub fn act(a: u32, n: u32) {
    unsafe { imparo_metal_act(a, n) }
}

/// A causal depthwise convolution; see `Backend::causal_conv`. `form` is the wire value
/// of `ConvForm`, which selects the pipeline.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv(
    form: u32,
    src: u32,
    w_off: u64,
    state: u32,
    state_off: u32,
    state_out_off: u32,
    out: u32,
    width: u32,
    kern: u32,
    n_tok: u32,
) {
    unsafe {
        imparo_metal_causal_conv(
            form, src, w_off, state, state_off, state_out_off, out, width, kern, n_tok,
        );
    }
}

/// The conv state at a boundary inside the batch; see `Backend::causal_conv_snapshot`.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv_snapshot(
    form: u32,
    src: u32,
    state: u32,
    state_off: u32,
    snap: u32,
    snap_off: u32,
    width: u32,
    kern: u32,
    n_tok: u32,
) {
    unsafe {
        imparo_metal_causal_conv_snapshot(
            form, src, state, state_off, snap, snap_off, width, kern, n_tok,
        );
    }
}

/// The gated delta rule; see `Backend::delta_net`. False when the pipeline was not built
/// or was built for other head widths, and then nothing is written.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn delta_net(
    qkv: u32,
    alpha: u32,
    beta: u32,
    a_off: u64,
    dt_off: u64,
    state: u32,
    state_off: u32,
    state_out_off: u32,
    out: u32,
    k_heads: u32,
    v_heads: u32,
    key_dim: u32,
    value_dim: u32,
    n_tok: u32,
    eps: f32,
    // The gated-RMS epilogue's weight offset, or `NO_EPILOGUE` for the unfused form
    // (then `gate` is ignored and the rule writes `out` as before).
    norm_w_off: u64,
    gate: u32,
) -> bool {
    unsafe {
        imparo_metal_delta_net(
            qkv, alpha, beta, a_off, dt_off, state, state_off, state_out_off, out,
            k_heads, v_heads, key_dim, value_dim, n_tok, eps, norm_w_off, gate,
        )
    }
}

/// `norm_w_off` for a `delta_net` call with no fused epilogue. Not a valid weight offset:
/// the same all-ones sentinel the norm kernels use for "this row has no weight".
pub const NO_EPILOGUE: u64 = u64::MAX;

/// Whether the built delta pipeline applies the gated-RMS epilogue itself; see
/// `Backend::delta_net_fuses_epilogue`.
#[must_use]
pub fn delta_net_fuses_epilogue() -> bool {
    // DEFAULT ON, opt OUT with IMPARO_DELTA_EPI=0.
    //
    //
    // The fold is measured and it wins -- +0.75% prefill at 512 tokens, decode exactly
    // flat, four rotated arms with no overlap -- and it is BIT-IDENTICAL to the two
    // dispatches it replaces: the 16-step decode stephash is 79f9225dce7c46b3 with the
    // fold on and off alike, which is also what the engine returned before the fold
    // existed. It was off until the last-bit difference was explained; the explanation is
    // the volatile store on `tout` in imparo_delta_net, which forces the rule's output to
    // round to f32 before the epilogue reads it back. One build, two arms: the gate is an
    // env read, not a comment-out, so the verified build and the measured build are the
    // same file.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DELTA_EPI").as_deref() != Ok("0"))
        && unsafe { imparo_metal_delta_net_fuses_epilogue() != 0 }
}

/// The recurrent head widths, before `init_weights`; see `Backend::set_recurrent_dims`.
pub fn set_recurrent_dims(key_dim: u32, value_dim: u32) {
    unsafe { imparo_metal_set_recurrent_dims(key_dim, value_dim) }
}

/// Whether the delta pipeline exists in the built library.
#[must_use]
pub fn supports_gated_delta() -> bool {
    unsafe { imparo_metal_supports_gated_delta() != 0 }
}

/// `a *= sigmoid(b)` over strided rows; see `Backend::mul_strided_sigmoid`.
#[allow(clippy::too_many_arguments)]
pub fn mul_strided_sigmoid(
    a: u32,
    b: u32,
    n: u32,
    b_off: u32,
    b_stride: u32,
    a_stride: u32,
    n_row: u32,
) {
    unsafe { imparo_metal_mul_sigmoid(a, b, n, b_off, b_stride, a_stride, n_row) }
}

/// One sub-block out of every row; see `Backend::copy_strided`.
pub fn copy_strided(dst: u32, src: u32, width: u32, src_off: u32, src_stride: u32, n_row: u32) {
    unsafe { imparo_metal_copy_strided(dst, src, width, src_off, src_stride, n_row) }
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
pub fn copy_range(dst: u32, dst_off: u32, src: u32, src_off: u32, n: u32) {
    unsafe { imparo_metal_copy_range(dst, dst_off, src, src_off, n) }
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

/// Wire layout of one load-time repack job; mirrors `WXformWire` in imparo_metal.mm.
///
/// The job carries THE RULE, not just the type ids. `TmRule` in imparo-gguf is the one
/// statement of where a block's scales are and where they go; a backend that re-derived
/// that from `from_type` would be a second copy of the layout, and the copy is what puts
/// every payload at a wrong offset the moment a format is added. The repack kernel is
/// therefore format-agnostic: it moves the spans it is given.
///
/// Spans are the scale spans first (`n_scale_spans` of them), then the payload spans, in
/// source order. Mirrors `imparo_backend::WeightBlockLayout`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct WXformWire {
    pub off: u64,
    pub bytes: u64,
    pub from_type: u32,
    pub to_type: u32,
    pub n_in: u32,
    pub n_out: u32,
    pub block_elems: u32,
    pub block_bytes: u32,
    pub unit_rows: u32,
    /// Scale spans + payload spans. At most `MAX_XFORM_SPANS`.
    pub n_spans: u32,
    pub n_scale_spans: u32,
    pub span_off: [u32; imparo_backend::MAX_BLOCK_SPANS],
    pub span_len: [u32; imparo_backend::MAX_BLOCK_SPANS],
}

impl WXformWire {
    /// Fills in the layout half from the job's `WeightBlockLayout`. This crate never
    /// derives a layout: it copies the one the common runtime read from `TmRule`.
    #[must_use]
    pub fn with_layout(mut self, l: &imparo_backend::WeightBlockLayout) -> Self {
        self.block_elems = l.block_elems;
        self.block_bytes = l.block_bytes;
        self.unit_rows = l.unit_rows;
        self.n_spans = l.n_spans;
        self.n_scale_spans = l.n_scale_spans;
        self.span_off = l.span_off;
        self.span_len = l.span_len;
        self
    }
}

/// Load-time repack of fast-tier tensors into private buffers on the GPU. One flag per
/// job: applied or left as it was. Err on a Metal failure.
pub fn transform_weights(jobs: &[WXformWire]) -> Result<Vec<bool>, String> {
    let mut applied = vec![0_u8; jobs.len()];
    let rc = unsafe {
        imparo_metal_transform_weights(
            jobs.as_ptr(),
            jobs.len() as u32,
            applied.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(format!("imparo_metal_transform_weights rc={rc}"));
    }
    Ok(applied.into_iter().map(|a| a != 0).collect())
}

/// Tells the backend which ggml type each wire weight kind is, as flat (wire, ggml) pairs.
/// The table is imparo-gguf's; this crate stores what it is handed and derives nothing.
pub fn set_weight_kind_types(pairs: &[(u32, u32)]) {
    let flat: Vec<u32> = pairs.iter().flat_map(|&(w, g)| [w, g]).collect();
    unsafe { imparo_metal_set_weight_kind_types(flat.as_ptr(), pairs.len() as u32) }
}

/// Dispatches this process refused rather than encoding -- an unserved weight kind, a
/// width no route can walk, a span past the kernel's slices. Nonzero means some op did
/// NOT run; a harness that must have dispatched reads this instead of scanning the log.
#[must_use]
pub fn refused_dispatches() -> u64 {
    unsafe { imparo_metal_refused_dispatches() }
}

/// The weight kinds whose matmul the register-tiled GEMM serves, one bit per wire kind.
/// Asked of the routing itself so an `applies` predicate cannot fall behind it; valid
/// only after `set_weight_kind_types`, which is where the kind -> ggml mapping arrives.
#[must_use]
pub fn rt_route_kinds() -> u64 {
    unsafe { imparo_metal_rt_route_kinds() }
}

/// Weight bytes as the GPU sees them at a file offset (private buffers read back through
/// a blit). Verification only.
pub fn read_weight_bytes(off: u64, dst: &mut [u8]) -> bool {
    unsafe {
        imparo_metal_read_weight_bytes(off, dst.len() as u64, dst.as_mut_ptr()) == 0
    }
}

/// Host-staged tier: the host copies this batch's rows of a row-gathered tensor into `dst`.
/// False when the tensor is not a host-staged segment of the placement (the GPU reads it).
pub fn stage_rows(off: u64, row_bytes: u32, ids: &[u32], dst: u32) -> bool {
    unsafe {
        imparo_metal_stage_rows(off, row_bytes, ids.as_ptr(), ids.len() as u32, dst)
            != 0
    }
}

/// `ple_gather_combine` over staged rows: row t of `rows` is token t.
pub fn ple_gather_combine_staged(
    proj: u32,
    rows: u32,
    width: u32,
    emb_scale: f32,
    comb_scale: f32,
    n_tok: u32,
) {
    unsafe {
        imparo_metal_ple_gather_combine_staged(
            proj, rows, width, emb_scale, comb_scale, n_tok,
        )
    }
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
            proj, tokens_buf,
            // WAS `as u32`, which truncated silently at the FFI boundary before the
            // kernel ever saw it.
            w_offset, width, emb_scale, comb_scale, n_tok,
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
/// Applies this host's MEASURED device profile: the ground truth that shape values are
/// derived from, read before the kernel library is compiled.
///
/// Unconditional, and separate from `apply_host_config`, because the two answer
/// different questions:
///
///   device profile   what this GPU does          measured, always applied
///   knob config      what we chose to do on it   tuned, skipped while tuning
///
/// A tuner must not seed itself from a previous tuning -- so it sets
/// IMPARO_NO_HOSTCONFIG and starts from compiled defaults. It must still derive its
/// shapes against the real machine, or it tunes a library the engine will not compile.
pub fn apply_device_profile() {
    let Some(vals) = imparo_host::read_device_profile("metal") else {
        return;
    };
    for (k, v) in &vals {
        if k == "device_max_accumulators" {
            set_measured_max_acc(u32::try_from(*v).unwrap_or(0));
        }
    }
}

#[must_use]
pub fn apply_host_config(model_bytes: u64) -> Option<usize> {
    use imparo_backend::BackendKnobs as _;
    // The registry is the one enumeration: every stored key is applied through its
    // declared hook, so a knob the tuner writes cannot be silently dropped here --
    // this function once hardcoded the list and lost first `lanes`, then
    // `gemv_max_tok`, and read space v10 after the space moved to v11.
    let space = MetalBackend.space_version();
    let c =
        imparo_host::read_for_device(model_bytes, false, ("metal", space, &kv_tag()))?;
    let reg = MetalBackend.knob_registry();
    let mut applied = 0usize;
    for (k, v) in &c.knobs {
        match reg.iter().find(|d| d.name == k.as_str()) {
            Some(d) => {
                (d.apply)(*v);
                applied += 1;
            }
            None => eprintln!(
                "[imparo] host config key '{k}' unknown to this build; \
                               ignored"
            ),
        }
    }
    // READ BACK WHAT TOOK (#74). A setter can refuse or clamp (rt_shape past the measured
    // accumulator cliff, a lane count off the ladder) and the registry swallows that, so the
    // file said one thing and the engine ran another with nothing but the tuner's own log to
    // tell. Every stored knob is re-read through its `current` hook; a derived knob that
    // differs is the same message from the other side (this is not the host it was tuned on).
    let missed = verify_applied(reg, c.knobs.iter().map(|(k, v)| (k.as_str(), *v)));
    for (name, requested, actual) in &missed {
        eprintln!(
            "[imparo] host config: knob '{name}' requested {requested} but the engine runs \
             {actual} (the setter refused or clamped it)"
        );
    }
    if missed.is_empty() {
        eprintln!(
            "[imparo] host config loaded from {} ({applied} knobs applied and read back)",
            c.path.display()
        );
    } else {
        eprintln!(
            "[imparo] host config loaded from {} ({applied} knobs applied, {} did NOT take -- \
             the stored file does not describe this run; re-run imparo-tune)",
            c.path.display(),
            missed.len()
        );
    }
    c.batch
}

/// The knobs whose value after `apply` is not the value asked for: (name, requested,
/// actual). Device-free: it only goes through the registry's `apply` / `current` hooks.
pub fn verify_applied<'a>(
    reg: &[imparo_backend::KnobDecl],
    knobs: impl Iterator<Item = (&'a str, u32)>,
) -> Vec<(String, u32, u32)> {
    let mut missed = Vec::new();
    for (k, v) in knobs {
        if let Some(d) = reg.iter().find(|d| d.name == k) {
            let actual = (d.current)();
            if actual != v {
                missed.push((k.to_string(), v, actual));
            }
        }
    }
    missed
}
