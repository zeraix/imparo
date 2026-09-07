//! `imparo_backend::Backend` implemented by delegation onto this crate's free
//! functions -- a pass-through, so routing a workflow through the trait is
//! bit-identical to calling the functions directly. The free functions stay public:
//! the tuner and dev binaries use them without the trait.

use imparo_backend::{
    Backend, BackendKnobs, BufId, Gemma4MegaLayer, KnobCategory, KnobDecl,
    Lfm2MegaLayer, Lfm2MegaMixer, NO_WEIGHT, WeightKindWire,
};

/// The Metal backend as a trait object. Zero-sized: all state lives in the
/// backend's own globals, exactly as before.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalBackend;

#[inline]
fn b(id: BufId) -> u32 {
    id as u32
}

impl Backend for MetalBackend {
    fn begin(&self) {
        crate::begin();
    }
    fn flush(&self) {
        crate::flush();
    }
    fn end(&self) -> Result<(), i32> {
        crate::end()
    }
    fn last_gpu_us(&self) -> f64 {
        crate::last_gpu_us()
    }
    fn spill_rate(&self, idx: u32, tgs: u32, tpg: u32, iters: u32) -> f64 {
        crate::spill_rate(idx, tgs, tpg, iters)
    }
    fn bw_read(&self, bytes: u64, reps: u32, tgs: u32, tpg: u32) -> f64 {
        crate::bw_read(bytes, reps, tgs, tpg)
    }
    fn scoremix_rate(
        &self,
        tgs: u32,
        sgs: u32,
        iters: u32,
        stride: u32,
        kspan: u32,
    ) -> f64 {
        crate::scoremix_rate(tgs, sgs, iters, stride, kspan)
    }
    fn commit_overhead(&self, n: u32) -> f64 {
        crate::commit_overhead(n)
    }
    fn sync_overhead(&self, n: u32) -> f64 {
        crate::sync_overhead(n)
    }
    fn kv_tag(&self) -> String {
        crate::kv_tag()
    }
    fn encode_cost(&self, n: u32) -> f64 {
        crate::encode_cost(n)
    }
    fn device_profile(&self) -> imparo_backend::DeviceProfile {
        imparo_backend::DeviceProfile {
            threadgroup_bytes: crate::threadgroup_bytes(),
            max_threads: crate::max_threads_per_threadgroup(),
            // Measured by the cache-knee probe, which does not yet run at init.
            // Measured by the tuner's discovery pass, not here: init must stay cheap.
            cache_knee_bytes: 0,
            dram_read_mbs: 0,
            max_accumulators: 0,
            fill_threadgroups: 0,
            attn_score_ceiling_gflops: 0,
            fill_threads_membound: 0,
            commit_overhead_ns: 0,
            encode_cost_ns: 0,
            layer_work_ns: 0,
            layer_work_prefill_ns: 0,
        }
    }
    fn flush_layers(&self, decode: bool) -> u32 {
        crate::flush_layers(decode)
    }

    fn alloc(&self, id: BufId, bytes: u64) -> Result<(), i32> {
        crate::alloc(b(id), bytes)
    }
    fn arena(&self, bytes: u64) -> Result<(), i32> {
        crate::arena(bytes)
    }
    fn place(&self, id: BufId, offset: u64, bytes: u64) -> Result<(), i32> {
        crate::place(b(id), offset, bytes)
    }
    fn page_round(&self, n: u64) -> u64 {
        crate::page_round(n)
    }
    fn fast_tier_budget(&self) -> Option<u64> {
        let b = crate::working_set_budget();
        (b > 0).then_some(b)
    }
    fn alloc_kv(&self, bytes: &[u64]) -> Result<(), i32> {
        crate::alloc_kv(bytes)
    }
    fn grow_kv(&self, bytes: &[u64]) -> Result<(), i32> {
        crate::grow_kv(bytes)
    }
    fn write(&self, id: BufId, off: u64, src: &[f32]) {
        crate::write(b(id), off, src);
    }
    fn write_u32(&self, id: BufId, off: u64, src: &[u32]) {
        crate::write_u32(b(id), off, src);
    }
    fn read(&self, id: BufId, off: u64, dst: &mut [f32]) {
        crate::read(b(id), off, dst);
    }
    fn read_kv_bytes(&self, layer: u32, is_v: bool, off: u64, dst: &mut [u8]) {
        crate::read_kv_bytes(layer, is_v, off, dst);
    }
    fn write_kv_bytes(&self, layer: u32, is_v: bool, off: u64, src: &[u8]) {
        crate::write_kv_bytes(layer, is_v, off, src);
    }
    fn kv_advise_free(&self, layer: u32, is_v: bool, off: u64, len: u64) {
        crate::kv_advise_free(layer, is_v, off, len);
    }
    fn kv_advise_reuse(&self, layer: u32, is_v: bool, off: u64, len: u64) {
        crate::kv_advise_reuse(layer, is_v, off, len);
    }
    fn set_kv_page_table(&self, layer: u32, entries: &[u32]) {
        crate::set_kv_page_table(layer, entries);
    }

    fn set_kv_region(&self, layer: u32, k_off: u64, v_off: u64) {
        crate::set_kv_region(layer, k_off, v_off);
    }

    fn matmat(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        n_in: u32,
        n_out: u32,
        src: BufId,
        dst: BufId,
        n_tok: u32,
    ) {
        crate::matmat(wkind, w_off, n_in, n_out, b(src), b(dst), n_tok);
    }
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
    ) {
        crate::matmat_from(wkind, w_off, n_in, n_out, b(src), b(dst), n_tok, src_row);
    }
    fn matmat_gated(
        &self,
        gate_kind: WeightKindWire,
        gate_off: u64,
        up_kind: WeightKindWire,
        up_off: u64,
        n_in: u32,
        n_out: u32,
        src: BufId,
        dst: BufId,
        _tmp: BufId,
        n_tok: u32,
    ) -> bool {
        crate::matmat_gated(
            gate_kind,
            gate_off,
            up_kind,
            up_off,
            n_in,
            n_out,
            b(src),
            b(dst),
            n_tok,
        )
    }
    fn end_async(&self) -> Result<(), i32> {
        crate::end_async()
    }
    fn wait_outstanding(&self) -> Result<(), i32> {
        crate::wait_outstanding()
    }
    fn mega_reserve(
        &self,
        n_mid: u32,
        attn_heads: u32,
        attn_hd: u32,
    ) -> Result<(), i32> {
        crate::mega_reserve(n_mid, attn_heads, attn_hd)
    }
    fn mega_recover(&self) -> Result<(), i32> {
        crate::mega_recover()
    }
    fn decode_pipelining(&self) -> bool {
        crate::decode_pipelining()
    }
    fn stages_rows(&self) -> bool {
        crate::stages_rows()
    }
    fn argmax_feed(
        &self,
        src: BufId,
        tokens: BufId,
        pick: BufId,
        pick_slot: u32,
        n: u32,
    ) {
        crate::argmax_feed(b(src), b(tokens), b(pick), pick_slot, n);
    }
    fn set_epilogue(&self, epi: imparo_backend::Epilogue) {
        crate::set_epilogue(epi as u32);
    }
    fn supports_epilogue(&self, _epi: imparo_backend::Epilogue) -> bool {
        true
    }
    fn row(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        width: u32,
        index: u32,
        scale: f32,
        dst: BufId,
        dst_off: u32,
    ) {
        crate::row(wkind, w_off, width, index, scale, b(dst), dst_off);
    }
    fn gather_rows(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        width: u32,
        table_rows: u32,
        scale: f32,
        dst: BufId,
        dst_off: u32,
        idx: BufId,
        n_rows: u32,
    ) -> bool {
        crate::gather_rows(
            wkind,
            w_off,
            width,
            table_rows,
            scale,
            b(dst),
            dst_off,
            b(idx),
            n_rows,
        )
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
    ) {
        crate::rms_norm(b(buf), w_off, width, eps, n_row, row_stride, base_off);
    }
    /// The sandwich boundary at ONE ROW as one dispatch (decode): norm + residual add, and
    /// for the attention side the following ffn_norm too. Prefill rows keep the separate
    /// dispatches (the fused kernel's per-row loop lost 0.8-1.0 % there on 2026-09-02).
    /// IMPARO_NORM_ROW=0 keeps the separate dispatches at decode too (the A/B arm).
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_add(
        &self,
        dst: BufId,
        src: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
        add: BufId,
        output_scale: f32,
    ) -> bool {
        let scaled = output_scale.to_bits() != 1.0_f32.to_bits();
        if !norm_row_fusion_on()
            || !norm_row_mask(if scaled { 2 } else { 1 })
            || n_row != 1
            || row_stride != width
            || base_off != 0
            || width % 4 != 0
            || w_off == imparo_backend::NO_WEIGHT
        {
            return false;
        }
        crate::rms_norm_add_row(
            b(dst),
            b(src),
            b(add),
            w_off,
            width,
            eps,
            n_row,
            output_scale,
            false,
            0,
            b(dst),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_add_dual_projection(
        &self,
        src: BufId,
        residual: BufId,
        first_w_off: u64,
        mid: BufId,
        second_w_off: u64,
        out: BufId,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
        output_scale: f32,
    ) -> bool {
        if !norm_row_fusion_on()
            || !norm_row_mask(0)
            || n_row != 1
            || row_stride != width
            || base_off != 0
            || width % 4 != 0
            || first_w_off == imparo_backend::NO_WEIGHT
            || second_w_off == imparo_backend::NO_WEIGHT
            || out == mid
            || out == residual
        {
            return false;
        }
        crate::rms_norm_add_row(
            b(mid),
            b(src),
            b(residual),
            first_w_off,
            width,
            eps,
            n_row,
            output_scale,
            true,
            second_w_off,
            b(out),
        )
    }
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
    ) {
        crate::rms_norm_from(
            b(buf),
            b(src),
            w_off,
            width,
            eps,
            n_row,
            row_stride,
            base_off,
        );
    }
    #[allow(clippy::too_many_arguments)]
    fn add_rms_norm(
        &self,
        dst: BufId,
        resid: BufId,
        other: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    ) -> bool {
        // Contiguous full-width rows only: that is the layout the kernel's pre-add mode
        // and the mirror rule were written for, and the only one LFM2 uses.
        if base_off != 0 || row_stride != width {
            return false;
        }
        crate::add_rms_norm(
            b(dst),
            b(resid),
            b(other),
            w_off,
            width,
            eps,
            n_row,
            row_stride,
            base_off,
        );
        true
    }
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
    ) {
        crate::rope(
            b(buf),
            n_rot,
            base,
            head_dim,
            n_heads,
            start_pos,
            n_tok,
            freqs,
        );
    }
    fn hadamard(&self, buf: BufId, n: u32, nrot: u32) {
        crate::hadamard(b(buf), n, nrot);
    }
    /// gemma4's per-head norm + rope as one dispatch; a Hadamard rotation (quantized KV)
    /// keeps the default three-dispatch sequence, which this reproduces verbatim.
    #[allow(clippy::too_many_arguments)]
    fn head_norm_rope_hadamard(
        &self,
        buf: BufId,
        w_off: u64,
        head_dim: u32,
        eps: f32,
        n_heads: u32,
        start_pos: u32,
        n_tok: u32,
        rope_dim: u32,
        rope_base: f32,
        freqs: Option<&[f32]>,
        hadamard_nrot: u32,
    ) {
        if hadamard_nrot != 0 {
            self.rms_norm(buf, w_off, head_dim, eps, n_tok * n_heads, head_dim, 0);
            self.rope(
                buf, rope_dim, rope_base, head_dim, n_heads, start_pos, n_tok, freqs,
            );
            self.hadamard(buf, n_tok * n_heads * head_dim, hadamard_nrot);
            return;
        }
        crate::head_norm_rope(
            b(buf),
            w_off,
            head_dim,
            eps,
            n_heads,
            start_pos,
            n_tok,
            rope_dim,
            rope_base,
            freqs,
        );
    }
    /// K: norm + rope fused; V: its weightless norm as before. K and V are different
    /// buffers, so the order K-then-V (the default runs K norm, V norm, K rope) changes
    /// no value. Any Hadamard rotation keeps the default sequence verbatim.
    #[allow(clippy::too_many_arguments)]
    fn kv_head_postprocess(
        &self,
        k: BufId,
        v: BufId,
        k_norm_off: u64,
        head_dim: u32,
        eps: f32,
        n_kv: u32,
        start_pos: u32,
        n_tok: u32,
        rope_dim: u32,
        rope_base: f32,
        freqs: Option<&[f32]>,
        k_hadamard_nrot: u32,
        v_hadamard_nrot: u32,
    ) {
        if k_hadamard_nrot != 0 || v_hadamard_nrot != 0 {
            self.rms_norm(k, k_norm_off, head_dim, eps, n_tok * n_kv, head_dim, 0);
            self.rms_norm(
                v,
                imparo_backend::NO_WEIGHT,
                head_dim,
                eps,
                n_tok * n_kv,
                head_dim,
                0,
            );
            self.rope(
                k, rope_dim, rope_base, head_dim, n_kv, start_pos, n_tok, freqs,
            );
            if k_hadamard_nrot != 0 {
                self.hadamard(k, n_tok * n_kv * head_dim, k_hadamard_nrot);
            }
            if v_hadamard_nrot != 0 {
                self.hadamard(v, n_tok * n_kv * head_dim, v_hadamard_nrot);
            }
            return;
        }
        crate::head_norm_rope(
            b(k),
            k_norm_off,
            head_dim,
            eps,
            n_kv,
            start_pos,
            n_tok,
            rope_dim,
            rope_base,
            freqs,
        );
        self.rms_norm(
            v,
            imparo_backend::NO_WEIGHT,
            head_dim,
            eps,
            n_tok * n_kv,
            head_dim,
            0,
        );
    }
    fn kv_store(
        &self,
        src: BufId,
        layer: u32,
        width: u32,
        start_pos: u32,
        n_tok: u32,
        is_v: bool,
        ring: u32,
    ) {
        crate::kv_store(b(src), layer, width, start_pos, n_tok, is_v, ring);
    }
    fn attention(
        &self,
        kv_layer: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv: u32,
        kv_width: u32,
        start_pos: u32,
        scale: f32,
        window: u32,
        n_tok: u32,
        max_scores: u32,
        ring: u32,
    ) {
        // The op applies its scale: Q is scaled in place before the kernel reads it. The
        // same dispatch the workflow used to issue, now where the contract lives.
        if scale.to_bits() != 1.0_f32.to_bits() {
            self.scale(BufId::Q, scale, n_tok * n_heads * head_dim);
        }
        crate::attention(
            kv_layer, head_dim, n_heads, n_kv, kv_width, start_pos, window, n_tok,
            max_scores, ring,
        );
    }
    #[allow(clippy::too_many_arguments)]
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
    ) {
        crate::shortconv(
            b(bcx),
            w_off,
            b(state),
            state_off,
            b(out),
            width,
            kernel,
            n_tok,
        );
    }
    #[allow(clippy::too_many_arguments)]
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
    ) {
        crate::shortconv_snapshot(
            b(bcx),
            b(state),
            state_off,
            b(snap),
            snap_off,
            width,
            kernel,
            n_tok,
        );
    }
    /// One decode layer as one entry of the mega kernel: the architecture's variant names the
    /// operands, the entry function below turns them into the program's slots and words.
    fn mega_layer(&self, e: &imparo_backend::MegaEntry<'_>) -> bool {
        match &e.layer {
            imparo_backend::MegaLayer::Gemma4(l) => mega_gemma4_entry(l, e.n_tok),
            imparo_backend::MegaLayer::Lfm2(l) => mega_lfm2_entry(l, e.n_tok),
        }
    }
    fn mega_qkv_wanted(&self) -> bool {
        crate::mega_qkv_wanted()
    }
    fn mega_program_end(&self) {
        crate::mega_program_end()
    }
    fn mega_front_wanted(&self) -> bool {
        crate::mega_front_wanted()
    }
    fn mega_attn_wanted(&self) -> bool {
        crate::mega_attn_wanted()
    }
    /// The whole gated FFN as one persistent dispatch: the mega FFN block (IMPARO_MEGA_FFN=1, #141).
    /// Decode only, Q4_0 only. `BufId::U` is the second scratch row -- the row the
    /// dispatch path this replaces writes its up projection into.
    #[allow(clippy::too_many_arguments)]
    fn ffn_gated_down(
        &self,
        gate_kind: WeightKindWire,
        gate_off: u64,
        up_kind: WeightKindWire,
        up_off: u64,
        down_kind: WeightKindWire,
        down_off: u64,
        n_in: u32,
        n_mid: u32,
        n_out: u32,
        src: BufId,
        gated_tmp: BufId,
        dst: BufId,
        n_tok: u32,
    ) -> bool {
        if n_tok != 1 || gate_kind != 1 || up_kind != 1 || down_kind != 1 {
            return false;
        }
        crate::ffn_persistent(
            gate_off,
            up_off,
            down_off,
            n_in,
            n_mid,
            n_out,
            b(src),
            b(gated_tmp),
            b(BufId::U),
            b(dst),
        )
    }
    fn act(&self, a: BufId, n: u32) {
        crate::act(b(a), n);
    }
    fn act_mul(&self, a: BufId, bb: BufId, n: u32) {
        crate::act_mul(b(a), b(bb), n);
    }
    fn set_activation(&self, act: imparo_backend::Epilogue) {
        crate::set_epilogue_act(act as u32);
    }
    fn add(&self, a: BufId, bb: BufId, n: u32) {
        crate::add(b(a), b(bb), n);
    }
    fn add_scale(&self, a: BufId, bb: BufId, k: f32, n: u32) {
        crate::add_scale(b(a), b(bb), k, n);
    }
    fn scale(&self, a: BufId, k: f32, n: u32) {
        crate::scale(b(a), k, n);
    }
    fn copy(&self, dst: BufId, src: BufId, n: u32) {
        crate::copy(b(dst), b(src), n);
    }
    fn copy_range(&self, dst: BufId, dst_off: u32, src: BufId, src_off: u32, n: u32) {
        crate::copy_range(b(dst), dst_off, b(src), src_off, n);
    }
    fn mul_strided(
        &self,
        a: BufId,
        bb: BufId,
        n: u32,
        b_off: u32,
        b_stride: u32,
        a_stride: u32,
        n_tok: u32,
    ) {
        crate::mul_strided(b(a), b(bb), n, b_off, b_stride, a_stride, n_tok);
    }
    fn softcap(&self, a: BufId, cap: f32, n: u32) {
        crate::softcap(b(a), cap, n);
    }
    fn argmax(&self, src: BufId, dst: BufId, n: u32) {
        crate::argmax(b(src), b(dst), n);
    }
    fn ple_gather_combine(
        &self,
        proj: BufId,
        tokens_buf: BufId,
        w_offset: u64,
        width: u32,
        emb_scale: f32,
        comb_scale: f32,
        n_tok: u32,
    ) {
        crate::ple_gather_combine(
            b(proj),
            b(tokens_buf),
            w_offset,
            width,
            emb_scale,
            comb_scale,
            n_tok,
        );
    }
    fn transform_weights(
        &self,
        jobs: &[imparo_backend::WeightTransform],
    ) -> Result<Vec<bool>, String> {
        let wire: Vec<crate::WXformWire> = jobs
            .iter()
            .map(|j| crate::WXformWire {
                off: j.offset,
                bytes: j.bytes,
                from_type: j.from_type,
                to_type: j.to_type,
                n_in: j.n_in,
                n_out: j.n_out,
            })
            .collect();
        crate::transform_weights(&wire)
    }
    fn serves_weight_type(&self, ggml_type: u32) -> bool {
        // F32, Q4_0, Q8_0 and the tile-major Q8_0 (fc 15 pipelines). Q4_0_TM has no kernels.
        matches!(ggml_type, 0 | 2 | 8 | 1000)
    }
    fn read_weight_bytes(&self, file_off: u64, dst: &mut [u8]) -> bool {
        crate::read_weight_bytes(file_off, dst)
    }
    fn stage_rows(
        &self,
        file_off: u64,
        row_bytes: u32,
        ids: &[u32],
        dst: BufId,
    ) -> bool {
        crate::stage_rows(file_off, row_bytes, ids, b(dst))
    }
    fn ple_gather_combine_staged(
        &self,
        proj: BufId,
        rows: BufId,
        width: u32,
        emb_scale: f32,
        comb_scale: f32,
        n_tok: u32,
    ) {
        crate::ple_gather_combine_staged(
            b(proj),
            b(rows),
            width,
            emb_scale,
            comb_scale,
            n_tok,
        );
    }
    fn set_attention_head_dims(&self, dims: &[u32]) {
        crate::set_attention_head_dims(dims);
    }
    fn set_attention_kv_widths(&self, widths: &[u32]) {
        crate::set_attention_kv_widths(widths);
    }
    unsafe fn init_weights(&self, base: *const u8, len: u64) -> Result<(), i32> {
        // Metal shares the CPU mmap with the GPU (unified memory, no copy).
        unsafe { crate::init_tuned(base, len) }
    }
    unsafe fn init_weights_with_placement(
        &self,
        base: *const u8,
        len: u64,
        placement: &imparo_backend::WeightPlacement,
    ) -> Result<(), i32> {
        crate::set_placement(&placement.segments, placement.budget.total.unwrap_or(0));
        unsafe { crate::init_tuned(base, len) }
    }
    fn set_kv_types(&self, k: u32, v: u32) {
        crate::set_kv_types(k, v);
    }
    fn prof_stats(&self) -> imparo_backend::ProfStats {
        // Barriers first: prof_read() resets the whole window, this counter included.
        let barriers = crate::prof_barriers();
        let p = crate::prof_read();
        imparo_backend::ProfStats {
            gpu_s: p.gpu_s,
            wall_s: p.wall_s,
            cbs: p.cbs,
            dispatches: p.dispatches,
            barriers,
            categories: crate::prof_categories(),
        }
    }
    fn prof_enable(&self, on: bool) {
        crate::prof_enable(on);
    }
    fn allocated_bytes(&self) -> u64 {
        crate::allocated_bytes()
    }
    fn device_tag(&self) -> String {
        "metal".to_string()
    }
    fn pool_caps(&self) -> imparo_backend::PoolCaps {
        use imparo_backend::Tier;
        imparo_backend::PoolCaps {
            // READ BACK from the engine, never typed again here: the shader is
            // compiled with this value as a preprocessor macro, so the page the block
            // table maps and the page this declares are one number by construction.
            page_cells: crate::kv_page_cells(),
            // The CEILING over every prefill attention kernel this backend can select,
            // not the value of the current selection -- which varies per layer, per
            // request length, per device and per KV type.
            //
            // 16 is `qb0 = q0 & ~15u`, the query group both prefill families take their
            // KV scan bounds from; the 8-run split inside a position block contributes
            // 8, which 16 already covers. MEASURED: the grid moved to 8 only when the
            // group was forced to 8, and did not move for the GEMM tile (32/64/128),
            // tail-split, nb8 or half-A staging. docs/kv-identity-grid.md.
            //
            // Raise this if a kernel with a wider query group is added, or the pool
            // will cut prefills where that kernel does not reproduce a cold pass.
            finest_cut_tokens: 16,
            // every full-attention KV access goes through the 64-cell block
            // table (kv paging 24a/24b; scattered placement gate-proven)
            paged_reads: true,
            shared_address: true,
            tiers: &[Tier::Unified, Tier::Disk],
        }
    }
}

/// The Metal knob registry: THE definition of the metal search space. Every Benched
/// knob is declared here once -- stage, sweep, workload, hooks -- and the shared tuner,
/// Powers of two from `lo` up to the DEVICE's threadgroup thread limit.
///
/// A thread-count ladder is a consequence of the hardware, not a choice. Written out, it
/// was right on the machine it was written on: `1024` is this Mac's limit and an ILLEGAL
/// dispatch on one whose limit is 512, and a machine with a wider limit never gets the
/// rungs it allows. `max_threads` comes from the queried device profile.
fn threads_upto(lo: u32, max_threads: u32) -> Vec<u32> {
    let cap = max_threads.max(lo);
    core::iter::successors(Some(lo), |t| t.checked_mul(2))
        .take_while(|t| *t <= cap)
        .collect()
}

/// The same ladder in SIMDGROUPS: a simdgroup is 32 threads, so the device's thread limit
/// caps how many of them one threadgroup may hold.
fn simdgroups_upto(lo: u32, max_threads: u32) -> Vec<u32> {
    threads_upto(lo, max_threads / 32)
}

/// the stored-file reader (apply_host_config) and the stage-2 descent all iterate this
/// table. Adding a knob is ONE entry here plus its engine global, then a space bump.
/// REGISTRY ORDER IS SWEEP ORDER (nb8_shape before the crossings that fight its winner).
use imparo_backend::{SweepKind as Sw, Workload as Wl};

// THE KERNEL OPS THIS REGISTRY GUARDS, and which decision belongs to which.
//
// Read this to learn what the Metal side does at a high level; each knob below then says
// what it changes about one op and what was measured. The registry is the honest place
// for it, because a knob that drifts away from the op it guards stops working, and this
// codebase has caught that five times.
//
// THE MATMUL FAMILY -- one logical op, four kernels, chosen by batch size:
//
//   n_tok == 1            the DECODE GEMV. One token against the whole weight matrix, so
//     ..gemv_max_tok      it is pure weight traffic and its shape is about how lanes
//                         divide a row and how many rows a lane group carries.
//                         guarded by: lanes x nr0 (one pipeline table indexed by both),
//                         and gemv_max_tok, which is where this path ends.
//
//   2..nb8_max            the NARROW tile. A handful of tokens is too many for the GEMV
//                         and too few to fill the wide tile's 64-token width.
//                         guarded by: nb8_shape (which of two tiles), nb8_max (where it
//                         gives way to the wide tile).
//
//   above that            the REGISTER-TILED PREFILL GEMM. A tile of output held in
//                         registers while the k dimension streams past; this is 78.6% of
//                         a 16k prefill and it is multiply-bound, not staging-bound.
//                         guarded by: rt_shape (rows x tokens x threads x accumulators),
//                         sgs (simdgroups per threadgroup).
//
// THE ATTENTION FAMILY -- split first by prefill vs decode, then by depth:
//
//   PREFILL, qcomb        query tiles against the whole prefix. A threadgroup takes QT
//                         query rows and walks the key/value scan in PT-wide position
//                         tiles, scoring into threadgroup memory, softmaxing, then
//                         multiplying by V. Its cost is the scan, so it grows with
//                         context and dominates deep prefills.
//                         guarded by: pt_512x / pt_256x (position tile width, DERIVED
//                         from the device's threadgroup budget), qcomb_blk (position
//                         blocks per work unit on the QT-8 variants that q4/q8 use),
//                         attn_threads_prefill (threads, which fixes the simdgroup
//                         count the accumulator is split across).
//
//   DECODE, score-tile    one query against the prefix, scores held in threadgroup
//                         memory. Fine while the prefix is short.
//                         guarded by: attn_threads, attn_min_tgs (below this many
//                         threadgroups the dispatch cannot fill the GPU, so it is sliced).
//
//   DECODE, streaming     the same query against a prefix too long for a score tile:
//                         the scan is cut into slices, each producing a partial
//                         (accumulator, max, sum), combined afterwards.
//                         guarded by: attn_stream_hq (query heads sharing a threadgroup,
//                         so one K/V row serves both), attn_stream_slices (how many
//                         slices), attn_stream_min_pos (the depth at which this path
//                         takes over from the score tile).
//
// THE SUBMISSION FAMILY -- not a kernel at all, and that is why it was missed. Every
// dispatch above reaches the GPU inside a command buffer, and how many layers share one
// is its own trade:
//
//   command buffer        commit too often and the fixed submission cost is paid on
//                         every batch; commit too rarely and the GPU sits idle while
//                         the host is still encoding.
//                         guarded by: flush_layers, flush_layers_prefill.
//
// The three boundaries -- gemv_max_tok, nb8_max, attn_stream_min_pos -- are what
// partition a request into regimes. Everything else is a shape within one regime.

/// The same arithmetic the backend runs at init, so the registry and the kernel cannot
/// disagree about what fits. Inverts the threadgroup layout: subtract the fixed regions,
/// divide what is left by the query-tile height, round down to a whole work unit.
const fn derive_pt(
    qt: u64,
    hd: u64,
    half_q: bool,
    device_spill: bool,
    blk: u64,
    budget_bytes: u64,
) -> u32 {
    let floats = budget_bytes / 4;
    let sq = if half_q && qt == 16 {
        qt * hd / 2
    } else {
        qt * hd
    };
    let spill = if device_spill { 0 } else { qt * hd };
    let fixed = sq + 2 * qt + (qt / 8) * 64 + spill;
    if fixed >= floats {
        return 0;
    }
    let unit = 8 * blk;
    (((floats - fixed) / qt) / unit * unit) as u32
}

/// LAYERS PER COMMAND BUFFER, from the two costs a flush trades between.
///
/// The engine encodes layers and commits every `flush_layers` of them. `flush` COMMITS AND
/// RETURNS -- it does not wait -- so the two terms are:
///
///   pay per buffer     one submission, `commit_ns`
///   pay per flush size the GPU cannot start until N layers are encoded: N x `encode_ns`
///                      x dispatches per layer
///
/// Over a model of L layers, N per buffer:
///
///   total = (L / N) x commit  +  N x encode_per_layer
///
/// which is a sum of a 1/N term and an N term, so it is smallest at
///
///   N = sqrt(L x commit / encode_per_layer)
///
/// MEASURED HERE (first form): commit ~3.7 us, encode ~0.128 us per dispatch, 21
/// dispatches per layer, 42 layers -> N = sqrt(42 x 3.7 / 2.69) = 7.6, deriving to 7 --
/// the compiled value. BOTH INPUTS WERE THE WRONG QUANTITY (docs/decode-turnaround.md,
/// 2026-09-04): the commit was the host's submission time, which is never on the
/// critical path at decode (the host encodes a token in ~0.4 ms against ~22 ms of GPU
/// work), and the encode was a trivial kernel's, 0.1 us, where the engine's real
/// dispatch binds three buffers and eight constants and costs ~1.1 us. Measured on a
/// live token the seat of 7 exposed 174 us of first-block encode per token; a flush of 3
/// exposed 32. The inputs are now what the step pays: the GPU-side gap between
/// consecutive buffers (`commit_overhead`) and a real GEMV's encode (`encode_cost`), and
/// the same formula derives the small value the measurement asks for.
///
/// AGREEING WITH THE OLD CONSTANT IS NOT THE POINT, and a derivation that only ever
/// reproduces constants would be worth nothing. What agreement buys is one less thing to
/// check. A DISAGREEMENT is the useful case: it says the code, the arithmetic, or the
/// constant is wrong, and which one has to be settled by reading, not by preferring the
/// newer number. Both outcomes happened in this file:
///
///   flush_layers          derived 7, compiled 7    agreed; nothing to check
///   flush_layers_prefill  derived 7, compiled 0    disagreed -> read the code -> the
///                                                  FORMULA was wrong (no work term), so
///                                                  the derivation was withdrawn
///   attn_min_tgs          benched 48, compiled 72  disagreed -> read the dispatch, then
///                                                  bracketed end to end -> the CONSTANT
///                                                  was wrong, 48 stands
///
/// The inputs drift with host state and the derived value should not: each is a median
/// of five readings (`submission_costs`), and the ratio is what the square root sees.
///
/// What the knob is worth: the exposed first-block encode, N x encode_per_layer, against
/// (L / N) GPU gaps of a few microseconds. On this machine that is ~150 us per token
/// between 7 and 3, under 1% of a 22 ms token -- real, and only visible on the probe.
///
/// A guard the arithmetic does not carry: 0 means "never flush mid-graph", which is a
/// legal setting and not what this returns. The derivation is only asked for a positive
/// answer, so it clamps to [1, L].
const fn derive_flush(
    commit_ns: u32,
    encode_ns: u32,
    dispatches: u32,
    layers: u32,
) -> u32 {
    if commit_ns == 0 || encode_ns == 0 || dispatches == 0 || layers == 0 {
        return 0; // unprobed: the compiled value stands
    }
    let encode_per_layer = (encode_ns as u64) * (dispatches as u64);
    let sq = (layers as u64) * (commit_ns as u64) / encode_per_layer;
    // Integer sqrt, const-friendly: Newton from a power-of-two seed.
    //
    // `midpoint` rather than `(a + b) / 2`: the sum can exceed u64 for a large seed, and
    // an overflow here would silently pick a flush interval from a wrapped value.
    let mut x = sq;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = x.midpoint(sq / x);
    }
    if x < 1 {
        1
    } else if x > layers as u64 {
        layers
    } else {
        x as u32
    }
}

/// TWO KINDS OF DERIVED, and they want opposite things from the stored config.
///
/// The design says "a derived value is REPORTED in the config and never APPLIED from it",
/// and that is right for a value the HOST can recompute -- pt_512x comes from the device's
/// QUERIED threadgroup limit, so every host has the input and restoring another machine's
/// answer is exactly the bug deriving it prevents. Those declare `apply: |_| {}`.
///
/// But flush_layers is derived from MEASURED inputs -- submission cost and encode cost,
/// which only the tuner probes. A host that never runs the tuner cannot recompute it, so
/// its stored value MUST be applied, and it keeps a real setter.
///
/// Nothing enforces the split: `apply_host_config` applies every stored key through its
/// declared hook, so the rule lives in whether each declaration hands over a real setter or
/// a no-op. That is a convention, and conventions drift -- worth making a mechanism the
/// next time a derived knob is added, keyed on where its inputs come from rather than on
/// the author remembering.
static METAL_KNOBS: &[KnobDecl] = &[
    KnobDecl {
        // COMPUTED, never swept. The widest position tile that fits this device's queried
        // threadgroup limit, rounded down to a whole number of work units. It was a
        // literal -- 240 -- computed once by hand against a 32768 byte budget and frozen,
        // which is correct on the machine it was computed on and silently wrong anywhere
        // else. The derivation reproduces it exactly here, which is the point.
        //
        // `apply` is a no-op: PT reaches the kernel as a preprocessor define at library
        // compile, so it cannot change after init. The value is REPORTED so a config says
        // what this host computed; another host recomputes its own.
        name: "pt_512x",
        category: KnobCategory::Arithmetic,
        values: &[],
        apply: |_| {},
        current: crate::pt_512x,
        screened: false,
        legal: None,
        bit_affecting: false,
        derive: Some(|_m, d| derive_pt(16, 512, true, true, 2, d.threadgroup_bytes)),
        candidates: None,
        after: &[],
        cross_check: None,
        applies: Some(|m| m.deep_head_dim == 512),
        tuple: None,
        sweep: Sw::Derived,
        workload: Wl::AttentionPrefillDeep,
    },
    KnobDecl {
        name: "pt_256x",
        category: KnobCategory::Arithmetic,
        values: &[],
        apply: |_| {},
        current: crate::pt_256x,
        screened: false,
        legal: None,
        bit_affecting: false,
        derive: Some(|_m, d| derive_pt(16, 256, true, false, 2, d.threadgroup_bytes)),
        candidates: None,
        after: &[],
        cross_check: None,
        applies: None,
        tuple: None,
        sweep: Sw::Derived,
        workload: Wl::AttentionPrefillDeep,
    },
    KnobDecl {
        // SIMDGROUPS PER THREADGROUP for the matmul dispatch -- how many 32-lane groups
        // share one threadgroup, and so how the output tile is divided among them. It
        // applies to both the decode GEMV and the prefill GEMM, which is why it is swept
        // on the decode mix where the cost is clearest.
        name: "sgs",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: Some(|_m, d| simdgroups_upto(2, d.max_threads)),
        after: &[],
        cross_check: None,
        // Q4-only kernel: on a Q8 model the route never dispatches it, and the sweep
        // used to spend DecodeMix reps on it and record a pick (review #116, D11).
        applies: Some(|m| m.weight_kinds & (1 << 1) != 0),
        // JOINS the gemv tuple. sgs, lanes and nr0 all appear in ONE expression
        // governing the decode GEMV's threadgroup geometry --
        //   rows_per_tg = sgs * (32 / lanes) * nr0
        //   threadsPerThreadgroup = 32 * sgs
        // -- so the rows a threadgroup covers is their product and none of them has a
        // best value alone. It was swept by itself, which measures one factor of a
        // product while holding the others at whatever they happened to be.
        tuple: Some("gemv"),
        category: KnobCategory::Benched,
        values: &[],
        apply: |v| crate::tune(v, 0),
        current: crate::sgs_current,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "lanes",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: Some(|_m, _d| {
            // COUNTED FROM THE PIPELINE TABLE, not written beside it. The build loop and
            // this ladder were the same four numbers in two places; extend the loop and
            // the new rung would never be swept, with nothing to say so.
            let (lo, hi) = crate::lanes_built();
            core::iter::successors(Some(lo), |t| t.checked_mul(2))
                .take_while(|t| *t <= hi.max(lo))
                .collect()
        }),
        after: &[],
        cross_check: None,
        // Q4-only kernel: on a Q8 model the route never dispatches it, and the sweep
        // used to spend DecodeMix reps on it and record a pick (review #116, D11).
        applies: Some(|m| m.weight_kinds & (1 << 1) != 0),
        tuple: Some("gemv"),
        // lanes and nr0 index ONE pipeline table together:
        // p_q4mm_lanes[g_lanes_log2][nr_log2]. They select a single kernel jointly,
        // so neither has a best value on its own.
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_lanes,
        current: crate::lanes_current,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "nb8_shape",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: None,
        // Q4-only kernel: on a Q8 model the route never dispatches it, and the sweep
        // used to spend DecodeMix reps on it and record a pick (review #116, D11).
        applies: Some(|m| m.weight_kinds & (1 << 1) != 0),
        tuple: Some("narrow"),
        // nb8_shape picks WHICH narrow tile; nb8_max picks the batch size at which
        // it engages. The right boundary depends on which tile sits behind it, so the
        // boundary search runs AFTER the shape is chosen, not before.
        category: KnobCategory::Benched,
        values: &[0, 1],
        apply: crate::set_nb8_shape,
        current: crate::nb8_shape_current,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::NarrowMix(8),
    },
    // Boundary crossings, in dependency order: each runs against the nb8 pick above.
    KnobDecl {
        // WHERE THE NARROW TILE GIVES WAY to the wide one. Same boundary shape as
        // gemv_max_tok, one level up: a handful of tokens is too many for the GEMV and
        // too few to fill a 64-token tile, and this is where "too few" ends.
        name: "nb8_max",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: None,
        // Q4-only kernel: on a Q8 model the route never dispatches it, and the sweep
        // used to spend DecodeMix reps on it and record a pick (review #116, D11).
        applies: Some(|m| m.weight_kinds & (1 << 1) != 0),
        tuple: Some("narrow"),
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_nb8_max,
        current: crate::nb8_max_current,
        screened: false,
        sweep: Sw::Crossing {
            ladder: &[4, 8, 12, 16, 24, 32, 48],
            hi: 64,
            lo: 0,
        },
        workload: Wl::NarrowMix(8),
    },
    KnobDecl {
        // WHERE THE DECODE GEMV ENDS. Batches up to this take the one-token kernel,
        // above it the GEMM family. A boundary, so it defines a regime rather than
        // choosing a shape within one: it is searched over a ladder of batch sizes,
        // looking for the size at which the wider kernel starts winning and keeps
        // winning.
        name: "gemv_max_tok",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: None,
        // Q4-only kernel: on a Q8 model the route never dispatches it, and the sweep
        // used to spend DecodeMix reps on it and record a pick (review #116, D11).
        applies: Some(|m| m.weight_kinds & (1 << 1) != 0),
        // JOINS the gemv tuple, as its BOUNDARY. Where the GEMV gives way to the GEMM
        // depends on how good the GEMV is, and that is what the tuple's other three
        // knobs decide -- so the crossover has to be searched AFTER them, against the
        // shape that won. Tuple members that are boundary searches always run last,
        // which is the same relationship attn_stream_min_pos has to the decode_stream
        // shape knobs.
        //
        // This is what "dependency order" turns out to be in practice: not a separate
        // ordering mechanism, but a boundary inside the tuple whose regimes it defines.
        tuple: Some("gemv"),
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_gemv_max_tok,
        current: crate::gemv_max_tok_current,
        screened: false,
        sweep: Sw::Crossing {
            ladder: &[2, 3, 4, 6, 8, 12, 16],
            hi: 64,
            lo: 1,
        },
        workload: Wl::NarrowMix(8),
    },
    // ---- THE Q8_0 WEIGHT FAMILY -----------------------------------------------------
    //
    // Ported from the LFM2.5 branch, where fourteen knobs governed these kernels and NINE
    // of them were judged end to end. This registry has no end-to-end stage, so each one
    // was either re-expressed against a micro workload or dropped, and the dropped ones
    // are listed with their reason at the end of the family. Every knob below is answered
    // by one kernel at one shape.
    //
    // `applies` gates the whole family on the model actually having Q8_0 projections.
    // Sweeping them on a Q4_0 model would rank candidates on a kernel the workload never
    // dispatches -- the same defect as declaring a prefill tile knob on DecodeMix.
    KnobDecl {
        name: "q8_decode_sgs",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: Some(|_m, d| simdgroups_upto(1, d.max_threads)),
        after: &[],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & ((1 << 2) | (1 << 3)) != 0),
        // The decode pair is a TUPLE: rows says how many outputs a threadgroup carries
        // and sgs how many simdgroups share them, and the cross-simdgroup reduction at
        // the end costs rows*sgs floats of threadgroup memory. Neither value means
        // anything without the other.
        tuple: Some("q8_decode"),
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_q8_decode_sgs,
        current: crate::q8_decode_sgs,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        // The TILE-MAJOR decode GEMV's simdgroups. The 256-byte unit fixes 8 rows per
        // threadgroup, so unlike the row-major pair this is a single knob, and it is
        // separate from q8_decode_sgs because the two layouts measured different winners
        // (8 vs 4: the converted file's deep-leg decode read +1.8/+2.5% at 8, the
        // isolated row-major GEMV prefers 4) -- one shared value would make the tune
        // file fight itself across layouts of the same model.
        name: "q8_tm_decode_sgs",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: Some(|_m, d| simdgroups_upto(1, d.max_threads)),
        after: &[],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & (1 << 3) != 0),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_q8_tm_decode_sgs,
        current: crate::q8_tm_decode_sgs,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "q8_decode_rows",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &["q8_decode_sgs"],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & ((1 << 2) | (1 << 3)) != 0),
        tuple: Some("q8_decode"),
        category: KnobCategory::Benched,
        // ROWS, not log2, so the stored config reads as the thing the kernel does. Only
        // 1, 2 and 4 have a compiled pipeline: Q8_DECODE_ROWS is a function constant and
        // the kernel holds its accumulators in a fixed four-element array.
        values: &[1, 2, 4],
        apply: crate::set_q8_decode_rows,
        current: crate::q8_decode_rows,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "q8_batch_sgs",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: Some(|_m, d| simdgroups_upto(1, d.max_threads)),
        after: &[],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & ((1 << 2) | (1 << 3)) != 0),
        tuple: Some("q8_narrow"),
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_q8_batch_sgs,
        current: crate::q8_batch_sgs,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::NarrowMix(4),
    },
    KnobDecl {
        name: "q8_token_tile",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &["q8_batch_sgs"],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & ((1 << 2) | (1 << 3)) != 0),
        tuple: Some("q8_narrow"),
        category: KnobCategory::Benched,
        // How many tokens reuse one dequantised weight byte. A function constant, so each
        // value is its own pipeline and the list is exactly what was compiled.
        values: &[1, 2, 4, 8],
        apply: crate::set_q8_token_tile,
        current: crate::q8_token_tile,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::NarrowMix(4),
    },
    KnobDecl {
        name: "st_gemm_shape",
        // Shape indices only; the bridge ignores anything past the table, which would
        // otherwise select a pipeline that was never built. COUNTED, not written: this
        // said `v < 10` and the table has twelve entries, so the two shapes added for the
        // deeper K chunk were illegal to store while being legal to sweep.
        legal: Some(|v, _m, _d| v < crate::st_gemm_shapes()),
        bit_affecting: false,
        derive: None,
        // COUNTED from ST_GEMM_CANDIDATES, which the .mm exports for exactly this. Same
        // reasoning as rt_shape: a written ladder goes stale the moment a shape is added,
        // and nothing reports it.
        candidates: Some(|_m, _d| (0..crate::st_gemm_shapes()).collect()),
        after: &[],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & ((1 << 2) | (1 << 3)) != 0),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_st_gemm_shape,
        current: crate::st_gemm_shape,
        // SCREENED, like rt_shape and for the same reason: the shapes span 128 to 512
        // threads and 32x16 to 128x32 output tiles, so the wide ones spill and read
        // orders of magnitude slow on the tiny-size screen. Shape 9 (128x32) is in the
        // list because the branch that introduced it measured it exact and SLOWER, and a
        // rejected candidate that stays reachable is evidence rather than folklore.
        //
        // PrefillGemmWidths, not PrefillGemm. What a tile trades is weight RE-READS
        // against padding, and PrefillGemm's slice made the weight tile cache-resident --
        // 2.23 MB against this device's measured 2.10 MB knee -- so the re-reads were
        // free in the harness and expensive in the engine, and the smallest tile won by
        // default. The ranking workload clears the knee 4x and sends two token widths.
        screened: true,
        sweep: Sw::Values,
        workload: Wl::PrefillGemmWidths,
    },
    KnobDecl {
        // THE SECOND TILE OF THE PREFILL PAIR. Not a threshold: the dispatch decides
        // between the two by comparing their padded token counts (see `q8_matmat`), which
        // is arithmetic and stores nothing. What is genuinely per-machine, and therefore
        // here, is WHICH second tile is worth having.
        //
        // Its workload is PrefillGemmUbatch: the whole-ubatch width at the model's full
        // projection. The dispatch selects this tile only where it pads no worse than the
        // first -- whole ubatches -- so a remainder width never runs it, and ranking it on
        // the two-width PrefillGemmWidths made half of every measurement dilution; the
        // sliced weight tile that workload used sat inside the last-level cache besides.
        // Measured on LFM2 (Q8): end to end by env pin 64x64 is worth +3.1% / +3.0% at
        // 5963 / 17123 tokens and -0.1% at 455 (one chunk, padded, the rule takes the
        // narrow tile). On PrefillGemmWidths it read -0.1% to +1.8%, inside the floor
        // every time, and the tuner wrote 3 over a stored 7; on PrefillGemmUbatch it
        // reads +4.0% on a 2.0% floor and the tuner picks 7 by itself.
        //
        // Equal to `st_gemm_shape` means one shape, which is the untuned path exactly. A
        // stored config from an earlier space has no line for this key and therefore
        // keeps that default, so an old file is not silently mis-tuned -- it is simply
        // the single-shape engine it has always been. That is why the space version does
        // not move for this knob: bumping it discards every OTHER knob's measured value
        // (task #85), and on this host that regressed E4B's deep prefill 535.0 -> 530.8
        // the last time it happened.
        name: "st_gemm_large_shape",
        legal: Some(|v, _m, _d| v < crate::st_gemm_shapes()),
        bit_affecting: false,
        derive: None,
        candidates: Some(|_m, _d| (0..crate::st_gemm_shapes()).collect()),
        after: &["st_gemm_shape"],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & ((1 << 2) | (1 << 3)) != 0),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_st_gemm_large_shape,
        current: crate::st_gemm_large_shape,
        screened: true,
        sweep: Sw::Values,
        workload: Wl::PrefillGemmChunk,
    },
    KnobDecl {
        name: "q8_full_tiles",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &["st_gemm_shape"],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & ((1 << 2) | (1 << 3)) != 0),
        tuple: None,
        category: KnobCategory::Benched,
        // May the host use the edge-predicate-free entry point when the grid is exact?
        // It is a separate compiled kernel with every masking arm deleted, so it should
        // be free when legal -- but the branch that added it kept it selectable rather
        // than always-on, which means it did not measure as strictly better. One kernel
        // at one shape answers that, so it stays a knob here.
        //
        // THE OLD ANSWER WAS A TWO-VARIABLE A/B. Until `imparo_st_gemm_N_full_h` existed,
        // the predicate-free entry point had no f16-activation twin and the host's mirror
        // condition carried `!full`, so selecting it also took the activations back to
        // f32. Measured that way it read 769.1 against 789.5 tok/s on LFM2 at 17123
        // tokens -- a preference for the mirror, recorded as a verdict on the predicates.
        values: &[0, 1],
        apply: crate::set_q8_full_tiles,
        current: crate::q8_full_tiles,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::PrefillGemmWidths,
    },
    KnobDecl {
        // WHICH GEMM DESIGN SERVES Q8_0 PREFILL. 0 = st_gemm, the staged design (both
        // operands copied to threadgroup memory, K chunk 32), with the tiles the two
        // knobs above chose; k >= 1 = rt_gemm<Q8> at RT_SHAPES[k - 1], the register-tiled
        // design the Q4 family runs (weights staged, activations through a register
        // prefetch pipeline, K chunk 64). One knob carries design AND rt shape because
        // the shape only exists under rt: swept as its own knob it would read flat under
        // st and the tuner would record a meaningless pick.
        //
        // Both designs stage identical half operands and accumulate in the same k order,
        // so moving this knob is not bit-affecting; the pins say so.
        name: "q8_design",
        // rt_gemm<Q8> reads the row-major layout only: on a Q8_0_TM (tile-major) model the
        // staged design is the one legal answer.
        legal: Some(|v, m, _d| {
            crate::q8_design_legal(v) && (v == 0 || m.weight_kinds & (1 << 3) == 0)
        }),
        bit_affecting: false,
        derive: None,
        candidates: Some(|_m, _d| (0..crate::q8_designs()).collect()),
        after: &["st_gemm_shape", "st_gemm_large_shape"],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & ((1 << 2) | (1 << 3)) != 0),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_q8_design,
        current: crate::q8_design,
        // SCREENED for the reason rt_shape is: the wide shapes spill and read orders of
        // magnitude slow on the tiny-size screen.
        screened: true,
        sweep: Sw::Values,
        workload: Wl::PrefillGemmWidths,
    },
    KnobDecl {
        name: "q8_gemv_max_tok",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &["q8_token_tile", "q8_batch_sgs", "st_gemm_shape"],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & ((1 << 2) | (1 << 3)) != 0),
        // The boundary between the narrow token-tile kernel and the wide GEMM, searched
        // AFTER both sides are settled -- the same relationship gemv_max_tok has to the
        // gemv tuple. hi = 64 keeps every rung on the token tile; lo = 1 sends every rung
        // above one token to the GEMM.
        tuple: Some("q8_narrow"),
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_q8_gemv_max_tok,
        current: crate::q8_gemv_max_tok,
        screened: false,
        sweep: Sw::Crossing {
            ladder: &[2, 4, 8, 16, 32],
            hi: 64,
            lo: 1,
        },
        workload: Wl::NarrowMix(8),
    },
    // NOT DECLARED, and each for a stated reason rather than by omission:
    //
    //   st_gemm_large_shape
    //       The SECOND tile of the prefill pair. There is no boundary knob any more: the
    //       dispatch picks between the two by comparing their PADDED token counts, which
    //       is arithmetic and needs nothing stored (see `q8_matmat`, and "What math
    //       decides" in docs/tuner-design.md). What remains genuinely tunable is WHICH
    //       second tile is worth building, and PrefillGemm still times one token count,
    //       so it cannot rank that yet -- the workload has to cover more than one ubatch
    //       width first. Until it does, the engine keeps large_shape equal to
    //       st_gemm_shape, which is the single-shape path exactly.
    //
    //   q8_grid_token_x / q8_typed_scale / q8_dev_a
    //       All three MEASURED WORSE end to end and stay reachable as evidence, the way
    //       shape 9 does. Grid order: 861.4/878.3 short, 783.3/790.8 deep. Typed scale:
    //       874.2/871.6 short, 811.8/827.2 deep. Device-side activations, at prefetch
    //       depth 0/2/5: 839.1/846.7/836.4 short against 869.2, 795.5/793.3/729.7 deep
    //       against 827.2. Each was verified to reach the dispatch and to give
    //       bit-identical logits before it was timed; none is a knob because a knob
    //       carries a value some machine wants, and no measurement here wants these.
    //
    //   q8_shared_proj / q8_shared_half
    //       Select a fused two-projection kernel. That is GRAPH fusion -- which two
    //       tensors are adjacent in the workflow -- not a kernel geometry a single
    //       dispatch can be ranked on, and no workflow here fuses two projections.
    //
    //   q8_silu_half
    //       A FUSED silu+half-store variant. The epilogue itself now takes the
    //       activation kind (imparo_backend::Epilogue), so SwiGLU is correct on every
    //       Q8 route without it; this knob would only add a second fused store shape.
    KnobDecl {
        name: "attn_threads",
        legal: None,
        // MEASURED to change decode output bits: at 256 the decode step hash is
        // 5d15bf53e803f597, at 512 it is 28bcea123dddca53, same prompt. The thread count
        // sets how the sliced decode attention splits and recombines its partials, and a
        // different split sums them in a different order.
        //
        // det_gate does NOT catch this: it pins prefill logits, and this knob is decode
        // only. So a tuner ranking on time alone can move the decode numerics with
        // nothing objecting -- which is why the caller has to opt in.
        bit_affecting: true,
        derive: None,
        candidates: Some(|_m, d| threads_upto(128, d.max_threads)),
        after: &[],
        cross_check: None,
        applies: None,
        // Sets the thread count for BOTH decode attention kernels -- the score-tile
        // path and the streaming path -- so the crossover between them depends on it.
        // Measured apart from the tuple it gave attn_stream_min_pos=2048; measured with
        // it, 6144. A knob that changes another knob's answer belongs in its tuple.
        tuple: Some("decode_stream"),
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_attn_threads,
        current: crate::attn_threads_current,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::AttentionDecodeDeep,
    },
    KnobDecl {
        // INERT ON AttentionPrefillDeep, found by the tuner's own inert check and then
        // confirmed by reading the dispatch site. The qcomb prefill attention kernels take
        // their thread count from the DERIVED simdgroup count (`g_nsg_512x * 32` for QT-16)
        // or from a hardcoded 256 (QT-8). Neither reads this knob. The only site that does
        // is the score-tile path, and the qcomb branch catches every prefill attention
        // dispatch before that path is reached.
        //
        // So it governs a FALLBACK: what runs if a qcomb pipeline failed to compile, or if
        // the QT-16 shape does not fit the device's threadgroup limit and QT-8 does not
        // exist either. That is real but unreachable here, which is why every candidate
        // lands within noise.
        //
        // Left declared rather than deleted: the fallback exists, and the sweep now SAYS
        // it cannot see the knob instead of quietly ranking noise.
        name: "attn_threads_prefill",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: Some(|_m, d| threads_upto(128, d.max_threads)),
        after: &[],
        cross_check: Some(Wl::AttentionPrefillMajor),
        applies: None,
        tuple: Some("prefill_attn"),
        // Thread count and BLK shape the same prefill attention dispatch: threads
        // fixes NSG, and the score phase's work-unit count is PT/(8*BLK) measured
        // against NSG. #37 records ten attempts across two sessions failing because
        // these terms only move together.
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_attn_threads_prefill,
        current: crate::attn_threads_pf_current,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::AttentionPrefillDeep,
    },
    // rt_shape IS screened: shape 7 spills registers and reads ~35-80x on the
    // tiny-size screen; stage 2 decides among survivors end to end (its micro
    // ranking was once backwards by 11%).
    KnobDecl {
        name: "rt_shape",
        legal: Some(|v, _m, _d| crate::rt_shape_legal(v)),
        bit_affecting: false,
        derive: None,
        // COUNTED, not written. The shape table lives in the .mm and already exports its
        // size; a literal ladder here is the same number in two places, and the failure is
        // silent -- add a shape and the tuner simply never tries it.
        candidates: Some(|_m, _d| (0..crate::rt_shape_count()).collect()),
        after: &[],
        cross_check: None,
        // Q4-only kernel: on a Q8 model the route never dispatches it, and the sweep
        // used to spend DecodeMix reps on it and record a pick (review #116, D11).
        applies: Some(|m| m.weight_kinds & (1 << 1) != 0),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: |v| {
            let _ = crate::set_rt_shape(v);
        },
        current: crate::rt_shape_current,
        screened: true,
        sweep: Sw::Values,
        workload: Wl::PrefillGemm,
    },
    KnobDecl {
        name: "nr0",
        // MATH, not a sweep. nr0 gives a THREAD more independent work by
        // handing it extra output rows, and a thread already has n_in / lanes
        // elements of it. The host says the same thing at the pipeline build:
        // "llama.cpp needs NR0=4 because it puts just 2 lanes on each block and
        // has to find independent work somewhere; 32 lanes per row already have
        // it."
        //
        // So nr0 > 1 can only pay when a thread is starved at the WIDEST lanes,
        // which needs n_in / 32 to be small. The narrowest decode row is n_embd,
        // and at n_embd 2560 that is 80 elements per thread -- an order of
        // magnitude past anything latency hiding needs. The measurement agrees
        // and always did: across all 16 pairs nr0=2 never once beat nr0=1, and
        // at lanes=4 it lost by up to 47%.
        //
        // Pruning it halves the gemv tuple, 32 combinations to 16. The condition
        // is kept rather than the knob deleted, so a model narrow enough to
        // starve a thread still gets the search.
        legal: Some(|v, m, _d| v == 1 || m.n_embd / 32 < 8),
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: None,
        // Q4-only kernel: on a Q8 model the route never dispatches it, and the sweep
        // used to spend DecodeMix reps on it and record a pick (review #116, D11).
        applies: Some(|m| m.weight_kinds & (1 << 1) != 0),
        tuple: Some("gemv"),
        // Was stage 2. Its WORKLOAD was already right -- NR0 applies to the decode
        // fast path only -- so it was end-to-end for no reason. Grouped with lanes,
        // which it shares a pipeline table with.
        category: KnobCategory::Benched,
        values: &[1, 2],
        apply: crate::set_nr0,
        current: crate::nr0_current,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        // "Threadgroups needed to fill the GPU" for the sliced decode attention path, so it
        // belongs on the ATTENTION workload, not on decode matvecs. It was stage 2 on
        // Wl::DecodeMix -- the same defect rt_shape had.
        //
        // GUARDS: the decode attention slice count. slices = ceil(attn_min_tgs /
        // direct_tgs), so the dispatch ends up with about attn_min_tgs threadgroups in
        // flight. That is a DEVICE quantity, and the probe that measures it now exists:
        // discovery's memory-bound fill point, which raises the threadgroup count against
        // a streaming read until aggregate bandwidth stops climbing.
        //
        // It measures 24 on an M3 Pro (~136 of ~155 GB/s), against the 72 compiled in
        // here -- so the literal was three times the point where this machine's memory
        // system saturates. It is NOT simply replaced by 24, because slicing further also
        // costs a combine pass over the partials, and the fill point does not know that
        // cost. It sets the candidates; the bench picks.
        name: "attn_min_tgs",
        legal: None,
        bit_affecting: false,
        // One value serves every context length, so a shallow decode is where it would
        // show if that is false. The winner is chosen on the deep workload because that
        // is where a decode step spends its attention time.
        derive: None,
        // The scale comes from the device, the choice from the bench. The memory-bound
        // fill point is what a latency-bound dispatch needs in flight; the range above it
        // is there because slicing further also costs a combine pass, and where that
        // trade lands is not something arithmetic answers.
        candidates: Some(|_m, d| {
            // The fill point is in THREADS; this knob counts THREADGROUPS of the decode
            // attention kernel, so divide by that kernel's own thread count. Taking the
            // probe's threadgroup count directly would have imported its 256-thread choice
            // into a kernel that runs 512, and the range would be wrong by 2x on any
            // machine where the two differ.
            let fill = d.fill_threads_membound;
            if fill == 0 {
                return vec![36, 72, 144, 288]; // unprobed: the compiled range stands
            }
            let per_tg = crate::attn_threads_current().max(32);
            let base = (fill / per_tg).max(1);
            // THE LADDER MUST OUTRUN ITS OWN ANSWER. A winner sitting at the top rung has
            // not been shown to be a maximum -- nothing beyond it was measured, and that is
            // the same defect as a boundary scan reporting "never" because its ladder
            // stopped too early. 12x is here so the curve visibly turns over: this kernel
            // wants 6x the bandwidth fill point (48 threadgroups against 8), which is the
            // fraction of its time spent in softmax barriers rather than issuing loads --
            // a kernel property no device probe can predict, which is why this is benched
            // at all. The RANGE is derived; only the pick is measured.
            vec![base, base * 2, base * 3, base * 6, base * 12]
        }),
        after: &[],
        // One value serves every context length, so a shallow decode is where it
        // would show if that is false. The winner is chosen on the deep workload because
        // that is where a decode step spends its attention time.
        cross_check: Some(Wl::AttentionDecode),
        applies: None,
        tuple: None,
        category: KnobCategory::Benched,
        values: &[36, 72, 144, 288],
        apply: crate::set_attn_min_tgs,
        current: crate::attn_min_tgs_current,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::AttentionDecodeDeep,
    },
    // The decode-attention routing boundary. MICRO-tuned as a span crossing:
    // it is a per-dispatch question (which kernel is cheaper for a span), and
    // measuring it end to end cannot resolve it -- 7 of 42 layers are
    // full-attention, so at shallow spans the choice is under 1% of a decode
    // step against 7% run-to-run spread. The compiled default (3072, measured
    // on the reference M3 Pro) stands when the scans disagree.
    KnobDecl {
        // THE POSITION TILE, tuned inside a derived bound. `qcomb_derive_pt` gives the
        // widest tile that FITS this device; it is not the fastest -- measured on E4B
        // q4_0 prefill the derived 480 was 1.3% slower than 128, because a wider tile
        // costs threadgroup occupancy. Values above what fits are clamped, so a ladder
        // written here cannot ask for a shape the device cannot hold.
        //
        // bit_affecting: a different tile changes where the online softmax rescales.
        // SIMDGROUPS PER QCOMB THREADGROUP, and the reason this is a knob at all.
        //
        // qcomb_derive_nsg answers "does this nsg FIT" -- NDB whole, room for a second
        // threadgroup, accumulator budget. That is a LIMIT. It was also being used to
        // answer "which nsg is FASTEST", by returning the first fit scanning up from a
        // written floor of 8, and at head_dim 64 that floor hid nsg 4: legal, bit-identical
        // (det_gate pin EXACT at every length) and 2.9% faster on the 17123-token leg,
        // 790.5 against 768.2. A limit doing a value's job.
        //
        // Do NOT replace this with a lower floor. First-fit does not rank, it only orders:
        // at qt 8 every legal nsg fits the accumulator rule, so a floor of 1 would pick
        // nsg 1 -- 32 threads per threadgroup, never measured, chosen by loop order. The
        // candidates are derived and the value is measured; that is the whole point.
        //
        // bit_affecting: false. It moves which simdgroup owns which dim slice, not the
        // order any output is summed in -- confirmed EXACT at all eight lengths.
        name: "attn_qcomb_nsg",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: Some(|m, _d| {
            // ASKED OF THE BACKEND, which owns the rule. It answers only once the device
            // profile has been measured; before that -- and in the pre-init registry test
            // -- it returns nothing, and the honest ladder is the compiled default alone,
            // exactly as attn_qcomb_pt falls back to its own. A knob may never resolve to
            // an EMPTY ladder: that reads as "swept" while sweeping nothing, which is the
            // failure this registry has a test for.
            let v = crate::qcomb_nsg_candidates(m.head_dim);
            if v.is_empty() { vec![8] } else { v }
        }),
        after: &[],
        cross_check: Some(Wl::AttentionPrefillReuse),
        applies: Some(|m| crate::qcomb_has_hd(m.head_dim)),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_qcomb_nsg,
        current: crate::qcomb_nsg,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::AttentionPrefillDeep,
    },
    KnobDecl {
        // SIMDGROUPS PER FA THREADGROUP. The FA op (the prefill default at head dim <= 128)
        // splits a block's column tiles, the queries and the output tiles over its
        // simdgroups; more simdgroups mean less work and fewer registers per simdgroup and
        // more barrier participants. The LIMIT is derived (fa_nsg_candidates: the kernel's
        // divisibility rules and the device's thread cap); the VALUE is measured: on the
        // reference M3 Pro at hd 64, 4 beat 8 by 29% (17.9 vs 23.0 ms at 16896 keys).
        //
        // bit_affecting: false. Which simdgroup owns which tile does not change the order
        // any score or output is summed in (det_gate pins EXACT at nsg 4 and 8).
        name: "attn_fa_nsg",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: Some(|m, _d| {
            let v = crate::fa_nsg_candidates(m.head_dim);
            if v.is_empty() {
                vec![crate::fa_nsg()]
            } else {
                v
            }
        }),
        after: &[],
        cross_check: Some(Wl::AttentionPrefillReuse),
        applies: Some(|m| crate::fa_has_hd(m.head_dim)),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_fa_nsg,
        current: crate::fa_nsg,
        // Screened on an attention dispatch (a spilling count reads 10x): the sweep then
        // never runs the deep workload on it.
        screened: true,
        sweep: Sw::Values,
        workload: Wl::AttentionPrefillDeep,
    },
    KnobDecl {
        // KEYS PER FLASH-DECODING SLICE (the decode attention route at head dim <= 128,
        // grouped by KV head). One threadgroup takes one chunk: a smaller chunk means more
        // threadgroups streaming in parallel and more partials for the combine; a larger one
        // fewer, longer streams. The LIMIT is derived (the slice's scores must fit the
        // device's threadgroup memory); the VALUE is measured: on the reference M3 Pro the
        // sweep read 128 / 256 / 512 / 1024 at 2532 / 2366 / 2510 / 3546 us and 256 is the
        // seat. bit_affecting: the split changes the order the combine sums the slices in.
        name: "attn_fd_chunk",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: Some(|m, _d| {
            let share = m.n_head.checked_div(m.n_kv).unwrap_or(0);
            let v = crate::attn_fd_chunk_candidates(m.head_dim, share);
            if v.is_empty() {
                vec![crate::attn_fd_chunk()]
            } else {
                v
            }
        }),
        after: &[],
        cross_check: None,
        applies: Some(|m| {
            let share = m.n_head.checked_div(m.n_kv).unwrap_or(0);
            !crate::attn_fd_chunk_candidates(m.head_dim, share).is_empty()
        }),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_attn_fd_chunk,
        current: crate::attn_fd_chunk,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeAttentionStep,
    },
    KnobDecl {
        // The vector decode kernel reads K/V once per QUERY head, so it wins while a span's
        // K/V is cache-resident and loses to the K-once-per-KV-head routes past that. The
        // crossover is measured per model on the deep decode step (1024 on both reference
        // models: at 16k keys the unsliced kernel loses 11% of the step on 512-dim heads,
        // the isolated probe notwithstanding); the ladder runs past the deep span for a
        // sliced form of the kernel. bit_affecting: the route changes the summation order.
        name: "attn_vec_max_keys",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: Some(|_m, _d| vec![512, 1024, 2048, 4096, 8192, 16384, 65536]),
        after: &[],
        cross_check: None,
        applies: Some(|m| m.head_dim % 32 == 0),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_attn_vec_max_keys,
        current: crate::attn_vec_max_keys,
        screened: false,
        sweep: Sw::Values,
        // NOTE: this workload runs one decode-attention step at depth, where every
        // candidate below the span is inert and the tuner keeps the default; a short-span
        // workload that crosses the ladder is the tuner follow-up (task #122).
        workload: Wl::DecodeAttentionStep,
    },
    KnobDecl {
        // MEGA BLOCK GRID (task #141). The persistent FFN(+PLE) dispatch's threadgroups all
        // wait on each other, so the grid is bounded by a DERIVED limit -- gpu_cores x the
        // pipeline's maxTotalThreadsPerThreadgroup (see the host) -- and the shape inside
        // it is a tuned value: the barrier cost grows with the number of arrivers, the
        // GEMV phases want simdgroups in flight, 1024-thread groups leave cores idle.
        // Candidates are per core: one, two-minus-slack, two per core.
        name: "mega_tgs",
        legal: Some(|v, _m, _d| {
            let lim = crate::mega_threads_limit();
            lim == 0 || v * crate::mega_nsg_current() * 32 <= lim
        }),
        bit_affecting: false,
        derive: None,
        candidates: Some(|_m, _d| {
            let c = crate::gpu_cores();
            if c == 0 {
                return vec![8, 16];
            }
            let mut v: Vec<u32> = [c, 2 * c - 4, 2 * c - 2, 2 * c]
                .into_iter()
                .filter(|&t| t > 0)
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        }),
        after: &[],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & (1 << 1) != 0 && crate::mega_level() >= 1),
        tuple: Some("mega"),
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_mega_tgs,
        current: crate::mega_tgs_current,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        // Simdgroups per mega threadgroup; with mega_tgs the product is the grid.
        name: "mega_nsg",
        legal: Some(|v, _m, d| {
            let lim = crate::mega_threads_limit();
            v * 32 <= d.max_threads
                && (lim == 0 || crate::mega_tgs_current() * v * 32 <= lim)
        }),
        bit_affecting: false,
        derive: None,
        candidates: Some(|_m, d| {
            [8u32, 16, 32]
                .into_iter()
                .filter(|&n| n * 32 <= d.max_threads)
                .collect()
        }),
        after: &[],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & (1 << 1) != 0 && crate::mega_level() >= 1),
        tuple: Some("mega"),
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_mega_nsg,
        current: crate::mega_nsg_current,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        name: "attn_qcomb_pt",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: Some(|m, _d| {
            // DERIVE THE LIMIT, TUNE THE VALUE. The ladder used to end at 512, which is
            // neither this device's limit nor a measured best -- it was a round number.
            // The widest tile the threadgroup budget allows is arithmetic; every power of
            // two below it is a rung, and the limit itself is the last one because the
            // widest legal tile is the interesting end of the range.
            let lim = crate::qcomb_pt_limit(m.head_dim);
            if lim == 0 {
                return vec![128];
            }
            let mut v: Vec<u32> =
                core::iter::successors(Some(64u32), |t| t.checked_mul(2))
                    .take_while(|t| *t <= lim)
                    .collect();
            if v.last() != Some(&lim) {
                v.push(lim);
            }
            v
        }),
        after: &[],
        cross_check: Some(Wl::AttentionPrefillReuse),
        applies: Some(|m| crate::qcomb_has_hd(m.head_dim)),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: crate::set_qcomb_pt,
        current: crate::qcomb_pt,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::AttentionPrefillDeep,
    },
    KnobDecl {
        // WHICH OF THIS MODEL'S HEAD DIMS TAKE QCOMB: one bit per dim it uses. A mask and
        // not a threshold, because a threshold assumes qcomb-worthiness rises with the dim
        // and nothing measured that; a mask asks the question the engine faces, per dim.
        //
        // bit_affecting: the two kernels sum in different orders, so the tuner reaches this
        // only under --allow-bit-changes and a default build routes as it does today.
        name: "attn_qcomb_mask",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: Some(|_m, _d| {
            // ONE BIT PER COMPILED SLOT, so a model with one head dim gets two masks and
            // one with three gets eight. The comment above this line described exactly
            // this and the hook was never written: the ladder resolved to `values`, which
            // is empty, so the knob had NOTHING to sweep and could never be chosen.
            (0..(1u32 << crate::qcomb_slot_count())).collect()
        }),
        after: &[],
        cross_check: Some(Wl::AttentionPrefillReuse),
        // Only where a second kernel exists for this model's head dim. A model at hd 128
        // has no qcomb instantiation, so every floor value routes it to qtile and the
        // coordinate is inert -- and an inert knob still costs a sweep and still reports
        // a winner, which is how a number that means nothing gets stored.
        applies: Some(|m| crate::qcomb_has_hd(m.head_dim)),
        tuple: None,
        category: KnobCategory::Benched,
        // DERIVED, not written: 2^slots candidates, so a model with one head dim gets
        // two and one with three gets eight. A literal ladder would be a list of numbers
        // meaningless for any model it was not written against.
        values: &[],
        apply: crate::set_qcomb_mask,
        current: crate::qcomb_mask,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::AttentionPrefillDeep,
    },
    KnobDecl {
        // BLK for the QT-8 qcomb prefill attention kernels -- the ones q4 and q8 prefill
        // run on. The score phase splits a PT-128 tile into ceil(PT/(8*BLK)) work units,
        // one per simdgroup, so the shipped BLK 4 gives 4 units over NSG 8 and leaves half
        // the machine idle; BLK 2 gives 8. The trade is arithmetic intensity, 1.25 loads
        // per MAC against 1.50. Whether double parallelism beats 1.2x the loads is a
        // DEVICE question, which is the whole reason this is a knob and not a literal --
        // the answer will not be the same on a 10-core and a 40-core GPU.
        // ALSO INERT on AttentionPrefillDeep, for a different reason: BLK is a template
        // argument of the QT-8 kernels, and QT-8 runs only when the KV cache is QUANTIZED.
        // The workload runs f16, so it takes the QT-16 kernels and never reaches these.
        //
        // The fix is a workload dimension the enum does not have: every attention workload
        // here is implicitly f16. Until Workload can say "prefill against a q4 cache", this
        // knob has nothing to be measured on, and the inert report is the honest output.
        name: "qcomb_blk",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: Some(Wl::AttentionPrefillMajor),
        applies: None,
        tuple: Some("prefill_attn"),
        category: KnobCategory::Benched,
        values: &[2, 4],
        apply: crate::set_qcomb_blk,
        current: crate::qcomb_blk_current,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::AttentionPrefillDeep,
    },
    KnobDecl {
        // DECODE GQA HEAD SHARING: two query heads per threadgroup instead of one. Worth
        // 15% of a decode dispatch at E4B's 8-over-2 geometry (595 us against 701-729),
        // and until now a compiled literal that no tuner could see -- the single most
        // valuable missing knob in the audit.
        //
        // MODEL-SPECIFIC. The kernel gates it on head_dim == 512, so on a model with no
        // such layers it selects nothing and must not be offered: measuring it would
        // spend samples on a value that changes nothing and then RECORD a pick, which
        // reads as a decision when none was made.
        name: "attn_stream_hq",
        category: KnobCategory::Benched,
        values: &[1, 2],
        apply: crate::set_attn_stream_hq,
        current: crate::attn_stream_hq_current,
        screened: false,
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: None,
        applies: Some(|m| m.deep_head_dim == 512),
        tuple: Some("decode_stream"),
        sweep: Sw::Values,
        workload: Wl::AttentionDecodeDeep,
    },
    KnobDecl {
        // Slices per decode attention dispatch. Coupled to hq: the GQA-grouped path has a
        // QUARTER of the threadgroups for the same work, so it needs about four times the
        // slices to fill the machine -- the right slice count is a function of which hq
        // won, which is exactly why they share a tuple.
        name: "attn_stream_slices",
        category: KnobCategory::Benched,
        values: &[8, 16, 32, 64],
        apply: crate::set_attn_stream_slices,
        current: crate::attn_stream_slices_current,
        screened: false,
        legal: None,
        bit_affecting: false,
        derive: None,
        // DERIVED RANGE, and the arithmetic also explains why this tuple reads INERT.
        // The streaming dispatch runs n_kv threadgroups per slice at a fixed 128 threads,
        // so `slices` threadgroups-worth of work is n_kv x slices x 128 threads. The
        // memory-bound fill point is 4096 threads here, which is 32 threadgroups, which is
        // 16 slices at n_kv=2.
        //
        // The old list started at 8 -- already 16 threadgroups, already half the fill
        // point -- so every candidate in it was at or past saturation and they measured
        // within 3% of each other, which is exactly what the inert check reported. A range
        // has to start below the thing it is looking for.
        candidates: Some(|m, d| {
            const STREAM_THREADS: u32 = 128; // the streaming kernel's fixed 4 simdgroups
            let fill = d.fill_threads_membound;
            if fill == 0 || m.n_kv == 0 {
                return vec![8, 16, 32, 64]; // unprobed: the compiled range stands
            }
            let sat = (fill / STREAM_THREADS / m.n_kv).max(1);
            // Straddle saturation: below it the machine is starved, above it each extra
            // slice is combine work bought for nothing.
            vec![(sat / 4).max(1), (sat / 2).max(1), sat, sat * 2, sat * 4]
        }),
        after: &[],
        cross_check: None,
        applies: Some(|m| m.deep_head_dim == 512),
        tuple: Some("decode_stream"),
        sweep: Sw::Values,
        workload: Wl::AttentionDecodeDeep,
    },
    KnobDecl {
        name: "attn_stream_min_pos",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        // MEASURED dependency, not a guessed one. attn_min_tgs decides how the
        // score-tile path slices, and that path is the side this boundary races the
        // streaming kernel against. Untuned it makes streaming look faster at every rung
        // (crossing 2048); tuned it makes streaming look slower at every rung (no
        // crossing). Same ladder, same kernels, opposite answer.
        after: &["attn_min_tgs"],
        cross_check: None,
        applies: None,
        // The crossover to the streaming kernel depends on that kernel's SHAPE, so it
        // is searched after hq and slices win -- tuple members that are boundary
        // searches always run last, against the winner.
        tuple: Some("decode_stream"),
        category: KnobCategory::Benched,
        values: &[],
        // NO-OP, like pt_512x and for the same reason: this knob's value is not
        // something the config should push back into the engine. See the note on the two
        // kinds of Derived above the registry. The env override still works -- it goes
        // through set_attn_stream_min_pos directly at init, not through this hook.
        apply: crate::set_attn_stream_min_pos,
        current: crate::attn_stream_min_pos_current,
        screened: false,
        // DERIVED FROM THE CACHE TYPE IN THE ENGINE, not stored here. The resolution and
        // the full measured table live at `stream_first` in imparo_metal.mm; the summary is
        // f16 8192, quantized 512, and UINT32_MAX in this knob means "derive".
        //
        // WHY THIS KNOB IS NO LONGER SEARCHED, and it is not a retreat from measuring:
        //
        //   The boundary is worth about 1% and this machine's end-to-end noise is about 1%
        //   per sample. A search cannot resolve it. Worse, bracket.py was reporting a
        //   SINGLE warm sample as `decode_median` -- `rows[1:]` after the warmup drop
        //   leaves one row at --repeat 2 -- so every A/B that set this knob compared one
        //   sample against one sample. That is how a 1% effect appeared to change sign
        //   between 5651 and 6518 tokens. Fixed: true median, plus decode_n and
        //   decode_spread in the line so an under-powered leg is visible in its own output.
        //
        //   Re-measured at --repeat 5 or better, the answer is monotone in depth and clean
        //   in direction, and it does not need a ladder to find -- it needs one bit,
        //   whether the cache is quantized.
        //
        // WHAT MADE EVERY EARLIER VALUE WRONG. This knob was raised 3072 -> 32768 on a
        // measurement that read "score-tile beats streaming by 1.4-1.8% at 5651 on all
        // three cache types". That was true of a streaming kernel running 512 threads.
        // Dropping it to 128 was worth +10.4% at 11941 and +20.6% at 24003, which reverses
        // the comparison outright. Every subsequent attempt to place the boundary --
        // ladders, crossing fits, per-KV tuned values -- was reasoning from a kernel that
        // no longer existed. When a kernel changes by 10%, the knobs that chose between it
        // and something else are stale, and no amount of better estimation fixes that.
        //
        // A STALE STORED CONFIG READS EXACTLY LIKE A TUNED ONE. While closing this out,
        // ~/.imparo held f16 32768 / q4_0 3072 / q8_0 3072, written hours before the commit
        // that changed what the tuner derives; 3072 was not even on the then-current
        // ladder. Nothing warned that the file predated the space it was tuned against.
        // NOT SEARCHED, AND THE LADDER IS WHY -- this was measured, not assumed, and the
        // ladder that failed is written down so nobody rebuilds it.
        //
        // The span ladder ran at [128, 512, 1024, 2048, 4096, 8192, 16384] against
        // hi = 0 (stream every rung) and lo = 1<<30 (no span reaches it, so score-tile),
        // three legs a side per rung, median, with each rung's own repeat scatter as its
        // noise band. TWO THINGS CAME OUT OF IT, and they contradict each other:
        //
        //   FAITHFUL (one attention dispatch per layer, as the engine issues):
        //     the scatter is 0.68%-6.13% per rung and the effect is 0.4%-2.5%.
        //     The noise is larger than the quantity. Verdicts: f16 16384, q4_0 16384,
        //     q8_0 4096, against a bracket that measures 8192 / ~512 / ~512.
        //
        //   AMPLIFIED (8 dispatches per layer, to lift attention above the scatter):
        //     the scatter falls to 0.13%-1.24% and every rung reads cleanly -- and the
        //     answer becomes 128 for all three cache types, because 56 attention dispatches
        //     per step is a concurrency regime the engine never enters. Streaming runs 128
        //     threads per threadgroup against score-tile's 512, and they do not scale alike
        //     when the machine is saturated with them. KV cache reuse was the first
        //     suspicion and it is REFUTED: round-robining the repeats over different layers
        //     so no dispatch re-reads its predecessor's KV moved f16@128 from +2.21% to
        //     +2.31%, i.e. not at all.
        //
        // So the workload is faithful at 1x and precise at 8x and not both, and the fitted
        // crossing cannot bridge it either: both kernels' intercepts carry ~18 ms of weight
        // streaming that belongs to neither, and n* = (F_hi - F_lo)/(m_lo - m_hi) is then a
        // ratio of two sub-1% differences. That produced 1587 on f16, a degenerate
        // `0 us + 0.0000/pos` on q8_0, and "the lines meet at -5388" on q4_0 -- three
        // answers from one physical situation.
        //
        // THE HONEST STATEMENT: a sub-1% attention difference inside a faithful decode step
        // is below this harness's resolution, and that is a property of the quantity, not a
        // defect to be tuned out. The engine derives the boundary from the cache type
        // instead -- see `stream_first` in imparo_metal.mm for the resolution and for the
        // 15 end-to-end points, every one of which the derive agrees with.
        sweep: Sw::Derived,
        workload: Wl::AttentionDecode,
    },
    // THE SUBMISSION FAMILY. Neither of these can be micro-benched: every workload here
    // is a single op, and these govern how a MULTI-LAYER encode loop is submitted, so
    // there is nothing for a one-op benchmark to see. Both are DERIVED instead, from two
    // costs the tuner measures directly. See `derive_flush`.
    //
    // Their inputs are MEASURED, not queried -- which is the difference from pt_512x. A
    // host that never runs the tuner has no commit or encode measurement, so the
    // derivation returns 0 there and the compiled value stands. That is the correct
    // behaviour and not a gap: the compiled 7 is within 0.014% of the derived answer on
    // the machine this was measured on, and a host that cares can run the tuner.
    KnobDecl {
        name: "flush_layers",
        legal: None,
        bit_affecting: false,
        derive: Some(|m, d| {
            derive_flush(
                d.commit_overhead_ns,
                d.encode_cost_ns,
                m.layer_dispatches,
                m.n_layers,
            )
        }),
        candidates: None,
        after: &[],
        cross_check: None,
        applies: None,
        tuple: None,
        category: KnobCategory::DeviceProfile,
        values: &[],
        apply: crate::set_flush_layers_decode,
        current: || crate::flush_layers(true),
        screened: false,
        sweep: Sw::Derived,
        workload: Wl::DecodeMix,
    },
    KnobDecl {
        // The work term the first attempt was missing. `derive_flush` alone returns 7 for
        // prefill against a shipped 0, because it trades submission cost against encode lag
        // and never asks how much GPU WORK a layer holds -- which is the whole difference
        // between the phases.
        //
        // What flushing buys is an earlier START: without it the GPU waits for all L layers
        // to be encoded. That lag is L x encode_per_layer, and what it costs RELATIVE to
        // the phase is
        //
        //     lag / total  =  (L x encode_per_layer) / (L x layer_work)  =  encode / work
        //
        // -- the layer count cancels, and the answer is a pure ratio of two measured times.
        // Decode and prefill pay the SAME encode and hold wildly different work, so the
        // same arithmetic separates them without a second formula.
        name: "flush_layers_prefill",
        legal: None,
        bit_affecting: false,
        derive: Some(|m, d| {
            // Below this the encode lag is not worth a single extra submission.
            const NEGLIGIBLE: f64 = 0.001; // 0.1% of the phase
            let work = f64::from(d.layer_work_prefill_ns);
            let encode = f64::from(d.encode_cost_ns) * f64::from(m.layer_dispatches);
            if work <= 0.0 || encode <= 0.0 {
                return 0; // unprobed: the compiled "never flush" stands
            }
            if encode / work < NEGLIGIBLE {
                return 0; // the GPU is never waiting on the host; do not pay a commit
            }
            derive_flush(
                d.commit_overhead_ns,
                d.encode_cost_ns,
                m.layer_dispatches,
                m.n_layers,
            )
        }),
        candidates: None,
        after: &[],
        cross_check: None,
        applies: None,
        tuple: None,
        category: KnobCategory::DeviceProfile,
        values: &[],
        apply: crate::set_flush_layers_prefill_only,
        current: || crate::flush_layers(false),
        screened: false,
        sweep: Sw::Derived,
        workload: Wl::PrefillGemm,
    },
];

impl BackendKnobs for MetalBackend {
    fn knob_registry(&self) -> &'static [KnobDecl] {
        METAL_KNOBS
    }
    /// The METAL search space version -- backend-owned since #19 (history: see
    /// imparo-host's version doctrine; v10 added nb8_max/nb8_shape, v11
    /// gemv_max_tok, v12 attn_stream_min_pos).
    ///
    /// v13: ADDED the Q8_0 weight family -- q8_decode_sgs, q8_decode_rows,
    /// q8_batch_sgs, q8_token_tile, st_gemm_shape, q8_full_tiles, q8_gemv_max_tok. The
    /// knob SET changed, which is what this version exists to invalidate: a v12 file has
    /// no line for any of them, so it would leave a Q8_0 model on compiled defaults while
    /// reporting itself as tuned. Same release: attn_stream_min_pos stopped being stored
    /// at all (the engine derives it from the cache type), so a v12 file's value for it
    /// would be applied as an explicit override of that derive.
    ///
    /// v14: the knob SET is unchanged, and this still has to move, because every
    /// attention value a v13 file holds was measured against the WRONG KERNELS. The tuner
    /// never injected the model's head dims, so its Metal library compiled no qcomb
    /// attention slots at all: its prefill workload dispatched qtile while the engine
    /// dispatched qcomb, and both qcomb knobs reported "N/A to this model". A v13 file is
    /// not stale, it is a measurement of a library this engine does not build.
    ///
    /// Also in v14, and harmless by comparison: the thread and simdgroup ladders are
    /// derived from the device's own limit rather than written down, and
    /// `attn_qcomb_mask` gained the `2^slots` candidate hook its comment had always
    /// described -- without it the mask resolved to an empty ladder and was never swept.
    fn space_version(&self) -> u32 {
        // 19: mega_tgs / mega_nsg join the space (the persistent mega block's grid shape).
        // 18: q8_tm_decode_sgs joins the space (the tile-major decode GEMV's simdgroups).
        // 18: attn_fd_chunk joins the space (keys per flash-decoding slice).
        // 16: attn_fa_nsg joins the space (the FA op's simdgroup count; its query tile is a
        //     structural 8, see the kernel).
        // 15: attn_qcomb_nsg joins the space. The knob set changed, so a v14 tune file
        // must not be read as if it described this space -- it has no value for the new
        // knob, and the derivation it silently fell back to is the thing being replaced.
        19
    }
}

/// IMPARO_NORM_ROW=0 disables the one-row norm+residual fusion (A/B arm on one binary).
/// IMPARO_NORM_ROW_MASK (test instrument): bit 0 the dual attention-side seam, bit 1 the
/// unscaled norm+add, bit 2 the scaled norm+add; default all.
fn norm_row_fusion_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_NORM_ROW").map_or(true, |v| v != "0"))
}
fn norm_row_mask(bit: u32) -> bool {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let m = *M.get_or_init(|| {
        std::env::var("IMPARO_NORM_ROW_MASK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7)
    });
    m & (1 << bit) != 0
}

#[cfg(test)]
mod ladder_tests {
    use super::{simdgroups_upto, threads_upto};

    /// The ladders that used to be written out must come back unchanged on the machine
    /// they were written on -- otherwise this change is a retune wearing a refactor's
    /// clothes, and the pinned numbers would move for a reason nobody recorded.
    #[test]
    fn the_written_ladders_are_reproduced_at_1024_threads() {
        assert_eq!(threads_upto(128, 1024), vec![128, 256, 512, 1024]);
        assert_eq!(simdgroups_upto(1, 1024), vec![1, 2, 4, 8, 16, 32]);
    }

    /// The point of deriving them: a device with half the budget must not be offered a
    /// dispatch it cannot run. 1024 threads was in the registry as a literal.
    #[test]
    fn a_narrower_device_is_never_offered_an_illegal_dispatch() {
        assert_eq!(threads_upto(128, 512), vec![128, 256, 512]);
        assert_eq!(simdgroups_upto(1, 512), vec![1, 2, 4, 8, 16]);
        for max in [256u32, 512, 1024, 2048] {
            assert!(threads_upto(128, max).iter().all(|t| *t <= max));
            assert!(simdgroups_upto(1, max).iter().all(|s| s * 32 <= max));
        }
    }

    /// A ladder is never empty: the lowest rung is the compiled default, and a device
    /// that cannot run it is one this backend cannot run at all.
    #[test]
    fn the_lowest_rung_survives_an_absurd_limit() {
        assert_eq!(threads_upto(128, 0), vec![128]);
        assert_eq!(simdgroups_upto(1, 0), vec![1]);
    }
}

/// One gemma4 layer (from the FFN input, or from the residual with the fronts) as one entry of
/// the mega kernel (task #158 step 2): the program's slots and words, the architecture's own
/// rules here, the Metal rules in the bridge.
fn mega_gemma4_entry(l: &Gemma4MegaLayer<'_>, n_tok: u32) -> bool {
    let &Gemma4MegaLayer {
        gate_kind,
        gate_off,
        up_kind,
        up_off,
        down_kind,
        down_off,
        post_ffw_norm_off,
        pg_kind,
        pg_off,
        pp_kind,
        pp_off,
        post_norm_off,
        next_norm_off,
        out_scale,
        n_embd,
        n_ff,
        ple,
        per_layer_off,
        eps,
        src,
        x,
        add,
        g,
        u,
        gate,
        per_layer,
        back,
        next_out,
        front,
    } = l;
    use crate::mega_slots::*;
    use crate::{MEGA_SLOT_BUF_R as R, MEGA_SLOT_BUF_RW as RW, MEGA_SLOT_BUF_W as W};
    use crate::{mega_slot_buf as sbuf, mega_slot_kv as skv, mega_slot_weight as sw};
    let qkv = front.and_then(|f| f.qkv);
    let attention = front.and_then(|f| f.attention);
    if n_tok != 1
        || gate_kind != 1
        || up_kind != 1
        || down_kind != 1
        || pg_kind != 1
        || pp_kind != 1
        || front.is_some_and(|f| f.wo_kind != 1)
        || qkv.is_some_and(|q| {
            q.wq_kind != 1
                || q.wk.is_some_and(|(k, _)| k != 1)
                || q.wv.is_some_and(|(k, _)| k != 1)
                || q.wk.is_some() != q.wv.is_some()
        })
    {
        return false;
    }
    // gemma4's own rules: Q4 row widths, the norm rows staged in the n_mid scratch, the
    // attention phase needs the front in the same dispatch, the q/k/v front needs the
    // attention phase (Q never leaves the block otherwise), the head geometry.
    if n_embd % 32 != 0 || n_ff % 32 != 0 || ple % 32 != 0 || n_ff < n_embd {
        return false;
    }
    if (attention.is_some() && front.is_none())
        || (qkv.is_some() && attention.is_none())
    {
        return false;
    }
    if front.is_some_and(|f| f.attn_in % 32 != 0) {
        return false;
    }
    if let (Some(q), Some(f), Some(a)) = (qkv, front, attention) {
        if q.rope_dim == 0 || q.rope_dim % 2 != 0 || q.rope_dim > f.head_dim {
            return false;
        }
        if q.wk.is_some() && a.kv_layer != q.layer {
            return false;
        }
        if a.n_heads * f.head_dim != f.attn_in {
            return false;
        }
    }
    let has_next = next_norm_off != NO_WEIGHT && next_norm_off != 0;
    let has_kv = qkv.is_some_and(|q| q.wk.is_some());
    let mut e = crate::MegaEntryFfi::new(0);
    e.min_level = if qkv.is_some() {
        5
    } else if attention.is_some() {
        4
    } else if front.is_some() {
        3
    } else {
        2
    };
    e.head_dim = front.map_or(0, |f| f.head_dim);
    e.attn_on = u32::from(attention.is_some());
    e.kv_write = u32::from(qkv.is_some() && has_kv);
    e.layer = qkv.map_or(0, |q| q.layer);
    e.scratch_rows = n_ff;
    if let Some(fr) = qkv.and_then(|q| q.freqs) {
        e.freqs = fr.as_ptr();
        e.n_freqs = fr.len() as u32;
    }
    e.slots[G4_W_GATE] = sw(gate_off, G4_O_GATE_OFF);
    e.slots[G4_W_UP] = sw(up_off, G4_O_UP_OFF);
    e.slots[G4_W_DOWN] = sw(down_off, G4_O_DOWN_OFF);
    e.slots[G4_W_PG] = sw(pg_off, G4_O_PG_OFF);
    e.slots[G4_W_PP] = sw(pp_off, G4_O_PP_OFF);
    e.slots[G4_W_N1] = sw(post_ffw_norm_off, G4_O_N1_OFF);
    e.slots[G4_W_N2] = sw(post_norm_off, G4_O_N2_OFF);
    if has_next {
        e.slots[G4_W_N3] = sw(next_norm_off, G4_O_N3_OFF);
        e.slots[G4_NXT] = sbuf(W, b(next_out), 0);
    }
    e.slots[G4_X] = sbuf(RW, b(x), 0);
    e.slots[G4_O] = sbuf(if front.is_some() { RW } else { R }, b(add), 0);
    e.slots[G4_CUR] = sbuf(R, b(src), 0);
    e.slots[G4_G] = sbuf(W, b(g), 0);
    e.slots[G4_U] = sbuf(W, b(u), 0);
    e.slots[G4_GATE] = sbuf(W, b(gate), 0);
    e.slots[G4_PER_LAYER] = sbuf(R, b(per_layer), 0);
    e.slots[G4_BACK] = sbuf(W, b(back), 0);
    if let Some(f) = front {
        e.slots[G4_W_O] = sw(f.wo_off, G4_O_WO_OFF);
        e.slots[G4_W_NPA] = sw(f.post_attn_norm_off, G4_O_NPA_OFF);
        e.slots[G4_W_NF] = sw(f.ffn_norm_off, G4_O_NF_OFF);
        e.slots[G4_ATTN] = sbuf(R, b(f.attn), 0);
        if attention.is_some() {
            e.slots[G4_ATTN_OUT] = sbuf(W, b(f.attn), 0);
        }
        e.u[G4_U_ATTN_IN] = f.attn_in;
        e.u[G4_U_Q_ROWS] = f.attn_in;
        e.u[G4_U_FRONT] = 1;
    }
    if let Some(a) = attention {
        e.kv_layer = a.kv_layer;
        e.start_pos = a.start_pos;
        e.slots[G4_QIN] = sbuf(if qkv.is_some() { RW } else { R }, b(a.q), 0);
        e.slots[G4_KC] = skv(crate::MEGA_SLOT_KV_K_R);
        e.slots[G4_VC] = skv(crate::MEGA_SLOT_KV_V_R);
        e.slots[G4_KV_PT] = skv(crate::MEGA_SLOT_KV_PT);
        e.u[MEGA_U_N_HEADS] = a.n_heads;
        e.u[MEGA_U_N_KV] = a.n_kv;
        e.u[MEGA_U_KV_WIDTH] = a.kv_width;
        e.u[MEGA_U_WINDOW] = a.window;
        e.u[MEGA_U_RING] = a.ring;
        e.u[MEGA_U_HAD_K] = a.had_k;
        e.u[MEGA_U_HAD_V] = a.had_v;
        e.u[G4_U_ATTN_PHASE] = 1;
    }
    if let Some(q) = qkv {
        e.slots[G4_W_Q] = sw(q.wq_off, G4_O_WQ_OFF);
        e.slots[G4_W_QN] = sw(q.q_norm_off, G4_O_QN_OFF);
        e.slots[G4_W_IN] = sw(q.in_norm_off, G4_O_IN_OFF);
        if let (Some((_, wk)), Some((_, wv))) = (q.wk, q.wv) {
            e.slots[G4_W_K] = sw(wk, G4_O_WK_OFF);
            e.slots[G4_W_V] = sw(wv, G4_O_WV_OFF);
            e.slots[G4_W_KN] = sw(q.k_norm_off, G4_O_KN_OFF);
            e.slots[G4_KC_W] = skv(crate::MEGA_SLOT_KV_K_W);
            e.slots[G4_VC_W] = skv(crate::MEGA_SLOT_KV_V_W);
        }
        e.slots[G4_KBUF] = sbuf(W, b(q.k), 0);
        e.slots[G4_VBUF] = sbuf(W, b(q.v), 0);
        e.u[G4_U_QKV_PHASE] = 1;
        e.u[G4_U_HAS_KV] = u32::from(has_kv);
        e.u[MEGA_U_N_ROT] = q.rope_dim;
        e.u[G4_U_N_FREQS] = e.n_freqs;
        e.f[MEGA_F_ROPE_BASE] = q.rope_base;
    }
    e.u[MEGA_U_N_EMBD] = n_embd;
    e.u[G4_U_N_MID] = n_ff;
    e.u[G4_U_PLE] = ple;
    e.u[G4_U_PL_OFF] = per_layer_off;
    e.u[G4_U_HAS_NEXT] = u32::from(has_next);
    e.f[MEGA_F_EPS] = eps;
    e.f[G4_F_OUT_SCALE] = out_scale;
    crate::mega_layer(&e)
}

/// One LFM2 layer as one persistent dispatch (task #147). Decode only, tile-major Q8_0
/// only (the kind the load-time repack produces).
fn mega_lfm2_entry(l: &Lfm2MegaLayer, n_tok: u32) -> bool {
    use crate::mega_slots::*;
    use crate::{MEGA_SLOT_BUF_RW as RW, MEGA_SLOT_BUF_W as W};
    use crate::{mega_slot_buf as sbuf, mega_slot_kv as skv, mega_slot_weight as sw};
    if n_tok != 1 || l.gate_kind != 3 || l.up_kind != 3 || l.down_kind != 3 {
        return false;
    }
    // LFM2's own rules: Q8 tile-major units of 8 rows, the conv taps within the kernel's
    // history, the head geometry of the attention mixer.
    if l.n_embd % 32 != 0 || l.n_ff % 32 != 0 || l.n_embd % 8 != 0 || l.n_ff % 8 != 0 {
        return false;
    }
    let mut e = crate::MegaEntryFfi::new(1);
    e.min_level = 2;
    e.slots[L2_W_GATE] = sw(l.gate_off, L2_O_GATE_OFF);
    e.slots[L2_W_UP] = sw(l.up_off, L2_O_UP_OFF);
    e.slots[L2_W_DOWN] = sw(l.down_off, L2_O_DOWN_OFF);
    e.slots[L2_W_FN] = sw(l.ffn_norm_off, L2_O_FN_OFF);
    e.slots[L2_X] = sbuf(RW, b(l.x), 0);
    e.slots[L2_O] = sbuf(RW, b(l.add), 0);
    e.slots[L2_G] = sbuf(W, b(l.g), 0);
    e.slots[L2_U] = sbuf(W, b(l.u), 0);
    e.u[MEGA_U_N_EMBD] = l.n_embd;
    e.u[L2_U_N_FF] = l.n_ff;
    e.f[MEGA_F_EPS] = l.eps;
    e.f[L2_F_Q_SCALE] = 1.0;
    match l.mixer {
        Lfm2MegaMixer::None => {}
        Lfm2MegaMixer::ShortConv {
            op_norm_off,
            in_kind,
            in_off,
            conv_off,
            out_kind,
            out_off,
            kernel,
            bcx,
            state,
            state_off,
            snap,
        } => {
            if in_kind != 3 || out_kind != 3 || kernel == 0 || kernel - 1 > 8 {
                return false;
            }
            e.u[L2_U_MIXER] = 1;
            e.u[L2_U_KERN] = kernel;
            e.slots[L2_W_OP] = sw(op_norm_off, L2_O_OP_OFF);
            e.slots[L2_W_IN] = sw(in_off, L2_O_IN_OFF);
            e.slots[L2_W_CONV] = sw(conv_off, L2_O_CONV_OFF);
            e.slots[L2_W_OUT] = sw(out_off, L2_O_OUT_OFF);
            e.slots[L2_BCX] = sbuf(RW, b(bcx), 0);
            e.slots[L2_STATE] = sbuf(RW, b(state), state_off);
            if let Some((sb, so)) = snap {
                e.slots[L2_SNAP] = sbuf(RW, b(sb), so);
                e.u[L2_U_HAS_SNAP] = 1;
            }
        }
        Lfm2MegaMixer::Attention {
            op_norm_off,
            wq_kind,
            wq_off,
            wk_kind,
            wk_off,
            wv_kind,
            wv_off,
            wo_kind,
            wo_off,
            q_norm_off,
            k_norm_off,
            head_dim,
            n_heads,
            n_kv,
            kv_width,
            kv_layer,
            start_pos,
            window,
            ring,
            rope_dim,
            rope_base,
            q_scale,
            q,
            k,
            v,
            attn,
            had_k,
            had_v,
        } => {
            if wq_kind != 3 || wk_kind != 3 || wv_kind != 3 || wo_kind != 3 {
                return false;
            }
            if head_dim % 32 != 0
                || kv_width != n_kv * head_dim
                || rope_dim == 0
                || rope_dim % 2 != 0
                || rope_dim > head_dim
            {
                return false;
            }
            e.u[L2_U_MIXER] = 2;
            e.head_dim = head_dim;
            e.attn_on = 1;
            e.kv_write = 1;
            e.kv_layer = kv_layer;
            e.start_pos = start_pos;
            e.scratch_rows = l.n_ff;
            e.slots[L2_W_OP] = sw(op_norm_off, L2_O_OP_OFF);
            e.slots[L2_W_Q] = sw(wq_off, L2_O_WQ_OFF);
            e.slots[L2_W_K] = sw(wk_off, L2_O_WK_OFF);
            e.slots[L2_W_V] = sw(wv_off, L2_O_WV_OFF);
            e.slots[L2_W_O] = sw(wo_off, L2_O_WO_OFF);
            e.slots[L2_W_QN] = sw(q_norm_off, L2_O_QN_OFF);
            e.slots[L2_W_KN] = sw(k_norm_off, L2_O_KN_OFF);
            e.slots[L2_QIN] = sbuf(RW, b(q), 0);
            e.slots[L2_KBUF] = sbuf(RW, b(k), 0);
            e.slots[L2_VBUF] = sbuf(RW, b(v), 0);
            e.slots[L2_ATTN] = sbuf(RW, b(attn), 0);
            e.slots[L2_KC_W] = skv(crate::MEGA_SLOT_KV_K_W);
            e.slots[L2_VC_W] = skv(crate::MEGA_SLOT_KV_V_W);
            e.slots[L2_KC] = skv(crate::MEGA_SLOT_KV_K_R);
            e.slots[L2_VC] = skv(crate::MEGA_SLOT_KV_V_R);
            e.slots[L2_KV_PT] = skv(crate::MEGA_SLOT_KV_PT);
            e.u[MEGA_U_N_HEADS] = n_heads;
            e.u[MEGA_U_N_KV] = n_kv;
            e.u[MEGA_U_KV_WIDTH] = kv_width;
            e.u[MEGA_U_WINDOW] = window;
            e.u[MEGA_U_RING] = ring;
            e.u[MEGA_U_HAD_K] = had_k;
            e.u[MEGA_U_HAD_V] = had_v;
            e.u[MEGA_U_N_ROT] = rope_dim;
            e.f[MEGA_F_ROPE_BASE] = rope_base;
            e.f[L2_F_Q_SCALE] = q_scale;
        }
    }
    crate::mega_layer(&e)
}
