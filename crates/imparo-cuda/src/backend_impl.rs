//! `Backend` for CUDA: FFI onto native/imparo_cuda.cu's extern "C" surface.
//! WRITTEN UNVERIFIED on a Mac (task #19) -- a CUDA host compiles and gates it.
//!
//! Per-op status (implemented-unverified means: real code, contracts mirrored from
//! the Metal backend, never run under nvcc here):
//!
//! | op                          | status                  |
//! |-----------------------------|-------------------------|
//! | buffers/arena/kv alloc+grow | implemented, unverified |
//! | matmat q4_0 (GEMV + GEMM + fused epilogue) | implemented, unverified; UNTUNED |
//! | matmat f32                  | implemented, unverified |
//! | rms_norm (+from)            | implemented, unverified |
//! | rope (neox, freqs)          | implemented, unverified (freqs upload TODO)     |
//! | hadamard (FWHT)             | implemented, unverified |
//! | kv_store f16/q4_0/q8_0      | implemented, unverified |
//! | attention (flash-style)     | implemented, unverified; per-query blocks, UNTUNED |
//! | elementwise + softcap       | implemented, unverified |
//! | argmax                      | implemented, unverified |
//! | row (embedding)             | implemented, unverified |
//! | matmat_from                 | implemented via matmat's src_row |
//! | ple_gather_combine          | STUB (todo!) -- gemma3n's PLE gather |
//! | shortconv                   | STUB (todo!) -- LFM2's gated short convolution |
//! | shortconv_snapshot          | STUB (todo!) -- its boundary snapshot          |
//! | set_epilogue / flush_layers | implemented (flush policy = compiled default) |
//!
//! The half-A mirror (XH/XH2) and the KVQ diagnostics are Metal-side host logic and
//! do not exist here; a CUDA port decides its own activation-precision staging.

use imparo_backend::{Backend, BufId, WeightKindWire};

unsafe extern "C" {
    fn imparo_cuda_init(weights: *const core::ffi::c_void, len: u64) -> i32;
    fn imparo_cuda_begin();
    fn imparo_cuda_flush();
    fn imparo_cuda_end() -> i32;
    fn imparo_cuda_alloc(id: u32, bytes: u64) -> i32;
    fn imparo_cuda_arena(bytes: u64) -> i32;
    fn imparo_cuda_place(id: u32, offset: u64, bytes: u64) -> i32;
    fn imparo_cuda_page_round(n: u64) -> u64;
    fn imparo_cuda_alloc_kv(n_layers: u32, bytes: *const u64) -> i32;
    fn imparo_cuda_grow_kv(n_layers: u32, bytes: *const u64) -> i32;
    fn imparo_cuda_write(id: u32, off: u64, src: *const f32, n: u64);
    fn imparo_cuda_write_u32(id: u32, off: u64, src: *const u32, n: u64);
    fn imparo_cuda_read(id: u32, off: u64, dst: *mut f32, n: u64);
    fn imparo_cuda_read_kv(layer: u32, is_v: u32, off: u64, dst: *mut u8, n: u64);
    fn imparo_cuda_set_epilogue(kind: u32);
    fn imparo_cuda_matmat(
        wkind: u32,
        w_off: u64,
        n_in: u32,
        n_out: u32,
        src: u32,
        dst: u32,
        n_tok: u32,
        src_row: u32,
    );
    fn imparo_cuda_rms_norm(
        buf: u32,
        src: u32,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
        has_w: u32,
    );
    fn imparo_cuda_rope(
        buf: u32,
        n_rot: u32,
        base: f32,
        head_dim: u32,
        n_heads: u32,
        start_pos: u32,
        n_tok: u32,
        freqs: *const f32,
    );
    fn imparo_cuda_hadamard(buf: u32, n: u32, nrot: u32);
    fn imparo_cuda_kv_store(
        src: u32,
        layer: u32,
        width: u32,
        start_pos: u32,
        n_tok: u32,
        is_v: u32,
        ring: u32,
    );
    fn imparo_cuda_attention(
        kv_layer: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv: u32,
        kv_width: u32,
        start_pos: u32,
        window: u32,
        n_tok: u32,
        ring: u32,
        q: u32,
        out: u32,
        kdq: u32,
        vdq: u32,
    );
    fn imparo_cuda_gelu(a: u32, n: u32);
    fn imparo_cuda_gelu_mul(a: u32, b: u32, n: u32);
    fn imparo_cuda_add(a: u32, b: u32, n: u32);
    fn imparo_cuda_add_scale(a: u32, b: u32, k: f32, n: u32);
    fn imparo_cuda_scale(a: u32, k: f32, n: u32);
    fn imparo_cuda_copy(dst: u32, src: u32, n: u32);
    fn imparo_cuda_mul_strided(
        a: u32,
        b: u32,
        n: u32,
        b_off: u32,
        b_stride: u32,
        a_stride: u32,
        n_tok: u32,
    );
    fn imparo_cuda_softcap(a: u32, cap: f32, n: u32);
    fn imparo_cuda_argmax(src: u32, dst: u32, n: u32);
    fn imparo_cuda_row(
        w_off: u64,
        width: u32,
        index: u32,
        scale: f32,
        dst: u32,
        dst_off: u32,
    );
}

/// Uploads the weight blob; call once before any op. Applies this host's stored
/// tuned configuration first (unless IMPARO_NO_HOSTCONFIG -- the tuner measures
/// from compiled defaults), mirroring the Metal backend's init.
pub fn init(weights: &[u8]) -> Result<(), i32> {
    if std::env::var("IMPARO_NO_HOSTCONFIG").is_err() {
        if let Some(batch) = apply_host_config(weights.len() as u64) {
            if std::env::var("IMPARO_BATCH").is_err() {
                unsafe { std::env::set_var("IMPARO_BATCH", batch.to_string()) };
            }
        }
    }
    let rc = unsafe { imparo_cuda_init(weights.as_ptr().cast(), weights.len() as u64) };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// Registry-driven config apply -- the CUDA analogue of imparo-metal's: every stored
/// key goes through the registry's declared hook, so a knob the tuner writes cannot
/// be silently dropped here. UNVERIFIED on real hardware, like the rest of this crate.
#[must_use]
pub fn apply_host_config(model_bytes: u64) -> Option<usize> {
    use imparo_backend::BackendKnobs as _;
    let space = CudaBackend.space_version();
    // CUDA has no quantized-cache path yet, so its tuning is f16 by construction.
    let c = imparo_host::read_for_device(model_bytes, false, ("cuda", space, "f16"))?;
    let reg = CudaBackend.knob_registry();
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
        imparo_host::path_for(&imparo_host::fingerprint_for("cuda", space, "f16")).display()
    );
    c.batch
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaBackend;

#[inline]
fn b(id: BufId) -> u32 {
    id as u32
}

#[allow(unused_variables)]
impl Backend for CudaBackend {
    fn begin(&self) {
        unsafe { imparo_cuda_begin() }
    }
    fn flush(&self) {
        unsafe { imparo_cuda_flush() }
    }
    fn end(&self) -> Result<(), i32> {
        match unsafe { imparo_cuda_end() } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn flush_layers(&self, decode: bool) -> u32 {
        // Stream-queued kernels need no cb-length policy; a registry knob later.
        let _ = decode;
        7
    }
    fn alloc(&self, id: BufId, bytes: u64) -> Result<(), i32> {
        match unsafe { imparo_cuda_alloc(b(id), bytes) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn arena(&self, bytes: u64) -> Result<(), i32> {
        match unsafe { imparo_cuda_arena(bytes) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn place(&self, id: BufId, offset: u64, bytes: u64) -> Result<(), i32> {
        match unsafe { imparo_cuda_place(b(id), offset, bytes) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn page_round(&self, n: u64) -> u64 {
        unsafe { imparo_cuda_page_round(n) }
    }
    fn alloc_kv(&self, bytes: &[u64]) -> Result<(), i32> {
        match unsafe { imparo_cuda_alloc_kv(bytes.len() as u32, bytes.as_ptr()) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn grow_kv(&self, bytes: &[u64]) -> Result<(), i32> {
        match unsafe { imparo_cuda_grow_kv(bytes.len() as u32, bytes.as_ptr()) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn write(&self, id: BufId, off: u64, src: &[f32]) {
        unsafe { imparo_cuda_write(b(id), off, src.as_ptr(), src.len() as u64) }
    }
    fn write_u32(&self, id: BufId, off: u64, src: &[u32]) {
        unsafe { imparo_cuda_write_u32(b(id), off, src.as_ptr(), src.len() as u64) }
    }
    fn read(&self, id: BufId, off: u64, dst: &mut [f32]) {
        unsafe { imparo_cuda_read(b(id), off, dst.as_mut_ptr(), dst.len() as u64) }
    }
    fn read_kv_bytes(&self, layer: u32, is_v: bool, off: u64, dst: &mut [u8]) {
        unsafe {
            imparo_cuda_read_kv(
                layer,
                u32::from(is_v),
                off,
                dst.as_mut_ptr(),
                dst.len() as u64,
            )
        }
        fn set_kv_page_table(&self, layer: u32, entries: &[u32]) {
            // TODO(cuda): upload the block table for the paged attention kernels.
            let _ = (layer, entries);
        }
        fn write_kv_bytes(&self, layer: u32, is_v: bool, off: u64, src: &[u8]) {
            // TODO(cuda): H2D staging into the layer's KV region.
            let _ = (layer, is_v, off, src);
        }
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
        unsafe {
            imparo_cuda_matmat(wkind, w_off, n_in, n_out, b(src), b(dst), n_tok, 0)
        }
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
        unsafe {
            imparo_cuda_matmat(
                wkind,
                w_off,
                n_in,
                n_out,
                b(src),
                b(dst),
                n_tok,
                src_row,
            )
        }
    }
    fn set_epilogue(&self, epi: imparo_backend::Epilogue) {
        unsafe { imparo_cuda_set_epilogue(epi as u32) }
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
        // The CUDA row kernel reads Q4_0 blocks. Refuse anything else rather than
        // unpack another layout with the wrong reader; Q8_0 weights are implemented on
        // Metal only so far.
        if wkind != 1 {
            eprintln!(
                "imparo cuda: row got weight kind {wkind}; this backend implements the \
                 Q4_0 embedding table only -- refusing the dispatch"
            );
            return;
        }
        unsafe { imparo_cuda_row(w_off, width, index, scale, b(dst), dst_off) }
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
        // NO_WEIGHT sentinel (u64::MAX) means normalize without a weight vector.
        let has_w = u32::from(w_off != u64::MAX);
        unsafe {
            imparo_cuda_rms_norm(
                b(buf),
                b(buf),
                w_off,
                width,
                eps,
                n_row,
                row_stride,
                base_off,
                has_w,
            )
        }
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
        let has_w = u32::from(w_off != u64::MAX);
        unsafe {
            imparo_cuda_rms_norm(
                b(buf),
                b(src),
                w_off,
                width,
                eps,
                n_row,
                row_stride,
                base_off,
                has_w,
            )
        }
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
        // TODO(cuda dev): freqs live on the host; upload once at init like the .mm
        // keeps them in a buffer. Passing the host pointer is a placeholder that a
        // real port replaces with a device-resident copy.
        let p = freqs.map_or(core::ptr::null(), <[f32]>::as_ptr);
        unsafe {
            imparo_cuda_rope(
                b(buf),
                n_rot,
                base,
                head_dim,
                n_heads,
                start_pos,
                n_tok,
                p,
            )
        }
    }
    fn hadamard(&self, buf: BufId, n: u32, nrot: u32) {
        unsafe { imparo_cuda_hadamard(b(buf), n, nrot) }
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
        unsafe {
            imparo_cuda_kv_store(
                b(src),
                layer,
                width,
                start_pos,
                n_tok,
                u32::from(is_v),
                ring,
            )
        }
    }
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
    ) {
        let _ = max_scores; // Metal's split-KV sizing hint; unused by the CUDA shape
        unsafe {
            imparo_cuda_attention(
                kv_layer,
                head_dim,
                n_heads,
                n_kv,
                kv_width,
                start_pos,
                window,
                n_tok,
                ring,
                b(BufId::Q),
                b(BufId::Attn),
                b(BufId::Kdq),
                b(BufId::Vdq),
            )
        }
    }
    fn gelu(&self, a: BufId, n: u32) {
        unsafe { imparo_cuda_gelu(b(a), n) }
    }
    fn gelu_mul(&self, a: BufId, bb: BufId, n: u32) {
        unsafe { imparo_cuda_gelu_mul(b(a), b(bb), n) }
    }
    fn add(&self, a: BufId, bb: BufId, n: u32) {
        unsafe { imparo_cuda_add(b(a), b(bb), n) }
    }
    fn add_scale(&self, a: BufId, bb: BufId, k: f32, n: u32) {
        unsafe { imparo_cuda_add_scale(b(a), b(bb), k, n) }
    }
    fn scale(&self, a: BufId, k: f32, n: u32) {
        unsafe { imparo_cuda_scale(b(a), k, n) }
    }
    fn copy(&self, dst: BufId, src: BufId, n: u32) {
        unsafe { imparo_cuda_copy(b(dst), b(src), n) }
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
        unsafe {
            imparo_cuda_mul_strided(b(a), b(bb), n, b_off, b_stride, a_stride, n_tok)
        }
    }
    fn softcap(&self, a: BufId, cap: f32, n: u32) {
        unsafe { imparo_cuda_softcap(b(a), cap, n) }
    }
    fn argmax(&self, src: BufId, dst: BufId, n: u32) {
        unsafe { imparo_cuda_argmax(b(src), b(dst), n) }
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
        todo!(
            "cuda: gemma3n per-layer-embedding gather -- port imparo_ple_gather_combine"
        )
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
        // TWO kernels, as on Metal: the state advance must not begin until every output
        // has read the state it overwrites, and a kernel cannot barrier its own grid.
        todo!("cuda: LFM2 short convolution -- port imparo_shortconv{{,_state}}")
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
        // The same kernel as the advance, with separate in and out pointers.
        todo!("cuda: short-conv boundary snapshot -- port imparo_shortconv_state")
    }
    unsafe fn init_weights(&self, base: *const u8, len: u64) -> Result<(), i32> {
        // CUDA uploads the blob to device memory.
        let bytes = unsafe { core::slice::from_raw_parts(base, len as usize) };
        init(bytes)
    }
    fn set_kv_types(&self, k: u32, v: u32) {
        // TODO(cuda): the .cu has imparo_cuda_set_kv_types; wire it.
        let _ = (k, v);
    }
    fn prof_stats(&self) -> imparo_backend::ProfStats {
        imparo_backend::ProfStats::default()
    }
    fn prof_enable(&self, on: bool) {
        let _ = on;
    }
    fn allocated_bytes(&self) -> u64 {
        0
    }
    fn pool_caps(&self) -> imparo_backend::PoolCaps {
        use imparo_backend::Tier;
        imparo_backend::PoolCaps {
            // PLACEHOLDER like the rest of this crate: re-derive from the real
            // CUDA kernels' tile geometry at bring-up.
            block_cells: 64,
            paged_reads: false,
            shared_address: false,
            tiers: &[Tier::Device, Tier::Host, Tier::Disk],
        }
    }
    fn device_tag(&self) -> String {
        "cuda".to_string()
    }
}
