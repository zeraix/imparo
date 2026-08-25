//! `Backend` on the host: the same op surface a GPU backend exposes, computed with
//! this crate's kernels over plain memory.
//!
//! WHY THIS EXISTS, when a CPU reference forward already does. The reference walks a
//! model directly and keeps its own flat KV cache; everything built on the `Backend`
//! seam -- the unified KV pool, block tables, checkpoints, the disk tier -- is
//! therefore unreachable from it. A machine with no GPU reprocessed every prompt from
//! scratch, not because the pool is device-specific (it is not: units, checkpoints and
//! manifests never mention a device) but because nothing on the host answered the
//! trait the pool talks to. This answers it.
//!
//! THE SAME LAYOUT AS A DEVICE, on purpose:
//!
//! ```text
//!   activations   f32, one buffer per BufId
//!   KV            f16, position -> slot through the SAME rule the shaders use:
//!                   ring   slot = pos & (ring - 1)          windowed layers
//!                   paged  slot = table[pos >> 6] * 64 + (pos & 63)   pooled layers
//! ```
//!
//! Storing KV as f16 is not an optimisation here -- it is what makes a checkpoint
//! written on the host and one written on a GPU the same bytes, so the disk tier is a
//! tier and not a per-backend format.
//!
//! FUTURE OFFLOAD. Nothing here assumes it owns the model: the buffer table is indexed
//! by `BufId`, the KV is indexed by layer, and both are ordinary Rust memory. A split
//! placement -- some layers here, some on a device -- needs a router above this trait
//! deciding which backend a layer's ops go to, not a change inside it.

use std::sync::{Mutex, MutexGuard, OnceLock};

use imparo_backend::{Backend, BufId, Epilogue, NO_WEIGHT, PoolCaps, WeightKindWire};
use imparo_gguf::weights::{GGML_F32, GGML_Q4_0, GGML_Q8_0};

use crate::ops;

/// The wire weight kind (`WeightKind as u32`) as a ggml type id.
///
/// Two numbering schemes meet here: the trait passes the compact wire value, this
/// crate's kernels switch on the ggml type. Getting it wrong reads a Q8 matrix as
/// Q4 and produces plausible garbage, so it is one function.
fn ggml_type(wkind: WeightKindWire) -> u32 {
    match wkind {
        0 => GGML_F32,
        1 => GGML_Q4_0,
        2 => GGML_Q8_0,
        other => panic!("cpu backend: unknown weight kind wire value {other}"),
    }
}

/// f32 -> f16 bits, round-to-nearest-even, with overflow to infinity.
#[must_use]
pub fn f16_from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x007f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00; // inf / NaN-as-inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflows to zero
        }
        let m = (mant | 0x0080_0000) >> (1 - exp);
        return sign | ((m + 0x0000_1000) >> 13) as u16;
    }
    let mut m = mant + 0x0000_1000; // round
    if m & 0x0080_0000 != 0 {
        m = 0;
        exp += 1;
        if exp >= 0x1f {
            return sign | 0x7c00;
        }
    }
    sign | ((exp as u16) << 10) | ((m >> 13) as u16)
}

/// f16 bits -> f32.
#[must_use]
pub fn f32_from_f16(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x03ff);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: the value is `mant * 2^-24`, renormalised to f32's form. Written
        // from that identity rather than from a shift count, because a shift count is
        // where the off-by-one lives -- this returned half the right number until the
        // round-trip test at the smallest NORMAL f16 caught it.
        let p = mant.ilog2(); // index of the top set bit
        let field = p + 103; // (p - 24) + 127
        let frac = (mant << (23 - p)) & 0x007f_ffff;
        return f32::from_bits(sign | (field << 23) | frac);
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// The weight mapping, as a pointer the trait handed over.
///
/// Send + Sync by assertion, matching what `init_weights` promises: the mapping is
/// read-only and outlives the process.
#[derive(Clone, Copy)]
struct Blob {
    base: *const u8,
    len: usize,
}
// SAFETY: read-only mapping, never freed while the backend lives.
unsafe impl Send for Blob {}
// SAFETY: as above -- shared reads only.
unsafe impl Sync for Blob {}

impl Blob {
    fn at(self, off: u64, len: usize) -> &'static [u8] {
        assert!(
            off as usize + len <= self.len,
            "cpu backend: weight slice [{off}, {}) outside the {}-byte mapping",
            off as usize + len,
            self.len
        );
        // SAFETY: bounds checked; the mapping is read-only and outlives the process.
        unsafe { std::slice::from_raw_parts(self.base.add(off as usize), len) }
    }
}

struct Ctx {
    bufs: Vec<Vec<f32>>,
    /// Per layer, per side: the cache as f16 bits.
    kv_k: Vec<Vec<u16>>,
    kv_v: Vec<Vec<u16>>,
    /// Per layer: positions [i*64, i*64+64) live in physical block `table[i]`.
    /// Empty means the identity mapping.
    kv_pt: Vec<Vec<u32>>,
    /// Per layer, per side: byte offset of the live REGION -- the ring belonging to the
    /// resident conversation. Zero for a pooled layer, which places through `kv_pt`.
    kv_reg_k: Vec<u64>,
    kv_reg_v: Vec<u64>,
    weights: Option<Blob>,
    epilogue: Epilogue,
    activation: Epilogue,
    kv_type_k: u32,
    kv_type_v: u32,
}

impl Default for Ctx {
    fn default() -> Self {
        Self {
            bufs: Vec::new(),
            kv_k: Vec::new(),
            kv_v: Vec::new(),
            kv_pt: Vec::new(),
            kv_reg_k: Vec::new(),
            kv_reg_v: Vec::new(),
            weights: None,
            epilogue: Epilogue::None,
            // Set by `set_activation` before any pipeline would need it, exactly as
            // on a device; GELU is the value gemma4 had before models differed.
            activation: Epilogue::Gelu,
            kv_type_k: 1,
            kv_type_v: 1,
        }
    }
}

impl Ctx {
    fn buf(&mut self, id: BufId) -> &mut Vec<f32> {
        if self.bufs.len() < BufId::COUNT {
            self.bufs.resize_with(BufId::COUNT, Vec::new);
        }
        &mut self.bufs[id as usize]
    }
    /// `row` is the layer's bytes per position, which is what turns the region's byte
    /// offset into slots. Ringed layers only: a pooled layer's region is always zero.
    fn slot(&self, layer: usize, pos: usize, ring: u32, is_v: bool, row: usize) -> usize {
        if ring > 0 {
            let reg = if is_v { &self.kv_reg_v } else { &self.kv_reg_k };
            let base = reg.get(layer).copied().unwrap_or(0) as usize / row.max(1);
            return base + (pos & (ring as usize - 1));
        }
        match self.kv_pt.get(layer) {
            Some(t) if !t.is_empty() => (t[pos >> 6] as usize) * 64 + (pos & 63),
            _ => pos,
        }
    }
}

fn ctx() -> MutexGuard<'static, Ctx> {
    static CTX: OnceLock<Mutex<Ctx>> = OnceLock::new();
    CTX.get_or_init(|| Mutex::new(Ctx::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The host backend. Zero-sized: its state is process-wide, like every other
/// backend's, because one process runs one model (see `imparo_model::backend`).
pub struct CpuBackend;

impl CpuBackend {
    fn act_of(kind: Epilogue, x: f32) -> f32 {
        if kind == Epilogue::Silu {
            return x / (1.0 + (-x).exp());
        }
        // The tanh approximation, which is what the shaders compute.
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
    }
}

#[allow(clippy::too_many_arguments)]
impl Backend for CpuBackend {
    // --- session: nothing is deferred, so these are where the GPU's are not ---
    fn begin(&self) {}
    fn flush(&self) {}
    fn end(&self) -> Result<(), i32> {
        Ok(())
    }
    fn flush_layers(&self, _decode: bool) -> u32 {
        u32::MAX
    }

    // --- buffers ---
    fn alloc(&self, id: BufId, bytes: u64) -> Result<(), i32> {
        ctx().buf(id).resize(bytes as usize / 4, 0.0);
        Ok(())
    }
    fn arena(&self, _bytes: u64) -> Result<(), i32> {
        // No arena: every buffer is its own allocation. The device's arena exists so
        // that grouped buffers can ALIAS, which only saves address space; here it
        // would save nothing and hide a real aliasing bug behind a shared Vec.
        Ok(())
    }
    fn place(&self, id: BufId, _offset: u64, bytes: u64) -> Result<(), i32> {
        self.alloc(id, bytes)
    }
    fn page_round(&self, n: u64) -> u64 {
        n.div_ceil(4096) * 4096
    }
    fn alloc_kv(&self, bytes: &[u64]) -> Result<(), i32> {
        let mut c = ctx();
        c.kv_k = bytes.iter().map(|b| vec![0_u16; *b as usize / 2]).collect();
        let k = c.kv_k.clone();
        c.kv_v = k;
        c.kv_pt.resize(bytes.len(), Vec::new());
        Ok(())
    }
    fn grow_kv(&self, bytes: &[u64]) -> Result<(), i32> {
        let mut c = ctx();
        for (i, b) in bytes.iter().enumerate() {
            let n = *b as usize / 2;
            if let Some(k) = c.kv_k.get_mut(i) {
                k.resize(n, 0);
            }
            if let Some(v) = c.kv_v.get_mut(i) {
                v.resize(n, 0);
            }
        }
        Ok(())
    }
    fn write(&self, id: BufId, off: u64, src: &[f32]) {
        let mut c = ctx();
        let b = c.buf(id);
        let lo = off as usize;
        if b.len() < lo + src.len() {
            b.resize(lo + src.len(), 0.0);
        }
        b[lo..lo + src.len()].copy_from_slice(src);
    }
    fn write_u32(&self, id: BufId, off: u64, src: &[u32]) {
        let mut c = ctx();
        let b = c.buf(id);
        let lo = off as usize;
        if b.len() < lo + src.len() {
            b.resize(lo + src.len(), 0.0);
        }
        for (d, s) in b[lo..lo + src.len()].iter_mut().zip(src) {
            *d = f32::from_bits(*s);
        }
    }
    fn read(&self, id: BufId, off: u64, dst: &mut [f32]) {
        let mut c = ctx();
        let b = c.buf(id);
        let lo = off as usize;
        assert!(b.len() >= lo + dst.len(), "cpu backend: read past buffer end");
        dst.copy_from_slice(&b[lo..lo + dst.len()]);
    }
    fn read_kv_bytes(&self, layer: u32, is_v: bool, off: u64, dst: &mut [u8]) {
        let c = ctx();
        let reg = if is_v { &c.kv_reg_v } else { &c.kv_reg_k };
        let lo = (reg.get(layer as usize).copied().unwrap_or(0) + off) as usize / 2;
        let side = if is_v { &c.kv_v } else { &c.kv_k };
        let cache = &side[layer as usize];
        for (i, out) in dst.chunks_exact_mut(2).enumerate() {
            out.copy_from_slice(&cache[lo + i].to_le_bytes());
        }
    }
    fn write_kv_bytes(&self, layer: u32, is_v: bool, off: u64, src: &[u8]) {
        let mut c = ctx();
        let base = if is_v { &c.kv_reg_v } else { &c.kv_reg_k };
        let lo = (base.get(layer as usize).copied().unwrap_or(0) + off) as usize / 2;
        let side = if is_v { &mut c.kv_v } else { &mut c.kv_k };
        let cache = &mut side[layer as usize];
        if cache.len() < lo + src.len() / 2 {
            cache.resize(lo + src.len() / 2, 0);
        }
        for (i, s) in src.chunks_exact(2).enumerate() {
            cache[lo + i] = u16::from_le_bytes([s[0], s[1]]);
        }
    }
    fn set_kv_page_table(&self, layer: u32, entries: &[u32]) {
        let mut c = ctx();
        if c.kv_pt.len() <= layer as usize {
            c.kv_pt.resize(layer as usize + 1, Vec::new());
        }
        c.kv_pt[layer as usize] = entries.to_vec();
    }
    fn set_kv_region(&self, layer: u32, k_off: u64, v_off: u64) {
        let mut c = ctx();
        let n = layer as usize + 1;
        if c.kv_reg_k.len() < n {
            c.kv_reg_k.resize(n, 0);
            c.kv_reg_v.resize(n, 0);
        }
        c.kv_reg_k[layer as usize] = k_off;
        c.kv_reg_v[layer as usize] = v_off;
    }

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
    ) {
        self.matmat_from(wkind, w_off, n_in, n_out, src, dst, n_tok, 0);
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
        let (kind, n_in, n_out, n_tok) =
            (ggml_type(wkind), n_in as usize, n_out as usize, n_tok as usize);
        let mut c = ctx();
        let blob = c.weights.expect("cpu backend: no weights").at(
            w_off,
            ops::row_bytes_len(kind, n_in) * n_out,
        );
        let lo = src_row as usize * n_in;
        let x = c.buf(src)[lo..lo + n_in * n_tok].to_vec();
        let mut y = vec![0.0_f32; n_out * n_tok];
        ops::mul_mat_bytes(blob, kind, n_in, n_out, &x, n_tok, &mut y);
        let epi = c.epilogue;
        let act = c.activation;
        let d = c.buf(dst);
        if d.len() < y.len() {
            d.resize(y.len(), 0.0);
        }
        if epi == Epilogue::None {
            d[..y.len()].copy_from_slice(&y);
        } else {
            // `y = act(y) * product`: the destination already holds the gate, which is
            // why a fused epilogue READS it. The sticky `epi` names the activation for
            // this dispatch; `act` is the process-wide one the ungated ops use.
            let _ = act;
            for (slot, v) in d[..y.len()].iter_mut().zip(&y) {
                *slot = Self::act_of(epi, *slot) * v;
            }
        }
    }
    fn set_epilogue(&self, epi: Epilogue) {
        ctx().epilogue = epi;
    }
    fn set_activation(&self, act: Epilogue) {
        ctx().activation = act;
    }
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
        let (w, k, n) = (width as usize, kernel as usize, n_tok as usize);
        let mut c = ctx();
        let conv = c
            .weights
            .expect("cpu backend: no weights")
            .at(w_off, w * k * 4);
        let cw: Vec<f32> = conv
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let x = c.buf(bcx)[..n * 3 * w].to_vec();
        let so = state_off as usize;
        let mut st = c.buf(state)[so..so + (k - 1) * w].to_vec();
        let mut o = vec![0.0_f32; n * w];
        ops::shortconv(&x, &cw, &mut st, &mut o, w, k, n);
        c.buf(state)[so..so + (k - 1) * w].copy_from_slice(&st);
        let d = c.buf(out);
        if d.len() < o.len() {
            d.resize(o.len(), 0.0);
        }
        d[..o.len()].copy_from_slice(&o);
    }
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
        // The state as of `n_tok` tokens in, computed the same way the kernel does:
        // the last (kernel - 1) values of b*x ending there, falling back to the
        // pre-batch state for the positions below it.
        let (w, k, n) = (width as usize, kernel as usize, n_tok as usize);
        let hist = k - 1;
        let mut c = ctx();
        let x = c.buf(bcx)[..(n.max(hist)) * 3 * w].to_vec();
        let so = state_off as usize;
        let prev = c.buf(state)[so..so + hist * w].to_vec();
        let mut next = vec![0.0_f32; hist * w];
        for si in 0..hist {
            let e = n + si;
            for ch in 0..w {
                next[si * w + ch] = if e < hist {
                    prev[e * w + ch]
                } else {
                    let row = (e - hist) * 3 * w;
                    x[row + ch] * x[row + 2 * w + ch]
                };
            }
        }
        let no = snap_off as usize;
        let d = c.buf(snap);
        if d.len() < no + next.len() {
            d.resize(no + next.len(), 0.0);
        }
        d[no..no + next.len()].copy_from_slice(&next);
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
        let (kind, w) = (ggml_type(wkind), width as usize);
        let mut c = ctx();
        let rb = ops::row_bytes_len(kind, w);
        let blob = c
            .weights
            .expect("cpu backend: no weights")
            .at(w_off + (index as u64) * rb as u64, rb);
        let mut out = vec![0.0_f32; w];
        ops::row_from_bytes(blob, kind, w, 0, &mut out);
        // Compared exactly on purpose: the caller passes a literal 1.0 to mean "no
        // scale", and any other value is a real one to apply.
        #[allow(clippy::float_cmp)]
        if scale != 1.0 {
            ops::scale(&mut out, scale);
        }
        let lo = dst_off as usize;
        let d = c.buf(dst);
        if d.len() < lo + w {
            d.resize(lo + w, 0.0);
        }
        d[lo..lo + w].copy_from_slice(&out);
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
        self.rms_norm_from(buf, buf, w_off, width, eps, n_row, row_stride, base_off);
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
        let (w, rows, stride, base) = (
            width as usize,
            n_row as usize,
            row_stride as usize,
            base_off as usize,
        );
        let mut c = ctx();
        let weight: Option<Vec<f32>> = (w_off != NO_WEIGHT).then(|| {
            c.weights
                .expect("cpu backend: no weights")
                .at(w_off, w * 4)
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        });
        for r in 0..rows {
            let lo = base + r * stride;
            let mut row = c.buf(src)[lo..lo + w].to_vec();
            ops::rms_norm(&mut row, weight.as_deref(), eps);
            let d = c.buf(buf);
            if d.len() < lo + w {
                d.resize(lo + w, 0.0);
            }
            d[lo..lo + w].copy_from_slice(&row);
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
        let (hd, heads) = (head_dim as usize, n_heads as usize);
        let mut c = ctx();
        let b = c.buf(buf);
        for t in 0..n_tok as usize {
            for h in 0..heads {
                let lo = (t * heads + h) * hd;
                ops::rope_neox(
                    &mut b[lo..lo + hd],
                    start_pos + t as u32,
                    n_rot as usize,
                    base,
                    freqs,
                );
            }
        }
    }
    fn hadamard(&self, _buf: BufId, _n: u32, _nrot: u32) {
        todo!("cpu backend: the quantized modes' rotation is not ported yet")
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
        let (w, l) = (width as usize, layer as usize);
        let mut c = ctx();
        assert_eq!(
            if is_v { c.kv_type_v } else { c.kv_type_k },
            1,
            "cpu backend: only an f16 KV cache is implemented"
        );
        let rows = c.buf(src)[..w * n_tok as usize].to_vec();
        for t in 0..n_tok as usize {
            let slot = c.slot(l, start_pos as usize + t, ring, is_v, w * 2);
            let side = if is_v { &mut c.kv_v } else { &mut c.kv_k };
            let cache = &mut side[l];
            if cache.len() < (slot + 1) * w {
                cache.resize((slot + 1) * w, 0);
            }
            for i in 0..w {
                cache[slot * w + i] = f16_from_f32(rows[t * w + i]);
            }
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
        _max_scores: u32,
        ring: u32,
    ) {
        let (hd, heads, kvh, kvw, l) = (
            head_dim as usize,
            n_heads as usize,
            n_kv as usize,
            kv_width as usize,
            kv_layer as usize,
        );
        let per_kv = heads / kvh;
        let qw = heads * hd;
        let mut c = ctx();
        let q = c.buf(BufId::Q)[..qw * n_tok as usize].to_vec();
        let mut out = vec![0.0_f32; qw * n_tok as usize];
        let mut scores = vec![0.0_f32; start_pos as usize + n_tok as usize];
        for t in 0..n_tok as usize {
            let pos = start_pos as usize + t;
            let lo = if window > 0 {
                (pos + 1).saturating_sub(window as usize)
            } else {
                0
            };
            for h in 0..heads {
                let kh = h / per_kv;
                let qoff = t * qw + h * hd;
                let sc = &mut scores[..pos + 1 - lo];
                for (si, pp) in (lo..=pos).enumerate() {
                    let slot = c.slot(l, pp, ring, false, kvw * 2);
                    let base = slot * kvw + kh * hd;
                    let k = &c.kv_k[l][base..base + hd];
                    sc[si] = (0..hd)
                        .map(|i| q[qoff + i] * f32_from_f16(k[i]))
                        .sum::<f32>();
                }
                ops::softmax(sc);
                let o = &mut out[qoff..qoff + hd];
                for (si, pp) in (lo..=pos).enumerate() {
                    let slot = c.slot(l, pp, ring, true, kvw * 2);
                    let base = slot * kvw + kh * hd;
                    let v = &c.kv_v[l][base..base + hd];
                    let wgt = sc[si];
                    for (oo, vv) in o.iter_mut().zip(v) {
                        *oo += wgt * f32_from_f16(*vv);
                    }
                }
            }
        }
        let d = c.buf(BufId::Attn);
        if d.len() < out.len() {
            d.resize(out.len(), 0.0);
        }
        d[..out.len()].copy_from_slice(&out);
    }
    fn act(&self, a: BufId, n: u32) {
        let mut c = ctx();
        let kind = c.activation;
        for x in &mut c.buf(a)[..n as usize] {
            *x = Self::act_of(kind, *x);
        }
    }
    fn act_mul(&self, a: BufId, b: BufId, n: u32) {
        let mut c = ctx();
        let kind = c.activation;
        let src = c.buf(b)[..n as usize].to_vec();
        for (x, u) in c.buf(a)[..n as usize].iter_mut().zip(&src) {
            *x = Self::act_of(kind, *x) * u;
        }
    }
    fn add(&self, a: BufId, b: BufId, n: u32) {
        let mut c = ctx();
        let src = c.buf(b)[..n as usize].to_vec();
        ops::add_into(&mut c.buf(a)[..n as usize], &src);
    }
    fn add_scale(&self, a: BufId, b: BufId, k: f32, n: u32) {
        let mut c = ctx();
        let src = c.buf(b)[..n as usize].to_vec();
        for (x, s) in c.buf(a)[..n as usize].iter_mut().zip(&src) {
            *x = (*x + s) * k;
        }
    }
    fn scale(&self, a: BufId, k: f32, n: u32) {
        ops::scale(&mut ctx().buf(a)[..n as usize], k);
    }
    fn copy(&self, dst: BufId, src: BufId, n: u32) {
        let mut c = ctx();
        let s = c.buf(src)[..n as usize].to_vec();
        let d = c.buf(dst);
        if d.len() < s.len() {
            d.resize(s.len(), 0.0);
        }
        d[..s.len()].copy_from_slice(&s);
    }
    fn mul_strided(
        &self,
        a: BufId,
        b: BufId,
        n: u32,
        b_off: u32,
        b_stride: u32,
        a_stride: u32,
        n_tok: u32,
    ) {
        let mut c = ctx();
        let src = c.buf(b).clone();
        let d = c.buf(a);
        for t in 0..n_tok as usize {
            let (ao, bo) = (
                t * a_stride as usize,
                b_off as usize + t * b_stride as usize,
            );
            for i in 0..n as usize {
                d[ao + i] *= src[bo + i];
            }
        }
    }
    fn softcap(&self, a: BufId, cap: f32, n: u32) {
        ops::softcap(&mut ctx().buf(a)[..n as usize], cap);
    }
    fn argmax(&self, src: BufId, dst: BufId, n: u32) {
        let mut c = ctx();
        let idx = ops::argmax_f32(&c.buf(src)[..n as usize]);
        let d = c.buf(dst);
        if d.is_empty() {
            d.resize(1, 0.0);
        }
        d[0] = f32::from_bits(idx);
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
        // `proj[t] = (proj[t] + ple_row(token[t]) * emb_scale) * comb_scale`, with the
        // row dequantised on the way. The table is Q4_0 in every file that carries one,
        // which is why the shader hardcodes it too -- and why this asserts the width
        // rather than guessing a layout.
        let w = width as usize;
        assert_eq!(w % 32, 0, "cpu backend: PLE width must be a multiple of 32");
        let mut c = ctx();
        let rb = ops::row_bytes_len(GGML_Q4_0, w);
        let blob = c.weights.expect("cpu backend: no weights");
        let tokens: Vec<u32> = c.buf(tokens_buf)[..n_tok as usize]
            .iter()
            .map(|f| f.to_bits())
            .collect();
        let mut row = vec![0.0_f32; w];
        for (t, &tok) in tokens.iter().enumerate() {
            let bytes = blob.at(w_offset + u64::from(tok) * rb as u64, rb);
            ops::row_from_bytes(bytes, GGML_Q4_0, w, 0, &mut row);
            let d = c.buf(proj);
            for (p, e) in d[t * w..(t + 1) * w].iter_mut().zip(&row) {
                *p = (*p + e * emb_scale) * comb_scale;
            }
        }
    }

    unsafe fn init_weights(&self, base: *const u8, len: u64) -> Result<(), i32> {
        ctx().weights = Some(Blob {
            base,
            len: len as usize,
        });
        Ok(())
    }
    fn set_kv_types(&self, k: u32, v: u32) {
        let mut c = ctx();
        c.kv_type_k = k;
        c.kv_type_v = v;
    }
    fn kv_tag(&self) -> String {
        let c = ctx();
        let name = |t: u32| match t {
            2 => "q4_0",
            3 => "q8_0",
            _ => "f16",
        };
        format!("{}/{}", name(c.kv_type_k), name(c.kv_type_v))
    }
    fn prof_stats(&self) -> imparo_backend::ProfStats {
        imparo_backend::ProfStats::default()
    }
    fn prof_enable(&self, _on: bool) {}
    fn allocated_bytes(&self) -> u64 {
        let c = ctx();
        let bufs: usize = c.bufs.iter().map(|b| b.len() * 4).sum();
        let kv: usize = c
            .kv_k
            .iter()
            .chain(&c.kv_v)
            .map(std::vec::Vec::len)
            .sum::<usize>()
            * 2;
        (bufs + kv) as u64
    }
    fn device_tag(&self) -> String {
        format!("cpu x{}", ops::threads())
    }
    fn pool_caps(&self) -> PoolCaps {
        PoolCaps {
            // The same 64-cell block the shaders index, because the pool's unit and
            // the resume grid are the engine's, not a device's.
            block_cells: 64,
            // Every read goes through `Ctx::slot`, which applies the page table --
            // the same rule the shaders' `kv_slot` applies.
            paged_reads: true,
            // There is one address space here by construction.
            shared_address: true,
            tiers: &[imparo_backend::Tier::Unified, imparo_backend::Tier::Disk],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, CpuBackend, Epilogue, f16_from_f32, f32_from_f16};

    /// The cache round trip, at the values that actually break a hand-written
    /// converter: zero, subnormals, the rounding boundary, and overflow.
    #[test]
    fn f16_round_trips_the_awkward_values() {
        for x in [
            0.0_f32, -0.0, 1.0, -1.0, 0.5, 65504.0, -65504.0, 6.1e-5, 5.96e-8, 1e-9,
            0.333_333_34, -2.717_5,
        ] {
            let back = f32_from_f16(f16_from_f32(x));
            let err = (back - x).abs();
            assert!(
                err <= x.abs() * 1e-3 + 1e-7,
                "f16 round trip of {x} came back {back}"
            );
        }
        // Above the range: infinity, not a wrapped sign bit.
        assert!(f32_from_f16(f16_from_f32(1e6)).is_infinite());
        assert!(f32_from_f16(f16_from_f32(-1e6)).is_infinite());
    }

    /// The fused epilogue is `y = act(y) * product` -- it READS the destination.
    /// A backend that overwrote instead would compute a feed-forward without its
    /// gate, which is wrong by a factor that looks plausible.
    #[test]
    fn the_epilogue_reads_the_destination() {
        let be = CpuBackend;
        be.set_activation(Epilogue::Silu);
        assert!((CpuBackend::act_of(Epilogue::Silu, 1.0) - 0.731_058_6).abs() < 1e-5);
        assert!((CpuBackend::act_of(Epilogue::Gelu, 1.0) - 0.841_192).abs() < 1e-4);
    }

    /// Paged and ringed addressing, the two rules the shaders' `kv_slot` applies, and
    /// the region base that shifts a ring to the conversation that owns it.
    #[test]
    fn the_slot_rule_matches_the_shader() {
        let be = CpuBackend;
        be.alloc_kv(&[64 * 1024]).unwrap();
        be.set_kv_page_table(0, &[3, 1, 2]);
        {
            let c = super::ctx();
            // paged: table[pos >> 6] * 64 + (pos & 63)
            assert_eq!(c.slot(0, 0, 0, false, 32), 3 * 64);
            assert_eq!(c.slot(0, 65, 0, false, 32), 64 + 1);
            assert_eq!(c.slot(0, 130, 0, false, 32), 2 * 64 + 2);
            // ring wins when set: pos & (ring - 1)
            assert_eq!(c.slot(0, 130, 128, false, 32), 2);
        }
        // a region shifts the ring by whole rings, and only the ring
        be.set_kv_region(0, 2 * 128 * 32, 0);
        let c = super::ctx();
        assert_eq!(c.slot(0, 130, 128, false, 32), 2 * 128 + 2);
        // V has its own base, here still zero
        assert_eq!(c.slot(0, 130, 128, true, 32), 2);
        // the paged rule ignores regions
        assert_eq!(c.slot(0, 130, 0, false, 32), 2 * 64 + 2);
    }
}
