//! Model-agnostic KV-pool glue at the crate level: everything a model needs to
//! join the pool that is derivable from the SHARED plan types. A model module
//! contributes only its geometry descriptor (see gemma4's `kv_state_geometry`).

// Engine-global KV cache-type configuration: which storage type the K and V
// caches use. Not a model property (every model reads the same config), so it
// lives at the crate level; models import it from here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum KvType {
    F16,
    Q4_0,
    Q8_0,
}

pub(crate) static KV_CFG_K: std::sync::OnceLock<KvType> = std::sync::OnceLock::new();
pub(crate) static KV_CFG_V: std::sync::OnceLock<KvType> = std::sync::OnceLock::new();

impl KvType {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "f16" => Some(Self::F16),
            "q4_0" => Some(Self::Q4_0),
            "q8_0" => Some(Self::Q8_0),
            _ => None,
        }
    }
    fn from_env(var: &str) -> Self {
        match std::env::var(var).as_deref() {
            Ok("q4_0") => Self::Q4_0,
            Ok("q8_0") => Self::Q8_0,
            Ok("f16") | Err(_) => Self::F16,
            Ok(other) => {
                eprintln!("[imparo] unknown {var}={other}; using f16");
                Self::F16
            }
        }
    }
    pub(crate) fn k() -> Self {
        *KV_CFG_K.get_or_init(|| Self::from_env("IMPARO_CTK"))
    }
    pub(crate) fn v() -> Self {
        *KV_CFG_V.get_or_init(|| Self::from_env("IMPARO_CTV"))
    }
    /// Bytes one cache row of `width` values takes in this type.
    pub(crate) fn row_bytes(self, width: usize) -> usize {
        debug_assert_eq!(width % 32, 0);
        match self {
            Self::F16 => width * 2,
            Self::Q4_0 => width / 32 * 18,
            Self::Q8_0 => width / 32 * 34,
        }
    }
    pub(crate) fn ggml_id(self) -> u32 {
        match self {
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
        }
    }
}
/// The configuration root for this plan + KV types: every input that shapes the
/// cache bytes, and nothing else (no tuned value may enter -- see the design's
/// identity rule). `model_digest` is the caller's cheap, stable model identity.
#[must_use]
pub fn config_root(
    plan: &crate::ModelPlan,
    model_digest: &[u8],
) -> imparo_kv::ConfigRoot {
    use crate::Attention;
    let c = &plan.config;
    let mut geom = Vec::new();
    geom.extend_from_slice(&c.n_kv_heads.to_le_bytes());
    geom.extend_from_slice(&c.n_heads.to_le_bytes());
    geom.extend_from_slice(&c.n_layers.to_le_bytes());
    for l in &plan.layers {
        geom.extend_from_slice(&l.index.to_le_bytes());
        match l.attention {
            Attention::Full {
                head_dim,
                rope_base,
                rope_dim,
            } => {
                geom.push(0);
                geom.extend_from_slice(&head_dim.to_le_bytes());
                geom.extend_from_slice(&rope_base.to_le_bytes());
                geom.extend_from_slice(&rope_dim.to_le_bytes());
            }
            Attention::Window {
                head_dim,
                rope_base,
                rope_dim,
                window,
            } => {
                geom.push(1);
                geom.extend_from_slice(&head_dim.to_le_bytes());
                geom.extend_from_slice(&rope_base.to_le_bytes());
                geom.extend_from_slice(&rope_dim.to_le_bytes());
                geom.extend_from_slice(&window.to_le_bytes());
            }
            // ITS OWN DISCRIMINANT, and the state SIZES with it. A recurrent block holds
            // state whose shape is nothing like a KV row, so two models that differ only
            // in where their recurrent blocks sit, or in how much state those blocks
            // carry, must not hash the same.
            Attention::Recurrent { r_elems, s_elems } => {
                geom.push(2);
                geom.extend_from_slice(&r_elems.to_le_bytes());
                geom.extend_from_slice(&s_elems.to_le_bytes());
            }
        }
        match l.kv_source {
            crate::KvSource::Own => geom.push(0xFF),
            crate::KvSource::SharedWith(s) => {
                geom.push(0xFE);
                geom.extend_from_slice(&s.to_le_bytes());
            }
        }
    }
    imparo_kv::ConfigRoot::new(
        model_digest,
        (KvType::k().ggml_id(), KvType::v().ggml_id()),
        &geom,
    )
}

/// Cheap, stable model identity for the configuration root: sha256 of the header
/// region (metadata + tensor table live at the front) plus the file length.
/// Hashing 4 GB per launch is not acceptable; the header pins architecture, quant
/// layout and vocab.
///
/// # Errors
/// Returns an error when the file cannot be read.
pub fn model_digest(path: &std::path::Path) -> Result<Vec<u8>, String> {
    use sha2::Digest as _;
    use std::io::Read as _;
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let len = f.metadata().map_err(|e| e.to_string())?.len();
    let mut head = vec![0u8; 1 << 20];
    let n = f.read(&mut head).map_err(|e| e.to_string())?;
    head.truncate(n);
    let mut out = len.to_le_bytes().to_vec();
    let mut h = sha2::Sha256::new();
    h.update(&head);
    let d: [u8; 32] = h.finalize().into();
    out.extend_from_slice(&d);
    Ok(out)
}

// ---- POOL GEOMETRY, DERIVED FROM THE PLAN ------------------------------------------
//
// These lived in gemma4/workflow_gpu.rs while gemma4 was the only model. Every input is a
// shared plan type, so by this module's own charter they belong here -- and the second
// model is when that stops being a matter of taste: LFM2 needs the same ring rule and the
// same byte budget over a different mix of block kinds, so a copy would have to be kept in
// step by hand.

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

/// Slots the first allocation covers. Growth is by `kv_round` from here.
pub(crate) const KV_FIRST_SLOTS: usize = 512;

/// Growth granularity in slots.
///
/// One block of headroom keeps decode from growing every token. 64 rather than 256 because
/// the block is pure headroom on the full-attention layers: MEASURED on gemma4 E4B, whose
/// four full layers carry 16384 bytes per position between them, a 256-block rounds 5674
/// positions to 6144 slots and a 64-block to 5760 -- 384 slots, 6 MiB, for one extra grow
/// every 64 decode tokens instead of every 256.
const KV_BLOCK: usize = 64;

/// Round a position count up to the next growth block, with one block of headroom.
///
/// NOT doubling. `kv_fit` runs once per request with the whole prompt, so a request never
/// grows more than once and geometric growth buys nothing -- while rounding 5642 up to the
/// next power of two is 8192, the entire context, which saves nothing at all. A block plus
/// one block of headroom keeps decode from growing every token: at 4 layers x 4 KiB a
/// position, a grow copies about 96 MiB, which is a millisecond every block of tokens.
#[must_use]
pub(crate) fn kv_round(positions: usize) -> usize {
    (positions + KV_BLOCK).div_ceil(KV_BLOCK) * KV_BLOCK
}

/// Slots a windowed layer's ring needs, or 0 for a layer that is not ringed.
///
/// NOT `window`. A batch writes every one of its K/V rows before any attention runs, so the
/// ring has to hold the union of the windows of all queries in the batch:
///
///   query at start_pos          needs [start_pos + 1 - window, start_pos]
///   query at start_pos + b - 1  needs [start_pos + b - window,  start_pos + b - 1]
///   union                        = window + b - 1 positions
///
/// At window 512 and ubatch 512 a `window`-sized ring is exactly one batch too small: the
/// row for start_pos + 512 lands on slot (start_pos & 511), the very slot the batch's first
/// query still needs. llama.cpp sizes its SWA cache the same way, as n_swa + n_ubatch.
///
/// Rounded to a power of two because the mapping is a MASK, not a modulus: `pos % ring` put
/// an integer division in the attention position loop and took decode from 33 to 15.9
/// tok/s. A layer whose window is not a power of two gets 0 and full-context allocation.
#[must_use]
pub fn ring_slots(attention: crate::Attention, max_batch: usize) -> usize {
    match attention {
        crate::Attention::Window { window, .. } if window.is_power_of_two() => {
            (window as usize + max_batch.max(1) - 1).next_power_of_two()
        }
        _ => 0,
    }
}

/// How many windowed conversations may be live at once, each with its own ring.
///
/// A windowed layer keeps ONE ring, so the conversation that ran last owns it and every
/// switch captures the outgoing window and restores the incoming one. A ring per
/// conversation removes that, and is what makes two conversations in one forward
/// expressible at all. The extra rings are address space: the caches are zero-fill
/// allocations that commit a page on first write, measured at E4B as 38.1 -> 125.2 MiB
/// allocated with the resident footprint unchanged.
#[must_use]
pub fn window_regions() -> usize {
    std::env::var("IMPARO_KV_REGIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

/// The ring mask for a layer, or 0 when the layer is not ringed.
#[must_use]
pub fn ring_mask(attention: crate::Attention, max_batch: usize) -> u32 {
    match ring_slots(attention, max_batch) {
        0 => 0,
        slots => slots as u32 - 1,
    }
}

/// Bytes each layer's cache needs to hold `positions`, indexed by layer.
///
/// Zero for a layer that owns no cache: one that borrows another layer's KV, and one whose
/// block does not attend at all. A recurrent block owns state, but its size is fixed by the
/// conv taps rather than by the position count, so it is not sized here.
///
/// A windowed layer is capped by its ring whatever the context is. Clamping a ring to
/// `capacity` is safe even though the MASK still spans the full ring: a position can never
/// reach `capacity`, so when capacity is the smaller of the two the mask is the identity and
/// no slot is ever aliased.
#[must_use]
pub fn kv_bytes_for(
    plan: &crate::ModelPlan,
    positions: usize,
    capacity: usize,
    ring_batch: usize,
) -> Vec<u64> {
    let c = &plan.config;
    let regions = window_regions();
    let mut kv_bytes = vec![0_u64; c.n_layers as usize];
    for layer in &plan.layers {
        if layer.kv_source != crate::KvSource::Own || !layer.attention.is_attention() {
            continue;
        }
        let w = c.n_kv_heads as usize * layer.attention.head_dim() as usize;
        let slots = match ring_slots(layer.attention, ring_batch) {
            0 => kv_round(positions).min(capacity),
            // One ring PER RESIDENT CONVERSATION. The extra rings are address space:
            // the cache buffers are zero-fill allocations that commit a page on first
            // write, so a conversation that never fills its window never pays for it.
            r => r.min(capacity) * regions,
        };
        // Sized by the configured cache type. K and V share one size per layer (the C
        // allocator takes one array for both), so a MIXED config sizes at the larger of the
        // two -- exact for the shipped configs (f16/f16, q4_0/q4_0).
        let row = KvType::k().row_bytes(w).max(KvType::v().row_bytes(w));
        kv_bytes[layer.index as usize] = (row * slots) as u64;
    }
    kv_bytes
}

// ---- POOL PARTICIPATION, DERIVED FROM THE PLAN --------------------------------------
//
// The descriptor a model hands the common capture/restore engine, and the three questions
// asked of it. All four are derivations over shared types: a model's remaining cost of
// joining the pool is binding them to its own runtime state (how many slots it has grown,
// whether the GPU is ready), which is what stays in the workflow.

use imparo_backend::Backend;
use imparo_kv::{KvState, LayerStateGeom, StateKind};

/// Reads a device recurrent buffer as little-endian f32 bytes.
///
/// One place, because two would be two chances to disagree about the encoding. `from` is
/// `BufId::Recur` for the live state, `BufId::RecurSnap` for a boundary snapshot.
#[must_use]
pub fn read_recurrent(elems: usize, from: imparo_backend::BufId) -> Vec<u8> {
    imparo_kv::state::capture_recurrent(backend(), elems, from)
}

/// Per-layer capture geometry for the layers that own a growing-or-bounded cache.
///
/// A block that does not attend is excluded: its state is constant in context, so by the
/// design's own rule it is a per-conversation fixed-size allocation and not the pool's --
/// "Only state that grows with context belongs in the pool ... The pool is the
/// full-attention KV allocator" (docs/unified-kv-pool.md). A recurrent kind arrives with
/// the first model that carries one, as a whole-state snapshot.
#[must_use]
pub fn state_geometry(
    plan: &crate::ModelPlan,
    ring_batch: usize,
) -> Vec<LayerStateGeom> {
    let c = &plan.config;
    plan.layers
        .iter()
        .filter(|l| l.kv_source == crate::KvSource::Own && l.attention.is_attention())
        .map(|l| {
            let w = c.n_kv_heads as usize * l.attention.head_dim() as usize;
            let ring = ring_slots(l.attention, ring_batch);
            let kind = match (l.attention, ring) {
                (crate::Attention::Window { window, .. }, r) if r > 0 => {
                    StateKind::Window {
                        window: window as usize,
                        ring: r,
                    }
                }
                _ => StateKind::Full,
            };
            LayerStateGeom {
                layer: l.index,
                kind,
                k_stride: KvType::k().row_bytes(w),
                v_stride: KvType::v().row_bytes(w),
            }
        })
        .collect()
}

/// The pooled set: layers whose state grows with context.
#[must_use]
pub fn full_layers(geom: &[LayerStateGeom]) -> Vec<u32> {
    geom.iter()
        .filter(|g| matches!(g.kind, StateKind::Full))
        .map(|g| g.layer)
        .collect()
}

/// The bounded set, which is what gets captured rather than pooled.
#[must_use]
pub fn bounded_geometry(geom: &[LayerStateGeom]) -> Vec<LayerStateGeom> {
    geom.iter()
        .filter(|g| !matches!(g.kind, StateKind::Full))
        .copied()
        .collect()
}

/// The furthest position a resumed prefill may start at, given how much of the request is
/// already computed (`resident`) and how long the request is (`len`).
///
/// A resume is only worth anything if it reproduces what a cold prefill would have
/// produced, and two engine properties decide where that holds (both measured, and both
/// gated by dev_harness/kv_gates.py `engine`):
///
/// - 64 is the token-tile width. Resuming off that grid does not reproduce the cold pass
///   -- 500 and 130 have differed since the pool landed.
/// - One token is a DECODE shape: it takes the GEMV, not the staged prefill GEMM, so a
///   1-token tail differs where a 2-token tail is byte-equal (n=705 split@704 against
///   n=706, q4_0).
///
/// The tail's width does not otherwise matter. It used to: the wo projection staged half
/// only at >= 64 tokens, so a narrow resumed chunk took the float path where the cold
/// pass had staged half, and callers carried rules to avoid landing there. The floor is 2
/// now (HALF_A_MIN, imparo_metal.mm) and this is the whole rule.
#[must_use]
pub fn resume_point(resident: usize, len: usize) -> usize {
    resident.min(len.saturating_sub(2)) & !63
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

/// See the IMPARO_KV_SCAN call site.
/// Per-layer KV drift scan: RMS and the widest per-dimension magnitudes, which is how a
/// quantised cache's outliers are spotted. IMPARO_KV_SCAN gates the caller.
///
/// Generic over any model that owns KV -- the layer set, the widths and the ring clamp all
/// come from the plan, so a second model would otherwise copy 66 lines to say the same
/// thing about its own cache.
pub fn scan(be: &dyn Backend, plan: &crate::ModelPlan, positions: usize) {
    fn half_bits_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
        let exp = ((bits >> 10) & 0x1f) as i32;
        let man = (bits & 0x3ff) as f32;
        match exp {
            0 => sign * man * 2f32.powi(-24),
            31 => {
                if man == 0.0 {
                    sign * f32::INFINITY
                } else {
                    f32::NAN
                }
            }
            _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
        }
    }
    let c = &plan.config;
    for (li, layer) in plan.layers.iter().enumerate() {
        if !matches!(layer.kv_source, crate::KvSource::Own) {
            continue;
        }
        let hd = layer.attention.head_dim();
        let width = (c.n_kv_heads * hd) as usize;
        let slots = match layer.attention {
            crate::Attention::Window { .. } => positions.min(1024),
            crate::Attention::Full { .. } => positions,
            crate::Attention::Recurrent { .. } => {
                unreachable!("gemma4 layer {li} planned as a recurrent block")
            }
        }
        .min(512);
        let mut bytes = vec![0u8; slots * width * 2];
        for (is_v, tag) in [(false, "K"), (true, "V")] {
            be.read_kv_bytes(li as u32, is_v, 0, &mut bytes);
            let vals: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|b| half_bits_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            let mut sq = 0.0f64;
            let mut dim_max = vec![0.0f32; width];
            for (i, &v) in vals.iter().enumerate() {
                sq += f64::from(v) * f64::from(v);
                let d = i % width;
                if v.abs() > dim_max[d] {
                    dim_max[d] = v.abs();
                }
            }
            let rms = (sq / vals.len() as f64).sqrt();
            let mut idx: Vec<usize> = (0..width).collect();
            idx.sort_by(|&a, &b| dim_max[b].partial_cmp(&dim_max[a]).unwrap());
            let med = dim_max[idx[width / 2]];
            let top: Vec<String> = idx
                .iter()
                .take(4)
                .map(|&d| format!("{d}:{:.1}", dim_max[d]))
                .collect();
            println!(
                "kvscan L{li:02} {tag} w={width} slots={slots} rms={rms:.3} \
                      med_dim_max={med:.2} top_dims {}",
                top.join(" ")
            );
        }
    }
}

/// TEST INSTRUMENT (the isolation invariant's mechanical half): scatter every
/// full-attention layer's placement by pair-swapping 64-cell blocks. Output
/// must be byte-identical to identity placement -- physical location must be
/// invisible. Pair-swap breaks block adjacency at every crossing, so the
/// kernels' 8-run fast/slow paths are both exercised.
pub fn set_scrambled_tables(
    be: &dyn Backend,
    geom: &[LayerStateGeom],
    slots: usize,
    capacity: usize,
) {
    // Clamped to CAPACITY: buffers are sized min(kv_round(pos), capacity), so a
    // swap must never map into blocks the allocation clamped away (the first
    // version did, at n=65, and produced wild stores -- in the TEST, not the
    // engine).
    let backed = slots.min(capacity);
    let blocks = (backed / 64) & !1; // even count; tail stays identity
    for geom in geom {
        if !matches!(geom.kind, StateKind::Full) {
            continue;
        }
        let ident = std::env::var("IMPARO_SCRAMBLE_IDENTITY").is_ok();
        let table: Vec<u32> = (0..blocks as u32)
            .map(|i| if ident { i } else { i ^ 1 })
            .chain((blocks as u32)..(blocks as u32 + 8))
            .collect();
        be.set_kv_page_table(geom.layer, &table);
    }
}

// ---- JOINING THE POOL --------------------------------------------------------------
//
// The design says "a model contributes ONLY this geometry descriptor -- the whole per-model
// cost of joining the pool". It was not true while a model also had to write fifteen
// wrappers, and the second model is what made that visible: LFM2 would have copied every
// one of them verbatim. A method every model writes identically is boilerplate whatever it
// is called.
//
// So the wrappers are defaults here, over the ten things only the model knows. What a model
// contributes is those ten accessors and its plan.

/// A model whose KV joins the pool.
///
/// Implement the required accessors; the rest come with it. Nothing here computes anything
/// a model would compute differently -- the geometry, the boundaries and the capture calls
/// are all derivations over the plan and the shared state kinds.
pub trait KvPoolMember {
    // --- what only the model knows ---

    /// The model's plan; the geometry is derived from it.
    fn plan(&self) -> &crate::ModelPlan;
    /// The state every workflow carries. One pair of accessors, not one per field:
    /// `kv_ring_batch`, `kv_runtime`, `weights_on_gpu` and `is_gpu_ready` were four
    /// separate methods a model had to write, and all four read one struct.
    fn state(&self) -> &crate::WorkflowState;
    /// The same, to write.
    fn state_mut(&mut self) -> &mut crate::WorkflowState;
    /// Bring the device path up if it is not already.
    ///
    /// Entered from the forward AND from the pool: `kv_restore` and `kv_prepare_pool`
    /// both arrive without a forward, so either can be the first thing to touch the
    /// device.
    ///
    /// # Errors
    /// Returns an error when preparation fails, or there is no device workflow.
    fn ensure_gpu_ready(&mut self) -> Result<(), String>;
    /// Grow every layer's cache to hold `positions`, keeping the contents.
    ///
    /// # Errors
    /// Returns an error when the backend cannot allocate.
    fn kv_fit(&mut self, positions: usize) -> Result<(), String>;

    // --- everything below is derived ---

    /// Largest batch the rings must cover; see `ring_slots`.
    fn kv_ring_batch(&self) -> usize {
        self.state().kv_ring_batch
    }
    /// The cache's capacity, allocation and fill.
    fn kv_runtime(&self) -> &KvRuntime {
        &self.state().kv_rt
    }
    /// The same, to write: growth updates `slots`, a resume point updates `filled`.
    fn kv_runtime_mut(&mut self) -> &mut KvRuntime {
        &mut self.state_mut().kv_rt
    }
    /// Whether a backend holds the weights at all -- the complement of running here.
    fn weights_on_gpu(&self) -> bool {
        !self.state().host_forward
    }
    /// Whether the device path has been brought up.
    fn is_gpu_ready(&self) -> bool {
        self.state().gpu_ready
    }

    /// Per-layer capture geometry.
    fn kv_state_geometry(&self) -> Vec<LayerStateGeom> {
        state_geometry(self.plan(), self.kv_ring_batch())
    }
    /// The pooled set: layers whose state grows with context.
    fn kv_full_layers(&self) -> Vec<u32> {
        full_layers(&self.kv_state_geometry())
    }
    /// The tightest ring slack across windowed layers.
    /// How far past a boundary the device may run and still produce a VALID checkpoint
    /// for it.
    ///
    /// For a windowed ring it is `ring - window`: the rows for `[boundary - window,
    /// boundary)` survive that many more writes. For RECURRENT state it is ZERO -- the
    /// buffer holds "now" and the very next token overwrites it -- so a model with any
    /// recurrent layer must checkpoint exactly where it stands, and the caller's
    /// deferred path is never valid for it.
    ///
    /// Named for the question rather than for windows, because the answer is no longer
    /// only about them.
    fn kv_checkpoint_slack(&self) -> usize {
        if self.plan().recurrent_elems() > 0 {
            return 0;
        }
        window_slack(&self.kv_state_geometry())
    }
    /// 64-cell blocks each pooled layer holds once prepared to capacity.
    fn kv_capacity_blocks(&self) -> u32 {
        (self.kv_runtime().capacity / 64) as u32
    }
    /// Declare the resume point without touching any state (bookkeeping after an external
    /// truncation decision).
    fn kv_set_filled(&mut self, filled: usize) {
        self.kv_runtime_mut().filled = filled;
    }
    /// Largest boundary spillable from live state: floor-256 discards at most 255
    /// positions, inside every ring's slack (ring >= window + batch).
    fn kv_spill_boundary(&self) -> usize {
        self.kv_runtime().filled & !255
    }
    /// Captures canonical state at `kv_spill_boundary()` (device idle).
    fn kv_spill(&self) -> Option<KvState> {
        if !self.is_gpu_ready() || !self.weights_on_gpu() {
            return None;
        }
        let boundary = self.kv_spill_boundary();
        if boundary == 0 {
            return None;
        }
        Some(imparo_kv::state::capture(
            backend(),
            &self.kv_state_geometry(),
            boundary,
            self.kv_recurrent_blob(boundary)?,
        ))
    }
    /// Restores a captured state; the next forward resumes at `state.boundary`.
    ///
    /// # Errors
    /// Returns an error when no backend holds the weights, or the cache cannot grow.
    fn kv_restore(&mut self, state: &KvState) -> Result<(), String> {
        if !self.weights_on_gpu() {
            return Err("kv_restore: GPU path inactive".into());
        }
        self.ensure_gpu_ready()?;
        self.kv_fit(state.boundary)?;
        imparo_kv::state::restore(
            backend(),
            &self.kv_state_geometry(),
            state,
            self.plan().recurrent_elems() as usize,
        )?;
        self.kv_note_restored_recurrent(state);
        self.kv_runtime_mut().filled = state.boundary;
        Ok(())
    }
    /// Grow to full capacity up front (pool mode): on unified memory untouched pages stay
    /// uncommitted, so this reserves address space, not RAM.
    ///
    /// # Errors
    /// Returns an error when the backend cannot allocate.
    fn kv_prepare_pool(&mut self) -> Result<(), String> {
        self.ensure_gpu_ready()?;
        self.kv_fit(self.kv_runtime().capacity)
    }
    /// Apply per-layer block tables (the pool's placement decision).
    fn kv_apply_tables(&self, tables: &std::collections::BTreeMap<u32, Vec<u32>>) {
        for (&layer, t) in tables {
            backend().set_kv_page_table(layer, t);
        }
    }
    /// Point every windowed layer at one conversation's own ring.
    ///
    /// The sibling of `kv_apply_tables`: pooled layers are placed by block table, windowed
    /// layers by region. `region` indexes the rings a windowed layer holds; region 0 is
    /// the single-ring layout, so a build with one region is byte-identical to none.
    fn kv_apply_region(&self, region: usize) {
        let be = backend();
        for g in self.kv_state_geometry() {
            let StateKind::Window { ring, .. } = g.kind else { continue };
            be.set_kv_region(
                g.layer,
                (region * ring * g.k_stride) as u64,
                (region * ring * g.v_stride) as u64,
            );
        }
    }
    /// Capture ONLY the bounded layers' state at a 256-aligned boundary (the switch cost).
    /// The boundary must be at most `kv_filled()` and within every ring's slack.
    /// The snapshot the device currently holds, as `(boundary, bytes)`.
    ///
    /// The pool takes this at a switch and gives it back at the next one, because the
    /// note belongs to a CONVERSATION while the buffer belongs to the device. With
    /// requests in flight for several conversations, one note reused across them is one
    /// conversation's state answering for another.
    fn kv_recurrent_note(&self) -> Option<(usize, Vec<u8>)> {
        let st = self.state();
        (!st.recur_ckpt.is_empty()).then(|| (st.recur_ckpt_at, st.recur_ckpt.clone()))
    }

    /// Makes a conversation's snapshot the live one: WRITES it to the device and records
    /// it.
    ///
    /// Writing is the point. Adopting resident KV blocks restores a conversation's
    /// attention cache without touching the recurrent buffer, so the device would keep
    /// whichever conversation ran last -- correct-looking KV over someone else's
    /// convolution history. Restoring from a checkpoint already writes it; this is the
    /// path that does not.
    ///
    /// # Errors
    /// When the blob's length disagrees with this model.
    fn kv_install_recurrent(&mut self, note: Option<(usize, Vec<u8>)>) -> Result<(), String> {
        let n = self.plan().recurrent_elems() as usize;
        if n == 0 {
            return Ok(());
        }
        let Some((at, blob)) = note else {
            // Nothing known for this conversation: forget the previous one rather than
            // let it answer for a conversation it does not describe.
            let st = self.state_mut();
            st.recur_ckpt_at = 0;
            st.recur_ckpt.clear();
            return Ok(());
        };
        imparo_kv::state::restore_recurrent(backend(), n, &blob)?;
        let st = self.state_mut();
        st.recur_ckpt_at = at;
        st.recur_ckpt = blob;
        Ok(())
    }

    /// Records a just-restored recurrent state as the process's known snapshot.
    ///
    /// Without this, a process that RESTORED a conversation rather than prefilling it has
    /// the right bytes on the device and no record of which boundary they belong to, so
    /// its own next checkpoint cannot be produced -- `noted at 0`. It bites hardest where
    /// conversations interleave: the field describes one conversation, and switching
    /// makes the previous one's note wrong. Restoring is exactly when the truth is known
    /// again.
    fn kv_note_restored_recurrent(&mut self, state: &KvState) {
        if self.plan().recurrent_elems() == 0 {
            return;
        }
        let b = state.boundary;
        let blob = state.recurrent.clone();
        let st = self.state_mut();
        st.recur_ckpt = blob;
        st.recur_ckpt_at = b;
    }

    /// The recurrent snapshot a checkpoint at `boundary` must carry, or None when this
    /// model cannot produce one for that position.
    ///
    /// Three cases, and the third is the point:
    ///
    /// ```text
    /// no recurrent layers      -> Some(empty)      nothing to carry
    /// boundary == filled       -> read the device  it is standing there
    /// boundary == the noted    -> the copy taken as that boundary was passed
    /// otherwise                -> None             REFUSE; it cannot be reconstructed
    /// ```
    fn kv_recurrent_blob(&self, boundary: usize) -> Option<Vec<u8>> {
        let n = self.plan().recurrent_elems() as usize;
        if n == 0 {
            return Some(Vec::new());
        }
        if boundary == self.kv_runtime().filled {
            return Some(read_recurrent(n, imparo_backend::BufId::Recur));
        }
        if boundary == self.state().recur_ckpt_at && !self.state().recur_ckpt.is_empty() {
            return Some(self.state().recur_ckpt.clone());
        }
        if crate::log_on() {
            eprintln!(
                "[imparo] no recurrent snapshot for boundary {boundary} (device at {}, \
                 noted at {})",
                self.kv_runtime().filled,
                self.state().recur_ckpt_at
            );
        }
        None
    }

    fn kv_capture_windows(&self, boundary: usize) -> KvState {
        let geom = bounded_geometry(&self.kv_state_geometry());
        imparo_kv::state::capture(
            backend(),
            &geom,
            boundary,
            self.kv_recurrent_blob(boundary).unwrap_or_default(),
        )
    }
    /// Capture a bounded-layer DELTA at `boundary` chained from a checkpoint at `from`:
    /// rows [max(from, boundary - window), boundary) per layer, so cost min(gap, window)
    /// instead of a whole window.
    fn kv_capture_delta(&self, boundary: usize, from: usize) -> imparo_kv::state::KvDelta {
        let geom = bounded_geometry(&self.kv_state_geometry());
        imparo_kv::state::capture_delta(
            backend(),
            &geom,
            self.kv_recurrent_blob(boundary).unwrap_or_default(),
            boundary,
            from,
        )
    }
    /// Assemble a full bounded state from a newest-first delta chain; None when a link is
    /// missing, which the caller reads as "no checkpoint".
    fn kv_assemble_chain(
        &self,
        chain: &[&imparo_kv::state::KvDelta],
    ) -> Option<KvState> {
        let geom = bounded_geometry(&self.kv_state_geometry());
        imparo_kv::state::assemble_chain(&geom, chain)
    }
    /// Restore bounded layers and set the resume point. Pooled layers are untouched --
    /// their state is wherever the tables point.
    ///
    /// # Errors
    /// Returns an error when the cache cannot grow.
    /// Puts a conversation's state on the device: its bounded layers AND its recurrent
    /// buffer, in ONE call.
    ///
    /// `boundary` 0 with no windows means "adopted from resident blocks" -- the attention
    /// cache is already in place via the block tables and only the recurrent buffer needs
    /// writing. That case used to be a bare `kv_set_filled(0)` with the recurrent install
    /// bolted on afterwards, which is how the device came to hold one conversation's
    /// convolution history under another's KV.
    ///
    /// # Errors
    /// When the cache cannot grow, or the recurrent blob disagrees with this model.
    fn kv_resume(&mut self, state: &KvState) -> Result<(), String> {
        if state.window.is_empty() && state.boundary == 0 {
            self.kv_install_recurrent(
                (!state.recurrent.is_empty()).then(|| (0, state.recurrent.clone())),
            )?;
            self.kv_set_filled(0);
            return Ok(());
        }
        self.kv_restore_windows(state)
    }

    fn kv_restore_windows(&mut self, state: &KvState) -> Result<(), String> {
        self.kv_fit(state.boundary)?;
        let geom = bounded_geometry(&self.kv_state_geometry());
        imparo_kv::state::restore(
            backend(),
            &geom,
            state,
            self.plan().recurrent_elems() as usize,
        )?;
        self.kv_note_restored_recurrent(state);
        self.kv_runtime_mut().filled = state.boundary;
        Ok(())
    }
    /// TEST INSTRUMENT: scatter placement so physical location must be invisible.
    fn kv_set_scrambled_tables(&self) {
        set_scrambled_tables(
            backend(),
            &self.kv_state_geometry(),
            self.kv_runtime().slots,
            self.kv_runtime().capacity,
        );
    }
    /// Diagnostic: per-layer K/V value structure (IMPARO_KV_SCAN gates the caller).
    fn kv_scan(&self, positions: usize) {
        scan(backend(), self.plan(), positions);
    }
}

/// The active backend. The defaults above reach it here rather than taking it as an
/// eleventh accessor: which backend is live is a process fact, not a model's.
fn backend() -> &'static dyn Backend {
    crate::backend::active().expect("kv pool entered with no active backend")
}

#[cfg(test)]
mod tests {
    use super::resume_point;

    /// The rule the engine gate measures, in one place: land on the 64 grid, and leave
    /// the resumed pass at least two tokens (one is a decode shape and still differs).
    #[test]
    fn resume_point_is_the_64_grid_with_a_two_token_tail() {
        // An exact resend: 744 resident, 744 asked -> 704, a 40-token tail.
        assert_eq!(resume_point(744, 744), 704);
        // A 1-token append would leave a 1-token tail at 704, so it stops a grid short.
        assert_eq!(resume_point(704, 705), 640);
        // Two tokens is enough.
        assert_eq!(resume_point(704, 706), 704);
        // Nothing resident, or nothing to resume into.
        assert_eq!(resume_point(0, 1000), 0);
        assert_eq!(resume_point(1000, 1), 0);
        assert_eq!(resume_point(1000, 0), 0);
        // Never past what is resident.
        assert_eq!(resume_point(100, 10_000), 64);
    }
}
