//! Gemma4 QAT assistant adapter over the shared greedy scheduler and target KV.
//! Explicit owner-lab pairing only; dense Q8 head, two proposals, no cached resume.
use crate::{
    Attention, KvSource, Model, ModelPlan,
    speculative::{DraftProvider, DraftTree},
    weights::Weights,
};
use imparo_cuda::gemma4_mtp::{Config, Layer};
use imparo_gguf::{Document, MetadataValue, Scalar};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

const BLOCK: usize = 3;
pub(crate) const TREE4_PARENTS: [i32; 4] = [-1, 0, 1, 0];
const TREE4_DELAYED_PARENTS: [i32; 4] = [-1, 0, 1, 1];
pub(crate) fn tree4_delayed_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("IMPARO_LAB_E4B_TREE_DELAYED").as_deref() == Ok("1"))
}
pub(crate) fn tree4_parents() -> &'static [i32; 4] {
    if tree4_delayed_enabled() { &TREE4_DELAYED_PARENTS } else { &TREE4_PARENTS }
}
pub(crate) fn tree4_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("IMPARO_DSPARK_TREE").as_deref() == Ok("4"))
}
pub(crate) fn tree4_fits(start: usize, capacity: usize, cell: usize) -> bool {
    capacity == 1024
        && (512..=765).contains(&start)
        && cell != 0
        && TREE4_PARENTS.len() <= cell - start % cell
}
fn tree4_from_chain(anchor: u32, chain: &[u32], alternate: u32, stops: &[u32], delayed: bool) -> Option<DraftTree> {
    if chain.len() != BLOCK - 1
        || alternate == chain[usize::from(delayed)]
        || chain.iter().chain(std::iter::once(&alternate)).any(|id| stops.contains(id))
    {
        return None;
    }
    Some(DraftTree::plain(
        vec![anchor, chain[0], chain[1], alternate],
        if delayed { TREE4_DELAYED_PARENTS.to_vec() } else { TREE4_PARENTS.to_vec() },
    ))
}
// Mirror the existing short-verification boundary contract, not a new tuning
// policy. The four-node tree has only three logical positions. An eager chain
// crossing a 32-position seam cannot price an eager non-crossing batch.
fn cost_domain(start: usize, rows: usize) -> u64 {
    if start >= 1023 { return 3; }
    let last = start.saturating_add(rows.min(BLOCK).saturating_sub(1));
    if start / 32 == last / 32 { 1 } else { 2 }
}
fn selected_chain(ids: &[u32; 4], vocab: u32) -> Result<[u32; 2], String> {
    if ids.iter().any(|&id| id >= vocab) || ids[2] == ids[3]
        || (ids[0] != ids[2] && ids[0] != ids[3])
    {
        return Err("MTP selected chain outside candidate contract".into());
    }
    Ok([ids[0], ids[1]])
}
const ASSISTANT_SHA256: &str =
    "4216d488258de0b66204f51b662cea2b75c643ebbd874672bdc75236d1003b9f";

pub struct Descriptor {
    pub config: Config,
    pub layers: Vec<Layer>,
    pub capacity: usize,
}
fn uint(doc: &Document, key: &str) -> Result<u32, String> {
    u32::try_from(
        doc.unsigned_value(key)
            .ok_or_else(|| format!("MTP missing {key}"))?,
    )
    .map_err(|_| format!("MTP {key} exceeds u32"))
}
fn positive(doc: &Document, key: &str) -> Result<f32, String> {
    match doc.metadata.get(key) {
        Some(MetadataValue::Scalar(Scalar::Float(x))) if x.is_finite() && *x > 0.0 => {
            let x = *x as f32;
            if x.is_finite() && x > 0.0 {
                Ok(x)
            } else {
                Err(format!("MTP invalid {key}"))
            }
        }
        _ => Err(format!("MTP missing positive float {key}")),
    }
}
fn bytes(kind: u32, shape: &[u64]) -> Result<u64, String> {
    if shape.is_empty() || shape.contains(&0) {
        return Err("MTP empty tensor".into());
    }
    let n = shape
        .iter()
        .try_fold(1_u64, |n, x| n.checked_mul(*x))
        .ok_or("MTP size overflow")?;
    match kind {
        0 => n.checked_mul(4),
        2 if shape[0] % 32 == 0 => (n / 32).checked_mul(18),
        8 if shape[0] % 32 == 0 => (n / 32).checked_mul(34),
        _ => None,
    }
    .ok_or_else(|| "MTP unsupported tensor storage".into())
}
fn span(offset: u64, size: u64, alignment: u64, limit: u64) -> Result<(), String> {
    if !alignment.is_power_of_two()
        || offset % alignment != 0
        || size == 0
        || offset.checked_add(size).is_none_or(|end| end > limit)
    {
        return Err("MTP tensor span/alignment invalid".into());
    }
    Ok(())
}
fn scalar(weights: &Weights, offset: u64) -> Result<f32, String> {
    span(offset, 4, 4, weights.byte_len())?;
    let at = usize::try_from(offset).map_err(|_| "MTP scalar address overflow")?;
    // The checked immutable paired mapping owns these four bytes.
    let value =
        unsafe { std::ptr::read_unaligned(weights.base_ptr().add(at).cast::<f32>()) };
    if value.is_finite() {
        Ok(value)
    } else {
        Err("MTP nonfinite layer scale".into())
    }
}

/// Build checked offsets against the paired mapping that holds the target.
/// This initial provider supports the verified E4B QAT assistant, not arbitrary Gemma models.
pub fn load_descriptor(
    doc: &Document,
    weights: &Weights,
    draft_base: u64,
    plan: &ModelPlan,
    capacity: usize,
    ring_batch: usize,
) -> Result<Descriptor, String> {
    let p = &plan.config;
    if p.architecture != "gemma4"
        || p.n_embd != 2560
        || p.vocab_size != 262_144
        || p.n_kv_heads != 2
        || plan.layers.is_empty()
        || plan.recurrent_elems() != 0
    {
        return Err("MTP requires its Gemma4 E4B target geometry".into());
    }
    if doc.string_value("general.architecture") != Some("gemma4-assistant")
        || doc.alignment < 4
        || doc.data_offset % doc.alignment != 0
    {
        return Err("MTP requires a valid Gemma4 assistant GGUF".into());
    }
    span(draft_base, doc.file_size, doc.alignment, weights.byte_len())?;
    if draft_base == 0
        || weights.tensors.values().any(|t| {
            (t.offset as u64)
                .checked_add(t.bytes as u64)
                .is_none_or(|end| end > draft_base)
        })
    {
        return Err("MTP assistant region overlaps target tensors".into());
    }
    let u = |suffix: &str| uint(doc, &format!("gemma4-assistant.{suffix}"));
    let f = |suffix: &str| positive(doc, &format!("gemma4-assistant.{suffix}"));
    let hidden = u("embedding_length")?;
    let target_hidden = u("embedding_length_out")?;
    let ffn = u("feed_forward_length")?;
    let heads = u("attention.head_count")?;
    let kv_heads = u("attention.head_count_kv")?;
    let count = u("block_count")?;
    if hidden != 256
        || target_hidden != p.n_embd
        || ffn != 2048
        || heads != 4
        || kv_heads != p.n_kv_heads
        || count != 4
        || u("attention.shared_kv_layers")? != count
        || u("nextn_predict_layers")? != count
        || u("embedding_length_per_layer_input")? != 0
    {
        return Err("MTP assistant capability/target mismatch".into());
    }
    if capacity < BLOCK
        || capacity > u("context_length")? as usize
        || capacity > u32::MAX as usize
        || ring_batch < BLOCK
        || ring_batch > i32::MAX as usize / p.n_embd as usize
    {
        return Err("MTP target/batch capacity unsupported".into());
    }
    let pattern = match doc
        .metadata
        .get("gemma4-assistant.attention.sliding_window_pattern")
    {
        Some(MetadataValue::Array(a))
            if a.element_count == 4 && a.preview.len() == 4 =>
        {
            a.preview
                .iter()
                .map(|x| match x {
                    Scalar::Bool(b) => Ok(*b),
                    _ => Err("MTP nonboolean SWA pattern"),
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("MTP missing/truncated SWA pattern".into()),
    };
    if pattern != [true, true, true, false] {
        return Err("MTP unsupported attention order".into());
    }
    let mut names = std::collections::BTreeSet::new();
    if doc.tensors.len() != 49 || doc.tensors.iter().any(|t| !names.insert(&t.name)) {
        return Err(
            "MTP dense-head tensor inventory differs from verified conversion".into(),
        );
    }
    let offset = |name: &str, kind: u32, shape: &[u64]| -> Result<u64, String> {
        let t = doc
            .tensor(name)
            .ok_or_else(|| format!("MTP missing tensor {name}"))?;
        let size = bytes(kind, shape)?;
        if t.ggml_type != kind
            || t.dimensions != shape
            || t.byte_size != size
            || doc.data_offset.checked_add(t.relative_offset) != Some(t.absolute_offset)
        {
            return Err(format!("MTP tensor contract mismatch: {name}"));
        }
        span(t.absolute_offset, size, doc.alignment, doc.file_size)?;
        let relocated = draft_base
            .checked_add(t.absolute_offset)
            .ok_or("MTP relocation overflow")?;
        span(relocated, size, doc.alignment, weights.byte_len())?;
        Ok(relocated)
    };
    let emb = weights
        .get("token_embd.weight")
        .ok_or("MTP target embedding missing")?;
    if emb.n_dims != 2
        || emb.ne[..2] != [u64::from(target_hidden), u64::from(p.vocab_size)]
        || !matches!(emb.ggml_type, 0 | 2 | 8)
        || emb.bytes as u64 != bytes(emb.ggml_type, &emb.ne[..2])?
    {
        return Err("MTP target embedding layout unsupported".into());
    }
    span(emb.offset as u64, emb.bytes as u64, 4, draft_base)?;
    let embedding_kind = imparo_gguf::weights::weight_kind(emb.ggml_type)
        .ok_or("MTP embedding kind")? as u32;
    let h = u64::from(hidden);
    let th = u64::from(target_hidden);
    let ff = u64::from(ffn);
    let config = Config {
        target_embedding: emb.offset as u64,
        pre_proj: offset("nextn.pre_projection.weight", 8, &[2 * th, h])?,
        post_proj: offset("nextn.post_projection.weight", 8, &[h, th])?,
        out_norm: offset("output_norm.weight", 0, &[h])?,
        head: offset("token_embd.weight", 8, &[h, u64::from(p.vocab_size)])?,
        hidden,
        target_hidden,
        ffn,
        vocab: p.vocab_size,
        heads,
        kv_heads,
        batch_capacity: u32::try_from(ring_batch).map_err(|_| "MTP batch overflow")?,
        target_final_layer: u32::try_from(plan.layers.len() - 1)
            .map_err(|_| "MTP target layer overflow")?,
        target_embedding_kind: embedding_kind,
        reserved: 0,
        eps: f("attention.layer_norm_rms_epsilon")?,
        embedding_scale: if plan.embed.scale_by_sqrt_embd {
            (target_hidden as f32).sqrt()
        } else {
            1.0
        },
    };
    let backend = crate::gpu_support::be();
    let route = crate::kv::effective_workflow_kv_route(
        plan,
        crate::kv::KvType::k(),
        crate::kv::KvType::v(),
        backend.kv_quantization_route(),
        backend.kv_quantization_route_override(),
    )?;
    let mut layers = Vec::with_capacity(4);
    for (i, windowed) in pattern.into_iter().enumerate() {
        let hd = u(if windowed {
            "attention.key_length_swa"
        } else {
            "attention.key_length"
        })?;
        let vd = u(if windowed {
            "attention.value_length_swa"
        } else {
            "attention.value_length"
        })?;
        let rope_dim = u(if windowed {
            "rope.dimension_count_swa"
        } else {
            "rope.dimension_count"
        })?;
        let rope_theta = f(if windowed {
            "rope.freq_base_swa"
        } else {
            "rope.freq_base"
        })?;
        let window = if windowed {
            u("attention.sliding_window")?
        } else {
            0
        };
        if hd != (if windowed { 256 } else { 512 })
            || vd != hd
            || rope_dim != hd
            || (windowed && window != 512)
        {
            return Err("MTP D256 SWA/D512 full attention capability mismatch".into());
        }
        // Match the target's last layer of this attention kind, then follow its
        // actual KV-source mapping instead of assuming that layer owns storage.
        let last = plan
            .layers
            .iter()
            .rposition(|l| {
                matches!(l.attention, Attention::Window { .. }) == windowed
                    && l.attention.is_attention()
            })
            .ok_or("MTP target attention kind missing")?;
        let mut owner = last;
        loop {
            match plan.layers[owner].kv_source {
                KvSource::Own => break,
                KvSource::SharedWith(source) if (source as usize) < owner => {
                    owner = source as usize;
                }
                KvSource::SharedWith(_) => {
                    return Err("MTP invalid target KV ownership chain".into());
                }
            }
        }
        let actual = plan.layers[owner].attention;
        let (actual_dim, actual_rope, actual_window) = match actual {
            Attention::Full {
                rope_dim,
                rope_base,
                ..
            } => (rope_dim, rope_base, 0),
            Attention::Window {
                rope_dim,
                rope_base,
                window,
                ..
            } => (rope_dim, rope_base, window),
            Attention::Recurrent { .. } => {
                return Err("MTP target source is not KV attention".into());
            }
        };
        if actual.head_dim() != hd
            || actual_dim != rope_dim
            || actual_rope.to_bits() != rope_theta.to_bits()
            || actual_window != window
            || plan.layers[last].attention != actual
            || plan.layers[owner].index as usize != owner
        {
            return Err(
                "MTP shared target KV geometry differs from assistant attention".into(),
            );
        }
        let w = |suffix: &str, kind: u32, shape: &[u64]| {
            offset(&format!("blk.{i}.{suffix}.weight"), kind, shape)
        };
        let q = u64::from(heads) * u64::from(hd);
        let scale_offset = w("layer_output_scale", 0, &[1])?;
        layers.push(Layer {
            attn_norm: w("attn_norm", 0, &[h])?,
            q: w("attn_q", 8, &[h, q])?,
            qn: w("attn_q_norm", 0, &[u64::from(hd)])?,
            o: w("attn_output", 8, &[q, h])?,
            attn_post_norm: w("post_attention_norm", 0, &[h])?,
            ffn_norm: w("ffn_norm", 0, &[h])?,
            gate: w("ffn_gate", 8, &[h, ff])?,
            up: w("ffn_up", 8, &[h, ff])?,
            down: w("ffn_down", 8, &[ff, h])?,
            ffn_post_norm: w("post_ffw_norm", 0, &[h])?,
            rope_freqs: if windowed {
                0
            } else {
                offset("rope_freqs.weight", 0, &[u64::from(hd / 2)])?
            },
            target_kv_layer: owner as u32,
            head_dim: hd,
            rope_dim,
            window,
            ring: crate::kv::ring_mask(actual, ring_batch),
            had_k: if crate::kv::KvType::k() == crate::kv::KvType::F16 {
                0
            } else {
                crate::kv::had_nrot("IMPARO_HAD_K", route.key, hd)?
            },
            had_v: if crate::kv::KvType::v() == crate::kv::KvType::F16 {
                0
            } else {
                crate::kv::had_nrot("IMPARO_HAD_V", route.value, hd)?
            },
            has_rope_freqs: u32::from(!windowed),
            rope_theta,
            out_scale: scalar(weights, scale_offset)?,
        });
    }
    Ok(Descriptor {
        config,
        layers,
        capacity,
    })
}

fn bridge(start: usize, remaining: usize, capacity: usize, cell: usize) -> bool {
    if cell < BLOCK {
        return false;
    }
    let distance = cell - start % cell;
    distance < BLOCK
        && distance.checked_add(BLOCK).is_some_and(|n| n <= remaining)
        && start
            .checked_add(distance)
            .and_then(|n| n.checked_add(BLOCK))
            .is_some_and(|end| end <= capacity)
}

/// A request-local attachment. No provider, borrowed KV pointer or GPU owner can
/// cross threads; Pairing stores only immutable CPU descriptors between requests.
pub struct Provider {
    descriptor: Descriptor,
    enabled: Option<Arc<AtomicBool>>,
    live: bool,
    history: usize,
    poison: Option<String>,
    minimum_probability: Option<f32>,
    confidence_trace: bool,
    tree_round: Option<(usize, u32, [u32; 3])>,
    tree_trace_done: bool,
    chain_selector: bool,
    selector_rounds: usize,
    selector_swaps: usize,
    // Request-local and diagnostic only. Never inherit another model/config's
    // prices and never make a tree proposal merely to populate a cost table.
    cost: Option<crate::verify_cost::TreeCostObserver>,
    _serial: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Provider {
    /// # Safety
    /// Keep the exclusive target and paired weight mapping alive until finish succeeds.
    pub unsafe fn attach<M: Model + ?Sized>(
        target: &mut M,
        descriptor: Descriptor,
    ) -> Result<Self, String> {
        let confidence_trace =
            std::env::var("IMPARO_LAB_MTP_CONFIDENCE_TRACE").as_deref() == Ok("1");
        let minimum = if crate::e4b_retained_decode_policy_enabled() {
            0.0
        } else {
            match std::env::var("IMPARO_LAB_MTP_MIN_PROB") {
                Ok(value) => {
                    let p = value
                        .parse::<f32>()
                        .map_err(|_| "MTP probability threshold must be numeric")?;
                    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                        return Err(
                            "MTP probability threshold must be within [0,1]".into()
                        );
                    }
                    p
                }
                Err(std::env::VarError::NotPresent) => 0.0,
                Err(_) => {
                    return Err("MTP probability threshold is not valid text".into());
                }
            }
        };
        let minimum_probability = if minimum > 0.0 || confidence_trace {
            Some(minimum)
        } else {
            None
        };
        let chain_selector = std::env::var("IMPARO_LAB_E4B_CHAIN_SELECTOR").as_deref() == Ok("1");
        if chain_selector && (tree4_enabled() || minimum_probability.is_some()
            || !crate::e4b_retained_decode_policy_enabled()
            || descriptor.capacity != 1024 || descriptor.config.target_embedding_kind != 1
            || descriptor.config.target_hidden != 2560
            || crate::kv::KvType::k() != crate::kv::KvType::Q4_0
            || crate::kv::KvType::v() != crate::kv::KvType::Q4_0)
        {
            return Err("MTP chain selector requires exclusive retained short E4B Q4 route".into());
        }
        if target.state().host_forward
            || target
                .state()
                .layer_outputs
                .as_ref()
                .is_some_and(super::layer_outputs::LayerOutputCapture::attached)
        {
            return Err(
                "MTP requires an exclusive CUDA target without another observer".into(),
            );
        }
        if let Err(error) = unsafe {
            imparo_cuda::gemma4_mtp::attach(&descriptor.config, &descriptor.layers)
        } {
            return match unsafe { imparo_cuda::gemma4_mtp::detach() } {
                Ok(()) => Err(error),
                Err(cleanup) => {
                    Err(format!("{error}; MTP failed-attach cleanup: {cleanup}"))
                }
            };
        }
        let enabled = Arc::new(AtomicBool::new(false));
        target.state_mut().layer_outputs =
            Some(crate::layer_outputs::LayerOutputCapture {
                enabled: Arc::downgrade(&enabled),
                layers: vec![descriptor.config.target_final_layer],
                capture: crate::layer_outputs::CaptureFn::observer(
                    |layer, start, rows, src| unsafe {
                        imparo_cuda::gemma4_mtp::capture_layer(
                            layer, start, rows, src as u32,
                        )
                    },
                ),
            });
        Ok(Self {
            descriptor,
            enabled: Some(enabled),
            live: true,
            history: 0,
            poison: None,
            minimum_probability,
            confidence_trace,
            tree_round: None,
            tree_trace_done: false,
            chain_selector,
            selector_rounds: 0,
            selector_swaps: 0,
            cost: (std::env::var("IMPARO_DSPARK_CLOCK").as_deref() == Ok("1"))
                .then(|| crate::verify_cost::TreeCostObserver::new(4)).flatten(),
            _serial: std::marker::PhantomData,
        })
    }
    fn ready(&self) -> Result<(), String> {
        if let Some(e) = &self.poison {
            return Err(e.clone());
        }
        if self.live {
            Ok(())
        } else {
            Err("MTP provider detached".into())
        }
    }
    fn checked(&mut self, result: Result<(), String>) -> Result<(), String> {
        if let Err(e) = &result {
            self.poison = Some(e.clone());
        }
        result
    }
    fn commit_inputs(
        &mut self,
        start: usize,
        inputs: &[u32],
        path: Option<&[i32]>,
    ) -> Result<(), String> {
        self.ready()?;
        let end = start
            .checked_add(inputs.len())
            .ok_or("MTP commit position overflow")?;
        if start != self.history
            || inputs.is_empty()
            || inputs.len() > self.descriptor.config.batch_capacity as usize
            || end > self.descriptor.capacity
            || inputs.iter().any(|&x| x >= self.descriptor.config.vocab)
        {
            return Err("MTP accepted-prefix commit invalid".into());
        }
        if let Some(path) = path {
            if path.len() != inputs.len()
                || path.iter().any(|&node| {
                    usize::try_from(node)
                        .ok()
                        .filter(|&node| node < self.descriptor.config.batch_capacity as usize)
                        .and_then(|node| start.checked_add(node))
                        .is_none_or(|position| position >= self.descriptor.capacity)
                })
            {
                return self.checked(Err("MTP accepted-tree commit invalid".into()));
            }
        }
        let native_start = u32::try_from(start).map_err(|_| "MTP commit overflow")?;
        let result = unsafe {
            match path {
                Some(path) => imparo_cuda::gemma4_mtp::append_tree(native_start, path),
                None => imparo_cuda::gemma4_mtp::append(native_start, inputs.len() as u32),
            }
        };
        self.checked(result)?;
        self.history = end;
        Ok(())
    }
}
impl DraftProvider for Provider {
    fn observes_chain_round(&self) -> bool { self.cost.is_some() }
    fn observe_chain_round(&mut self, timing: &crate::speculative::TreeRoundTiming) {
        // A different candidate policy changes round yield/cost even at the
        // same target shape. Keep its diagnostic prices out of ordinary Chain3.
        let selector_domain = if self.chain_selector && (512..=765).contains(&timing.context) { 1 << 16 } else { 0 };
        if let Some(cost) = self.cost.as_mut() {
            cost.observe_chain_in_domain(timing, cost_domain(timing.context, timing.rows) | selector_domain);
        }
    }
    fn observe_tree_round(&mut self, tree: &DraftTree, timing: &crate::speculative::TreeRoundTiming) {
        if let Some(cost) = self.cost.as_mut() {
            cost.observe_in_domain(tree, timing, cost_domain(timing.context, timing.rows));
        }
    }
    fn tree_proposal(
        &mut self,
        start: usize,
        anchor: u32,
        chain: &[u32],
        stops: &[u32],
    ) -> Result<Option<DraftTree>, String> {
        let Some((draft_start, draft_anchor, ids)) = self.tree_round.take() else {
            return Ok(None);
        };
        if draft_start != start || draft_anchor != anchor || chain != &ids[..2] {
            return self.checked(Err("MTP tree proposal differs from its chain".into()))
                .map(|()| None);
        }
        let delayed = tree4_delayed_enabled();
        let tree = tree4_from_chain(anchor, chain, ids[2], stops, delayed);
        if tree.is_some() && !self.tree_trace_done {
            eprintln!("[gemma4-mtp-tree4] start={start} nodes=4 parents={:?} assistant_forwards=2 original_chain=preserved delayed={delayed}", tree4_parents());
            self.tree_trace_done = true;
        }
        Ok(tree)
    }
    fn block_size(&self) -> usize {
        BLOCK
    }
    fn initialize(&mut self) -> Result<(), String> {
        self.ready()?;
        let result = unsafe { imparo_cuda::gemma4_mtp::reset() };
        self.checked(result)?;
        if let Some(e) = &self.enabled {
            e.store(false, Ordering::Relaxed);
        }
        self.history = 0;
        self.tree_round = None;
        self.selector_rounds = 0;
        self.selector_swaps = 0;
        Ok(())
    }
    fn set_capture(&mut self, enabled: bool) -> Result<(), String> {
        if !self.live {
            return if enabled {
                Err("MTP provider detached".into())
            } else {
                Ok(())
            };
        }
        if enabled {
            self.ready()?;
        }
        if !enabled {
            if let Some(e) = &self.enabled {
                e.store(false, Ordering::Relaxed);
            }
        }
        let result = unsafe { imparo_cuda::gemma4_mtp::capture_mode(enabled) };
        self.checked(result)?;
        if let Some(e) = &self.enabled {
            e.store(enabled, Ordering::Relaxed);
        }
        Ok(())
    }
    fn can_draft(&self, start: usize, _remaining: usize) -> bool {
        let cell = crate::prefill_batch().max(1);
        self.live
            && self.poison.is_none()
            && start > 0
            && BLOCK <= cell - start % cell
            && start
                .checked_add(BLOCK)
                .is_some_and(|end| end <= self.descriptor.capacity)
    }
    fn can_bridge(&self, start: usize, remaining: usize) -> bool {
        self.live
            && self.poison.is_none()
            && bridge(
                start,
                remaining,
                self.descriptor.capacity,
                crate::prefill_batch().max(1),
            )
    }
    fn draft(
        &mut self,
        _target: &mut dyn crate::Model,
        start: usize,
        anchor: u32,
    ) -> Result<Vec<u32>, String> {
        self.ready()?;
        self.tree_round = None;
        if self.history != start || anchor >= self.descriptor.config.vocab {
            return Err("MTP history/anchor mismatch".into());
        }
        if self.chain_selector && (512..=765).contains(&start) {
            let mut ids = [0_u32; 4];
            let result = unsafe { imparo_cuda::gemma4_mtp::generate_selected(start as u32, anchor, &mut ids) };
            self.checked(result)?;
            let chain = match selected_chain(&ids, self.descriptor.config.vocab) {
                Ok(chain) => chain,
                Err(error) => return self.checked(Err(error)).map(|()| Vec::new()),
            };
            self.selector_rounds += 1;
            self.selector_swaps += usize::from(ids[0] != ids[2]);
            if std::env::var("IMPARO_LAB_DRAFT_ACCEPTANCE_TRACE").as_deref() == Ok("1") {
                eprintln!("[mtp-chain-selector] start={start} original={} alternate={} selected={}", ids[2], ids[3], ids[0]);
            }
            return Ok(chain.to_vec());
        }
        if tree4_enabled()
            && crate::e4b_retained_decode_policy_enabled()
            && self.minimum_probability.is_none()
            && self.descriptor.config.batch_capacity >= 4
            && crate::kv::KvType::k() == crate::kv::KvType::Q4_0
            && crate::kv::KvType::v() == crate::kv::KvType::Q4_0
            && tree4_fits(start, self.descriptor.capacity, crate::prefill_batch().max(1))
        {
            let mut ids = [0_u32; 3];
            let result = unsafe {
                imparo_cuda::gemma4_mtp::generate_tree4(start as u32, anchor, &mut ids)
            };
            self.checked(result)?;
            if ids.iter().any(|&id| id >= self.descriptor.config.vocab)
                || ids[usize::from(tree4_delayed_enabled())] == ids[2]
            {
                return self.checked(Err("MTP tree prediction outside contract".into()))
                    .map(|()| Vec::new());
            }
            self.tree_round = Some((start, anchor, ids));
            return Ok(ids[..2].to_vec());
        }
        let mut ids = [0_u32; 2];
        let result = unsafe {
            imparo_cuda::gemma4_mtp::generate(
                u32::try_from(start).map_err(|_| "MTP position overflow")?,
                anchor,
                &mut ids,
            )
        };
        self.checked(result)?;
        if ids.iter().any(|&id| id >= self.descriptor.config.vocab) {
            return self
                .checked(Err("MTP prediction outside vocabulary".into()))
                .map(|()| Vec::new());
        }
        Ok(ids.to_vec())
    }
    fn try_draft(
        &mut self,
        target: &mut dyn crate::Model,
        start: usize,
        anchor: u32,
    ) -> Result<Option<Vec<u32>>, String> {
        let Some(minimum) = self.minimum_probability else {
            return self.draft(target, start, anchor).map(Some);
        };
        self.ready()?;
        if self.history != start || anchor >= self.descriptor.config.vocab {
            return Err("MTP history/anchor mismatch".into());
        }
        let mut ids = [0_u32; 2];
        let result = unsafe {
            imparo_cuda::gemma4_mtp::generate_gated(
                u32::try_from(start).map_err(|_| "MTP position overflow")?,
                anchor,
                minimum,
                &mut ids,
            )
        };
        let (count, probability) = match result {
            Ok(value) => value,
            Err(e) => {
                self.poison = Some(e.clone());
                return Err(e);
            }
        };
        if self.confidence_trace {
            eprintln!(
                "[gemma4-mtp] confidence start={start} probability={probability:.9} proposals={count}"
            );
        }
        if count == 0 {
            return Ok(None);
        }
        if count != 2 || ids.iter().any(|&id| id >= self.descriptor.config.vocab) {
            return self
                .checked(Err("MTP prediction outside vocabulary".into()))
                .map(|()| None);
        }
        Ok(Some(ids.to_vec()))
    }
    fn commit(&mut self, start: usize, inputs: &[u32]) -> Result<(), String> {
        self.commit_inputs(start, inputs, None)
    }
    fn commit_tree(
        &mut self,
        start: usize,
        inputs: &[u32],
        path: &[i32],
        _next: u32,
    ) -> Result<(), String> {
        self.commit_inputs(start, inputs, Some(path))
    }
    fn keep_tree(
        &mut self,
        start: usize,
        inputs: &[u32],
        path: &[i32],
    ) -> Result<(), String> {
        self.commit_inputs(start, inputs, Some(path))
    }
    fn finish(&mut self) -> Result<(), String> {
        if let Some(e) = &self.enabled {
            e.store(false, Ordering::Relaxed);
        }
        if self.live {
            unsafe {
                imparo_cuda::gemma4_mtp::detach()?;
            }
            self.live = false;
            self.enabled = None;
            if self.chain_selector {
                eprintln!("[mtp-chain-selector-summary] rounds={} swaps={} target_rows=3", self.selector_rounds, self.selector_swaps);
            }
        }
        if let Some(report) = self.cost.as_mut().and_then(|cost| cost.take_report()) {
            eprintln!("[gemma4-mtp-cost] {report}");
        }
        Ok(())
    }
}
impl Drop for Provider {
    fn drop(&mut self) {
        if let Err(e) = self.finish() {
            eprintln!("Gemma4 MTP teardown: {e}");
        }
    }
}

/// The Gemma4 assistant, admitted like DSpark's drafter: mapped after the target
/// (`Weights::open_with_appended`). It does not allocate another model, execution engine or
/// target KV cache.
pub struct Pairing {
    document: Document,
    /// The assistant's file, as the manifest names it: the mapping must hold this one.
    draft: std::path::PathBuf,
}
impl Pairing {
    /// Declare immutable assistant weights to the common placement planner.
    /// The native provider owns activations/KV borrowing, not weight residency.
    pub(crate) fn appended_spans(
        &self,
        weights: &Weights,
    ) -> Result<Vec<(u64, u64)>, String> {
        let base = crate::speculative::drafter_offset(weights, &self.draft)?;
        span(
            base,
            self.document.file_size,
            self.document.alignment,
            weights.byte_len(),
        )?;
        self.document
            .tensors
            .iter()
            .map(|tensor| {
                let offset = base
                    .checked_add(tensor.absolute_offset)
                    .ok_or("MTP appended tensor address overflow")?;
                span(
                    offset,
                    tensor.byte_size,
                    self.document.alignment,
                    weights.byte_len(),
                )?;
                Ok((offset, tensor.byte_size))
            })
            .collect()
    }
    /// The assistant a pairing manifest names for `target` (`speculative::admit_pairing`), and
    /// its file's path, which the caller maps after the target's.
    ///
    /// # Errors
    /// When the pairing does not admit, the assistant is not the verified conversion, or a
    /// sidecar or the assistant's header does not check out.
    pub fn load(
        manifest: &std::path::Path,
        target: &std::path::Path,
    ) -> Result<(std::path::PathBuf, Self), String> {
        use sha2::{Digest, Sha256};
        use std::{fs::File, io::Read, path::PathBuf};
        let (v, draft) = crate::speculative::admit_pairing(manifest, target)?;
        if v["draft_sha256"].as_str() != Some(ASSISTANT_SHA256) {
            return Err("MTP assistant is not the verified QAT Q8 conversion".into());
        }
        // Startup identity only: these sidecars belong to the fixed assistant above.
        if let Some(root) = std::env::var_os("IMPARO_LAB_MTP_CLUSTER_HEAD_DIR")
            .filter(|v| !v.is_empty())
        {
            let root = PathBuf::from(root);
            for (name, size, expected) in [
                (
                    "centroids.f32",
                    2_097_152_u64,
                    "d293fc2fc2b68dea9716cc6cad81c4847084640d393962c637ef593415aa68c7",
                ),
                (
                    "ordering.u32",
                    1_048_576_u64,
                    "2d4a619b6fdf687972daaf298bf5ce341f4e2f1d05c20de10c77684e20481b07",
                ),
            ] {
                let path = root.join(name);
                let mut file = File::open(&path)
                    .map_err(|e| format!("MTP cluster head {}: {e}", path.display()))?;
                if file.metadata().map_err(|e| e.to_string())?.len() != size {
                    return Err(format!(
                        "MTP cluster head file size changed: {}",
                        path.display()
                    ));
                }
                let mut bytes = vec![0_u8; size as usize];
                file.read_exact(&mut bytes)
                    .map_err(|e| format!("MTP cluster head {}: {e}", path.display()))?;
                let actual = crate::identity::hex(&Sha256::digest(&bytes));
                if actual != expected {
                    return Err(format!(
                        "MTP cluster head checksum mismatch: {}",
                        path.display()
                    ));
                }
            }
        }
        let document = imparo_gguf::read(&draft).map_err(|e| e.to_string())?;
        Ok((draft.clone(), Self { document, draft }))
    }
    // speculative.rs calls every drafter's methods through one match, so these keep the
    // shared signature whether or not this drafter needs `self` or can fail.
    #[allow(clippy::unused_self)]
    pub(crate) fn can_start(
        &self,
        start: usize,
        limit: usize,
        capacity: usize,
    ) -> bool {
        let cell = crate::prefill_batch().max(1);
        let remaining = limit.saturating_sub(1);
        start > 0
            && remaining >= BLOCK
            && start.checked_add(BLOCK).is_some_and(|end| end <= capacity)
            && (BLOCK <= cell - start % cell
                || bridge(start, remaining, capacity, cell))
    }
    #[allow(clippy::unused_self)]
    pub(crate) fn can_resume(&self, _prompt: &[u32], _start: usize) -> bool {
        false
    }
    #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
    pub(crate) fn run_cached<A: crate::Architecture>(
        &self,
        _target: &mut crate::Workflow<A>,
        _prompt: &[u32],
        _start: usize,
        _run: &mut crate::speculative::DraftRun<'_>,
    ) -> Result<bool, String> {
        Ok(false)
    }
    pub(crate) fn run<A: crate::Architecture>(
        &self,
        target: &mut crate::Workflow<A>,
        run: &mut crate::speculative::DraftRun<'_>,
    ) -> Result<(), String> {
        if !cfg!(feature = "cuda-speculative")
            || !A::DEVICE_ALL_LOGITS
            || !A::DEVICE_PREFIX_VERIFICATION
            || (!crate::e4b_retained_decode_policy_enabled()
                && std::env::var("IMPARO_KV_HISTORY_LAB").as_deref() != Ok("1"))
        {
            return Err("MTP requires device all-position/prefix verification and IMPARO_KV_HISTORY_LAB=1".into());
        }
        if target.state.host_forward
            || target
                .state
                .layer_outputs
                .as_ref()
                .is_some_and(super::layer_outputs::LayerOutputCapture::attached)
        {
            return Err("MTP requires its exclusive CUDA target".into());
        }
        target.ensure_gpu_ready()?;
        let ring_batch = if target.state.kv_ring_batch == 0 {
            crate::prefill_batch()
        } else {
            target.state.kv_ring_batch
        };
        let descriptor = load_descriptor(
            &self.document,
            &target.weights,
            crate::speculative::drafter_offset(&target.weights, &self.draft)?,
            &target.plan,
            target.state.kv_rt.capacity,
            ring_batch,
        )?;
        // Mutable workflow scope retains the mapping and target owner. Provider is
        // !Send and cannot escape this callback; teardown runs on success and error.
        let mut provider = unsafe { Provider::attach(target, descriptor)? };
        let result = run(target, &mut provider);
        let teardown = provider.finish();
        if teardown.is_ok() {
            target.state.layer_outputs = None;
        }
        match (result, teardown) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(e), Ok(())) => Err(e),
            (Ok(()), Err(e)) => Err(format!("MTP teardown: {e}")),
            (Err(e), Err(c)) => Err(format!("{e}; MTP teardown: {c}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TREE4_PARENTS, TREE4_DELAYED_PARENTS, tree4_fits, tree4_from_chain};

    #[test]
    fn selector_never_emits_a_candidate_outside_the_verified_pair() {
        assert_eq!(super::selected_chain(&[7, 9, 7, 8], 10).unwrap(), [7, 9]);
        assert_eq!(super::selected_chain(&[8, 6, 7, 8], 10).unwrap(), [8, 6]);
        for ids in [[6, 9, 7, 8], [7, 9, 7, 7], [7, 10, 7, 8], [7, 9, 7, 10]] {
            assert!(super::selected_chain(&ids, 10).is_err());
        }
    }

    #[test]
    fn cost_domains_keep_boundary_fallback_apart_from_common_short_work() {
        for rows in [3, 4] {
            assert_eq!(super::cost_domain(541, rows), 1);
            assert_eq!(super::cost_domain(542, rows), 2);
            assert_eq!(super::cost_domain(543, rows), 2);
            assert_eq!(super::cost_domain(544, rows), 1);
            assert_eq!(super::cost_domain(1023, rows), 3);
        }
        assert_eq!(super::cost_domain(542, 2), 1);
    }

    #[test]
    fn tree4_preserves_the_chain_and_adds_only_the_first_sibling() {
        let tree = tree4_from_chain(10, &[11, 12], 20, &[], false).unwrap();
        assert_eq!(tree.tokens, [10, 11, 12, 20]);
        assert_eq!(tree.parents, TREE4_PARENTS);
        // Equal token IDs under different parents remain distinct candidates.
        assert!(tree4_from_chain(10, &[11, 12], 12, &[], false).is_some());
        assert!(tree4_from_chain(10, &[11, 12], 11, &[], false).is_none());
        for stop in [11, 12, 20] {
            assert!(tree4_from_chain(10, &[11, 12], 20, &[stop], false).is_none());
        }
    }

    #[test]
    fn delayed_tree4_preserves_two_steps_and_deduplicates_only_the_second_sibling() {
        let tree = tree4_from_chain(10, &[11, 12], 20, &[], true).unwrap();
        assert_eq!(tree.tokens, [10, 11, 12, 20]);
        assert_eq!(tree.parents, TREE4_DELAYED_PARENTS);
        assert!(tree4_from_chain(10, &[11, 12], 11, &[], true).is_some());
        assert!(tree4_from_chain(10, &[11, 12], 12, &[], true).is_none());
        for stop in [11, 12, 20] {
            assert!(tree4_from_chain(10, &[11, 12], 20, &[stop], true).is_none());
        }
    }

    #[test]
    fn tree4_never_extends_beyond_the_admitted_short_schedule_or_cell() {
        assert!(tree4_fits(512, 1024, 512));
        assert!(tree4_fits(765, 1024, 512));
        for (start, capacity, cell) in [
            (511, 1024, 512), (766, 1024, 512), (512, 2048, 512),
            (765, 1024, 256), (512, 1024, 0),
        ] {
            assert!(!tree4_fits(start, capacity, cell));
        }
    }
}
