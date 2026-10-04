//! DSpark model adapter. Metadata and tensor semantics stay out of the shared scheduler.
//! The drafter's description is read from its file and names no backend; each backend
//! admits the geometry it serves. `DsparkProvider` drives a drafter session: CUDA's native
//! session under `cuda-speculative`, otherwise the drafter's forward on the active backend's
//! ops (`dspark_forward`).
use crate::weights::Tensor;
use crate::{Model, speculative::DraftProvider, speculative::TreeBudget};
#[cfg(feature = "cuda-speculative")]
use imparo_cuda::dspark::{Config, Layer};
use imparo_gguf::{Document, MetadataValue, Scalar};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

/// One drafter tensor, addressed in the paired weight mapping. The target's tensors keep
/// their offsets; the drafter's are moved by the offset its file is mapped at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftTensor {
    pub offset: u64,
    pub bytes: u64,
    pub ggml_type: u32,
    /// GGUF order: `dims[0]` is a projection's input width.
    pub dims: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct DraftLayer {
    pub attn_norm: DraftTensor,
    pub q: DraftTensor,
    pub k: DraftTensor,
    pub v: DraftTensor,
    pub o: DraftTensor,
    pub q_norm: DraftTensor,
    pub k_norm: DraftTensor,
    pub ffn_norm: DraftTensor,
    pub gate: DraftTensor,
    pub up: DraftTensor,
    pub down: DraftTensor,
}

/// A DSpark drafter as its file declares it. Every width comes from the metadata and the
/// tensor list, so a drafter whose attention width is not its hidden width reads the same way.
#[derive(Clone, Debug)]
pub struct DsparkDescriptor {
    pub hidden: u32,
    pub ffn: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub context_length: u32,
    pub rank: u32,
    pub vocab: u32,
    /// The target's residual width: one tapped layer's share of `fc`'s input.
    pub target_hidden: u32,
    pub mask_token: u32,
    pub eps: f32,
    pub rope_theta: f32,
    /// 0-based target layers whose residual output feeds the drafter. The file stores each
    /// as the index of the layer that output is the input of.
    pub target_layers: Vec<u32>,
    /// The target's embedding, and its output head (the embedding again when tied).
    pub embedding: DraftTensor,
    pub head: DraftTensor,
    pub fc: DraftTensor,
    pub enc_norm: DraftTensor,
    pub out_norm: DraftTensor,
    pub markov1: DraftTensor,
    pub markov2: DraftTensor,
    pub confidence: DraftTensor,
    pub confidence_bias: DraftTensor,
    pub layers: Vec<DraftLayer>,
}

impl DsparkDescriptor {
    /// The confidence head as floats, read from `weights`, the mapping this descriptor
    /// addresses: its `hidden + rank` weights, then its bias.
    ///
    /// # Errors
    /// When a tensor falls outside the mapping, is not a float type, or holds a value that is
    /// not finite.
    pub fn confidence_head(
        &self,
        weights: &crate::weights::Weights,
    ) -> Result<Vec<f32>, String> {
        let want = self.hidden as usize + self.rank as usize + 1;
        let mut head = Vec::with_capacity(want);
        for t in [&self.confidence, &self.confidence_bias] {
            span(t.offset, t.bytes, 1, weights.byte_len())?;
            let at = usize::try_from(t.offset)
                .map_err(|_| "DSpark confidence head address overflow")?;
            let len = usize::try_from(t.bytes)
                .map_err(|_| "DSpark confidence head size overflow")?;
            // SAFETY: `span` keeps these bytes inside the live, immutable mapping.
            let bytes =
                unsafe { std::slice::from_raw_parts(weights.base_ptr().add(at), len) };
            match t.ggml_type {
                0 => head.extend(
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                ),
                1 => head.extend(bytes.chunks_exact(2).map(|b| {
                    crate::weights::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
                })),
                other => {
                    return Err(format!(
                        "DSpark confidence head: ggml type {other} is not a float type"
                    ));
                }
            }
        }
        if head.len() != want {
            return Err(format!(
                "DSpark confidence head: {} values, the head needs {want}",
                head.len()
            ));
        }
        if head.iter().any(|x| !x.is_finite()) {
            return Err("DSpark confidence head: a value is not finite".into());
        }
        Ok(head)
    }

    /// What the target's plan must carry for this drafter to run beside it.
    #[must_use]
    pub fn drafter_plan(&self) -> crate::DrafterPlan {
        crate::DrafterPlan {
            layers: self.layers.len() as u32,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            taps: self.target_layers.clone(),
        }
    }

    /// (offset, bytes) of every drafter tensor in the paired mapping: the spans the weight
    /// placement adds past the target's own. The embedding and the head are the target's.
    #[must_use]
    pub fn appended_spans(&self) -> Vec<(u64, u64)> {
        let mut spans: Vec<(u64, u64)> = [
            &self.fc,
            &self.enc_norm,
            &self.out_norm,
            &self.markov1,
            &self.markov2,
            &self.confidence,
            &self.confidence_bias,
        ]
        .iter()
        .map(|t| (t.offset, t.bytes))
        .collect();
        for l in &self.layers {
            for t in [
                &l.attn_norm,
                &l.q,
                &l.k,
                &l.v,
                &l.o,
                &l.q_norm,
                &l.k_norm,
                &l.ffn_norm,
                &l.gate,
                &l.up,
                &l.down,
            ] {
                spans.push((t.offset, t.bytes));
            }
        }
        spans
    }
}

/// `dflash.*` keys the reader interprets.
const READ_KEYS: [&str; 12] = [
    "dflash.embedding_length",
    "dflash.feed_forward_length",
    "dflash.attention.head_count",
    "dflash.attention.head_count_kv",
    "dflash.attention.key_length",
    "dflash.attention.value_length",
    "dflash.attention.layer_norm_rms_epsilon",
    "dflash.rope.freq_base",
    "dflash.block_size",
    "dflash.block_count",
    "dflash.context_length",
    "dflash.target_layers",
];
/// Keys that select another layout: a bonus anchor slot, a causal block, no confidence head.
/// Absent, or holding this value, they leave the layout this reader reads.
const DEFAULT_KEYS: [(&str, bool); 3] = [
    ("dflash.sample_from_anchor", true),
    ("dflash.attention.causal", false),
    ("dflash.has_confidence_head", true),
];
const TOP_TENSORS: [&str; 7] = [
    "fc.weight",
    "enc.output_norm.weight",
    "output_norm.weight",
    "markov_w1.weight",
    "markov_w2.weight",
    "conf_proj.weight",
    "conf_proj.bias",
];
const LAYER_TENSORS: [&str; 11] = [
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "attn_q_norm",
    "attn_k_norm",
    "ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

fn uint(doc: &Document, key: &str) -> Result<u32, String> {
    u32::try_from(
        doc.unsigned_value(key)
            .ok_or_else(|| format!("DSpark missing {key}"))?,
    )
    .map_err(|_| format!("DSpark {key} exceeds u32"))
}
fn number(doc: &Document, key: &str) -> Result<f32, String> {
    match doc.metadata.get(key) {
        Some(MetadataValue::Scalar(Scalar::Float(x))) if x.is_finite() && *x > 0.0 => {
            let v = *x as f32;
            if v.is_finite() {
                Ok(v)
            } else {
                Err(format!("DSpark float overflow {key}"))
            }
        }
        _ => Err(format!("DSpark invalid float {key}")),
    }
}
/// Bytes a tensor of this type and shape occupies, from the GGUF layout table.
fn tensor_bytes(kind: u32, dims: &[u64]) -> Result<u64, String> {
    if dims.is_empty() || dims.contains(&0) {
        return Err("DSpark empty tensor".into());
    }
    let layout = imparo_gguf::tensor_layout(kind)
        .map_err(|e| format!("DSpark tensor type {kind}: {e}"))?;
    if dims[0] % layout.block_elements != 0 {
        return Err("DSpark tensor rows are not whole blocks".into());
    }
    dims.iter()
        .try_fold(1u64, |x, d| x.checked_mul(*d))
        .and_then(|n| (n / layout.block_elements).checked_mul(layout.block_bytes))
        .ok_or_else(|| "DSpark tensor overflow".into())
}
fn span(off: u64, n: u64, align: u64, limit: u64) -> Result<(), String> {
    if !align.is_power_of_two()
        || off % align != 0
        || n == 0
        || off.checked_add(n).is_none_or(|e| e > limit)
    {
        Err("DSpark tensor span/alignment invalid".into())
    } else {
        Ok(())
    }
}
/// Read a drafter appended to a target's weight mapping.
///
/// `target` is the target's tensor table (offsets in the mapping), `mapping_bytes` the
/// mapping's length and `draft_base` where the drafter's file starts in it. Only a
/// checksummed pairing establishes that the two files are a trained pair; valid geometry
/// alone cannot.
///
/// # Errors
/// When the file breaks the DSpark contract or does not fit the mapping, with the reason.
pub fn read_descriptor(
    doc: &Document,
    target: &BTreeMap<String, Tensor>,
    mapping_bytes: u64,
    draft_base: u64,
    target_layer_count: u32,
    mask_token: u32,
) -> Result<DsparkDescriptor, String> {
    if doc.string_value("general.architecture") != Some("dflash")
        || doc.alignment < 4
        || doc.data_offset % doc.alignment != 0
    {
        return Err("DSpark requires valid dflash GGUF".into());
    }
    span(draft_base, doc.file_size, doc.alignment, mapping_bytes)?;
    if draft_base == 0
        || target.values().any(|t| {
            (t.offset as u64)
                .checked_add(t.bytes as u64)
                .is_none_or(|e| e > draft_base)
        })
    {
        return Err("DSpark appended region overlaps target tensors".into());
    }
    for (key, value) in &doc.metadata {
        if !key.starts_with("dflash.") || READ_KEYS.contains(&key.as_str()) {
            continue;
        }
        let Some(&(_, default)) = DEFAULT_KEYS.iter().find(|(k, _)| *k == key.as_str())
        else {
            return Err(format!("DSpark metadata {key} is not read by this adapter"));
        };
        let holds_default = match value {
            MetadataValue::Scalar(Scalar::Bool(b)) => *b == default,
            MetadataValue::Scalar(Scalar::String(s)) => *s == default.to_string(),
            _ => false,
        };
        if !holds_default {
            return Err(format!(
                "DSpark {key} other than {default} is not read by this adapter"
            ));
        }
    }
    let h = uint(doc, "dflash.embedding_length")?;
    let f = uint(doc, "dflash.feed_forward_length")?;
    let heads = uint(doc, "dflash.attention.head_count")?;
    let kv_heads = uint(doc, "dflash.attention.head_count_kv")?;
    let d = uint(doc, "dflash.attention.key_length")?;
    let block_size = uint(doc, "dflash.block_size")?;
    let nl = uint(doc, "dflash.block_count")?;
    let context_length = uint(doc, "dflash.context_length")?;
    if uint(doc, "dflash.attention.value_length")? != d
        || [h, f, heads, kv_heads, d, nl].contains(&0)
        || heads % kv_heads != 0
        || block_size < 2
    {
        return Err("DSpark geometry invalid".into());
    }
    let mut seen = BTreeSet::new();
    for t in &doc.tensors {
        let known = TOP_TENSORS.contains(&t.name.as_str())
            || t.name
                .strip_prefix("blk.")
                .and_then(|rest| rest.split_once('.'))
                .is_some_and(|(l, rest)| {
                    l.parse::<u32>().is_ok_and(|n| n < nl && n.to_string() == l)
                        && rest
                            .strip_suffix(".weight")
                            .is_some_and(|s| LAYER_TENSORS.contains(&s))
                });
        if !known {
            return Err(format!(
                "DSpark tensor {} is not read by this adapter",
                t.name
            ));
        }
        if !seen.insert(t.name.as_str()) {
            return Err(format!("DSpark tensor {} appears twice", t.name));
        }
    }
    let Some(MetadataValue::Array(a)) = doc.metadata.get("dflash.target_layers") else {
        return Err("DSpark target feature layers missing".into());
    };
    if a.element_count == 0
        || a.element_count > 16
        || a.element_count != a.preview.len() as u64
    {
        return Err("DSpark target feature array missing/truncated".into());
    }
    let mut target_layers = Vec::new();
    for v in &a.preview {
        let one = match v {
            Scalar::Unsigned(x) => *x,
            Scalar::Signed(x) => {
                u64::try_from(*x).map_err(|_| "negative feature layer")?
            }
            _ => return Err("noninteger feature layer".into()),
        };
        if one == 0 || one > u64::from(target_layer_count) {
            return Err("target feature layer out of range".into());
        }
        target_layers.push((one - 1) as u32);
    }
    if target_layers.windows(2).any(|w| w[0] >= w[1]) {
        return Err("DSpark feature layers must be strictly increasing".into());
    }
    let from_target = |t: &Tensor| DraftTensor {
        offset: t.offset as u64,
        bytes: t.bytes as u64,
        ggml_type: t.ggml_type,
        dims: vec![t.ne[0], t.ne[1]],
    };
    let emb = target
        .get("token_embd.weight")
        .ok_or("target embedding absent")?;
    if emb.n_dims != 2 || emb.ne[0] == 0 || emb.ne[1] == 0 {
        return Err("target embedding shape invalid".into());
    }
    let target_hidden =
        u32::try_from(emb.ne[0]).map_err(|_| "target width overflow")?;
    let vocab = u32::try_from(emb.ne[1]).map_err(|_| "vocabulary overflow")?;
    if mask_token >= vocab {
        return Err("DSpark mask/vocabulary invalid".into());
    }
    let embedding = from_target(emb);
    let head = match target.get("output.weight") {
        None => embedding.clone(),
        Some(o) if o.n_dims == 2 && o.ne[..2] == emb.ne[..2] => from_target(o),
        Some(_) => return Err("target output head and embedding shapes differ".into()),
    };
    let markov = doc
        .tensor("markov_w1.weight")
        .ok_or("missing DSpark tensor markov_w1.weight")?;
    let rank = match markov.dimensions.as_slice() {
        [r, _] if *r > 0 => u32::try_from(*r).map_err(|_| "rank overflow")?,
        _ => return Err("Markov shape invalid".into()),
    };
    let tensor = |name: &str, shape: &[u64]| -> Result<DraftTensor, String> {
        let t = doc
            .tensor(name)
            .ok_or_else(|| format!("missing DSpark tensor {name}"))?;
        let n = tensor_bytes(t.ggml_type, shape)?;
        if t.dimensions != shape
            || t.byte_size != n
            || doc.data_offset.checked_add(t.relative_offset) != Some(t.absolute_offset)
        {
            return Err(format!("DSpark tensor contract mismatch {name}"));
        }
        span(t.absolute_offset, n, doc.alignment, doc.file_size)?;
        let offset = draft_base
            .checked_add(t.absolute_offset)
            .ok_or("relocation overflow")?;
        span(offset, n, doc.alignment, mapping_bytes)?;
        Ok(DraftTensor {
            offset,
            bytes: n,
            ggml_type: t.ggml_type,
            dims: shape.to_vec(),
        })
    };
    let (hw, ffw, dw) = (u64::from(h), u64::from(f), u64::from(d));
    let qw = u64::from(heads) * dw;
    let kw = u64::from(kv_heads) * dw;
    let (rw, vw) = (u64::from(rank), u64::from(vocab));
    let feature_width = (target_layers.len() as u64)
        .checked_mul(u64::from(target_hidden))
        .ok_or("feature width overflow")?;
    let layers = (0..nl)
        .map(|l| {
            let w =
                |s: &str, shape: &[u64]| tensor(&format!("blk.{l}.{s}.weight"), shape);
            Ok(DraftLayer {
                attn_norm: w("attn_norm", &[hw])?,
                q: w("attn_q", &[hw, qw])?,
                k: w("attn_k", &[hw, kw])?,
                v: w("attn_v", &[hw, kw])?,
                o: w("attn_output", &[qw, hw])?,
                q_norm: w("attn_q_norm", &[dw])?,
                k_norm: w("attn_k_norm", &[dw])?,
                ffn_norm: w("ffn_norm", &[hw])?,
                gate: w("ffn_gate", &[hw, ffw])?,
                up: w("ffn_up", &[hw, ffw])?,
                down: w("ffn_down", &[ffw, hw])?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(DsparkDescriptor {
        hidden: h,
        ffn: f,
        heads,
        kv_heads,
        head_dim: d,
        block_size,
        context_length,
        rank,
        vocab,
        target_hidden,
        mask_token,
        eps: number(doc, "dflash.attention.layer_norm_rms_epsilon")?,
        rope_theta: number(doc, "dflash.rope.freq_base")?,
        fc: tensor("fc.weight", &[feature_width, hw])?,
        enc_norm: tensor("enc.output_norm.weight", &[hw])?,
        out_norm: tensor("output_norm.weight", &[hw])?,
        markov1: tensor("markov_w1.weight", &[rw, vw])?,
        markov2: tensor("markov_w2.weight", &[rw, vw])?,
        confidence: tensor("conf_proj.weight", &[hw + rw])?,
        confidence_bias: tensor("conf_proj.bias", &[1])?,
        target_layers,
        embedding,
        head,
        layers,
    })
}

/// A drafter in the form the CUDA operators read.
#[cfg(feature = "cuda-speculative")]
pub struct CudaDescriptor {
    pub config: Config,
    pub layers: Vec<Layer>,
    pub target_layers: Vec<u32>,
}

/// What the CUDA drafter serves: 32 / 8 heads x 64, block 9, Q8_0 matrices, a tied Q8_0
/// embedding, and index arithmetic that fits i32.
#[cfg(feature = "cuda-speculative")]
fn cuda_descriptor(
    d: &DsparkDescriptor,
    kv_capacity: u32,
    batch_capacity: u32,
) -> Result<CudaDescriptor, String> {
    let (h, f, m) = (d.hidden, d.ffn, d.block_size);
    if d.heads != 32
        || d.kv_heads != 8
        || d.head_dim != 64
        || m != 9
        || h != d.heads * d.head_dim
        || f % 32 != 0
        || d.layers.len() > 64
    {
        return Err(
            "DSpark CUDA capability requires Q8 D64/32:8 heads and block9".into(),
        );
    }
    if kv_capacity < m || kv_capacity > d.context_length || batch_capacity < m {
        return Err("DSpark capacity invalid".into());
    }
    let emb = &d.embedding;
    if emb.ggml_type != 8 || d.target_hidden != h {
        return Err("DSpark requires compatible canonical Q8 shared embedding".into());
    }
    if emb.bytes != tensor_bytes(8, &[u64::from(h), u64::from(d.vocab)])? {
        return Err("target embedding byte count mismatch".into());
    }
    if emb.offset % 4 != 0 {
        return Err("DSpark tensor span/alignment invalid".into());
    }
    if d.head != *emb {
        return Err("DSpark requires tied target output embedding".into());
    }
    if d.rank % 32 != 0 {
        return Err("Q8 Markov rank invalid".into());
    }
    let fw = h
        .checked_mul(d.target_layers.len() as u32)
        .ok_or("feature width overflow")?;
    let cw = h.checked_add(d.rank).ok_or("confidence width overflow")?;
    for width in [h, f, d.vocab, d.rank, fw, cw] {
        if u64::from(batch_capacity) * u64::from(width) > i32::MAX as u64 {
            return Err("DSpark index capacity exceeded".into());
        }
    }
    if u64::from(kv_capacity) * u64::from(d.kv_heads * d.head_dim) > i32::MAX as u64 {
        return Err("DSpark KV index capacity exceeded".into());
    }
    let want = |t: &DraftTensor, kind: u32, name: &str| {
        if t.ggml_type == kind {
            Ok(t.offset)
        } else {
            Err(format!("DSpark tensor contract mismatch {name}"))
        }
    };
    let config = Config {
        embedding: emb.offset,
        fc: want(&d.fc, 8, "fc.weight")?,
        enc_norm: want(&d.enc_norm, 0, "enc.output_norm.weight")?,
        out_norm: want(&d.out_norm, 0, "output_norm.weight")?,
        markov1: want(&d.markov1, 8, "markov_w1.weight")?,
        markov2: want(&d.markov2, 8, "markov_w2.weight")?,
        confidence: want(&d.confidence, 1, "conf_proj.weight")?,
        confidence_bias: want(&d.confidence_bias, 0, "conf_proj.bias")?,
        hidden: h,
        ffn: f,
        vocab: d.vocab,
        heads: d.heads,
        kv_heads: d.kv_heads,
        head_dim: d.head_dim,
        rank: d.rank,
        block_size: m,
        mask_token: d.mask_token,
        kv_capacity,
        batch_capacity,
        target_hidden: h,
        eps: d.eps,
        rope_theta: d.rope_theta,
    };
    let layers = d
        .layers
        .iter()
        .enumerate()
        .map(|(l, x)| {
            let w = |t: &DraftTensor, kind: u32, s: &str| {
                want(t, kind, &format!("blk.{l}.{s}.weight"))
            };
            Ok(Layer {
                attn_norm: w(&x.attn_norm, 0, "attn_norm")?,
                q: w(&x.q, 8, "attn_q")?,
                k: w(&x.k, 8, "attn_k")?,
                v: w(&x.v, 8, "attn_v")?,
                o: w(&x.o, 8, "attn_output")?,
                qn: w(&x.q_norm, 0, "attn_q_norm")?,
                kn: w(&x.k_norm, 0, "attn_k_norm")?,
                ffn_norm: w(&x.ffn_norm, 0, "ffn_norm")?,
                gate: w(&x.gate, 8, "ffn_gate")?,
                up: w(&x.up, 8, "ffn_up")?,
                down: w(&x.down, 8, "ffn_down")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(CudaDescriptor {
        config,
        layers,
        target_layers: d.target_layers.clone(),
    })
}

fn boundary_resume_enabled() -> bool {
    crate::lfm_retained_domain() != 0
        || std::env::var("IMPARO_LAB_DRAFT_BOUNDARY_RESUME").as_deref() == Ok("1")
}

/// Whether the provider this build serves requests with bridges a prefill cell's edge: Metal's
/// always does (`DsparkProvider::<BackendSession>::attach`), CUDA's only under
/// `IMPARO_LAB_DRAFT_BOUNDARY_RESUME=1`. A request's admission must read the same answer as the
/// provider that will serve it.
fn served_provider_bridges() -> bool {
    !cfg!(feature = "cuda-speculative") || boundary_resume_enabled()
}

/// WHETHER A REQUEST MAY START DRAFTING at `start` (its prompt's end) with `limit` tokens to generate.
///
/// ```text
///   the first block fits before the cell's edge          -> admitted
///   it does not, and the provider bridges edges          -> admitted: one-token steps to the edge,
///                                                           then the first block (can_bridge_cell)
///   it does not, and the provider does not bridge        -> refused for the whole reply
/// ```
///
/// MEASURED (evidence 2026-09-17-step3-ngram-in-the-budget.md, the goal test): admission read the
/// CUDA switch on Metal, so a prompt ending within one block of a 512-token cell edge ran its whole
/// reply without the drafter (2981 tokens at 45 tok/s where the fixed tree ran the same turn at
/// 124), and the next turn could not resume the drafter's history either.
fn start_admits(
    start: usize,
    limit: usize,
    capacity: usize,
    block: usize,
    cell: usize,
    bridges: bool,
) -> bool {
    let remaining = limit.saturating_sub(1);
    block >= 2
        && start > 1
        && remaining >= block
        && start.checked_add(block).is_some_and(|end| end <= capacity)
        && (block <= cell - start % cell
            || (bridges && can_bridge_cell(start, remaining, capacity, block, cell)))
}

// Only a short cell tail is recoverable. Reserve a full draft block after the
// bridge so every accepted feature append retains native's start+n+M capacity.
fn can_bridge_cell(
    start: usize,
    remaining: usize,
    capacity: usize,
    block: usize,
    cell: usize,
) -> bool {
    if block < 2 || cell < block {
        return false;
    }
    let distance = cell - start % cell;
    distance < block
        && distance
            .checked_add(block)
            .is_some_and(|needed| needed <= remaining)
        && start
            .checked_add(distance)
            .and_then(|next| next.checked_add(block))
            .is_some_and(|end| end <= capacity)
}

#[cfg(feature = "cuda-speculative")]
// A named callback lets the tree workflow distinguish this native feature
// subscriber from arbitrary layer-output observers before skipping host callbacks.
pub(crate) fn capture_layer_output(
    layer: u32,
    start: u32,
    rows: u32,
    source: imparo_backend::BufId,
) -> Result<(), String> {
    unsafe { imparo_cuda::dspark::capture_layer(layer, start, rows, source as u32) }
}

/// A DSpark drafter on one backend: the operations `DsparkProvider` drives. The provider keeps
/// the history and the verify rules; the session keeps the drafter's state.
pub trait DsparkSession {
    /// Drafted ids per block, the anchor excluded.
    fn drafts(&self) -> usize;
    /// Forgets the drafter's history and stops capturing.
    ///
    /// # Errors
    /// When the backend fails.
    fn reset(&mut self) -> Result<(), String>;
    /// Starts or stops writing the target's tapped rows on its forwards.
    ///
    /// # Errors
    /// When the backend fails.
    fn capture(&mut self, enabled: bool) -> Result<(), String>;
    /// The tapped rows of the target's last forward, its first `n`, into the drafter's caches at
    /// positions `start..start + n`.
    ///
    /// # Errors
    /// When those rows are not the last forward's, or the backend fails.
    fn append(&mut self, start: usize, n: usize) -> Result<(), String>;
    /// After a tree verify: the accepted path's rows moved to the first `path.len()` rows, so
    /// `append` reads them as a chain.
    ///
    /// # Errors
    /// When the path is not a path of the last forward's rows, or the backend fails.
    fn compact(&mut self, path: &[i32]) -> Result<(), String>;
    /// One block at `start` from `anchor`: `drafts()` ids.
    ///
    /// # Errors
    /// When the block does not fit the target, or the backend fails.
    fn generate(
        &mut self,
        target: &mut dyn Model,
        start: usize,
        anchor: u32,
    ) -> Result<Vec<u32>, String>;
    /// A backend-owned tree, if available. Its session proves its own row,
    /// capacity and cell limits; explicit widths may use the shared fixed selector.
    /// It is not repriced by the adaptive candidate budget.
    /// Errors propagate without silently changing the verification path.
    fn tree(
        &self,
        _start: usize,
        _anchor: u32,
        _chain: &[u32],
        _stops: &[u32],
    ) -> Result<Option<crate::speculative::DraftTree>, String> {
        Ok(None)
    }
    /// The lowest cache position that holds this request's rows: `generate` reads the committed
    /// cache from there up. 0 for a history that starts at the prompt; the restore point for one
    /// rebuilt there (design 5.5). False when the session cannot bound its read, which then serves
    /// only 0.
    fn set_floor(&mut self, floor: usize) -> bool {
        floor == 0
    }
    /// The candidates a tree over the drafted chain is built from, when the session proposes a
    /// tree; `None` verifies the chain. The provider prices them and builds the tree, because the
    /// acceptance model that prices them lives as long as the provider, not the block.
    ///
    /// # Errors
    /// When the backend fails.
    fn candidates(
        &self,
        _start: usize,
        _anchor: u32,
        _chain: &[u32],
    ) -> Result<Option<Candidates>, String> {
        Ok(None)
    }
    /// The session's actual tree-row limit, independent of backend-wide layouts.
    fn verify_max_rows(&self) -> usize {
        imparo_backend::ROW_LAYOUT_MAX_ROWS
    }
    /// Native trees retain their existing selector. Explicit clock diagnostics
    /// may observe full rounds, but cannot probe rows or seed the shared budget.
    fn passive_tree_costs(&self) -> bool {
        false
    }
    /// A native adapter may price only completed transactions, including its
    /// accepted-path history commit. Early phase callbacks then do not train.
    fn complete_tree_costs(&self) -> bool { false }
    /// A persisted table requires an execution identity, not just a model name.
    fn cost_cache_allowed(&self) -> bool { true }
    fn cost_cache_key(&self) -> Option<(String, u64)> {
        self.cost_cache_allowed().then(|| crate::gpu_support::be().config_key()).flatten()
    }
    /// The actual feature group, which native adapters may fix independently of
    /// the generic drafter's configurable candidate count.
    fn acceptance_top_k(&self) -> u32 { crate::dspark_forward::candidates_per_position() }
    /// A native candidate pool can reuse shared selection at its existing fixed
    /// row budget before CUDA has qualified adaptive prices. None keeps the
    /// backend's established path, including Metal's learned width controller.
    fn fixed_candidate_rows(&self) -> Option<usize> { None }
    fn candidate_depth_limit(&self) -> usize { CHAIN_DEPTH }
    fn has_accept_features(&self) -> bool { true }
    /// Without matching persisted state, a new feature contract starts from the
    /// derived prior rather than importing another backend's offline prior.
    fn fresh_accept_model(&self) -> bool { false }
    /// Free rows an `append` needs past its own rows when no draft follows it, or `None` when
    /// the session takes no such append: its history then ends at the last round that another
    /// round drafted from.
    fn tail_room(&self) -> Option<usize> {
        None
    }
    /// Stops capturing and releases the drafter.
    ///
    /// # Errors
    /// When the backend fails.
    fn detach(&mut self) -> Result<(), String>;
}

/// A round's tree candidates: today's estimate as each one's `q`, and what the acceptance model
/// reads about each, index for index.
pub struct Candidates {
    pub candidates: Vec<crate::speculative::TreeCandidate>,
    /// Native frontiers may have probabilities without the confidence-head features the
    /// acceptance model was trained on. Preserve that absence instead of inventing features.
    pub features: Vec<Option<crate::accept_model::Features>>,
    /// Which source proposed each candidate (design 6.6).
    pub sources: Vec<crate::speculative::NodeSource>,
    /// Native beam nodes remain n-gram branch points even when their local kind is
    /// Sibling. Appended TopK siblings have no native continuation of their own.
    /// None preserves the ordinary drafter's Pick-only branching convention.
    pub ngram_branch_prefix: Option<usize>,
}

/// WHAT ONE ROUND LEAVES BEHIND, so the commit that follows can price a width the round did
/// NOT run. Every prefix of the best-first order is itself a tree, so the accepted walk at
/// `hi` also says what `lo` would have taken -- both sides of the marginal from one round.
struct RoundPlan {
    /// Row -> best-first rank, the anchor -1.
    rank: Vec<i32>,
    /// Prefix sums of Q over the whole best-first order.
    s: Vec<f64>,
    /// The part of `s` the n-gram's and agreed nodes carry.
    s_n: Vec<f64>,
    /// An exploration round: its order is not best-first at the chosen width, so it teaches no gain.
    explored: bool,
    /// Row -> its parent row, the anchor -1. Design 11.3's label rule needs it: a node is
    /// labelled exactly when its PARENT is on the accepted path.
    parents: Vec<i32>,
    /// The marginal's reference width (`gain_reference`) and the width this round ran.
    lo: usize,
    hi: usize,
    /// The narrowest width the budget was offered this round: the bottom of the marginal BELOW
    /// the reference, which every round reveals.
    first: usize,
    /// The context the round priced at. The gain is a function of it, so the observation
    /// must carry it rather than read whatever the table is pointing at by the time the
    /// commit lands.
    ctx: usize,
    /// Row -> the candidate it holds, the anchor -1.
    node: Vec<i32>,
    /// Every candidate of the round, placed or not, with today's estimate as its `q` -- not the
    /// probability the tree was priced with, which may be the acceptance model's.
    found: Candidates,
    /// The offset today's estimate carried this round: the guard scores today's estimate after it.
    offset: f64,
}

/// One explicitly attached provider, using the target's shared immutable weights.
/// Does not implement Send/Sync; backend calls remain serialized on this thread.
#[allow(clippy::struct_excessive_bools)] // independent per-run switches, not one state
pub struct DsparkProvider<S: DsparkSession> {
    session: S,
    /// Positions the drafter's caches hold.
    capacity: usize,
    /// Rows one append takes.
    batch_capacity: usize,
    live: bool,
    history: usize,
    /// Where the drafter's caches hold this request's rows from: 0, or the restore point of a
    /// history rebuilt there (design 5.5). Its rows below are another stream's, and the drafter
    /// never reads them.
    floor: usize,
    resume_from: Option<usize>,
    tokens: Option<Vec<u32>>,
    boundary_resume: bool,
    boundary_trace: bool,
    bridge_start: Option<usize>,
    boundary_witness_logged: bool,
    /// What a verify of each row count costs, measured from the rounds this provider runs.
    /// `None` for native sessions whose costs are observation-only.
    cost: Option<crate::verify_cost::VerifyCost>,
    native_cost: Option<crate::verify_cost::TreeCostObserver>,
    /// How the stored table is keyed: the backend's own config key, so the measurement
    /// lands beside the tuning it was measured under. `None` when no config was loaded.
    cost_key: Option<(String, u64)>,
    /// The widest start position a verify ran at, recorded in the stored file so a reader
    /// can see what state produced the numbers. Advisory, never part of the key.
    cost_at: usize,
    /// THE WIDTH THIS PROVIDER IS SITTING ON. The budget keeps it unless an alternative
    /// clears it by a margin, so the policy's floor is whatever fixed width it settles on
    /// -- a per-round argmax has no floor and flips on noise.
    hold: usize,
    /// What the last tree left for the commit that follows. The value model's only
    /// uncensored evidence.
    last_plan: Option<RoundPlan>,
    /// DESIGN 11.4'S PER-REQUEST OFFSET on the acceptance estimate. Level 1: it is reset
    /// where the drafter's own context is, because it corrects an estimate ABOUT that
    /// context. `IMPARO_DSPARK_OFFSET=0` turns it off.
    offset: crate::accept_offset::AcceptOffset,
    /// DESIGN 11.4'S REQUEST N-GRAM INDEX. Level 1, same lifetime as the offset. It is fed
    /// from `commit`, which is the one funnel every committed token passes through -- the
    /// prompt's prefill and every round's accepted path alike.
    ngram: crate::ngram::NgramIndex,
    /// DESIGN 11.5B'S STORED TABLE, when the dictionary is on: a separate map with its own bound,
    /// read only where the request's index holds nothing for a context, and taught the request's
    /// committed text keyed by the request's context.
    table: Option<crate::ngram::NgramIndex>,
    /// Tree rounds this provider built, for the exploration row's schedule.
    tree_rounds: u64,
    /// Whether this request's first chain round has printed the key it formed at the anchor.
    key0_said: bool,
    /// Written once. `finish` can be called more than once, and a second save would decay a
    /// table that had already been decayed -- measured as 119 rows becoming 39.
    dictionary_saved: bool,
    /// What the n-gram WOULD have proposed, scored against what the target actually committed,
    /// before it learns the token: tokens judged, then per evidence slot (`NgramTerms::slot`: the
    /// request's 3-token index by count, its copy matches by length, the stored table by count)
    /// `(it proposed, it was right)`. Design 11.4's gate is these numbers, and the n-gram-only
    /// nodes' starting estimate is their slot's precision.
    ngram_score: (u64, [(u64, u64); 8]),
    /// Verifies THIS provider timed. A run that measured nothing -- a plain chain decode,
    /// which is the default -- has nothing to store and must not rewrite the file, or a
    /// stale table would keep renewing itself on every process exit.
    cost_seen: u64,
    /// Rounds this provider has timed, and whether the one in flight is the request's first --
    /// the cold one the cost model does not learn from (`observe_round_fixed`).
    rounds_timed: u64,
    cold_round: bool,
    /// DESIGN 6.6'S LONG-RUN RATE, rho: the tokens per microsecond this process delivers, by
    /// context band. Kept across requests, because it prices time, not text.
    rate: crate::cost_model::LongRunRate,
    /// The round in flight so far, microseconds: its fixed part and its verify. The commit
    /// adds its own time and folds the round into `rate`.
    round_us: f64,
    /// DESIGN 6.6'S ACCEPTANCE MODEL, `None` when switched off. Its coefficients live as long
    /// as the provider; its intercept and its report reset with each request.
    accept: Option<crate::accept_model::AcceptModel>,
    /// The acceptance model as the store held it at attach. Written back unchanged when the model
    /// is switched off, so a run without it does not erase what an earlier run learned.
    stored_accept: Option<imparo_host::AcceptRow>,
    /// This request's rounds the acceptance model priced, and its tree rounds.
    accept_rounds: (u64, u64),
    _serial: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl<S: DsparkSession> DsparkProvider<S> {
    fn new(
        session: S,
        capacity: usize,
        batch_capacity: usize,
        history: usize,
        floor: usize,
        tokens: Option<Vec<u32>>,
        boundary_resume: bool,
    ) -> Self {
        // THE TABLE'S STEPS ARE LEARNED from served rounds (`VerifyCost::new`): a ladder of
        // widths, split where a width inside a step could be chosen.
        // Native CUDA sessions retain fixed selection until their execution costs
        // are qualified; observations must not activate the generic adaptive ladder.
        let mut cost = if session.passive_tree_costs() { None } else {
            crate::verify_cost::VerifyCost::new(session.verify_max_rows())
        };
        let native_cost = ((session.passive_tree_costs() || session.complete_tree_costs())
            && std::env::var("IMPARO_DSPARK_CLOCK").as_deref() == Ok("1"))
            .then(|| crate::verify_cost::TreeCostObserver::new(session.verify_max_rows()))
            .flatten();
        // SEED FROM THE LAST RUN so a restart does not spend its first rounds re-probing.
        // The stored file is keyed the way the tuner's config is keyed, so a retune reads its
        // own file, and it carries the widths the last run split; a file whose widths this
        // build's ladder cannot hold is refused whole rather than mixed.
        // A native adapter supplies its own exact execution/feature identity;
        // passive observations cannot read or write adaptive prices.
        let key = if session.passive_tree_costs() { None }
            else { session.cost_cache_key() };
        // ONE READ of the stored table. The cost side seeds from it only when this build's
        // ladder holds its widths; the long-run rate and the acceptance model are not kernel
        // measurements, so they seed from it either way.
        let stored = key.as_ref()
            .filter(|_| std::env::var("IMPARO_DSPARK_COST_CACHE").as_deref() != Ok("0"))
            .and_then(|(fp, model_bytes)| {
            imparo_host::read_verify_cost(fp, *model_bytes)
        });
        let mut cost_seeded = false;
        if let (Some(cost), Some(table)) = (cost.as_mut(), stored.as_ref()) {
            if cost.seed(&table.steps) {
                cost_seeded = true;
                // THE LAWS ONLY AFTER the anchors took: a law belongs to the class
                // boundaries it was measured under, and those are what `seed` checks.
                cost.seed_laws(&table.laws);
                // and the samples each law's refit trains on (`CostLaw::bank`)
                cost.seed_banks(&table.banks);
                // ARRIVES TRAINED. The gain needs rounds at more than one width, which
                // one short request rarely provides; from the store it is already there.
                let (gxx, gxy, gs) = table.gain;
                cost.seed_gain(gxx, gxy, gs);
                let (nxx, nxy, ns) = table.gain_n;
                cost.seed_gain_n(nxx, nxy, ns);
                cost.seed_gain_below(table.gain_below, table.gain_n_below);
            } else {
                eprintln!(
                    "[imparo] dspark verify cost: the stored table's widths do not fit \
                     this build's ladder; measuring again"
                );
            }
        }
        let stored_accept = stored.as_ref().and_then(|t| t.accept.clone());
        let accept = session.has_accept_features()
            .then(|| accept_start(stored_accept.as_ref(), session.acceptance_top_k(),
                session.fresh_accept_model())).flatten();
        let mut provider = Self {
            session,
            capacity,
            batch_capacity,
            live: true,
            history,
            floor,
            resume_from: (history > 0).then_some(history),
            tokens,
            boundary_resume,
            boundary_trace: std::env::var("IMPARO_DSPARK_BOUNDARY_TRACE").as_deref()
                == Ok("1"),
            bridge_start: None,
            boundary_witness_logged: false,
            cost,
            native_cost,
            cost_at: 0,
            cost_seen: 0,
            rounds_timed: 0,
            cold_round: false,
            // The incumbent starts unset: the first round has nothing to hold and takes the
            // plain argmax, which is also what the cold-start probe wants.
            hold: 0,
            last_plan: None,
            offset: crate::accept_offset::AcceptOffset::default(),
            ngram: crate::ngram::NgramIndex::default(),
            table: dictionary_on().then(|| {
                let mut table = crate::ngram::NgramIndex::with_limit(2 * DICTIONARY_CONTEXTS);
                if let Some((fp, bytes)) = key.as_ref() {
                    let rows = imparo_host::read_dspark_dictionary(fp, *bytes);
                    for (k, t, c) in &rows {
                        table.seed(*k, *t, *c);
                    }
                    if !rows.is_empty() {
                        eprintln!(
                            "[imparo] dspark dictionary: {} rows over {} contexts loaded",
                            rows.len(),
                            table.contexts()
                        );
                    }
                }
                table
            }),
            tree_rounds: 0,
            key0_said: false,
            cost_key: key,
            dictionary_saved: false,
            ngram_score: (0, [(0, 0); 8]),
            rate: {
                let mut rate = crate::cost_model::LongRunRate::default();
                if let Some(table) = stored.as_ref() {
                    rate.seed(&table.rate);
                }
                rate
            },
            round_us: 0.0,
            accept,
            stored_accept,
            accept_rounds: (0, 0),
            _serial: std::marker::PhantomData,
        };
        // A RESUMED provider's index holds what a cold start over the same prompt would have
        // built: the tokens below the resume point were committed by an earlier request, not
        // by this one, and level 1's index is this request's context.
        if let Some(tokens) = provider.tokens.as_ref() {
            provider.ngram.observe_all(tokens);
        }
        if provider.session.complete_tree_costs() {
            eprintln!("[dspark-cost-cache] {}", serde_json::json!({
                "event":"load", "key":provider.cost_key.as_ref().map(|k| &k.0),
                "path":provider.cost_key.as_ref().map(|(fp,n)| imparo_host::verify_cost_path_for(fp,*n)),
                "enabled":std::env::var("IMPARO_DSPARK_COST_CACHE").as_deref()!=Ok("0"),
                "found":stored.is_some(), "cost_seeded":cost_seeded,
                "steps":provider.cost.as_ref().map(crate::verify_cost::VerifyCost::rows),
                "next_probe":provider.cost.as_ref().and_then(crate::verify_cost::VerifyCost::probe),
                "rate":provider.rate.rows(), "top_k":provider.session.acceptance_top_k(),
                "accept_seen":provider.accept.as_ref().map(|m| m.level2().seen),
                "stored_accept_seen":provider.stored_accept.as_ref().map(|r|r.seen),
            }));
        }
        provider
    }

    /// The history this provider's commits leave in the drafter's caches: the tokens at
    /// positions `0..history` and the floor its rows start at, or `None` when it did not track
    /// them. CUDA parks its history natively (`park`) and never reads it back this way.
    #[cfg(not(feature = "cuda-speculative"))]
    fn committed_history(&mut self) -> Option<(Vec<u32>, usize)> {
        self.tokens
            .take()
            .filter(|t| t.len() == self.history)
            .map(|t| (t, self.floor))
    }

    /// What this round taught the value model about the MARGINAL's scale.
    ///
    /// The round ran at `hi` rows and the target reported which of them it took. Because
    /// every prefix of the best-first order is itself a tree, that one walk also says what
    /// would have been accepted at `lo` -- so the round reports BOTH sides of the marginal
    /// without drafting or verifying anything extra.
    ///
    /// Only a round that ran wide can do this. A round at `lo` says nothing about what `hi`
    /// would have bought, which is why the evidence is censored and why the width the policy
    /// sits on decides what it is able to learn next.
    fn learn_margin(&mut self, path: &[i32]) {
        let Some(RoundPlan {
            rank,
            s,
            s_n,
            explored,
            lo,
            hi,
            first,
            ctx,
            node,
            found,
            ..
        }) = self.last_plan.take()
        else {
            return;
        };
        // THE MARGIN NEEDS TWO WIDTHS TO DIFFERENCE. `lo` is 0 until the cost table has an
        // envelope, and `s_at(0)` would index from the wrong end -- so this guard lives here,
        // where the marginal is formed, and not on recording the plan. The offset (11.4) reads
        // the same plan and needs no envelope at all. An exploration round's order is not
        // best-first at its width, so its narrower prefixes are not the trees they stand for.
        if lo < 2 || s.len() < 2 || explored {
            return;
        }
        // Accepted rows inside the first `n` rows, split by source: the drafter's alone, and the
        // n-gram's and agreed ones (design 6.6's g_d and g_n read each their own).
        let accept_at = |n: usize| -> (f64, f64) {
            let cut = i32::try_from(n.saturating_sub(1)).unwrap_or(i32::MAX);
            let (mut d, mut g) = (0.0, 0.0);
            for &row in path.iter().skip(1) {
                let Ok(r) = usize::try_from(row) else { break };
                match rank.get(r) {
                    Some(&q) if q >= 0 && q < cut => {
                        let ngram = node
                            .get(r)
                            .and_then(|&c| usize::try_from(c).ok())
                            .and_then(|c| found.sources.get(c))
                            .is_some_and(|s| {
                                *s != crate::speculative::NodeSource::Drafter
                            });
                        if ngram {
                            g += 1.0;
                        } else {
                            d += 1.0;
                        }
                    }
                    _ => break,
                }
            }
            (d, g)
        };
        let at = |v: &[f64], n: usize| v[(n - 1).min(v.len() - 1)];
        // One marginal, predicted and realised, split by source: `(drafter, n-gram)`, each as
        // `(predicted, realised)`.
        let marginal = |a: usize, b: usize| {
            let ((b_d, b_n), (a_d, a_n)) = (accept_at(b), accept_at(a));
            let p_n = at(&s_n, b) - at(&s_n, a);
            ((at(&s, b) - at(&s, a) - p_n, b_d - a_d), (p_n, b_n - a_n))
        };
        let Some(cost) = self.cost.as_mut() else {
            return;
        };
        // ABOVE THE REFERENCE only a round wider than it reveals the marginal: censored.
        if hi > lo {
            let ((p_d, r_d), (p_n, r_n)) = marginal(lo, hi);
            cost.observe_gain(ctx, p_d, r_d);
            cost.observe_gain_n(ctx, p_n, r_n);
        }
        // BELOW IT every round does: a verify of `hi` rows holds each narrower prefix, so the rows
        // between the narrowest offered width and min(hi, reference) are priced without censoring.
        let top = hi.min(lo);
        if first >= 2 && top > first {
            let (drafter, ngram) = marginal(first, top);
            cost.observe_gain_below(ctx, drafter, ngram);
        }
    }

    /// Design 11.4's offset in logit units, or 0 when the level is switched off. Read once
    /// per round so every node of one tree is built from the same correction.
    fn offset_value(&self) -> f64 {
        if offset_on() {
            self.offset.value()
        } else {
            0.0
        }
    }

    /// DESIGN 11.4'S UPDATE: every node the target actually judged, labelled.
    ///
    /// The label RULE is `speculative::acceptance_labels` -- one function, because level 3's
    /// confidence-head refit reads the same supervision (design 11.6). This only folds the
    /// pairs in.
    fn learn_offset(&mut self, path: &[i32]) {
        if !offset_on() {
            return;
        }
        let Some(plan) = self.last_plan.as_ref() else {
            return;
        };
        let labels = crate::speculative::acceptance_labels(
            &plan.parents,
            &plan.node,
            &plan.found.candidates,
            path,
        );
        for (q, accepted) in labels {
            self.offset.observe(q, accepted);
        }
        if crate::speculative::score_probe() {
            eprintln!(
                "dspark offset b={:.4} seen={}",
                self.offset.value(),
                self.offset.seen()
            );
        }
    }

    /// DESIGN 6.6'S LABELS: at every accepted parent, each of its candidates, placed or not,
    /// against the target's pick there.
    ///
    /// ```text
    ///   path     anchor -> row a -> row b          (the accepted rows)
    ///   parents  anchor, a, b                      each one's candidates are a group
    ///   picks    token of a, token of b, next      next = the target's pick after b
    /// ```
    ///
    /// The label is the candidate carrying the pick, or NONE when no candidate does. A parent
    /// with no candidates teaches nothing.
    fn learn_accept(&mut self, path: &[i32], next: u32) {
        let (Some(model), Some(plan)) = (self.accept.as_mut(), self.last_plan.as_ref())
        else {
            return;
        };
        if plan.found.features.iter().any(Option::is_none) {
            return;
        }
        let cands = &plan.found.candidates;
        let kids = children(cands);
        let of_row = |row: i32| -> Option<usize> {
            let c = *plan.node.get(usize::try_from(row).ok()?)?;
            usize::try_from(c).ok().filter(|&c| c < cands.len())
        };
        for (i, &row) in path.iter().enumerate() {
            let parent = if i == 0 {
                cands.len()
            } else {
                match of_row(row) {
                    Some(c) => c,
                    None => continue,
                }
            };
            let group = &kids[parent];
            if group.is_empty() {
                continue;
            }
            let pick = match path.get(i + 1) {
                Some(&child) => match of_row(child) {
                    Some(c) => cands[c].token,
                    None => continue,
                },
                None => next,
            };
            let features: Vec<_> = group
                .iter()
                .filter_map(|&c| plan.found.features[c])
                .collect();
            let today: Vec<f64> = group
                .iter()
                .map(|&c| crate::accept_offset::shift(cands[c].q, plan.offset))
                .collect();
            let label = group.iter().position(|&c| cands[c].token == pick);
            model.observe(&features, &today, label);
        }
    }

    /// Design 6.6's verdict for one request, printed when it ends: the log loss per labelled node
    /// of the model and of today's estimate, each scored before either learned from the label.
    /// Then the request's intercept and sums start again; the coefficients carry on.
    fn accept_report(&mut self) {
        let Some(m) = self.accept.as_mut() else {
            return;
        };
        let (model_ll, today_ll, nodes) = m.request_losses();
        let (priced, rounds) = std::mem::take(&mut self.accept_rounds);
        if nodes > 0 {
            let (w, seen) = m.state();
            let w: Vec<String> = w.iter().map(|x| format!("{x:.4}")).collect();
            eprintln!(
                "[imparo] dspark accept model_ll={model_ll:.5} today_ll={today_ll:.5} \
                 nodes={nodes} priced={priced}/{rounds} b={:.4} seen={seen} w={}",
                m.intercept(),
                w.join(",")
            );
            // Design 6.6's per-bucket check: what the model predicted for each n-gram bucket's
            // labelled nodes against how many the target took, and the bucket's coefficient.
            for (name, r, weight) in m.bucket_reports() {
                #[allow(clippy::cast_precision_loss)]
                let rate = r.accepted as f64 / r.labels as f64;
                #[allow(clippy::cast_precision_loss)]
                let mean = r.predicted / r.labels as f64;
                eprintln!(
                    "[imparo] dspark accept bucket={name} labels={} predicted={mean:.4} \
                     realised={rate:.4} w={weight:.4}",
                    r.labels
                );
            }
        }
        m.reset_request();
    }

    /// DESIGN 11.5B: the dictionary is written once, when the provider finishes.
    ///
    /// DECAY IS THE EVICTION MECHANISM, NOT A TAX ON EVERY EXIT. Decaying unconditionally was
    /// measured and it erases the table: counts are integers and most contexts are seen once,
    /// so one halving takes them to zero. A 796-context index stored 119 rows, and a second
    /// save took that to 39. Decay now runs only when the table is over its bound, which is
    /// what "eviction by decayed count" means -- a context that keeps appearing outruns its
    /// own decay, one that stopped fades, and nothing fades while there is room.
    fn save_dictionary(&mut self) {
        if self.dictionary_saved {
            return;
        }
        let (Some(table), Some((fp, bytes))) =
            (self.table.as_mut(), self.cost_key.clone())
        else {
            return;
        };
        self.dictionary_saved = true;
        while table.contexts() > DICTIONARY_CONTEXTS {
            let before = table.contexts();
            table.decay();
            if table.contexts() >= before {
                break; // nothing fell out; decaying again would only erase what is left
            }
        }
        let rows: Vec<(u64, u32, u32)> = table.rows().collect();
        match imparo_host::write_dspark_dictionary(&fp, bytes, &rows) {
            Ok(()) => eprintln!(
                "[imparo] dspark dictionary: {} rows over {} contexts stored",
                rows.len(),
                table.contexts()
            ),
            Err(e) => eprintln!("[imparo] dspark dictionary: not stored ({e})"),
        }
    }

    /// Design 11.4's index, scored on the stream it would propose into. Printed when the
    /// request ends, because that is when the denominator is final.
    fn ngram_report(&mut self) {
        let (n, by_source) = std::mem::take(&mut self.ngram_score);
        if n == 0 || !crate::speculative::score_probe() {
            return;
        }
        let (ctx, seen) = self.ngram.size();
        #[allow(clippy::cast_precision_loss)]
        let pct = |x: u64| 100.0 * x as f64 / n as f64;
        const NAMES: [&str; 8] = [
            "request/n1",
            "request/n2+",
            "request/copy4-15",
            "request/copy16+",
            "table/n1",
            "table/n2",
            "table/n3-4",
            "table/n5+",
        ];
        for (name, (matched, right)) in NAMES.iter().zip(by_source) {
            if matched == 0 {
                continue;
            }
            eprintln!(
                "dspark ngram source={name} tokens={n} matched={matched} ({:.1}%) \
                 top1_right={right} ({:.1}%) precision={:.1}% contexts={ctx} observed={seen}",
                pct(matched),
                pct(right),
                if matched == 0 {
                    0.0
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    let p = 100.0 * right as f64 / matched as f64;
                    p
                },
            );
        }
    }

    /// Each source's top-1 precision on the committed stream so far, `(right + 1) / (matched + 2)`:
    /// an n-gram-only node's starting estimate. The pseudo-counts keep a source with no evidence at
    /// one half rather than at 0 or 1, where a logit is unbounded.
    fn ngram_precision(&self) -> [f64; 8] {
        self.ngram_score.1.map(|(matched, right)| {
            #[allow(clippy::cast_precision_loss)]
            let p = (right as f64 + 1.0) / (matched as f64 + 2.0);
            p
        })
    }

    /// The row counts the budget picks between, with what each one costs: the table's
    /// envelope, so a class a wider one beats on price is never offered.
    fn widths(&self) -> Vec<(usize, f64)> {
        self.cost
            .as_ref()
            .map_or_else(Vec::new, crate::verify_cost::VerifyCost::envelope)
    }

    /// How this round's tree is sized, or `None` to verify the chain.
    ///
    /// The BUDGET's cold start is the table's own probe: a class with no measurement has
    /// no price, so the first rounds run the row counts that need one. Only once every
    /// class has a price does design 6.3's ratio have anything to compare.
    fn tree_budget<'w>(
        &self,
        widths: &'w [(usize, f64)],
        start: usize,
        room: usize,
        offset: f64,
        base_rows: usize,
    ) -> Result<Option<TreeBudget<'w>>, String> {
        Ok(match tree_width()? {
            None => None,
            Some(TreeWidth::Fixed(n)) => Some(TreeBudget::Fixed { nodes: n, offset }),
            Some(TreeWidth::Budget) => match self.cost.as_ref() {
                // No table -- the widest verify leaves no width to choose: nothing to probe and
                // nothing to choose from, so the round falls back to the chain.
                None => None,
                Some(cost) => {
                    let probe = cost.probe();
                    if crate::speculative::score_probe() {
                        let env: Vec<String> = widths
                            .iter()
                            .map(|&(n, us)| format!("{n}:{us:.0}"))
                            .collect();
                        // `gain` and `hold` are what the budget's value and hysteresis read this
                        // round, so an offline pass can reproduce the width it chose -- and price
                        // the widths it did not choose under another objective.
                        // `rate_tok_s` and `rate_n` are the band's long-run rate and its rounds,
                        // printed whether or not the rule reads them, so both arms of an A/B
                        // carry the same evidence.
                        eprintln!(
                            "dspark cost probe={} fixed_us={:.0} scale={:.3} gain={:.4} \
                             gain_n={:.4} gain_below={:.4} hold={} rate_tok_s={} rate_n={} env={}",
                            probe.map_or("-".to_string(), |r| r.to_string()),
                            cost.fixed_us().unwrap_or(0.0),
                            cost.scale(),
                            cost.gain(),
                            cost.gain_n(),
                            cost.gain_below().0,
                            self.hold,
                            self.rate.at(start).map_or_else(
                                || "-".to_string(),
                                |r| format!("{:.3}", r * 1e6)
                            ),
                            self.rate.seen(start),
                            env.join(",")
                        );
                    }
                    // A PROBE WIDTH THAT DOES NOT FIT THE CELL waits for a round that has room, and
                    // this round is chosen as if nothing were being probed -- not dropped to the
                    // plain chain, which would make every room-short round pay for the probe.
                    match probe.filter(|&rows| rows <= room) {
                        Some(rows) => Some(TreeBudget::Fixed {
                            nodes: rows,
                            offset,
                        }),
                        None if widths.is_empty() => None,
                        None => Some(TreeBudget::Choose(crate::speculative::Choice {
                            widths,
                            fixed_us: cost.fixed_us().unwrap_or(0.0),
                            gain: cost.gain(),
                            gain_n: cost.gain_n(),
                            gain_below: cost.gain_below(),
                            // AN UNSET INCUMBENT SEEDS AT THE WIDEST OFFERED WIDTH, never
                            // at whatever the first round's argmax happened to pick. Two
                            // reasons, and they are one reason: the evidence is censored,
                            // so only a wide round teaches -- and hysteresis then holds
                            // whatever it is given, which would make one noisy round the
                            // policy's floor for the whole session.
                            hold: match self.hold {
                                0 => widths.iter().map(|&(n, _)| n).max().unwrap_or(0),
                                n => n,
                            },
                            offset,
                            // Until the band holds enough rounds this is `None`, and the
                            // budget keeps the per-round ratio: the rule's cold start.
                            rho: if rate_rule_on() {
                                self.rate.at(start)
                            } else {
                                None
                            },
                            base_rows,
                        })),
                    }
                }
            },
        })
    }
}

fn draft_minimum_remaining(domain: u32, block: usize, lab_bounded: bool) -> usize {
    // All retained LFM domains use the already bounded output commit. Keeping
    // the short domain at a full block needlessly sends its last 3–8 tokens
    // through sequential decode; target tree/cell/capacity checks still apply.
    if domain != 0 || lab_bounded { 3.min(block) } else { block }
}

impl<S: DsparkSession> DraftProvider for DsparkProvider<S> {
    fn tree_proposal(
        &mut self,
        start: usize,
        anchor: u32,
        chain: &[u32],
        stops: &[u32],
    ) -> Result<Option<crate::speculative::DraftTree>, String> {
        // A new round: whatever an earlier round left unfolded (a round kept without learning,
        // `keep_tree`) does not belong to this one.
        self.round_us = 0.0;
        self.last_plan = None;
        // CUDA retains its admitted native frontier by default; explicit smaller
        // widths use the shared fixed selector within that session's bounds.
        // The learned candidate path remains unchanged for other sessions.
        if let Some(tree) = self.session.tree(start, anchor, chain, stops)? {
            return Ok(Some(tree));
        }
        // A TREE NEEDS ITS WIDEST LAYOUT'S SLOTS. Its rows occupy the cache slots after `start`,
        // one per node whatever the node's depth, so within ROW_LAYOUT_MAX_ROWS of the capacity
        // a wide tree would write past the cache; such a round drafts a chain, whose block
        // `can_draft` already bounds. The server reserves the same slots in the pool.
        if start.checked_add(self.session.verify_max_rows())
            .is_none_or(|end| end > self.capacity) {
            self.last_plan = None;
            return Ok(None);
        }
        // THE ROUND'S CONTEXT, before anything is priced. Cost is a function of it, and
        // this is the one place that knows it: the table prices and learns at `start`.
        if let Some(cost) = self.cost.as_mut() {
            cost.set_context(start);
        }
        // THE CELL'S ROOM. A tree verify runs inside one prefill cell, so a tree wider than the room
        // left is refused and the round falls back to the drafter's plain chain. The budget offers
        // only the widths that fit; a fixed width stays fixed.
        let cell = crate::prefill_batch().max(1);
        let room = cell - start % cell;
        let all_widths = self.widths();
        let widths: Vec<(usize, f64)> = all_widths
            .iter()
            .copied()
            .filter(|&(n, _)| n <= room)
            .collect();
        let capped = widths.len() < all_widths.len();
        // THE WIDTH THE GAINS ARE MEASURED FROM. The table offers narrower widths as it finds
        // them; the gains must keep measuring the marginal above the same width, or the value
        // model changes under the cost model's feet (`Choice::base_rows`).
        let narrow = gain_reference(&widths);
        // WHO PRICES THIS ROUND. The model's probabilities carry the request's intercept, so a
        // round it prices applies no offset on top; otherwise today's estimate takes today's.
        let today_offset = self.offset_value();
        let model = self.accept.as_ref().filter(|m| m.prices());
        let tree_offset = if model.is_some() { 0.0 } else { today_offset };
        let budget = if let Some(rows) = self.session.fixed_candidate_rows() {
            // Reuse the current physical budget. Missing CUDA prices must never
            // trigger the generic 2..64 cold probe or import Metal measurements.
            match tree_width()? {
                None => None,
                Some(TreeWidth::Fixed(n)) if n > rows =>
                    return Err("candidate width exceeds session capability".into()),
                Some(TreeWidth::Fixed(n)) if n <= room =>
                    Some(TreeBudget::Fixed { nodes: n, offset: tree_offset }),
                Some(TreeWidth::Budget) if rows <= room =>
                    Some(TreeBudget::Fixed { nodes: rows, offset: tree_offset }),
                _ => None,
            }
        } else { self.tree_budget(&widths, start, room, tree_offset, narrow)? };
        let Some(budget) = budget else {
            self.last_plan = None;
            return Ok(None);
        };
        // The width this round probes, if it is a probe: the tree it builds may fall short of it.
        let probed = match &budget {
            crate::speculative::TreeBudget::Fixed { nodes, .. }
                if self.cost.as_ref().and_then(crate::verify_cost::VerifyCost::probe)
                    == Some(*nodes) =>
            {
                Some(*nodes)
            }
            _ => None,
        };
        let Some(mut found) = self.session.candidates(start, anchor, chain)? else {
            self.last_plan = None;
            return Ok(None);
        };
        // DESIGN 6.6'S CHAINS. The index is fed and scored on every committed token either way
        // (design 11.4's gate is the score); this decides only whether it proposes nodes.
        if ngram_candidates_on() {
            let lookup = NgramLookup {
                request: &self.ngram,
                table: self.table.as_ref(),
            };
            let key0 = ngram_chains(
                &mut found,
                &lookup,
                anchor,
                &self.ngram_precision(),
                stops,
                start,
                self.session.candidate_depth_limit().min(room.saturating_sub(1)).max(1),
                self.session.verify_max_rows().saturating_sub(1),
            );
            if crate::speculative::score_probe() && !self.key0_said {
                // The proof that position 0's key ends in the anchor: printed once a request.
                self.key0_said = true;
                let k: Vec<String> = key0.iter().map(ToString::to_string).collect();
                eprintln!("dspark ngram key0={} anchor={anchor}", k.join(","));
            }
        }
        self.tree_rounds += 1;
        let priced = model.map(|m| priced_candidates(m, &found));
        // THE EXPLORATION ROW, every 16th tree round, and only where the model can say which
        // bucket has the fewest labels.
        let explore: Option<Vec<u64>> = (self.tree_rounds % EXPLORE_EVERY == 0)
            .then_some(self.accept.as_ref())
            .flatten()
            .filter(|_| {
                found
                    .sources
                    .contains(&crate::speculative::NodeSource::Ngram)
            })
            .map(|m| {
                found
                    .features
                    .iter()
                    .zip(&found.sources)
                    .map(|(f, s)| match (s, f.as_ref().and_then(|f| f.bucket_coefficient())) {
                        (crate::speculative::NodeSource::Ngram, Some(c)) => m.labels(c),
                        _ => u64::MAX,
                    })
                    .collect()
            });
        let tree = budget.tree(
            anchor,
            priced.as_deref().unwrap_or(&found.candidates),
            stops,
            crate::speculative::TreeRound {
                of: &found.sources,
                explore: explore.as_deref(),
                shape: tree_shape()?,
            },
        )?;
        if tree.tokens.len() < 2 { return Ok(None); }
        if crate::speculative::score_probe() {
            // Each candidate's source, index for index with the `dspark cands` line: d the
            // drafter's alone, n the n-gram's alone, a agreed. With the path, it splits rows,
            // accepted rows and S by source offline.
            let src: String = found
                .sources
                .iter()
                .map(|s| match s {
                    crate::speculative::NodeSource::Drafter => 'd',
                    crate::speculative::NodeSource::Ngram => 'n',
                    crate::speculative::NodeSource::Agreed => 'a',
                })
                .collect();
            eprintln!("dspark src={src} explored={}", u8::from(tree.explored));
        }
        // A PROBE THE TREE COULD NOT FILL tells the table how wide a tree gets here, before the
        // verify reports a width that belongs to another step (`VerifyCost::unreached`).
        if let (Some(asked), Some(cost)) = (probed, self.cost.as_mut()) {
            cost.unreached(asked, tree.tokens.len());
        }
        if self.accept.is_some() {
            self.accept_rounds.0 += u64::from(priced.is_some());
            self.accept_rounds.1 += 1;
            if crate::speculative::score_probe() {
                probe_accept(self.accept.as_ref(), priced.is_some(), &found);
            }
        }
        // What this round can teach the value model, kept for the commit that follows: the
        // ranks and S it would need to price a width it did NOT run, and every candidate with
        // the estimate it came with.
        self.last_plan = if tree.s.len() > 1 {
            // A round the cell's room narrowed is not the policy's choice: the hysteresis keeps
            // the width it held.
            if !capped {
                self.hold = tree.tokens.len();
            }
            Some(RoundPlan {
                rank: tree.rank.clone(),
                s: tree.s.clone(),
                s_n: tree.s_n.clone(),
                explored: tree.explored,
                parents: tree.parents.clone(),
                lo: narrow,
                hi: tree.tokens.len(),
                first: widths.first().map_or(0, |&(n, _)| n),
                ctx: start,
                node: tree.node.clone(),
                found,
                offset: today_offset,
            })
        } else {
            None
        };
        // THE COST TABLE LEARNS WHERE ITS STEPS ARE only where a width could be chosen: the gate
        // asks the question the chooser just asked, with this round's tokens (`VerifyCost::refine`).
        if tree.s.len() > 1 {
            let hold = self.hold;
            let rho = if rate_rule_on() { self.rate.at(start) } else { None };
            if let Some(cost) = self.cost.as_mut() {
                if let Some(held_us) = cost.cost_us(hold) {
                    let choice = crate::speculative::Choice {
                        widths: &widths,
                        fixed_us: cost.fixed_us().unwrap_or(0.0),
                        gain: cost.gain(),
                        gain_n: cost.gain_n(),
                        gain_below: cost.gain_below(),
                        hold,
                        offset: 0.0,
                        rho,
                        base_rows: narrow,
                    };
                    let valuer = crate::speculative::Valuer::new(
                        &tree.s,
                        &tree.s_n,
                        tree.s.len() - 1,
                        &choice,
                        rho,
                    );
                    let added =
                        cost.refine(hold, |rows, us| valuer.could_beat(rows, us, hold, held_us));
                    if let (Some(added), true) = (added, crate::speculative::score_probe()) {
                        eprintln!("dspark cost split={added} held={hold} steps={}", cost.report());
                    }
                }
            }
        }
        Ok(Some(tree))
    }
    fn observe_verify(&mut self, rows: usize, elapsed: std::time::Duration) {
        self.round_us += elapsed.as_secs_f64() * 1e6;
        if self.cold_round || self.session.complete_tree_costs() {
            return;
        }
        if let Some(cost) = self.cost.as_mut() {
            cost.observe(rows, elapsed);
            self.cost_seen += 1;
        }
    }
    fn observe_tree_round(
        &mut self,
        tree: &crate::speculative::DraftTree,
        round: &crate::speculative::TreeRoundTiming,
    ) {
        if let Some(cost) = self.native_cost.as_mut() {
            cost.observe(tree, round);
        }
        if self.session.complete_tree_costs() && !self.cold_round
            && round.continuing
            && round.submission == crate::speculative::TreeSubmission::Ordinary
            && tree.tokens.len() == round.rows
            && (2..=self.session.verify_max_rows()).contains(&round.rows)
            && round.accepted > 0 && round.accepted <= round.rows
            && !round.verify.is_zero()
            && round.total >= round.fixed.saturating_add(round.verify).saturating_add(round.commit)
        {
            if let Some(cost) = self.cost.as_mut() {
                cost.set_context(round.context);
                cost.observe_fixed(round.total - round.verify);
                cost.observe(round.rows, round.verify);
                self.cost_seen += 1;
                self.rate.observe(round.context, round.accepted, round.total.as_secs_f64() * 1e6);
            }
        }
    }
    fn observe_round_fixed(&mut self, elapsed: std::time::Duration) {
        self.round_us += elapsed.as_secs_f64() * 1e6;
        // A REQUEST'S FIRST ROUND IS COLD, and the cost model does not learn from it. Measured
        // over 36 requests per target, the first round's verify ran 5-11% above that width's
        // typical time (median) and its drafter 3-7% above; the second round is within 0.3%. Fed
        // in, it set a one-sample level that priced its width high enough to drop out of the
        // offer -- LFM2.5-2.6B at 8,444 tokens then held 6 rows for 25 rounds -- and a one-sample
        // clock that priced every other width 9% high. It is left out rather than discounted: its
        // excess is not a fixed ratio (5-11%), and the round still counts toward the decode time.
        self.cold_round = self.rounds_timed == 0;
        self.rounds_timed += 1;
        if self.cold_round || self.session.complete_tree_costs() {
            return;
        }
        if let Some(cost) = self.cost.as_mut() {
            cost.observe_fixed(elapsed);
        }
    }
    fn commit_tree(
        &mut self,
        start: usize,
        inputs: &[u32],
        path: &[i32],
        next: u32,
    ) -> Result<(), String> {
        let at = std::time::Instant::now();
        // The deepest position a tree round reached, for the stored table's advisory line.
        self.cost_at = self.cost_at.max(start);
        // Read before learn_margin TAKES the plan: only a round this provider planned is a
        // round of the rate's.
        let planned = self.last_plan.as_ref().map(|p| p.ctx);
        // BEFORE learn_margin, which TAKES last_plan; all three read the same round. The model
        // first: its guard scores today's estimate after the offset the round carried, which is
        // recorded in the plan, so the order against learn_offset does not matter to it.
        self.learn_accept(path, next);
        self.learn_offset(path);
        self.learn_margin(path);
        self.session.compact(path)?;
        self.commit(start, inputs)?;
        // THE ROUND'S TOKENS AND TIME. The path is the anchor plus every accepted node, and
        // the round emits one token per path entry (the accepted drafts and the target's pick
        // after them), so its length is `1 + accepted`. The time is the drafter and tree build,
        // the verify, and this commit: everything the round spent that the next round will
        // spend again.
        if let Some(ctx) = planned.filter(|_| !self.session.complete_tree_costs()) {
            let us = self.round_us + at.elapsed().as_secs_f64() * 1e6;
            self.rate.observe(ctx, path.len(), us);
            if crate::speculative::score_probe() {
                eprintln!("dspark rate ctx={ctx} tokens={} us={us:.0}", path.len());
            }
        }
        Ok(())
    }
    /// A history that outlives the request follows the stream to its end (design 5.5), so the
    /// next turn can resume at the stream's grid point.
    fn keeps_rows(&self, start: usize, n: usize) -> bool {
        self.live
            && self.tokens.is_some()
            && start == self.history
            && n <= self.batch_capacity
            && self.session.tail_room().is_some_and(|room| {
                start
                    .checked_add(n)
                    .and_then(|end| end.checked_add(room))
                    .is_some_and(|end| end <= self.capacity)
            })
    }
    fn keep_tree(
        &mut self,
        start: usize,
        inputs: &[u32],
        path: &[i32],
    ) -> Result<(), String> {
        // Kept without learning, as these rounds were before the history kept them: the reply's
        // last round ends at a stop token or at the output budget, not where the target disagreed.
        self.last_plan = None;
        self.session.compact(path)?;
        self.commit(start, inputs)
    }
    fn block_size(&self) -> usize {
        self.session.drafts() + 1
    }
    fn minimum_remaining(&self) -> usize {
        draft_minimum_remaining(
            crate::lfm_retained_domain(), self.block_size(),
            std::env::var("IMPARO_LAB_DRAFT_BOUNDED_TAIL").as_deref() == Ok("1"),
        )
    }
    fn initialize(&mut self) -> Result<(), String> {
        if !self.live {
            return Err("DSpark provider detached".into());
        }
        self.session.reset()?;
        self.history = 0;
        self.floor = 0;
        if !self.session.set_floor(0) {
            return Err("DSpark session refused a floor of 0".into());
        }
        if let Some(t) = self.tokens.as_mut() {
            t.clear();
        }
        self.resume_from = None;
        self.bridge_start = None;
        self.boundary_witness_logged = false;
        // LEVEL 1 ENDS HERE for everything with that lifetime: the offset corrects an estimate
        // about the context this call is throwing away, and the index holds that context's own
        // text (design 11.2).
        self.ngram_report();
        self.accept_report();
        self.offset.reset();
        // LEVEL 1 resets the counts with the context. The stored table is a separate map and
        // keeps its counts; its keys come from the request's context, which goes here.
        self.ngram.reset();
        self.key0_said = false;
        Ok(())
    }
    fn initialize_at(&mut self, start: usize) -> Result<(), String> {
        if start == 0 {
            return self.initialize();
        }
        if self.live && self.resume_from.take() == Some(start) && self.history == start
        {
            Ok(())
        } else {
            Err("draft resume point changed".into())
        }
    }
    fn set_capture(&mut self, enabled: bool) -> Result<(), String> {
        if !self.live {
            return if enabled {
                Err("DSpark provider detached".into())
            } else {
                Ok(())
            };
        }
        self.session.capture(enabled)
    }
    fn can_draft(&self, start: usize, _remaining: usize) -> bool {
        let cell = crate::prefill_batch().max(1);
        self.live
            && start
                .checked_add(self.block_size())
                .is_some_and(|end| end <= self.capacity)
            && self.block_size() <= cell - start % cell
    }
    fn can_bridge(&self, start: usize, remaining: usize) -> bool {
        self.live
            && self.boundary_resume
            && can_bridge_cell(
                start,
                remaining,
                self.capacity,
                self.block_size(),
                crate::prefill_batch().max(1),
            )
    }
    fn draft(
        &mut self,
        target: &mut dyn Model,
        start: usize,
        anchor: u32,
    ) -> Result<Vec<u32>, String> {
        if self.history != start {
            return Err("DSpark history/target mismatch".into());
        }
        let ids = self.session.generate(target, start, anchor)?;
        if let Some(from) = self.bridge_start.take() {
            let cell = crate::prefill_batch().max(1);
            if self.boundary_trace
                && !self.boundary_witness_logged
                && start / cell > from / cell
            {
                eprintln!(
                    "[dspark-boundary-resume] bridge_from={from} start={start} history={} block={} cell={cell}",
                    self.history,
                    self.block_size()
                );
                self.boundary_witness_logged = true;
            }
        }
        Ok(ids)
    }
    fn commit(&mut self, start: usize, inputs: &[u32]) -> Result<(), String> {
        if start != self.history
            || inputs.is_empty()
            || inputs.len() > self.batch_capacity
        {
            return Err("DSpark feature commit range invalid".into());
        }
        self.session.append(start, inputs.len())?;
        self.history = start + inputs.len();
        if let Some(t) = self.tokens.as_mut() {
            t.extend_from_slice(inputs);
        }
        // LEVEL 1's INDEX, and level 2's table when it is on. Each token is SCORED BEFORE it is
        // learned, so the count is an honest prediction and not a lookup of something already
        // inserted; the score goes to whichever source a lookup would have read.
        for &tok in inputs {
            let lookup = NgramLookup {
                request: &self.ngram,
                table: self.table.as_ref(),
            };
            if let Some(step) = lookup.step(self.ngram.tail()) {
                let slot = &mut self.ngram_score.1[step.terms.slot()];
                slot.0 += 1;
                slot.1 += u64::from(step.token == tok);
            }
            self.ngram_score.0 += 1;
            if let Some(table) = self.table.as_mut() {
                table.count(self.ngram.tail(), tok);
            }
            self.ngram.observe(tok);
        }
        if self.boundary_trace
            && self.boundary_resume
            && !self.boundary_witness_logged
            && inputs.len() == 1
            && self.bridge_start.is_none()
        {
            let cell = crate::prefill_batch().max(1);
            if self.block_size() > cell - start % cell {
                self.bridge_start = Some(start);
            }
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<(), String> {
        // The last request's index is scored here as well as at the next `initialize`, because
        // a process that serves ONE request never reaches an initialize after it.
        self.ngram_report();
        self.accept_report();
        self.save_dictionary();
        if let Some(report) = self.native_cost.as_mut().and_then(|c| c.take_report()) {
            eprintln!("[dspark-tree-cost] {report}");
        }
        if self.session.complete_tree_costs() {
            if let Some(cost) = self.cost.as_ref() {
                eprintln!("[dspark-cuda-budget] rows_max={} eager_only=1 whole_round=1 samples={} cost={} rate={:?}",
                    self.session.verify_max_rows(), self.cost_seen, cost.report(), self.rate.rows());
            }
        }
        if self.live {
            self.session.detach()?;
            self.live = false;
        }
        // THE SLOW TIMESCALE RUNS HERE. The round's learner is one pass at 399 ns and its
        // weights reach the very next round; this refits each law over its retained window
        // and keeps the result only if it beats the incumbent on held-out samples. A
        // request boundary is where it belongs: ~0.04 ms a law (measured, show_what_a_refit_costs), and
        // nothing is waiting once a request ends. It runs BEFORE the store so the refitted
        // law is what persists and what the next request seeds from.
        if let Some(cost) = self.cost.as_mut() {
            if self.cost_seen > 0 {
                let took = cost.refit();
                if took > 0
                    && std::env::var("IMPARO_DSPARK_CLOCK").as_deref() == Ok("1")
                {
                    eprintln!("dspark cost refit: {took} law(s) beat the incumbent");
                }
            }
        }
        // The table is what design 6.3's budget will read, so it is printed beside the
        // round clock that produced it: one entry per cost class, `lo..hi=ms(samples)`.
        // WRITTEN BACK so the next process starts where this one finished. The
        // existing rows/seed contract also preserves partial probes; unseen
        // classes remain probes and cannot silently become measured choices.
        // `IMPARO_DSPARK_COST_CACHE=0` turns the write off.
        if self.cost_seen > 0
            && std::env::var("IMPARO_DSPARK_COST_CACHE").as_deref() != Ok("0")
        {
            if let (Some(cost), Some((fp, model_bytes))) =
                (self.cost.as_ref(), self.cost_key.as_ref())
            {
                let table = imparo_host::VerifyCostTable {
                    steps: cost.rows(),
                    laws: cost.laws(),
                    banks: cost.banks(),
                    gain: cost.gain_state(),
                    gain_n: cost.gain_n_state(),
                    gain_below: cost.gain_below_state().0,
                    gain_n_below: cost.gain_below_state().1,
                    rate: self.rate.rows(),
                    accept: self
                        .accept
                        .as_ref()
                        .map(|m| accept_row(m, self.session.acceptance_top_k()))
                        .or_else(|| self.stored_accept.clone()),
                };
                if !table.steps.is_empty() {
                    if let Err(e) = imparo_host::write_verify_cost(
                        fp,
                        *model_bytes,
                        self.cost_at,
                        &table,
                    ) {
                        eprintln!("[imparo] dspark verify cost: not stored ({e})");
                    } else if self.session.complete_tree_costs() {
                        eprintln!("[dspark-cost-cache] {}", serde_json::json!({
                            "event":"store", "key":fp,
                            "path":imparo_host::verify_cost_path_for(fp,*model_bytes),
                            "steps":table.steps, "rate":table.rate,
                            "top_k":self.session.acceptance_top_k(),
                            "accept_seen":table.accept.as_ref().map(|r|r.seen),
                        }));
                    }
                }
            }
        }
        if std::env::var("IMPARO_DSPARK_CLOCK").as_deref() == Ok("1") {
            match self.cost.as_ref() {
                Some(cost) => {
                    eprintln!("[imparo] dspark verify cost {}", cost.report());
                }
                None => eprintln!(
                    "[imparo] dspark verify cost: the backend does not declare where its                      matmul cost steps; no table"
                ),
            }
        }
        Ok(())
    }
}

impl<S: DsparkSession> Drop for DsparkProvider<S> {
    fn drop(&mut self) {
        if let Err(e) = self.finish() {
            eprintln!("DSpark teardown: {e}");
        }
    }
}

/// Adapt cached native cumulative log probabilities to the shared selector's
/// conditional edge probabilities. No confidence-head features are fabricated;
/// this fixed-width adapter does not train or invoke the acceptance/cost models.
#[cfg(feature = "cuda-speculative")]
fn cuda_frontier_candidates(
    anchor: u32,
    tokens: &[u32],
    parents: &[i32],
    scores: &[f32],
) -> Result<Candidates, String> {
    use crate::speculative::{NodeSource, TreeCandidate};
    if !(2..=33).contains(&tokens.len()) || parents.len() != tokens.len()
        || scores.len() != tokens.len() || tokens[0] != anchor
        || parents[0] != -1 || scores[0] != 0.0 {
        return Err("CUDA candidate pool shape/root".into());
    }
    let mut candidates = Vec::with_capacity(tokens.len() - 1);
    let mut depths = vec![0_u32; tokens.len()];
    for row in 1..tokens.len() {
        let parent = usize::try_from(parents[row]).ok().filter(|&p| p < row)
            .ok_or("CUDA candidate pool parent order")?;
        depths[row] = depths[parent] + 1;
        if depths[row] > 8 || !scores[row].is_finite() || scores[row] > scores[parent] {
            return Err("CUDA candidate pool probability/depth".into());
        }
        candidates.push(TreeCandidate {
            token: tokens[row], parent: parents[row] - 1,
            // Native beam scores are cumulative log probabilities; this is an
            // estimate for the real edge, never a fabricated confidence head.
            q: (f64::from(scores[row]) - f64::from(scores[parent])).exp(),
        });
    }
    Ok(Candidates {
        features: vec![None; candidates.len()],
        ngram_branch_prefix: Some(candidates.len()),
        sources: vec![NodeSource::Drafter; candidates.len()], candidates,
    })
}

/// Complete each expanded native parent's Top4 before constructing acceptance
/// features. The global beam's 32 survivors alone omit siblings from the model's
/// softmax. Keep those survivors first so their original n-gram branches and
/// parent indices remain stable, then append the missing leaf candidates.
#[cfg(feature = "cuda-speculative")]
fn cuda_acceptance_candidates(
    anchor: u32,
    tokens: &[u32],
    parents: &[i32],
    scores: &[f32],
    groups: &[imparo_cuda::dspark::FrontierGroup],
    context: usize,
    vocab: u32,
) -> Result<Candidates, String> {
    use crate::accept_model::Features;
    use crate::speculative::{NodeSource, TreeCandidate};
    if tokens.len() != 33 || groups.len() != 29 || tokens.iter().any(|&t| t >= vocab) {
        return Err("CUDA acceptance group/pool shape or token".into());
    }
    let mut found = cuda_frontier_candidates(anchor, tokens, parents, scores)?;
    let mut edges = std::collections::HashMap::new();
    let mut depths = vec![0_u32; tokens.len()];
    for row in 1..tokens.len() {
        depths[row] = depths[parents[row] as usize] + 1;
        if edges.insert((parents[row], tokens[row]), row - 1).is_some() {
            return Err("CUDA acceptance duplicate beam edge".into());
        }
    }
    let mut seen = [false; 29];
    for group in groups {
        let parent = group.parent as usize;
        if parent >= seen.len() || seen[parent] || depths[parent] >= 8
            || !group.confidence.is_finite() || !(0.0..=1.0).contains(&group.confidence)
        {
            return Err("CUDA acceptance parent/confidence contract".into());
        }
        seen[parent] = true;
        let mut mass = 0.0;
        for rank in 0..4 {
            let (token, value) = (group.tokens[rank], group.logp[rank]);
            if token >= vocab || !value.is_finite() || value > 0.0
                || group.tokens[..rank].contains(&token)
                // Normalization can round distinct logits to an equal logp. The
                // native order still names the true local pick in that case.
                || (rank > 0 && value > group.logp[rank - 1])
            {
                return Err("CUDA acceptance Top4 token/probability/order contract".into());
            }
            mass += f64::from(value).exp();
        }
        if mass > 1.0 + 1e-5 {
            return Err("CUDA acceptance Top4 probability mass".into());
        }
        // A common vocabulary log-normalizer cancels from every feature in
        // Features::position, so local logp gives the same complete-group shares.
        let features = Features::position(group.confidence, &group.logp);
        for (rank, mut feature) in features.into_iter().enumerate() {
            let token = group.tokens[rank];
            let edge = (group.parent as i32, token);
            let q = f64::from(group.logp[rank]).exp();
            let c = if let Some(&c) = edges.get(&edge) {
                let expected = scores[parent] + group.logp[rank];
                let actual = scores[c + 1];
                if (expected - actual).abs() > 2e-5 * (1.0 + actual.abs())
                    || found.features[c].is_some()
                {
                    return Err("CUDA acceptance group/beam score mismatch".into());
                }
                // Retain the actual conditional probability, not the head's
                // acceptance estimate. The model guard compares against this q.
                found.candidates[c].q = q;
                c
            } else {
                let c = found.candidates.len();
                edges.insert(edge, c);
                found.candidates.push(TreeCandidate {
                    token, parent: group.parent as i32 - 1, q,
                });
                found.features.push(None);
                found.sources.push(NodeSource::Drafter);
                c
            };
            feature.depth = depths[parent];
            feature.context = context;
            found.features[c] = Some(feature);
        }
    }
    if !seen.into_iter().all(|v| v) || found.candidates.len() != 116
        || found.features.iter().any(Option::is_none)
    {
        return Err("CUDA acceptance incomplete expanded-parent Top4 groups".into());
    }
    Ok(found)
}

#[cfg(feature = "cuda-speculative")]
fn select_cuda_frontier(
    anchor: u32,
    tokens: &[u32],
    parents: &[i32],
    scores: &[f32],
    nodes: usize,
    stops: &[u32],
) -> Result<Option<crate::speculative::DraftTree>, String> {
    use crate::speculative::{TreeBudget, TreeCandidate, TreeRound};
    if !(2..=16).contains(&nodes) || !(2..=16).contains(&tokens.len())
        || parents.len() != tokens.len() || scores.len() != tokens.len()
        || tokens[0] != anchor || parents[0] != -1 || scores[0] != 0.0
    {
        return Err("CUDA frontier candidate shape/root".into());
    }
    let mut candidates = Vec::with_capacity(tokens.len()-1);
    let mut depths = vec![0_u32; tokens.len()];
    for row in 1..tokens.len() {
        let parent = usize::try_from(parents[row]).ok().filter(|&p| p < row)
            .ok_or("CUDA frontier parent order")?;
        depths[row] = depths[parent]+1;
        if depths[row] > 8 || !scores[row].is_finite() || scores[row] > scores[parent] {
            return Err("CUDA frontier probability/depth".into());
        }
        candidates.push(TreeCandidate {
            token: tokens[row],
            parent: parents[row]-1,
            q: (f64::from(scores[row])-f64::from(scores[parent])).exp(),
        });
    }
    let tree = TreeBudget::Fixed { nodes, offset: 0.0 }
        .tree(anchor, &candidates, stops, TreeRound::default())?;
    // If every first prediction is a stop, ordinary verification emits that stop.
    Ok((tree.tokens.len() >= 2).then_some(tree))
}

/// The drafter as imparo-cuda's native session, one per process.
#[cfg(feature = "cuda-speculative")]
pub struct CudaSession {
    descriptor: CudaDescriptor,
    enabled: Option<Arc<AtomicBool>>,
}

/// Stable session identity on top of the backend's actual model/knob identity.
/// It intentionally has no host search-space migration tag: different execution
/// contracts must not inherit prices or acceptance state from older CUDA routes.
#[cfg(feature = "cuda-speculative")]
fn cuda_cost_cache_key(
    base: (String, u64), descriptor: &CudaDescriptor, settings: &[(String, String)],
) -> (String, u64) {
    use sha2::{Digest, Sha256};
    static SOURCES: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();
    let sources = SOURCES.get_or_init(|| {
        let mut hash = Sha256::new();
        for source in [include_bytes!("dspark.rs").as_slice(), include_bytes!("verify_cost.rs").as_slice(),
            include_bytes!("cost_model.rs").as_slice(), include_bytes!("accept_model.rs").as_slice(),
            include_bytes!("accept_offset.rs").as_slice(), include_bytes!("speculative.rs").as_slice(),
            include_bytes!("ngram.rs").as_slice(), include_bytes!("verification.rs").as_slice()] {
            hash.update((source.len() as u64).to_le_bytes()); hash.update(source);
        }
        hash.finalize().into()
    });
    // Debug on these value-only structs encodes every named scalar/offset, with
    // no raw C padding, pointer addresses or process-local handles.
    let body = serde_json::to_vec(&serde_json::json!({
        "version":2, "backend":base.0, "model_bytes":base.1,
        "contract":"native-top4-v1;eager;maxrows16;depth8;whole-successful-warm-round;derived-or-matching-stored",
        "source_sha256":sources, "config":format!("{:?}",descriptor.config),
        "layers":format!("{:?}",descriptor.layers), "target_layers":descriptor.target_layers,
        "settings":settings,
    })).expect("value-only cache identity serializes");
    let digest = Sha256::digest(body);
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    (format!("cuda-dspark-cost-v2-{hex}"), base.1)
}

#[cfg(feature = "cuda-speculative")]
fn cuda_cost_cache_settings() -> Vec<(String, String)> {
    let mut settings: Vec<_> = std::env::vars().filter(|(key, _)| {
        (key.starts_with("IMPARO_DSPARK_") && key != "IMPARO_DSPARK_COST_CACHE")
            || key.starts_with("IMPARO_LAB_")
    }).collect();
    settings.sort();
    settings
}

#[cfg(feature = "cuda-speculative")]
impl DsparkSession for CudaSession {
    fn verify_max_rows(&self) -> usize { 16 }
    fn passive_tree_costs(&self) -> bool { !imparo_cuda::knobs::lfm_tree_adaptive_enabled() }
    fn complete_tree_costs(&self) -> bool { imparo_cuda::knobs::lfm_tree_adaptive_enabled() }
    fn cost_cache_allowed(&self) -> bool { imparo_cuda::knobs::lfm_tree_adaptive_enabled() }
    fn cost_cache_key(&self) -> Option<(String, u64)> {
        if !self.cost_cache_allowed() { return None; }
        let base = crate::gpu_support::be().config_key()?;
        Some(cuda_cost_cache_key(base, &self.descriptor, &cuda_cost_cache_settings()))
    }
    fn acceptance_top_k(&self) -> u32 { 4 }
    fn fixed_candidate_rows(&self) -> Option<usize> {
        (imparo_cuda::knobs::lfm_tree_candidates_enabled()
            && !imparo_cuda::knobs::lfm_tree_adaptive_enabled()).then_some(16)
    }
    fn candidate_depth_limit(&self) -> usize { 8 }
    fn has_accept_features(&self) -> bool {
        imparo_cuda::knobs::lfm_tree_acceptance_enabled()
    }
    fn fresh_accept_model(&self) -> bool { true }
    fn drafts(&self) -> usize {
        // The session drafts `block_size` slots and drops the last one.
        self.descriptor.config.block_size as usize - 1
    }
    fn reset(&mut self) -> Result<(), String> {
        if let Some(e) = &self.enabled {
            e.store(false, Ordering::Relaxed);
        }
        unsafe { imparo_cuda::dspark::reset() }
    }
    fn capture(&mut self, enabled: bool) -> Result<(), String> {
        unsafe {
            imparo_cuda::dspark::capture_mode(enabled)?;
        }
        if let Some(e) = &self.enabled {
            e.store(enabled, Ordering::Relaxed);
        }
        Ok(())
    }
    fn append(&mut self, start: usize, n: usize) -> Result<(), String> {
        unsafe {
            imparo_cuda::dspark::append(
                u32::try_from(start).map_err(|_| "commit start overflow")?,
                u32::try_from(n).map_err(|_| "commit length overflow")?,
            )
        }
    }
    fn compact(&mut self, path: &[i32]) -> Result<(), String> {
        unsafe { imparo_cuda::dspark::compact_features(path) }
    }
    fn generate(
        &mut self,
        _target: &mut dyn Model,
        start: usize,
        anchor: u32,
    ) -> Result<Vec<u32>, String> {
        let n = self.descriptor.config.block_size as usize;
        let mut ids = vec![0; n];
        let mut conf = vec![0.; n];
        unsafe {
            let start = u32::try_from(start).map_err(|_| "draft position overflow")?;
            if imparo_cuda::knobs::lfm_tree_acceptance_enabled() {
                imparo_cuda::dspark::generate_with_accept_features(
                    start, anchor, &mut ids, &mut conf, n,
                )?;
            } else {
                imparo_cuda::dspark::generate(start, anchor, &mut ids, &mut conf, n)?;
            }
        }
        if ids.iter().any(|&t| t >= self.descriptor.config.vocab)
            || conf.iter().any(|x| !x.is_finite())
        {
            return Err("invalid DSpark output".into());
        }
        ids.truncate(n - 1);
        Ok(ids)
    }
    fn tree(
        &self,
        start: usize,
        anchor: u32,
        chain: &[u32],
        stops: &[u32],
    ) -> Result<Option<crate::speculative::DraftTree>, String> {
        // An explicit fixed width reuses the shared best-first selector. The default
        // remains the qualified native16 topology; CUDA has no cost-class table yet.
        let width = tree_width()?;
        if matches!(width, Some(TreeWidth::Fixed(n)) if n > 16) {
            return Err("CUDA tree supports 2 to 16 nodes".into());
        }
        if width.is_none() {
            return Ok(None);
        }
        // Diagnostic only: retain the native frontier draft and verify its existing chain.
        if std::env::var("IMPARO_LAB_DSPARK_CHAIN_VERIFY").as_deref() == Ok("1") {
            return Ok(None);
        }
        if crate::lfm_retained_domain() == 0
            && std::env::var("IMPARO_LAB_DSPARK_TREE16").as_deref() != Ok("1")
        {
            return Ok(None);
        }
        let cell = crate::prefill_batch().max(1);
        if chain.len() != 8
            || 16 > cell - start % cell
            || start
                .checked_add(16)
                .is_none_or(|x| x > self.descriptor.config.kv_capacity as usize)
        {
            return Ok(None);
        }
        if imparo_cuda::knobs::lfm_tree_candidates_enabled() {
            return Ok(None);
        }
        if crate::lfm_retained_domain() != 0
            || std::env::var("IMPARO_LAB_DSPARK_FRONTIER16").as_deref() == Ok("1")
        {
            let (tokens, parents) =
                unsafe { imparo_cuda::dspark::frontier_tree(start as u32, anchor)? };
            if let Some(TreeWidth::Fixed(nodes @ 2..=15)) = width {
                let scores = unsafe {
                    imparo_cuda::dspark::frontier_scores(start as u32, anchor)?
                };
                let selected = select_cuda_frontier(anchor, &tokens, &parents, &scores, nodes, stops)?;
                static SAID: std::sync::Once = std::sync::Once::new();
                if let Some(tree) = &selected {
                    SAID.call_once(|| eprintln!(
                        "[dspark-tree-select] shared=1 source=cached-frontier requested={nodes} rows={} max_depth=8",
                        tree.tokens.len()
                    ));
                }
                return Ok(selected);
            }
            return Ok(Some(crate::speculative::DraftTree::plain(tokens, parents)));
        }
        if matches!(width, Some(TreeWidth::Fixed(2..=15))) {
            return Err("CUDA small-tree selection requires the cached frontier provider".into());
        }
        let leaves = unsafe { imparo_cuda::dspark::tree_leaves(start as u32, anchor)? };
        let mut tokens = vec![anchor];
        tokens.extend_from_slice(chain);
        tokens.extend(leaves);
        let mut parents = vec![-1];
        parents.extend(0..8);
        parents.extend(0..7);
        Ok(Some(crate::speculative::DraftTree::plain(tokens, parents)))
    }
    fn candidates(
        &self,
        start: usize,
        anchor: u32,
        chain: &[u32],
    ) -> Result<Option<Candidates>, String> {
        if !imparo_cuda::knobs::lfm_tree_candidates_enabled() { return Ok(None); }
        if crate::lfm_retained_domain() != 1 || chain.len() != self.drafts() {
            return Err("shared CUDA candidates require the admitted short retained domain".into());
        }
        let cell = crate::prefill_batch().max(1);
        if 16 > cell - start % cell || start.checked_add(16)
            .is_none_or(|end| end > self.descriptor.config.kv_capacity as usize) {
            return Ok(None);
        }
        let (tokens, parents, scores) = unsafe {
            imparo_cuda::dspark::frontier_candidates(
                u32::try_from(start).map_err(|_| "candidate context overflow")?, anchor,
            )?
        };
        if imparo_cuda::knobs::lfm_tree_acceptance_enabled() {
            let groups = unsafe {
                imparo_cuda::dspark::frontier_groups(
                    u32::try_from(start).map_err(|_| "candidate context overflow")?, anchor,
                )?
            };
            let found = cuda_acceptance_candidates(
                anchor, &tokens, &parents, &scores, &groups, start,
                self.descriptor.config.vocab,
            )?;
            static ACCEPT_SAID: std::sync::Once = std::sync::Once::new();
            ACCEPT_SAID.call_once(|| eprintln!(
                "[dspark-tree-acceptance] groups={} candidates={} head=real prior=derived max_rows=16 max_depth=8 ngram_branches=32 learner_enabled={}",
                groups.len(), found.candidates.len(), u8::from(accept_on()),
            ));
            return Ok(Some(found));
        }
        let candidates = cuda_frontier_candidates(anchor, &tokens, &parents, &scores)?;
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| eprintln!(
            "[dspark-tree-candidates] shared=1 cached_nodes={} max_rows=16 max_depth=8 confidence_features=absent ngram={}",
            tokens.len(), u8::from(ngram_candidates_on())
        ));
        Ok(Some(candidates))
    }
    /// The native append needs a block free past its rows whether or not a draft follows:
    /// `start + n + M <= kv_capacity`, M the block size (imparo-cuda's `dspark.cuh`).
    fn tail_room(&self) -> Option<usize> {
        Some(self.descriptor.config.block_size as usize)
    }
    fn detach(&mut self) -> Result<(), String> {
        if let Some(e) = &self.enabled {
            e.store(false, Ordering::Relaxed);
        }
        unsafe {
            imparo_cuda::dspark::detach()?;
        }
        self.enabled = None;
        Ok(())
    }
}

#[cfg(feature = "cuda-speculative")]
impl DsparkProvider<CudaSession> {
    /// # Safety
    /// Keep this target and its already verified paired weight mapping alive until
    /// finish succeeds. No other thread or model may submit to the backend concurrently.
    pub unsafe fn attach<M: Model + ?Sized>(
        target: &mut M,
        descriptor: CudaDescriptor,
    ) -> Result<Self, String> {
        if target.state().host_forward
            || target
                .state()
                .layer_outputs
                .as_ref()
                .is_some_and(super::layer_outputs::LayerOutputCapture::attached)
        {
            return Err("DSpark requires an exclusive CUDA target with no other output subscriber".into());
        }
        unsafe {
            imparo_cuda::dspark::attach(
                &descriptor.config,
                &descriptor.layers,
                &descriptor.target_layers,
            )?;
        }
        Ok(Self::bind(target, descriptor, 0, None))
    }
    fn bind<M: Model + ?Sized>(
        target: &mut M,
        descriptor: CudaDescriptor,
        history: usize,
        tokens: Option<Vec<u32>>,
    ) -> Self {
        let enabled = Arc::new(AtomicBool::new(false));
        target.state_mut().layer_outputs =
            Some(crate::layer_outputs::LayerOutputCapture {
                enabled: Arc::downgrade(&enabled),
                layers: descriptor.target_layers.clone(),
                capture: crate::layer_outputs::CaptureFn::CudaDspark,
            });
        let capacity = descriptor.config.kv_capacity as usize;
        let batch_capacity = descriptor.config.batch_capacity as usize;
        Self::new(
            CudaSession {
                descriptor,
                enabled: Some(enabled),
            },
            capacity,
            batch_capacity,
            history,
            0,
            tokens,
            boundary_resume_enabled(),
        )
    }
    fn park(&mut self) -> Result<CachedHistory, String> {
        self.set_capture(false)?;
        let (ticket, coverage) = unsafe { imparo_cuda::dspark::suspend()? };
        if coverage as usize != self.history
            || self.tokens.as_ref().is_none_or(|x| x.len() != self.history)
        {
            return Err("parked draft history coverage mismatch".into());
        }
        self.live = false;
        self.session.enabled = None;
        eprintln!("[dspark-cache] parked coverage={coverage} ticket={ticket}");
        Ok(CachedHistory {
            ticket,
            tokens: self.tokens.take().unwrap(),
            floor: 0,
        })
    }
}

/// The drafter on the active backend's ops: its forward (`DsparkForward`) and the target rows it
/// reads (`FeatureTaps`).
pub struct BackendSession {
    forward: crate::dspark_forward::DsparkForward,
    taps: FeatureTaps,
    /// The last drafted block and where it was drafted (start, anchor): a tree reads its
    /// candidates.
    last: Option<(usize, u32, crate::dspark_forward::DraftBlock)>,
    /// Where the committed cache holds this request's rows from (`DsparkSession::set_floor`).
    floor: u32,
}

impl DsparkSession for BackendSession {
    fn drafts(&self) -> usize {
        // Anchor-first slots: slot k proposes token k + 1, so every slot is a draft.
        self.forward.block_size() as usize
    }
    fn reset(&mut self) -> Result<(), String> {
        // The caches need no clearing: an append overwrites every row attention reads.
        self.taps.set_enabled(false);
        self.last = None;
        Ok(())
    }
    fn capture(&mut self, enabled: bool) -> Result<(), String> {
        self.taps.set_enabled(enabled);
        Ok(())
    }
    fn append(&mut self, start: usize, n: usize) -> Result<(), String> {
        let start =
            u32::try_from(start).map_err(|_| "DSpark append position overflow")?;
        let n = u32::try_from(n).map_err(|_| "DSpark append length overflow")?;
        self.taps.check_rows(start, n)?;
        self.forward.append(start, n)
    }
    fn compact(&mut self, path: &[i32]) -> Result<(), String> {
        self.taps.compact(path)
    }
    fn generate(
        &mut self,
        target: &mut dyn Model,
        start: usize,
        anchor: u32,
    ) -> Result<Vec<u32>, String> {
        let at = u32::try_from(start).map_err(|_| "DSpark draft position overflow")?;
        if self.floor > at {
            return Err(format!(
                "DSpark draft at {at} below the history's floor {}",
                self.floor
            ));
        }
        let block = self.forward.generate(target, at, anchor, self.floor)?;
        let ids = block.ids.clone();
        self.last = Some((start, anchor, block));
        Ok(ids)
    }
    fn set_floor(&mut self, floor: usize) -> bool {
        u32::try_from(floor).is_ok_and(|floor| {
            self.floor = floor;
            true
        })
    }
    fn candidates(
        &self,
        start: usize,
        anchor: u32,
        chain: &[u32],
    ) -> Result<Option<Candidates>, String> {
        let Some((at, from, block)) = &self.last else {
            return Err("DSpark tree: no block was drafted".into());
        };
        if *at != start || *from != anchor || block.ids != chain {
            return Err(format!(
                "DSpark tree: the last block was drafted at {at} from {from}, the round is at \
                 {start} from {anchor}"
            ));
        }
        level1_candidates(block, start).map(Some)
    }
    /// An append writes its own rows only, at positions the target's forward has just written.
    fn tail_room(&self) -> Option<usize> {
        Some(0)
    }
    fn detach(&mut self) -> Result<(), String> {
        self.taps.set_enabled(false);
        Ok(())
    }
}

/// Design 6.1's level 1 as tree candidates: at each drafted position the chain's pick, hanging
/// from the chain's previous pick (the anchor at position 0), and its siblings, candidates 2 to K
/// of the same biased column. Estimate, first form: the confidence head for the pick; for a
/// sibling, its probability among the K candidates (a softmax over their biased values).
/// `context` is the round's start, which the acceptance model's context term reads.
///
/// # Errors
/// When the block carries fewer than 2 candidates per position, its rows do not match its picks,
/// or a position's first candidate is not its pick.
fn level1_candidates(
    block: &crate::dspark_forward::DraftBlock,
    context: usize,
) -> Result<Candidates, String> {
    use crate::accept_model::Features;
    use crate::speculative::TreeCandidate;
    let (m, k) = (block.ids.len(), block.candidates);
    if k < 2 {
        return Err(format!(
            "DSpark tree: {k} candidates per position; a tree needs IMPARO_DSPARK_TOPK of 2 or more"
        ));
    }
    if block.candidate_ids.len() != m * k
        || block.candidate_values.len() != m * k
        || block.confidence.len() != m
    {
        return Err(
            "DSpark tree: the block's candidate rows do not match its picks".into(),
        );
    }
    let mut out = Vec::with_capacity(m * k);
    let mut features = Vec::with_capacity(m * k);
    let mut chain_parent = -1_i32;
    for (pos, &pick) in block.ids.iter().enumerate() {
        let ids = &block.candidate_ids[pos * k..(pos + 1) * k];
        let values = &block.candidate_values[pos * k..(pos + 1) * k];
        if ids[0] != pick {
            return Err(format!(
                "DSpark tree: position {pos}'s first candidate {} is not its pick {pick}",
                ids[0]
            ));
        }
        // The first value is the largest, so no weight overflows.
        let top = f64::from(values[0]);
        let weights: Vec<f64> =
            values.iter().map(|&v| (f64::from(v) - top).exp()).collect();
        let total: f64 = weights.iter().sum();
        let parent = chain_parent;
        chain_parent =
            i32::try_from(out.len()).map_err(|_| "DSpark tree: too many candidates")?;
        out.push(TreeCandidate {
            token: pick,
            parent,
            q: f64::from(block.confidence[pos]),
        });
        for (&token, &w) in ids.iter().zip(&weights).skip(1) {
            out.push(TreeCandidate {
                token,
                parent,
                q: w / total,
            });
        }
        let depth =
            u32::try_from(pos).map_err(|_| "DSpark tree: too many positions")?;
        features.extend(
            Features::position(block.confidence[pos], values)
                .into_iter()
                .map(|f| Some(Features {
                    depth,
                    context,
                    ..f
                })),
        );
    }
    let sources = vec![crate::speculative::NodeSource::Drafter; out.len()];
    Ok(Candidates {
        candidates: out,
        features,
        sources,
        ngram_branch_prefix: None,
    })
}

/// A round's n-gram lookup: the request's own index first, the stored table where the request's
/// index holds nothing for the context (design 6.6's "the stored table as each lookup's fallback").
struct NgramLookup<'a> {
    request: &'a crate::ngram::NgramIndex,
    table: Option<&'a crate::ngram::NgramIndex>,
}

impl NgramLookup<'_> {
    /// The followers of `context`'s last key, most frequent first, and which table answered.
    fn follows(
        &self,
        context: &[u32],
    ) -> Option<(&[(u32, u32)], crate::accept_model::Source)> {
        use crate::accept_model::Source;
        let f = self.request.follows(context);
        if !f.is_empty() {
            return Some((f, Source::Request));
        }
        self.table
            .map(|t| t.follows(context))
            .filter(|f| !f.is_empty())
            .map(|f| (f, Source::Table))
    }
}

/// What one chain step proposes after a context, with the evidence its pricing reads.
#[derive(Clone, Copy, Debug)]
struct Step {
    token: u32,
    /// Source, count, copy length; the caller sets `run`.
    terms: crate::accept_model::NgramTerms,
    /// For a copy match, the position of `token` in the request's committed text, so the next step
    /// can read on from there.
    copied_from: Option<usize>,
}

impl NgramLookup<'_> {
    /// THE ONE RULE both the chains and the score use: a copy match of the request's own text first,
    /// then the request's 3-token index, then the stored table.
    fn step(&self, context: &[u32]) -> Option<Step> {
        use crate::accept_model::{NgramTerms, Source};
        if let Some((len, at)) = self.request.longest(context) {
            return Some(Step {
                token: self.request.tail()[at],
                terms: NgramTerms {
                    source: Source::Request,
                    count: 0,
                    matched: u32::try_from(len).unwrap_or(u32::MAX),
                    run: 0,
                },
                copied_from: Some(at),
            });
        }
        let (followers, source) = self.follows(context)?;
        let (token, count) = followers[0];
        Some(Step {
            token,
            terms: NgramTerms {
                source,
                count,
                matched: 0,
                run: 0,
            },
            copied_from: None,
        })
    }

    /// The step after a copy step at `at`: the next token of the same passage, one token longer
    /// match, while the passage has one. Reading on is what a longest-match lookup at the extended
    /// context would return while the copy continues, without searching again.
    fn copy_on(&self, at: usize, matched: u32) -> Option<Step> {
        use crate::accept_model::{NgramTerms, Source};
        let token = *self.request.tail().get(at + 1)?;
        Some(Step {
            token,
            terms: NgramTerms {
                source: Source::Request,
                count: 0,
                matched: matched
                    .saturating_add(1)
                    .min(u32::try_from(crate::ngram::MATCH_MAX).unwrap_or(u32::MAX)),
                run: 0,
            },
            copied_from: Some(at + 1),
        })
    }
}

/// Tree rounds between exploration rows (design 6.6, theory item 3).
const EXPLORE_EVERY: u64 = 16;

/// HOW DEEP A CHAIN MAY GO below the anchor. Past the drafter's block only the n-gram proposes,
/// and only a copy match reaches that far in practice; the budget's rows decide how much of it a
/// round verifies. MEASURED before building (the replay over full replies): copy chains past the
/// block added tokens per round, depth 16 and 32 alike.
const CHAIN_DEPTH: usize = 32;

/// N-gram-only nodes one round may add, all chains together: a verify never holds more rows, so
/// more would only cost host time.
const CHAIN_NODES: usize = imparo_backend::ROW_LAYOUT_MAX_ROWS - 1;

/// DESIGN 6.6'S N-GRAM CHAINS, added to a round's drafter candidates.
///
/// ```text
///   branch at   the anchor, then each drafter pick or featureless native candidate
///   context     committed text + anchor + the actual parent path + the chain's own tokens
///   step        a copy match of the request's own text (then read on along the copied passage),
///               else the top follower of the context's last 3 tokens, else the stored table's
///                 = a child the node already has  -> that child is AGREED; a pick ends the chain
///                                                    (its own branch continues from it), a sibling
///                                                    carries it on
///                 otherwise                       -> a new n-gram node, and the chain continues below
///   stops       at `depth_limit` below the anchor, when nothing is proposed, after a stop token, or
///               when the round has added `max_added` n-gram nodes
/// ```
///
/// The committed text is the request index's tail, which ends one token BEFORE the anchor: the
/// anchor is committed with the round's accepted path, after this tree. So the anchor is appended
/// here, and position 0's key ends in it. Returns the last `KEY_TOKENS` of that context, for the
/// probe.
///
/// A new node's `q` is its evidence slot's measured precision -- today's estimate for a token the
/// drafter did not offer. An agreed node keeps the drafter's `q`; the acceptance model prices what
/// agreement adds.
fn ngram_chains(
    found: &mut Candidates,
    lookup: &NgramLookup<'_>,
    anchor: u32,
    precision: &[f64; 8],
    stops: &[u32],
    context: usize,
    depth_limit: usize,
    max_added: usize,
) -> Vec<u32> {
    use crate::accept_model::{Features, Kind};
    use crate::speculative::{NodeSource, TreeCandidate};
    let drafted = found.candidates.len();
    let kids = children(&found.candidates);
    let branches: Vec<usize> = (0..drafted)
        .filter(|&c| found.ngram_branch_prefix.map_or_else(
            || found.features[c].is_none_or(|f| f.kind == Kind::Pick),
            |prefix| c < prefix,
        ))
        .collect();
    let mut is_branch = vec![false; drafted];
    for &c in &branches {
        is_branch[c] = true;
    }
    // Keep this index live as nodes are added: multiple branch walks must share an
    // already proposed (parent, token), including a node added earlier this round.
    let mut by_edge: std::collections::HashMap<(i32, u32), usize> = found
        .candidates
        .iter()
        .enumerate()
        .map(|(c, node)| ((node.parent, node.token), c))
        .collect();
    // The committed tail's last tokens -- as many as a copy match can compare -- and the anchor:
    // every branch point's context starts here.
    let tail = lookup.request.tail();
    let mut base: Vec<u32> =
        tail[tail.len().saturating_sub(crate::ngram::MATCH_MAX)..].to_vec();
    base.push(anchor);
    let key0: Vec<u32> =
        base[base.len().saturating_sub(crate::ngram::KEY_TOKENS)..].to_vec();
    // The n-gram tokens directly above each drafter candidate, once a chain agreed with it.
    let mut run_at = vec![0_u32; drafted];
    let mut added = 0_usize;
    // Parents precede children; candidate order need not be one chain or depth order.
    let branches = std::iter::once(None).chain(branches.into_iter().map(Some));
    for branch in branches {
        if added >= max_added {
            break;
        }
        let mut path = Vec::new();
        let mut at = branch;
        let mut valid = true;
        while let Some(c) = at {
            let candidate = &found.candidates[c];
            path.push(candidate.token);
            at = match candidate.parent {
                -1 => None,
                p if p >= 0 && (p as usize) < c => Some(p as usize),
                _ => {
                    valid = false;
                    break;
                }
            };
        }
        let depth0 = path.len();
        if !valid || depth0 >= depth_limit || path.iter().any(|t| stops.contains(t)) {
            continue;
        }
        let mut ctx = base.clone();
        ctx.extend(path.into_iter().rev());
        let mut node = branch;
        let mut depth = depth0;
        let mut run = branch.map_or(0, |c| run_at[c]);
        let mut last: Option<Step> = None;
        while depth < depth_limit && added < max_added {
            let next = match last {
                Some(Step {
                    copied_from: Some(at),
                    terms,
                    ..
                }) => lookup
                    .copy_on(at, terms.matched)
                    .or_else(|| lookup.step(&ctx)),
                _ => lookup.step(&ctx),
            };
            let Some(mut step) = next else {
                break;
            };
            step.terms.run = run;
            let token = step.token;
            let at_parent = match node {
                None => kids[drafted].as_slice(),
                Some(c) if c < drafted => kids[c].as_slice(),
                Some(_) => &[],
            };
            let parent = node.map_or(-1, |c| i32::try_from(c).unwrap_or(i32::MAX));
            let existing = by_edge.get(&(parent, token)).copied();
            run += 1;
            depth += 1;
            let p = precision[step.terms.slot()];
            // Every OTHER drafter child of this parent is contradicted by the same proposal.
            for &c in at_parent {
                if Some(c) != existing {
                    found.features[c] = found.features[c]
                        .map(|f| f.contradicted(step.terms, p));
                }
            }
            if let Some(c) = existing {
                if found.sources[c] != NodeSource::Ngram {
                    found.features[c] = found.features[c].map(|f| f.agreed(step.terms, p));
                    found.sources[c] = NodeSource::Agreed;
                }
                if c < drafted {
                    run_at[c] = run;
                }
                if stops.contains(&token) || (c < drafted && is_branch[c]) {
                    break;
                }
                node = Some(c);
            } else {
                let c = found.candidates.len();
                found.candidates.push(TreeCandidate {
                    token,
                    parent,
                    q: p,
                });
                found.features.push(Some(Features::ngram(
                    step.terms,
                    p,
                    u32::try_from(depth - 1).unwrap_or(u32::MAX),
                    context,
                )));
                found.sources.push(NodeSource::Ngram);
                by_edge.insert((parent, token), c);
                added += 1;
                if stops.contains(&token) {
                    break;
                }
                node = Some(c);
            }
            ctx.push(token);
            last = Some(step);
        }
    }
    key0
}

/// How wide this run's trees are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TreeWidth {
    /// A fixed node count, the anchor included.
    Fixed(usize),
    /// Design 6.3: the row count with the most expected accepted tokens per millisecond,
    /// after the cost table has priced every class.
    Budget,
}

/// HOW WIDE A ROUND'S TREE IS. Unset is design 6.3's budget: the row count that maximises
/// expected accepted tokens per millisecond, priced by the verify-cost table.
///
/// ```text
///   unset / budget   the budget                            the default
///   N                a fixed tree of N nodes               an instrument
///   chain            no tree at all, the drafter's chain   the arm the budget was measured against
/// ```
///
/// MEASURED on LFM2, ms/token, table seeded, two interleaved rounds with the arm order
/// reversed, three contexts (443 / 1596 / 8444):
///
/// ```text
///   chain     11.131   13.042   14.629
///   fixed 16  10.271   10.592   12.123
///   budget     9.782   10.506   12.132     -4.8% / -0.8% / +0.1% against the best fixed width
/// ```
///
/// A backend that declares no cost classes has no table, and the round falls back to the
/// chain -- design 9's untuned host.
///
/// # Errors
/// When the value is none of `budget`, `chain`, or a node count a row layout holds.
/// Contexts the stored dictionary holds before decay starts evicting. At ~800 contexts per
/// thousand committed tokens this is a few hundred requests' worth; the file is three numbers
/// a row, so 200k contexts is a few MB.
const DICTIONARY_CONTEXTS: usize = 200_000;

/// Whether design 11.5B's DICTIONARY is live: the same index kept ACROSS requests and stored
/// on disk, instead of reset with each one. Off until a held-out measurement says otherwise.
///
/// It holds fragments of the user's text, so it is local only, keyed to this device and this
/// model, and one `rm` of the file removes it -- the same privacy class as the KV disk tier.
fn dictionary_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_DICT").as_deref() == Ok("1"))
}

/// Whether design 11.4's index proposes candidates. It is FED and SCORED either way -- that
/// costs one hash per committed token and answers the gate; this switch decides only whether it
/// is allowed to spend verify rows. ON: with the acceptance model it is the complete form, which
/// beat a fixed 16-row tree x1.111 in decode over 36 full-reply turns (evidence
/// 2026-09-17-step3-ngram-in-the-budget.md). `IMPARO_DSPARK_NGRAM=0` turns it off.
fn ngram_candidates_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_NGRAM").as_deref() != Ok("0"))
}

/// Each parent's candidates by index, in candidate order; the anchor's at `candidates.len()`. A
/// candidate whose parent is not an earlier candidate is left out: the tree builder refuses it.
fn children(candidates: &[crate::speculative::TreeCandidate]) -> Vec<Vec<usize>> {
    let n = candidates.len();
    let mut kids = vec![Vec::new(); n + 1];
    for (i, c) in candidates.iter().enumerate() {
        match usize::try_from(c.parent) {
            Ok(p) if p < i => kids[p].push(i),
            Err(_) if c.parent == -1 => kids[n].push(i),
            _ => {}
        }
    }
    kids
}

/// The candidates as the acceptance model prices them: each parent's candidates from one softmax.
fn priced_candidates(
    model: &crate::accept_model::AcceptModel,
    found: &Candidates,
) -> Vec<crate::speculative::TreeCandidate> {
    let mut out = found.candidates.clone();
    if found.features.iter().any(Option::is_none) {
        return out;
    }
    for kids in children(&found.candidates) {
        if kids.is_empty() {
            continue;
        }
        let group: Vec<_> = kids.iter().filter_map(|&c| found.features[c]).collect();
        for (&c, pj) in kids.iter().zip(model.probabilities(&group)) {
            out[c].q = pj;
        }
    }
    out
}

/// The acceptance model's half of the value probe, inside the caller's gate: whether it priced the
/// round, its request intercept and coefficients, and today's estimate for every candidate at full
/// precision. With the probe's `raw` (the q the tree was built from), an offline pass recomputes
/// the model's probabilities from today's estimate and checks them against the engine's.
fn probe_accept(
    model: Option<&crate::accept_model::AcceptModel>,
    priced: bool,
    found: &Candidates,
) {
    if found.features.iter().any(Option::is_none) {
        return;
    }
    let Some(m) = model else {
        return;
    };
    let (w, seen) = m.state();
    let join = |it: &mut dyn Iterator<Item = String>| it.collect::<Vec<_>>().join(",");
    eprintln!(
        "dspark accept priced={} b={:e} seen={seen} w={} today={}",
        u8::from(priced),
        m.intercept(),
        join(&mut w.iter().map(|x| format!("{x:e}"))),
        join(&mut found.candidates.iter().map(|c| format!("{:e}", c.q)))
    );
}

/// Whether design 6.6's acceptance model runs: it learns from every round's labels and prices the
/// tree while its guard says it beats today's estimate. ON, as part of the complete form (see
/// `ngram_candidates_on`); `IMPARO_DSPARK_ACCEPT=0` turns it off.
fn accept_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_ACCEPT").as_deref() != Ok("0"))
}

/// The acceptance model's row for the store, keyed by the top-K its features were computed over.
fn accept_row(model: &crate::accept_model::AcceptModel, top_k: u32) -> imparo_host::AcceptRow {
    let l = model.level2();
    imparo_host::AcceptRow {
        top_k,
        fitted: l.fitted,
        seen: l.seen,
        guard: [l.guard.0, l.guard.1],
        prior: l.prior.to_vec(),
        z: l.z.to_vec(),
        n: l.n.to_vec(),
        labels: l.labels.to_vec(),
    }
}

/// The acceptance model this request starts from, or `None` when it is off.
///
/// ```text
///   a stored model, same top-K                               what earlier requests learned, resumed
///   IMPARO_DSPARK_ACCEPT_PRIOR=a_p,b_p,c_p,a_s,b_s,c_s,d_s   fitted on measured rounds; prices from
///                                                            its first round
///   neither                                                  derived, kappa 0.5 (uninformed); prices
///                                                            only once it beats today's estimate
/// ```
///
/// A prior that is not seven finite numbers switches the model OFF and says so: an arm that asked
/// for a fitted prior and silently ran the derived one would measure the wrong model.
fn accept_start(
    stored: Option<&imparo_host::AcceptRow>,
    top_k: u32,
    derived_only: bool,
) -> Option<crate::accept_model::AcceptModel> {
    use crate::accept_model::{AcceptModel, COEFS, DRAFTER_COEFS, Level2};
    if !accept_on() {
        return None;
    }
    if let Some(row) = stored {
        let coefs = |v: &[f64]| <[f64; COEFS]>::try_from(v).ok();
        match (
            row.top_k == top_k,
            coefs(&row.prior),
            coefs(&row.z),
            coefs(&row.n),
            <[u64; COEFS]>::try_from(row.labels.as_slice()).ok(),
        ) {
            (true, Some(prior), Some(z), Some(n), Some(labels)) => {
                eprintln!(
                    "[imparo] dspark accept: prior=stored seen={} fitted={}",
                    row.seen,
                    u8::from(row.fitted)
                );
                return Some(AcceptModel::resume(&Level2 {
                    fitted: row.fitted,
                    seen: row.seen,
                    guard: (row.guard[0], row.guard[1]),
                    prior,
                    z,
                    n,
                    labels,
                }));
            }
            _ => eprintln!(
                "[imparo] dspark accept: the stored model was learned under top-K {} with {} \
                 coefficients, this run has top-K {top_k} and {COEFS}; starting again",
                row.top_k,
                row.prior.len()
            ),
        }
    }
    if derived_only {
        eprintln!("[imparo] dspark accept: prior=derived kappa=0.5 source=native-top4 fresh=1");
        return Some(AcceptModel::derived(0.5));
    }
    let Ok(text) = std::env::var("IMPARO_DSPARK_ACCEPT_PRIOR") else {
        eprintln!("[imparo] dspark accept: prior=derived kappa=0.5");
        return Some(AcceptModel::derived(0.5));
    };
    let values: Vec<f64> = text
        .split(',')
        .filter_map(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .collect();
    let Ok(prior) = <[f64; DRAFTER_COEFS]>::try_from(values.as_slice()) else {
        eprintln!(
            "[imparo] dspark accept: IMPARO_DSPARK_ACCEPT_PRIOR={text} is not {DRAFTER_COEFS} \
             finite numbers; the model is OFF"
        );
        return None;
    };
    eprintln!("[imparo] dspark accept: prior=fitted {text}");
    Some(AcceptModel::fitted(prior))
}

/// Whether design 11.4's per-request offset is applied and trained.
///
/// ON by default since its gate: 80 held-out public prompts, paired, one session.
///
/// ```text
///   budget   b-on - b-off    -0.0929 +- 0.0272 ms/tok   t = -3.41   -0.91%
///   fixed16  b-on - b-off    +0.0016 +- 0.0159 ms/tok   t = +0.10    a clean zero
/// ```
///
/// The fixed-width arm being EXACTLY zero is what names the channel: at a pinned width the
/// offset can only reshape the tree, and reshaping is worth nothing. At the budget it moves
/// `S(n)` at every n, so it moves WHICH WIDTH is picked -- the budget was choosing too narrow
/// (4.987 tokens per round against a fixed 16's 5.296) and the offset pushed it wider (5.057).
///
/// `IMPARO_DSPARK_OFFSET=0` turns it off for an A/B.
fn offset_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_OFFSET").as_deref() != Ok("0"))
}

/// THE WIDTH THE GAINS ARE MEASURED FROM: the narrowest offered width of at least this many rows.
/// It is a reference, not a tuned value: the gain only needs a width that stays put while the table
/// splits narrower ones in. 8 is a width every target offered when the backend declared the classes
/// (the top of the 8-bit kernel's first tile), so gains stored then still measure the same marginal.
const GAIN_BASE_ROWS: usize = 8;

/// The marginal's reference width (`Choice::base_rows`): the narrowest offered width of at least
/// `GAIN_BASE_ROWS`, or the narrowest offered when none is that wide.
fn gain_reference(widths: &[(usize, f64)]) -> usize {
    let first = widths.first().map_or(0, |&(n, _)| n);
    widths
        .iter()
        .map(|&(n, _)| n)
        .find(|&n| n >= GAIN_BASE_ROWS)
        .unwrap_or(first)
}

/// Whether the budget prices a round's time at the process's long-run rate, rho, instead of the
/// round's own ratio (design 6.6, theory item 1). The rate is measured either way; this decides
/// only whether the width decision reads it.
///
/// ON by default since its gate: 16 held-out prompts, two interleaved rounds, non-inferiority.
///
/// ```text
///   rho - ratio   -0.39% +- 0.46% ms/token   upper 95% bound +0.37%, margin +1.0%   answers 32/32 equal
/// ```
///
/// The widths moved most at 12k-16k tokens of context, where the ratio rule sat on 8 rows and the
/// rate widened to 9-13; that is also where the only consistent gains were.
///
/// `IMPARO_DSPARK_RHO=0` turns it off for an A/B.
fn rate_rule_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_RHO").as_deref() != Ok("0"))
}

/// THE SHAPE OF A ROUND'S TREE: `IMPARO_DSPARK_SHAPE=chain` keeps one child per node, the ablation
/// that measures what branching is worth with the candidates, the pricing and the budget unchanged;
/// unset or `tree` is the tree.
///
/// # Errors
/// When the value is neither.
pub(crate) fn tree_shape() -> Result<crate::speculative::Shape, String> {
    use crate::speculative::Shape;
    static SHAPE: std::sync::OnceLock<Result<Shape, String>> =
        std::sync::OnceLock::new();
    SHAPE
        .get_or_init(|| match std::env::var("IMPARO_DSPARK_SHAPE") {
            Err(_) => Ok(Shape::Tree),
            Ok(v) if v == "tree" => Ok(Shape::Tree),
            Ok(v) if v == "chain" => Ok(Shape::Chain),
            Ok(v) => Err(format!("IMPARO_DSPARK_SHAPE={v}: `tree` or `chain`")),
        })
        .clone()
}

pub(crate) fn tree_width() -> Result<Option<TreeWidth>, String> {
    static WIDTH: std::sync::OnceLock<Result<Option<TreeWidth>, String>> =
        std::sync::OnceLock::new();
    WIDTH
        .get_or_init(|| match std::env::var("IMPARO_DSPARK_TREE") {
            Err(_) => Ok(Some(TreeWidth::Budget)),
            Ok(v) if v == "budget" => Ok(Some(TreeWidth::Budget)),
            Ok(v) if v == "chain" => Ok(None),
            Ok(v) => match v.parse::<usize>() {
                Ok(n) if (2..=imparo_backend::ROW_LAYOUT_MAX_ROWS).contains(&n) => {
                    Ok(Some(TreeWidth::Fixed(n)))
                }
                _ => Err(format!(
                    "IMPARO_DSPARK_TREE={v}: `budget`, `chain`, or a tree of 2 to {} nodes",
                    imparo_backend::ROW_LAYOUT_MAX_ROWS
                )),
            },
        })
        .clone()
}

impl DsparkProvider<BackendSession> {
    /// The drafter bound to `target` on the active backend, capture off. A round that would
    /// cross a prefill cell steps one token at a time to the cell's edge, then drafts again.
    ///
    /// `history` is where the drafter's caches already hold this request's prefix: 0 for a
    /// fresh start, or the resume point of a history an earlier request left there, whose
    /// tokens are `tokens` (see `CachedHistory`). `floor` is where those rows start: 0, or the
    /// restore point of a history rebuilt there, below which the drafter reads nothing.
    /// `tokens` is `None` when the provider need not report its history at the end.
    ///
    /// # Errors
    /// As `DsparkForward::attach` and `FeatureTaps::attach`, or a floor above `history`.
    pub fn attach<M: Model + ?Sized>(
        target: &mut M,
        descriptor: &DsparkDescriptor,
        history: usize,
        floor: usize,
        tokens: Option<Vec<u32>>,
    ) -> Result<Self, String> {
        if floor > history {
            return Err(format!(
                "DSpark attach: floor {floor} above the history's end {history}"
            ));
        }
        let floor_u32 =
            u32::try_from(floor).map_err(|_| "DSpark: history floor too large")?;
        let batch = crate::prefill_batch();
        let rows =
            u32::try_from(batch).map_err(|_| "DSpark: prefill batch too large")?;
        let forward = crate::dspark_forward::DsparkForward::attach(target, descriptor)?;
        let taps = FeatureTaps::attach(target, descriptor, rows)?;
        taps.set_enabled(false);
        let capacity = target.state().kv_rt.capacity;
        Ok(Self::new(
            BackendSession {
                forward,
                taps,
                last: None,
                floor: floor_u32,
            },
            capacity,
            batch,
            history,
            floor,
            tokens,
            true,
        ))
    }
}

/// THE DRAFTER'S CACHES HOLD ONE HISTORY: the token stream of the last request that drafted, at
/// positions `0..tokens.len()`. Its rows came from the target's features at those positions, and
/// no store keeps features, so this is the only copy. A later request whose prompt agrees with
/// these tokens below its restore point resumes from them; any other request that drafts writes
/// its own stream over them, and a request that does not draft leaves them alone.
///
/// On Metal the caches are the target's per-layer KV arrays past its own layers. On CUDA the
/// native backend parks them apart, and `ticket` names that parked owner.
struct CachedHistory {
    #[cfg(feature = "cuda-speculative")]
    ticket: u64,
    tokens: Vec<u32>,
    /// Where the rows start: 0, or the restore point the history was rebuilt from. The tokens
    /// below it are the stream's, kept so a later prompt can be matched; the rows there are not.
    floor: usize,
}
impl CachedHistory {
    /// Whether a prompt restored at `start` resumes this history: it agrees with the stream below
    /// `start`, and the history's rows reach down to at most `start`.
    fn matches(&self, prompt: &[u32], start: usize) -> bool {
        start > 0
            && self.floor <= start
            && start <= self.tokens.len()
            && start <= prompt.len()
            && self.tokens[..start] == prompt[..start]
    }
}

/// The drafter history a request restored at `start` runs on, and where its rows start (design
/// 5.5): the resident history when it agrees with the prompt below `start`, otherwise one rebuilt
/// from `start`, whose attention reads no key below it.
#[cfg(not(feature = "cuda-speculative"))]
fn history_for(
    previous: Option<CachedHistory>,
    prompt: &[u32],
    start: usize,
) -> (Vec<u32>, usize, &'static str) {
    match previous {
        Some(h) if h.matches(prompt, start) => {
            let mut tokens = h.tokens;
            tokens.truncate(start);
            (tokens, h.floor, "resumed")
        }
        _ if start == 0 => (Vec::new(), 0, "fresh"),
        _ => (prompt[..start].to_vec(), start, "rebuilt"),
    }
}

/// A drafter admitted at startup, mapped after the target (`Weights::open_with_appended`).
/// Native descriptors are rebuilt against the live target mapping; no provider or GPU owner
/// crosses request threads.
pub struct Pairing {
    document: Document,
    /// The drafter's file, as the caller named it: the mapping must hold this one.
    draft: std::path::PathBuf,
    mask_token: u32,
    cached: std::sync::Mutex<Option<CachedHistory>>,
}
impl Pairing {
    /// The drafter as its file declares it, addressed in this target's paired mapping.
    ///
    /// # Errors
    /// When the mapping holds no drafter or another file, or the drafter file breaks the
    /// DSpark contract or does not fit the mapping.
    pub fn descriptor(
        &self,
        weights: &crate::weights::Weights,
        target_layer_count: u32,
    ) -> Result<DsparkDescriptor, String> {
        read_descriptor(
            &self.document,
            &weights.tensors,
            weights.byte_len(),
            crate::speculative::drafter_offset(weights, &self.draft)?,
            target_layer_count,
            self.mask_token,
        )
    }
    pub(crate) fn can_resume(&self, prompt: &[u32], start: usize) -> bool {
        self.cached
            .lock()
            .is_ok_and(|c| c.as_ref().is_some_and(|h| h.matches(prompt, start)))
    }
    /// One request's provider on the active backend. At `start > 0` it resumes the drafter's
    /// history when that history covers this prompt below `start` (see `CachedHistory`), and
    /// otherwise REBUILDS the history from `start` (design 5.5): the rows begin at the restore
    /// point, the drafter's attention reads nothing below it, and the prompt's tokens below it
    /// are kept only so a later prompt can be matched. A conversation that adopted another's
    /// prefix, or came back after another conversation drafted, drafts over the rows since its
    /// restore instead of not drafting at all. Always `Ok(true)` on this backend.
    ///
    /// The history is TAKEN before the run and put back only by a run that succeeds: a run that
    /// fails part way has written rows its record would not describe.
    #[cfg(not(feature = "cuda-speculative"))]
    pub(crate) fn run_cached<A: crate::Architecture>(
        &self,
        target: &mut crate::Workflow<A>,
        prompt: &[u32],
        start: usize,
        run: &mut crate::speculative::DraftRun<'_>,
    ) -> Result<bool, String> {
        if start > prompt.len() {
            return Err(format!(
                "draft restore point {start} past the prompt's {} tokens",
                prompt.len()
            ));
        }
        let (tokens, floor) = {
            let mut cache = self
                .cached
                .lock()
                .map_err(|_| "draft cache metadata poisoned")?;
            let held = cache.as_ref().map_or(0, |h| h.tokens.len());
            let (tokens, floor, how) = history_for(cache.take(), prompt, start);
            if crate::log_on() {
                eprintln!(
                    "[imparo] dspark history: start={start} held={held} floor={floor} -> {how}"
                );
            }
            (tokens, floor)
        };
        let descriptor =
            self.descriptor(&target.weights, target.plan.layers.len() as u32)?;
        let mut provider = DsparkProvider::<BackendSession>::attach(
            target,
            &descriptor,
            start,
            floor,
            Some(tokens),
        )?;
        let result = run(target, &mut provider);
        let history = provider.committed_history();
        if crate::log_on() {
            // The next turn resumes only below the history's end: equal to the target's
            // filled, a turn can resume at any grid point of this stream.
            eprintln!(
                "[imparo] dspark history: holds {} of the target's {} tokens, rows from {}",
                history.as_ref().map_or(0, |(t, _)| t.len()),
                target.state.kv_rt.filled,
                history.as_ref().map_or(0, |(_, f)| *f)
            );
        }
        match (result, provider.finish()) {
            (Ok(()), Ok(())) => {
                if let Some((tokens, floor)) = history {
                    *self
                        .cached
                        .lock()
                        .map_err(|_| "draft cache metadata poisoned")? =
                        Some(CachedHistory { tokens, floor });
                }
                Ok(true)
            }
            (Err(e), Ok(())) => Err(e),
            (Ok(()), Err(e)) => Err(format!("draft teardown: {e}")),
            (Err(e), Err(c)) => Err(format!("{e}; draft teardown: {c}")),
        }
    }
    /// One request's provider on the active backend: attached before the callback, finished
    /// after it. It records no history, and its appends overwrite whatever history the drafter's
    /// caches held, so that record goes first.
    #[cfg(not(feature = "cuda-speculative"))]
    pub(crate) fn run<A: crate::Architecture>(
        &self,
        target: &mut crate::Workflow<A>,
        run: &mut crate::speculative::DraftRun<'_>,
    ) -> Result<(), String> {
        self.cached
            .lock()
            .map_err(|_| "draft cache metadata poisoned")?
            .take();
        let descriptor =
            self.descriptor(&target.weights, target.plan.layers.len() as u32)?;
        let mut provider =
            DsparkProvider::<BackendSession>::attach(target, &descriptor, 0, 0, None)?;
        let result = run(target, &mut provider);
        match (result, provider.finish()) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(e), Ok(())) => Err(e),
            (Ok(()), Err(e)) => Err(format!("draft teardown: {e}")),
            (Err(e), Err(c)) => Err(format!("{e}; draft teardown: {c}")),
        }
    }
    #[cfg(feature = "cuda-speculative")]
    pub(crate) fn run_cached<A: crate::Architecture>(
        &self,
        target: &mut crate::Workflow<A>,
        prompt: &[u32],
        start: usize,
        run: &mut crate::speculative::DraftRun<'_>,
    ) -> Result<bool, String> {
        if target.plan.config.architecture != "lfm2"
            || target.state.host_forward
            || target
                .state
                .layer_outputs
                .as_ref()
                .is_some_and(super::layer_outputs::LayerOutputCapture::attached)
        {
            return Err("cached draft requires its exclusive CUDA target".into());
        }
        if start > 0 && start != target.state.kv_rt.filled {
            return Err("cached draft differs from actual target restore".into());
        }
        let descriptor = cuda_descriptor(
            &self.descriptor(&target.weights, target.plan.layers.len() as u32)?,
            u32::try_from(target.state.kv_rt.capacity)
                .map_err(|_| "draft capacity overflow")?,
            u32::try_from(crate::prefill_batch())
                .map_err(|_| "draft batch overflow")?,
        )?;
        let mut cache = self
            .cached
            .lock()
            .map_err(|_| "draft cache metadata poisoned")?;
        let mut provider = if start > 0 {
            let Some(mut previous) = cache.take().filter(|h| h.matches(prompt, start))
            else {
                return Ok(false);
            };
            let coverage = previous.tokens.len();
            // Native identity rejection leaves the pool-restored target unchanged.
            if unsafe {
                imparo_cuda::dspark::resume(
                    previous.ticket,
                    u32::try_from(start).map_err(|_| "draft start overflow")?,
                    &descriptor.config,
                    &descriptor.layers,
                    &descriptor.target_layers,
                )
            }
            .is_err()
            {
                eprintln!(
                    "[dspark-cache] native identity rejected; ordinary continuation"
                );
                return Ok(false);
            }
            eprintln!(
                "[dspark-cache] resumed start={start} coverage={coverage} ticket={}",
                previous.ticket
            );
            previous.tokens.truncate(start);
            DsparkProvider::<CudaSession>::bind(
                target,
                descriptor,
                start,
                Some(previous.tokens),
            )
        } else {
            let previous = cache.take();
            // Reuse only the checked physical owner. A new request has no valid
            // logical history; initialize_at(0) and observed Prefill rebuild it.
            let reused = (match crate::lfm_retained_domain() {
                0 => {
                    std::env::var("IMPARO_LAB_DSPARK_COLD_OWNER_REUSE").as_deref()
                        == Ok("1")
                }
                domain => domain == 1,
            }) && previous.as_ref().is_some_and(|h| unsafe {
                imparo_cuda::dspark::resume(
                    h.ticket,
                    0,
                    &descriptor.config,
                    &descriptor.layers,
                    &descriptor.target_layers,
                )
                .is_ok()
            });
            if reused {
                eprintln!("[dspark-cache] cold-owner-reused history=0");
                DsparkProvider::<CudaSession>::bind(
                    target,
                    descriptor,
                    0,
                    Some(Vec::new()),
                )
            } else {
                let mut p = unsafe {
                    DsparkProvider::<CudaSession>::attach(target, descriptor)?
                };
                p.tokens = Some(Vec::new());
                p
            }
        };
        let result = run(target, &mut provider);
        if result.is_ok() {
            match provider.park() {
                Ok(h) => {
                    *cache = Some(h);
                    return Ok(true);
                }
                Err(e) => {
                    let _ = provider.finish();
                    return Err(e);
                }
            }
        }
        match (result, provider.finish()) {
            (Err(e), Ok(())) => Err(e),
            (Err(e), Err(c)) => Err(format!("{e}; draft teardown: {c}")),
            _ => unreachable!(),
        }
    }
    pub(crate) fn can_start(
        &self,
        start: usize,
        limit: usize,
        capacity: usize,
    ) -> bool {
        let Some(block) = self
            .document
            .unsigned_value("dflash.block_size")
            .and_then(|n| usize::try_from(n).ok())
        else {
            return false;
        };
        start_admits(
            start,
            limit,
            capacity,
            block,
            crate::prefill_batch().max(1),
            served_provider_bridges(),
        )
    }

    /// The DSpark drafter in the file `draft`, opened as it is: no manifest and no offline
    /// step -- the caller names the drafter and maps it after the target's file. The drafter's
    /// fit to the target is checked where it is read (`descriptor`). `mask_token` is the
    /// caller's, else the drafter's own `tokenizer.ggml.mask_token_id`.
    ///
    /// # Errors
    /// When the file does not parse, is not a DFlash drafter with the Markov head (only DSpark
    /// pairs), or names no mask token and none is given, or names one the caller's disagrees
    /// with.
    pub fn open(
        draft: &std::path::Path,
        mask_token: Option<u32>,
    ) -> Result<(std::path::PathBuf, Self), String> {
        let document =
            imparo_gguf::read(draft).map_err(|e| format!("{}: {e}", draft.display()))?;
        if document.string_value("general.architecture") != Some("dflash") {
            return Err(format!(
                "{}: not a dflash drafter (general.architecture {:?})",
                draft.display(),
                document.string_value("general.architecture")
            ));
        }
        if document.tensor("markov_w1.weight").is_none() {
            return Err(format!(
                "{}: a DFlash drafter without the Markov head; only DSpark pairs",
                draft.display()
            ));
        }
        let declared = document
            .unsigned_value("tokenizer.ggml.mask_token_id")
            .and_then(|m| u32::try_from(m).ok());
        let mask_token = match (mask_token, declared) {
            (Some(m), Some(d)) if m != d => {
                return Err(format!(
                    "--draft-mask-token {m} disagrees with the drafter's mask token {d}"
                ));
            }
            (Some(m), _) | (None, Some(m)) => m,
            (None, None) => {
                return Err(format!(
                    "{}: names no mask token; pass --draft-mask-token",
                    draft.display()
                ));
            }
        };
        let draft = draft
            .canonicalize()
            .map_err(|e| format!("{}: {e}", draft.display()))?;
        Ok((
            draft.clone(),
            Self {
                document,
                draft,
                mask_token,
                cached: std::sync::Mutex::new(None),
            },
        ))
    }
    /// The mask token the drafter fills its block with.
    #[must_use]
    pub fn mask_token(&self) -> u32 {
        self.mask_token
    }
    #[cfg(feature = "cuda-speculative")]
    pub(crate) fn run<A: crate::Architecture>(
        &self,
        target: &mut crate::Workflow<A>,
        run: &mut crate::speculative::DraftRun<'_>,
    ) -> Result<(), String> {
        if target.plan.config.architecture != "lfm2" {
            return Err("DSpark pairing requires its LFM2 target adapter".into());
        }
        let descriptor = cuda_descriptor(
            &self.descriptor(&target.weights, target.plan.layers.len() as u32)?,
            u32::try_from(target.state.kv_rt.capacity)
                .map_err(|_| "draft capacity overflow")?,
            u32::try_from(crate::prefill_batch())
                .map_err(|_| "draft batch overflow")?,
        )?;
        // The mutable workflow borrow keeps its checked mapping alive for the whole
        // callback. Provider is !Send and cannot escape; finish precedes return.
        let mut provider =
            unsafe { DsparkProvider::<CudaSession>::attach(target, descriptor)? };
        let result = run(target, &mut provider);
        match (result, provider.finish()) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(e), Ok(())) => Err(e),
            (Ok(()), Err(e)) => Err(format!("draft teardown: {e}")),
            (Err(e), Err(c)) => Err(format!("{e}; draft teardown: {c}")),
        }
    }
}

/// The target's tapped layer outputs, written on the device into `BufId::DraftFeatures` as
/// each tapped layer finishes. In a forward's row r, tap k's residual sits at
/// `r * taps * n_embd + k * n_embd`: the taps side by side, as the drafter's `fc` reads them.
///
/// While capturing, the target keeps the full residual at every tapped layer: an active
/// subscriber turns off the forward's tail cuts and the mega-kernel decode.
pub struct FeatureTaps {
    /// Read by the target's capture on every forward; the subscription lives as long as this.
    enabled: Arc<AtomicBool>,
    /// The last forward the taps wrote: its start position in the high word, its rows in the
    /// low word; `NO_ROWS` when none was captured since capture was turned on.
    last: Arc<AtomicU64>,
    /// Floats per feature row: taps times the target's residual width.
    stride: u32,
}

/// `FeatureTaps::last` with no forward captured.
const NO_ROWS: u64 = u64::MAX;

impl FeatureTaps {
    /// Subscribes to the descriptor's taps, capturing, and sizes the feature buffer for `rows`
    /// rows per forward. With `IMPARO_DRAFT_TAP_PROBE=1` every forward's feature rows are read
    /// back and compared with the tapped layers' outputs, bit for bit.
    ///
    /// # Errors
    /// When the target has no device forward or already has a subscriber, its plan does not
    /// carry this drafter, or the buffer cannot be allocated.
    pub fn attach<M: crate::Model + ?Sized>(
        target: &mut M,
        descriptor: &DsparkDescriptor,
        rows: u32,
    ) -> Result<Self, String> {
        if target.state().host_forward
            || target
                .state()
                .layer_outputs
                .as_ref()
                .is_some_and(crate::layer_outputs::LayerOutputCapture::attached)
        {
            return Err(
                "DSpark feature taps need a device target with no other output subscriber"
                    .into(),
            );
        }
        let plan = target.plan();
        // The drafter's caches and attention dims were made from the plan at load.
        if plan.drafter.as_ref() != Some(&descriptor.drafter_plan()) {
            return Err(
                "DSpark feature taps: the target's plan does not carry this drafter"
                    .into(),
            );
        }
        if plan.config.n_embd != descriptor.target_hidden {
            return Err(format!(
                "DSpark feature taps: the target's residual is {} wide, the drafter reads {}",
                plan.config.n_embd, descriptor.target_hidden
            ));
        }
        let hidden = descriptor.target_hidden;
        let taps = descriptor.target_layers.clone();
        let stride = u32::try_from(taps.len())
            .ok()
            .and_then(|n| n.checked_mul(hidden))
            .ok_or("DSpark feature row width overflows")?;
        crate::gpu_support::be()
            .alloc(
                imparo_backend::BufId::DraftFeatures,
                plan.draft_feature_bytes(rows as usize),
            )
            .map_err(|rc| format!("draft features allocation rc={rc}"))?;
        let probe = (std::env::var("IMPARO_DRAFT_TAP_PROBE").as_deref() == Ok("1"))
            .then(|| std::sync::Mutex::new(Vec::<Vec<f32>>::new()));
        let enabled = Arc::new(AtomicBool::new(true));
        let last = Arc::new(AtomicU64::new(NO_ROWS));
        let written = Arc::clone(&last);
        target.state_mut().layer_outputs = Some(
            crate::layer_outputs::LayerOutputCapture {
                enabled: Arc::downgrade(&enabled),
                layers: taps.clone(),
                capture: crate::layer_outputs::CaptureFn::observer(
                    move |layer, start, n, source| {
                        let k = taps.binary_search(&layer).map_err(|_| {
                            format!("DSpark feature taps: layer {layer} is not a tap")
                        })?;
                        if n > rows {
                            return Err(format!(
                                "DSpark feature taps: a forward of {n} rows, the buffer holds {rows}"
                            ));
                        }
                        crate::gpu_support::be().scatter_strided(
                            imparo_backend::BufId::DraftFeatures,
                            source,
                            hidden,
                            k as u32 * hidden,
                            stride,
                            n,
                        );
                        written.store(
                            (u64::from(start) << 32) | u64::from(n),
                            Ordering::Relaxed,
                        );
                        if let Some(seen) = &probe {
                            probe_feature_rows(
                                seen, &taps, k, start, n, hidden, source,
                            );
                        }
                        Ok(())
                    },
                ),
            },
        );
        Ok(Self {
            enabled,
            last,
            stride,
        })
    }

    /// Starts or stops capturing. Stopping forgets the last forward, so an append needs a
    /// forward captured since capture was last turned on.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
        if !enabled {
            self.last.store(NO_ROWS, Ordering::Relaxed);
        }
    }

    /// The last captured forward: its start position and row count.
    fn last_forward(&self) -> Option<(u32, u32)> {
        let last = self.last.load(Ordering::Relaxed);
        (last != NO_ROWS).then_some(((last >> 32) as u32, last as u32))
    }

    /// Refuses unless the last captured forward started at `start` and wrote at least `n` rows.
    /// Then its first `n` feature rows belong to those positions, and the target's buffers hold
    /// `n` rows.
    ///
    /// # Errors
    /// When the last captured forward was another, or none was captured.
    pub fn check_rows(&self, start: u32, n: u32) -> Result<(), String> {
        match self.last_forward() {
            Some((s, r)) if s == start && n <= r => Ok(()),
            Some((s, r)) => Err(format!(
                "DSpark features: {n} rows at {start} asked, the last forward wrote {r} rows at {s}"
            )),
            None => Err(format!(
                "DSpark features: {n} rows at {start} asked, no forward was captured"
            )),
        }
    }

    /// After a tree verify: node `path[i]`'s feature row moves to row `i`, so the accepted path
    /// reads as a chain. A node's row is never below its depth, so no copy overwrites the row a
    /// later copy reads.
    ///
    /// # Errors
    /// When `path` is not a root-first path of rows the last forward wrote, or the backend
    /// fails.
    pub fn compact(&self, path: &[i32]) -> Result<(), String> {
        let rows = self.last_forward().map_or(0, |(_, r)| r);
        let nodes: Vec<u32> =
            path.iter().filter_map(|&p| u32::try_from(p).ok()).collect();
        if nodes.len() != path.len()
            || nodes.first() != Some(&0)
            || nodes.windows(2).any(|w| w[0] >= w[1])
            || nodes.last().is_some_and(|&p| p >= rows)
        {
            return Err(format!(
                "DSpark features: {path:?} is not a path of the last forward's {rows} rows"
            ));
        }
        let be = crate::gpu_support::be();
        be.begin();
        for (depth, &node) in (0_u32..).zip(&nodes) {
            if node != depth {
                be.copy_range(
                    imparo_backend::BufId::DraftFeatures,
                    depth * self.stride,
                    imparo_backend::BufId::DraftFeatures,
                    node * self.stride,
                    self.stride,
                );
            }
        }
        be.end()
            .map_err(|rc| format!("DSpark feature compaction rc={rc}"))
    }
}

/// `IMPARO_DRAFT_TAP_PROBE=1`: keeps each tap's residual rows, read back as the tap is
/// written, and once the forward's last tap is written reads the feature rows and compares
/// every tap's sub-block with them. Comparing after the LAST tap is what catches a tap
/// written over another tap's sub-block.
fn probe_feature_rows(
    seen: &std::sync::Mutex<Vec<Vec<f32>>>,
    taps: &[u32],
    k: usize,
    start: u32,
    n: u32,
    hidden: u32,
    source: imparo_backend::BufId,
) {
    let be = crate::gpu_support::be();
    let _ = be.end();
    let (n, hidden) = (n as usize, hidden as usize);
    let stride = taps.len() * hidden;
    let mut rows = vec![0.0_f32; n * hidden];
    be.read(source, 0, &mut rows);
    let mut features = Vec::new();
    if k + 1 == taps.len() {
        features = vec![0.0_f32; n * stride];
        be.read(imparo_backend::BufId::DraftFeatures, 0, &mut features);
    }
    // The forward goes on encoding after this call: reopen the region ended above, as gprobe
    // does. Without it every later dispatch of the forward is dropped.
    be.begin();
    let mut seen = seen
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if k == 0 {
        seen.clear();
    }
    if seen.len() != k {
        eprintln!(
            "[imparo] draft tap probe: start={start} tap {k} arrived after {} taps",
            seen.len()
        );
        seen.clear();
        return;
    }
    seen.push(rows);
    if k + 1 < taps.len() {
        return;
    }
    let differing: Vec<usize> = seen
        .iter()
        .enumerate()
        .map(|(tap, x)| {
            (0..n * hidden)
                .filter(|&e| {
                    features[(e / hidden) * stride + tap * hidden + e % hidden]
                        .to_bits()
                        != x[e].to_bits()
                })
                .count()
        })
        .collect();
    eprintln!(
        "[imparo] draft tap probe: start={start} rows={n} layers={taps:?} values_per_tap={} \
         differing={differing:?} bit_equal={}",
        n * hidden,
        differing.iter().all(|&d| d == 0)
    );
    seen.clear();
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn cuda_cost_cache_identity_is_stable_and_separates_execution_contracts() {
        let descriptor = || super::CudaDescriptor {
            config: super::Config { embedding:0, fc:1, enc_norm:2, out_norm:3,
                markov1:4, markov2:5, confidence:6, confidence_bias:7,
                hidden:2048, ffn:7168, vocab:128256, heads:32, kv_heads:8, head_dim:64,
                rank:64, block_size:9, mask_token:125017, kv_capacity:1024,
                batch_capacity:512, target_hidden:2048, eps:1e-5, rope_theta:10000. },
            layers: vec![super::Layer { attn_norm:10, q:11, k:12, v:13, o:14, qn:15,
                kn:16, ffn_norm:17, gate:18, up:19, down:20 }],
            target_layers:vec![2,5],
        };
        let base=("cuda-dspark-cache-v1:full-backend-identity".to_string(),12345);
        let settings=vec![("IMPARO_DSPARK_TREE".to_string(),"budget".to_string())];
        let make=|d:&super::CudaDescriptor,s:&[(String,String)]| super::cuda_cost_cache_key(base.clone(),d,s);
        let expected=make(&descriptor(),&settings);
        assert_eq!(expected,make(&descriptor(),&settings));
        assert!(!expected.0.contains("|space=v"));
        assert_eq!(expected.1,12345);
        for variant in 0..6 {
            let mut d=descriptor();
            match variant {
                0=>d.config.kv_capacity+=128,
                1=>d.config.batch_capacity=128,
                2=>d.config.mask_token+=1,
                3=>d.config.eps=f32::from_bits(d.config.eps.to_bits()+1),
                4=>d.target_layers[0]+=1,
                _=>d.layers[0].gate+=1,
            }
            assert_ne!(expected,make(&d,&settings));
        }
        assert_ne!(expected,make(&descriptor(),&[]));
        assert_ne!(expected,super::cuda_cost_cache_key(("other-backend".into(),12345),&descriptor(),&settings));
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn cuda_top4_acceptance_restores_matching_state_and_rejects_other_features() {
        use crate::accept_model::{AcceptModel, Features};
        let mut original=AcceptModel::derived(0.5);
        let probabilities=[0.4_f32,0.3,0.2,0.1];
        let features=Features::position(0.65,&probabilities.map(f32::ln));
        for _ in 0..8 { original.observe(&features,&[0.4,0.3,0.2,0.1],Some(1)); }
        let row=super::accept_row(&original,4);
        let table=imparo_host::VerifyCostTable {
            steps:vec![(2,2,2000.,3),(0,0,1000.,3)], accept:Some(row), ..Default::default()
        };
        let key="cuda-dspark-cost-v2-test";
        let text=imparo_host::verify_cost_body(key,12345,512,&table);
        let parsed=imparo_host::parse_verify_cost(&text,key).unwrap();
        let stored=parsed.accept.as_ref().unwrap();
        assert_eq!(stored.top_k,4);
        let resumed=super::accept_start(Some(stored),4,true).unwrap();
        assert_eq!(original.state(),resumed.state());
        let before=original.level2();let after=resumed.level2();
        assert_eq!(before.prior,after.prior);assert_eq!(before.z,after.z);
        assert_eq!(before.n,after.n);assert_eq!(before.labels,after.labels);
        assert_eq!(before.guard,after.guard);assert_eq!(before.seen,after.seen);
        let mut different=stored.clone();different.top_k=8;
        assert_eq!(super::accept_start(Some(&different),4,true).unwrap().state(),
            AcceptModel::derived(0.5).state());
        assert!(imparo_host::parse_verify_cost(&text,"different-contract").is_none());
        let mut cost=crate::verify_cost::VerifyCost::new(16).unwrap();
        assert!(cost.seed(&parsed.steps));
        assert_eq!(cost.probe(),Some(4),"partial table resumes at the next unmeasured width");
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn cuda_full_pool_keeps_edge_scores_and_omits_unavailable_head_features() {
        let found = super::cuda_frontier_candidates(
            100, &[100, 11, 12, 13, 14], &[-1, 0, 0, 1, 2],
            &[1.0_f32, 0.5, 0.4, 0.45, 0.08].map(f32::ln),
        ).unwrap();
        assert!((found.candidates[2].q - 0.9).abs() < 1e-6);
        assert!((found.candidates[3].q - 0.2).abs() < 1e-6);
        assert_eq!(found.candidates[2].parent, 0);
        assert_eq!(found.candidates[3].parent, 1);
        assert!(found.features.iter().all(Option::is_none));
        let tokens: Vec<_> = (0..33).collect();
        let mut parents = vec![-1];
        parents.extend((1..33).map(|i| if i < 5 {0} else {i-4}));
        let scores: Vec<_> = (0..33).map(|i| -((i+3)/4) as f32).collect();
        assert_eq!(super::cuda_frontier_candidates(0, &tokens, &parents, &scores)
            .unwrap().candidates.len(), 32);
        parents[32] = 32;
        assert!(super::cuda_frontier_candidates(0, &tokens, &parents, &scores).is_err());
    }

    #[cfg(feature = "cuda-speculative")]
    fn acceptance_fixture() -> (Vec<u32>, Vec<i32>, Vec<f32>, Vec<imparo_cuda::dspark::FrontierGroup>) {
        let tokens: Vec<u32> = (0..33).collect();
        let mut parents = vec![-1];
        parents.extend((1..33).map(|i| if i < 5 { 0 } else { i - 4 }));
        let logp = [0.4_f32, 0.3, 0.2, 0.1].map(f32::ln);
        let mut scores = vec![0.0_f32; 33];
        for row in 1..33 {
            scores[row] = scores[parents[row] as usize] + logp[if row < 5 { row - 1 } else { 0 }];
        }
        let groups = (0..29).map(|parent| imparo_cuda::dspark::FrontierGroup {
            parent,
            tokens: if parent == 0 { [1, 2, 3, 4] }
                else { [parent + 4, 1000 + parent * 4 + 1, 1000 + parent * 4 + 2, 1000 + parent * 4 + 3] },
            logp,
            confidence: 0.65 + (parent % 3) as f32 * 0.05,
        }).collect();
        (tokens, parents, scores, groups)
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn cuda_acceptance_restores_complete_parent_groups_and_real_features() {
        use crate::accept_model::{Features, Kind};
        let (tokens, parents, scores, groups) = acceptance_fixture();
        let found = super::cuda_acceptance_candidates(0, &tokens, &parents, &scores, &groups, 512, 20000).unwrap();
        assert_eq!(found.candidates.len(), 116);
        assert_eq!(found.ngram_branch_prefix, Some(32));
        assert_eq!(found.candidates[..32].iter().map(|c| c.token).collect::<Vec<_>>(), tokens[1..]);
        let children = super::children(&found.candidates);
        for group in &groups {
            let kids = &children[if group.parent == 0 { 116 } else { group.parent as usize - 1 }];
            assert_eq!(kids.len(), 4, "each expanded parent has all four siblings");
            let expected = Features::position(group.confidence, &group.logp);
            for rank in 0..4 {
                let c = *kids.iter().find(|&&c| found.candidates[c].token == group.tokens[rank]).unwrap();
                let f = found.features[c].unwrap();
                assert_eq!(f.kind, if rank == 0 { Kind::Pick } else { Kind::Sibling });
                assert_eq!(f.context, 512);
                assert_eq!(f.depth, (group.parent + 3) / 4);
                assert_eq!(f.logit_head, expected[rank].logit_head);
                assert_eq!(f.logit_share_pick, expected[rank].logit_share_pick);
                assert_eq!(f.log_r, expected[rank].log_r);
                assert_eq!(found.candidates[c].q, f64::from(group.logp[rank]).exp());
            }
        }
        assert_ne!(found.candidates[0].q, f64::from(groups[0].confidence), "head must not replace baseline q");
        assert!(found.features.iter().all(Option::is_some));
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn cuda_acceptance_rejects_missing_duplicate_and_invalid_group_evidence() {
        let (tokens, parents, scores, groups) = acceptance_fixture();
        let check = |groups: &[imparo_cuda::dspark::FrontierGroup]| {
            super::cuda_acceptance_candidates(0, &tokens, &parents, &scores, groups, 512, 20000)
        };
        assert!(check(&groups[..28]).is_err());
        for invalid in 0..7 {
            let (_, _, _, mut bad) = acceptance_fixture();
            match invalid {
                0 => bad[1].parent = 0,
                1 => bad[0].tokens[1] = bad[0].tokens[0],
                2 => bad[0].confidence = f32::NAN,
                3 => bad[0].logp[0] = 0.1,
                4 => bad[0].logp = [0.0; 4],
                5 => bad[0].tokens[0] = 20000,
                6 => bad[0].logp[0] -= 0.01,
                _ => unreachable!(),
            }
            assert!(check(&bad).is_err(), "invalid group variant {invalid}");
        }
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn cuda_acceptance_preserves_native_sibling_ngram_branch_points() {
        let (tokens, parents, scores, groups) = acceptance_fixture();
        let mut found = super::cuda_acceptance_candidates(0, &tokens, &parents, &scores, &groups, 512, 20000).unwrap();
        assert_eq!(found.features[1].unwrap().kind, crate::accept_model::Kind::Sibling);
        let mut index = crate::ngram::NgramIndex::default();
        index.observe_all(&[90, 91]);
        index.count(&[91, 0, 2], 9000);
        index.count(&[0, 1, 1005], 9001);
        let lookup = super::NgramLookup { request: &index, table: None };
        super::ngram_chains(&mut found, &lookup, 0, &[0.75; 8], &[], 512, 8, 15);
        let added: Vec<_> = found.candidates[116..].iter().map(|c| (c.token, c.parent)).collect();
        assert_eq!(added, [(9000, 1)]);
        assert_eq!(index.tail(), &[90, 91]);
    }

    #[cfg(feature = "cuda-speculative")]
    struct AcceptanceTestSession<const ADAPTIVE: bool = false>;

    #[cfg(feature = "cuda-speculative")]
    impl<const ADAPTIVE: bool> super::DsparkSession for AcceptanceTestSession<ADAPTIVE> {
        fn drafts(&self) -> usize { 8 }
        fn reset(&mut self) -> Result<(), String> { Ok(()) }
        fn capture(&mut self, _: bool) -> Result<(), String> { Ok(()) }
        fn append(&mut self, _: usize, _: usize) -> Result<(), String> { Ok(()) }
        fn compact(&mut self, _: &[i32]) -> Result<(), String> { Ok(()) }
        fn generate(&mut self, _: &mut dyn super::Model, _: usize, _: u32) -> Result<Vec<u32>, String> {
            Err("test does not generate".into())
        }
        fn detach(&mut self) -> Result<(), String> { Ok(()) }
        fn passive_tree_costs(&self) -> bool { !ADAPTIVE }
        fn complete_tree_costs(&self) -> bool { ADAPTIVE }
        fn cost_cache_allowed(&self) -> bool { false }
        fn verify_max_rows(&self) -> usize { 16 }
        fn fresh_accept_model(&self) -> bool { true }
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn cuda_acceptance_labels_complete_groups_at_accepted_parents_only() {
        let (tokens, parents, scores, groups) = acceptance_fixture();
        let found = super::cuda_acceptance_candidates(0, &tokens, &parents, &scores, &groups, 512, 20000).unwrap();
        let tree = crate::speculative::TreeBudget::Fixed { nodes: 16, offset: 0.0 }
            .tree(0, &found.candidates, &[], crate::speculative::TreeRound::default()).unwrap();
        let row = |token| tree.tokens.iter().position(|&t| t == token).unwrap() as i32;
        let path = [0, row(1), row(5)];
        // The target's next pick is a true sibling omitted by the original beam
        // and by this verification tree. The complete group still labels it.
        let next = groups[5].tokens[3];
        assert!(!tree.tokens.contains(&next));
        let mut provider = super::DsparkProvider::new(AcceptanceTestSession::<false>, 1024, 512, 0, 0, None, false);
        provider.accept = Some(crate::accept_model::AcceptModel::derived(0.5));
        provider.last_plan = Some(super::RoundPlan {
            rank: tree.rank.clone(), s: tree.s.clone(), s_n: tree.s_n.clone(),
            explored: false, parents: tree.parents.clone(), lo: 0, hi: 16,
            ctx: 512, node: tree.node.clone(), found, offset: 0.0,
        });
        provider.learn_accept(&path, next);
        let model = provider.accept.as_ref().unwrap();
        assert_eq!(model.state().1, 3, "anchor, token 1, token 5 are the only labelled parents");
        assert_eq!(model.request_losses().2, 12, "all four true siblings at each accepted parent are labelled");
        let plan = provider.last_plan.as_ref().unwrap();
        let priced = super::priced_candidates(model, &plan.found);
        assert!(priced.iter().zip(&plan.found.candidates).any(|(a, b)| a.q != b.q));
        for kids in super::children(&priced).into_iter().filter(|k| !k.is_empty()) {
            assert!(kids.iter().map(|&c| priced[c].q).sum::<f64>() < 1.0);
        }
    }
    #[test]
    fn retained_short_domain_reuses_bounded_tail_without_changing_unbound_sessions() {
        assert_eq!(super::draft_minimum_remaining(0, 9, false), 9);
        assert_eq!(super::draft_minimum_remaining(0, 9, true), 3);
        for domain in [1, 2, 3] {
            assert_eq!(super::draft_minimum_remaining(domain, 9, false), 3);
            assert_eq!(super::draft_minimum_remaining(domain, 2, false), 2);
        }
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn cuda_budget_prices_complete_successful_warm_rounds_once() {
        use crate::speculative::{DraftProvider, DraftTree, TreeRoundTiming, TreeSubmission};
        use std::time::Duration as D;
        let mut p = super::DsparkProvider::new(AcceptanceTestSession::<true>, 1024, 512, 0, 0, None, false);
        assert!(p.cost_key.is_none());
        let tree = DraftTree::plain(vec![1,2,3,4], vec![-1,0,1,1]);
        let timing = TreeRoundTiming { context:512, rows:4, accepted:3,
            fixed:D::from_micros(1000), verify:D::from_micros(2000),
            commit:D::from_micros(1000), total:D::from_micros(4500),
            submission:TreeSubmission::Ordinary, continuing:true };
        p.observe_round_fixed(timing.fixed);
        p.observe_verify(4,timing.verify);
        p.observe_tree_round(&tree,&timing);
        assert_eq!(p.cost_seen,0,"cold round is not a steady-state sample");
        for _ in 0..3 {
            p.observe_round_fixed(timing.fixed);
            p.observe_verify(4,timing.verify);
            p.observe_tree_round(&tree,&timing);
        }
        assert_eq!(p.cost_seen,3);
        let cost=p.cost.as_ref().unwrap();
        assert_eq!(cost.steps().iter().map(|s|s.seen).sum::<u64>(),3);
        assert!(cost.fixed_us().unwrap() > 2400.,"price includes history commit and host remainder");
        assert_eq!(p.rate.seen(512),3);
        for bad in [TreeRoundTiming {continuing:false,..timing},
            TreeRoundTiming {submission:TreeSubmission::Capture,..timing},
            TreeRoundTiming {submission:TreeSubmission::Replayed,..timing},
            TreeRoundTiming {total:D::from_micros(1000),..timing}] {
            p.observe_tree_round(&tree,&bad);
        }
        assert_eq!(p.cost_seen,3);
        assert_eq!(p.rate.seen(512),3);
    }
    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn cuda_frontier_uses_conditional_scores_and_remaps_dfs_parents() {
        let tokens = [100, 11, 12, 13, 14];
        let parents = [-1, 0, 0, 1, 2];
        let scores = [1.0_f32, 0.5, 0.45, 0.49, 0.10].map(f32::ln);
        let tree = super::select_cuda_frontier(100, &tokens, &parents, &scores, 3, &[]).unwrap().unwrap();
        // 13 has joint probability .49, ahead of sibling12's .45. Applying
        // cumulative score as an edge again would incorrectly choose12 instead.
        assert_eq!(tree.tokens, [100, 11, 13]);
        assert_eq!(tree.parents, [-1, 0, 1]);
        let tree = super::select_cuda_frontier(100, &tokens, &parents, &scores, 4, &[]).unwrap().unwrap();
        assert_eq!(tree.tokens, [100, 11, 13, 12]);
        assert_eq!(tree.parents, [-1, 0, 1, 0]);
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn cuda_frontier_prunes_stop_subtrees_and_declines_an_anchor_only_tree() {
        let tokens = [100, 11, 12, 13, 14];
        let parents = [-1, 0, 0, 1, 2];
        let scores = [1.0_f32, 0.5, 0.45, 0.49, 0.10].map(f32::ln);
        let tree = super::select_cuda_frontier(100, &tokens, &parents, &scores, 4, &[11]).unwrap().unwrap();
        assert_eq!(tree.tokens, [100, 12, 14]);
        assert_eq!(tree.parents, [-1, 0, 1]);
        assert!(super::select_cuda_frontier(100, &tokens, &parents, &scores, 4, &[11,12]).unwrap().is_none());
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn cuda_frontier_rejects_invalid_metadata_before_selection() {
        assert!(super::select_cuda_frontier(100, &[100,11], &[-1,0], &[0.0, f32::NAN], 2, &[]).is_err());
        assert!(super::select_cuda_frontier(100, &[100,11], &[-1,0], &[0.0, 0.1], 2, &[]).is_err());
        assert!(super::select_cuda_frontier(100, &[100,11], &[-1,1], &[0.0, -0.1], 2, &[]).is_err());
        assert!(super::select_cuda_frontier(100, &[100,11], &[-1,0], &[0.0, -0.1], 17, &[]).is_err());
        assert!(super::select_cuda_frontier(100, &[99,11], &[-1,0], &[0.0, -0.1], 2, &[]).is_err());
    }

    use super::*;
    use imparo_gguf::{ArraySummary, TensorInfo, ValueType};

    /// A prompt that ends within one block of a cell edge starts drafting on a provider that bridges
    /// edges, and only there; the other refusals stay.
    #[test]
    fn a_prompt_ending_near_a_cell_edge_is_admitted_when_the_provider_bridges() {
        let (cap, block, cell) = (65536, 9, 512);
        // 3578 = 6 * 512 + 506: six positions left in the cell, a block needs nine.
        assert!(start_admits(3578, 16384, cap, block, cell, true));
        assert!(!start_admits(3578, 16384, cap, block, cell, false));
        // Twelve left: the block fits either way.
        assert!(start_admits(3572, 16384, cap, block, cell, false));
        // The bridge needs the edge's steps AND a block after them inside the output budget.
        assert!(!start_admits(3578, 6 + block, cap, block, cell, true));
        assert!(start_admits(3578, 6 + block + 1, cap, block, cell, true));
        // No room for a block in the output budget or the cache.
        assert!(!start_admits(1000, block, cap, block, cell, true));
        assert!(!start_admits(cap - 4, 16384, cap, block, cell, true));
        // Metal's provider bridges every edge, so its admission does too.
        #[cfg(not(feature = "cuda-speculative"))]
        assert!(served_provider_bridges());
    }

    /// A restored request resumes the resident history only where that history's rows cover its
    /// prefix; any other restore is rebuilt from the restore point, never refused (design 5.5).
    #[cfg(not(feature = "cuda-speculative"))]
    #[test]
    fn a_restored_request_resumes_a_covering_history_and_rebuilds_otherwise() {
        let prompt: Vec<u32> = (0..700).collect();
        let held =
            |tokens: Vec<u32>, floor: usize| Some(CachedHistory { tokens, floor });
        let plan = |h, start| {
            let (tokens, floor, how) = history_for(h, &prompt, start);
            (tokens.len(), floor, how)
        };
        // Nothing held: a request from 0 starts fresh; a restored one rebuilds at its restore
        // point, carrying the prompt's tokens below it so a later prompt can be matched.
        assert_eq!(plan(None, 0), (0, 0, "fresh"));
        assert_eq!(plan(None, 576), (576, 576, "rebuilt"));
        // The same stream, longer than the restore point: resumed, cut back to it.
        assert_eq!(
            plan(held(prompt[..650].to_vec(), 0), 576),
            (576, 0, "resumed")
        );
        // Another stream (one token differs below the restore point): rebuilt.
        let mut other = prompt[..650].to_vec();
        other[300] += 1;
        assert_eq!(plan(held(other, 0), 576), (576, 576, "rebuilt"));
        // A held stream shorter than the restore point does not cover it: rebuilt.
        assert_eq!(
            plan(held(prompt[..500].to_vec(), 0), 576),
            (576, 576, "rebuilt")
        );
        // A rebuilt history keeps its floor when resumed again above it...
        assert_eq!(
            plan(held(prompt[..650].to_vec(), 512), 640),
            (640, 512, "resumed")
        );
        // ...and is rebuilt when a restore lands below it: its rows start above that point.
        assert_eq!(
            plan(held(prompt[..650].to_vec(), 640), 576),
            (576, 576, "rebuilt")
        );
    }

    struct Shape {
        hidden: u64,
        ffn: u64,
        heads: u64,
        kv_heads: u64,
        head_dim: u64,
        block: u64,
        layers: u64,
        rank: u64,
        vocab: u64,
        target_hidden: u64,
        taps: Vec<i64>,
    }

    fn lfm25() -> Shape {
        Shape {
            hidden: 2048,
            ffn: 6144,
            heads: 32,
            kv_heads: 8,
            head_dim: 64,
            block: 9,
            layers: 5,
            rank: 256,
            vocab: 128_000,
            target_hidden: 2048,
            taps: vec![3, 10, 18, 22, 28],
        }
    }

    /// Qwen3.8-27B's DimInfer drafter as the design's table gives it: attention width 4096
    /// under a hidden width of 5120, and an untied target head.
    fn qwen27b() -> Shape {
        Shape {
            hidden: 5120,
            ffn: 17408,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            block: 15,
            layers: 5,
            rank: 256,
            vocab: 248_320,
            target_hidden: 5120,
            taps: vec![6, 20, 34, 48, 62],
        }
    }

    fn tensors(s: &Shape) -> Vec<(String, u32, Vec<u64>)> {
        let (qw, kw) = (s.heads * s.head_dim, s.kv_heads * s.head_dim);
        let mut t: Vec<(String, u32, Vec<u64>)> = vec![
            (
                "fc.weight".into(),
                8,
                vec![s.taps.len() as u64 * s.target_hidden, s.hidden],
            ),
            ("enc.output_norm.weight".into(), 0, vec![s.hidden]),
            ("output_norm.weight".into(), 0, vec![s.hidden]),
            ("markov_w1.weight".into(), 8, vec![s.rank, s.vocab]),
            ("markov_w2.weight".into(), 8, vec![s.rank, s.vocab]),
            ("conf_proj.weight".into(), 1, vec![s.hidden + s.rank]),
            ("conf_proj.bias".into(), 0, vec![1]),
        ];
        for l in 0..s.layers {
            for (name, kind, dims) in [
                ("attn_norm", 0, vec![s.hidden]),
                ("attn_q", 8, vec![s.hidden, qw]),
                ("attn_k", 8, vec![s.hidden, kw]),
                ("attn_v", 8, vec![s.hidden, kw]),
                ("attn_output", 8, vec![qw, s.hidden]),
                ("attn_q_norm", 0, vec![s.head_dim]),
                ("attn_k_norm", 0, vec![s.head_dim]),
                ("ffn_norm", 0, vec![s.hidden]),
                ("ffn_gate", 8, vec![s.hidden, s.ffn]),
                ("ffn_up", 8, vec![s.hidden, s.ffn]),
                ("ffn_down", 8, vec![s.ffn, s.hidden]),
            ] {
                t.push((format!("blk.{l}.{name}.weight"), kind, dims));
            }
        }
        t
    }

    const ALIGN: u64 = 32;
    const DATA: u64 = 4096;

    fn document(s: &Shape, tensors: &[(String, u32, Vec<u64>)]) -> Document {
        let mut rel = 0;
        let mut infos = Vec::new();
        for (name, kind, dims) in tensors {
            let byte_size = tensor_bytes(*kind, dims).unwrap();
            infos.push(TensorInfo {
                name: name.clone(),
                dimensions: dims.clone(),
                ggml_type: *kind,
                relative_offset: rel,
                absolute_offset: DATA + rel,
                byte_size,
                type_field_offset: 0,
            });
            rel = (rel + byte_size).div_ceil(ALIGN) * ALIGN;
        }
        let mut metadata = BTreeMap::new();
        let mut put = |k: &str, v: MetadataValue| {
            metadata.insert(k.to_string(), v);
        };
        put(
            "general.architecture",
            MetadataValue::Scalar(Scalar::String("dflash".into())),
        );
        for (k, v) in [
            ("dflash.embedding_length", s.hidden),
            ("dflash.feed_forward_length", s.ffn),
            ("dflash.attention.head_count", s.heads),
            ("dflash.attention.head_count_kv", s.kv_heads),
            ("dflash.attention.key_length", s.head_dim),
            ("dflash.attention.value_length", s.head_dim),
            ("dflash.block_size", s.block),
            ("dflash.block_count", s.layers),
            ("dflash.context_length", 128_000),
        ] {
            put(k, MetadataValue::Scalar(Scalar::Unsigned(v)));
        }
        put(
            "dflash.rope.freq_base",
            MetadataValue::Scalar(Scalar::Float(1e7)),
        );
        put(
            "dflash.attention.layer_norm_rms_epsilon",
            MetadataValue::Scalar(Scalar::Float(1e-5)),
        );
        put(
            "dflash.target_layers",
            MetadataValue::Array(ArraySummary {
                element_type: ValueType::Int32,
                element_count: s.taps.len() as u64,
                preview: s.taps.iter().map(|&t| Scalar::Signed(t)).collect(),
            }),
        );
        Document {
            version: 3,
            alignment: ALIGN,
            data_offset: DATA,
            file_size: DATA + rel,
            metadata,
            tensors: infos,
        }
    }

    /// The target's tensor table (the embedding, and an untied Q6_K head when asked) and the
    /// offset the drafter's file is mapped at.
    fn target(s: &Shape, untied: bool) -> (BTreeMap<String, Tensor>, u64) {
        let mut table = BTreeMap::new();
        let mut end = DATA;
        let mut add = |name: &str, kind: u32| {
            let bytes = tensor_bytes(kind, &[s.target_hidden, s.vocab]).unwrap();
            table.insert(
                name.to_string(),
                Tensor {
                    offset: end as usize,
                    bytes: bytes as usize,
                    ggml_type: kind,
                    ne: [s.target_hidden, s.vocab, 1, 1],
                    n_dims: 2,
                },
            );
            end = (end + bytes).div_ceil(ALIGN) * ALIGN;
        };
        add("token_embd.weight", 8);
        if untied {
            add("output.weight", 14);
        }
        (table, end)
    }

    fn read(
        s: &Shape,
        doc: &Document,
        untied: bool,
    ) -> Result<DsparkDescriptor, String> {
        let (table, base) = target(s, untied);
        read_descriptor(doc, &table, base + doc.file_size, base, 64, 7)
    }

    #[test]
    fn lfm25_drafter_reads_with_a_tied_head() {
        let s = lfm25();
        let doc = document(&s, &tensors(&s));
        let d = read(&s, &doc, false).unwrap();
        assert_eq!(d.target_layers, vec![2, 9, 17, 21, 27]);
        assert_eq!(d.head, d.embedding);
        assert_eq!(
            (
                d.hidden,
                d.heads,
                d.kv_heads,
                d.head_dim,
                d.block_size,
                d.rank
            ),
            (2048, 32, 8, 64, 9, 256)
        );
        assert_eq!(d.fc.dims, vec![10240, 2048]);
        assert_eq!(d.layers.len(), 5);
        let (_, base) = target(&s, false);
        let q = doc.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(d.layers[0].q.offset, base + q.absolute_offset);
    }

    #[test]
    fn qwen27b_drafter_reads_its_own_attention_width_and_untied_head() {
        let s = qwen27b();
        let doc = document(&s, &tensors(&s));
        let d = read(&s, &doc, true).unwrap();
        assert_eq!(d.layers[0].q.dims, vec![5120, 4096]);
        assert_eq!(d.layers[0].k.dims, vec![5120, 1024]);
        assert_eq!(d.layers[0].o.dims, vec![4096, 5120]);
        assert_eq!(d.head.ggml_type, 14);
        assert_ne!(d.head, d.embedding);
        assert_eq!(d.target_layers, vec![5, 19, 33, 47, 61]);
    }

    #[test]
    fn a_tensor_the_adapter_does_not_read_is_refused_by_name() {
        let s = lfm25();
        let mut t = tensors(&s);
        t.push(("d2t".into(), 26, vec![128_000]));
        let e = read(&s, &document(&s, &t), false).unwrap_err();
        assert!(e.contains("d2t"), "{e}");
    }

    #[test]
    fn a_shape_off_the_metadata_is_refused_by_name() {
        let s = lfm25();
        let mut t = tensors(&s);
        let k = t.iter_mut().find(|x| x.0 == "blk.3.attn_k.weight").unwrap();
        k.2 = vec![2048, 1024];
        let e = read(&s, &document(&s, &t), false).unwrap_err();
        assert!(e.contains("blk.3.attn_k.weight"), "{e}");
    }

    #[test]
    fn metadata_that_changes_the_layout_is_refused_by_name() {
        let s = lfm25();
        let mut doc = document(&s, &tensors(&s));
        doc.metadata.insert(
            "dflash.attention.causal".into(),
            MetadataValue::Scalar(Scalar::Bool(false)),
        );
        doc.metadata.insert(
            "dflash.sample_from_anchor".into(),
            MetadataValue::Scalar(Scalar::String("true".into())),
        );
        assert!(read(&s, &doc, false).is_ok());
        doc.metadata.insert(
            "dflash.sample_from_anchor".into(),
            MetadataValue::Scalar(Scalar::Bool(false)),
        );
        let e = read(&s, &doc, false).unwrap_err();
        assert!(e.contains("dflash.sample_from_anchor"), "{e}");
        doc.metadata.remove("dflash.sample_from_anchor");
        doc.metadata.insert(
            "dflash.attention.sliding_window".into(),
            MetadataValue::Scalar(Scalar::Unsigned(512)),
        );
        let e = read(&s, &doc, false).unwrap_err();
        assert!(e.contains("dflash.attention.sliding_window"), "{e}");
    }

    #[test]
    fn appended_spans_cover_every_drafter_tensor_past_the_target() {
        let s = lfm25();
        let doc = document(&s, &tensors(&s));
        let d = read(&s, &doc, false).unwrap();
        let spans = d.appended_spans();
        let (_, base) = target(&s, false);
        assert_eq!(spans.len(), doc.tensors.len());
        assert_eq!(
            spans.iter().map(|s| s.1).sum::<u64>(),
            doc.tensors.iter().map(|t| t.byte_size).sum::<u64>()
        );
        assert!(spans.iter().all(|&(offset, _)| offset >= base));
    }

    #[test]
    fn target_layers_must_increase_and_fit_the_target() {
        let mut s = lfm25();
        s.taps = vec![3, 3, 18, 22, 28];
        assert!(read(&s, &document(&s, &tensors(&s)), false).is_err());
        s.taps = vec![3, 10, 18, 22, 65];
        assert!(read(&s, &document(&s, &tensors(&s)), false).is_err());
    }

    /// A block of `picks` with K = 3: each position's pick and two siblings `10 * pick` and
    /// `10 * pick + 1`, the pick's value largest.
    fn block_of(picks: &[u32]) -> crate::dspark_forward::DraftBlock {
        let mut candidate_ids = Vec::new();
        let mut candidate_values = Vec::new();
        for &p in picks {
            candidate_ids.extend([p, 10 * p, 10 * p + 1]);
            candidate_values.extend([2.0_f32, 1.0, 0.0]);
        }
        crate::dspark_forward::DraftBlock {
            ids: picks.to_vec(),
            candidates: 3,
            candidate_ids,
            candidate_values,
            confidence: vec![0.5; picks.len()],
        }
    }

    fn chains(
        picks: &[u32],
        committed: &[u32],
        anchor: u32,
        stops: &[u32],
    ) -> (Candidates, Vec<u32>) {
        chains_to(picks, committed, anchor, stops, picks.len())
    }

    /// `chains` with its own depth limit. Precision per evidence slot: 0.75 for the request's
    /// 3-token index, 0.9 / 0.95 for its copy matches, 0.5 for the table.
    fn chains_to(
        picks: &[u32],
        committed: &[u32],
        anchor: u32,
        stops: &[u32],
        depth_limit: usize,
    ) -> (Candidates, Vec<u32>) {
        let mut ix = crate::ngram::NgramIndex::default();
        ix.observe_all(committed);
        let mut found = level1_candidates(&block_of(picks), 100).unwrap();
        let lookup = NgramLookup {
            request: &ix,
            table: None,
        };
        let precision = [0.75, 0.75, 0.9, 0.95, 0.5, 0.5, 0.5, 0.5];
        let key0 = ngram_chains(
            &mut found,
            &lookup,
            anchor,
            &precision,
            stops,
            100,
            depth_limit,
            CHAIN_NODES,
        );
        (found, key0)
    }

    /// THE ANCHOR IS IN THE KEY. The index's tail ends one token before the anchor, because the
    /// anchor is committed after the tree; a key formed from the tail alone predicts the anchor's
    /// own position. On a stream repeating `10 11 12 13`, with `13` the anchor and the drafter
    /// right, every pick is the index's follower too: all agreed, no new node, and each agreed
    /// pick counts the n-gram tokens above it.
    #[test]
    fn the_key_ends_in_the_anchor_and_a_right_drafter_is_agreed_with() {
        use crate::speculative::NodeSource;
        let (found, key0) =
            chains(&[10, 11, 12, 13], &[10, 11, 12, 13, 10, 11, 12], 13, &[]);
        assert_eq!(key0, vec![11, 12, 13]);
        assert_eq!(
            found.candidates.len(),
            12,
            "a follower equal to a pick added a node"
        );
        for (pos, c) in [0_usize, 3, 6, 9].into_iter().enumerate() {
            assert_eq!(found.sources[c], NodeSource::Agreed, "pick {pos}");
            let terms = found.features[c]
                .unwrap()
                .ngram
                .expect("an agreed pick carries its bucket");
            assert_eq!(terms.run, u32::try_from(pos).unwrap(), "pick {pos}");
            assert_eq!(terms.source, crate::accept_model::Source::Request);
        }
        assert!(
            found
                .sources
                .iter()
                .enumerate()
                .all(|(i, s)| i % 3 == 0 || *s == NodeSource::Drafter),
            "a sibling was marked agreed"
        );
    }

    /// A follower the drafter did not offer starts a chain of new nodes below its branch point,
    /// each at its evidence slot's precision, and the chain stops at its depth limit. The first
    /// node comes from the 3-token index (only "1 2 3" is shared); the second is already a copy
    /// match, because "1 2 3 500" has been written before.
    #[test]
    fn a_follower_the_drafter_did_not_offer_starts_a_chain_to_the_drafter_s_depth() {
        use crate::accept_model::Kind;
        use crate::speculative::NodeSource;
        let (found, _) = chains(&[7, 8], &[1, 2, 3, 500, 501, 502, 503, 1, 2], 3, &[]);
        assert_eq!(
            found.candidates.len(),
            8,
            "two new nodes, then the depth stops the chain"
        );
        let (a, b) = (&found.candidates[6], &found.candidates[7]);
        assert_eq!((a.token, a.parent, a.q), (500, -1, 0.75));
        assert_eq!((b.token, b.parent, b.q), (501, 6, 0.9));
        assert_eq!(found.sources[6..], [NodeSource::Ngram, NodeSource::Ngram]);
        let (fa, fb) = (found.features[6].unwrap(), found.features[7].unwrap());
        assert_eq!(
            (fa.kind, fa.depth, fa.ngram.unwrap().run),
            (Kind::Ngram, 0, 0)
        );
        assert_eq!(
            (fb.kind, fb.depth, fb.ngram.unwrap().run),
            (Kind::Ngram, 1, 1)
        );
        assert!(
            (fa.base - 3.0_f64.ln()).abs() < 1e-12,
            "base is logit(0.75)"
        );
    }

    /// A follower equal to a SIBLING agrees with it and the chain carries on below the sibling;
    /// a follower equal to a stop token takes its node and ends the chain.
    #[test]
    fn an_agreed_sibling_carries_the_chain_and_a_stop_token_ends_it() {
        use crate::speculative::NodeSource;
        // After "2 3 | anchor 4": 70 (the sibling of pick 7), then 9, then the stop token 99.
        let (found, _) = chains(&[7, 8, 5], &[2, 3, 4, 70, 9, 99, 2, 3], 4, &[99]);
        assert_eq!(
            found.sources[1],
            NodeSource::Agreed,
            "sibling 70 at position 0"
        );
        let added: Vec<(u32, i32)> = found.candidates[9..]
            .iter()
            .map(|c| (c.token, c.parent))
            .collect();
        assert_eq!(
            added,
            vec![(9, 1), (99, 9)],
            "9 below the sibling, then the stop token"
        );
    }

    /// The stored table answers only where the request's index holds nothing for the context.
    #[test]
    fn the_stored_table_is_each_lookup_s_fallback() {
        use crate::accept_model::Source;
        let mut request = crate::ngram::NgramIndex::default();
        request.observe_all(&[1, 2, 3, 40]);
        let mut table = crate::ngram::NgramIndex::with_limit(8);
        table.count(&[1, 2, 3], 41);
        table.count(&[5, 6, 7], 42);
        let lookup = NgramLookup {
            request: &request,
            table: Some(&table),
        };
        let (f, s) = lookup.follows(&[1, 2, 3]).unwrap();
        assert_eq!((f[0].0, s), (40, Source::Request));
        let (f, s) = lookup.follows(&[5, 6, 7]).unwrap();
        assert_eq!((f[0].0, s), (42, Source::Table));
        assert!(lookup.follows(&[8, 8, 8]).is_none());
    }

    /// A COPY GOES PAST THE DRAFTER'S DEPTH. The committed text holds a passage once; the round's
    /// context ends in its first four tokens, the drafter offers something else, and the chain reads
    /// the rest of the passage on, one node per token, to the depth limit -- each node carrying the
    /// copy's length, which grows as it reads on.
    #[test]
    fn a_copy_chain_reads_the_passage_on_past_the_drafter_s_depth() {
        use crate::accept_model::Kind;
        use crate::speculative::NodeSource;
        let passage: Vec<u32> = (100..130).collect();
        let mut committed = vec![1, 2, 3];
        committed.extend(&passage);
        committed.extend([7, 8]);
        committed.extend(&passage[..3]);
        // The anchor is the passage's fourth token: the context ends in 100 101 102 103.
        let anchor = passage[3];
        let (found, _) = chains_to(&[5, 6], &committed, anchor, &[], 12);
        let added: Vec<u32> = found.candidates[6..].iter().map(|c| c.token).collect();
        assert_eq!(
            added,
            (104..116).collect::<Vec<u32>>(),
            "twelve copied tokens"
        );
        assert!(found.sources[6..].iter().all(|s| *s == NodeSource::Ngram));
        let first = found.features[6].unwrap();
        let last = found.features[found.features.len() - 1].unwrap();
        assert_eq!(first.kind, Kind::Ngram);
        let (m0, m11) = (first.ngram.unwrap().matched, last.ngram.unwrap().matched);
        assert!(
            m0 >= 4 && m11 == m0 + 11,
            "the copy's length grows: {m0} -> {m11}"
        );
        assert!(
            (found.candidates[6].q - 0.9).abs() < 1e-12,
            "a copy of 4-15 tokens reads its own slot"
        );
        // The copy proposed 104 at the anchor, where the drafter offered 5, 50 and 51: all three are
        // contradicted by it; position 1's candidates hang from pick 5, where the n-gram said nothing.
        for c in 0..3 {
            let contra = found.features[c]
                .unwrap()
                .contra
                .expect("contradicted at the anchor");
            assert_eq!(contra.matched, m0);
            assert!(found.features[c].unwrap().base < 0.0);
        }
        assert!(found.features[3..6].iter().all(|f| f.unwrap().contra.is_none()));
        // Each node hangs from the one before it.
        for (i, c) in found.candidates[7..].iter().enumerate() {
            assert_eq!(c.parent, i32::try_from(6 + i).unwrap());
        }
    }

    #[test]
    fn level1_candidates_hang_siblings_from_the_chain_node_before_them() {
        let block = crate::dspark_forward::DraftBlock {
            ids: vec![7, 8],
            candidates: 3,
            candidate_ids: vec![7, 70, 71, 8, 80, 81],
            candidate_values: vec![2.0, 1.0, 0.0, 5.0, 5.0, 3.0],
            confidence: vec![0.75, 0.5],
        };
        let found = level1_candidates(&block, 0).unwrap();
        // The model's inputs line up with the candidates, index for index.
        let kinds: Vec<_> = found.features.iter().map(|f| f.unwrap().kind).collect();
        {
            use crate::accept_model::Kind::{Pick, Sibling};
            assert_eq!(kinds, vec![Pick, Sibling, Sibling, Pick, Sibling, Sibling]);
        }
        let c = found.candidates;
        let tokens: Vec<u32> = c.iter().map(|x| x.token).collect();
        let parents: Vec<i32> = c.iter().map(|x| x.parent).collect();
        assert_eq!(tokens, vec![7, 70, 71, 8, 80, 81]);
        assert_eq!(parents, vec![-1, -1, -1, 0, 0, 0]);
        assert_eq!((c[0].q, c[3].q), (0.75, 0.5));
        let z = 1.0 + (-1.0_f64).exp() + (-2.0_f64).exp();
        assert!((c[1].q - (-1.0_f64).exp() / z).abs() < 1e-12);
        assert!((c[2].q - (-2.0_f64).exp() / z).abs() < 1e-12);
        // A sibling whose biased value equals the pick's shares its probability.
        let z2 = 2.0 + (-2.0_f64).exp();
        assert!((c[4].q - 1.0 / z2).abs() < 1e-12);
        assert!((c[5].q - (-2.0_f64).exp() / z2).abs() < 1e-12);
    }

    fn native_candidates(edges: &[(u32, i32)]) -> Candidates {
        Candidates {
            candidates: edges
                .iter()
                .map(|&(token, parent)| crate::speculative::TreeCandidate {
                    token,
                    parent,
                    q: 0.5,
                })
                .collect(),
            features: vec![None; edges.len()],
            sources: vec![crate::speculative::NodeSource::Drafter; edges.len()],
            ngram_branch_prefix: Some(edges.len()),
        }
    }

    #[test]
    fn native_ngram_context_follows_each_real_parent_path() {
        use crate::speculative::NodeSource;
        let mut ix = crate::ngram::NgramIndex::default();
        ix.observe_all(&[1, 2]);
        // Two real branches: anchor 3 -> 10 -> 11 and anchor 3 -> 20 -> 21.
        ix.count(&[2, 3, 10], 11);
        ix.count(&[2, 3, 20], 21);
        ix.count(&[3, 10, 11], 100);
        ix.count(&[3, 20, 21], 200);
        // These fabricated concatenations of two beams must never become contexts.
        ix.count(&[10, 20, 11], 999);
        ix.count(&[20, 11, 21], 998);
        let lookup = NgramLookup { request: &ix, table: None };
        let mut found = native_candidates(&[(10, -1), (20, -1), (11, 0), (21, 1)]);
        ngram_chains(&mut found, &lookup, 3, &[0.75; 8], &[], 100, 8, 15);
        let added: Vec<_> = found.candidates[4..]
            .iter().map(|c| (c.token, c.parent)).collect();
        assert_eq!(added, [(100, 2), (200, 3)]);
        assert!(found.features[..4].iter().all(Option::is_none));
        assert_eq!(&found.sources[2..4], &[NodeSource::Agreed; 2]);
        assert!(found.features[4..].iter().all(|f| f.unwrap().depth == 2));
        assert_eq!(ix.tail(), &[1, 2], "speculative branches must not feed the index");
        assert!(ix.follows(&[10, 11, 100]).is_empty());
    }

    #[test]
    fn native_ngram_nodes_are_deduplicated_and_keep_their_source() {
        use crate::speculative::NodeSource;
        let mut ix = crate::ngram::NgramIndex::default();
        ix.observe_all(&[1, 2, 3, 40, 41, 42, 1, 2]);
        let lookup = NgramLookup { request: &ix, table: None };
        let mut found = native_candidates(&[(9, -1)]);
        ngram_chains(&mut found, &lookup, 3, &[0.75; 8], &[], 100, 3, 15);
        let once: Vec<_> = found.candidates.iter().map(|c| (c.token, c.parent, c.q)).collect();
        assert_eq!(once.len(), 4);
        // All three n-gram nodes now exist, including below n-gram parents. Following
        // the same copy must reuse those edges rather than adding a duplicate chain.
        ngram_chains(&mut found, &lookup, 3, &[0.75; 8], &[], 100, 3, 15);
        let twice: Vec<_> = found.candidates.iter().map(|c| (c.token, c.parent, c.q)).collect();
        assert_eq!(once, twice);
        assert!(found.sources[1..].iter().all(|s| *s == NodeSource::Ngram));
    }

    #[test]
    fn native_ngram_depth_and_new_node_budgets_are_independent() {
        let passage: Vec<u32> = (100..130).collect();
        let mut committed = passage.clone();
        committed.extend([7, 8]);
        committed.extend(&passage[..3]);
        let mut ix = crate::ngram::NgramIndex::default();
        ix.observe_all(&committed);
        let lookup = NgramLookup { request: &ix, table: None };
        for (limit, expected) in [(15, 8), (3, 3)] {
            let mut found = native_candidates(&[(9, -1)]);
            ngram_chains(&mut found, &lookup, 103, &[0.75; 8], &[], 100, 8, limit);
            assert_eq!(found.candidates.len(), 1 + expected);
            for f in &found.features[1..] {
                assert!(f.unwrap().depth < 8);
            }
        }
    }

    #[test]
    fn no_ngram_match_preserves_native_candidates_and_does_not_price_missing_features() {
        let mut ix = crate::ngram::NgramIndex::default();
        ix.observe_all(&[1, 2]);
        let lookup = NgramLookup { request: &ix, table: None };
        let mut found = native_candidates(&[(10, -1), (20, -1), (11, 0), (21, 1)]);
        let before: Vec<_> = found.candidates.iter().map(|c| (c.token, c.parent, c.q)).collect();
        ngram_chains(&mut found, &lookup, 3, &[0.75; 8], &[], 100, 8, 15);
        assert_eq!(before, found.candidates.iter().map(|c| (c.token, c.parent, c.q)).collect::<Vec<_>>());
        assert!(found.features.iter().all(Option::is_none));
        assert!(found.sources.iter().all(|s| *s == crate::speculative::NodeSource::Drafter));
        let model = crate::accept_model::AcceptModel::derived(0.5);
        let priced = priced_candidates(&model, &found);
        assert_eq!(before, priced.iter().map(|c| (c.token, c.parent, c.q)).collect::<Vec<_>>());
        assert_eq!(ix.tail(), &[1, 2]);
    }

    #[test]
    fn level1_candidates_refuse_a_pick_that_is_not_first_or_a_single_candidate() {
        let swapped = crate::dspark_forward::DraftBlock {
            ids: vec![7],
            candidates: 2,
            candidate_ids: vec![70, 7],
            candidate_values: vec![1.0, 1.0],
            confidence: vec![0.5],
        };
        assert!(level1_candidates(&swapped, 0).is_err());
        let single = crate::dspark_forward::DraftBlock {
            ids: vec![7],
            candidates: 1,
            candidate_ids: vec![7],
            candidate_values: vec![1.0],
            confidence: vec![0.5],
        };
        assert!(level1_candidates(&single, 0).is_err());
    }
}
