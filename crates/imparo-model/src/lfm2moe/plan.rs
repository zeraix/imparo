//! LFM2-MoE plan builder: GGUF metadata -> ModelPlan.
//!
//! The same tower as LFM2 -- short-convolution blocks interleaved with grouped-query
//! attention, one SwiGLU feed-forward per block, the output norm spelled
//! `token_embd_norm` -- with ONE difference, and it is per layer: from
//! `leading_dense_block_count` onward the feed-forward is routed.
//!
//! EVERY FACT HERE WAS READ from the file's metadata and from the reference builder
//! (`llama.cpp/src/models/lfm2moe.cpp`, whose graph is `lfm2.cpp`'s). The four that would
//! have been easy to get wrong:
//!
//!   - The dense/MoE split is `il >= leading_dense_block_count`, and the key is OPTIONAL:
//!     a file without it routes every block.
//!   - A block is recurrent exactly when its per-layer `head_count_kv` entry is ZERO --
//!     the same array rule LFM2 uses, and orthogonal to the dense/MoE split. A layer can
//!     be recurrent AND routed.
//!   - THERE IS NO SHARED EXPERT. `expert_shared_feed_forward_length` exists as a key and
//!     the reference's lfm2moe never reads it, so planning one would add a projection the
//!     file has no weights for.
//!   - `expert_gating_func` decides what the router's scores mean, and the bias
//!     `exp_probs_b` is added BEFORE the top-k, so it changes which experts run. Neither
//!     is a detail the forward can default.
//!
//! This is its own directory rather than an arm inside `lfm2/`: the dense model is what
//! the shipping speculation work measures, and it does not change to make room for this.

use std::path::Path;

use crate::ggufscan;

use crate::{
    Activation, Attention, EmbedPlan, ExpertGating, Ffn, KvSource, LayerPlan,
    ModelConfig, ModelPlan, OutputPlan, PlanError, f32_opt, u32_at, u32_or,
};

/// Builds an LFM2-MoE plan from GGUF metadata.
///
/// # Errors
///
/// Returns [`PlanError`] when a required key is absent or malformed, when the per-layer
/// `head_count_kv` array disagrees with the block count, when the router's shape does not
/// match the expert count, or when a routed layer is missing the tensors its forward
/// dispatches.
pub fn build(
    document: &imparo_gguf::Document,
    path: &Path,
) -> Result<ModelPlan, PlanError> {
    let n_layers = u32_at(document, "lfm2moe.block_count")?;
    let n_embd = u32_at(document, "lfm2moe.embedding_length")?;
    let n_ff = u32_at(document, "lfm2moe.feed_forward_length")?;
    let n_heads = u32_at(document, "lfm2moe.attention.head_count")?;
    let context_length = u32_at(document, "lfm2moe.context_length")?;
    let norm_eps =
        f32_opt(document, "lfm2moe.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
    let rope_base = f32_opt(document, "lfm2moe.rope.freq_base").unwrap_or(10_000.0);

    let kernel = u32_at(document, "lfm2moe.shortconv.l_cache")?;
    if kernel < 2 {
        return Err(PlanError::Inconsistent(format!(
            "lfm2moe.shortconv.l_cache is {kernel}; a short conv needs at least 2 taps to \
             carry state"
        )));
    }

    if n_heads == 0 || n_embd % n_heads != 0 {
        return Err(PlanError::Inconsistent(format!(
            "lfm2moe.embedding_length {n_embd} is not a multiple of head_count {n_heads}"
        )));
    }
    let head_dim = n_embd / n_heads;
    let rope_dim = u32_or(document, "lfm2moe.rope.dimension_count", head_dim);

    let kv_per_layer = head_count_kv_per_layer(path, n_layers)?;
    let attention_kv: Vec<u32> =
        kv_per_layer.iter().copied().filter(|&n| n != 0).collect();
    let n_kv_heads = attention_kv.first().copied().unwrap_or(0);
    if attention_kv.iter().any(|&n| n != n_kv_heads) {
        return Err(PlanError::Inconsistent(format!(
            "lfm2moe.attention.head_count_kv mixes GQA ratios across layers: {attention_kv:?}"
        )));
    }
    let window = u32_or(document, "lfm2moe.attention.sliding_window", 0);

    // THE ROUTING FACTS. Every one is read; the forward cannot default any of them.
    let experts = u32_at(document, "lfm2moe.expert_count")?;
    let experts_used = u32_at(document, "lfm2moe.expert_used_count")?;
    let expert_hidden = u32_at(document, "lfm2moe.expert_feed_forward_length")?;
    if experts == 0 || experts_used == 0 || experts_used > experts {
        return Err(PlanError::Inconsistent(format!(
            "lfm2moe routes {experts_used} of {experts} experts, which is not a routing"
        )));
    }
    let gating = match u32_at(document, "lfm2moe.expert_gating_func")? {
        1 => ExpertGating::Softmax,
        2 => ExpertGating::Sigmoid,
        // Not folded into the nearer of the two: the reference's enum has five values and
        // the three this engine has not implemented must announce themselves.
        other => {
            return Err(PlanError::Inconsistent(format!(
                "lfm2moe.expert_gating_func is {other}; this workflow runs softmax (1) \
                 and sigmoid (2)"
            )));
        }
    };
    // RENORMALISATION IS THE ARCHITECTURE'S, NOT THE FILE'S. The reference's lfm2moe
    // graph passes a literal `true` for norm_w (`src/models/lfm2.cpp`, build_moe_ffn),
    // and this file carries no `expert_weights_norm` key at all -- so reading one and
    // defaulting it to false, which is what this did first, left the picked weights
    // unnormalised and the logits wrong: 2 of 10 top ids shared with llama.cpp.
    let normalise_weights = true;
    // THE PLAN CARRIES THE MULTIPLIER, not the file's raw value. The reference stores 0.0
    // for an absent key and then skips its scale step for a value that is neither 0 nor 1,
    // so BOTH of those mean "multiply by one" -- and a forward that took the raw 0.0 and
    // multiplied by it zeroed every routing weight, which is what the device path did
    // before this line: the routed blocks contributed nothing and the tower's output came
    // through untouched.
    let weights_scale = match f32_opt(document, "lfm2moe.expert_weights_scale") {
        Some(v) if v != 0.0 => v,
        _ => 1.0,
    };
    // A file with no key routes EVERY block; the reference reads the key as optional and
    // defaults it to zero.
    let dense_lead = u32_or(document, "lfm2moe.leading_dense_block_count", 0);

    let mut layers = Vec::with_capacity(n_layers as usize);
    for index in 0..n_layers {
        let recurrent = kv_per_layer[index as usize] == 0;
        let attention = if recurrent {
            Attention::Recurrent {
                r_elems: n_embd * (kernel - 1),
                s_elems: 0,
                key_dim: 0,
                value_dim: 0,
            }
        } else if window > 0 {
            Attention::Window {
                head_dim,
                rope_base,
                rope_dim,
                window,
            }
        } else {
            Attention::Full {
                head_dim,
                rope_base,
                rope_dim,
            }
        };
        // THE SPLIT IS PER LAYER, and the tensors decide rather than the arithmetic: a
        // file whose split is phased differently from its key would otherwise be planned
        // wrongly and still run.
        let routed_by_key = index >= dense_lead;
        let routed_by_tensors =
            tensor(document, index, "ffn_gate_inp.weight").is_some();
        if routed_by_key != routed_by_tensors {
            return Err(PlanError::Inconsistent(format!(
                "blk.{index} {} a router, but lfm2moe.leading_dense_block_count \
                 {dense_lead} says it is {}",
                if routed_by_tensors {
                    "carries"
                } else {
                    "carries no"
                },
                if routed_by_key { "routed" } else { "dense" }
            )));
        }
        let ffn = if routed_by_tensors {
            for suffix in [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ] {
                let t = tensor(document, index, suffix).ok_or_else(|| {
                    PlanError::Inconsistent(format!(
                        "blk.{index}.{suffix} is absent; a routed block dispatches it"
                    ))
                })?;
                // The expert count is the LAST dimension of each stacked tensor. A file
                // that disagreed with its own key would index past the weights.
                if t.dimensions.last().copied() != Some(u64::from(experts)) {
                    return Err(PlanError::Inconsistent(format!(
                        "blk.{index}.{suffix} is {:?}, whose last dimension is not the \
                         expert count {experts}",
                        t.dimensions
                    )));
                }
            }
            let router =
                tensor(document, index, "ffn_gate_inp.weight").ok_or_else(|| {
                    PlanError::Inconsistent(format!(
                        "blk.{index}.ffn_gate_inp.weight absent"
                    ))
                })?;
            if router.dimensions.as_slice() != [u64::from(n_embd), u64::from(experts)] {
                return Err(PlanError::Inconsistent(format!(
                    "blk.{index}.ffn_gate_inp.weight is {:?}, not [{n_embd}, {experts}]",
                    router.dimensions
                )));
            }
            Ffn::Moe {
                activation: Activation::Silu,
                expert_hidden,
                experts,
                experts_used,
                // Read, not assumed: the reference's lfm2moe never reads
                // expert_shared_feed_forward_length, and no shared-expert tensor is in
                // the file.
                shared_hidden: 0,
                gating,
                normalise_weights,
                weights_scale,
                expert_bias: tensor(document, index, "exp_probs_b.bias").is_some(),
            }
        } else {
            Ffn::Dense {
                activation: Activation::Silu,
                hidden: n_ff,
            }
        };
        layers.push(LayerPlan {
            index,
            attention,
            ffn,
            kv_source: KvSource::Own,
        });
    }

    let vocab_size = vocab_from_tensors(document)
        .or_else(|| u32_at(document, "lfm2moe.vocab_size").ok())
        .unwrap_or(0);

    Ok(ModelPlan {
        config: ModelConfig {
            architecture: "lfm2moe".into(),
            n_layers,
            n_embd,
            n_ff,
            n_heads,
            n_kv_heads,
            context_length,
            vocab_size,
            norm_eps,
        },
        embed: EmbedPlan {
            scale_by_sqrt_embd: false,
            per_layer_dim: None,
            per_layer_row_bytes: None,
        },
        kv_storage_basis: crate::KvStorageBasisPolicy::BackendOverrideOrCanonical,
        weight_residency: crate::WeightResidencyPlan::default(),
        layers,
        // BOTH DECODE ROUTES, each an A/B away from its measurement:
        //
        //   decode_interleave   the host encodes step t+1 before step t retires, which hides
        //                       the round trip between tokens (IMPARO_DECODE_PIPE=0 is the
        //                       other arm). It was off only because nobody had measured it here.
        //   mega_decode         a routed layer's feed-forward runs as ONE persistent dispatch
        //                       (IMPARO_MEGA_FFN=0 is the other arm). A mega region can time out
        //                       and be re-run on the dispatch path from the plane it read, which
        //                       is what the extra recurrent plane this flag buys is for.
        //
        // The mega entry is the ROUTED TAIL -- the chain of seven dependent dispatches the
        // barrier probe priced (docs/moe-decode-attribution.md) -- not a fold of the small ops
        // around grid-hungry GEMVs: the experts' rows run inside the program at its grid, and
        // whether that pays is the measurement, not an argument.
        decode_interleave: true,
        mega_decode: true,
        drafter: None,
        output: OutputPlan {
            final_norm: true,
            logit_softcap: None,
            tied_embeddings: document.tensor("output.weight").is_none(),
        },
    })
}

fn tensor<'a>(
    document: &'a imparo_gguf::Document,
    layer: u32,
    suffix: &str,
) -> Option<&'a imparo_gguf::TensorInfo> {
    let name = format!("blk.{layer}.{suffix}");
    document.tensors.iter().find(|t| t.name == name)
}

/// The per-layer KV head counts. Read through the scanner because this key is an ARRAY.
fn head_count_kv_per_layer(path: &Path, n_layers: u32) -> Result<Vec<u32>, PlanError> {
    let key = "lfm2moe.attention.head_count_kv";
    let scan = ggufscan::Metadata::read(path)
        .map_err(|_| PlanError::MissingMetadata(key.into()))?;
    let ints = scan
        .ints(key)
        .ok_or_else(|| PlanError::MissingMetadata(key.into()))?;
    if ints.len() != n_layers as usize {
        return Err(PlanError::Inconsistent(format!(
            "{key} has {} entries but block_count is {n_layers}",
            ints.len()
        )));
    }
    Ok(ints
        .iter()
        .map(|v| u32::try_from(*v).unwrap_or(0))
        .collect())
}

fn vocab_from_tensors(document: &imparo_gguf::Document) -> Option<u32> {
    document
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .and_then(|t| t.dimensions.last().copied())
        .and_then(|v| u32::try_from(v).ok())
}

/// The rolling state of each recurrent block, the same shape LFM2's carries.
#[must_use]
pub fn conv_windows(plan: &crate::ModelPlan) -> Vec<crate::ConvWindow> {
    crate::lfm2::plan::conv_windows(plan)
}
