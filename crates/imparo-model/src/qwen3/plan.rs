//! Qwen3 plan builder: GGUF metadata -> ModelPlan.
//!
//! Qwen3 is the plainest tower this engine serves: every block attends, every block runs
//! one SwiGLU FFN, nothing is shared and nothing recurs. What is NOT plain, and is read
//! from the file rather than assumed:
//!
//!   - The head dim is `attention.key_length`, NOT `embedding_length / head_count`. On
//!     Qwen3-8B the two agree (4096 / 32 = 128); the derived form is a coincidence of
//!     that file's widths, and a sibling whose embedding is narrower than q_heads x
//!     head_dim would be planned wrongly by it and still run.
//!   - Q AND K CARRY A PER-HEAD RMS NORM, `attn_q_norm` / `attn_k_norm`, one vector of
//!     `head_dim` shared by every head of that block. A forward that skips them produces
//!     plausible text and wrong logits, so their presence is REQUIRED here rather than
//!     inferred at dispatch.
//!   - Whether the output head is tied is the file's answer, read from the presence of
//!     `output.weight`. Qwen3-8B carries it. A file without it reuses the embedding
//!     table, and the plan says so rather than choosing by model size.
//!   - `rope.freq_base` is 1e6, not the 10k default, and the rotation covers the whole
//!     head unless the file says otherwise.
//!
//! No `qwen3.expert_count` key exists in these files; an MoE Qwen3 would declare a
//! different architecture string, so this builder refuses one rather than planning its
//! FFN as dense.

use std::path::Path;

use crate::{
    Activation, Attention, EmbedPlan, Ffn, KvSource, LayerPlan, ModelConfig, ModelPlan,
    OutputPlan, PlanError, f32_opt, u32_at, u32_or,
};

/// Builds a Qwen3 plan from GGUF metadata.
///
/// # Errors
///
/// Returns [`PlanError`] when a required key is absent, when the file declares experts
/// (which this workflow does not run), when a block is missing the attention or FFN
/// tensors the forward dispatches, or when the per-head Q/K norms are absent or are not
/// `key_length` wide.
pub fn build(
    document: &imparo_gguf::Document,
    path: &Path,
) -> Result<ModelPlan, PlanError> {
    let _ = path;
    let n_layers = u32_at(document, "qwen3.block_count")?;
    let n_embd = u32_at(document, "qwen3.embedding_length")?;
    let n_ff = u32_at(document, "qwen3.feed_forward_length")?;
    let n_heads = u32_at(document, "qwen3.attention.head_count")?;
    let n_kv_heads = u32_at(document, "qwen3.attention.head_count_kv")?;
    let context_length = u32_at(document, "qwen3.context_length")?;
    let norm_eps =
        f32_opt(document, "qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
    let rope_base = f32_opt(document, "qwen3.rope.freq_base").unwrap_or(10_000.0);

    if n_layers == 0 {
        return Err(PlanError::Inconsistent("qwen3.block_count is 0".into()));
    }
    if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(PlanError::Inconsistent(format!(
            "qwen3 head counts {n_heads} q / {n_kv_heads} kv do not group evenly"
        )));
    }
    if u32_or(document, "qwen3.expert_count", 0) != 0 {
        return Err(PlanError::Inconsistent(
            "qwen3.expert_count is set; this workflow runs one dense FFN per block"
                .into(),
        ));
    }

    // THE HEAD DIM IS DECLARED, NOT DERIVED. key_length and value_length are separate
    // keys and a file may disagree with itself; the attention kernel takes one width for
    // both, so a mismatch is refused rather than resolved by picking one.
    let head_dim = u32_at(document, "qwen3.attention.key_length")?;
    let value_dim = u32_or(document, "qwen3.attention.value_length", head_dim);
    if head_dim == 0 || value_dim != head_dim {
        return Err(PlanError::Inconsistent(format!(
            "qwen3 key_length {head_dim} and value_length {value_dim} must be equal and \
             non-zero"
        )));
    }
    let rope_dim = u32_or(document, "qwen3.rope.dimension_count", head_dim);
    if rope_dim == 0 || rope_dim > head_dim || rope_dim % 2 != 0 {
        return Err(PlanError::Inconsistent(format!(
            "qwen3.rope.dimension_count {rope_dim} must be even and at most the head dim \
             {head_dim}"
        )));
    }

    let mut layers = Vec::with_capacity(n_layers as usize);
    for index in 0..n_layers {
        // Every tensor the forward dispatches is checked here, once, against the file's
        // own table. A missing norm is the dangerous case: attention would still run and
        // still produce text.
        for suffix in [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            if tensor(document, index, suffix).is_none() {
                return Err(PlanError::Inconsistent(format!(
                    "blk.{index}.{suffix} is absent; the qwen3 forward dispatches it on \
                     every block"
                )));
            }
        }
        for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
            let norm = tensor(document, index, suffix).ok_or_else(|| {
                PlanError::Inconsistent(format!(
                    "blk.{index}.{suffix} is absent; Qwen3 normalises every head of Q and K \
                     before rope"
                ))
            })?;
            if norm.dimensions.as_slice() != [u64::from(head_dim)] {
                return Err(PlanError::Inconsistent(format!(
                    "blk.{index}.{suffix} is {:?}, not one vector of the head dim \
                     {head_dim}",
                    norm.dimensions
                )));
            }
        }
        layers.push(LayerPlan {
            index,
            attention: Attention::Full {
                head_dim,
                rope_base,
                rope_dim,
            },
            ffn: Ffn::Dense {
                activation: Activation::Silu,
                hidden: n_ff,
            },
            // Nothing is shared: every block owns its K and V.
            kv_source: KvSource::Own,
        });
    }

    let vocab_size = vocab_from_tensors(document)
        .or_else(|| u32_at(document, "qwen3.vocab_size").ok())
        .unwrap_or(0);

    // TIED OR NOT IS THE FILE'S ANSWER: Qwen3-8B ships output.weight, and a file that
    // does not reuses the embedding table as its head.
    let tied_embeddings = match document.tensor("output.weight") {
        None => true,
        Some(head) => {
            let embd = document.tensor("token_embd.weight").ok_or_else(|| {
                PlanError::Inconsistent(
                    "qwen3 file carries no token_embd.weight".into(),
                )
            })?;
            if head.dimensions != embd.dimensions {
                return Err(PlanError::Inconsistent(format!(
                    "qwen3 output.weight {:?} and token_embd.weight {:?} differ in shape",
                    head.dimensions, embd.dimensions
                )));
            }
            false
        }
    };

    Ok(ModelPlan {
        config: ModelConfig {
            architecture: "qwen3".into(),
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
            // No embedding scale key, and no norm between the table and blk.0's attn_norm.
            scale_by_sqrt_embd: false,
            per_layer_dim: None,
            per_layer_row_bytes: None,
        },
        kv_storage_basis: crate::KvStorageBasisPolicy::BackendOverrideOrCanonical,
        // ROW-GATHERED ONLY WHEN IT IS NOT ALSO THE HEAD. A tied file multiplies the
        // whole table at the tail, and the fit may stage a row-gathered tensor by its
        // requested rows alone -- so declaring it here would let the tail read pages the
        // fit never brought in.
        weight_residency: if tied_embeddings {
            crate::WeightResidencyPlan::default()
        } else {
            crate::WeightResidencyPlan {
                row_gathered: &["token_embd.weight"],
            }
        },
        layers,
        decode_interleave: false,
        mega_decode: false,
        drafter: None,
        output: OutputPlan {
            final_norm: true,
            logit_softcap: None,
            tied_embeddings,
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

fn vocab_from_tensors(document: &imparo_gguf::Document) -> Option<u32> {
    document
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .and_then(|t| t.dimensions.last().copied())
        .and_then(|v| u32::try_from(v).ok())
}
