//! Gemma4 (including E4B) plan builder.
//!
//! Everything gemma4-specific lives here. The traits it exercises that other architectures
//! do not: two attention geometries in one network with different head dimensions AND
//! different rope bases, a KV-sharing suffix, a per-layer input embedding, and output
//! logit softcapping.

use std::path::Path;

use crate::ggufscan;

use crate::{
    Activation, Attention, EmbedPlan, Ffn, KvSource, LayerPlan, ModelConfig, ModelPlan,
    OutputPlan, PlanError, f32_opt, u32_at, u32_or,
};

/// Builds a gemma4 plan from GGUF metadata.
///
/// # Errors
///
/// Returns [`PlanError`] when a required key is absent, malformed, or when the layer
/// pattern length disagrees with the block count.
pub fn build(
    document: &imparo_gguf::Document,
    path: &Path,
) -> Result<ModelPlan, PlanError> {
    let n_layers = u32_at(document, "gemma4.block_count")?;
    let n_embd = u32_at(document, "gemma4.embedding_length")?;
    let n_ff = u32_at(document, "gemma4.feed_forward_length")?;
    let n_heads = u32_at(document, "gemma4.attention.head_count")?;
    let n_kv_heads = u32_at(document, "gemma4.attention.head_count_kv")?;
    let context_length = u32_at(document, "gemma4.context_length")?;
    let norm_eps =
        f32_opt(document, "gemma4.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);

    // Full-attention layers and windowed layers have DIFFERENT head dimensions and
    // DIFFERENT rope bases. Both pairs are read; the pattern decides which a layer gets.
    let head_dim_full = u32_at(document, "gemma4.attention.key_length")?;
    let head_dim_swa =
        u32_or(document, "gemma4.attention.key_length_swa", head_dim_full);
    let rope_dim_full = u32_or(document, "gemma4.rope.dimension_count", head_dim_full);
    let rope_dim_swa =
        u32_or(document, "gemma4.rope.dimension_count_swa", head_dim_swa);
    let rope_base_full = f32_opt(document, "gemma4.rope.freq_base").unwrap_or(10_000.0);
    let rope_base_swa =
        f32_opt(document, "gemma4.rope.freq_base_swa").unwrap_or(rope_base_full);
    let window = u32_or(document, "gemma4.attention.sliding_window", 0);

    let pattern = sliding_pattern(path, n_layers)?;

    // `shared_kv_layers` counts how many layers at the END of the network reuse an
    // earlier layer's KV, so the boundary is n_layers - shared_kv_layers. Reading it as
    // "layers at or after index 18 share" produced 3 KV-owning full-attention layers;
    // the file has attn_k on layers 0..23 only, and the owning full layers are
    // [5, 11, 17, 23] -- four, which matches the fork's measured E4B KV floor.
    let shared_count = u32_at(document, "gemma4.attention.shared_kv_layers").ok();
    let shared_from = shared_count.map(|n| n_layers.saturating_sub(n));

    // ROUTED OR DENSE, read rather than assumed. gemma-4-26B-A4B carries 128 experts
    // under this same architecture string; planning it from feed_forward_length alone
    // would describe a routed model as dense and be believed.
    let experts = u32_or(document, "gemma4.expert_count", 0);
    let ffn_plan = if experts > 0 {
        Ffn::Moe {
            activation: Activation::Gelu,
            expert_hidden: u32_or(document, "gemma4.expert_feed_forward_length", n_ff),
            experts,
            experts_used: u32_or(document, "gemma4.expert_used_count", 0),
            shared_hidden: u32_or(
                document,
                "gemma4.expert_shared_feed_forward_length",
                0,
            ),
        }
    } else {
        Ffn::Dense {
            activation: Activation::Gelu,
            hidden: n_ff,
        }
    };

    let mut layers = Vec::with_capacity(n_layers as usize);
    for index in 0..n_layers {
        let windowed = pattern[index as usize];
        let attention = if windowed && window > 0 {
            Attention::Window {
                head_dim: head_dim_swa,
                rope_base: rope_base_swa,
                rope_dim: rope_dim_swa,
                window,
            }
        } else {
            Attention::Full {
                head_dim: head_dim_full,
                rope_base: rope_base_full,
                rope_dim: rope_dim_full,
            }
        };
        let kv_source = match shared_from {
            Some(boundary) if index >= boundary => {
                KvSource::SharedWith(last_owner_before(boundary, &pattern, windowed))
            }
            _ => KvSource::Own,
        };
        layers.push(LayerPlan {
            index,
            attention,
            ffn: ffn_plan,
            kv_source,
        });
    }

    let vocab_size = vocab_from_tensors(document).unwrap_or(0);

    Ok(ModelPlan {
        config: ModelConfig {
            architecture: "gemma4".into(),
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
            scale_by_sqrt_embd: true,
            per_layer_dim: u32_at(document, "gemma4.embedding_length_per_layer_input")
                .ok(),
            // One row = one token's per-layer embedding; ne[1] is the vocabulary.
            per_layer_row_bytes: document
                .tensor("per_layer_token_embd.weight")
                .filter(|t| t.dimensions.len() >= 2 && t.dimensions[1] > 0)
                .map(|t| t.byte_size / t.dimensions[1]),
        },
        kv_storage_basis: crate::KvStorageBasisPolicy::BackendRoute,
        weight_residency: crate::WeightResidencyPlan {
            row_gathered: &["per_layer_token_embd.weight"],
        },
        layers,
        output: OutputPlan {
            final_norm: true,
            logit_softcap: f32_opt(document, "gemma4.final_logit_softcapping"),
            tied_embeddings: true,
        },
    })
}

/// The most recent KV-owning layer before `boundary` with the same attention kind. A
/// windowed layer cannot borrow a full layer's cache: the head dimensions differ.
fn last_owner_before(boundary: u32, pattern: &[bool], windowed: bool) -> u32 {
    (0..boundary)
        .rev()
        .find(|&i| pattern[i as usize] == windowed)
        .unwrap_or(0)
}

fn sliding_pattern(path: &Path, n_layers: u32) -> Result<Vec<bool>, PlanError> {
    let key = "gemma4.attention.sliding_window_pattern";
    let scan = ggufscan::Metadata::read(path)
        .map_err(|_| PlanError::MissingMetadata(key.into()))?;
    let ints = scan
        .ints(key)
        .ok_or_else(|| PlanError::MissingMetadata(key.into()))?;
    let pattern: Vec<bool> = ints.iter().map(|v| *v != 0).collect();
    if pattern.len() != n_layers as usize {
        return Err(PlanError::Inconsistent(format!(
            "{key} has {} entries but block_count is {n_layers}",
            pattern.len()
        )));
    }
    Ok(pattern)
}

fn vocab_from_tensors(document: &imparo_gguf::Document) -> Option<u32> {
    document
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .and_then(|t| t.dimensions.last().copied())
        .and_then(|v| u32::try_from(v).ok())
}
