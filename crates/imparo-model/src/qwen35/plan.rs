//! Qwen3.8 plan builder: GGUF metadata -> ModelPlan.
//!
//! EVERY FACT HERE WAS READ from the file's own metadata and tensor table, not from the
//! model card. Four that would have been easy to get wrong:
//!
//!   - `qwen35.block_count` is 65 but the main tower is 64. The last block is the
//!     multi-token-prediction head: it carries `nextn.eh_proj`, `nextn.enorm`,
//!     `nextn.hnorm` and `nextn.shared_head_norm` on top of an ordinary attention block.
//!     The count to drop is `qwen35.nextn_predict_layers`, which is READ rather than
//!     assumed to be 1, so a file with a deeper MTP stack describes itself correctly the
//!     day one appears.
//!   - The FULL-ATTENTION layers are `i % full_attention_interval == interval - 1`, so
//!     3, 7, ... 63: sixteen of them. The other forty-eight are gated delta-net. Verified
//!     against the tensor table -- blk.0..2 carry `ssm_*`, blk.3 carries `attn_q/k/v` --
//!     and the builder CHECKS it rather than trusting the arithmetic, because a file that
//!     phased the pattern differently would otherwise be described wrongly and run.
//!   - The head dim is `attention.key_length` (256), NOT `n_embd / head_count`, which
//!     would be 213 and is not even an integer. A model whose head dim is not the
//!     embedding width over the head count is exactly the case that silently produces a
//!     plausible plan.
//!   - Some exports carry an independent `output.weight`; smaller tied-embedding
//!     models omit it and use `token_embd.weight` for both lookup and output projection.
//!
//! The delta-net state is two counts, both derived:
//!
//! ```text
//!   head_dim  = ssm.inner_size / ssm.time_step_rank      6144 / 48       = 128
//!   qkv       = 2*group_count*head_dim + inner_size      2048+2048+6144  = 10240
//!   r_elems   = (ssm.conv_kernel - 1) * qkv              3 * 10240       = 30720
//!   s_elems   = time_step_rank * head_dim * state_size   48 * 128 * 128  = 786432
//! ```
//!
//! `qkv` is the width the conv1d runs over, and the tensor table agrees: `ssm_conv1d` is
//! (4, 10240) and `attn_qkv` is 5120 -> 10240. Q and K are `group_count` heads wide each
//! and V is all `time_step_rank` heads, which is why the three are not equal.
//!
//! 48 layers x (30720 + 786432) x 4 bytes = 149.6 MiB of recurrent state per conversation.
//! It does not grow with context, which is what keeps these layers out of the KV pool.

use std::path::Path;

use crate::{
    Activation, Attention, EmbedPlan, Ffn, KvSource, LayerPlan, ModelConfig, ModelPlan,
    OutputPlan, PlanError, f32_opt, u32_at, u32_or,
};

/// Builds a Qwen3.8 plan from GGUF metadata.
///
/// # Errors
///
/// Returns [`PlanError`] when a required key is absent, when the MTP head count leaves no
/// main tower, or when the derived full-attention pattern disagrees with the tensors the
/// file actually carries.
pub fn build(
    document: &imparo_gguf::Document,
    path: &Path,
) -> Result<ModelPlan, PlanError> {
    let _ = path;
    super::weight_basis::validate_header(document).map_err(PlanError::Inconsistent)?;
    let tied_embeddings = document.tensor("output.weight").is_none();
    let blocks = u32_at(document, "qwen35.block_count")?;
    let n_embd = u32_at(document, "qwen35.embedding_length")?;
    let n_ff = u32_at(document, "qwen35.feed_forward_length")?;
    let n_heads = u32_at(document, "qwen35.attention.head_count")?;
    let n_kv_heads = u32_at(document, "qwen35.attention.head_count_kv")?;
    let context_length = u32_at(document, "qwen35.context_length")?;
    let norm_eps =
        f32_opt(document, "qwen35.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
    let rope_base = f32_opt(document, "qwen35.rope.freq_base").unwrap_or(10_000.0);

    // THE MAIN TOWER IS block_count MINUS THE MTP HEAD. Read the count; do not assume 1.
    let mtp_layers = u32_or(document, "qwen35.nextn_predict_layers", 0);
    let n_layers = blocks
        .checked_sub(mtp_layers)
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            PlanError::Inconsistent(format!(
                "qwen35.block_count is {blocks} and qwen35.nextn_predict_layers is \
             {mtp_layers}, which leaves no main tower"
            ))
        })?;

    // PARTIAL ROPE. `rope.dimension_count` is 64 of a 256-wide head, and
    // `rope.dimension_sections` [11, 11, 10, 0] is the multimodal section split whose
    // first three entries sum to 32 -- the pair count, so 64 rotated components. The text
    // tower uses the sections only through that total, so the plan carries the count.
    let head_dim = u32_at(document, "qwen35.attention.key_length")?;
    let rope_dim = u32_or(document, "qwen35.rope.dimension_count", head_dim);

    // The delta-net geometry, all derived. `time_step_rank` is the V head count and
    // `group_count` the Q/K head count -- the names are the reference's, the meaning is
    // the tensor table's.
    let v_heads = u32_at(document, "qwen35.ssm.time_step_rank")?;
    let kv_groups = u32_at(document, "qwen35.ssm.group_count")?;
    let inner = u32_at(document, "qwen35.ssm.inner_size")?;
    let state = u32_at(document, "qwen35.ssm.state_size")?;
    let conv_kernel = u32_at(document, "qwen35.ssm.conv_kernel")?;
    if v_heads == 0 || inner % v_heads != 0 {
        return Err(PlanError::Inconsistent(format!(
            "qwen35.ssm.inner_size {inner} is not a multiple of time_step_rank {v_heads}"
        )));
    }
    if conv_kernel < 2 {
        return Err(PlanError::Inconsistent(format!(
            "qwen35.ssm.conv_kernel is {conv_kernel}; a conv needs at least 2 taps to \
             carry state"
        )));
    }
    let ssm_head_dim = inner / v_heads;
    let qkv_width = 2 * kv_groups * ssm_head_dim + inner;
    let r_elems = (conv_kernel - 1) * qkv_width;
    let s_elems = v_heads * ssm_head_dim * state;

    // THE PATTERN, CHECKED AGAINST THE TENSORS. `full_attention_interval` says every
    // fourth block attends, the last of each group of four. Arithmetic alone would
    // describe a differently phased file wrongly and then run it, so the tensor table
    // decides and the arithmetic only has to agree.
    let interval = u32_or(document, "qwen35.full_attention_interval", 0);
    if interval < 2 {
        return Err(PlanError::Inconsistent(format!(
            "qwen35.full_attention_interval is {interval}; expected at least 2"
        )));
    }
    let mut layers = Vec::with_capacity(n_layers as usize);
    for index in 0..n_layers {
        let attends = has_tensor(document, index, "attn_q.weight");
        let recurs = has_tensor(document, index, "ssm_conv1d.weight");
        if attends == recurs {
            return Err(PlanError::Inconsistent(format!(
                "blk.{index} carries {} -- a block is either full attention (attn_q) or \
                 gated delta-net (ssm_conv1d), never both and never neither",
                if attends {
                    "both attn_q and ssm_conv1d"
                } else {
                    "neither attn_q nor ssm_conv1d"
                }
            )));
        }
        if attends != (index % interval == interval - 1) {
            return Err(PlanError::Inconsistent(format!(
                "blk.{index} is {} but qwen35.full_attention_interval {interval} says it \
                 is {}; the file's layer pattern is not the one the key describes",
                if attends {
                    "full attention"
                } else {
                    "gated delta-net"
                },
                if index % interval == interval - 1 {
                    "full attention"
                } else {
                    "gated delta-net"
                }
            )));
        }
        let attention = if attends {
            Attention::Full {
                head_dim,
                rope_base,
                rope_dim,
            }
        } else {
            Attention::Recurrent {
                r_elems,
                s_elems,
                // The state matrix is [value_head][value_coord][key_coord] with the key
                // coordinate contiguous; both coordinates are `ssm_head_dim` here, and
                // the state_size key confirms the key one.
                key_dim: state,
                value_dim: ssm_head_dim,
            }
        };
        layers.push(LayerPlan {
            index,
            attention,
            // Both block kinds carry the same SwiGLU FFN: ffn_gate, ffn_up, ffn_down are
            // present on every block including the attention ones.
            ffn: Ffn::Dense {
                activation: Activation::Silu,
                hidden: n_ff,
            },
            // Nothing shares here; there is no shared_kv_layers key. A delta-net block is
            // kept out of the KV pool by Recurrent::grows_with_context(), not by this.
            kv_source: KvSource::Own,
        });
    }

    let vocab_size = vocab_from_tensors(document)
        .or_else(|| u32_at(document, "qwen35.vocab_size").ok())
        .unwrap_or(0);

    Ok(ModelPlan {
        config: ModelConfig {
            architecture: "qwen35".into(),
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
            // Not scaled: there is no embedding scale key and no norm between the table
            // and the first block's attn_norm.
            scale_by_sqrt_embd: false,
            per_layer_dim: None,
            per_layer_row_bytes: None,
        },
        kv_storage_basis: crate::KvStorageBasisPolicy::BackendOverrideOrCanonical,
        // A tied table is also the full output projection: retain the common owner
        // instead of allowing embedding-only row staging, as in the LFM2 plan.
        weight_residency: crate::WeightResidencyPlan {
            row_gathered: if tied_embeddings { &[] } else { &["token_embd.weight"] },
        },
        layers,
        // Neither decode route: the interleave is worth 0.65% at this model's 119 ms step
        // and there is no mega-kernel route for the architecture, so the recurrent state
        // is one plane (149.6 MiB), updated in place.
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

fn has_tensor(document: &imparo_gguf::Document, layer: u32, suffix: &str) -> bool {
    let name = format!("blk.{layer}.{suffix}");
    document.tensors.iter().any(|t| t.name == name)
}

fn vocab_from_tensors(document: &imparo_gguf::Document) -> Option<u32> {
    document
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .and_then(|t| t.dimensions.last().copied())
        .and_then(|v| u32::try_from(v).ok())
}
