//! LFM2 plan builder: GGUF metadata -> ModelPlan.
//!
//! EVERY FACT HERE WAS READ, from the file's own metadata and from the reference
//! implementation's builder (`llama.cpp/src/models/lfm2.cpp`), not inferred from the model
//! card. The three that would have been easy to guess wrong:
//!
//!   - the input embedding is NOT scaled and NOT normalised (`build_inp_embd(tok_embd)`
//!     with nothing after it), unlike gemma4 which scales by sqrt(n_embd);
//!   - `token_embd_norm` is the OUTPUT norm, not an embedding norm -- the reference maps
//!     LLM_TENSOR_OUTPUT_NORM_LFM2 to that name with the comment "fix for wrong tensor
//!     name", and there is no `output_norm.weight` in the file;
//!   - a block is recurrent exactly when its per-layer `head_count_kv` entry is ZERO.
//!     The key is an ARRAY here, where gemma4 reads the same key as a scalar.

use std::path::Path;

use crate::ggufscan;

use crate::{
    Activation, Attention, EmbedPlan, Ffn, KvSource, LayerPlan, ModelConfig, ModelPlan,
    OutputPlan, PlanError, f32_opt, u32_at, u32_or,
};

/// Builds an LFM2 plan from GGUF metadata.
///
/// # Errors
///
/// Returns [`PlanError`] when a required key is absent or malformed, or when the per-layer
/// `head_count_kv` array disagrees with the block count.
pub fn build(
    document: &imparo_gguf::Document,
    path: &Path,
) -> Result<ModelPlan, PlanError> {
    let n_layers = u32_at(document, "lfm2.block_count")?;
    let n_embd = u32_at(document, "lfm2.embedding_length")?;
    let n_ff = u32_at(document, "lfm2.feed_forward_length")?;
    let n_heads = u32_at(document, "lfm2.attention.head_count")?;
    let context_length = u32_at(document, "lfm2.context_length")?;
    let norm_eps =
        f32_opt(document, "lfm2.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
    let rope_base = f32_opt(document, "lfm2.rope.freq_base").unwrap_or(10_000.0);

    // Conv taps. The rolling state is `l_cache - 1` past values per channel, which is what
    // the reference calls d_conv; a 1-tap conv would have no state at all and the
    // reference asserts against it.
    let kernel = u32_at(document, "lfm2.shortconv.l_cache")?;
    if kernel < 2 {
        return Err(PlanError::Inconsistent(format!(
            "lfm2.shortconv.l_cache is {kernel}; a short conv needs at least 2 taps to \
             carry state"
        )));
    }

    // HEAD DIM IS DERIVED. There is no key_length key in this file, and the reference
    // takes n_embd_head_k = n_embd / n_head. rope_dim follows it: n_rot defaults to the
    // head dim with no rope.dimension_count present, so the whole head is rotated.
    if n_heads == 0 || n_embd % n_heads != 0 {
        return Err(PlanError::Inconsistent(format!(
            "lfm2.embedding_length {n_embd} is not a multiple of head_count {n_heads}"
        )));
    }
    let head_dim = n_embd / n_heads;
    let rope_dim = u32_or(document, "lfm2.rope.dimension_count", head_dim);

    // PER-LAYER, AND THE KEY IS AN ARRAY. Zero marks a recurrent (short conv) block; a
    // non-zero entry is that block's KV head count. LFM2.5-2.6B reads
    // [0,0,8,0,0,8,0,0,0,8,0,0,0,8,0,0,0,8,0,0,0,8,0,0,8,0,0,8,0,0]: 8 attention blocks
    // at 2,5,9,13,17,21,24,27 and 22 conv blocks.
    let kv_per_layer = head_count_kv_per_layer(path, n_layers)?;
    let attention_kv: Vec<u32> =
        kv_per_layer.iter().copied().filter(|&n| n != 0).collect();
    let n_kv_heads = attention_kv.first().copied().unwrap_or(0);
    // One GQA ratio for the whole network is what ModelConfig can express, and every
    // attention block here shares it. A file that mixed ratios would need a per-layer
    // field, so refuse rather than silently describe it with the first one.
    if attention_kv.iter().any(|&n| n != n_kv_heads) {
        return Err(PlanError::Inconsistent(format!(
            "lfm2.attention.head_count_kv mixes GQA ratios across layers: {attention_kv:?}"
        )));
    }

    // A sliding window is legal for this architecture and the reference applies it to the
    // ATTENTION blocks only. LFM2.5-2.6B does not carry the key.
    let window = u32_or(document, "lfm2.attention.sliding_window", 0);

    let mut layers = Vec::with_capacity(n_layers as usize);
    for index in 0..n_layers {
        let recurrent = kv_per_layer[index as usize] == 0;
        let attention = if recurrent {
            // The conv runs over the full model width -- in_proj is n_embd -> 3 * n_embd
            // and the three chunks (b, c, x) are each n_embd wide -- and holds `kernel - 1`
            // past values per channel. That product is the whole of this architecture's
            // state; there is no recurrent matrix, so s is zero. Width and taps are not
            // shared facts and stay with the model's own records.
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
        layers.push(LayerPlan {
            index,
            attention,
            // BOTH block kinds carry the same FFN, and it is SwiGLU: the reference builds
            // every layer's FFN with LLM_FFN_SILU and LLM_FFN_PAR.
            ffn: Ffn::Dense {
                activation: Activation::Silu,
                hidden: n_ff,
            },
            // Every block owns whatever state it has. Nothing shares here -- there is no
            // shared_kv_layers key -- and a conv block's state is excluded from the KV
            // pool by Recurrent::grows_with_context() rather than by its kv_source.
            kv_source: KvSource::Own,
        });
    }

    let vocab_size = vocab_from_tensors(document)
        .or_else(|| u32_at(document, "lfm2.vocab_size").ok())
        .unwrap_or(0);

    Ok(ModelPlan {
        config: ModelConfig {
            architecture: "lfm2".into(),
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
            // NOT scaled: the reference feeds build_inp_embd straight into the first
            // block's norm. token_embd_norm is the output norm, not an input one.
            scale_by_sqrt_embd: false,
            per_layer_dim: None,
            per_layer_row_bytes: None,
        },
        // The established Metal representation is unrotated. CUDA may opt into its
        // independently receipted llama-compatible full-head route without changing
        // the shared workflow semantics for backends that do not make that declaration.
        kv_storage_basis: crate::KvStorageBasisPolicy::BackendOverrideOrCanonical,
        weight_residency: crate::WeightResidencyPlan::default(),
        layers,
        output: OutputPlan {
            final_norm: true,
            logit_softcap: None,
            // `output.weight` is absent, so the reference duplicates token_embd for the
            // head.
            tied_embeddings: true,
        },
    })
}

/// The per-layer KV head counts. Read through the scanner because this key is an ARRAY,
/// which `u32_at` cannot express -- the same route gemma4 uses for its sliding pattern.
fn head_count_kv_per_layer(path: &Path, n_layers: u32) -> Result<Vec<u32>, PlanError> {
    let key = "lfm2.attention.head_count_kv";
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
