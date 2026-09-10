//! The knob taxonomy, EXPLICIT. Every performance knob the engine carries belongs to
//! exactly one category, and the category decides how its value is obtained:
//!
//! 1. `ModelShape` -- derived from the model's tensor shapes by arithmetic. COMPUTED,
//!    never benched (e.g. rms-norm thread count from row width).
//! 2. `DeviceProfile` -- a property of the machine, measured once per machine by
//!    `imparo-metalbench` and reused for every model.
//! 3. `Arithmetic` -- model x device arithmetic over known quantities. COMPUTED.
//! 4. `Benched` -- genuinely uncertain at an operator boundary: swept by this tuner,
//!    stored per (host fingerprint, model, search-space version).
//! 5. `EndToEnd` -- workflow/graph policies with no faithful operator proxy. The micro
//!    tuner preserves their incumbent; a whole-engine bracket and correctness receipt
//!    are the only authority allowed to promote another value.
//!
//! A model-specific bench extension (see `ModelBench`) declares the category of every
//! knob it adds. If a knob's measured effect sits under the tuner's noise floor, it
//! stays exposed with a stored default rather than becoming a search axis.

// NOT-BENCHED knobs, recorded so nobody re-benches a computed quantity. These are
// deliberately absent from every backend registry -- the registry declares only what
// must be measured; the operative list of measured knobs IS the registry
// (imparo-metal/src/backend_impl.rs, imparo-cuda/src/knobs.rs), stated once there:
//
// - norm_threads (ModelShape): derived from row width inside the norm kernel;
//   v3 removed it from the store for exactly this reason.
// - KV scratch sizing (ModelShape): computed from capacity x kv heads x head size.
// - memory bandwidth / peak TFLOPS (DeviceProfile): imparo-metalbench, once per
//   machine; used to sanity-check sweeps, never swept.

/// A model-specific bench extension. The tuner core (device probing, sweep machinery,
/// config store + versioning + stale rejection, decline-to-store, screening) is
/// model-agnostic; what varies per model is WHICH tensors the decode mix touches and
/// any shape-specific extra cases.
pub struct ModelBench {
    /// GGUF architecture string this extension serves (e.g. "gemma4").
    pub arch: &'static str,
    /// Tensor names (blk.0.*) whose real offsets stage 1's decode sweeps touch.
    pub decode_tensors: &'static [&'static str],
    /// Optional exact PLE gate/projection pair. These are deliberately named rather
    /// than inferred from the decode list: the exact-128 route is one PLE+FFN atomic
    /// candidate, so its synthetic transaction must carry the same two tensors as the
    /// Gemma4 workflow before it may issue a correctness receipt.
    pub ple_tensors: Option<(&'static str, &'static str)>,
    /// Optional short-convolution weight name. Its dimensions and recurrent-state
    /// compatibility are validated from GGUF metadata and the model plan.
    pub shortconv_tensor: Option<&'static str>,
    /// Compute dispatches ONE layer encodes in a decode step.
    ///
    /// COUNTED FROM THE FORWARD CODE, not measured and not guessed: the tuner never runs
    /// the graph, and the engine's own dispatch counter reads 0 unless `IMPARO_PROF=1` is
    /// set. It is the work term in the command-buffer trade -- see `flush_layers`.
    pub layer_dispatches: u32,
}

pub const MODEL_BENCHES: &[ModelBench] = &[
    ModelBench {
        arch: "gemma4",
        // gemma4's two attention geometries (256/512) are handled by measuring the
        // majority geometry (main.rs picks head_dim by layer count), not extra cases.
        decode_tensors: &[
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ],
        ple_tensors: Some(("inp_gate", "proj")),
        shortconv_tensor: None,
        // Counted from gemma4's decode layer in workflow_gpu.rs: 8 matmuls (Q/K/V, attention
        // output, FFN gate/up/down, and the per-layer embedding projection), 7 rms_norms, 2
        // ropes, 2 kv_stores, 1 attention, 1 strided multiply. Shared-KV layers skip the K/V
        // projections and their stores, so this is the full-attention layer's count.
        layer_dispatches: 21,
    },
    ModelBench {
        arch: "lfm2",
        // Layer zero is recurrent in the current LFM2.5 export. Its decode mix is the
        // short-convolution input/output projections plus the shared SwiGLU triplet.
        // Attention layers use the same FFN tensors and are covered by the explicit
        // attention workloads, so no Gemma tensor name is borrowed here.
        decode_tensors: &[
            "shortconv.in_proj",
            "shortconv.out_proj",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ],
        ple_tensors: None,
        shortconv_tensor: Some("shortconv.conv"),
        // Counted from lfm2/workflow_gpu.rs for a one-token recurrent layer: operator
        // norm, in projection, shortconv, out projection, residual add, FFN norm,
        // gate/up projections, standalone SiLU-mul, down projection, residual add.
        layer_dispatches: 11,
    },
    ModelBench {
        arch: "qwen35",
        // blk.0 is a GATED DELTA-NET block, and 48 of this model's 64 blocks are: the
        // file's `full_attention_interval` is 4 and the attending block is the LAST of
        // each group, so blk.0 carries attn_qkv / ssm_* and no attn_q at all. Naming
        // gemma4's attn_q/attn_k/attn_v here would look up tensors this file does not
        // have on block 0 and silently measure four fewer projections. This is the
        // majority block's decode mix, the same rule gemma4's row follows.
        //
        // ssm_alpha and ssm_beta are n_out = 48 -- six work units on any grid. They are
        // in the list because they really are dispatched once per delta block at decode;
        // dropping them would rank the knobs on a mix the engine never runs.
        decode_tensors: &[
            "attn_qkv",
            "attn_gate",
            "ssm_alpha",
            "ssm_beta",
            "ssm_out",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ],
        ple_tensors: None,
        // NOT ssm_conv1d. The short-convolution workload is for a conv that IS the whole
        // mixer state (LFM2: width = n_embd, no matrix state). qwen35's conv is one stage
        // of the gated delta rule -- it runs over the qkv projection's width and hands a
        // [value_head][value][key] matrix state to the delta rule after it. The
        // transaction's own guard rejects that shape, so naming it would add a non-
        // projection tensor to the decode mix and buy nothing.
        shortconv_tensor: None,
        // Counted from qwen35/workflow_gpu.rs for a one-token DELTA block (48 of 64):
        // attn norm, qkv / gate / alpha / beta projections, the causal conv (TWO: the
        // conv output and the state shift are separate dispatches on this route), the
        // delta rule, the per-value-head norm, act_mul, the out projection, residual add,
        // FFN norm, three FFN projections, act_mul, residual add.
        //
        // Cross-checked against the engine's own counter (IMPARO_PROF=1, 8 decode steps
        // on Qwen3.8-27B-UD-Q4_K_S): 9496 dispatches / 8 = 1187 per token, and
        // 48 delta x 18 + 16 attention x 20 + 3 outside the layers = 1187 exactly.
        layer_dispatches: 18,
    },
];

#[must_use]
pub fn bench_for(arch: &str) -> Option<&'static ModelBench> {
    MODEL_BENCHES.iter().find(|m| m.arch == arch)
}

#[cfg(test)]
mod tests {
    #[test]
    fn model_benches_are_explicit_and_unknown_architectures_fail_closed() {
        assert_eq!(super::bench_for("gemma4").unwrap().layer_dispatches, 21);
        assert_eq!(super::bench_for("lfm2").unwrap().layer_dispatches, 11);
        assert_eq!(super::bench_for("gemma4").unwrap().shortconv_tensor, None);
        assert_eq!(
            super::bench_for("lfm2").unwrap().shortconv_tensor,
            Some("shortconv.conv")
        );
        assert!(super::bench_for("unknown").is_none());
    }
}
