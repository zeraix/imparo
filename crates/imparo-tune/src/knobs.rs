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
    /// Whether this architecture's decode WORKFLOW offers mega entries
    /// (`Backend::mega_layer`) at all. The tuner combines it with the backend's form
    /// policy into `ModelFacts::mega_seat` (task #203). Mirrors the workflow the way
    /// `decode_tensors` mirrors its tensor use: gemma4 and LFM2 dispatch entries; qwen35's
    /// workflow does not yet (#165), so the grid knobs must not be offered for it.
    pub mega_entries: bool,
    /// Builds this architecture's mega entries for the tuner's `MegaDecodeStep` workload
    /// (task #203): one `MegaLayer` per layer over the synthetic tensor offsets, positions
    /// left at 0 (the workload sets the span). Mirrors the workflow's own entry literal the
    /// way `decode_tensors` mirrors its tensor use. `None` = the workload is unavailable for
    /// this architecture; the grid knobs then keep their incumbent, and the sweep says so.
    pub mega_layers:
        Option<fn(&MegaBenchInputs<'_>) -> Vec<imparo_backend::MegaLayer<'static>>>,
}

/// What a mega entry builder reads: the plan (geometry and KV source per layer, rope), the
/// synthetic offset of each decode tensor by name, and one zero row every norm weight points
/// at -- a norm's cost does not depend on its values, and the rows are kilobytes.
pub struct MegaBenchInputs<'a> {
    pub plan: &'a imparo_model::ModelPlan,
    pub lookup: &'a dyn Fn(&str) -> Option<(u64, u32, u32, u32)>,
    pub norm_off: u64,
}

/// gemma4's level-5 entry -- the whole layer from the residual: q/k/v front, attention over
/// the cache, o_proj + sandwich, FFN, PLE, tails -- as `gemma4/workflow_gpu.rs` builds it,
/// over the synthetic tensors. Differences from the engine's entry, each timing-neutral:
/// every norm weight is the zero row, `out_scale` is 1.0 (a scalar), rope has no factor
/// table, and there is no next-norm tail (level 5 forms the next layer's input itself).
/// A shared-KV layer carries no K/V projection and reads its source layer's cache, as in
/// the engine. Empty when the model has no per-layer embedding (then the workflow offers no
/// entry either) or a tensor the entry needs is missing.
fn gemma4_mega_layers(
    i: &MegaBenchInputs<'_>,
) -> Vec<imparo_backend::MegaLayer<'static>> {
    use imparo_backend::{
        BufId, Gemma4MegaLayer, MegaAttn, MegaFront, MegaLayer, MegaQkv, NO_WEIGHT,
    };
    use imparo_model::{Attention, KvSource};
    let c = &i.plan.config;
    let Some(ple) = i.plan.embed.per_layer_dim.filter(|&p| p > 0) else {
        return Vec::new();
    };
    let (
        Some(gate),
        Some(up),
        Some(down),
        Some(pg),
        Some(pp),
        Some(wq),
        Some(wk),
        Some(wv),
        Some(wo),
    ) = (
        (i.lookup)("ffn_gate"),
        (i.lookup)("ffn_up"),
        (i.lookup)("ffn_down"),
        (i.lookup)("inp_gate"),
        (i.lookup)("proj"),
        (i.lookup)("attn_q"),
        (i.lookup)("attn_k"),
        (i.lookup)("attn_v"),
        (i.lookup)("attn_output"),
    )
    else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(i.plan.layers.len());
    for (li, layer) in i.plan.layers.iter().enumerate() {
        let (hd, rope_base, rope_dim, window) = match layer.attention {
            Attention::Full {
                head_dim,
                rope_base,
                rope_dim,
            } => (head_dim, rope_base, rope_dim, 0),
            Attention::Window {
                head_dim,
                rope_base,
                rope_dim,
                window,
            } => (head_dim, rope_base, rope_dim, window),
            Attention::Recurrent { .. } => return Vec::new(),
        };
        let (kv_layer, shared) = match layer.kv_source {
            KvSource::Own => (li as u32, false),
            KvSource::SharedWith(src) => (src, true),
        };
        let kv_hd = i
            .plan
            .layers
            .get(kv_layer as usize)
            .map_or(hd, |l| l.attention.head_dim());
        out.push(MegaLayer::Gemma4(Gemma4MegaLayer {
            gate_kind: gate.3,
            gate_off: gate.0,
            up_kind: up.3,
            up_off: up.0,
            down_kind: down.3,
            down_off: down.0,
            post_ffw_norm_off: i.norm_off,
            pg_kind: pg.3,
            pg_off: pg.0,
            pp_kind: pp.3,
            pp_off: pp.0,
            post_norm_off: i.norm_off,
            next_norm_off: NO_WEIGHT,
            out_scale: 1.0,
            n_embd: c.n_embd,
            n_ff: c.n_ff,
            ple,
            per_layer_off: (li as u32) * ple,
            eps: c.norm_eps,
            src: BufId::Cur,
            x: BufId::X,
            add: BufId::O,
            g: BufId::G,
            u: BufId::U,
            // The workflow's GATE / BACK / PER_LAYER scratch ids.
            gate: BufId::Model0,
            per_layer: BufId::Model2,
            back: BufId::Model1,
            next_out: BufId::X,
            front: Some(MegaFront {
                wo_kind: wo.3,
                wo_off: wo.0,
                attn: BufId::Attn,
                attn_in: c.n_heads * hd,
                post_attn_norm_off: i.norm_off,
                ffn_norm_off: i.norm_off,
                head_dim: hd,
                attention: Some(MegaAttn {
                    kv_layer,
                    n_heads: c.n_heads,
                    n_kv: c.n_kv_heads,
                    kv_width: c.n_kv_heads * kv_hd,
                    start_pos: 0,
                    window,
                    ring: 0,
                    q: BufId::Q,
                    had_k: 0,
                    had_v: 0,
                }),
                qkv: Some(MegaQkv {
                    wq_kind: wq.3,
                    wq_off: wq.0,
                    wk: (!shared).then_some((wk.3, wk.0)),
                    wv: (!shared).then_some((wv.3, wv.0)),
                    q_norm_off: i.norm_off,
                    k_norm_off: i.norm_off,
                    in_norm_off: i.norm_off,
                    rope_dim,
                    rope_base,
                    freqs: None,
                    k: BufId::K,
                    v: BufId::V,
                    layer: li as u32,
                }),
            }),
        }));
    }
    out
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
        mega_entries: true,
        mega_layers: Some(gemma4_mega_layers),
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
        mega_entries: true,
        // The LFM2 entry (short-conv / attention mixers, recurrent state) is not built for
        // the tuner yet: mega_nsg applies to LFM2 but keeps its incumbent until it is.
        mega_layers: None,
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
        mega_entries: false,
        mega_layers: None,
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
