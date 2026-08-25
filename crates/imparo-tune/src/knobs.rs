//! The knob taxonomy, EXPLICIT. Every performance knob the engine carries belongs to
//! exactly one category, and the category decides how its value is obtained:
//!
//! 1. `ModelShape` -- derived from the model's tensor shapes by arithmetic. COMPUTED,
//!    never benched (e.g. rms-norm thread count from row width).
//! 2. `DeviceProfile` -- a property of the machine, measured once per machine by
//!    `imparo-metalbench` and reused for every model.
//! 3. `Arithmetic` -- model x device arithmetic over known quantities. COMPUTED.
//! 4. `Benched` -- genuinely uncertain: swept by this tuner, stored per
//!    (host fingerprint, model, search-space version).
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
    /// Compute dispatches ONE layer encodes in a decode step.
    ///
    /// COUNTED FROM THE FORWARD CODE, not measured and not guessed: the tuner never runs
    /// the graph, and the engine's own dispatch counter reads 0 unless `IMPARO_PROF=1` is
    /// set. It is the work term in the command-buffer trade -- see `flush_layers`.
    pub layer_dispatches: u32,
}

pub const MODEL_BENCHES: &[ModelBench] = &[ModelBench {
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
    // Counted from gemma4's decode layer in workflow_gpu.rs: 8 matmuls (Q/K/V, attention
    // output, FFN gate/up/down, and the per-layer embedding projection), 7 rms_norms, 2
    // ropes, 2 kv_stores, 1 attention, 1 strided multiply. Shared-KV layers skip the K/V
    // projections and their stores, so this is the full-attention layer's count.
    layer_dispatches: 21,
}];

#[must_use]
pub fn bench_for(arch: &str) -> &'static ModelBench {
    MODEL_BENCHES
        .iter()
        .find(|m| m.arch == arch)
        .unwrap_or(&MODEL_BENCHES[0])
}
