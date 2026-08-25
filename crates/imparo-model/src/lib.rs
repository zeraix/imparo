//! The MODEL LAYER: architecture + workflows, one module per model.
//!
//! Since imparo-runtime dissolved (2026-08-20) this crate owns the whole model layer.
//! For each architecture, `gemma4/` holds three roles:
//!   - `plan`         GGUF metadata -> ModelPlan (shapes, attention geometry, KV plan);
//!   - `workflow_gpu` the forward pass over the backend trait (imparo-backend);
//!   - `workflow_cpu` the reference forward.
//! `host` carries execution-host glue (profiling, footprint, KV config, argmax) behind
//! its one macOS gate. The plan types and `build_plan` registry live at the crate root.
//!
//! Adding a model = a sibling `<name>/` module with the same three roles plus any
//! backend kernels it needs, and one arm in `build_plan`. Nothing else changes.
//! Dependency direction: imparo-gguf (format) <- imparo-model (here) <- backends/server.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::path::Path;

use imparo_gguf::{Document, MetadataValue, Scalar};

pub mod backend;
pub mod chat;
pub mod cpu_support;
pub mod gemma4;
pub mod gpu_support;
pub mod host;
pub mod kv;
pub mod lfm2;
pub use imparo_cpu::ops;
pub use imparo_gguf::scan as ggufscan;
pub use imparo_gguf::weights;

/// Re-exported so tools can set backend tuning without depending on the backend crate.
#[cfg(target_os = "macos")]
pub use imparo_metal as imparo_metal_reexport;



/// Attention geometry for ONE layer.
///
/// Head dimension and rope base sit here, not on the model, because gemma4 mixes two
/// geometries in one network: windowed layers use head_dim 256 with rope base 1e4, full
/// layers use head_dim 512 with rope base 1e6. A model-level field cannot express that.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Attention {
    Full {
        head_dim: u32,
        rope_base: f32,
        rope_dim: u32,
    },
    Window {
        head_dim: u32,
        rope_base: f32,
        rope_dim: u32,
        window: u32,
    },
    /// A block carrying RECURRENT state instead of attending: a gated short convolution,
    /// a state-space scan, a linear-attention delta rule. It is in THIS enum because it
    /// occupies the same slot in a LayerPlan -- what a block does instead of attending --
    /// and because the pool has to see "this layer holds fixed-size state, not KV".
    ///
    /// WHAT IS SHARED IS THE SIZE, NOT THE SHAPE. Every architecture in this family
    /// describes its state as two per-layer element counts, which is exactly what the
    /// reference computes (llama-hparams.cpp, n_embd_r and n_embd_s):
    ///
    /// ```text
    ///           r (rolling / conv history)         s (recurrent matrix)
    /// LFM2      n_embd * (l_cache - 1)             0
    /// RWKV      token_shift_count * n_embd         n_embd * wkv_head_size
    /// Mamba     (d_conv-1)*(d_inner+2*g*d_state)   ssm_d_state * ssm_d_inner
    /// Kimi KDA  3 * (d_conv-1) * d_inner           head_dim^2 * n_head
    /// MiniMax   -                                  head_dim^2 * n_head
    /// ```
    ///
    /// Taps and widths are a MODEL's derivation of `r`, not a shared fact, so they stay in
    /// the model's own records where its kernels read them. Neither count grows with
    /// context, which is what `grows_with_context` reports and what keeps these blocks out
    /// of the KV pool.
    Recurrent {
        /// Elements per layer of rolling state: a conv history, a token shift.
        r_elems: u32,
        /// Elements per layer of recurrent matrix state; 0 for a block that has none.
        s_elems: u32,
    },
}

impl Attention {
    /// Head dimension, or ZERO for a block that does not attend.
    ///
    /// Zero is not a placeholder here, it is the right answer: every shared caller
    /// multiplies this by a KV element count, and a recurrent block contributes no KV per
    /// token. A caller that wants the geometry rather than the KV width must ask
    /// `is_attention` first.
    #[must_use]
    pub fn head_dim(&self) -> u32 {
        match *self {
            Self::Full { head_dim, .. } | Self::Window { head_dim, .. } => head_dim,
            Self::Recurrent { .. } => 0,
        }
    }
    /// True when this layer's state grows with context, i.e. it belongs in the KV pool.
    /// Windowed layers are bounded by the window and stay per-conversation.
    #[must_use]
    pub fn grows_with_context(&self) -> bool {
        matches!(self, Self::Full { .. })
    }
    /// False for a block that does not attend, so a caller can tell "head_dim 0" from a
    /// real geometry.
    #[must_use]
    pub fn is_attention(&self) -> bool {
        !matches!(self, Self::Recurrent { .. })
    }
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Full { .. } => "full",
            Self::Window { .. } => "window",
            Self::Recurrent { .. } => "recurrent",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    Gelu,
    Silu,
}

impl ModelPlan {
    /// Where each layer's recurrent state sits in the FLAT per-conversation buffer, as
    /// `(r_offset, s_offset, r_elems, s_elems)` in elements, indexed by layer.
    ///
    /// A fact about the PLAN, so it lives with the plan: the host keeps one `Vec` per
    /// layer and the device keeps one buffer, and two derivations of one layout is how a
    /// restored conversation reads another layer's state. Layers are laid out in index
    /// order; a non-recurrent layer contributes nothing.
    #[must_use]
    pub fn recurrent_layout(&self) -> Vec<(u32, u32, u32, u32)> {
        let mut at = 0_u32;
        let mut out = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let (r, sz) = match l.attention {
                Attention::Recurrent { r_elems, s_elems } => (r_elems, s_elems),
                _ => (0, 0),
            };
            out.push((at, at + r, r, sz));
            at += r + sz;
        }
        out
    }

    /// Elements the whole per-conversation recurrent state occupies; 0 when there is none.
    #[must_use]
    pub fn recurrent_elems(&self) -> u32 {
        self.recurrent_layout()
            .last()
            .map_or(0, |&(r_off, _, r, sz)| r_off + r + sz)
    }
}

impl Activation {
    /// The device epilogue that computes this activation.
    ///
    /// The workflow used to pass a bool and get GELU, which was true of gemma4 and of
    /// nothing else. The plan already knows; this is how it reaches the kernel.
    #[must_use]
    pub fn epilogue(self) -> imparo_backend::Epilogue {
        match self {
            Self::Gelu => imparo_backend::Epilogue::Gelu,
            Self::Silu => imparo_backend::Epilogue::Silu,
        }
    }
}

/// The feed-forward half of a block.
///
/// Dense and MoE are different SHAPES, not one shape with optional fields. A routed layer
/// has no single hidden width: it has an expert width, a count, and how many of them a
/// token actually runs. Writing that as a struct with `experts: Option<u32>` invites a
/// reader to use `hidden` without checking, which is how a 128-expert model gets planned
/// as dense -- and `gemma-4-26B-A4B` is exactly that model, under the same `gemma4`
/// architecture string as the dense E4B.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ffn {
    Dense {
        activation: Activation,
        hidden: u32,
    },
    /// Routed experts, with an optional shared expert every token also runs.
    Moe {
        activation: Activation,
        /// Hidden width of ONE expert.
        expert_hidden: u32,
        /// Experts in the layer.
        experts: u32,
        /// Experts a token is routed to.
        experts_used: u32,
        /// Hidden width of the always-on shared expert; 0 when there is none.
        shared_hidden: u32,
    },
}

impl Ffn {
    #[must_use]
    pub fn activation(&self) -> Activation {
        match *self {
            Self::Dense { activation, .. } | Self::Moe { activation, .. } => activation,
        }
    }
    /// Hidden width a token actually computes through: one expert's width times the
    /// experts it is routed to, plus any shared expert. This is the ACTIVE width, which is
    /// what an activation buffer must hold -- not the sum over all experts.
    #[must_use]
    pub fn active_hidden(&self) -> u32 {
        match *self {
            Self::Dense { hidden, .. } => hidden,
            Self::Moe {
                expert_hidden,
                experts_used,
                shared_hidden,
                ..
            } => expert_hidden * experts_used + shared_hidden,
        }
    }
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Dense { .. } => "dense",
            Self::Moe { .. } => "moe",
        }
    }
}

/// Which layer owns this layer's KV. Gemma4 shares KV across a suffix of its layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvSource {
    Own,
    SharedWith(u32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerPlan {
    pub index: u32,
    pub attention: Attention,
    pub ffn: Ffn,
    pub kv_source: KvSource,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub architecture: String,
    pub n_layers: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub context_length: u32,
    pub vocab_size: u32,
    pub norm_eps: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbedPlan {
    /// Gemma scales embeddings by sqrt(n_embd) at input.
    pub scale_by_sqrt_embd: bool,
    /// Gemma4 feeds an extra per-layer embedding of this width.
    pub per_layer_dim: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputPlan {
    pub final_norm: bool,
    pub logit_softcap: Option<f32>,
    pub tied_embeddings: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelPlan {
    pub config: ModelConfig,
    pub embed: EmbedPlan,
    pub layers: Vec<LayerPlan>,
    pub output: OutputPlan,
}

impl ModelPlan {
    /// Layers whose state grows with context. Only these belong in the KV pool.
    #[must_use]
    pub fn pooled_layers(&self) -> Vec<u32> {
        self.layers
            .iter()
            .filter(|l| {
                l.attention.grows_with_context() && l.kv_source == KvSource::Own
            })
            .map(|l| l.index)
            .collect()
    }

    /// Bytes of KV per token per layer kind, at a given element width.
    #[must_use]
    pub fn kv_bytes_per_token(
        &self,
        element_bytes: u32,
    ) -> BTreeMap<&'static str, u64> {
        let mut out = BTreeMap::new();
        for layer in &self.layers {
            // A recurrent block owns state but no KV, and its state does not scale with
            // tokens, so it has no place in a per-token figure. head_dim() would already
            // make it contribute zero; skipping it keeps the label out of the map too.
            if layer.kv_source != KvSource::Own || !layer.attention.is_attention() {
                continue;
            }
            let per = u64::from(self.config.n_kv_heads)
                * u64::from(layer.attention.head_dim())
                * u64::from(element_bytes)
                * 2; // K and V
            *out.entry(layer.attention.label()).or_insert(0) += per;
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PlanError {
    UnknownArchitecture(String),
    MissingMetadata(String),
    BadMetadata(String),
    Inconsistent(String),
}

impl Display for PlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownArchitecture(a) => write!(f, "unknown architecture {a:?}"),
            Self::MissingMetadata(k) => write!(f, "missing metadata {k:?}"),
            Self::BadMetadata(k) => write!(f, "bad metadata {k:?}"),
            Self::Inconsistent(m) => write!(f, "inconsistent plan: {m}"),
        }
    }
}

impl std::error::Error for PlanError {}

/// The registry. Adding a model means adding a module and one arm here.
///
/// # Errors
///
/// Returns [`PlanError::UnknownArchitecture`] when no builder claims the file.
/// The state every workflow carries, whatever its architecture.
///
/// Five of gemma4's seven fields and five of LFM2's seven were these, and both
/// constructors filled them the same way. Held by composition and reached through one
/// accessor pair -- the shape already used for `KvRuntime`, and what a base class's
/// protected fields become in Rust.
///
/// The test for what belongs here is "would a second model write this identically?".
/// `gemma4`'s per-layer-embedding buffers would not, so they are not here; the batch
/// width the activation buffers are sized for would, so it is.
pub struct WorkflowState {
    /// Host KV arrays. Empty when the device holds the weights -- see `host_forward`.
    pub kv: cpu_support::KvCache,
    /// Device-side cache counters: capacity, allocated slots, positions filled.
    pub kv_rt: kv::KvRuntime,
    /// Batch width the windowed KV rings were sized for; see `kv::ring_slots`.
    pub kv_ring_batch: usize,
    /// Batch width the activation buffers are currently sized for.
    pub gpu_batch: usize,
    /// Whether the device buffers have been built.
    pub gpu_ready: bool,
    /// Whether THIS workflow computes on the host.
    ///
    /// Not "is there a GPU": it is the answer to "will the arithmetic happen here", and
    /// the host KV cache is allocated exactly when it is true. Written as
    /// `!weights.gpu_enabled()` at the call site it read as the CPU path asking about
    /// the GPU, which is not what it means.
    pub host_forward: bool,
    /// Per-conversation recurrent state, empty for a model that has none.
    ///
    /// Here rather than on the one model that uses it so the "reset at position 0" rule
    /// is stated once, in the generic forward: any model with recurrent layers needs it,
    /// and for one without, `reset` walks an empty list.
    pub recurrent: cpu_support::RecurrentState,
    /// The most recent recurrent snapshot taken AT a unit boundary, and which one.
    ///
    /// A convolution history is one buffer holding "now", so a checkpoint for an earlier
    /// boundary cannot be read off the device after the fact. It is copied aside as each
    /// boundary is passed instead -- 0.34 MiB per 256 tokens for LFM2.5 -- which is what
    /// lets a shutdown spill checkpoint a position the forward has already run past.
    ///
    /// Empty for a model with no recurrent layers, and then every path touching it is a
    /// no-op.
    pub recur_ckpt: Vec<u8>,
    /// The position `recur_ckpt` was taken at; meaningless while it is empty.
    pub recur_ckpt_at: usize,
    /// Set for ONE batch: "a checkpoint boundary falls this many tokens into you".
    ///
    /// A recurrent model's device graph reads it and dispatches its state kernel a second
    /// time, into `BufId::RecurSnap`, so the batch can run to its natural end instead of
    /// being cut short to stand on the boundary. `arm_recurrent_snapshot` sets it,
    /// `take_recurrent_snapshot` clears it; None means this batch crosses no boundary.
    pub recur_snap: Option<u32>,
    /// `IMPARO_LOG`, read once.
    pub log: bool,
}

impl WorkflowState {
    /// Everything a workflow needs before it has resolved a single tensor.
    #[must_use]
    pub fn new(
        weights: &weights::Weights,
        plan: &ModelPlan,
        capacity: usize,
    ) -> Self {
        let host_forward = !weights.gpu_enabled();
        Self {
            // Storage only on the host path: the device keeps its own half-precision
            // ring buffers and never reads these, so allocating them anyway cost
            // 223 MiB -- more than half the engine's footprint.
            kv: cpu_support::KvCache::new(plan, capacity, host_forward),
            // Capacity comes from the caller's context bound. It used to live inside
            // the cache, which is why splitting the two nearly lost it: `Default` gives
            // 0, and "kv capacity 0 exceeded at pos 0" is what that looks like.
            kv_rt: kv::KvRuntime {
                capacity,
                slots: 0,
                filled: 0,
            },
            kv_ring_batch: 0,
            gpu_batch: 0,
            gpu_ready: false,
            host_forward,
            recurrent: cpu_support::RecurrentState::new(plan, host_forward),
            recur_ckpt: Vec::new(),
            recur_ckpt_at: 0,
            recur_snap: None,
            log: log_on(),
        }
    }
}

/// Whether the per-stage log is on (`IMPARO_LOG`).
///
/// At the crate root because it belongs to NEITHER path: the host reference, the device
/// workflow, the pool and the buffer placement all read it. It lived in `cpu_support`,
/// which had `gpu_support` calling into the host module six times for a switch.
#[must_use]
pub fn log_on() -> bool {
    std::env::var("IMPARO_LOG").is_ok_and(|v| v != "0" && !v.is_empty())
}

/// Tokens per prefill chunk when nothing overrides it.
///
/// 512, not 32. Prefill re-reads every weight row once per chunk, so a small chunk pays
/// the whole weight traffic many more times. Measured on M3 Pro, gemma4 E4B, cold_compare
/// ctx 4096, prefill kernel on:
///
/// ```text
///   batch=32     194.4 tok/s prefill
///   batch=512    550.2 tok/s prefill      2.8x, decode and footprint unchanged
/// ```
///
/// The tuner sweeps this, but the value an untuned engine uses has to be a good one on
/// its own -- most installs will never run the tuner. Defined ONCE, here, and imported by
/// the tuner and the server: a second copy is how the search-space version came to
/// disagree with itself, and it had already happened again (gemma4 and lfm2 each had one).
pub const PREFILL_BATCH: usize = 512;

/// The chunk width this process will use: `IMPARO_BATCH` when set, else [`PREFILL_BATCH`].
///
/// A free function because it needs no model: it reads an environment variable and a
/// constant. It was a method on the forward trait, and `ensure_gpu_ready` -- which is on
/// the OTHER trait -- had its own copy of the same four lines.
#[must_use]
pub fn prefill_batch() -> usize {
    std::env::var("IMPARO_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(PREFILL_BATCH)
        .max(1)
}

/// The message a workflow gives when the device was asked for and it has none.
///
/// One sentence, one place: `batch_device`, `ensure_gpu_ready` and `kv_fit` all refuse
/// for the same reason, and three separately-worded refusals read as three problems.
fn no_device_workflow(plan: &ModelPlan) -> String {
    format!(
        "{}: no GPU workflow; unset IMPARO_GPU to run the CPU reference deliberately",
        plan.config.architecture
    )
}

/// One architecture's contribution: the weights it resolves, and its arithmetic.
///
/// WHY THIS IS SEPARATE FROM [`Model`], since they describe the same thing from two
/// sides: Rust will not let one trait do both jobs. This one has static methods and an
/// associated type, so it can never be a trait object --
///
/// ```text
/// error[E0038]: the trait `Architecture` is not dyn compatible
/// ```
///
/// -- and a caller needs `Box<dyn Model>`, because the alternative the compiler itself
/// suggests ("define an enum where each variant holds one of these types") means every
/// new model edits a caller, which is the thing this seam exists to prevent.
///
/// So: `Architecture` is the GENERIC side, what a model supplies. `Model` is the DYN
/// side, what a caller uses. `impl<A: Architecture> Model for Workflow<A>` is the bridge,
/// written once.
///
/// Everything a workflow has that is NOT one of these is [`Workflow`], written once for
/// every architecture there will ever be -- the weight mapping, the plan, the cache, the
/// counters, the constructor, the capacity check, the chunk loop, the pool accessors.
/// Before this existed each model retyped about thirty lines of it, and that number grows
/// with every model added.
///
/// The methods take `&mut Workflow<Self>` rather than `&mut self` because the
/// architecture is a NAME, not a value: it carries no data, so there is nothing for
/// `self` to be.
pub trait Architecture: Sized + 'static {
    /// This architecture's weights, resolved once at load. gemma4 and LFM2 both call
    /// theirs `ModelW`, and they have almost no fields in common.
    type Weights: 'static;

    /// Whether this architecture has a device forward at all.
    ///
    /// Derived by `architecture!` from the presence of a `batch_device` row, so it cannot
    /// disagree with the table. It exists so the device path refuses at BRING-UP rather
    /// than after allocating buffers it will never dispatch against.
    const DEVICE: bool = false;

    /// Resolve them, rejecting a file that cannot be run rather than misreading it.
    ///
    /// # Errors
    /// When a required tensor is absent or carries a quant no kernel can read.
    fn prepare(
        weights: &weights::Weights,
        plan: &ModelPlan,
    ) -> Result<Self::Weights, String>;

    /// Runs ONE chunk on the host, leaving the LAST token's logits in `out`.
    ///
    /// `at` is the chunk's absolute start position, not its offset in the request.
    ///
    /// # Errors
    /// When a shape disagrees with the plan.
    fn batch_host(
        wf: &mut Workflow<Self>,
        tokens: &[u32],
        at: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), String>;

    /// Runs ONE chunk on the device.
    ///
    /// ONE row, not two, because the two callers differ only in a flag. `argmax` true
    /// means: do not write the logits, write the greedy pick's index into `out[0]` as
    /// raw bits. Splitting it gave every architecture a pair of wrappers whose bodies
    /// were textually identical across models.
    ///
    /// The default REFUSES rather than falling back to the host: the caller asked for the
    /// device, and a quiet fall back to CPU reads as a catastrophic regression. This is
    /// the rule `backend::enable_gpu` already states for init failures, and it is the
    /// whole statement that an architecture has no device workflow yet -- structural,
    /// not an `if` in that model's constructor.
    ///
    /// # Errors
    /// Always, unless overridden.
    fn device_batch(
        wf: &mut Workflow<Self>,
        _tokens: &[u32],
        _at: usize,
        _out: &mut Vec<f32>,
        _argmax: bool,
    ) -> Result<(), String> {
        Err(no_device_workflow(&wf.plan))
    }

    /// Every activation buffer this architecture needs for a batch of `b`, and which of
    /// them may share bytes.
    ///
    /// A function of the PLAN and two numbers, not of a loaded model: that is what makes
    /// a buffer list testable without a multi-gigabyte file on disk, and a buffer list
    /// nothing can check is a buffer list nobody has checked.
    ///
    /// `capacity` is the context bound. It is here because a quantized KV cache needs a
    /// dequant scratch sized for the largest layer at FULL capacity, which the batch
    /// width does not tell you.
    ///
    /// Empty by default -- an architecture with no device forward declares nothing.
    fn buffer_requirements(
        _plan: &ModelPlan,
        _b: usize,
        _capacity: usize,
    ) -> Vec<gpu_support::BufferRequirement> {
        Vec::new()
    }
}

/// Declares an architecture: the unit type, the workflow alias, and the `Architecture`
/// impl -- from a table of `role: path` rows.
///
/// WHY A MACRO. Every row of that impl is one call. Written by hand, each architecture
/// retypes six four-line signatures in order to put ONE path in each body, and two of the
/// rows (`prepare`, `batch_host`) come out textually identical in every model. The
/// signatures are the boilerplate, so they live here once and a model states facts:
///
/// ```ignore
/// architecture!(Lfm2Arch => Lfm2 {
///     weights:    workflow_cpu::ModelW,
///     prepare:    workflow_cpu::prepare,
///     batch_host: workflow_cpu::batch,
/// });
/// ```
///
/// The four device rows are OPTIONAL. Omitting them leaves the trait's defaults, which
/// refuse -- that is the whole statement that an architecture has no GPU workflow yet.
///
/// Each row names the file that implements it, which is what keeps host code in
/// `workflow_cpu.rs` and device code in `workflow_gpu.rs`: one `impl` block cannot be
/// split across files, so the block has to be a table and not the code itself.
#[macro_export]
macro_rules! architecture {
    ($arch:ident => $alias:ident {
        weights:    $weights:ty,
        prepare:    $prepare:path,
        batch_host: $batch_host:path,
        $(device_batch:        $device_batch:path,)?
        $(buffer_requirements: $buffer_requirements:path,)?
    }) => {
        /// The architecture. A unit type: it carries no data, it NAMES a set of methods.
        /// The data is in the workflow, which is every model's data.
        pub struct $arch;

        #[doc = concat!("`", stringify!($arch), "`'s workflow. Differs from every other \
                         architecture's in exactly one field's type.")]
        pub type $alias = $crate::Workflow<$arch>;

        impl $crate::Architecture for $arch {
            type Weights = $weights;

            fn prepare(
                weights: &$crate::weights::Weights,
                plan: &$crate::ModelPlan,
            ) -> ::std::result::Result<Self::Weights, ::std::string::String> {
                $prepare(weights, plan)
            }

            fn batch_host(
                wf: &mut $alias,
                tokens: &[u32],
                at: usize,
                out: &mut ::std::vec::Vec<f32>,
            ) -> ::std::result::Result<(), ::std::string::String> {
                $batch_host(wf, tokens, at, out)
            }

            $(
                // Presence of a device batch IS the declaration that a device forward
                // exists; deriving the flag from the table means it cannot disagree
                // with it.
                const DEVICE: bool = { let _ = $device_batch as fn(_, _, _, _, _) -> _; true };

                fn device_batch(
                    wf: &mut $alias,
                    tokens: &[u32],
                    at: usize,
                    out: &mut ::std::vec::Vec<f32>,
                    argmax: bool,
                ) -> ::std::result::Result<(), ::std::string::String> {
                    $device_batch(wf, tokens, at, out, argmax)
                }
            )?

            $(
                fn buffer_requirements(
                    plan: &$crate::ModelPlan,
                    b: usize,
                    capacity: usize,
                ) -> ::std::vec::Vec<$crate::gpu_support::BufferRequirement> {
                    $buffer_requirements(plan, b, capacity)
                }
            )?

        }
    };
}

/// A loaded model: one architecture's weights plus everything every model has.
///
/// WHY [`WorkflowState`] IS A SEPARATE STRUCT and not just more fields here: this type is
/// generic, so a trait object cannot name it -- `KvPoolMember` has no `A` to write. It
/// needs a handle on something CONCRETE, and `WorkflowState` is exactly the part of a
/// workflow that does not depend on the architecture. Flatten the two and the trait needs
/// one accessor per field again, which is the eight-accessor version this replaced.
///
/// `Workflow<Gemma4Arch>` and `Workflow<Lfm2Arch>` differ in exactly one field's type.
pub struct Workflow<A: Architecture> {
    /// The weight mapping, and whether a device holds it.
    pub weights: weights::Weights,
    pub plan: ModelPlan,
    /// The cache, the counters, the device flags, the log switch.
    pub state: WorkflowState,
    /// The architecture's resolved weights.
    pub w: A::Weights,
}

impl<A: Architecture> Workflow<A> {
    /// Loads one file's weights against a plan.
    ///
    /// # Errors
    /// When a required tensor is absent from the file.
    pub fn new(
        weights: weights::Weights,
        plan: ModelPlan,
        capacity: usize,
    ) -> Result<Self, String> {
        let state = WorkflowState::new(&weights, &plan, capacity);
        let w = A::prepare(&weights, &plan)?;
        Ok(Self {
            weights,
            plan,
            state,
            w,
        })
    }
}

/// Written once, for every architecture. This was five methods per model, three of them
/// one-liners returning a field.
impl<A: Architecture> kv::KvPoolMember for Workflow<A> {
    fn plan(&self) -> &ModelPlan {
        &self.plan
    }
    fn state(&self) -> &WorkflowState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut WorkflowState {
        &mut self.state
    }
    // Both bodies are generic and live in `gpu_support`, next to the rest of the device
    // machinery. These two are the object-safe face of them.
    fn ensure_gpu_ready(&mut self) -> Result<(), String> {
        Workflow::ensure_gpu_ready(self)
    }
    fn kv_fit(&mut self, positions: usize) -> Result<(), String> {
        Workflow::kv_fit(self, positions)
    }
}

/// Likewise: the object-safe face of every architecture, written once.
/// The forward, written once for every architecture.
///
/// The capacity check, the chunk width, anchoring chunk boundaries at ABSOLUTE positions,
/// bringing the device up, growing the cache, resetting recurrent state, updating
/// `filled` and the timing logs are all here. Each model used to carry a copy, and the
/// copies drifted: gemma4's re-implemented its own `ensure_gpu_ready` inline, and LFM2's
/// forgot the absolute-position anchoring.
impl<A: Architecture> Model for Workflow<A> {
    fn forward_into(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        let capacity = self.state.kv_rt.capacity;
        if start_pos + tokens.len() > capacity {
            return Err(format!("kv capacity {capacity} exceeded at pos {start_pos}"));
        }
        let batch = prefill_batch();
        let t_start = std::time::Instant::now();

        // Routing asks a runtime question -- is a backend holding the weights? -- not an
        // OS one. On a build with no GPU backend compiled in this is always false and the
        // host reference path runs.
        let on_gpu = !self.state.host_forward;
        if on_gpu {
            // A pool entry point can arrive before any forward, so this is the same call
            // the pool makes. gemma4 used to inline a second copy of it here.
            kv::KvPoolMember::ensure_gpu_ready(self)?;
            // Grow the cache to what THIS request reaches, not to the context bound. The
            // full-attention layers are 128 MiB of a 148 MiB cache at ctx 8192, and a
            // short conversation touches a fraction of it.
            Workflow::kv_fit(self, start_pos + tokens.len())?;
        }

        // Recurrent state is only valid for the positions it has already absorbed.
        // Resuming at 0 on a dirty state gives the right shape and the wrong numbers,
        // which is the failure that reads as success -- so the reset is here, not left
        // to the caller. A no-op for a model with no recurrent layers.
        //
        // AFTER the device is up, not before: `gpu_ready` is false on the first forward,
        // so a reset placed earlier would skip the device buffer and be correct only
        // because `gpu_prepare` happens to zero it too. One rule, one place.
        if start_pos == 0 {
            self.state.recurrent.reset();
            if on_gpu {
                self.zero_recurrent();
            }
        }

        let prof = std::env::var("IMPARO_PROF").is_ok_and(|v| v == "1");
        // The last unit boundary this call reaches; only that one can be checkpointed,
        // and everything below it is the previous call's copy.
        let last_boundary = (start_pos + tokens.len()) / imparo_kv::UNIT_TOKENS
            * imparo_kv::UNIT_TOKENS;
        let mut done = 0;
        while done < tokens.len() {
            // Chunk boundaries anchor at ABSOLUTE positions (multiples of `batch`), not
            // at the tail's start: a continued conversation -- a forward at start_pos > 0,
            // KV pool reuse -- must chunk exactly where a cold prefill of the same context
            // would, or the two produce different batch shapes for the same tokens. A
            // no-op for start_pos = 0 and for single-token decode steps.
            let at = start_pos + done;
            let n = (batch - (at % batch)).min(tokens.len() - done);
            // A checkpoint boundary can fall INSIDE this chunk -- chunks are cut on the
            // prefill grid (512) and units are 256, so an 800-token prefill runs
            // [0,512) [512,800) and passes straight through 768. The batch is not cut to
            // stand on it; it is told where it is and writes the state aside in passing.
            // Cutting cost +29 ms of 1191 on that prefill; this costs one dispatch.
            //
            // Device path only, and not because the host cannot do it: nothing on the
            // host path can USE the result. `kv_spill` refuses without a ready device,
            // so a host-only run has no checkpoint to write. The old code read the
            // DEVICE buffer here whichever path had run, which for a host forward
            // described nothing.
            if on_gpu {
                self.arm_recurrent_snapshot(at, n, last_boundary);
            }
            let chunk = &tokens[done..done + n];
            let t_chunk = std::time::Instant::now();
            if on_gpu {
                A::device_batch(self, chunk, at, out, false)?;
            } else {
                A::batch_host(self, chunk, at, out)?;
            }
            // Per-chunk series (IMPARO_PROF=1): where a long prefill's time accumulates.
            // Each chunk syncs -- its logits are read back -- so this is real wall time,
            // not encode time.
            if prof {
                eprintln!(
                    "[prof] chunk at={at} n={n} ms={:.1}",
                    t_chunk.elapsed().as_secs_f64() * 1e3
                );
            }
            done += n;
            if on_gpu {
                self.take_recurrent_snapshot(at);
            }
        }
        self.state.kv_rt.filled = start_pos + tokens.len();
        if self.state.log {
            let ms = t_start.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "[imparo] forward tokens={} start_pos={start_pos} batch={batch} \
backend={} ms={ms:.1} ms_per_token={:.2}",
                tokens.len(),
                if on_gpu { "gpu" } else { "cpu" },
                ms / tokens.len() as f64
            );
        }
        Ok(())
    }

    fn forward_next(&mut self, token: u32, start_pos: usize) -> Result<u32, String> {
        // The device path picks the token ON the device and copies back only the index:
        // the host argmax over 262144 rows cost 1 ms per token, which a GPU-only
        // attribution missed for a whole session. `argmax` true means the graph writes
        // that index into out[0] as raw bits -- decoded HERE, once, rather than in a
        // wrapper each architecture writes for itself.
        // No `A::DEVICE` here: `gpu_ready` is set only by `ensure_gpu_ready`, which
        // goes through `gpu_prepare`, which refuses without a device workflow. Asking
        // twice invites the two answers to disagree.
        if !self.state.host_forward && self.state.gpu_ready {
            let capacity = self.state.kv_rt.capacity;
            if start_pos + 1 > capacity {
                return Err(format!("kv capacity {capacity} exceeded at pos {start_pos}"));
            }
            Workflow::kv_fit(self, start_pos + 1)?;
            let mut pick = Vec::new();
            // Decode advances one position at a time, so a step either lands on a
            // boundary or does not; either way this call reaches at most one.
            self.arm_recurrent_snapshot(start_pos, 1, start_pos + 1);
            A::device_batch(self, &[token], start_pos, &mut pick, true)?;
            self.state.kv_rt.filled = start_pos + 1;
            self.take_recurrent_snapshot(start_pos);
            return Ok(pick[0].to_bits());
        }
        let mut logits = Vec::new();
        self.forward_into(&[token], start_pos, &mut logits)?;
        Ok(imparo_cpu::ops::argmax_f32(&logits))
    }
}

/// What the engine needs from a workflow to run it.
///
/// THE OUTWARD FACE, and nothing else: three methods a caller uses. `batch_host`,
/// `batch_device` and `device_argmax` were on here and should not have been -- they are
/// how a forward is built, not something a server ever calls, and putting them on the
/// object-safe trait published them as API.
///
/// The KV-pool half is [`kv::KvPoolMember`]. Together they are the whole seam: above this
/// line nothing names an architecture.
///
/// Every method is implemented ONCE, by `Workflow<A>`. It is a trait rather than inherent
/// methods only because callers hold `Box<dyn Model>` -- `Architecture` has static methods
/// and can never be a trait object.
pub trait Model: kv::KvPoolMember {
    /// Runs `tokens` starting at `start_pos` and leaves the LAST token's logits in `out`.
    ///
    /// `out` rather than a return value: the decode loop calls this once per token, and a
    /// fresh `Vec` per step is a 1 MB allocation for a 262144-row vocabulary.
    ///
    /// # Errors
    /// When the cache would overflow, or a device call fails.
    fn forward_into(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), String>;

    /// One greedy decode step, returning the chosen token.
    ///
    /// Which backend ran is not observable in the output: the device pick and the host
    /// argmax apply the same rule, the smallest index attaining the maximum.
    ///
    /// # Errors
    /// As [`Self::forward_into`].
    fn forward_next(&mut self, token: u32, start_pos: usize) -> Result<u32, String>;

    /// Allocating wrapper. A decode loop should use [`Self::forward_into`].
    ///
    /// # Errors
    /// As [`Self::forward_into`].
    fn forward(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>, String> {
        let mut out = Vec::new();
        self.forward_into(tokens, start_pos, &mut out)?;
        Ok(out)
    }
}

/// Builds the workflow for a plan's architecture.
///
/// The one place an architecture name maps to a type, and the companion to
/// [`build_plan`]: that turns a file into a description, this turns a description into
/// something that runs. A caller holding a `Box<dyn Model>` never learns which.
///
/// `+ Send` because a server holds the model behind a mutex and hands it to a worker
/// thread. Every `Workflow<A>` is Send; erasing the type must not erase that.
///
/// # Errors
/// When the architecture has no workflow, or a required tensor is absent.
pub fn load(
    weights: weights::Weights,
    plan: ModelPlan,
    capacity: usize,
) -> Result<Box<dyn Model + Send>, String> {
    match plan.config.architecture.as_str() {
        "gemma4" => Ok(Box::new(gemma4::Gemma4::new(weights, plan, capacity)?)),
        "lfm2" => Ok(Box::new(lfm2::Lfm2::new(weights, plan, capacity)?)),
        other => Err(format!(
            "no workflow for architecture '{other}'; build_plan accepted it, so the \
             plan arm and the load arm disagree"
        )),
    }
}

pub fn build_plan(document: &Document, path: &Path) -> Result<ModelPlan, PlanError> {
    let arch = document
        .string_value("general.architecture")
        .ok_or_else(|| PlanError::MissingMetadata("general.architecture".into()))?;
    match arch {
        "gemma4" => gemma4::build(document, path),
        "lfm2" => lfm2::build(document, path),
        other => Err(PlanError::UnknownArchitecture(other.into())),
    }
}

// ---- metadata helpers, shared by every model builder ----

pub(crate) fn u32_at(doc: &Document, key: &str) -> Result<u32, PlanError> {
    match doc.metadata.get(key) {
        Some(MetadataValue::Scalar(Scalar::Unsigned(v))) => {
            u32::try_from(*v).map_err(|_| PlanError::BadMetadata(key.into()))
        }
        Some(MetadataValue::Scalar(Scalar::Signed(v))) => {
            u32::try_from(*v).map_err(|_| PlanError::BadMetadata(key.into()))
        }
        Some(_) => Err(PlanError::BadMetadata(key.into())),
        None => Err(PlanError::MissingMetadata(key.into())),
    }
}

pub(crate) fn u32_or(doc: &Document, key: &str, fallback: u32) -> u32 {
    u32_at(doc, key).unwrap_or(fallback)
}

pub(crate) fn f32_at(doc: &Document, key: &str) -> Result<f32, PlanError> {
    match doc.metadata.get(key) {
        Some(MetadataValue::Scalar(Scalar::Float(v))) => Ok(*v as f32),
        Some(_) => Err(PlanError::BadMetadata(key.into())),
        None => Err(PlanError::MissingMetadata(key.into())),
    }
}

pub(crate) fn f32_opt(doc: &Document, key: &str) -> Option<f32> {
    f32_at(doc, key).ok()
}
