//! Qwen3.8 (Qwen3.8-27B): the architecture's own directory.
//!
//! What it exercises that neither shipped model does: three quarters of its blocks are
//! GATED DELTA-NET -- a linear-attention delta rule carrying both a conv history and a
//! recurrent matrix, where LFM2's conv blocks carry only the first -- and the sixteen that
//! do attend pack Q with a per-head sigmoid gate in one 12288-wide tensor.
//!
//! The host path in `workflow_cpu` is the oracle; `workflow_gpu` runs the same graph on a
//! device, and the two dispatch the SAME two recurrent ops in the same order.
pub mod chat_format;
pub mod plan;
pub mod workflow_cpu;
pub mod workflow_gpu;

pub use plan::build;

// What Qwen3.8 contributes to the engine: SIX rows. The macro derives `DEVICE` from the
// presence of `device_batch` rather than from a flag someone has to remember to flip.
crate::architecture!(Qwen35Arch => Qwen35 {
    weights:             workflow_cpu::ModelW,
    prepare:             workflow_cpu::prepare,
    batch_host:          workflow_cpu::batch,
    device_batch:        workflow_gpu::batch,
    device_prepare:      workflow_gpu::prepare_device,
    buffer_requirements: workflow_gpu::buffer_requirements,
});
