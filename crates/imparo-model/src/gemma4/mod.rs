//! Gemma4, including E4B: the whole architecture in one directory.
//!
//! `plan` derives the layer plan and attention geometry from the GGUF; `workflow_cpu` is
//! the reference forward; `workflow_gpu` runs it over the backend trait. This file is the
//! seam between them and the engine. Nothing else in the tree changes to add a model.

pub mod chat_format;
pub mod plan;
pub mod workflow_cpu;
pub mod workflow_gpu;

pub use plan::build;

// What gemma4 contributes to the engine, and which file implements each row.
//
// Not here, because they are the same for every architecture and live on `Workflow`:
// `new`, `plan`, the pool accessors, `ensure_gpu_ready`, `gpu_prepare`, `gpu_fit_batch`,
// `kv_fit`, `kv_bytes_for`, `kv_scan`, the capacity check, the chunk loop with its
// absolute-position anchoring, `forward`, `forward_into`, `forward_next`.
//
// The last two rows are DECLARATIONS, not code: what buffers gemma4 needs, and the one
// buffer of its own that aliases inside the layout.
crate::architecture!(Gemma4Arch => Gemma4 {
    weights:             workflow_cpu::ModelW,
    prepare:             workflow_cpu::prepare,
    batch_host:          workflow_cpu::batch,
    device_batch:        workflow_gpu::batch,
    buffer_requirements: workflow_gpu::buffer_requirements,
});
