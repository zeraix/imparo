//! Qwen3 (Qwen3-8B, Qwen3-4B): the architecture's own directory.
//!
//! The plainest tower here, and that is its value: every block attends with grouped-query
//! attention, normalises each head of Q and K, rotates, and runs one SwiGLU FFN. Nothing
//! recurs, nothing is shared, nothing is windowed, no block is an exception. Where LFM2
//! and Qwen3.8 interleave a state-carrying layer and Gemma-4 carries per-layer
//! embeddings, this model exercises the attention and GEMM paths alone -- which is what
//! makes it the control in a speculation comparison.
//!
//! The host path in `workflow_cpu` is the oracle; `workflow_gpu` runs the same graph on a
//! device.
pub mod chat_format;
pub mod plan;
pub mod workflow_cpu;
pub mod workflow_gpu;

pub use plan::build;

// FIVE ROWS, NOT SIX: there is no `device_rows` yet, so co-batched decode declines this
// architecture rather than running a path nothing has measured. One client at a time is
// what the speculation work needs first.
crate::architecture!(Qwen3Arch => Qwen3 {
    weights:             workflow_cpu::ModelW,
    prepare:             workflow_cpu::prepare,
    batch_host:          workflow_cpu::batch,
    device_batch:        workflow_gpu::batch,
    device_prepare:      workflow_gpu::prepare_device,
    buffer_requirements: workflow_gpu::buffer_requirements,
    // A VERIFY IS ONE BATCH, AND IT RETURNS EVERY ROW'S LOGITS. Without this the engine
    // verifies a drafted chain one token at a time, and the drafter's feature bookkeeping
    // -- which asks for the rows of ONE forward -- cannot be satisfied: it asked for two
    // rows at 1862 while the last forward had written one row at 1863.
    device_all_logits: true,
    // And the verify of a drafted chain is that one batch: there is no recurrent state to
    // snapshot per prefix, so a partial accept rolls back by moving the KV cursor.
    device_prefix_verification: true,
    // And a TREE verify is one row-layout batch: every row carries its own position and
    // visibility mask, so siblings do not see each other.
    row_layout_forward: true,
});
