//! LFM2 / LFM2.5 (LiquidAI): the whole architecture in one directory.
//!
//! What it exercises that gemma4 does not: most blocks do not attend at all -- 22 of 30 in
//! LFM2.5-2.6B are gated short convolutions with a fixed-size rolling state -- the FFN is
//! SwiGLU rather than GELU, and the output norm is spelled `token_embd_norm` because the
//! exported name is wrong (the reference carries the same fix-up).

pub mod chat_format;
pub mod plan;
pub mod workflow_cpu;
pub mod workflow_gpu;

pub use plan::build;

// What LFM2 contributes to the engine: FIVE rows.
crate::architecture!(Lfm2Arch => Lfm2 {
    weights:             workflow_cpu::ModelW,
    prepare:             workflow_cpu::prepare,
    batch_host:          workflow_cpu::batch,
    device_batch:        workflow_gpu::batch,
    buffer_requirements: workflow_gpu::buffer_requirements,
});
