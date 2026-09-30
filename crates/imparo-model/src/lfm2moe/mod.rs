//! LFM2-MoE (LFM2.5-8B-A1B): the architecture's own directory.
//!
//! LFM2's tower with a routed feed-forward from `leading_dense_block_count` onward. It is
//! a separate directory rather than an arm inside `lfm2/` because the dense model is what
//! the shipping speculation work measures, and it does not move to make room for this.
//!
//! The device path is `workflow_gpu`, and what it needed that no other architecture here
//! does is a matmul that selects its weight per token (`docs/moe-support-design.md`). The
//! host workflow stays the oracle that route is gated against.
pub mod plan;
pub mod workflow_cpu;
pub mod workflow_gpu;

pub use plan::build;

crate::architecture!(Lfm2MoeArch => Lfm2Moe {
    weights:        workflow_cpu::ModelW,
    prepare:        workflow_cpu::prepare,
    batch_host:     workflow_cpu::batch,
    device_batch:   workflow_gpu::batch,
    device_prepare: workflow_gpu::prepare_device,
    buffer_requirements: workflow_gpu::buffer_requirements,
    // A tree verify rebuilds these windows from the accepted path's kept inputs; without them
    // the windows do not cover the recurrent state and every draft verifies as a chain.
    conv_windows:   plan::conv_windows,
    device_all_logits: true,
    device_prefix_verification: true,
    row_layout_forward: true,
});
