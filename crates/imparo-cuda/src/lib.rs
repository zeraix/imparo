//! CUDA backend SKELETON (task #16). The structure a Windows/CUDA developer fills
//! without touching shared code; study references are llama.cpp's ggml-cuda and
//! candle's CUDA backend:
//!
//! - one host-side module per op family under `ops/` (ggml-cuda's file-per-op shape),
//!   each owning its kernels and their launch geometry;
//! - a context owning the device, stream(s), the weight blob upload, and the named
//!   buffer slots (`BufId` -> device allocation) -- the CUDA analogue of the Metal
//!   backend's globals + arena;
//! - `CudaBackend` implements `imparo_backend::Backend`; every method is `todo!()`
//!   until its op lands. The trait is the CONTRACT: the gemma4 workflow already runs
//!   entirely through it, so filling these stubs (plus registering in the runtime's
//!   `be()` selection point behind `feature = "cuda"`) is the whole port.
//!
//! Porting order that mirrors how the Metal backend was built and gated:
//!   1. buffers/arena + begin/end, 2. matmat (F32 then Q4_0), 3. rms_norm + rope +
//!   elementwise, 4. kv_store + attention, 5. the rest; gate every step with
//!   dev_harness/det_gate.py --model <gguf> against pinned logits produced by the
//!   CPU reference.
//!
//! With the feature OFF this crate compiles to just this skeleton on any OS.

#[cfg(feature = "cuda")]
mod backend_impl;
#[cfg(feature = "cuda")]
pub mod context;
#[cfg(feature = "cuda")]
pub mod knobs;
#[cfg(feature = "cuda")]
pub use backend_impl::CudaBackend;
