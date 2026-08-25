#![doc = "The host backend: plain-math kernels over imparo-gguf's weight mapping, and \
the `Backend` implementation that puts them behind the same seam a GPU sits behind. \
Two ways in, for two different jobs: `ops` is what the model layer's reference forward \
calls directly, and `CpuBackend` is what the engine, the KV pool and the disk tier talk \
to when no device is present. No model knowledge, no execution policy: imparo-model \
decides what to run and on which backend. Correctness bisection runs against the \
reference engine by layer (dev_harness/README.md), not against this crate."]

pub mod backend_impl;
pub mod ops;

pub use backend_impl::CpuBackend;
