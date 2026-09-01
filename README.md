<div align="center">

<img src="assets/imparo-wordmark-black.png" alt="Imparo by Zeraix" width="560" />

### Local inference, fitted to your hardware.

**An independent, hardware-adaptive inference engine for local models.**

[Features](#why-imparo) ·
[Status](#current-status) ·
[Contributing](#contributing) ·
[Issues](https://github.com/zeraix/imparo/issues)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-38D6B4?style=flat-square)](LICENSE)

</div>

## About

Imparo is built around a Rust runtime with direct Metal, CUDA, and CPU backends.
It adapts model execution to the hardware, model architecture, and request shape
it runs with.

GGUF is an input format; model execution is implemented by Imparo rather than
delegated to llama.cpp, GGML, or a general-purpose machine-learning framework.

## Why Imparo

- **Lean, direct execution.** Model workflows call backend kernels directly,
  without a general graph runtime, heavyweight scheduler, or runtime-fusion layer.

- **Tuned to the machine.** Imparo profiles the device, reads model geometry from
  GGUF, and benchmarks only the execution choices that cannot be safely derived.

- **Shaped by the request.** Decode, narrow-batch, prefill, attention, and KV
  paths are selected from the current token width, context depth, KV format,
  prefix reuse, and per-layer model structure.

- **State built for repeated work.** Content-addressed paged state allows
  matching prefixes to be reused across conversations. Inactive state can be
  managed across memory and disk under explicit resource limits.

- **Evidence before claims.** Optimizations are checked for correctness and
  determinism before adoption. Performance results remain tied to the exact
  model, hardware, configuration, and engine version that produced them.

## Current Status

Apple Silicon with Metal is the primary development and validation platform
today. The CUDA backend is under active hardware-specific validation, and the
CPU backend provides a reference and fallback path.

Current model work focuses on Gemma4 E4B and LFM2.5. Broader model coverage,
sampling, continuous batching, and additional hardware validation are in
progress.

Imparo is under active development. Support claims apply only to the model,
weight format, KV format, backend, and hardware combinations that have been
explicitly validated.

## Contributing

Contributions are welcome across model support, Metal and CUDA kernels,
hardware validation, tuning, KV and state management, correctness testing,
benchmarks, tooling, and documentation.

Performance contributions should include a reproducible baseline and the
corresponding correctness checks.

Start with an open [issue](https://github.com/zeraix/imparo/issues).

## License

Imparo is licensed under the [Apache License 2.0](LICENSE).
