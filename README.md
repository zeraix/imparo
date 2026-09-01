<div align="center">

<img src="assets/imparo-wordmark-black.png" alt="Imparo by Zeraix" width="560" />

### LLM inference, fitted to your hardware.

**An independent, hardware-adaptive LLM inference engine.**

[Why Imparo](#why-imparo) ·
[Quick Start](#quick-start) ·
[Contributing](#contributing) ·
[Issues](https://github.com/zeraix/imparo/issues)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-38D6B4?style=flat-square)](LICENSE)

</div>

## About

Imparo is an independent LLM inference engine built around a Rust runtime and
direct Metal, CUDA, and CPU backends. It fits model execution to the hardware,
model architecture, and shape of the workload instead of assuming one fixed
deployment environment.

GGUF is an input format; model execution is implemented by Imparo rather than
delegated to llama.cpp, GGML, or a general-purpose machine-learning framework.

## Why Imparo

- **Fitted to the machine.** Imparo reads the model and device geometry, then
  measures only the execution choices that cannot be safely derived. Tuning is
  tied to the exact model, hardware, backend, and engine version that produced it.

- **Shaped by the workload.** Prefill, decode, attention, KV, and narrow-batch
  paths can be selected from the current token width, context depth, cache
  format, prefix reuse, and per-layer model structure.

- **Persistent, paged state.** Content-addressed state allows matching prefixes
  to be reused across conversations, while inactive KV and recurrent state can
  move between memory and disk under explicit resource limits. This is designed
  for multi-turn conversations, repeated tool use, and long-running workloads.

- **Memory-efficient model-native acceleration.** Imparo's native MTP research
  reuses target-model components and keeps only the architecture-specific
  auxiliary state, rather than loading a complete second draft model.

## Quick Start

Build Imparo from source with Rust 1.85 or later:

```sh
git clone https://github.com/zeraix/imparo.git
cd imparo
cargo build --release --locked --bin imparo-server
```

Start the server with a supported GGUF model. On Apple Silicon, enable the Metal
backend with `IMPARO_GPU=1`:

```sh
IMPARO_GPU=1 ./target/release/imparo-server \
  --model /path/to/model.gguf \
  --port 8420 \
  --ctx 4096
```

Send an OpenAI-compatible chat-completion request:

```sh
curl http://127.0.0.1:8420/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "imparo",
    "messages": [
      {"role": "user", "content": "Explain why low-latency inference matters."}
    ],
    "max_tokens": 128,
    "stream": false
  }'
```

Backend and model support is qualification-specific. Apple Silicon with Metal is
the primary development and validation path today; other backend combinations
should be treated as experimental until explicitly validated.

## Contributing

Contributions are welcome across model support, Metal and CUDA kernels, hardware
validation, per-machine tuning, KV and state management, correctness testing,
benchmarks, tooling, and documentation.

Performance contributions should include a reproducible baseline and the
corresponding correctness checks.

Start with an open [issue](https://github.com/zeraix/imparo/issues).

## License

Imparo is licensed under the [Apache License 2.0](LICENSE).
