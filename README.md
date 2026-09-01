<div align="center">

<img src="assets/imparo-wordmark-black.png" alt="Imparo by Zeraix" width="560" />

### LLM inference that adapts to your hardware and workload.

**Native model execution, measurement-driven tuning, and persistent inference state—unified in a Rust runtime.**

[Why Imparo](#why-imparo) ·
[Quick Start](#quick-start) ·
[Contributing](#contributing) ·
[Issues](https://github.com/zeraix/imparo/issues)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-38D6B4?style=flat-square)](LICENSE)

</div>

## About

Imparo is a hardware- and workload-adaptive LLM inference engine built in Rust
with native Metal, CUDA, and CPU backends. It brings model execution,
measurement-driven hardware tuning, and persistent inference state into one runtime.

Execution adapts to the model architecture, hardware topology, and live workload
shape—from short decode steps to long prefills and repeated-prefix workloads.

## Why Imparo

- **Lean native execution.** Model workflows call backend kernels directly,
  keeping the hot path compact and giving each architecture an execution path
  designed around its actual operators and state.

- **Fitted to your hardware.** Imparo reads the model and device geometry, then
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

- **Correctness-gated optimization.** Candidate optimizations must preserve
  model outputs and state transitions before they can be selected. Performance
  results remain bound to the configuration and hardware that produced them.

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
validation, hardware-specific tuning, KV and state management, correctness testing,
benchmarks, tooling, and documentation.

Performance contributions should include a reproducible baseline and the
corresponding correctness checks.

Start with an open [issue](https://github.com/zeraix/imparo/issues).

## License

Imparo is licensed under the [Apache License 2.0](LICENSE).
