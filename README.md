<div align="center">

<img src="assets/imparo-wordmark-black.png" alt="Imparo by Zeraix" width="560" />

### LLM inference that adapts to your hardware and workload.

**Native model execution, measurement-driven tuning, and persistent inference state—unified in a Rust runtime.**

[Why Imparo](#why-imparo) ·
[Design Rules](#design-rules) ·
[Megakernel Decode](#megakernel-decode) ·
[Performance](#performance) ·
[Latest Updates](#latest-updates) ·
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

## Design rules

Two rules define the layering; everything else is their consequence.

- **A model contributes a plan and a workflow, never a scattered code path.** No
  `if arch == "gemma4"` outside the model layer.
- **A backend contributes kernels, never model knowledge.** No model name in a kernel. The one
  deliberate exception is the Metal megakernel's phase list: a decode layer is emitted from a
  per-architecture list of phases held in the backend, so a new architecture is a program rather
  than a new hand-written kernel.

And two principles govern how the code is allowed to grow:

- **Simple is the strength.** Middleware is thin: common abstraction exists only where it
  removes redundant code. There is no graph IR, no scheduler, no runtime fusion machinery —
  a model's workflow is a direct, hand-scheduled function calling a backend trait. Fusion
  happens by hand, where a measurement justified it.
- **Nothing is believed without a measurement.** Every kernel change is gated on pinned
  logits, determinism suites, and interleaved same-session comparisons. Rejected
  experiments are recorded with their numbers so they are not re-attempted on a hunch.

## Megakernel decode

On Metal a decode step used to be about 180 (Gemma 4 E4B) or 34 (LFM2) small GPU dispatches
per token, each carrying 5-9 microseconds of fixed cost around less than a microsecond of
arithmetic. Imparo now runs a whole decode layer as **one persistent dispatch**: the
threadgroups stay resident for the layer and a grid barrier stands where each dispatch
boundary used to be. On LFM2 a single dispatch covers all of a token's layers.

```
  dispatches per decode token     Gemma 4 E4B  ~180 -> 42     LFM2  34 -> 5
```

Both kernels are generated at build time from a phase list the host writes -- a phase is the
entry condition that enables it, the code it runs, and what separates it from the next phase
-- so a new architecture is a list of phases rather than a hand-written kernel. The route
refuses a layer it cannot serve (weights not resident, a shape the pipeline was not compiled
for) and leaves it on the ordinary dispatch path; a region that fails to complete is rolled
back and re-run there, so the fast route cannot return a wrong answer.

## Performance

Apple M3 Pro, Gemma 4 E4B (`UD-Q4_K_XL`) and LFM2.5-2.6B (`Q8_0`), f16 KV cache, 512-token
prefill chunks, thinking disabled on every engine. tok/s, median of two interleaved rounds.
Every engine is measured the same way and in the same session: a client sends the same prompt
to each in turn and times it, so prefill is derived from time-to-first-token and decode from
the token stream — no engine reports its own numbers. A reference cell shows that engine's
tok/s and imparo's lead over it.

| model | prompt | phase | imparo | llama.cpp | oMLX | rapid-mlx |
|---|---|---|---|---|---|---|
| E4B | 449 | prefill | **932** | 556 (+68%) | 590 (+58%) | 830 (+12%) |
| E4B | 5651 | prefill | **1063** | 570 (+86%) | 904 (+18%) | 960 (+11%) |
| E4B | 16191 | prefill | **1006** | 522 (+93%) | 888 (+13%) | 914 (+10%) |
| LFM2 | 455 | prefill | **1023** | 933 (+10%) | 675 (+52%) | 857 (+19%) |
| LFM2 | 5963 | prefill | **1060** | 980 (+8%) | 964 (+10%) | 1007 (+5%) |
| LFM2 | 17123 | prefill | **972** | 900 (+8%) | 921 (+6%) | 957 (+2%) |
| E4B | 449 | decode | **46.6** | 40.5 (+15%) | 43.9 (+6%) | 42.8 (+9%) |
| E4B | 5651 | decode | **44.2** | 38.5 (+15%) | 41.8 (+6%) | 40.5 (+9%) |
| E4B | 16191 | decode | **40.0** | 34.5 (+16%) | 37.9 (+6%) | 36.7 (+9%) |
| LFM2 | 455 | decode | **46.6** | 43.9 (+6%) | 46.8 (tie) | 44.6 (+4%) |
| LFM2 | 5963 | decode | **45.1** | 42.4 (+7%) | 44.6 (+1%) | 42.6 (+6%) |
| LFM2 | 17123 | decode | **42.7** | 40.0 (+7%) | 40.8 (+5%) | 39.6 (+8%) |

Ahead on every cell but one: LFM2's short-prompt decode ties oMLX (46.6 against 46.8).

## Latest Updates

Updates below reflect code already merged into the public `main` branch.
Model and backend availability remains qualification-specific.

- **[2026-09-07]** Published Metal megakernel decode for Gemma 4 E4B and LFM2,
  reducing GPU dispatches per token from about 180 to 42 and from 34 to 5,
  respectively, with unsupported layers and failed regions safely returning to
  the ordinary dispatch path.
- **[2026-09-07]** Published interleaved Apple M3 Pro benchmarks for short,
  medium, and long prefill and decode workloads against llama.cpp, oMLX, and
  rapid-mlx.
- **[2026-09-07]** Documented the model-plan/backend-kernel boundary and the
  measurement- and correctness-gated process used to select hardware- and
  workload-specific execution paths.

- **[2026-08-25]** Published the initial Rust runtime with an OpenAI-compatible
  chat-completions server.
- **[2026-08-25]** Published initial Gemma 4 and LFM2 architecture support with
  native CPU and Metal execution paths.
- **[2026-08-25]** Published content-addressed paged inference state with
  in-memory reuse and disk-backed persistence.

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
