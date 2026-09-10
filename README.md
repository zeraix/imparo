<div align="center">

<img src="assets/imparo-wordmark-black.png" alt="Imparo by Zeraix" width="560" />

### LLM inference that adapts to your hardware and workload.

**Lower execution overhead, reusable context, and persistent state for long-running AI workloads.**

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

Imparo is a hardware- and workload-adaptive LLM inference engine built to reduce
execution overhead and repeated context computation, with persistent state for
long conversations and multi-agent workflows.

Built in Rust with native Metal, CUDA, and CPU backends, it combines measured
hardware tuning, workload-aware execution, and shared, paged inference state.
Supported configurations and measured results are documented below.

## Why Imparo

- **Execution that fits your hardware and workload.** Hardware limits, model
  layouts, and request size guide execution choices, helping avoid costly
  mismatches in short prompts and token generation. Imparo's tuning tools
  measure execution choices and save configurations for reuse.

- **Spend less time on execution overhead.** Direct backend calls keep the hot
  path compact. On supported Metal paths, megakernel decode combines small GPU
  dispatches, improving token generation in the [published benchmarks](#performance).
  Unsupported regions use ordinary kernels; recoverable decode failures can
  restore state and retry there.

- **Reuse context across agents.** Agents using the same model and matching
  prompt prefixes can share cached computation and full-attention KV pages,
  reducing repeated prefill work and duplicate cache storage. Sharing requires
  compatible model and cache settings.

- **Resume and branch with less recomputation.** Persisted KV and recurrent
  state let conversations restore matching saved boundaries after a switch or
  restart. Retained checkpoints also support rewinds and branches, so reusable
  history does not have to be processed from scratch.

- **Keep long tool loops manageable.** Superseded tool-step checkpoints are
  collapsed within a conversation turn. Paged allocation, residency budgets,
  and disk-cache eviction help control redundant state and inactive-cache
  growth as conversations accumulate.

- **Cleaner streaming for agent integrations.** Separate handling of reasoning,
  visible text, and tool calls helps clients render responses correctly and
  preserve the handoff to external tools.

- **Explore acceleration with a smaller memory overhead.** Imparo's native
  MTP research reuses target-model components and keeps architecture-specific
  auxiliary state, aiming to reduce the extra memory needed for speculative
  decoding. This remains a research path, not a default server capability.

- **Check correctness alongside speed.** Tuning screens measured gains against
  timing noise. Output and state-transition checks help identify numerical
  regressions, with results bound to the tested model, configuration, and hardware.

- **Extend the engine through focused contributions.** Model plans describe
  architecture-specific execution and state; backends provide kernels through
  shared interfaces. Contributors can add model support or improve a kernel
  without spreading model-specific branches throughout the runtime.

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

- **[2026-09-10]** Extended validation to **Apple M4 Pro**, reproducing the
  optimization benefits previously demonstrated on **M3 Pro**.
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

Shared paged residency currently targets the supported Metal path. The public
server executes generation requests serially: multi-agent context reuse does
not imply concurrent decoding or continuous batching. Persisted-state reuse
requires matching model/cache settings and retained cache data.

## Contributing

Contributions are welcome across model support, Metal and CUDA kernels, hardware
validation, hardware-specific tuning, KV and state management, correctness testing,
benchmarks, tooling, and documentation.

Performance contributions should include a reproducible baseline and the
corresponding correctness checks.

New to the codebase? Start with the [contribution guide](CONTRIBUTING.md) for an
architecture map, development checks, and small ways to help. Documentation fixes
and hardware reports are welcome; no CLA or copyright transfer is required.

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md). Report security concerns
privately using the contact in [SECURITY.md](SECURITY.md).

Questions or ideas? Open an [issue](https://github.com/zeraix/imparo/issues/new/choose).

## License

Imparo is licensed under the [Apache License 2.0](LICENSE).
