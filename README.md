<div align="center">

<img src="assets/imparo-wordmark-black.png" alt="Imparo by Zeraix" width="560" />

### LLM inference that adapts to your hardware and workload.

**Lower execution overhead, reusable context, and persistent state for long-running AI workloads.**

[Why Imparo](#why-imparo) ·
[Design Rules](#design-rules) ·
[Megakernel Decode](#megakernel-decode) ·
[Performance](#performance) ·
[SM86 Original vs Imparo](#sm86-original-runner-vs-imparo) ·
[Bonsai CUDA Validation](#bonsai-cuda-validation) ·
[Latest Updates](#latest-updates) ·
[Quick Start](#quick-start) ·
[Contributing](#contributing) ·
[Issues](https://github.com/zeraix/imparo/issues)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-38D6B4?style=flat-square)](LICENSE)

</div>

## CUDA MoE on SM86

LFM2.5-8B-A1B Q4_0 now runs through the common MoE workflow, weight owners and CUDA expert routing. Reusing the existing D64/F16/M1 attention kernels raises observed Decode throughput from 32.26/7.21/2.99 to **45.99/41.81/36.35 token/s** at 512/6144/16384 input tokens, each generating 128 tokens. Complete request time falls by **18.23%/32.33%/33.47%**; Prefill has no demonstrated gain. One warmed observation per configuration, not a llama comparison.

Outputs differ between numerical routes. Independent FP64, six fixed answers and a limited long-prefix likelihood check pass. Normal receipted static/dynamic service and state reset pass; final ordinary-service 6144 Decode is **41.80 token/s**. The existing tuner knob selects this qualified route; other models/hardware require their own admission. See [full comparisons, architecture and limits](docs/cuda-moe-attention-2026-09-23.md).

## CUDA CoBatch on SM86

CUDA CoBatch now reuses the common scheduler, KV pool, per-request state owners and existing tuner/layout providers. E4B, LFM2 and Qwen3.5 existing optimized profiles pass final-build-011 admission and normal loading; E4B MTP and LFM2 DSpark join/leave checks preserve complete outputs. Local source/dynamic delivery and GPU error recovery pass. Hardware validation is SM86; Fast rows are supported, Exact rows are explicitly unavailable.

In the final ordinary Qwen two-request check (418 prompt tokens and 128 output tokens each), aggregate output throughput is 40.36→49.97 token/s and complete time is 6.343→5.123 seconds (1.238×), with identical outputs. This is one observation including prefill, not pure decode or a llama comparison. Long E4B and the Qwen middle profile on a short workload have no demonstrated benefit. See [architecture, complete observations and evidence limits](docs/cuda-cobatch-status-2026-09-23.md).

## Qwen3.5-4B Q8 migration validation

Existing-method validation is complete for the reviewed Qwen3.5-4B Q8/F16/SM86 setup: all 65 registered knobs and 14 supplemental families have measured, inherited, inapplicable or parked dispositions. This does not mean every method wins. Reusing the existing F32 GEMM and layout/placement methods reduced complete request time by 19.45% / 21.23% / 29.49% in the earlier 512 / 6144 / 16384 cohort; that three-way llama.cpp b11065 comparison retains its original binary identities.

The final build additionally retains a middle-length private GateUp layout using the original canonical GGUF. In its separate ordinary-service 6144→128 comparison, Prefill is 1755.20→1840.96 token/s, Decode 38.33→38.53 token/s, and request time 6.813→6.640 seconds, with identical output. One cold observation per configuration, without clock locking. Short/long keep the canonical layout. Three receipted profiles and normal loading pass; no new GPU algorithm was added. Independent numerics and the limited 63-token quality contract pass, with original failures retained. See the [complete results and configuration links](docs/qwen35-4b-all-methods-2026-09-21.md) and [coverage closure](docs/qwen35-existing-methods-full-check-2026-09-21.md).

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

## SM86 original runner vs Imparo

Here **original** means the reference runner without Imparo modifications: llama.cpp
for E4B/LFM2 and the official Prism llama.cpp fork for Bonsai's PTQ format. Those
runners retain their own optimizations. This is a comparison between engines,
not a claim that the reference has all optimization disabled, and not the
intermediate Imparo FFN A/B comparison.

### E4B and LFM2

RTX 3060 Laptop 6 GiB; latest available six-row matched comparison from
2026-09-18. Speeds are token/s. Same input token IDs and 256 generated tokens;
both engines enable speculation (E4B MTP, LFM2 DSpark). Two warm samples per
engine in one fixed ABBA sequence; displayed values are arithmetic means.
Imparo `37ff1f39`, llama.cpp `a2878d30` / build10909. These are retained
measurements, **not reruns of the latest compatibility binary**. Cross-engine
output text differs; output equivalence is not claimed.

| Model | Input tokens | Original runner Prefill | Imparo Prefill | Original runner Decode | Imparo Decode | Complete request: original → Imparo (s) |
|---|---:|---:|---:|---:|---:|---:|
| E4B | 512 | 2282.90 | **3181.90** | 96.69 | **132.41** | 2.884 → **2.107** |
| E4B | 6144 | 2117.11 | **3679.82** | 85.75 | **114.05** | 5.899 → **3.923** |
| E4B | 16384 | 1930.89 | **3430.62** | 74.88 | **107.21** | 11.920 → **7.173** |
| LFM2 | 512 | 3642.20 | **4498.64** | 145.58 | **206.21** | 1.905 → **1.375** |
| LFM2 | 6144 | 3463.27 | **4304.80** | 131.78 | **160.83** | 3.732 → **3.028** |
| LFM2 | 16384 | 3256.96 | **4069.69** | 106.78 | **139.04** | 7.443 → **5.884** |

### Bonsai2-27B PTQ — historical v4 comparison

Matched **512 / 6144 / 16384 input IDs and 128 generated tokens**, RTX 3060 Laptop
6 GiB, same PTQ1_0 model file, F16 KV, batch128, no speculation or prompt-cache
hits. One cold full request per engine/length. Earlier v4 receipted Imparo server
`8c75e61c` versus official Prism llama.cpp `b10683-d8f26eec7`, auto GPU fit with
256 MiB margin and 8 CPU threads. Engine-reported PP/Decode timing; client HTTP
latency excludes model loading. These are retained v4-build comparisons,
not the earlier incremental Imparo FFN A/B.

| Input tokens | Original runner Prefill | Imparo Prefill | Original runner Decode | Imparo Decode | Original request (s) | Imparo request (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 91.55 | 123.81 | 1.386 | 6.893 | 97.266 | 22.562 |
| 6144 | 147.40 | 123.24 | 0.861 | 5.844 | 189.219 | 71.609 |
| 16384 | 140.27 | 115.30 | 0.640 | 4.571 | 315.234 | 169.907 |

Both engines retain their own optimizations and placement strategies. Prism's
logs report CPU-assigned layers and disabled Gated Delta fusion for those mixed
placements; this is a complete engine comparison, not an equal-residency kernel
benchmark. Cross-engine output equivalence is not claimed. Slower Prefill cells
are retained; no best-of selection or broad quality claim is made.

[Full comparison, conditions and limitations](docs/bonsai2-official-comparison-2026-09-19.md)
· [Portable inputs, results and logs](evidence/bonsai2-complete-2026-09-19/comparison.json)
· [Earlier short-only baseline and FFN observations](docs/bonsai2-sm86-validation.md)

## Bonsai CUDA validation

**2026-09-19 · PR #14 branch results, not a public-main release.** Target:
Ternary-Bonsai-2-27B PTQ1_0 on RTX 3060 Laptop 6 GiB (SM86), F16 KV,
batch 128 for the historical v4 results below; see the current profile follow-up at the end. **Not every optimization method has been experimentally tested.**
Applicability review, a GPU prototype, full-model validation, and production
admission are different evidence levels; the status below applies only to this model.

| Method / route | What was actually verified |
|---|---|
| PTQ/BF16, Hadamard and grouped basis transforms | Adapted; independent numerical checks and full-model execution. These transforms are required model mathematics. |
| Common placement, reserve, streaming and execution owners | Adapted and exercised in full requests; retains the existing architecture. |
| DeltaNet, causal convolution, snapshots and restore | Adapted; output/state probes and full-model cold/split/reset evidence. |
| HeadNorm/RoPE, last-position head and tail work elimination | Existing routes reused and observed; no isolated gain claimed for each one. |
| Strided copies | GPU bitwise check and complete request A/B. |
| PTQ Tensor Core FFN and non-FFN projections | Independent GPU checks; seven projection geometries and short/medium/long complete requests. M9–128; small-M/head/BF16 routes retain their own implementations. |
| GQA6/D256 Decode and Prefill attention | Independent GPU checks and complete request comparisons; only the documented contiguous F16 domains. |
| KV reservation, tuner/knob and receipt integration | Actual runtime selection and three request lengths verified; opt-in configuration, no global parameter sweep. |
| Registered-host, two-slot weight prefetch | Earlier single-slice long-context result remains parked. New same-budget grouped prefetch plus common spread placement passed three-length full A/B (Decode +41–46%), state/reset checks and formal HostConfig admission. |
| Bounded PTQ→F16/cuBLAS projections | Existing FFN provider extended to seven validated projection geometries; independent FP64 checks, limited quality, three-length full A/B and v4 receipt. FP32 accumulation/output, bounded scratch, SM86/M96–128 only. |
| Triton | Complete FFN small-graph prototype executed and numerical checks passed; register spilling and slower diagnostic timings led to retaining native CUDA. No full-model Triton integration or A/B. |
| TensorRT + Triton | Joint bounded dynamic-weight FFN prototype attempted. TRT found no Cast→MatMul tactic with either4 or32MiB workspace; both failures retained. No running TRT/full-model speed claim. |
| FlashInfer / Marlin | Source/mechanism and compatibility references. Existing native attention paths were adapted; neither library was integrated and benchmarked as a Bonsai PTQ backend. |
| Alternative Q8 TM / W4A16 weight representations | Layout/math/memory applicability reviewed; no Bonsai full-model A/B. Bounded PTQ dequantization above is a separate tested implementation. |
| Shared Hadamard reuse, extra RMS/Add or Delta/conv tail fusion | Reviewed and deferred; no independent Bonsai end-to-end gain measured. |
| Split-K / Stream-K variants | One real M1 complete-FFN four-warp K-split passed independent math but was4.24% slower; parked without a parameter sweep. |
| CUDA Graph / full decoder mega-kernel / Program Pack | Current bounded resident FFN Graph passed changed-input correctness but gained only~0.5% locally, so expansion was parked. Full decoder mega-kernel/Program Pack remain unadapted; no whole-model gain claimed. |
| E4B cross-layer KV reuse / fixed E4B-LFM kernels | Not applicable to the current model semantics/shapes; no misleading cross-model gain claim. |
| DSpark / MTP / tree verification | No compatible Bonsai drafter/state contract; not enabled or benchmarked for this model. |

The [original-runner comparison above](#sm86-original-runner-vs-imparo) lists
fresh same-length absolute speeds. The separate [FFN A/B report](docs/bonsai2-sm86-validation.md)
is an incremental comparison between two already optimized Imparo configurations;
its earlier column must not be called original or unoptimized performance.
The new provider requires source-built CUDA with cuBLAS; driver-only plugins
do not support it.

Admission uses independent numerical checks plus **limited quality non-regression**.
The historical **4/6 absolute-quality failure remains recorded**. The new fixed
64-token NLL check and state checks are not broad or long-context quality certification.
E4B/LFM2 speculation OFF/ON compatibility checks passed on the previously published compatibility build;
those checks are not Bonsai speculation or new cross-engine performance results.

See the [validation report](docs/bonsai2-sm86-validation.md),
[portable results and runtime identities](evidence/bonsai2-sm86-2026-09-19/summary.json),
[new-model selection workflow](docs/new-model-optimization.md), and
[short handoff](handoff/sm86-optimization-toolchain.md).

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

Build Imparo from source with Rust 1.85 or later. On a Mac it needs Apple Silicon and
macOS 15 or later:

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


### Bonsai retained-profile follow-up (2026-09-19)

The existing optimized providers now load through ordinary HostConfig with separate v5 receipts for batch256 and512. The final server is `f5cc3fa0`. This follow-up adds configuration plumbing only: the admitted batch now reaches memory placement before loading, recovering 51/64 resident layers for the short profile. No new GPU kernel or automatic request classifier was added.

| Input tokens | Prefill token/s | Decode token/s | Full request seconds |
|---:|---:|---:|---:|
|512|194.04|6.742|21.500|
|6144|261.92|5.704|45.750|
|16384|251.29|4.352|94.406|

Short/middle/long inputs are512/6144/16384;128 generated tokens each, F16 KV, no speculation or cache hits. Each is one fresh-process observation, with output text/usage identical to the earlier optimized same-profile result. This is ordinary-loader delivery validation, not new algorithmic speedup or a fresh llama comparison. Batch128 was negative in a separate512→512 request and was not adopted; existing tail routes were audited without widening their domain.

Independent raw math replay, same-profile full logits/state/reset and limited fixed continuation NLL pass. Historical4/6 failures remain; no broad quality certification is claimed. Profiles are explicitly selected before load through the existing HostConfig; automatic per-request selection is not implemented. Details and raw evidence: [method-validation ledger](docs/bonsai2-all-methods-validation-2026-09-19.md), [current handoff](handoff/sm86-optimization-toolchain.md).
