<div align="center">

<img src="assets/imparo-wordmark-black.png" alt="Imparo by Zeraix" width="680" />

### Direct-kernel local inference, tuned to the machine it runs on.

<p>
  <a href="#design-rules">Design</a> |
  <a href="#crates">Crates</a> |
  <a href="#correctness-and-determinism">Correctness</a> |
  <a href="#harness-and-tuner">Harness & Tuner</a> |
  <a href="https://github.com/zeraix/Imparo/issues">Issues</a>
</p>

**Imparo is Zeraix's independent, cross-platform inference engine for local agentic workloads.**

Rust everywhere, direct GPU kernels (Metal today, CUDA prepared), no ML-framework
dependencies — not a fork of llama.cpp and free of GGML. Measured against llama.cpp on
the same hardware and models: at or ahead on speed and memory at the shipped
configuration, with byte-stable, run-to-run deterministic output.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-38D6B4?style=flat-square)](LICENSE)

</div>

---

```
              prefill      decode      memory        (gemma4 E4B, M3 Pro, vs llama.cpp fork)
  f16 long    tie          ahead       lighter at rest after CPU levers
  f16 short   ~+10%        ahead       ~-45 MiB
  q4 KV       ~+4%         ~+10%       lighter    <- the app configuration
  q8 KV       ~+5%         ~+8%        lighter
```

## Design rules

Two rules define the layering; everything else is their consequence.

- **A model contributes a plan and a workflow, never a scattered code path.** No
  `if arch == "gemma4"` outside the model layer.
- **A backend contributes kernels, never model knowledge.** No model name in a kernel.

And two principles govern how the code is allowed to grow:

- **Simple is the strength.** Middleware is thin: common abstraction exists only where it
  removes redundant code. There is no graph IR, no scheduler, no runtime fusion machinery —
  a model's workflow is a direct, hand-scheduled function calling a backend trait. Fusion
  happens by hand, where a measurement justified it.
- **Nothing is believed without a measurement.** Every kernel change is gated on pinned
  logits, determinism suites, and interleaved same-session comparisons. Rejected
  experiments are recorded with their numbers so they are not re-attempted on a hunch.

## Crates

```
imparo-gguf       GGUF container: mmap, metadata spans, tensor index, quant kinds
imparo-tokenize   tokenizer (BPE, byte-level); borrows token bytes from the GGUF mmap
imparo-model      THE MODEL LAYER: chat format, plan + workflows
imparo-backend    the Backend trait: one method per op, derived from the real op surface
imparo-metal      Metal backend: MSL kernels + the one sanctioned Obj-C++ host file
imparo-cuda       CUDA backend: .cu kernels + FFI, nvcc-gated (written, unverified on Mac)
imparo-cpu        plain-math CPU backend; the no-GPU fallback path
imparo-kv         unified KV pool: content-addressed units, refcounted sharing (in progress)
imparo-host       host identity, tuned-config store, search-space versioning (per backend)
imparo-tune       auto-tuner: shared sweep machinery + backend-owned knob registries
imparo-server     OpenAI-compatible HTTP server, jinja chat templates, request loop
```

Dependencies point downward only. The model layer meets backends exclusively through the
`Backend` trait; `cfg(target_os)` appears only in `imparo-host`, on whole-crate backend
gates, and at the composition root — the OS is never used to mean "a GPU exists".

## Supported models

| Model | Weights | KV cache | Status |
|---|---|---|---|
| gemma4 E4B (`gemma-4-E4B-it-qat`, GGUF) | Q4_0 + F32 norms | f16 / q4_0 / q8_0 | measured daily against llama.cpp; the reference target |

One model, deliberately: E4B mixes two attention geometries, GQA, shared KV layers,
per-layer embeddings and logit softcapping, so it exercises every axis the contracts
have. The registry and the per-model plan/workflow seam exist for the next model to
prove; a GGUF whose tensor types the engine does not implement is rejected loudly at
load, by tensor name.

## The model contract

A model builder reads GGUF metadata and emits a plan. It allocates nothing and calls no
kernel. The plan carries attention geometry **per layer**, because real models mix them —
gemma4 E4B runs `[W W W W W F] x 7`: windowed layers at head_dim 256 / rope 1e4 / window
512, full layers at head_dim 512 / rope 1e6, 8 query heads over 2 KV heads, 18 layers
sharing KV, a per-layer input embedding, and logit softcapping. One model exercising every
axis is why it was the first target.

Weight quantisation is a **per-tensor property** (`WeightKind`), dispatched through a
type→kernel table; an unsupported kind is rejected loudly at load, by tensor name, never
silently misread.

## The backend contract

`imparo-backend` defines the trait — buffers/arena, matmat, norms, rope, KV store/dequant,
attention, argmax, profiling — and each backend implements it with direct kernels. No
binding libraries (no metal-rs, no cudarc): ideas may be borrowed from candle/wgpu/ggml;
dependencies may not.

When a backend needs to diverge from the shared workflow, it does so at the narrowest
level measurement justifies: (1) inside its trait impl, (2) an ask-the-backend hint,
(3) a per-stage override, (4) a full per-backend workflow as a last resort bought with
numbers. Every divergence is recorded with its measurement.

## Harness and tuner

`imparo-tune` measures what cannot be derived. Every knob declares its category:
shape-derived and arithmetic knobs are computed; device-only knobs come from the
once-per-machine profile (`imparo-metalbench`); only genuinely uncertain knobs are benched.
Backends own their knob registries with independent search-space versions; stored configs
are rejected on version mismatch, and the tuner declines to store when compiled defaults
win. Sweeps are screened at small sizes and duty-cycled so a tuner run never freezes the
host.

## Where the KV pool will sit

The pool goes below the model layer and above the backend: the plan declares per-layer
attention kind and head dimension, the pool turns that into block geometry, the backend
receives a block table it does not interpret. On gemma4, 35 layers hold window-bounded
state and only 7 grow with context — that determination comes from the plan, not from a
model check inside the pool. The disk tier and cross-turn reuse land behind the same seam.

## TODO

What a typical inference engine has that Imparo does not yet. Each fits an existing seam;
none requires re-architecture.

- **Sampling** — only greedy argmax today. Temperature, top-k, top-p, min-p, repetition
  penalties; sits in `imparo-server` above the workflow, with the argmax kernel as its
  fast path.
- **Speculative decoding (MTP / draft models)** — the multi-token verification path is
  already built for it: nb8 handles narrow batches deterministically, so verifying k
  drafted tokens is one GEMM-family pass. Missing: the drafter loop and accept/reject
  logic.
- **Multimodal (mmproj)** — vision tower + projector loading from GGUF, image token
  splicing into the prompt. A second model plan feeding the same workflow seam.
- **Unified KV pool + disk tier** — block-table allocator below the model layer (see
  "Where the KV pool will sit"), cross-turn prefix reuse, cold blocks to disk. In design.
- **Continuous batching** — the server loop is one-request-at-a-time; agentic workloads
  want parallel sub-agent requests sharing the GPU.
- **More models** — gemma4 E4B is the only wired model; the registry and per-model
  plan/workflow contract exist for the next one to prove the seam.
- **More quant types** — weights: F32/Q4_0 only (per-tensor dispatch table is ready for
  more); KV: f16/q4_0/q8_0.
- **CUDA verification** — the backend is written but has never run on a CUDA host; first
  Windows/CUDA bring-up will exercise the tuner's per-backend knob registry for real.
