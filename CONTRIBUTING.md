# Contributing to Imparo

Thanks for helping build Imparo. We welcome first-time contributors, experienced
kernel developers, and people who simply want to make the engine work better on
their hardware. Bug reports, documentation fixes, and reproducible measurements
are valuable contributions too.

You do not need an expensive GPU, a complete implementation, or permission to
start a small fix. Open an [issue](https://github.com/zeraix/imparo/issues/new/choose)
if you would like help finding a starting point. English and Chinese are both
welcome; clear examples matter more than polished writing.

## Find your starting point

| Area | Where to look | Examples of useful contributions |
| --- | --- | --- |
| Model support | [imparo-model](crates/imparo-model/src) | Architecture plans, CPU/GPU workflows, model-specific chat formatting. |
| Backend contracts and kernels | [imparo-backend](crates/imparo-backend/src), [imparo-cpu](crates/imparo-cpu/src), [imparo-metal](crates/imparo-metal), [imparo-cuda](crates/imparo-cuda) | Operators, dispatch overhead, memory layout, backend correctness. |
| Hardware and workload tuning | [imparo-host](crates/imparo-host/src), [imparo-tune](crates/imparo-tune/src) | Hardware identification, measured execution choices, reusable tuning configurations. |
| KV and persistent state | [imparo-kv](crates/imparo-kv/src) | Prefix reuse, paged residency, save/restore, branching, eviction, and bounded cache growth. |
| Model input and tokenization | [imparo-gguf](crates/imparo-gguf/src), [imparo-tokenize](crates/imparo-tokenize/src) | Format validation, quantized weights, tokenizer compatibility. |
| Serving and integrations | [imparo-server](crates/imparo-server/src) | HTTP requests, streaming, tool-call formatting, and conversation-state handling. |
| Kernel program packs | [imparo-program-pack](crates/imparo-program-pack/src), [Triton pack tools](tools/triton-pack/README.md) | Manifest validation, trust policy, installation, and backend integration. |
| Tests and documentation | Tests beside the code, [dev_harness](dev_harness), [README](README.md) | Reproductions, clearer setup instructions, and results on additional hardware. |

Apple Silicon with Metal is the primary development and validation path today.
Other backend/model combinations are qualification-specific. A successful build
alone is not evidence of GPU correctness or performance.

## Make a small, reviewable change

1. Fork the repository and create a branch from `main`.
2. For a small fix, go straight to a pull request. For a new architecture, public
   API, state format, or substantial redesign, open an issue first so we can agree
   on the direction and avoid duplicated work.
3. Keep the change focused. A draft PR is welcome when you want early feedback.
4. Explain what changed, why it helps, and what you tested. If you could not run a
   test or do not own the relevant hardware, say so; we can discuss validation.

Documentation-only changes do not need GPU benchmarks. You do not need to fix
unrelated failures, reformat the whole repository, or test every supported device.

## Development and checks

Use stable Rust, at least the version declared in [Cargo.toml](Cargo.toml)
(currently 1.85), with a native compiler/linker for your platform. Metal development
also needs the Apple command-line developer tools; CUDA native builds need the
matching NVIDIA toolkit. See the [CUDA build notes](crates/imparo-cuda/native/README.md)
before working on that backend.

From your checkout, these focused tests need neither model weights nor a GPU:

```sh
cargo test --release --locked -p imparo-backend -p imparo-kv
```

For other changes, run the relevant crate's tests and check the platform jobs in
[CI](.github/workflows/ci.yml). The full-workspace checks used there include:

```sh
cargo build --release --locked
cargo test --release --locked
cargo clippy --release --locked --all-targets -- -D warnings
```

Format changed Rust code with rustfmt, and keep unrelated formatting out of the
PR. See [Quick Start](README.md#quick-start) for server usage with a supported GGUF.

> Public-checkout note, 2026-09-10: full-server compilation at `f0959bf` is blocked
> by missing source files, including `crates/imparo-metal/mega_slots.rs`. The
> focused backend/KV tests above pass at that revision. You are not expected to
> reconstruct missing files or repair this baseline issue in an unrelated PR.
> Include your commit and error if you encounter it; maintainers need to resolve
> the source-sync gap. This note should be removed once a fresh public checkout
> builds successfully.

## Work with the architecture

- **Keep model knowledge in the model layer.** Start with the existing Gemma 4 or
  LFM2 modules: a plan describes shapes and state, workflows execute it, and chat
  formatting stays with the architecture.
- **Keep hardware details in the backend.** Extend shared operation contracts
  when needed, rather than scattering model-name checks through the server or
  introducing a new execution framework for a single kernel. Follow the existing
  architecture-specific Metal phase-list boundary where it applies.
- **Measure tuning choices.** Bind saved configurations to the hardware, backend,
  and numerical/cache settings they actually apply to. Report the tested domain
  instead of assuming one device's best setting is universal.
- **Preserve state semantics.** Changes to KV layout, identity, residency, or
  persistence should cover matching and mismatched configurations, shared
  prefixes, restart/restore, rewind/branch behavior, and failure cleanup as relevant.
- **Keep fallback paths correct.** New fast paths should reject unsupported
  shapes safely and preserve the intended state boundary when retrying.

These are review guides, not a requirement to understand the entire engine before
contributing. Ask when a boundary is unclear.

## Performance changes: speed and correctness together

Please include, as applicable:

- Baseline and candidate commits, hardware/RAM, OS and toolchain/driver versions.
- Model source/revision, quantization, context and output lengths, launch commands,
  and relevant tuning or cache settings.
- Whether timing is client-observed or internal, cold or warm, and whether prefix
  reuse, thinking, or speculative decoding was enabled.
- Repeated before/after measurements under comparable conditions; interleaving
  runs helps separate a real gain from temperature, cache, and timing noise.
- Output/logit or reference checks, plus state-transition tests for affected
  save/restore, shared-cache, or speculative paths. Explain any numerical tolerance.

Label microbenchmarks, compile-only checks, and end-to-end measurements separately.
A preliminary result is welcome; clearly label what remains unverified. Never
upload private prompts, model weights you cannot redistribute, credentials, or
personal cache files just to make a report complete.

## Review, credit, and licensing

We aim for constructive review: explain concerns, help narrow the next step, and
credit contributions in Git history and, where appropriate, release notes. If a
direction does not fit, maintainers should explain why. If you need to pause,
leave a short note so someone else can help continue the work.

Imparo is free and open-source software under [Apache-2.0](LICENSE). Contributions
are made under that same license; **no CLA or copyright transfer is required**.
You retain copyright in your work. Please contribute material you have the right
to share and preserve relevant third-party attribution and license notices.

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md). For a security concern,
use the private reporting channel in [SECURITY.md](SECURITY.md), not a public issue.
