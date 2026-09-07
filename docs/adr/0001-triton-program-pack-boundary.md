# ADR 0001: Triton AOT Program Pack boundary

- Status: accepted for implementation
- Date: 2026-08-28
- Baseline: `origin/v2@364b777af0d229b9222761af62b8b87d66634335`
- Scope: Program Pack v1 and CUDA integration

## Decision

Imparo treats Triton as an optional, build-time candidate generator for CUDA Program
Packs. Triton is not a backend and is never a production runtime dependency. A
production pack contains signed, data-only `cubin` modules, a strictly parsed
manifest, an SBOM, notices, and provenance. It cannot contain or execute host code,
Python, PyTorch, Triton JIT/compiler components, PTX, arbitrary bytecode, or scripts.

The engine always retains its built-in native CUDA route. A pack candidate is merely
another implementation of a public, versioned semantic contract. It may be selected
only after admission, resource validation, numerical gates, and measured tuning. Pack
origin (`community` or `commercial`) cannot alter trust, correctness, or ranking.

Program Packs use a backend-generic `BackendPrograms` surface. CUDA owns module load,
function/resource validation, launch, and graph lifetime; the generic pack crate owns
manifest bytes, content identity, signature policy, and atomic installation. Model
workflows, KV layout/ownership, and Metal behavior remain unchanged.

## Fixed contracts

Only these boundaries are fixed and versioned:

- semantic operation contracts and their legal slots/effects;
- Program Pack wire ABI, manifest schema, and signature message domain;
- engine support for contract revisions;
- receipt and route identity fields;
- bounded policy ceilings and fail-closed fallback rules.

Variant count, tile shape, warp/stage count, fusion choice, thresholds, and winners are
not engine constants. A manifest declares them, centralized policy bounds exploration,
and the tuner chooses from measured eligible candidates. Exact-SM results are separate
evidence domains; SM86 measurements never certify another SM.

## Independent gates

Three proofs are independent and none substitutes for another:

1. Pack signature proves approved immutable bytes.
2. Correctness receipt proves a precise route passed required numerical gates.
3. Entitlement, when present, proves permission to use a commercial feature set.

Missing, expired, mismatched, or revoked proof fails closed to the built-in route.
Pack/driver/SM/ABI/contract/eligible-set/route changes invalidate prior tuning and
receipts. Fatal CUDA errors poison the affected context; execution must not continue
through it.

## Version boundary

The following coordinated version change is reserved and must be activated together
after the wire/schema implementation is complete, never piecemeal:

| Identity | Current | Program Pack boundary |
| --- | ---: | ---: |
| CUDA native ABI | 25 | 26 |
| CUDA tuning space | 24 | 25 |
| CUDA selector | 2 | 3 |
| gate suite | 3 | 4 |
| correctness receipt schema | 3 | 4 |
| Program Pack ABI | - | 1 |
| kernel contract ABI | - | 1 |

This ADR reserves the numbers but does not change runtime constants. A later ABI PR
must change every identity and its tests atomically.

## Repository and release boundary

Public source and schemas are exported by an exact per-file allowlist. Private kernel
source must never enter the public Git history, CI cache, artifact, or log. Community
and commercial packs use the same public ABI/schema/conformance suite. Production
releases publish the exact tested bytes; signed artifacts are not rebuilt afterward.

Legal review of Apache-2.0 project material, Triton MIT material, LLVM components, and
CUDA redistribution terms is a human release gate. Documentation and notices in this
change record the intended boundary but do not claim legal approval.

## Consequences

- Native CUDA remains usable when packs are absent, corrupt, incompatible, or slower.
- The runtime stays offline and telemetry-free by default.
- Adding variants does not require new Rust enums or per-kernel FFI exports.
- Adding new semantics requires a public contract revision and engine adapter.
- CUDA Graph capture may reference a pack module only while its immutable catalog and
  module lifetime are frozen; changing a pack or binding requires a quiescence barrier
  and graph destruction/re-capture.
- No production or commercial Triton kernel work starts until PR-A, PR-B, and PR-C of
  the handoff have passed their respective gates.
