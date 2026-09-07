# Program Pack operation contract ABI v1

This document defines the public semantic boundary between Imparo and data-only CUDA
Program Pack variants. It is independent of Triton: any AOT producer may emit a
conforming cubin. A manifest identifies a contract by namespaced `contract_id`,
integer `revision`, and digest; the engine dispatches only exact revisions it knows.

## CUDA host ABI 26 surface

Program ABI v1 first appears in CUDA backend ABI 26. The complete stable host surface
is exactly these six generic symbols; a pack never adds a DLL/SO export per kernel:

```text
imparo_cuda_program_pack_install
imparo_cuda_program_catalog_identity
imparo_cuda_program_bind
imparo_cuda_program_freeze
imparo_cuda_program_launch
imparo_cuda_program_reset
```

The 64-bit C/Rust wires are versioned and size checked. Their v1 sizes are module 56,
argument descriptor 16, function 200, pack 160, catalog identity 128, launch argument
16, and launch 72 bytes. Every top-level request carries `struct_size` and Program ABI
version 1. Module images and argument-schema arrays are borrowed only for the
synchronous install call; native copies Driver-owned module/function handles and every
descriptor into its registry. Kernel arguments use fixed-width POD storage and the
launch call receives only the declarative grid, shared-memory extent, selected opaque
variant identity, and the already validated ordered arguments.

The bridge initializes and verifies the same CUDA Runtime primary context and stream
used by the existing backend. It loads only authenticated cubin bytes through the CUDA
Driver API, queries actual function resources, freezes before binding, invalidates an
old decode Graph before changing routes, and pins modules until checked Graph teardown
and stream quiescence. `reset` is the only teardown entry point; a failed quiescence or
Graph destroy keeps modules pinned rather than unloading live code. The built-in native
provider remains the safe choice for every admitted choice group and is not a loadable
Program function.

PR-E deliberately reports `identity_ready = false` and zero pack-set, candidate-catalog,
and eligible-set hashes. Their canonical encodings, profile activation, route-map hash,
and receipt schema 4 are PR-G work. A caller must not persist or accept a Program route
while that identity is pending.

## Authority and evolution

- Contract IDs are stable, public, namespaced strings. Semantics never change in place;
  incompatible changes create a new revision.
- The engine-owned registry fixes slots, tensor/state meaning, legal dtypes/layouts,
  read/write/alias effects, numerical classes, scratch ownership, stream behavior, and
  graph rules. Each descriptor binds an immutable compiled engine-adapter ID, revision,
  and raw-byte digest that owns tensor rank/extent/stride, legal byte regions, shape
  relations, operation semantics, and state reset. A pack cannot add authority.
- A variant may narrow shapes, alignment, dtype, layout, and resources but cannot widen
  or reinterpret its contract.
- Unknown IDs, revisions, or digests make a variant ineligible and cause native
  fallback.
- Canonical contract identity is SHA-256 over the tracked UTF-8/LF descriptor bytes,
  with domain `imparo-program-contract-v1` and a fixed-width little-endian byte length.

PR-A proves only that an exact raw contract descriptor and its adapter reference are
bound together. PR-E adds the engine-owned adapter authority check keyed by adapter ID
and revision, compares the declared contract/adapter/workload/effect/resource/Graph
claims to compiled authority, and binds execution to CUDA backend ABI 26. A valid
contract digest without that complete registry match never authorizes a launch.

Manifest choice-group IDs remain auditable namespaced strings. The native ABI receives
their 32-byte identity:

```text
SHA256(ASCII("imparo-program-choice-group-v1") || 0x00
       || LE64(id_utf8_byte_length) || UTF8(id))
```

## Invocation slots and effects

Every slot has a stable name, ABI index, kind, wire type, role, access mode, lifetime,
graph mutability, alignment, and alias policy. Pack v1 supports fixed-width values:

| Kind | Wire value | Meaning |
| --- | --- | --- |
| `tensor_ptr` | device `u64` | Engine-owned CUDA allocation base or view |
| `state_ptr` | device `u64` | Engine-owned mutable contract state |
| `scratch_ptr` | device `u64` | Engine-owned bounded scratch |
| `scalar_i32` | signed 32-bit | Contract-declared scalar |
| `scalar_u32` | unsigned 32-bit | Contract-declared scalar/dimension |
| `scalar_u64` | unsigned 64-bit | Contract-declared size/stride |
| `scalar_f32` | IEEE-754 binary32 bits | Contract-declared scalar |
| `manifest_u32` | unsigned 32-bit | Immutable manifest constant |

Access is `read`, `write`, or `read_write`. Inputs not declared writable remain
bitwise unchanged. Aliasing is `forbidden`, `may_alias`, or `must_alias` against a
named slot. Scratch cannot alias model tensors or persistent state. Manifest argument
bindings must match contract index, kind, width, effects, and dynamic/static status
exactly; missing, extra, duplicate, or reordered bindings are rejected.

All slot ABI indices are dense, zero-based, and serialized in increasing order. Kernel
arguments are likewise dense: required slots appear first in slot order, followed only
by contract-declared bounded `manifest_u32` constants. Rust/C wires are
`repr(C)`/POD and little-endian fixed-width values. Device pointers use 64-bit
`CUdeviceptr` storage; the host `void **kernelParams` array points to that fixed-width
host storage for the duration of `cuLaunchKernel`, never to a Rust reference or a host
address intended as a device pointer. `usize`, `bool`, `String`, `Vec`, C++ objects,
exceptions, and implementation-defined enums never cross the ABI.

Contract validation enforces the complete kind/wire/role/access/lifetime/state matrix.
Tensor/state pointers have a positive rank and nonempty dtype/quant/layout domains;
scalar slots have rank zero and no tensor domain. Scratch is engine-owned, bounded, and
cannot alias model tensors or persistent state. Alias relations are target-unique,
symmetric, and exact-match between manifest and contract.

For v1, `alignment` on a pointer slot is the minimum alignment of the referenced device
address, not the alignment of the host-side kernel-parameter storage. A variant may
require one stronger alignment for a slot; duplicate alignment constraints are invalid.
Scalar parameter storage uses its wire type's natural C alignment.

Role/lifetime combinations are closed in v1: tensor inputs may be `invocation`, `graph`,
or `model`; tensor outputs may be `invocation` or `graph`; mutable state is
`conversation`; scratch is `graph` and capture-static so its address remains stable from warmup until every
captured graph and asynchronous launch is destroyed/synchronized; scalar input/shape
slots are `invocation` or `graph`. `replay_update` is legal only for an argument whose
contract adapter supplies a new value/address before replay; model-lifetime arguments
are capture-static. State initialization and reset are owned by the exact adapter
revision. `state_reset.revision` must equal `engine_adapter.revision`; a reset recipe
from another adapter revision has no authority.

An omitted alias pair means `forbidden`. `may_alias` and `must_alias` are symmetric;
`must_alias` forms a complete transitive equivalence class, and all members must have
compatible adapter-owned extents and satisfy the strongest alignment. A pair cannot
carry more than one mode. Scratch never aliases tensor or state slots. Manifest effects
repeat the complete contract relation exactly; they cannot add authority or omit a
declared exception to the forbidden-by-default rule.

Pack v1 has zero or one scratch slot. `scratch_max_bytes == 0` requires no scratch slot,
no scratch kernel argument, and a variant maximum of zero. A positive maximum requires
exactly one `scratch_ptr`/`scratch`/`read_write` slot and its matching dense kernel
argument. The variant maximum cannot exceed the contract maximum, and its scratch
alignment cannot be weaker than the slot alignment. `zero_initialized=true` means the
engine zeros the full selected scratch extent before every production or tuning
invocation that consumes it, including candidate cross-checks; the reset cost is part
of measurement. The adapter must explicitly permit that requirement. Hidden compiler
global/profile scratch remains forbidden.

## Restricted launch recipe

A recipe is declarative data, not executable host code. It contains the opaque
module/symbol, ordered argument bindings, three constant block dimensions, bounded
dynamic shared memory, and three grid expressions. Arguments may reference a declared
slot, a typed manifest constant, or engine scratch; host pointers, raw addresses,
pointer arithmetic, strings, paths, environment access, callbacks, and bytecode are
forbidden.

Grid/shared-memory expressions are depth- and node-bounded typed ASTs. Leaves are
unsigned constants or declared unsigned scalar/dimension slots. Operators are checked
`add`, `mul`, `ceil_div`, `min`, and `max`. Every intermediate uses checked
`u64`; division by zero, overflow, zero grid dimension, truncation, or policy/device
limit excess rejects the candidate. Block dimensions are nonzero and their checked
product cannot exceed manifest policy, device, or function limits.

The CUDA bridge verifies register count, static/dynamic shared memory, maximum threads,
and other required attributes from the loaded `CUfunction`. Manifest resource values
are ceilings, not trusted observations. Hidden global/profile scratch is forbidden;
all workspace is bounded, engine-allocated, aligned, and passed through a declared
scratch slot.

## Numerical and graph classes

The descriptor maps directly to `imparo_backend::numerical::NumericalClass`:

- `bit_exact`: nonempty reference plus nonzero contract version;
- `gate_bounded`: nonempty gate-suite ID plus nonzero contract version;
- `diagnostic_only`: development only and never receipt-admissible.

Determinism is `required` or `not_required`. Bit-affecting routes require explicit
gates and an exact receipt. `graph_capture` is `forbidden`, `capture_only`, or
`replay_update_safe`. Replay-safe variants list all updateable argument slots and
must match contract graph mutability. A graph cannot outlive its module or frozen
catalog.

Program ABI v1 has a closed replay-update source registry. Source `0` means no update
and is required for capture-static arguments. Source `1` is the engine-owned
`decode_start_pos_u32` value; it is legal only on a `scalar_u32` argument declared
`replay_update`. It is not a manifest slot index or callback. Every other source/kind
combination fails closed until a later engine adapter and ABI contract define it.

## Workload ABI

Choice groups refer to a revisioned public workload ID, a bounded parameter object, a
domain-separated parameter digest, and an invocation-fixture digest. Revision 1 maps
exactly to the existing backend workload enum:

| Manifest ID | Backend `Workload` |
| --- | --- |
| `imparo.workload.decode_mix` | `DecodeMix` |
| `imparo.workload.narrow_mix` | `NarrowMix(narrow_tokens)` |
| `imparo.workload.attention_decode` | `AttentionDecode` |
| `imparo.workload.attention_decode_deep` | `AttentionDecodeDeep` |
| `imparo.workload.attention_prefill` | `AttentionPrefill` |
| `imparo.workload.attention_prefill_deep` | `AttentionPrefillDeep` |
| `imparo.workload.decode_attention_step` | `DecodeAttentionStep` |
| `imparo.workload.attention_prefill_reuse` | `AttentionPrefillReuse` |
| `imparo.workload.prefill_gemm` | `PrefillGemm` |

Only `narrow_mix` accepts `narrow_tokens`; all other revision-1 parameter objects are
empty. The engine contract allowlists exact `(workload_id, revision, fixture_sha256)`
tuples, and the fixture digest binds the exact contract invocation used for
screening/cross-checking so tuning cannot benchmark
a route that never dispatches the candidate.

Revision-1 parameter bytes are canonical ASCII, not generic implementation-dependent
JSON serialization:

```text
parameterless workload:       {}
narrow_mix with value N:      {"narrow_tokens":N}
```

There is no whitespace, key reordering, sign, exponent, or leading zero; `N` is the
base-10 rendering of a positive `u32`. The digest is:

```text
SHA256(ASCII("imparo-program-workload-parameters-v1") || 0x00
       || LE64(parameter_byte_length) || parameter_bytes)
```

Normative test vectors:

| Parameter bytes | Length | SHA-256 |
| --- | ---: | --- |
| `{}` | 2 | `1aa54ff326d23ef77b0dd6db032f335ae5d32710cff814846b28b7a2fc3ad351` |
| `{"narrow_tokens":8}` | 19 | `5ba74de335a289f64aa0c79cbb0a3600dbf49b43114ebf89648dc7593039368a` |
| `{"narrow_tokens":16}` | 20 | `c74fbccbd835cb491305945cdd29e3eb212e7b5e910bb46c678ad84645ebc411` |

The workload fixture is also engine-owned. Each allowed workload reference binds ID,
revision, and fixture digest in the contract/adapter registry. A pack-provided arbitrary
64-hex value is not authority. PR-E verifies that adapter/fixture authority before
native installation; PR-G must prove instrumentation observed dispatch of the bound
variant before accepting its timing.

## Candidate and route identity

`variant_id` and `config_id` are opaque lowercase 32-byte identities represented as
64 hex characters. The engine does not infer algorithm meaning from them. Candidate
identity also binds manifest/module digests, contract ID/revision/digest, constraints,
effects, launch recipe, resource claims, numerical/graph classes, and config ID.

Tuning and receipt identity additionally bind model/KV layout, backend artifact, native
ABI, Program Pack ABI, contract ABI, CUDA driver, exact SM, pack set, candidate catalog,
eligible candidate set, selected route map, math mode, and applicable entitlement
features. Distribution/provider never changes trust or ranking.

The exact pack-set, candidate-catalog, eligible-set, and selected-route encodings are
reserved until PR-G. No PR-A fixture or pre-PR-G profile is evidence that those
identities have been activated.
