# Program Pack v1 threat model

Program Packs are untrusted input until fully admitted. Attackers may control transport
archives, manifest bytes, signatures from revoked or unknown keys, module bytes,
filenames, metadata, and concurrent installation attempts. A valid but faulty kernel
may return incorrect values, write out of bounds, hang, exhaust resources, or poison a
CUDA context.

The transport-to-installer handoff is an engine-owned, private, quiescent directory.
Pack bytes are hostile, but an unrelated local process with permission to rename cache
or staging ancestors is outside this boundary; PR-I must create and ACL that directory
before invoking PR-D. PR-D still rejects links/reparse points, hardlinked files, special
files, unexpected directories, and a lock whose handle identity changes during acquire.

A cubin is native GPU code and CUDA provides no in-context memory-safety sandbox for
it. Schema checks, resource attributes, and numerical canaries cannot prove arbitrary
SASS will not read another allocation, write out of bounds, or hang. Therefore a
delegated signing key is a native-code execution trust root, not merely a metadata key.
Adding a downstream/custom publisher key is an explicit user decision to trust that
publisher. Production deployments should isolate inference workers/processes and CUDA
contexts so a fatal pack failure can restart a worker instead of continuing in a
poisoned context.

## Trust boundaries

| Boundary | Trusted responsibility | Untrusted input |
| --- | --- | --- |
| Installer | bounded extraction, hashes, signature, atomic publish | transport/archive and pack files |
| Generic pack crate | strict schema, content identity, key policy | raw manifest/signature bytes |
| CUDA admission | exact SM/driver/ABI/contract and function resources | manifest claims and cubin |
| Tuner | identical workload, CUDA-event timing, numerical gates | candidate performance/outputs |
| Receipt loader | exact fingerprint and route binding | cached profile/receipt |
| Runtime | frozen catalog/module/graph lifetimes and native fallback | selected external function |

The runtime reads only a successfully installed content-addressed directory. It never
extracts a general archive and never executes pack-provided host code.

## Three validation layers

Validation is deliberately split because JSON Schema alone cannot express every safety
property:

1. The raw parser bounds bytes, requires UTF-8, and rejects duplicate JSON keys before
   conversion to ordinary maps.
2. JSON Schema 2020-12 rejects unknown fields and enforces local types, formats, counts,
   and sizes.
3. Semantic validation checks cross references, uniqueness, range ordering, exact SM
   and ABI compatibility, normalized path collisions, launch arithmetic, contract
   authority, and actual CUDA function resources.

Passing one layer never bypasses the others.

## Admission controls

Before a pack becomes a candidate, all of the following must pass:

- bounded raw manifest size and duplicate-key-aware parsing;
- strict schema with unknown fields rejected outside explicit extension lists;
- relative allowlisted paths only, with no absolute/UNC/drive path, parent traversal,
  backslash, alternate data stream, trailing dot/space, symlink/junction/reparse point,
  duplicate normalized path, or Windows case-fold collision;
- declared byte length and lowercase SHA-256 for every immutable file;
- a trusted, non-revoked key and the `imparo-program-pack-v1` domain-separated signed
  message over raw manifest bytes, raw length, and SHA-256;
- exact backend, SM, compatible driver, native ABI, Program Pack ABI, and supported
  operation contract revision and digest;
- module, variant, choice-group, symbol, string, grid, block, shared-memory, scratch,
  expression-depth, and expression-node ceilings;
- actual `CUfunction` resource attributes no greater than admitted limits;
- a separately verified commercial entitlement for each variant-level
  `required_entitlement_features` member; the root feature list must exactly equal the
  union of all variant lists.

An unknown required extension rejects the entire pack. An unknown optional extension
rejects only variants that depend on it. An unknown contract is never scheduled.

## Signature, receipt, and entitlement

These proofs are independent:

- Signature authenticates immutable pack bytes.
- Correctness receipt proves numerical gates for an exact engine/model/device/driver/
  ABI/contract/catalog/eligible-set/route fingerprint.
- Entitlement authorizes a declared commercial feature set.

A valid signature does not make a kernel correct. A receipt does not authenticate the
pack. Entitlement neither authenticates nor proves correctness. Production selection
requires every applicable proof.

Dependency `provides` capabilities and entitlement requirements are separate
namespaces and proofs. A selected kernel can provide a catalog capability but can never
manufacture a license entitlement.

Trust roots use key IDs and support rotation and revocation. Unsigned developer packs
require an explicit compile-time feature plus an explicit developer path; their
profiles and receipts are non-production and cannot be loaded by a production build.

## Installation and rollback

The installer validates into a private temporary directory, fsyncs file bytes (and
directory metadata where the OS exposes it), then atomically renames to a
content-addressed final directory. Windows state-file replacement uses write-through
rename because directory handles cannot be flushed. It never overwrites a known-good
directory in place. Concurrent installers converge on the same
digest or fail without exposing partial data. The previous known-good pack is retained
for rollback. A before/after journal has an explicit durable committed phase, so crash
recovery rolls back only unacknowledged transitions and completes acknowledged ones.
Failed-update quarantine and retry policy remain release-installer work in PR-I.

## CUDA execution and Graph lifetime

Only the CUDA Driver bridge sees cubin bytes. Rust keeps input bytes alive until
synchronous `cuModuleLoadDataEx` returns. Symbols are opaque IDs; the loader does not
infer safety from names. Invocation uses a restricted recipe containing only known
slots, typed manifest constants, overflow-checked integer grid operations, bounded
block/shared-memory values, and engine-owned scratch.

Catalog and bindings freeze before graph capture. A module remains alive until all
using streams synchronize and every graph/graph executable referencing it is destroyed.
Pack install/update/removal, binding change, or profile change requires a quiescence
barrier and graph re-capture. The first release forbids forward-path hot swap.

CUDA illegal access, launch failure, module corruption, or another fatal asynchronous
error poisons the context. The engine reports failure and rebuilds a clean context or
exits; it never continues inference on the poisoned context.

## Privacy, fallback, and negative tests

Inference is offline and telemetry-free by default. Pack selection never contacts a
network. Logs exclude prompts, tokens, model data, private source paths, entitlement
secrets, signatures, and raw proprietary metadata.

The built-in CUDA route is always present. Missing/corrupt/unsigned packs, hash or
identity mismatch, wrong SM/driver/ABI, unsupported contract, expired entitlement,
failed numerical gate, resource excess, or stale receipt disable the affected external
candidate and fall back conservatively.

Required negative tests cover duplicate/unknown fields, invalid UTF-8, oversized or
truncated input, absolute/traversal/symlink/colliding paths, wrong length/hash/signature,
unknown/revoked/rotated keys, dangling or duplicate IDs, wrong SM/driver/ABI/contract,
invalid effects/aliasing, launch overflow/division-by-zero/resource excess, install
races/interruption/rollback, stale identity/receipt, entitlement expiry, graph lifetime,
poisoned context, no-Pack regression, and absence of production Python/PyTorch/Triton
or compiler dependencies.
