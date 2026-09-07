# Program Pack validation plan

Status: implementation gate plan. A row is complete only when its named phase lands
code, negative tests, exact commands, exit codes, and retained evidence. Later-phase
rows are not claims of current support.

| Phase | Accountable reviewer | Required evidence |
| --- | --- | --- |
| PR-A schema/contract | schema + ABI reviewer | strict raw JSON tests, official Draft 2020-12 validator, closed schemas, semantic/engine-contract negatives, raw-byte digest vectors |
| PR-B public boundary | public-export + security reviewer | exact allowlist, fixed-tree export, secret/private-reference scan, cross-platform dry-run |
| PR-C CUDA timing | CUDA runtime reviewer | event lifecycle/stream semantics, nonzero stable timing, DeviceProfile probe tests |
| PR-D parser/trust/install | security reviewer | parser fuzzing, signature/key rotation/revocation, path/race/rollback limits |
| PR-E Driver bridge | CUDA ABI reviewer | adapter/workload-fixture registry match, POD/static layout, Driver resource checks, module/context/Graph lifetime, poisoned-context restart |
| PR-F AOT builder | build/release reviewer | pinned no-PyTorch toolchain, reproducible cubin, no PTX/IR/debug/source paths, SBOM/NOTICE/provenance |
| PR-G tuner/receipt | numerical + tuner reviewer | catalog/eligible/route identity invalidation, exact receipt schema 4, no-Pack/native fallback |
| PR-H Community route | numerical + performance reviewer | real SM-specific logit/decode gates, profiler evidence, end-to-end dispatch proof, native regression |
| PR-I release/install | release + legal reviewer | exact tested bytes, signed channel/rollback, entitlement expiry, offline/no-telemetry, license approval |

## PR-A implemented negative matrix

The dependency-free conformance test currently covers:

- raw UTF-8 decode and duplicate root/nested JSON keys;
- unknown root/nested fields and every declared object being closed;
- schema/backend/version/hash/path/symbol and representative local-bound failures;
- trailing-newline hash bypass and malformed stable SemVer;
- distinct dtype, quantization, and layout domains;
- empty/reversed ABI ranges, duplicate module/group/variant/config IDs;
- extension ID/revision conflicts and community entitlement misuse;
- missing group/module/contract references and group/variant authority disagreement;
- workload ID/parameter mapping and domain-separated parameter digest; exact
  engine-owned fixture admission remains a PR-E gate;
- shape/effect/alias relationships, block/grid/shared-memory ceilings;
- unknown grid slots, divide-by-zero, checked-u64 overflow, and AST depth;
- dependency cycle/self/conflict symmetry and joint-group relationships;
- dense slot/kernel argument ABI, kind/wire/role/access/rank/state matrices;
- contract adapter/raw-byte digest, exact effects and argument bindings;
- dtype/layout/quant narrowing, scratch/resources, graph updates, and numerical class;
- signature envelope constants, fixed-length fields, and canonical Ed25519 base64url.

The PR-A test suite must not claim dependency, alias, entitlement, duplicate-ID, count,
or bound coverage from an unexecuted validator branch. Each claimed rejection has a
named mutation/assertion. The workload canonical-byte vectors in the ABI document are
cross-language fixtures, not illustrative hashes.

The examples are schema/contract fixtures, not installable packs. Their module,
SBOM/NOTICE/provenance, and Ed25519 signature are deliberately not presented as real
release artifacts. PR-D must replace or supplement them with actual sidecar bytes,
trusted test keys, valid signatures, and tamper cases. Structural schema validation is
independently run by the pinned `jsonschema` Draft 2020-12 implementation in addition
to the dependency-free semantic suite.

## Deferred hard gates

PR-D owns oversize-before-parse, truncated JSON, symlink/junction/reparse and normalized
path collisions, actual file length/hash, missing/extra files, install concurrency,
fsync/atomic publish, immutable content/admission directories, key rotation/revocation,
and N-1 admission rollback. PR-E owns engine adapter/workload
fixture registry validation, `CUfunction` attribute verification, launch-time device
limits, graph/module lifetime, and fatal-context handling. PR-F owns exact toolchain
lock and binary scans. PR-G owns the final pack-set/catalog/eligible/route canonical
encodings, cross-language vectors, and identity/receipt invalidation. PR-H owns real
numerical and performance evidence. PR-I owns transport/download, signed channel index,
quarantine UX, and commercial entitlement lifecycle tests.

No deferred item may be used to claim a preceding implementation exists. Failure at
any gate disables the external candidate and preserves or restores the built-in native
route; fatal CUDA failures restart a worker/context before fallback.
