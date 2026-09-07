# Program Packs

Status: strict schema, trust, identity, and content-addressed installer implemented.
No GPU module loader, runtime selection, or Triton-produced production kernel is
enabled by this PR.

Program Packs are designed to add optional AOT CUDA implementations without making
Triton or another compiler a production dependency. The implemented runtime will
consume signed data-only cubin plus a strict manifest, measure eligible variants
against built-in CUDA, and retain built-in CUDA as the safe fallback.

## Installed layout and immutable admission

```text
<pack-cache>/v1/objects/<content-digest>/
  manifest.json
  modules/<opaque>.cubin
  SBOM.spdx.json
  THIRD_PARTY_NOTICES
  provenance.json

<pack-cache>/v1/admissions/<install-digest>/
  content.digest
  manifest.sig

<pack-cache>/v1/channels/<channel-id>/
  current.json
  known-good.json
```

The installer, not inference runtime, handles transport containers. It validates into
a temporary directory and atomically publishes content-addressed bytes. Runtime never
parses a general archive or loads pack-provided host DLLs.

The object directory and every admission directory are immutable after publication.
`content.digest` is the lowercase hexadecimal digest naming the referenced object.
`current.json` and `known-good.json` are engine-owned, bounded pointer records written
with fsync plus atomic rename; they are not pack-controlled symlinks and contain only
previously admitted `install-digest` values.

`content.digest` has exactly 64 lowercase ASCII hexadecimal bytes: no BOM, whitespace,
or trailing newline. Each channel pointer is exact canonical UTF-8 JSON with this field
order and no insignificant whitespace or trailing newline:

```json
{"schema":1,"generation":7,"install_digest":"<64 lowercase hex>"}
```

`current.json` and `known-good.json`, when both present, must have the same nonzero
generation. A bounded canonical `transaction.json` records complete before/after pairs
before either pointer changes. After both pointers are durably replaced, the same
journal is durably marked committed before success can be acknowledged. Recovery
restores the complete before pair for an uncommitted journal and the complete after
pair for a committed journal, then removes it; a crash cannot expose a split generation
or roll back an acknowledged update. First install has no known-good pointer; every
later advance retains exactly the previous current (N-1).

An Ed25519 key rotation creates a new admission and therefore a new `install_digest`,
while reusing the unchanged content object. It never rewrites
`objects/<content-digest>` or an existing `manifest.sig`. Revoking the old key makes
the old admission ineligible; the channel pointer may advance only after the new
admission passes current trust policy. N-1 rollback retains previous admissible
install digests, not mutable copies of a signature in a content directory.

## Signature message

`manifest.sig` is a strict JSON envelope defined by
`schemas/program-pack-signature-v1.schema.json`. Its Ed25519 signature covers this
single byte string:

```text
ASCII("imparo-program-pack-v1") || 0x00
|| LE64(raw_manifest_byte_length)
|| SHA256(raw_manifest_bytes)
|| raw_manifest_bytes
```

The parser bounds and validates raw UTF-8/duplicate-key behavior before signature
admission, then checks the envelope length and digest using the exact bytes. Community
and commercial release channels use separately delegated keys. Key IDs, rotation, and
revocation are public trust-policy inputs; private signing keys remain outside source,
artifacts, logs, and inference hosts. Signed bytes are never reformatted or rebuilt.

The signature envelope itself is frozen canonical ASCII JSON: fields appear as
`schema`, `domain`, `algorithm`, `key_id`, `manifest_bytes`, `manifest_sha256`, then
`signature`, with no whitespace, BOM, or newline. Semantically equivalent JSON with a
different byte representation is rejected. This removes signature-envelope aliases
from `install_digest` and makes key rotation deterministic.

PR-D accepts an already unpacked, engine-owned and quiescent staging directory only.
The transport layer must create that directory with private OS permissions and close
all writers before admission; a path writable by an untrusted concurrent local process
is not a valid staging capability. The installer opens each source entry
without following links, hashes and copies through that single handle into a new
installer-owned same-volume stage, syncs copied files, and publishes new immutable
directories without overwriting. It never renames the caller's tree and is not a
general archive parser. Transport, network download, quarantine UX, signed channel
indexes, and commercial entitlement enforcement remain PR-I responsibilities.

## Content identities

All length fields below are little-endian fixed-width integers. Hash bytes are decoded
32-byte SHA-256 values, never hexadecimal text inside the hash message.

```text
contract_digest =
  SHA256(ASCII("imparo-program-contract-v1") || 0x00
         || LE64(contract_bytes_len) || raw_contract_bytes)

choice_group_digest =
  SHA256(ASCII("imparo-program-choice-group-v1") || 0x00
         || LE64(group_id_utf8_len) || UTF8(group_id))
```

`content_digest` excludes `manifest.sig` so delegated-key rotation does not change the
content identity. It covers the following immutable object entries: `manifest.json`, every
declared module, `SBOM.spdx.json`, `THIRD_PARTY_NOTICES`, and `provenance.json`.
Entries are sorted by raw UTF-8 path bytes and encoded:

```text
SHA256(ASCII("imparo-program-pack-content-v1") || 0x00
       || LE32(entry_count)
       || repeated(LE32(path_len) || path_utf8
                   || LE64(file_len) || SHA256(file_bytes)))
```

`install_digest` binds authentication to content without making either directory
mutable:

```text
install_digest =
  SHA256(ASCII("imparo-program-pack-install-v1") || 0x00
         || content_digest_bytes
         || LE64(manifest_sig_len) || SHA256(raw_manifest_sig_bytes))
```

`pack_set_sha256`, `candidate_catalog_sha256`,
`eligible_candidate_set_sha256`, and `selected_route_map_sha256` remain reserved until
PR-G freezes their exact field order, duplicate handling, canonical byte encoding, and
cross-language test vectors. They must use separate domains and length-prefixed fields;
no implementation may persist a profile or receipt using an ad-hoc interpretation
before that coordinated boundary lands.

## Manifest version and producer identity

Program Pack v1 uses a stable SemVer core for release-bearing version strings:
`MAJOR.MINOR.PATCH`, with no prerelease or build suffix. This is intentionally narrower
than the full SemVer grammar; allowing suffixes later requires a schema revision rather
than inconsistent parser behavior.

Program Pack is producer-neutral. Toolchain metadata is a closed, discriminated union
with common immutable builder-image and build-recipe identities. Each producer arm
then carries its own exact identity/version fields: the Triton arm requires pinned
Triton revision/version, Python, and CUDA fields; the CUDA C++ arm binds compiler and
toolkit versions; an external AOT arm binds namespaced producer/revision plus locked
source and toolchain hashes. A non-Triton producer never fabricates Triton metadata.
Adding a producer arm does
not grant runtime code execution: every production output remains data-only cubin plus
the same public contract, admission, tuner, and correctness gates.

## Merged catalog and dependency semantics

Catalog merge is provider-neutral:

- `choice_group_id` is a global namespaced identity. Multiple packs may contribute
  variants to the same group only when contract, workload, cross-check, screening,
  bit-affecting, and joint-group declarations match exactly.
- A variant is addressed as `(content_digest, variant_id)` because `variant_id` is only
  required to be unique inside one pack. A bare variant reference in a manifest is
  pack-local. Persisted route and catalog identities bind the content digest so the
  same opaque ID in another catalog cannot alias it.
- Choice-group and feature references are namespaced catalog identities. A choice
  selects at most one variant from each active group. `joint_with` is a symmetric,
  bounded group relation; every participant must declare the same relationship.
- `requires` must be satisfied by the final selected set. `conflicts` forbids the final
  set from containing the referenced variant, any selected member of the referenced
  group, or the referenced feature. The same reference cannot appear in both lists.
- `provides` contains capability features, not license entitlements. More than one
  alternative variant may provide the same feature. A feature requirement is satisfied
  when at least one selected variant provides it.
- A variant cannot require itself, another variant from its own choice group, or a hard
  dependency cycle. Structurally invalid local references reject the pack; contradictions
  discovered only after merge make the affected candidate component ineligible while
  leaving built-in native fallback available.

No provider, distribution, release channel, or entitlement adds an ordering edge or a
priority. Ranking begins only after the merged eligible graph passes these rules.

## Selection flow

```text
installed bytes
  -> signature/hash/schema/path validation
  -> exact backend/SM/driver/ABI/contract/resource/entitlement admission
  -> catalog and eligible-set identities
  -> identical workload + CUDA event timing
  -> numerical gate and correctness receipt
  -> measured winner or native fallback
  -> catalog freeze before CUDA Graph capture
```

Community and commercial distributions follow the same flow. Commercial candidates
have no priority and win only by passing identical gates and measuring faster. Removing
all packs restores a complete built-in CUDA path.

Each variant declares its own `required_entitlement_features`. The root
`required_entitlement_features` is a redundant, sorted set equal to the exact union of
all variant requirements; admission rejects a missing member, an extra member, or a
duplicate. Community packs require both the root and every variant set to be empty.
Commercial distribution does not itself authorize anything, and a commercial pack may
therefore contain both free and entitled variants. Eligibility filters each variant by
its own namespaced feature set using a separately verified offline entitlement. Both
the root union and the per-variant result enter catalog/eligible-set/receipt identity.
Dependency `provides` features never satisfy an entitlement requirement.

Search budgets and safety ceilings live in versioned `program-pack-policy.toml`; they
are not scattered through workflows or kernels. A manifest can add variants within a
known contract. New semantics require a public contract revision and engine adapter.
Evidence is exact-SM and exact-driver scoped; current local evidence targets SM86 only.

Inference is offline and telemetry-free. Pack download/update is an explicit product
operation outside inference and cannot be triggered by a manifest or kernel.

## Ownership and approval

| Boundary | Accountable role | Required independent approval |
| --- | --- | --- |
| Public engine, ABI, schema, contracts | Imparo engine maintainers | security/ABI reviewer |
| Public export allowlist and leak scan | public-export maintainer | security reviewer |
| Community source and reproducible builder | Community Pack maintainers | engine + license reviewer |
| Private candidate source/build | private Pack maintainers in a separate repository | private security reviewer |
| Signing keys and exact release bytes | release signer/HSM job owner | release manager |
| Apache/Triton/LLVM/CUDA terms | legal/IP approver | human sign-off before release |

Private source/build owners receive only a tagged public SDK/ABI. Public engine and CI
do not know a private repository URL, source path, candidate list, customer profile, or
credential. Named human assignments and branch protection are repository-administration
gates; this specification cannot claim those approvals on their behalf.
