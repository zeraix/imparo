# Imparo Triton AOT builder

This directory is development/CI tooling. It is not linked into an Imparo
runtime binary and never runs from Cargo `build.rs`. The formal output is a
data-only Program Pack: canonical manifest, cubin modules, SPDX SBOM, notices,
provenance, and an optional detached Ed25519 signature.

The checked-in lock currently pins the official Triton `release/3.8.x` source
commit `b252c7c4...`. On 2026-08-28, upstream had not published a `v3.8.0` tag or
PyPI wheel, so the lock says `source_pin_pre_release`; it does not claim a
release wheel exists. The source archive, upstream LLVM binary, Linux x86-64
ptxas wheel, CUDA image, Python 3.12.3 image, and build recipe are all
hash-pinned. The Dockerfile builds Triton in a throwaway stage and copies only
its Python package into the final CUDA image; torch is neither installed nor
imported. Image-build Python dependencies are version-and-hash locked in
`requirements-build.lock` and installed with pip `--require-hashes`.
The pinned Triton `setup.py` explicitly translates `MAX_JOBS` into Ninja's
`-j` option, so the image recipe fixes `MAX_JOBS=2` to prevent unconstrained
host CPU detection from exhausting WSL memory. `CMAKE_BUILD_PARALLEL_LEVEL`
alone is insufficient because upstream supplies its own `-j` argument.
The CUDA devel image exposes driver headers through its target-specific include
directory rather than adding that directory to every C++ compile. The recipe
therefore pins `CPATH=/usr/local/cuda/targets/x86_64-linux/include`; without it,
the upstream Proton backend fails closed while compiling `CudaApi.h`.

`builder.base_image_sha256` is only the upstream CUDA base identity. It is never
written into a pack as the final builder identity. Until the Docker image has
been built and `docker image inspect` has supplied its distinct content ID, the
lock remains `source_pin_pre_release`, its final `image_sha256` is null, and
production `build.py` refuses to run even if environment variables are forged.
After a real local image build, its inspected content ID may be recorded while
the status remains `source_pin_pre_release`. Such an image can produce explicit
non-release infrastructure evidence with `--feasibility-source-pin`; formal
release builds remain disabled until an official Triton release pin exists and
the lock is reviewed as `release_ready`.
The temporary lock labels Docker's local `.Id` as
`oci_config_digest_local_feasibility`. A formal lock only accepts
`oci_manifest_digest`, which must identify a pullable registry manifest; the
two digest meanings are never interchangeable.

## Source contract

The source may import only `triton` and `triton.language`. It exports one
`IMPARO_AOT` dictionary with exactly these keys:

```python
IMPARO_AOT = {
    "kernel": "ip_<64 lowercase hex>",
    "signature": ["*fp32:16", "*fp32:16", "i32"],
    "num_warps": 1,
    "num_stages": 1,
    "manifest": {...},
}
```

`manifest` supplies the semantic declaration: pack/channel/version, ABI
ranges, driver/math mode, one complete choice group, and one complete variant
without `variant_id`, `config_id`, `module_id`, `symbol`, or `resources`. The
builder owns those five fields and derives opaque identities from canonical
inputs and the cubin. Pack source cannot override toolchain or target facts.

The adapter calls pinned `ASTSource`, `GPUTarget`, and `triton.compile` APIs. It
rejects non-zero `global_scratch_size` or `profile_scratch_size`. Explicit
engine-owned scratch remains a manifest/contract argument and is independent.

## Build and verify

The controlled Linux image job must set these attestations to their exact lock
values before invoking the builder:

```text
IMPARO_BUILDER_IMAGE_SHA256
IMPARO_TRITON_SOURCE_REVISION
IMPARO_TRITON_LLVM_REVISION
IMPARO_PTXAS_VERSION
IMPARO_CUOBJDUMP_VERSION
```

With the checked-in pre-release source pin, build one exact-SM infrastructure
smoke artifact explicitly:

```text
python tools/triton-pack/build.py --lock tools/triton-pack/toolchain.lock \
  --feasibility-source-pin \
  --target cuda:86:32 --source program-packs/community/cuda/smoke.py \
  --out artifacts/program-packs/sm86-smoke
python tools/triton-pack/verify.py --allow-feasibility-smoke \
  artifacts/program-packs/sm86-smoke/manifest.json
```

This pre-release opt-in is recorded as
`toolchain_status: source_pin_pre_release` and `feasibility_only: true` in
provenance and must not be treated
as release evidence. Verify that a copied image identity really came from
Docker before building:

```text
python tools/triton-pack/lock_verify.py \
  --lock tools/triton-pack/toolchain.lock \
  --builder-image-id sha256:<docker-image-inspect-id>
```

Only infrastructure smoke verification may opt in with
`verify.py --allow-feasibility-smoke`; the default verifier rejects such a
pack. The runtime manifest schema is intentionally unchanged because its Rust
parser denies unknown toolchain fields; feasibility state lives in the hashed
provenance sidecar.

After the reviewed lock becomes `release_ready`, the formal release lane uses
the same commands without `--feasibility-source-pin` and
`--allow-feasibility-smoke`; those default paths remain fail-closed.

Build `cuda:80:32` and `cuda:86:32` in separate clean directories. Compare all
output bytes, not timestamps. The compiler creates and destroys a fresh
Triton/XDG cache for every invocation, so the second build cannot reuse the
first cubin. The manifest and sidecars contain no wall-clock
build time or absolute source path; their only timestamp is the locked upstream
commit epoch used by SPDX.

Signing is a separate release boundary; the private key never enters source or
the builder image:

```text
python tools/triton-pack/sign.py \
  --manifest artifacts/program-packs/sm86-smoke/manifest.json \
  --private-key /secure/community-ed25519.pem \
  --key-id imparo.community.release
python tools/triton-pack/verify.py --require-signature \
  --public-key /secure/community-ed25519-public.pem \
  artifacts/program-packs/sm86-smoke/manifest.json
```

No signing-message file is emitted. The final inventory therefore remains
exactly the files admitted by `imparo-program-pack`.
