# Phase A2 LAB-A RMSNorm -> Q8_1 MMQ builder evidence

Status: **builder/codegen evidence only**.  RTX 3060 correctness, canary,
sanitizer and interleaved native/Triton timing are still pending.  This file
does not authorize a production Program Pack, ABI, receipt, installer, trust
root, winner, or performance claim.

## Scope and identity

- Branch/starting HEAD: `triton@e53b18a9093f189ea67994b8d00ca7d7e463c7c2`.
- Candidate: `rms_norm_q8_1_mmq`, fixed lab shape `width=2560`, `n_tok=512`.
- Target: exact `cuda:86:32`; no other SM was compiled locally.
- Source: `tools/triton-kernel-lab/rms_q8.py`, SHA-256
  `589f41a0ee77c5941a9bbd373e5bbb51586b147567cc097a5d6a7eeb99e235cc`.
- Triton source pin: commit
  `b252c7c4cea49216b27e5a47db061f9d0dbf49f3` (`3.8.0` source tree).
  There was no official `v3.8.0` tag or PyPI wheel at this evidence point, so
  this remains `source_pin_pre_release` feasibility evidence.
- Python `3.12.3`; Triton `3.8.0`; torch absent; isolated Python flag `1`.
- CUDA tools: ptxas `V12.9.86`; cuobjdump `V12.9.82`.
- Builder semantic recipe SHA-256:
  `5165e28709da883bd44423c6f980726ed5413b89094f3efa76d3c617cbe4efe7`.
- Local Docker image-store/manifest-list ID:
  `sha256:357ea84b8e099d8b1884bde98172feef40a1acd9dc91f6b256dd16245573a1cf`.
- Linux/amd64 platform manifest:
  `sha256:cc77b73970499241bf4e5c9f70a1f4f9571994a7a992937a3dc6827241f0ab76`.
- OCI config digest:
  `sha256:91239d1dae39c4006010a7ca5f2d0c90f342a6c9327053553ee3b1b36c6a4e54`.

The local image-store ID/config digest is not a pullable registry manifest
identity and was not written into the checked-in release lock.

## Frozen lab launch contract

Each variant is a separate cubin/module; no unverified cubin link step is used.
Both use grid `(512,1,1)`, `num_stages=1`, and this argument order:

```text
src *fp32:16, dst *fp32:16, mul *fp32:16,
q8_bytes *i8:16, q8_scales *fp32:16, eps fp32,
global_scratch device-pointer, profile_scratch device-pointer
```

The first six parameters are the visible source ABI.  Triton 3.8 appends the
two scratch pointers as cubin parameter ordinals 6 and 7 even when compiler
metadata reports both allocation sizes as zero.  A Driver launch must pass the
addresses of two zero-valued `CUdeviceptr` variables; truncating the argument
array to six entries caused the initial Windows host access violation inside
`cuLaunchKernel`.

`q8_scales` aliases `q8_bytes + 128` bytes.  A 144-byte `BlockQ8_1Mmq`
record stores `int8 qs[128]` then four fp32 scales.  Its record index is
`(block32/4)*n_tok+tok`.  Stored scales take the native fp16-to-fp32
round-trip; the quantizer uses the unrounded reciprocal, matching native
operation order.  Numerical tolerance remains a hardware-gate decision; no
exact byte-identity claim is made here.

## Clean build results

The command below was run twice into fresh `run-e` and `run-f` directories.
It requires no caller-supplied Triton/CUDA tool environment overrides:

```powershell
wsl.exe -e docker run --rm `
  -v /mnt/d/imparo:/workspace -w /workspace `
  --entrypoint python3 imparo-triton-builder:step4 -I `
  tools/triton-kernel-lab/build.py `
  --out program-packs/lab/rms-q8-sm86/<run> `
  --builder-image-id sha256:357ea84b8e099d8b1884bde98172feef40a1acd9dc91f6b256dd16245573a1cf
```

Both invocations exited `0`.  Cubin bytes were identical:

| Variant | Opaque symbol suffix | Block | Dynamic shared | Registers | Local/hidden scratch | Bytes | SHA-256 | Compile seconds run-e / run-f |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| w4-s1 | `1f773034...918834` | 128 | 2,048 B | 72 | 0 / 0 | 41,584 | `ed52e1d59988e69463b95c4ff4b02828288c489f45efece22b961051b8d3eacf` | 1.389 / 1.476 |
| w8-s1 | `fd28e32e...d0efcc` | 256 | 4,096 B | 48 | 0 / 0 | 27,120 | `3f28b5cdb7c73837642142464368af8df06f819fec0e0a3bf31b9b5da6d9d8c8` | 0.163 / 0.139 |

Artifacts and full metadata:

- `program-packs/lab/rms-q8-sm86/run-e/lab-metadata.json`
- `program-packs/lab/rms-q8-sm86/run-f/lab-metadata.json`

## Bring-up failures retained as evidence

Failures were not counted as passes:

1. `run-a`, exit `1`: final image did not export `TRITON_PTXAS_PATH`; compiler
   reported `Cannot find ptxas`.
2. `run-b`, exit `1`: pinned `ASTSource` did not resolve an entrypoint calling
   a global JIT helper.  Each opaque entrypoint was made self-contained; no
   active-driver binder or cubin linker was introduced.
3. `run-c`, exit `1`: Triton 3.8 requires instantiated global constexprs such
   as `tl.constexpr(4096)`, not annotation-only syntax.
4. `run-d`, exit `1` after successful codegen: a production release verifier
   rejected Triton's lab `.debug_frame`.  LAB-A now applies an independent,
   narrow ELF64/NVIDIA-OSABI/EM_CUDA/exact-SM gate; production release policy
   was not weakened.

Builder bring-up also found that CUDA 12.9.1 ships ptxas `V12.9.86` but
cuobjdump `V12.9.82`.  The Docker identity check and `toolchain.lock` now pin
the real distinct versions rather than masking the mismatch.

## Software gates

- `python -m unittest discover -s tools/triton-pack/tests -v`: 17 passed,
  exit `0`.
- `python -m unittest discover -s tools/triton-kernel-lab/tests -v`: 3 passed,
  exit `0`.
- Docker cached rebuild after final tool-path pinning: exit `0`.

## Pending before any Gate A conclusion

- Load both modules through the existing CUDA Driver bridge in the same
  primary context and exact engine stream.
- Compare against `k_rms_norm_q8_1_mmq<1024>` using the same inputs and
  outputs; check alignment, bounds, canaries and non-finite values.
- Record cold/warm module/launch cost, device memory delta, CUDA-event samples,
  noise and interleaved ABBA/BAAB native/Triton results.
- Run the declared sanitizer gate.
- Do not start a second candidate or production integration until the first
  candidate's correctness result is known.
