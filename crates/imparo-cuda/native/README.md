# CUDA native kernel layers

`imparo_cuda.cu` is the stable backend boundary. It owns ABI validation, memory
policy, generic fallbacks, and dispatch by tensor geometry. Architecture-specific
code lives below `smXX/` and must not leak model names or workflow decisions into
the backend.

## Layering contract

1. The Rust workflow calls semantic backend operations only. Metal keeps the default
   implementation of every new semantic hook unless its own implementation is changed
   and tested separately.
2. `imparo_cuda.cu` validates an operation and selects a supported geometry. Every
   specialized route must retain a correct common-kernel fallback.
3. `smXX/` owns instruction choice, tile geometry, and architecture-specific numerical
   ordering. A newer SM family may reuse an older family only when the catalog declares
   that compatibility.
4. Temporary memory is bounded and reusable. Its size must be included in
   `imparo_cuda_memory_info`; allocation failure reduces the tile chunk or falls back
   instead of making model loading fail.
5. A specialization remains explicitly gated until it passes the external logit gate
   and the workload speed gate. Diagnostic experiments do not become release defaults.

## Adding a kernel

- Dispatch on semantic shape and data type, not a model filename.
- Put the implementation in the lowest compatible SM family directory.
- Document layout, accumulator precision, and reduction order beside the kernel.
- Test the generic fallback, the specialized path, bounded-workspace chunking, and an
  allocation-failure fallback.
- Run the pinned llama/FA logit gate before performance measurement, then report both
  cold and steady-state timings.

The release workflow compiles the same boundary once per catalog SM target. Each DLL is
independently downloadable and ABI-checked; there is no all-SM fat package.

## Data-only Program module bridge

CUDA backend ABI 26 adds Program ABI v1 without adding another backend or changing the
model workflow. `program_pack.h` freezes fixed-width C wires and `program_pack.cu` is
included into the existing `imparo_cuda.cu` translation unit, so Program modules share
the backend's selected device, retained primary context, one stream, error state, and
decode-Graph owner.

The frozen 64-bit Program ABI v1 wire sizes are module 56, argument descriptor 16,
function 200, pack 160, catalog identity 128, launch argument 16, and launch 72 bytes.
The only replay-update source currently defined is engine-owned source 1,
`decode_start_pos_u32`, and it is valid only for a replay-update `scalar_u32` argument.
Source 0 is required for capture-static arguments; all other source/kind combinations
fail closed.

The public DLL/SO surface is exactly six generic functions:

```text
imparo_cuda_program_pack_install
imparo_cuda_program_catalog_identity
imparo_cuda_program_bind
imparo_cuda_program_freeze
imparo_cuda_program_launch
imparo_cuda_program_reset
```

There are no pack-specific or kernel-specific exports. Installation accepts only
already authenticated, owned cubin bytes from `imparo-program-pack`; native loads them
with the CUDA Driver API, resolves opaque symbols, and checks actual registers, static
and dynamic shared memory, local memory, and thread limits against both manifest and
device ceilings. Pack DLL/SO, PTX, Python, callbacks, and host code are not executable
inputs to this bridge.

The catalog is mutable only before freeze. Binding a different route first destroys an
old decode Graph using the existing checked teardown; launches use the existing stream.
Reset synchronizes, destroys Graph state, then unloads modules. If safe teardown cannot
be proven, reset fails and leaves modules pinned. Every choice group exposes a built-in
native fallback marker which can be selected but never launched as a Driver function.
Exact target SM, CUDA driver minimum, backend ABI range, math mode, compiled contract
authority, launch slots/effects, and resource ceilings all fail closed.

PR-E keeps catalog identity hashes zero and `identity_ready=false`. Under the current
R2 Gate A No-Go, this surface remains permanently dormant for this evaluation: do not
execute the old PR-G or interpret the ABI 26 fields as an active receipted Program
route. Only new evidence that first reopens Gate A as Go and then passes Gate B may
activate these identities and issue receipts, through a new reviewed PR.

## Runtime identity contract

`imparo_cuda_runtime_identity` is available before model weights are initialized and
uses the same `IMPARO_CUDA_DEVICE` selection as backend initialization. Its ABI-versioned
wire value reports the selected device UUID and SM, CUDA driver/runtime versions, the
backend ABI, and a semantic native build SHA-256. The build identity is produced by the
shared Cargo/release helper from sorted native sources plus the target, SM set, math mode,
compiler identity, ABI, and build/catalog inputs. It must never be replaced by a hash of
the common server or tuner executable.

Dynamic runtimes additionally hash the exact resolved DLL/SO. Correctness receipts use
that artifact hash when present and otherwise use the semantic build hash embedded by a
static build, so independently built tuner and server binaries can still agree on the
same native backend. A change to the identity wire, exported identity function, or its
meaning requires a backend ABI bump. Missing or incomplete identity is an error; callers
must fall back to safe, untuned defaults instead of weakening the receipt fingerprint.

## Decode MMVQ

SM80+ owns compile-time 2/4/8-warp specializations for single-token Q4xQ8 projection,
alongside its gated and paired forms. The common dispatcher chooses them only for the
resident Q4, one-token geometry and retains the runtime-warp kernel behind
`IMPARO_CUDA_NO_SM80_DECODE_MMVQ=1`. A specialization must preserve the generic path's
weight-block visitation, ascending cross-warp combine, and XOR reduction order; changing
any of those is a new numerical route and requires the recurrent external gate.

Single-token projection policy is workload-shaped rather than model-shaped. General-K
and narrow-K projections have independent warp knobs, with a tuned input-width boundary;
narrow projections may additionally group 1/2/4 adjacent output rows in one CTA. The
group shares Q8_1 activation traversal but gives every row the same accumulation and
reduction order as the one-row kernel. These choices are registered as
`gemv_warps`, `narrow_gemv_warps`, `narrow_gemv_max_width`, and
`narrow_gemv_rows_per_cta`; do not replace the boundary with a PLE or model-width test.
Gated and paired semantic kernels retain their own one-row scheduling.

## Quantized KV expansion

SM80+ expands staged Q4_0/Q8_0 KV with one lane per output value. The generic fallback
assigns one quantized block to one thread and serializes 32 converts/stores; it remains
available through `IMPARO_CUDA_NO_PARALLEL_KV_DEQUANT=1`. The parallel kernel must retain
the exact packed-byte interpretation and per-value f32-to-half rounding. It is bounded by
the existing shared scratch and never creates a persistent f16 mirror of the KV cache.

## Prefill attention

SM80 fused prefill attention keeps D256 and D512 as separate architecture kernels. On
ringed D256 layers, the ring argument is a power-of-two mask supplied by the common
workflow; absolute key cycles must therefore be recovered with that mask, never an
integer modulus or division. The ringed specialization folds causal and lower-window
bounds into one unsigned-age comparison, while a non-ringed instantiation preserves the
general fallback.

Each D256 warp publishes and consumes its own 16-column Q fragment through shared memory.
That ownership permits warp synchronization during Q transposition; CTA barriers remain
mandatory for K/V tiles because their async copies are shared across warps. These are
architecture-local scheduling choices: cache allocation, ring sizing, Metal, and the
backend attention contract remain unchanged.

D512 has a different ownership graph: four warps consume each disjoint 16-key K/V
partition, while one even/odd warp pair owns each 16-column output tile. Its async cache
copies and accumulator combine therefore use named 128-thread partition barriers and
64-thread column-pair barriers respectively. Do not widen those barriers back to the CTA
unless a future kernel introduces a real cross-partition dependency. Each column pair
also publishes its own Q rows before entering the same 64-thread barrier. Barrier IDs are
register operands derived from partition/pair ownership; do not reintroduce a hot-loop
branch or switch merely to select a constant PTX barrier number.

The staged prefill score kernel shares each K tile across the column warps in its
partition. Producer count must be derived from `query_tokens`: an 8-query medium tile
has two column warps per partition, while a 16-query wide tile has four. Each geometry
must cooperatively publish all 32 aligned sectors before the CTA barrier. Assuming four
producers left half of medium K shared memory uninitialized and caused long-context
run-to-run nondeterminism; never encode one launch geometry into this shared ownership.

The small-query D512 path consumes K/V in aligned adjacent pairs. Its cache accessor
therefore loads one f16 pair directly, or shares the Q4/Q8 block address and scale across
the two independently rounded values. Callers must preserve the even-index contract;
scalar or unaligned consumers belong on `cache_half` instead. The following partial
softmaxes are independent warp reductions. SM80+ packs several reductions into each CTA
to avoid the architectural resident-block limit leaving warp slots empty. The packing
width is the benched `attn_softmax_warps` knob (1/2/4/8, default 2), not workflow policy;
new architectures may tune it without changing attention semantics or Metal.

`attn_stream_part_cap` bounds the topology-derived small-query Stream-K partition count
(4/8/12/16/24/32, default 8). It balances parallel partial-softmax work against the
value kernel's per-part fixup; it is a shape/tuner decision, not a device-name check.
Because partition seams are numerically visible, every candidate must clear recurrent
agreement and determinism gates in addition to its end-to-end timing.

The value stage loads a probability tile before issuing the V-cache loads, preventing
the two unrelated global streams from being interleaved by the per-thread staging loop.
Column rescale and partial max/sum metadata are staged once per CTA because every output
row reuses them. Direct Q4 values use one aligned 16-bit packed load per adjacent pair
and stage the block scale once per key row; loading the next scale is folded into the
existing post-MMA warp barrier. `attn_value_tiles` can reuse one probability tile across
one or two adjacent 16-value output tiles. One tile is the portable default: two increases live
accumulator state and lost materially on SM86, but remains a tuner candidate for later
architectures with a different register/occupancy balance. `IMPARO_CUDA_PROFILE_ATTN_STAGES=1`
reports `small-scores`, `small-softmax`, and `small-values`; disable decode graphs while
using the synchronizing diagnostic profiler.

Large reductions use a 1024-thread argmax when the vocabulary has at least 65536 entries;
`IMPARO_CUDA_NO_WIDE_ARGMAX=1` retains the 256-thread diagnostic path. Both forms enforce
the smallest-index-at-maximum rule.

Q4 target verification reuses the same four-token small-attention ownership tile. Wider
causal chains are covered by consecutive tiles with matching logical-position and
Q/output-row offsets; this preserves the architecture kernel's 16-column register shape
while reducing an eight-row verifier from eight one-row attention schedules to two.
The DLL reports both the eight-row capability and this four-row physical tile to the
common scheduler. The prompt-lookup proposer raises continuation support only when a
proposal would cross the reported tile boundary, keeping architecture shape out of the
server and allowing another SM family to publish a different tile independently.
`IMPARO_CUDA_NO_Q4_VERIFY_GROUPED=1` retains the sequential diagnostic path.

## MMQ token tails

SM80+ keeps the 128-token MMQ ownership grid and numerical K order, but bounds activation
staging to the valid portion of the final token tile. A warp may skip tensor-core work
only when its complete 8-token fragment is outside that valid tail. This avoids coupling
the common workflow to an architecture tile size and keeps Stream-K seams stable.
`IMPARO_CUDA_NO_MMQ_TAIL_TRIM=1` restores the padded zero-work path for A/B localization.
Do not use the fallback as a release mode; both paths must remain output-identical.

## Packed Q4 staging

SM80 MMQ maps four unsigned Q4 nibbles to signed bytes with one carry-free packed
transform before `mma.m16n8k32`. Bits 0--2 pass through, while inverted bit 3 is
multiplied by 31 to fill bits 3--7; the per-byte product is at most 248, so no carry
may cross a byte lane. Keep this operation in the architecture layer and prove any
replacement over all 16 nibble values. It is part of tensor-core staging, not a
license to route the common workflow through a generic DP4A fallback.

## D64 wide prefill

SM80+ D64/GQA4 prefill batches of 9--16 tokens use one native 64-column
attention family instead of chaining four-token score, softmax and value launches.
Four warps own four independent 16-column fragments, while each CTA owns one
64-key partition. A bounded second pass combines partitions from last to first.
Every key tile has distinct shared probability storage; reusing one tile across
the four PV updates is a correctness bug.

Workspace is chunked by KV head. Allocation pressure reduces the number of heads
processed together, and failure to stage even one head returns to the established
small-query chain rather than making an otherwise fitting model fail. The general
budget remains `IMPARO_CUDA_ATTN_WORKSPACE_MIB`; the shape-specific override is
`IMPARO_CUDA_ATTN_D64_WORKSPACE_MIB`. `IMPARO_CUDA_NO_ATTN_D64_WIDE=1` selects the
fallback for numerical and performance A/B. The route consumes the public float-Q
ABI and applies its supplied QK scale; it does not depend on LFM2 preprocessing.

## Projection-ready activations

The backend contract exposes semantic hooks for normalized, transformed activations that
immediately feed a quantized projection. CUDA may use those hooks to publish the common
f32 result and its reusable MMQ Q8 layout in one kernel; Metal and other backends retain
the ordinary operation sequence through the trait defaults. The fused producer must use
the same reduction and output arithmetic as the unfused path, and the Q8 cache is valid
only for the exact buffer epoch, row span, width, and layout it produced.

The SM80+ 64-point FWHT producer is gated by
`IMPARO_CUDA_NO_HADAMARD_Q8=1`. Post-FFW RMS/add activations that semantically feed the
per-layer-embedding gate projection are gated by `IMPARO_CUDA_NO_RMS_ADD_Q8=1`. These
switches are diagnostic oracles; workflow code must select the semantic hook from tensor
flow, never from a model filename or CUDA tile detail.

The PLE projection hook also lets SM80+ combine its gate GELU, token-specific scaling,
and group-major MMQ Q8 production without adding a workflow operation. It is gated by
`IMPARO_CUDA_NO_PLE_GATE_Q8=1`; non-Q4 projections, small MMVQ batches, and unsupported
geometries retain the explicit elementwise and quantization sequence.

The PLE norm/gather semantic hook owns both surrounding model scales as well as RMSNorm
and the selected Q4 table row. CUDA applies the input scale with an explicit rounded
multiply inside the fused kernel and applies the combine scale at publication, removing
two full activation passes while the Backend default retains the original operation
sequence for Metal. Widths no larger than the CTA keep their single owned value in a
register across the reduction barrier; wider PLE rows use the general strided loop.
Changing this native signature requires a CUDA backend ABI bump so the common runtime
cannot pair with an older independently downloaded SM library.

For tiled attention, CUDA's fused head postprocess may publish an epoch- and
shape-versioned FP16 Q mirror beside the public f32 output. Attention reuses it only when
the requested QK scale is exactly one; arbitrary scales and cache misses retain the
standalone scaled conversion. `IMPARO_CUDA_NO_HEAD_Q_CACHE=1` is the diagnostic fallback.
The mirror is a bounded optional allocation, included in memory reporting, and allocation
failure must never make model fit fail.

## Decode execution graph

Whole-forward CUDA Graph replay is a backend capability, not a second model workflow.
The common workflow asks the backend to replay and otherwise submits its normal semantic
operations; Metal and other backends use that fallback by default. CUDA captures only a
stable one-token argmax shape, keeps host-originated inputs in reusable pinned storage,
and updates position-dependent kernel parameters within a validated attention schedule
bucket. If any dynamic node is unrecognized, memory moves, or geometry changes, the graph
is discarded and the semantic workflow remains authoritative. New CUDA kernels whose
arguments depend on token position must therefore be registered in the graph matcher and
included in its expected-node count before replay can remain enabled.

The production replay uploads one pinned `u32` position control value per token. Every
registered position-dependent kernel derives its start/valid span from that value;
`IMPARO_CUDA_GRAPH_NODE_UPDATES=1` retains the slower per-node update path as a diagnostic
oracle. The control buffer is allocated lazily only after a warm decode has published
stable input descriptors and a capturable attention schedule.

Resident token embeddings use the batched `rows` operation even for one-token decode.
That path registers a token-only replay descriptor: the graph updates the pinned token
id read by the resident table kernel and must not copy an embedding row into the paged
weight cache. Streamed PLE rows retain their separate pinned staging descriptor.

Unsaturated sliding-window attention is graph-safe only within one physical 32-key
schedule bucket. The runtime intersects that range with every full-attention schedule,
replays inside it, and automatically discards/recaptures at a boundary. A saturated ring
accepts only positions at or above its mask, preventing a graph from an old long request
being reused for a new short conversation. `IMPARO_CUDA_NO_RING_GRAPH_BUCKETS=1` restores
the conservative wait-until-saturation policy for diagnostics.

## Kernel contract probes

Small instruction-level contracts live in `native/tests/` and run without a GGUF file.
They are intentionally separate from the backend ABI and release DLL. The maintained
SM86 validation entry points are the Rust/native contract suites:

```text
cargo test -p imparo-cuda --features cuda-static --tests --locked
cargo test -p imparo-cuda --features cuda-dynamic --tests --locked
```

On Windows, run the static command from a Visual Studio developer shell so `nvcc` can
find the MSVC host compiler, and set `IMPARO_CUDA_ARCHS=86` for this hardware gate.
The release workflow may separately compile every catalog SM DLL, but compile coverage
is not hardware correctness or performance evidence. This handoff validates only SM86.

## KV arena, staging and page-table state (ABI 23)

CUDA KV storage is one transactionally replaced device arena. Every layer's K and V
pointer names a stable, aligned slice; growth prepares a fresh arena, preserves exact
logical bytes with queued D2D copies, synchronizes once, and only then swaps all slice
pointers. Allocation or copy failure leaves the old arena and ownership metadata live.
`allocated_bytes` reports the actual aligned arena allocation once, while
`imparo_cuda_kv_live_bytes` reports logical pool ownership separately.

Ownership is not tied to a CUDA page size. The first advise for each `(layer, K/V)`
side establishes its runtime block quantum, and later transitions require that exact
length and aligned fixed offset. `NeverClaimed -> Live -> Free -> Live` is explicit;
double-free/double-reuse and transfer through `Free` fail closed. Contiguous legacy
access may use `NeverClaimed`, so enabling the arena does not change non-pool behavior.
H2D and D2H restore use a capacity-growing three-slot pinned ring derived from the
actual transfer length; no model/unit byte size is compiled in.

The ownership/layout state test requires no CUDA device:
`cl /nologo /std:c++17 /EHsc native\tests\kv_memory_test.cpp /Fe:kv_memory_test.exe`.

ABI 22 adds an explicit `PagingLayout` for every layer: logical slots and independent
K/V byte strides come from Rust and are never inferred from padded allocation bytes. The
native backend allocates one device page-table arena only at KV alloc/grow, identity-fills
each layer, preserves installed prefixes across growth and extends the tail with identity
entries. `imparo_cuda_set_kv_pages` only updates the preallocated arena; it rejects short
capacity overrun and out-of-range entries, while an empty table explicitly restores the
identity mapping. Raw KV byte read/write APIs remain physical and unchanged.

Step 8.3-B gives f16/Q4/Q8 KV stores and both generic/SM80 quantized-dequant paths one
shared logical-row mapping: a nonzero ring mask wins; otherwise the stable per-layer
device table maps 64-row block IDs, with identity as the no-table fallback. Store graph
arguments append that stable pointer without changing the existing start/ring slots.
Step 8.3-C applies the same mapping exactly once when generic, decode, small-query,
staged-prefill, flash and Stream-K attention routes form K/V row addresses. Masks,
workspace indices and schedules remain logical; ring addressing still wins. The
identity route retains its pre-paging kernel symbols and code generation; dedicated
paged symbols carry the appended table argument. Full-model identity Graph/default,
Graph-off and node-update gates, plus the five-layout cold/resume scramble matrix, pass.
Step 8.3-D therefore advertised `paged_reads=true` while CUDA still advertised only
`[Device]`. That sentence is historical: completed Step 8.4 and ABI 23 added explicit
Host transfers, and current CUDA advertises the complete `[Device, Host, Disk]` ladder.

ABI 23 adds backend-owned persistent pinned Host buffers referenced only by
generation-safe opaque handles. Demotion and promotion accept spans from multiple
handles, enqueue the complete batch, and synchronize the CUDA stream once. They require
live device ranges but never change lifecycle state: the common pool remains the sole
authority for `Reuse` and `Free`. Host allocation, access, transfer and profiling reject
graph capture. The native Host profile reports available RAM and measured pinned H2D/D2H
bandwidth facts, using zero for unknown measurements; Host-tier sizing, admission and
fallback policy remain common-runtime decisions.
## LFM2 numerical operations (ABI 24)

ABI 24 adds standalone SiLU/SiLU-multiply and LFM2 short-convolution entry points.
CUDA advertises fused `None` and `GELU` epilogues only; a fused SiLU request is rejected
through the pending-error channel rather than being treated as another nonzero GELU
selector. The model workflow consults this capability and routes CUDA SiLU through the
standalone kernels. CPU and Metal keep their established SiLU fusion.

`shortconv` consumes token-major `(b,c,x)`, channel-major weights with oldest tap first,
and time-major state. Its output and state commit are separate stream-ordered dispatches.
The commit has one thread per channel and advances slots in ascending order, which makes
the `state -> state` update race-free when a boundary is shorter than history. Snapshot
uses the same state kernel with a distinct destination and must be queued before commit.
All shape arithmetic, buffer extents, aliases and weight ranges fail closed; F32
convolution weights use `weight_slice`, so resident and paged-weight modes share one
path. No kernel-history ceiling is compiled in.

The existing row and batched-row exports also gain a `wkind` argument in ABI 24. Q4
remains implemented here; other kinds fail closed until their typed kernels land, without
requiring another ABI change.

## Current release contract (ABI 26)

ABI 26 is the authoritative CUDA DLL contract. It retains paging, persistent Host
transfer/profile surfaces and the LFM2 exports, and adds the versioned Prefill-cache/
FFN transaction surface. Its wire layout is enforced by the static and dynamic
release-contract tests. The current SM86 tuning identity is space 30, selector 4 and
correctness gate-suite 7. Older ABI/space/selector receipts, including the tracked
Step 9 ABI-25 review pair, remain historical evidence and fail closed under the current
loader. The current E4B exact-128 candidate and receipt are under
`docs/evidence/cuda-sm86-e4b-open-source/exact128-route5-graph-v30/`.

The Program bridge declarations share ABI 26 but remain dormant with
`identity_ready=false` after the R2 Gate A No-Go. They do not reinterpret the
historical ABI-25 Step 9 receipt or the E4B native receipt as Program-route authority.
Activating a Program candidate still requires Gate A to be reopened as Go, a successful
Gate B, and a separately reviewed identity/receipt change.

The SM86 default enables the receipted D64/GQA4 MMA prefill route through registry slot
37. There is no positive environment flag required for production selection;
`IMPARO_CUDA_NO_ATTN_D64_MMA_PREFILL=1` is a negative diagnostic fallback. CUDA now
implements the typed Q8 row/matmul paths and LFM2 SiLU/short-convolution/snapshot
operations required by Step 9. The existing Metal workflow is outside this DLL and was
not changed.

The E4B SM86 exact-128 production candidate is safe-off by default and becomes active
only through an adjacent fingerprint-matched receipt. Slot 39 value 5 selects the
receipted Route-5 transaction; slot 40 value 1 adds exact-key Graph replay after its
fixed full-logit replay gate. Other token counts, models, SM families and Metal retain
their established routes.

The project still targets performance beyond llama.cpp, but the historical at-least-2x
number is a non-blocking optimization goal. It never weakens route correctness, receipt
fingerprinting, determinism or regression gates.
