"""Phase A2 RMSNorm -> Q8_1 MMQ candidates; lab-only, never runtime-imported."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


WIDTH = tl.constexpr(2560)
N_TOK = tl.constexpr(512)
BLOCK = tl.constexpr(4096)
Q8_BLOCKS = tl.constexpr(80)
Q8_GROUPS_PADDED = tl.constexpr(128)
Q8_RECORD_BYTES = tl.constexpr(144)
Q8_RECORD_FLOATS = tl.constexpr(36)


@triton.jit
def ip_1f7730341ef8debbde72f83b24bc35b271fa3422b4bf10c8b7de43dded918834(
    src, dst, mul, q8_bytes, q8_scales, eps
):
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    active = cols < WIDTH
    src_offsets = tok * WIDTH + cols
    values = tl.load(src + src_offsets, mask=active, other=0.0)
    square_sum = tl.sum(values * values, axis=0)
    norm_scale = libdevice.rsqrt(square_sum / WIDTH + eps)
    weights = tl.load(mul + cols, mask=active, other=0.0)
    normalized = values * weights * norm_scale
    tl.store(dst + src_offsets, normalized, mask=active)

    # Reduce each contiguous 32-value Q8_1 block independently.  The padded
    # rows are masked from every store and therefore cannot touch the output.
    matrix = tl.reshape(normalized, (Q8_GROUPS_PADDED, 32))
    amax = tl.max(tl.abs(matrix), axis=1)
    safe_amax = tl.where(amax > 0.0, amax, 1.0)
    inv_scale = 127.0 / safe_amax
    quantized = libdevice.round(matrix * tl.reshape(inv_scale, (Q8_GROUPS_PADDED, 1)))
    quantized = tl.reshape(quantized, (BLOCK,)).to(tl.int8)

    block = cols // 32
    block_in_group = block % 4
    group = block // 4
    lane = cols % 32
    record = group * N_TOK + tok
    q8_offset = record * Q8_RECORD_BYTES + block_in_group * 32 + lane
    tl.store(q8_bytes + q8_offset, quantized, mask=active)

    block_ids = tl.arange(0, Q8_GROUPS_PADDED)
    scale_active = block_ids < Q8_BLOCKS
    scale_group = block_ids // 4
    scale_in_group = block_ids % 4
    scale_record = scale_group * N_TOK + tok
    scale_offset = scale_record * Q8_RECORD_FLOATS + scale_in_group
    # Native stores d through fp16 and then expands it back to fp32.  Keep
    # quantization on the unrounded reciprocal, matching the CUDA order.
    scales = tl.where(amax > 0.0, amax / 127.0, 0.0)
    scales = scales.to(tl.float16).to(tl.float32)
    tl.store(q8_scales + scale_offset, scales, mask=scale_active)


@triton.jit
def ip_fd28e32eb73acb9f96ea19a08c73529e5cdeea92d60f3ec29ac80a43a0d0efcc(
    src, dst, mul, q8_bytes, q8_scales, eps
):
    # Kept self-contained because pinned ASTSource compilation intentionally
    # does not consult the active-driver dependency binder for JIT helpers.
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    active = cols < WIDTH
    src_offsets = tok * WIDTH + cols
    values = tl.load(src + src_offsets, mask=active, other=0.0)
    square_sum = tl.sum(values * values, axis=0)
    norm_scale = libdevice.rsqrt(square_sum / WIDTH + eps)
    weights = tl.load(mul + cols, mask=active, other=0.0)
    normalized = values * weights * norm_scale
    tl.store(dst + src_offsets, normalized, mask=active)

    matrix = tl.reshape(normalized, (Q8_GROUPS_PADDED, 32))
    amax = tl.max(tl.abs(matrix), axis=1)
    safe_amax = tl.where(amax > 0.0, amax, 1.0)
    inv_scale = 127.0 / safe_amax
    quantized = libdevice.round(matrix * tl.reshape(inv_scale, (Q8_GROUPS_PADDED, 1)))
    quantized = tl.reshape(quantized, (BLOCK,)).to(tl.int8)

    block = cols // 32
    block_in_group = block % 4
    group = block // 4
    lane = cols % 32
    record = group * N_TOK + tok
    q8_offset = record * Q8_RECORD_BYTES + block_in_group * 32 + lane
    tl.store(q8_bytes + q8_offset, quantized, mask=active)

    block_ids = tl.arange(0, Q8_GROUPS_PADDED)
    scale_active = block_ids < Q8_BLOCKS
    scale_group = block_ids // 4
    scale_in_group = block_ids % 4
    scale_record = scale_group * N_TOK + tok
    scale_offset = scale_record * Q8_RECORD_FLOATS + scale_in_group
    scales = tl.where(amax > 0.0, amax / 127.0, 0.0)
    scales = scales.to(tl.float16).to(tl.float32)
    tl.store(q8_scales + scale_offset, scales, mask=scale_active)
