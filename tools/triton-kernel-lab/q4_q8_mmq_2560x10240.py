"""LAB-only Q4_0 x Q8_1 MMQ: K=2560, M=10240, N=512."""

import triton
import triton.language as tl


N_BLOCKS = tl.constexpr(80)
N_OUT = tl.constexpr(10240)
BLOCK_M = tl.constexpr(64)
BLOCK_N = tl.constexpr(64)


@triton.jit
def ip_a1da19c631b940e2f64b0e46f33bb0ced69e830472eae28b4c06560d2509fda6(
    w_qs, w_d, x_qs, x_d, y, n_tok, out_stride, numeric_stream_grid
):
    program_row = tl.program_id(0)
    program_token = tl.program_id(1)
    rows = program_row * BLOCK_M + tl.arange(0, BLOCK_M)
    tokens = program_token * BLOCK_N + tl.arange(0, BLOCK_N)
    k32 = tl.arange(0, 32)
    prefix = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    suffix = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Reproduce the native 128x128 tile's Stream-K seam.  For expansion the
    # caller passes numeric_stream_grid=320, so every boundary is block-aligned
    # and seam is zero; retaining the calculation freezes the eighth ABI slot.
    token_valid = tokens < n_tok
    native_ntx = (n_tok + 127) // 128
    native_tile = (program_row // 2) * native_ntx + program_token // 2
    native_tiles = (N_OUT // 128) * native_ntx
    worker = (native_tile * numeric_stream_grid + native_tiles - 1) // native_tiles
    worker = tl.maximum(worker, 1)
    total_work = native_tiles * N_BLOCKS
    boundary = worker * total_work // numeric_stream_grid
    boundary -= (boundary % N_BLOCKS) % 8
    seam = tl.where(
        (worker < numeric_stream_grid) & (boundary // N_BLOCKS == native_tile),
        boundary % N_BLOCKS,
        0,
    )

    for block in tl.range(0, N_BLOCKS):
        weight_record = rows[:, None] * N_BLOCKS + block
        packed = tl.load(
            w_qs
            + weight_record * 18
            + (k32[None, :] & 15)
        ).to(tl.uint8)
        nibble = tl.where(
            k32[None, :] < 16, packed & 15, packed >> 4
        )
        q4 = (nibble.to(tl.int16) - 8).to(tl.int8)
        d4 = tl.load(w_d + (rows * N_BLOCKS + block) * 9).to(tl.float32)

        group = block // 4
        block_in_group = block % 4
        activation_record = group * n_tok + tokens[None, :]
        q8 = tl.load(
            x_qs
            + activation_record * 144
            + block_in_group * 32
            + k32[:, None],
            mask=token_valid[None, :],
            other=0,
        )
        d8 = tl.load(
            x_d
            + (group * n_tok + tokens) * 36
            + block_in_group,
            mask=token_valid,
            other=0.0,
        )
        integer_dot = tl.dot(q4, q8, out_dtype=tl.int32)
        contribution = integer_dot.to(tl.float32) * d4[:, None] * d8[None, :]
        to_prefix = (seam == 0) | (block < seam)
        prefix += tl.where(to_prefix, contribution, 0.0)
        suffix += tl.where((seam != 0) & (block >= seam), contribution, 0.0)

    acc = tl.where(seam != 0, suffix + prefix, prefix)
    output = tokens[None, :] * out_stride + rows[:, None]
    tl.store(y + output, acc, mask=token_valid[None, :])
