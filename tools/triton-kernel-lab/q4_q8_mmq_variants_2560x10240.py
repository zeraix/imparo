"""LAB-only bounded Q4_0 x Q8_1 MMQ variants: K=2560, M=10240."""

import triton
import triton.language as tl


N_BLOCKS = tl.constexpr(80)
N_OUT = tl.constexpr(10240)


@triton.jit
def _q4_q8_mmq(
    w_qs,
    w_d,
    x_qs,
    x_d,
    y,
    n_tok,
    out_stride,
    numeric_stream_grid,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program_row = tl.program_id(0)
    program_token = tl.program_id(1)
    rows = program_row * BLOCK_M + tl.arange(0, BLOCK_M)
    tokens = program_token * BLOCK_N + tl.arange(0, BLOCK_N)
    k32 = tl.arange(0, 32)
    prefix = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    suffix = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Every bounded variant divides the native 128x128 numerical tile.  Derive
    # that tile from element coordinates so changing BM/BN cannot silently
    # change the native Stream-K seam or its suffix-then-prefix final fold.
    token_valid = tokens < n_tok
    native_ntx = (n_tok + 127) // 128
    native_row_tile = (program_row * BLOCK_M) // 128
    native_token_tile = (program_token * BLOCK_N) // 128
    native_tile = native_row_tile * native_ntx + native_token_tile
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
            w_qs + weight_record * 18 + (k32[None, :] & 15)
        ).to(tl.uint8)
        nibble = tl.where(k32[None, :] < 16, packed & 15, packed >> 4)
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
            x_d + (group * n_tok + tokens) * 36 + block_in_group,
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


@triton.jit
def ip_cd1dd7eea1c5a04fa3fca1d1dbbde58075df25baad5830bf8d1fb179d1c7d5ed(
    w_qs, w_d, x_qs, x_d, y, n_tok, out_stride, numeric_stream_grid
):
    _q4_q8_mmq(
        w_qs, w_d, x_qs, x_d, y, n_tok, out_stride, numeric_stream_grid,
        BLOCK_M=128, BLOCK_N=64,
    )


@triton.jit
def ip_040845b88cef1fe24b7900e33d507b81592c31ddc35ba8516ba7151c03741bc0(
    w_qs, w_d, x_qs, x_d, y, n_tok, out_stride, numeric_stream_grid
):
    _q4_q8_mmq(
        w_qs, w_d, x_qs, x_d, y, n_tok, out_stride, numeric_stream_grid,
        BLOCK_M=64, BLOCK_N=128,
    )


@triton.jit
def ip_58bab621edc125e72372075f257ab39554b74abc49ff8855e6d836bbcdfd857c(
    w_qs, w_d, x_qs, x_d, y, n_tok, out_stride, numeric_stream_grid
):
    _q4_q8_mmq(
        w_qs, w_d, x_qs, x_d, y, n_tok, out_stride, numeric_stream_grid,
        BLOCK_M=64, BLOCK_N=64,
    )


@triton.jit
def ip_2f16dc07d913306ecc0325a784eab13a396850af0b5c8bc1f8c59e6b69f7198b(
    w_qs, w_d, x_qs, x_d, y, n_tok, out_stride, numeric_stream_grid
):
    _q4_q8_mmq(
        w_qs, w_d, x_qs, x_d, y, n_tok, out_stride, numeric_stream_grid,
        BLOCK_M=64, BLOCK_N=64,
    )
