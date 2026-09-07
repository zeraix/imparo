"""Infrastructure-only Triton AOT smoke kernel.

This source is consumed by ``tools/triton-pack`` in a controlled builder. It is
never imported by the Imparo runtime and intentionally imports neither torch nor
any project module.
"""

import triton
import triton.language as tl


@triton.jit
def ip_ab8511ce441e25e37a94fd947a072a657077a099a3ce4b944cbcc7e002a34a2c(
    output,
    input,
    n_elements,
):
    offsets = tl.arange(0, 256)
    active = offsets < n_elements
    values = tl.load(input + offsets, mask=active, other=0.0)
    tl.store(output + offsets, values, mask=active)


IMPARO_AOT = {
    "kernel": "ip_ab8511ce441e25e37a94fd947a072a657077a099a3ce4b944cbcc7e002a34a2c",
    "signature": ["*fp32:16", "*fp32:16", "i32"],
    "num_warps": 1,
    "num_stages": 1,
    "manifest": {
        "pack_id": "com.zeraix.imparo.community.cuda.smoke",
        "pack_version": "1.0.0",
        "distribution": "community",
        "release_channel": "imparo.community.stable",
        "required_entitlement_features": [],
        "engine_api": {"min": 1, "max_exclusive": 2},
        "backend_abi": {"min": 26, "max_exclusive": 27},
        "driver_min": 12000,
        "math_mode": "strict",
        "choice_group": {
            "choice_group_id": "imparo.cuda.infrastructure.copy.v1",
            "contract": {
                "id": "imparo.cuda.infrastructure.copy",
                "revision": 1,
                "sha256": "37d08404f20f83b2076f435208883dee937b985414629a4040df0fe896dd9d75",
            },
            "workload": {
                "workload_id": "imparo.workload.attention_decode",
                "revision": 1,
                "parameters": {},
                "parameters_sha256": "1aa54ff326d23ef77b0dd6db032f335ae5d32710cff814846b28b7a2fc3ad351",
                "fixture_sha256": "e29e17cdcc8449b75d7291d0c53273279b67c7f5b0e4751b4c2ab00d63d3a8eb",
            },
            "screened": True,
            "bit_affecting": False,
            "joint_with": [],
        },
        "variant": {
            "choice_group_id": "imparo.cuda.infrastructure.copy.v1",
            "contract": {
                "id": "imparo.cuda.infrastructure.copy",
                "revision": 1,
                "sha256": "37d08404f20f83b2076f435208883dee937b985414629a4040df0fe896dd9d75",
            },
            "constraints": {
                "shapes": [
                    {
                        "slot": "input",
                        "axis": 0,
                        "min": 0,
                        "max": 256,
                        "multiple_of": 1,
                    },
                    {
                        "slot": "output",
                        "axis": 0,
                        "min": 0,
                        "max": 256,
                        "multiple_of": 1,
                    },
                ],
                "dtypes": [
                    {"slot": "input", "allowed": ["f32"]},
                    {"slot": "output", "allowed": ["f32"]},
                ],
                "quantizations": [
                    {"slot": "input", "allowed": ["none"]},
                    {"slot": "output", "allowed": ["none"]},
                ],
                "layouts": [
                    {"slot": "input", "allowed": ["contiguous"]},
                    {"slot": "output", "allowed": ["contiguous"]},
                ],
                "alignments": [
                    {"slot": "input", "bytes": 16},
                    {"slot": "output", "bytes": 16},
                ],
            },
            "effects": [
                {"slot": "output", "access": "write", "aliasing": []},
                {"slot": "input", "access": "read", "aliasing": []},
                {"slot": "n_elements", "access": "read", "aliasing": []},
            ],
            "scratch": {
                "max_bytes": 0,
                "alignment": 16,
                "zero_initialized": False,
            },
            "launch": {
                "arguments": [
                    {"kind": "slot", "slot": "output", "wire_type": "tensor_ptr"},
                    {"kind": "slot", "slot": "input", "wire_type": "tensor_ptr"},
                    {"kind": "slot", "slot": "n_elements", "wire_type": "scalar_u32"},
                ],
                "grid": {
                    "x": {"kind": "const", "value": 1},
                    "y": {"kind": "const", "value": 1},
                    "z": {"kind": "const", "value": 1},
                },
                "block": {"x": 32, "y": 1, "z": 1},
                "dynamic_shared_bytes": {"kind": "const", "value": 0},
            },
            "graph_capture": "forbidden",
            "graph_update_slots": [],
            "numerical_class": {"kind": "diagnostic_only"},
            "determinism": "required",
            "bit_affecting": False,
            "required_entitlement_features": [],
            "requires": [],
            "conflicts": [],
            "provides": [],
            "joint_with": [],
        },
    },
}
