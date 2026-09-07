"""Dependency-free structural and semantic checks for Program Pack v1 drafts."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import struct
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "examples"
U64_MAX = (1 << 64) - 1


class ContractError(ValueError):
    """A schema or cross-field contract was violated."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ContractError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def loads_strict(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_strict_object)


def load_json(path: Path) -> Any:
    return loads_strict(path.read_bytes().decode("utf-8", errors="strict"))


def domain_sha256(domain: str, payload: bytes) -> str:
    message = domain.encode("ascii") + b"\0" + struct.pack("<Q", len(payload)) + payload
    return hashlib.sha256(message).hexdigest()


def canonical_json_sha256(domain: str, value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return domain_sha256(domain, payload)


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ContractError(f"non-local schema reference: {ref}")
    node: Any = root
    for raw in ref[2:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        node = node[key]
    if not isinstance(node, dict):
        raise ContractError(f"schema reference is not an object: {ref}")
    return node


def _type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "boolean": type(value) is bool,
    }.get(expected, False)


def _json_equal(left: Any, right: Any) -> bool:
    """Use JSON Schema equality, keeping booleans distinct from JSON numbers."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def validate_schema(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str = "$",
    _depth: int = 0,
) -> None:
    """Validate the JSON Schema subset used by the tracked v1 drafts."""

    if _depth > 128:
        raise ContractError(f"{path}: schema value exceeds nesting limit")

    if "$ref" in schema:
        validate_schema(
            value, _resolve_ref(root, schema["$ref"]), root, path, _depth + 1
        )
        return

    if "oneOf" in schema:
        matches = 0
        reasons: list[str] = []
        for branch in schema["oneOf"]:
            try:
                validate_schema(value, branch, root, path, _depth + 1)
                matches += 1
            except ContractError as error:
                reasons.append(str(error))
        if matches != 1:
            raise ContractError(
                f"{path}: expected exactly one schema branch, matched {matches}; "
                + " | ".join(reasons)
            )
        return

    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ContractError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        raise ContractError(f"{path}: value is outside enum")

    expected = schema.get("type")
    if expected is not None and not _type_matches(value, expected):
        raise ContractError(f"{path}: expected {expected}, got {type(value).__name__}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                raise ContractError(f"{path}: missing required field {required}")
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise ContractError(f"{path}: unknown fields {extras}")
        for key, child in value.items():
            if key in properties:
                validate_schema(
                    child, properties[key], root, f"{path}.{key}", _depth + 1
                )

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ContractError(f"{path}: too few items")
        if len(value) > schema.get("maxItems", len(value)):
            raise ContractError(f"{path}: too many items")
        if schema.get("uniqueItems"):
            normalized = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value]
            if len(normalized) != len(set(normalized)):
                raise ContractError(f"{path}: duplicate items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, child in enumerate(value):
                validate_schema(
                    child, item_schema, root, f"{path}[{index}]", _depth + 1
                )

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ContractError(f"{path}: string too short")
        if len(value) > schema.get("maxLength", len(value)):
            raise ContractError(f"{path}: string too long")
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(pattern, value) is None:
            raise ContractError(f"{path}: string does not match {pattern}")

    if type(value) is int:
        if value < schema.get("minimum", value):
            raise ContractError(f"{path}: integer below minimum")
        if value > schema.get("maximum", value):
            raise ContractError(f"{path}: integer above maximum")


def _unique(values: list[Any], label: str) -> None:
    if len(values) != len(set(values)):
        raise ContractError(f"duplicate {label}")


def _walk_expression(
    expr: dict[str, Any],
    allowed_slots: set[str],
    depth: int = 1,
) -> tuple[int, int | None]:
    if depth > 8:
        raise ContractError("launch expression exceeds depth limit")
    if expr["kind"] == "const":
        return 1, expr["value"]
    if expr["kind"] == "slot":
        if expr["slot"] not in allowed_slots:
            raise ContractError("launch expression references an unknown unsigned slot")
        return 1, None

    left_nodes, left = _walk_expression(expr["args"][0], allowed_slots, depth + 1)
    right_nodes, right = _walk_expression(expr["args"][1], allowed_slots, depth + 1)
    nodes = 1 + left_nodes + right_nodes
    if nodes > 64:
        raise ContractError("launch expression exceeds node limit")
    if expr["op"] == "ceil_div" and right == 0:
        raise ContractError("launch expression divides by zero")
    if left is None or right is None:
        return nodes, None

    op = expr["op"]
    if op == "add":
        result = left + right
    elif op == "mul":
        result = left * right
    elif op == "ceil_div":
        if right == 0:
            raise ContractError("launch expression divides by zero")
        numerator = left + right - 1
        if numerator > U64_MAX:
            raise ContractError("launch expression overflows u64")
        result = numerator // right
    elif op == "min":
        result = min(left, right)
    else:
        result = max(left, right)
    if result > U64_MAX:
        raise ContractError("launch expression overflows u64")
    return nodes, result


def validate_manifest_semantics(manifest: dict[str, Any]) -> None:
    for name in ("engine_api", "backend_abi"):
        version_range = manifest[name]
        if version_range["min"] >= version_range["max_exclusive"]:
            raise ContractError(f"{name} is empty or reversed")

    if manifest["required_entitlement_features"] != sorted(
        manifest["required_entitlement_features"]
    ):
        raise ContractError("pack entitlement features are not sorted")
    if manifest["distribution"] == "community" and manifest["required_entitlement_features"]:
        raise ContractError("community pack requires a commercial entitlement")

    required_extensions = {item["id"]: item["revision"] for item in manifest["required_extensions"]}
    optional_extensions = {item["id"]: item["revision"] for item in manifest["optional_extensions"]}
    if len(required_extensions) != len(manifest["required_extensions"]):
        raise ContractError("duplicate required extension id")
    if len(optional_extensions) != len(manifest["optional_extensions"]):
        raise ContractError("duplicate optional extension id")
    if set(required_extensions) & set(optional_extensions):
        raise ContractError("required and optional extensions overlap")
    declared_extensions = {**required_extensions, **optional_extensions}

    modules = manifest["modules"]
    _unique([item["id"] for item in modules], "module id")
    _unique([item["file"] for item in modules], "module path")
    _unique([item["sha256"] for item in modules], "module digest")
    if sum(item["bytes"] for item in modules) > 536_870_912:
        raise ContractError("declared module bytes exceed pack ceiling")
    module_ids = {item["id"] for item in modules}

    groups = manifest["choice_groups"]
    _unique([item["choice_group_id"] for item in groups], "choice-group id")
    group_by_id = {item["choice_group_id"]: item for item in groups}
    for group in groups:
        workload = group["workload"]
        parameters = workload["parameters"]
        if workload["workload_id"] == "imparo.workload.narrow_mix":
            if set(parameters) != {"narrow_tokens"}:
                raise ContractError("narrow_mix requires exactly narrow_tokens")
        elif parameters:
            raise ContractError("parameterless workload contains parameters")
        expected_parameters = canonical_json_sha256(
            "imparo-program-workload-parameters-v1", parameters
        )
        if workload["parameters_sha256"] != expected_parameters:
            raise ContractError("workload parameter digest mismatch")
        if "cross_check" in group:
            cross = group["cross_check"]
            cross_parameters = cross["parameters"]
            if cross["workload_id"] == "imparo.workload.narrow_mix":
                if set(cross_parameters) != {"narrow_tokens"}:
                    raise ContractError("narrow_mix cross-check requires narrow_tokens")
            elif cross_parameters:
                raise ContractError("parameterless cross-check contains parameters")
            if cross["parameters_sha256"] != canonical_json_sha256(
                "imparo-program-workload-parameters-v1", cross_parameters
            ):
                raise ContractError("cross-check parameter digest mismatch")
        if group["choice_group_id"] in group["joint_with"]:
            raise ContractError("choice group is joint with itself")
    for group in groups:
        for joint in group["joint_with"]:
            if (
                joint in group_by_id
                and group["choice_group_id"] not in group_by_id[joint]["joint_with"]
            ):
                raise ContractError("choice-group joint relation is not symmetric")

    variants = manifest["variants"]
    _unique([item["variant_id"] for item in variants], "variant id")
    _unique([item["config_id"] for item in variants], "config id")
    variant_ids = {item["variant_id"] for item in variants}
    variant_by_id = {item["variant_id"]: item for item in variants}
    required_entitlements: set[str] = set()

    for variant in variants:
        if variant["required_entitlement_features"] != sorted(
            variant["required_entitlement_features"]
        ):
            raise ContractError("variant entitlement features are not sorted")
        group_id = variant["choice_group_id"]
        if group_id not in group_by_id:
            raise ContractError("variant references missing choice group")
        if variant["module_id"] not in module_ids:
            raise ContractError("variant references missing module")
        if variant["contract"] != group_by_id[group_id]["contract"]:
            raise ContractError("variant and choice-group contracts differ")
        if variant["bit_affecting"] != group_by_id[group_id]["bit_affecting"]:
            raise ContractError("variant and choice-group bit-affecting flags differ")
        if set(variant["joint_with"]) != set(group_by_id[group_id]["joint_with"]):
            raise ContractError("variant and choice-group joint sets differ")
        extension = variant.get("extension")
        if extension is not None:
            if extension["id"] not in declared_extensions:
                raise ContractError("variant references undeclared extension")
            if declared_extensions[extension["id"]] != extension["revision"]:
                raise ContractError("variant references wrong extension revision")

        shapes = variant["constraints"]["shapes"]
        shape_keys = [(item["slot"], item["axis"]) for item in shapes]
        _unique(shape_keys, "shape constraint")
        for item in shapes:
            if item["min"] > item["max"]:
                raise ContractError("shape range is reversed")
            for allowed in item.get("one_of", []):
                if not item["min"] <= allowed <= item["max"]:
                    raise ContractError("shape one_of value is outside range")
                if allowed % item["multiple_of"] != 0:
                    raise ContractError("shape one_of value violates multiple_of")
            first_multiple = (
                (item["min"] + item["multiple_of"] - 1) // item["multiple_of"]
            ) * item["multiple_of"]
            if first_multiple > item["max"]:
                raise ContractError("shape constraint has no satisfiable value")

        for domain in ("dtypes", "quantizations", "layouts", "alignments"):
            _unique(
                [item["slot"] for item in variant["constraints"][domain]],
                f"{domain} slot constraint",
            )

        effect_slots = {item["slot"] for item in variant["effects"]}
        _unique([item["slot"] for item in variant["effects"]], "effect slot")
        for effect in variant["effects"]:
            _unique(
                [alias["target_slot"] for alias in effect["aliasing"]],
                "alias target",
            )
            for alias in effect["aliasing"]:
                if alias["target_slot"] == effect["slot"]:
                    raise ContractError("effect aliases itself")
                if alias["target_slot"] not in effect_slots:
                    raise ContractError("effect aliases an unknown slot")

        launch = variant["launch"]
        block = launch["block"]
        threads = block["x"] * block["y"] * block["z"]
        if threads > 1024 or threads > variant["resources"]["threads_per_block_max"]:
            raise ContractError("block exceeds thread ceiling")
        unsigned_slots = {
            item["slot"]
            for item in launch["arguments"]
            if item["kind"] == "slot"
            and item["wire_type"] in {"scalar_u32", "scalar_u64"}
        }
        for axis, ceiling in (("x", 2_147_483_647), ("y", 65_535), ("z", 65_535)):
            _, constant = _walk_expression(launch["grid"][axis], unsigned_slots)
            if constant is not None and not 1 <= constant <= ceiling:
                raise ContractError(f"grid {axis} is outside limits")
        _, dynamic_shared = _walk_expression(
            launch["dynamic_shared_bytes"], unsigned_slots
        )
        if dynamic_shared is not None and dynamic_shared > min(
            262_144, variant["resources"]["dynamic_shared_bytes_max"]
        ):
            raise ContractError("dynamic shared memory exceeds ceiling")
        if variant["graph_capture"] != "replay_update_safe" and variant["graph_update_slots"]:
            raise ContractError("non-replay graph route declares update slots")

        required_refs = {(item["kind"], item["id"]) for item in variant["requires"]}
        conflict_refs = {(item["kind"], item["id"]) for item in variant["conflicts"]}
        if len(required_refs) != len(variant["requires"]):
            raise ContractError("duplicate required dependency")
        if len(conflict_refs) != len(variant["conflicts"]):
            raise ContractError("duplicate conflict dependency")
        if required_refs & conflict_refs:
            raise ContractError("dependency is both required and conflicted")
        for relation_name, relation in (
            ("requires", variant["requires"]),
            ("conflicts", variant["conflicts"]),
        ):
            for reference in relation:
                if reference["kind"] == "variant" and reference["id"] not in variant_ids:
                    raise ContractError("dependency references missing variant")
                if reference["kind"] == "variant" and reference["id"] == variant["variant_id"]:
                    raise ContractError("variant references itself")
                if reference["kind"] == "choice_group" and reference["id"] == group_id:
                    raise ContractError("variant references its own choice group")
                if (
                    relation_name == "requires"
                    and reference["kind"] == "variant"
                    and variant_by_id[reference["id"]]["choice_group_id"] == group_id
                ):
                    raise ContractError(
                        f"variant {relation_name} another variant in its exclusive choice group"
                    )
        required_entitlements.update(variant["required_entitlement_features"])

    if required_entitlements != set(manifest["required_entitlement_features"]):
        raise ContractError("pack and variant entitlement feature sets differ")

    for variant in variants:
        for conflict in variant["conflicts"]:
            if conflict["kind"] != "variant":
                continue
            reverse = {
                (item["kind"], item["id"])
                for item in variant_by_id[conflict["id"]]["conflicts"]
            }
            if ("variant", variant["variant_id"]) not in reverse:
                raise ContractError("variant conflict relation is not symmetric")

    visiting: set[str] = set()
    visited: set[str] = set()
    variants_by_group: dict[str, list[str]] = {}
    for variant in variants:
        variants_by_group.setdefault(variant["choice_group_id"], []).append(
            variant["variant_id"]
        )

    def visit(variant_id: str) -> None:
        if variant_id in visiting:
            raise ContractError("variant dependency cycle")
        if variant_id in visited:
            return
        visiting.add(variant_id)
        for relation in variant_by_id[variant_id]["requires"]:
            if relation["kind"] == "variant":
                visit(relation["id"])
            elif relation["kind"] == "choice_group":
                targets = variants_by_group.get(relation["id"], [])
                if len(targets) == 1:
                    visit(targets[0])
        visiting.remove(variant_id)
        visited.add(variant_id)

    for variant_id in variant_ids:
        visit(variant_id)

    selected_groups = {variant["choice_group_id"] for variant in variants}
    if selected_groups != set(group_by_id):
        raise ContractError("choice group has no variant")


def validate_contract_semantics(contract: dict[str, Any]) -> None:
    if contract["state_reset"]["revision"] != contract["engine_adapter"]["revision"]:
        raise ContractError("state reset revision differs from engine adapter")
    slots = contract["slots"]
    _unique([slot["slot_id"] for slot in slots], "contract slot id")
    _unique([slot["abi_index"] for slot in slots], "contract ABI index")
    if [slot["abi_index"] for slot in slots] != list(range(len(slots))):
        raise ContractError("contract slots are not dense ABI order")
    slot_ids = {slot["slot_id"] for slot in slots}
    slot_by_id = {slot["slot_id"]: slot for slot in slots}
    wire_for_kind = {
        "tensor_ptr": "device_u64",
        "state_ptr": "device_u64",
        "scratch_ptr": "device_u64",
        "scalar_i32": "i32",
        "scalar_u32": "u32",
        "scalar_u64": "u64",
        "scalar_f32": "f32_bits",
    }
    roles_for_kind = {
        "tensor_ptr": {"input", "output"},
        "state_ptr": {"state"},
        "scratch_ptr": {"scratch"},
        "scalar_i32": {"input"},
        "scalar_u32": {"input", "shape"},
        "scalar_u64": {"input", "shape"},
        "scalar_f32": {"input"},
    }
    for slot in slots:
        if slot["wire_type"] != wire_for_kind[slot["kind"]]:
            raise ContractError("slot kind and wire type differ")
        if slot["role"] not in roles_for_kind[slot["kind"]]:
            raise ContractError("slot kind and role differ")
        if slot["kind"] == "tensor_ptr" and slot["role"] == "input":
            allowed_lifetimes = {"invocation", "graph", "model"}
        elif slot["role"] == "output":
            allowed_lifetimes = {"invocation", "graph"}
        elif slot["role"] == "state":
            allowed_lifetimes = {"conversation"}
        elif slot["role"] == "scratch":
            allowed_lifetimes = {"graph"}
        else:
            allowed_lifetimes = {"invocation", "graph"}
        if slot["lifetime"] not in allowed_lifetimes:
            raise ContractError("slot role and lifetime differ")
        if slot["lifetime"] == "model" and slot["graph_mutability"] != "capture_static":
            raise ContractError("model lifetime slot is not capture-static")
        if slot["role"] == "scratch" and slot["graph_mutability"] != "capture_static":
            raise ContractError("scratch slot is not capture-static")
        if slot["role"] in {"input", "shape"} and slot["access"] != "read":
            raise ContractError("input or shape slot is writable")
        if slot["role"] == "output" and slot["access"] == "read":
            raise ContractError("output slot is read-only")
        if slot["role"] in {"state", "scratch"} and slot["access"] != "read_write":
            raise ContractError("state or scratch slot is not read_write")
        if slot["role"] == "state" and slot["state_initialization"] == "none":
            raise ContractError("state slot lacks initialization semantics")
        if slot["role"] != "state" and slot["state_initialization"] != "none":
            raise ContractError("non-state slot declares state initialization")
        is_scalar = slot["kind"].startswith("scalar_")
        if is_scalar:
            natural_alignment = 8 if slot["kind"] == "scalar_u64" else 4
            if slot["alignment"] != natural_alignment:
                raise ContractError("scalar slot does not use natural C alignment")
        if is_scalar and slot["rank"] != 0:
            raise ContractError("scalar slot has nonzero rank")
        if not is_scalar and slot["rank"] == 0:
            raise ContractError("pointer slot has zero rank")
        owns_tensor_domain = slot["kind"] in {"tensor_ptr", "state_ptr"}
        domains = (
            slot["allowed_dtypes"],
            slot["allowed_quantizations"],
            slot["allowed_layouts"],
        )
        if owns_tensor_domain and any(not domain for domain in domains):
            raise ContractError("tensor/state slot has an empty data domain")
        if not owns_tensor_domain and any(domains):
            raise ContractError("scalar/scratch slot declares a tensor data domain")
        _unique([alias["target_slot"] for alias in slot["aliasing"]], "contract alias")
        for alias in slot["aliasing"]:
            if alias["target_slot"] not in slot_ids:
                raise ContractError("alias references missing contract slot")
            if alias["target_slot"] == slot["slot_id"]:
                raise ContractError("slot aliases itself")
            other = slot_by_id[alias["target_slot"]]
            if alias["mode"] != "forbidden" and slot["wire_type"] != other["wire_type"]:
                raise ContractError("alias relation has incompatible wire types")
            if (
                alias["mode"] != "forbidden"
                and (slot["role"] == "scratch") != (other["role"] == "scratch")
            ):
                raise ContractError("scratch aliases non-scratch state")
            reverse = {
                (item["target_slot"], item["mode"]) for item in other["aliasing"]
            }
            if (slot["slot_id"], alias["mode"]) not in reverse:
                raise ContractError("contract alias relation is not symmetric")

    must_alias: dict[str, set[str]] = {slot_id: set() for slot_id in slot_ids}
    for slot in slots:
        for alias in slot["aliasing"]:
            if alias["mode"] == "must_alias":
                must_alias[slot["slot_id"]].add(alias["target_slot"])
    remaining = set(slot_ids)
    while remaining:
        seed = remaining.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for target in must_alias[current] - component:
                component.add(target)
                remaining.discard(target)
                frontier.append(target)
        if len(component) > 1:
            for slot_id in component:
                if must_alias[slot_id] != component - {slot_id}:
                    raise ContractError("must_alias component is not a complete relation")

    arguments = contract["kernel_arguments"]
    if [item["abi_index"] for item in arguments] != list(range(len(arguments))):
        raise ContractError("kernel arguments are not dense ABI order")
    if len(arguments) < len(slots):
        raise ContractError("kernel ABI omits contract slots")
    for index, slot in enumerate(slots):
        argument = arguments[index]
        if argument["kind"] != "slot":
            raise ContractError("manifest constant precedes a required slot")
        if argument["slot"] != slot["slot_id"] or argument["wire_type"] != slot["kind"]:
            raise ContractError("kernel slot argument differs from contract slot")
    constant_names: list[str] = []
    for argument in arguments[len(slots) :]:
        if argument["kind"] != "manifest_u32":
            raise ContractError("duplicate or reordered kernel slot argument")
        if argument["min"] > argument["max"]:
            raise ContractError("manifest constant range is reversed")
        constant_names.append(argument["name"])
    _unique(constant_names, "manifest constant")

    replay_safe = "replay_update_safe" in contract["allowed_graph_capture"]
    for slot in slots:
        if slot["graph_mutability"] == "replay_update" and not replay_safe:
            raise ContractError("replay-update slot lacks replay-safe graph class")
        if (
            replay_safe
            and slot["lifetime"] == "invocation"
            and slot["graph_mutability"] != "replay_update"
        ):
            raise ContractError("replay-safe graph freezes invocation-lifetime slot")

    scratch_slots = [slot for slot in slots if slot["kind"] == "scratch_ptr"]
    if contract["scratch_max_bytes"] == 0:
        if scratch_slots:
            raise ContractError("zero scratch contract declares scratch slot")
        if contract["scratch_zero_initialized"]:
            raise ContractError("zero scratch contract requires initialization")
    elif len(scratch_slots) != 1:
        raise ContractError("nonzero scratch contract requires exactly one scratch slot")

    workload_keys = [
        (item["workload_id"], item["revision"]) for item in contract["allowed_workloads"]
    ]
    _unique(workload_keys, "contract workload authority")


def validate_signature_envelope(envelope: dict[str, Any]) -> None:
    encoded = envelope["signature"]
    try:
        raw = base64.urlsafe_b64decode(encoded + "==")
    except (ValueError, TypeError) as error:
        raise ContractError("signature is not base64url") from error
    canonical = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    if len(raw) != 64 or canonical != encoded:
        raise ContractError("signature is not canonical Ed25519 base64url")


def validate_manifest_against_contract_registry(
    manifest: dict[str, Any],
    registry: dict[tuple[str, int], tuple[dict[str, Any], bytes]],
) -> None:
    validate_manifest_semantics(manifest)
    contract_schema = load_json(ROOT / "program-contract-v1.schema.json")
    group_contracts: dict[str, dict[str, Any]] = {}
    for group in manifest["choice_groups"]:
        reference = group["contract"]
        key = (reference["id"], reference["revision"])
        if key not in registry:
            raise ContractError("choice group references unsupported contract")
        contract, raw_contract = registry[key]
        try:
            parsed_raw_contract = loads_strict(
                raw_contract.decode("utf-8", errors="strict")
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ContractError) as error:
            raise ContractError("raw contract descriptor is not strict JSON") from error
        if not _json_equal(contract, parsed_raw_contract):
            raise ContractError("parsed contract differs from raw descriptor")
        validate_schema(contract, contract_schema, contract_schema)
        if (contract["contract_id"], contract["revision"]) != key:
            raise ContractError("registry key differs from contract descriptor")
        validate_contract_semantics(contract)
        digest = domain_sha256("imparo-program-contract-v1", raw_contract)
        if reference["sha256"] != digest:
            raise ContractError("contract digest mismatch")
        allowed_workloads = {
            (item["workload_id"], item["revision"]): item["fixture_sha256"]
            for item in contract["allowed_workloads"]
        }
        workload = group["workload"]
        workload_key = (workload["workload_id"], workload["revision"])
        if allowed_workloads.get(workload_key) != workload["fixture_sha256"]:
            raise ContractError("workload is outside contract authority")
        if "cross_check" in group:
            cross_check = group["cross_check"]
            cross_key = (cross_check["workload_id"], cross_check["revision"])
            if allowed_workloads.get(cross_key) != cross_check["fixture_sha256"]:
                raise ContractError("cross-check workload is outside contract authority")
        group_contracts[group["choice_group_id"]] = contract

    for variant in manifest["variants"]:
        contract = group_contracts[variant["choice_group_id"]]
        slots = contract["slots"]
        slot_by_id = {slot["slot_id"]: slot for slot in slots}

        expected_effects = {
            slot["slot_id"]: (
                slot["access"],
                sorted(
                    (alias["mode"], alias["target_slot"])
                    for alias in slot["aliasing"]
                ),
            )
            for slot in slots
        }
        actual_effects = {
            effect["slot"]: (
                effect["access"],
                sorted(
                    (alias["mode"], alias["target_slot"])
                    for alias in effect["aliasing"]
                ),
            )
            for effect in variant["effects"]
        }
        if actual_effects != expected_effects:
            raise ContractError("manifest effects differ from engine contract")

        constraint_domains = (
            ("dtypes", "allowed_dtypes"),
            ("quantizations", "allowed_quantizations"),
            ("layouts", "allowed_layouts"),
        )
        for manifest_key, contract_key in constraint_domains:
            seen: set[str] = set()
            for constraint in variant["constraints"][manifest_key]:
                slot_id = constraint["slot"]
                if slot_id not in slot_by_id:
                    raise ContractError("constraint references unknown contract slot")
                if slot_id in seen:
                    raise ContractError("duplicate slot domain constraint")
                seen.add(slot_id)
                if not set(constraint["allowed"]).issubset(
                    set(slot_by_id[slot_id][contract_key])
                ):
                    raise ContractError("constraint widens contract data domain")
        for constraint in variant["constraints"]["alignments"]:
            slot_id = constraint["slot"]
            if slot_id not in slot_by_id:
                raise ContractError("alignment references unknown contract slot")
            if constraint["bytes"] < slot_by_id[slot_id]["alignment"]:
                raise ContractError("alignment is weaker than contract")
        for constraint in variant["constraints"]["shapes"]:
            slot = slot_by_id.get(constraint["slot"])
            if slot is None:
                raise ContractError("shape references unknown contract slot")
            if slot["kind"] not in {"tensor_ptr", "state_ptr"}:
                raise ContractError("shape references a non-tensor contract slot")
            if constraint["axis"] >= slot["rank"]:
                raise ContractError("shape axis exceeds contract slot rank")

        expected_arguments = contract["kernel_arguments"]
        actual_arguments = variant["launch"]["arguments"]
        if len(actual_arguments) != len(expected_arguments):
            raise ContractError("kernel argument count differs from contract")
        for expected, actual in zip(expected_arguments, actual_arguments, strict=True):
            if expected["kind"] != actual["kind"]:
                raise ContractError("kernel argument kind differs from contract")
            if expected["kind"] == "slot":
                if (
                    expected["slot"] != actual["slot"]
                    or expected["wire_type"] != actual["wire_type"]
                ):
                    raise ContractError("kernel slot binding differs from contract")
            elif (
                expected["name"] != actual["name"]
                or not expected["min"] <= actual["value"] <= expected["max"]
            ):
                raise ContractError("manifest constant differs from contract")

        if variant["scratch"]["max_bytes"] > contract["scratch_max_bytes"]:
            raise ContractError("scratch exceeds contract")
        if variant["scratch"]["zero_initialized"] != contract["scratch_zero_initialized"]:
            raise ContractError("scratch initialization differs from contract")
        scratch_slots = [slot for slot in slots if slot["kind"] == "scratch_ptr"]
        if variant["scratch"]["max_bytes"] > 0:
            if len(scratch_slots) != 1:
                raise ContractError("scratch allocation lacks contract slot")
            if variant["scratch"]["alignment"] < scratch_slots[0]["alignment"]:
                raise ContractError("scratch alignment is weaker than contract")
        elif any(
            item["kind"] == "slot" and item["wire_type"] == "scratch_ptr"
            for item in actual_arguments
        ):
            raise ContractError("zero scratch variant binds a scratch argument")
        for resource, ceiling in contract["resource_ceilings"].items():
            if variant["resources"][resource] > ceiling:
                raise ContractError("resource claim exceeds contract")
        if variant["graph_capture"] not in contract["allowed_graph_capture"]:
            raise ContractError("graph class is outside contract")
        numerical_kind = variant["numerical_class"]["kind"]
        if numerical_kind not in contract["allowed_numerical_classes"]:
            raise ContractError("numerical class is outside contract")
        required_updates = {
            slot["slot_id"]
            for slot in slots
            if slot["graph_mutability"] == "replay_update"
        }
        actual_updates = set(variant["graph_update_slots"])
        if variant["graph_capture"] == "replay_update_safe":
            if actual_updates != required_updates:
                raise ContractError("graph update slots differ from contract")
        elif actual_updates:
            raise ContractError("non-replay route has graph update slots")


class ProgramPackSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_schema = load_json(ROOT / "program-pack-v1.schema.json")
        cls.signature_schema = load_json(ROOT / "program-pack-signature-v1.schema.json")
        cls.contract_schema = load_json(ROOT / "program-contract-v1.schema.json")
        cls.manifest = load_json(EXAMPLES / "program-pack-v1.minimal.json")
        cls.signature = load_json(EXAMPLES / "program-pack-signature-v1.minimal.json")
        contract_path = EXAMPLES / "program-contract-v1.minimal.json"
        cls.contract_bytes = contract_path.read_bytes()
        cls.contract = loads_strict(cls.contract_bytes.decode("utf-8", errors="strict"))
        cls.registry = {
            (cls.contract["contract_id"], cls.contract["revision"]): (
                cls.contract,
                cls.contract_bytes,
            )
        }

    def test_schema_examples_pass_declared_pr_a_contracts(self) -> None:
        validate_schema(self.manifest, self.manifest_schema, self.manifest_schema)
        validate_schema(self.signature, self.signature_schema, self.signature_schema)
        validate_signature_envelope(self.signature)
        validate_schema(self.contract, self.contract_schema, self.contract_schema)
        validate_contract_semantics(self.contract)
        validate_manifest_against_contract_registry(self.manifest, self.registry)
        digest = domain_sha256("imparo-program-contract-v1", self.contract_bytes)
        self.assertEqual(self.manifest["choice_groups"][0]["contract"]["sha256"], digest)

    def test_every_declared_object_is_closed(self) -> None:
        def walk(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False)
                for child in node.values():
                    walk(child)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        for schema in (self.manifest_schema, self.signature_schema, self.contract_schema):
            walk(schema)

    def test_duplicate_keys_are_rejected_before_schema_validation(self) -> None:
        with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
            loads_strict('{"schema":1,"schema":1}')
        with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
            loads_strict('{"outer":{"id":"a","id":"b"}}')
        with self.assertRaises(UnicodeDecodeError):
            b'{"value":"\xff"}'.decode("utf-8", errors="strict")

    def test_unknown_root_and_nested_fields_are_rejected(self) -> None:
        root = copy.deepcopy(self.manifest)
        root["priority"] = 100
        with self.assertRaises(ContractError):
            validate_schema(root, self.manifest_schema, self.manifest_schema)
        nested = copy.deepcopy(self.manifest)
        nested["variants"][0]["launch"]["script"] = "run_me"
        with self.assertRaises(ContractError):
            validate_schema(nested, self.manifest_schema, self.manifest_schema)

    def test_local_bounds_and_path_grammar_are_rejected(self) -> None:
        cases: list[tuple[str, Any]] = [
            ("schema", lambda value: value.__setitem__("schema", 2)),
            ("boolean schema", lambda value: value.__setitem__("schema", True)),
            ("backend", lambda value: value.__setitem__("backend", "metal")),
            ("bytes", lambda value: value["modules"][0].__setitem__("bytes", 0)),
            ("path", lambda value: value["modules"][0].__setitem__("file", "modules/../x.cubin")),
            ("hash", lambda value: value["modules"][0].__setitem__("sha256", "A" * 64)),
            (
                "hash trailing newline",
                lambda value: value["modules"][0].__setitem__("sha256", "a" * 64 + "\n"),
            ),
            ("bad semver", lambda value: value.__setitem__("pack_version", "1.0.0-a..b")),
            ("namespaced id newline", lambda value: value.__setitem__("pack_id", "com.example.pack\n")),
            ("symbol", lambda value: value["variants"][0].__setitem__("symbol", "private_tile_64")),
            (
                "dtype from layout domain",
                lambda value: value["variants"][0]["constraints"]["dtypes"][0].__setitem__(
                    "allowed", ["paged_kv"]
                ),
            ),
            (
                "quant from dtype domain",
                lambda value: value["variants"][0]["constraints"]["quantizations"][0].__setitem__(
                    "allowed", ["f32"]
                ),
            ),
            (
                "layout from quant domain",
                lambda value: value["variants"][0]["constraints"]["layouts"][0].__setitem__(
                    "allowed", ["q4_0"]
                ),
            ),
            (
                "boolean alignment",
                lambda value: value["variants"][0]["constraints"]["alignments"][0].__setitem__(
                    "bytes", True
                ),
            ),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.manifest)
                mutate(candidate)
                with self.assertRaises(ContractError):
                    validate_schema(candidate, self.manifest_schema, self.manifest_schema)

    def test_cross_field_failures_are_rejected(self) -> None:
        mutations: list[tuple[str, Any]] = [
            (
                "empty ABI range",
                lambda value: value["backend_abi"].__setitem__("max_exclusive", 26),
            ),
            (
                "duplicate module",
                lambda value: value["modules"].append(copy.deepcopy(value["modules"][0])),
            ),
            (
                "dangling group",
                lambda value: value["variants"][0].__setitem__(
                    "choice_group_id", "imparo.cuda.missing.v1"
                ),
            ),
            (
                "dangling module",
                lambda value: value["variants"][0].__setitem__("module_id", "f" * 64),
            ),
            (
                "contract mismatch",
                lambda value: value["variants"][0]["contract"].__setitem__("revision", 2),
            ),
            (
                "oversized block",
                lambda value: value["variants"][0]["launch"]["block"].__setitem__("x", 64),
            ),
            (
                "invalid graph updates",
                lambda value: value["variants"][0]["graph_update_slots"].append("input"),
            ),
            (
                "bit-affecting disagreement",
                lambda value: value["variants"][0].__setitem__("bit_affecting", True),
            ),
            (
                "same extension id at two revisions",
                lambda value: (
                    value["required_extensions"].append(
                        {"id": "imparo.extension.test", "revision": 1}
                    ),
                    value["optional_extensions"].append(
                        {"id": "imparo.extension.test", "revision": 2}
                    ),
                ),
            ),
            (
                "parameterless workload has parameter",
                lambda value: value["choice_groups"][0]["workload"]["parameters"].__setitem__(
                    "narrow_tokens", 1
                ),
            ),
            (
                "unknown grid slot",
                lambda value: value["variants"][0]["launch"]["grid"].__setitem__(
                    "x", {"kind": "slot", "slot": "missing"}
                ),
            ),
            (
                "dynamic shared-memory excess",
                lambda value: value["variants"][0]["launch"].__setitem__(
                    "dynamic_shared_bytes", {"kind": "const", "value": 1}
                ),
            ),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.manifest)
                mutate(candidate)
                validate_schema(candidate, self.manifest_schema, self.manifest_schema)
                with self.assertRaises(ContractError):
                    validate_manifest_semantics(candidate)

    def test_launch_expression_rejects_division_overflow_and_depth(self) -> None:
        def set_grid(candidate: dict[str, Any], expression: dict[str, Any]) -> None:
            candidate["variants"][0]["launch"]["grid"]["x"] = expression

        divide_zero = {
            "kind": "op",
            "op": "ceil_div",
            "args": [
                {"kind": "const", "value": 1},
                {"kind": "const", "value": 0},
            ],
        }
        dynamic_divide_zero = {
            "kind": "op",
            "op": "ceil_div",
            "args": [
                {"kind": "slot", "slot": "tokens"},
                {"kind": "const", "value": 0},
            ],
        }
        overflow = {
            "kind": "op",
            "op": "mul",
            "args": [
                {"kind": "const", "value": 9_007_199_254_740_991},
                {"kind": "const", "value": 9_007_199_254_740_991},
            ],
        }
        too_deep: dict[str, Any] = {"kind": "const", "value": 1}
        for _ in range(9):
            too_deep = {
                "kind": "op",
                "op": "add",
                "args": [too_deep, {"kind": "const", "value": 1}],
            }
        for name, expression in (
            ("divide zero", divide_zero),
            ("dynamic divide zero", dynamic_divide_zero),
            ("overflow", overflow),
            ("depth", too_deep),
        ):
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.manifest)
                if name == "dynamic divide zero":
                    candidate["variants"][0]["launch"]["arguments"].append(
                        {
                            "kind": "slot",
                            "slot": "tokens",
                            "wire_type": "scalar_u32",
                        }
                    )
                set_grid(candidate, expression)
                validate_schema(candidate, self.manifest_schema, self.manifest_schema)
                expected = "divides by zero" if "divide zero" in name else ".*"
                with self.assertRaisesRegex(ContractError, expected):
                    validate_manifest_semantics(candidate)

        pathological: dict[str, Any] = {"kind": "const", "value": 1}
        for _ in range(1200):
            pathological = {
                "kind": "op",
                "op": "add",
                "args": [pathological, {"kind": "const", "value": 1}],
            }
        candidate = copy.deepcopy(self.manifest)
        set_grid(candidate, pathological)
        with self.assertRaisesRegex(ContractError, "nesting limit"):
            validate_schema(candidate, self.manifest_schema, self.manifest_schema)

    def test_contract_rejects_slot_abi_role_and_alias_failures(self) -> None:
        duplicate = copy.deepcopy(self.contract)
        duplicate["slots"].append(copy.deepcopy(duplicate["slots"][0]))
        validate_schema(duplicate, self.contract_schema, self.contract_schema)
        with self.assertRaisesRegex(ContractError, "duplicate contract slot"):
            validate_contract_semantics(duplicate)

        dangling = copy.deepcopy(self.contract)
        dangling["slots"][0]["aliasing"].append(
            {"mode": "may_alias", "target_slot": "missing"}
        )
        validate_schema(dangling, self.contract_schema, self.contract_schema)
        with self.assertRaisesRegex(ContractError, "missing contract slot"):
            validate_contract_semantics(dangling)

        bad_wire = copy.deepcopy(self.contract)
        bad_wire["slots"][0]["kind"] = "scalar_i32"
        bad_wire["slots"][0]["wire_type"] = "u64"
        bad_wire["slots"][0]["rank"] = 0
        bad_wire["slots"][0]["allowed_dtypes"] = []
        bad_wire["slots"][0]["allowed_quantizations"] = []
        bad_wire["slots"][0]["allowed_layouts"] = []
        validate_schema(bad_wire, self.contract_schema, self.contract_schema)
        with self.assertRaisesRegex(ContractError, "kind and wire"):
            validate_contract_semantics(bad_wire)

        writable_input = copy.deepcopy(self.contract)
        writable_input["slots"][0]["access"] = "write"
        validate_schema(writable_input, self.contract_schema, self.contract_schema)
        with self.assertRaisesRegex(ContractError, "writable"):
            validate_contract_semantics(writable_input)

        sparse = copy.deepcopy(self.contract)
        sparse["slots"][0]["abi_index"] = 3
        validate_schema(sparse, self.contract_schema, self.contract_schema)
        with self.assertRaisesRegex(ContractError, "dense ABI"):
            validate_contract_semantics(sparse)

        bad_lifetime = copy.deepcopy(self.contract)
        bad_lifetime["slots"][0].update(role="output", access="write", lifetime="model")
        validate_schema(bad_lifetime, self.contract_schema, self.contract_schema)
        with self.assertRaisesRegex(ContractError, "role and lifetime"):
            validate_contract_semantics(bad_lifetime)

        incompatible_alias = copy.deepcopy(self.contract)
        scalar = copy.deepcopy(incompatible_alias["slots"][0])
        scalar.update(
            slot_id="scalar",
            abi_index=1,
            kind="scalar_u64",
            wire_type="u64",
            role="input",
            access="read",
            lifetime="invocation",
            rank=0,
            alignment=8,
            allowed_dtypes=[],
            allowed_quantizations=[],
            allowed_layouts=[],
            aliasing=[{"mode": "must_alias", "target_slot": "input"}],
        )
        incompatible_alias["slots"][0]["aliasing"] = [
            {"mode": "must_alias", "target_slot": "scalar"}
        ]
        incompatible_alias["slots"].append(scalar)
        incompatible_alias["kernel_arguments"].append(
            {
                "abi_index": 1,
                "kind": "slot",
                "slot": "scalar",
                "wire_type": "scalar_u64",
            }
        )
        validate_schema(incompatible_alias, self.contract_schema, self.contract_schema)
        with self.assertRaisesRegex(ContractError, "incompatible wire"):
            validate_contract_semantics(incompatible_alias)

        bad_reset = copy.deepcopy(self.contract)
        bad_reset["state_reset"]["revision"] = 2
        with self.assertRaisesRegex(ContractError, "reset revision"):
            validate_contract_semantics(bad_reset)

        bad_scalar_alignment = copy.deepcopy(self.contract)
        bad_scalar_alignment["slots"][0].update(
            kind="scalar_u32",
            wire_type="u32",
            role="input",
            access="read",
            lifetime="invocation",
            rank=0,
            alignment=16,
            allowed_dtypes=[],
            allowed_quantizations=[],
            allowed_layouts=[],
        )
        bad_scalar_alignment["kernel_arguments"][0]["wire_type"] = "scalar_u32"
        validate_schema(
            bad_scalar_alignment, self.contract_schema, self.contract_schema
        )
        with self.assertRaisesRegex(ContractError, "natural C alignment"):
            validate_contract_semantics(bad_scalar_alignment)

        model_update = copy.deepcopy(self.contract)
        model_update["slots"][0].update(
            lifetime="model", graph_mutability="replay_update"
        )
        with self.assertRaisesRegex(ContractError, "model lifetime"):
            validate_contract_semantics(model_update)

        scratch_update = copy.deepcopy(self.contract)
        scratch_update["slots"][0].update(
            kind="scratch_ptr",
            role="scratch",
            access="read_write",
            lifetime="graph",
            graph_mutability="replay_update",
            allowed_dtypes=[],
            allowed_quantizations=[],
            allowed_layouts=[],
        )
        scratch_update["kernel_arguments"][0]["wire_type"] = "scratch_ptr"
        scratch_update["scratch_max_bytes"] = 16
        scratch_update["allowed_graph_capture"] = ["replay_update_safe"]
        with self.assertRaisesRegex(ContractError, "scratch slot"):
            validate_contract_semantics(scratch_update)

        frozen_invocation = copy.deepcopy(self.contract)
        frozen_invocation["allowed_graph_capture"] = ["replay_update_safe"]
        with self.assertRaisesRegex(ContractError, "freezes invocation"):
            validate_contract_semantics(frozen_invocation)

    def test_manifest_must_match_engine_owned_contract_exactly(self) -> None:
        mutations: list[tuple[str, Any]] = [
            (
                "contract digest",
                lambda value: value["choice_groups"][0]["contract"].__setitem__(
                    "sha256", "0" * 64
                ),
            ),
            (
                "effect authority",
                lambda value: value["variants"][0]["effects"][0].__setitem__(
                    "access", "read_write"
                ),
            ),
            (
                "kernel wire",
                lambda value: value["variants"][0]["launch"]["arguments"][0].__setitem__(
                    "wire_type", "state_ptr"
                ),
            ),
            (
                "dtype widening",
                lambda value: value["variants"][0]["constraints"]["dtypes"][0].__setitem__(
                    "allowed", ["f32"]
                ),
            ),
            (
                "scratch widening",
                lambda value: value["variants"][0]["scratch"].__setitem__("max_bytes", 1),
            ),
            (
                "resource widening",
                lambda value: value["variants"][0]["resources"].__setitem__(
                    "registers_per_thread_max", 65
                ),
            ),
            (
                "graph widening",
                lambda value: value["variants"][0].__setitem__(
                    "graph_capture", "capture_only"
                ),
            ),
            (
                "numerical widening",
                lambda value: value["variants"][0].__setitem__(
                    "numerical_class", {"kind": "diagnostic_only"}
                ),
            ),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.manifest)
                mutate(candidate)
                validate_schema(candidate, self.manifest_schema, self.manifest_schema)
                with self.assertRaises(ContractError):
                    validate_manifest_against_contract_registry(candidate, self.registry)

        raw_mismatch = {
            (self.contract["contract_id"], self.contract["revision"]): (
                self.contract,
                b'{"contract_abi":1}',
            )
        }
        with self.assertRaisesRegex(ContractError, "differs from raw"):
            validate_manifest_against_contract_registry(self.manifest, raw_mismatch)

        wrong_descriptor = copy.deepcopy(self.contract)
        wrong_descriptor["contract_id"] = "other.cuda.contract"
        wrong_descriptor["revision"] = 99
        wrong_raw = json.dumps(
            wrong_descriptor, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        wrong_registry = {
            (self.contract["contract_id"], self.contract["revision"]): (
                wrong_descriptor,
                wrong_raw,
            )
        }
        with self.assertRaisesRegex(ContractError, "registry key"):
            validate_manifest_against_contract_registry(self.manifest, wrong_registry)

        unknown_contract_field = copy.deepcopy(self.contract)
        unknown_contract_field["unknown_authority"] = "yes"
        unknown_raw = json.dumps(
            unknown_contract_field, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        unknown_manifest = copy.deepcopy(self.manifest)
        unknown_digest = domain_sha256("imparo-program-contract-v1", unknown_raw)
        for owner in (
            unknown_manifest["choice_groups"][0],
            unknown_manifest["variants"][0],
        ):
            owner["contract"]["sha256"] = unknown_digest
        unknown_registry = {
            (self.contract["contract_id"], self.contract["revision"]): (
                unknown_contract_field,
                unknown_raw,
            )
        }
        with self.assertRaisesRegex(ContractError, "unknown fields"):
            validate_manifest_against_contract_registry(
                unknown_manifest, unknown_registry
            )

        unsupported_contract = copy.deepcopy(self.manifest)
        for owner in (
            unsupported_contract["choice_groups"][0],
            unsupported_contract["variants"][0],
        ):
            owner["contract"]["id"] = "imparo.cuda.unsupported"
        validate_schema(
            unsupported_contract, self.manifest_schema, self.manifest_schema
        )
        with self.assertRaisesRegex(ContractError, "unsupported contract"):
            validate_manifest_against_contract_registry(
                unsupported_contract, self.registry
            )

    def test_toolchain_producer_union_is_strict_and_extensible(self) -> None:
        cuda_cpp = copy.deepcopy(self.manifest)
        cuda_cpp["toolchain"]["producer"] = {
            "kind": "cuda_cpp",
            "compiler_id": "nvcc",
            "compiler_version": "13.0",
            "cuda_toolkit": "13.0",
        }
        validate_schema(cuda_cpp, self.manifest_schema, self.manifest_schema)

        external = copy.deepcopy(self.manifest)
        external["toolchain"]["producer"] = {
            "kind": "external_aot",
            "producer_id": "org.example.aot",
            "producer_revision": "1" * 40,
            "source_sha256": "2" * 64,
            "toolchain_sha256": "3" * 64,
        }
        validate_schema(external, self.manifest_schema, self.manifest_schema)

        for name, mutate in (
            (
                "missing discriminant",
                lambda value: value["toolchain"]["producer"].pop("kind"),
            ),
            (
                "cross-producer field",
                lambda value: value["toolchain"]["producer"].__setitem__(
                    "compiler_id", "nvcc"
                ),
            ),
        ):
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.manifest)
                mutate(candidate)
                with self.assertRaises(ContractError):
                    validate_schema(candidate, self.manifest_schema, self.manifest_schema)

    def test_identity_and_entitlement_invariants_reject_duplicates(self) -> None:
        mutations: list[tuple[str, Any]] = [
            (
                "duplicate group",
                lambda value: value["choice_groups"].append(
                    copy.deepcopy(value["choice_groups"][0])
                ),
            ),
            (
                "duplicate variant",
                lambda value: value["variants"].append(copy.deepcopy(value["variants"][0])),
            ),
            (
                "duplicate config",
                lambda value: (
                    value["variants"].append(copy.deepcopy(value["variants"][0])),
                    value["variants"][1].__setitem__("variant_id", "9" * 64),
                ),
            ),
            (
                "community entitlement",
                lambda value: (
                    value["required_entitlement_features"].append("imparo.pro.pack"),
                    value["variants"][0]["required_entitlement_features"].append(
                        "imparo.pro.pack"
                    ),
                ),
            ),
            (
                "entitlement union mismatch",
                lambda value: value["required_entitlement_features"].append(
                    "imparo.pro.pack"
                ),
            ),
            (
                "unsorted commercial entitlements",
                lambda value: (
                    value.__setitem__("distribution", "commercial"),
                    value.__setitem__(
                        "required_entitlement_features",
                        ["imparo.pro.zeta", "imparo.pro.alpha"],
                    ),
                    value["variants"][0].__setitem__(
                        "required_entitlement_features",
                        ["imparo.pro.zeta", "imparo.pro.alpha"],
                    ),
                ),
            ),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.manifest)
                mutate(candidate)
                validate_schema(candidate, self.manifest_schema, self.manifest_schema)
                with self.assertRaises(ContractError):
                    validate_manifest_semantics(candidate)

    def test_workload_authority_binds_parameters_revision_and_fixture(self) -> None:
        bad_parameters = copy.deepcopy(self.manifest)
        bad_parameters["choice_groups"][0]["workload"]["parameters_sha256"] = "0" * 64
        validate_schema(bad_parameters, self.manifest_schema, self.manifest_schema)
        with self.assertRaisesRegex(ContractError, "parameter digest"):
            validate_manifest_semantics(bad_parameters)

        revision = copy.deepcopy(self.manifest)
        revision["choice_groups"][0]["workload"]["revision"] = 2
        with self.assertRaises(ContractError):
            validate_schema(revision, self.manifest_schema, self.manifest_schema)

        for field, value in (("fixture_sha256", "0" * 64),):
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.manifest)
                candidate["choice_groups"][0]["workload"][field] = value
                validate_schema(candidate, self.manifest_schema, self.manifest_schema)
                with self.assertRaisesRegex(ContractError, "workload is outside"):
                    validate_manifest_against_contract_registry(candidate, self.registry)

        narrow = copy.deepcopy(self.manifest)
        workload = narrow["choice_groups"][0]["workload"]
        workload["workload_id"] = "imparo.workload.narrow_mix"
        workload["parameters"] = {"narrow_tokens": 0}
        workload["parameters_sha256"] = canonical_json_sha256(
            "imparo-program-workload-parameters-v1", workload["parameters"]
        )
        with self.assertRaises(ContractError):
            validate_schema(narrow, self.manifest_schema, self.manifest_schema)

    def test_shape_alignment_and_scratch_closure(self) -> None:
        axis = copy.deepcopy(self.manifest)
        axis["variants"][0]["constraints"]["shapes"] = [
            {"slot": "input", "axis": 1, "min": 1, "max": 8, "multiple_of": 1}
        ]
        validate_schema(axis, self.manifest_schema, self.manifest_schema)
        with self.assertRaisesRegex(ContractError, "axis exceeds"):
            validate_manifest_against_contract_registry(axis, self.registry)

        unsatisfiable = copy.deepcopy(self.manifest)
        unsatisfiable["variants"][0]["constraints"]["shapes"] = [
            {"slot": "input", "axis": 0, "min": 3, "max": 3, "multiple_of": 2}
        ]
        validate_schema(unsatisfiable, self.manifest_schema, self.manifest_schema)
        with self.assertRaisesRegex(ContractError, "no satisfiable"):
            validate_manifest_semantics(unsatisfiable)

        duplicate_alignment = copy.deepcopy(self.manifest)
        duplicate_alignment["variants"][0]["constraints"]["alignments"].append(
            {"slot": "input", "bytes": 32}
        )
        validate_schema(duplicate_alignment, self.manifest_schema, self.manifest_schema)
        with self.assertRaisesRegex(ContractError, "alignments slot"):
            validate_manifest_semantics(duplicate_alignment)

        missing_scratch_slot = copy.deepcopy(self.contract)
        missing_scratch_slot["scratch_max_bytes"] = 1
        with self.assertRaisesRegex(ContractError, "exactly one scratch"):
            validate_contract_semantics(missing_scratch_slot)

        bad_zero_init = copy.deepcopy(self.contract)
        bad_zero_init["scratch_zero_initialized"] = True
        with self.assertRaisesRegex(ContractError, "requires initialization"):
            validate_contract_semantics(bad_zero_init)

        variant_zero_init = copy.deepcopy(self.manifest)
        variant_zero_init["variants"][0]["scratch"]["zero_initialized"] = True
        validate_schema(variant_zero_init, self.manifest_schema, self.manifest_schema)
        with self.assertRaisesRegex(ContractError, "initialization differs"):
            validate_manifest_against_contract_registry(variant_zero_init, self.registry)

    def test_dependency_and_alias_graph_rules(self) -> None:
        def two_groups() -> dict[str, Any]:
            candidate = copy.deepcopy(self.manifest)
            second_group = copy.deepcopy(candidate["choice_groups"][0])
            second_group["choice_group_id"] = "imparo.cuda.noop.second.v1"
            candidate["choice_groups"].append(second_group)
            second_variant = copy.deepcopy(candidate["variants"][0])
            second_variant["variant_id"] = "9" * 64
            second_variant["config_id"] = "8" * 64
            second_variant["choice_group_id"] = second_group["choice_group_id"]
            candidate["variants"].append(second_variant)
            return candidate

        duplicated = copy.deepcopy(self.manifest)
        ref = {"kind": "feature", "id": "imparo.feature.decode"}
        duplicated["variants"][0]["requires"] = [ref, copy.deepcopy(ref)]
        with self.assertRaises(ContractError):
            validate_schema(duplicated, self.manifest_schema, self.manifest_schema)

        contradictory = copy.deepcopy(self.manifest)
        contradictory["variants"][0]["requires"] = [ref]
        contradictory["variants"][0]["conflicts"] = [copy.deepcopy(ref)]
        validate_schema(contradictory, self.manifest_schema, self.manifest_schema)
        with self.assertRaisesRegex(ContractError, "both required"):
            validate_manifest_semantics(contradictory)

        alternatives = copy.deepcopy(self.manifest)
        alternatives["variants"][0]["provides"] = [ref]
        alternatives["variants"].append(copy.deepcopy(alternatives["variants"][0]))
        alternatives["variants"][1]["variant_id"] = "9" * 64
        alternatives["variants"][1]["config_id"] = "8" * 64
        validate_schema(alternatives, self.manifest_schema, self.manifest_schema)
        validate_manifest_semantics(alternatives)

        cycle = two_groups()
        cycle["variants"][0]["requires"] = [
            {"kind": "variant", "id": cycle["variants"][1]["variant_id"]}
        ]
        cycle["variants"][1]["requires"] = [
            {"kind": "variant", "id": cycle["variants"][0]["variant_id"]}
        ]
        validate_schema(cycle, self.manifest_schema, self.manifest_schema)
        with self.assertRaisesRegex(ContractError, "dependency cycle"):
            validate_manifest_semantics(cycle)

        group_cycle = two_groups()
        group_cycle["variants"][0]["requires"] = [
            {
                "kind": "choice_group",
                "id": group_cycle["variants"][1]["choice_group_id"],
            }
        ]
        group_cycle["variants"][1]["requires"] = [
            {
                "kind": "choice_group",
                "id": group_cycle["variants"][0]["choice_group_id"],
            }
        ]
        validate_schema(group_cycle, self.manifest_schema, self.manifest_schema)
        with self.assertRaisesRegex(ContractError, "dependency cycle"):
            validate_manifest_semantics(group_cycle)

        external_catalog_refs = copy.deepcopy(self.manifest)
        external_group = "org.example.external.group"
        external_catalog_refs["choice_groups"][0]["joint_with"] = [external_group]
        external_catalog_refs["variants"][0]["joint_with"] = [external_group]
        external_catalog_refs["variants"][0]["requires"] = [
            {"kind": "choice_group", "id": external_group}
        ]
        validate_schema(
            external_catalog_refs, self.manifest_schema, self.manifest_schema
        )
        validate_manifest_semantics(external_catalog_refs)

        asymmetric = two_groups()
        asymmetric["variants"][0]["conflicts"] = [
            {"kind": "variant", "id": asymmetric["variants"][1]["variant_id"]}
        ]
        validate_schema(asymmetric, self.manifest_schema, self.manifest_schema)
        with self.assertRaisesRegex(ContractError, "not symmetric"):
            validate_manifest_semantics(asymmetric)

        alias_contract = copy.deepcopy(self.contract)
        for index, slot_id in ((1, "output_a"), (2, "output_b")):
            slot = copy.deepcopy(alias_contract["slots"][0])
            slot.update(
                slot_id=slot_id,
                abi_index=index,
                role="output",
                access="write",
                aliasing=[],
            )
            alias_contract["slots"].append(slot)
            alias_contract["kernel_arguments"].append(
                {
                    "abi_index": index,
                    "kind": "slot",
                    "slot": slot_id,
                    "wire_type": "tensor_ptr",
                }
            )
        alias_contract["slots"][0]["aliasing"] = [
            {"mode": "must_alias", "target_slot": "output_a"}
        ]
        alias_contract["slots"][1]["aliasing"] = [
            {"mode": "must_alias", "target_slot": "input"},
            {"mode": "must_alias", "target_slot": "output_b"},
        ]
        alias_contract["slots"][2]["aliasing"] = [
            {"mode": "must_alias", "target_slot": "output_a"}
        ]
        validate_schema(alias_contract, self.contract_schema, self.contract_schema)
        with self.assertRaisesRegex(ContractError, "complete relation"):
            validate_contract_semantics(alias_contract)

    def test_signature_requires_fixed_fields_and_canonical_base64url(self) -> None:
        boolean_signature_schema = copy.deepcopy(self.signature)
        boolean_signature_schema["schema"] = True
        with self.assertRaises(ContractError):
            validate_schema(
                boolean_signature_schema, self.signature_schema, self.signature_schema
            )

        boolean_contract_abi = copy.deepcopy(self.contract)
        boolean_contract_abi["contract_abi"] = True
        with self.assertRaises(ContractError):
            validate_schema(
                boolean_contract_abi, self.contract_schema, self.contract_schema
            )

        cases = [
            ("hash newline", "manifest_sha256", "a" * 64 + "\n"),
            ("wrong domain", "domain", "imparo-cuda-release-v1"),
        ]
        for name, key, value in cases:
            with self.subTest(name=name):
                envelope = copy.deepcopy(self.signature)
                envelope[key] = value
                with self.assertRaises(ContractError):
                    validate_schema(envelope, self.signature_schema, self.signature_schema)

        noncanonical = copy.deepcopy(self.signature)
        noncanonical["signature"] = "A" * 85 + "B"
        validate_schema(noncanonical, self.signature_schema, self.signature_schema)
        with self.assertRaisesRegex(ContractError, "canonical"):
            validate_signature_envelope(noncanonical)


if __name__ == "__main__":
    unittest.main()
