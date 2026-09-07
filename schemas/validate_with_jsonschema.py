"""Validate examples and parity negatives with official Draft 2020-12 semantics."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError


ROOT = Path(__file__).resolve().parent
PAIRS = (
    ("program-pack-v1.schema.json", "examples/program-pack-v1.minimal.json"),
    ("program-contract-v1.schema.json", "examples/program-contract-v1.minimal.json"),
    (
        "program-pack-signature-v1.schema.json",
        "examples/program-pack-signature-v1.minimal.json",
    ),
)


def main() -> None:
    loaded: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for schema_name, example_name in PAIRS:
        schema = json.loads((ROOT / schema_name).read_bytes())
        example = json.loads((ROOT / example_name).read_bytes())
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(example)
        loaded[schema_name] = (schema, example)
        print(f"OK {schema_name} {example_name}")

    manifest_schema, manifest = loaded["program-pack-v1.schema.json"]
    contract_schema, contract = loaded["program-contract-v1.schema.json"]
    signature_schema, signature = loaded["program-pack-signature-v1.schema.json"]

    negatives: list[tuple[str, dict[str, object], dict[str, object]]] = []
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["schema"] = True
    negatives.append(("manifest boolean const", manifest_schema, bad_manifest))
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["pack_id"] = "com.example.pack\n"
    negatives.append(("manifest trailing newline", manifest_schema, bad_manifest))
    bad_manifest = copy.deepcopy(manifest)
    producer = bad_manifest["toolchain"]["producer"]
    producer["compiler_id"] = "nvcc"
    negatives.append(("producer union bleed", manifest_schema, bad_manifest))

    bad_contract = copy.deepcopy(contract)
    bad_contract["contract_abi"] = True
    negatives.append(("contract boolean const", contract_schema, bad_contract))
    bad_contract = copy.deepcopy(contract)
    bad_contract["contract_id"] = "imparo.cuda.noop\n"
    negatives.append(("contract trailing newline", contract_schema, bad_contract))

    bad_signature = copy.deepcopy(signature)
    bad_signature["schema"] = True
    negatives.append(("signature boolean const", signature_schema, bad_signature))
    bad_signature = copy.deepcopy(signature)
    bad_signature["manifest_sha256"] = "a" * 64 + "\n"
    negatives.append(("signature trailing newline", signature_schema, bad_signature))

    for name, schema, candidate in negatives:
        try:
            Draft202012Validator(schema).validate(candidate)
        except ValidationError:
            print(f"OK reject {name}")
        else:
            raise AssertionError(f"official validator accepted negative: {name}")


if __name__ == "__main__":
    main()
