from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .common import SYMBOL_RE, exact_keys, require_int, require_string
from .errors import BuildError

ALLOWED_TARGETS = {"cuda:80:32": 80, "cuda:86:32": 86}


@dataclass(frozen=True)
class AotSpec:
    kernel_name: str
    signature: tuple[str, ...]
    num_warps: int
    num_stages: int
    manifest: dict[str, Any]


@dataclass(frozen=True)
class CompileResult:
    cubin: bytes
    symbol: str
    registers_per_thread: int
    static_shared_bytes: int
    local_memory_bytes: int
    global_scratch_bytes: int
    profile_scratch_bytes: int


class Compiler(Protocol):
    def compile(
        self, source: Path, spec: AotSpec, target: str
    ) -> CompileResult: ...


def parse_target(target: str) -> int:
    try:
        return ALLOWED_TARGETS[target]
    except KeyError as error:
        raise BuildError(
            "target must be exactly cuda:80:32 or cuda:86:32"
        ) from error


def audit_source_imports(source: Path) -> None:
    try:
        text = source.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=source.name)
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise BuildError(f"parse Triton source {source}: {error}") from error
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            if name not in {"triton", "triton.language"} and not name.startswith(
                "triton.language."
            ):
                raise BuildError(
                    f"AOT source imports forbidden module {name!r}; "
                    "only triton and triton.language are allowed"
                )


def load_aot_spec(source: Path) -> AotSpec:
    audit_source_imports(source)
    if "torch" in sys.modules:
        raise BuildError("torch was already imported in the AOT builder process")
    module_name = f"_imparo_aot_{source.stem}"
    module_spec = importlib.util.spec_from_file_location(module_name, source)
    if module_spec is None or module_spec.loader is None:
        raise BuildError(f"cannot import AOT source {source}")
    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(source.parent))
    try:
        module_spec.loader.exec_module(module)
    except Exception as error:
        raise BuildError(f"execute AOT source {source.name}: {error}") from error
    finally:
        sys.path.pop(0)
    if "torch" in sys.modules:
        raise BuildError("AOT source or Triton imported torch")
    value = getattr(module, "IMPARO_AOT", None)
    if not isinstance(value, dict):
        raise BuildError("AOT source must export an IMPARO_AOT object")
    exact_keys(
        value,
        {"kernel", "signature", "num_warps", "num_stages", "manifest"},
        "IMPARO_AOT",
    )
    kernel_name = require_string(value["kernel"], "IMPARO_AOT.kernel")
    if not SYMBOL_RE.fullmatch(kernel_name):
        raise BuildError("kernel name must be an opaque ip_<sha256> symbol")
    if not hasattr(module, kernel_name):
        raise BuildError("IMPARO_AOT.kernel is not defined by the source")
    signature = value["signature"]
    if (
        not isinstance(signature, list)
        or not signature
        or len(signature) > 64
        or any(not isinstance(item, str) or not item for item in signature)
    ):
        raise BuildError("IMPARO_AOT.signature must be 1..64 non-empty strings")
    num_warps = require_int(value["num_warps"], "IMPARO_AOT.num_warps", 1)
    if num_warps not in {1, 2, 4, 8}:
        raise BuildError("num_warps must be one of 1, 2, 4, 8")
    num_stages = require_int(value["num_stages"], "IMPARO_AOT.num_stages", 1)
    if num_stages > 8:
        raise BuildError("num_stages exceeds the builder limit")
    manifest = value["manifest"]
    if not isinstance(manifest, dict):
        raise BuildError("IMPARO_AOT.manifest must be an object")
    return AotSpec(
        kernel_name=kernel_name,
        signature=tuple(signature),
        num_warps=num_warps,
        num_stages=num_stages,
        manifest=manifest,
    )


class TritonCompiler:
    """Small pinned adapter over Triton's compiler API; never its unstable CLI."""

    def __init__(self, expected_version: str | None = None) -> None:
        self.expected_version = expected_version

    def compile(self, source: Path, spec: AotSpec, target: str) -> CompileResult:
        sm = parse_target(target)
        try:
            import triton
            from triton.backends.compiler import GPUTarget
            from triton.compiler import ASTSource
        except ImportError as error:
            raise BuildError("the pinned Triton compiler is not installed") from error
        if self.expected_version is not None and triton.__version__ != self.expected_version:
            raise BuildError(
                f"installed Triton {triton.__version__!r} differs from locked "
                f"version {self.expected_version!r}"
            )
        if "torch" in sys.modules:
            raise BuildError("importing Triton pulled torch into the builder")

        module_name = f"_imparo_compile_{source.stem}"
        module_spec = importlib.util.spec_from_file_location(module_name, source)
        if module_spec is None or module_spec.loader is None:
            raise BuildError(f"cannot import AOT source {source}")
        module = importlib.util.module_from_spec(module_spec)
        sys.path.insert(0, str(source.parent))
        try:
            module_spec.loader.exec_module(module)
        except Exception as error:
            raise BuildError(f"execute AOT source {source.name}: {error}") from error
        finally:
            sys.path.pop(0)
        if "torch" in sys.modules:
            raise BuildError("AOT compilation imported torch")
        kernel = getattr(module, spec.kernel_name)
        signature: dict[str, str] = {}
        constants: dict[str, int | float] = {}
        attributes: dict[tuple[int, ...], list[list[object]]] = {}
        if len(kernel.arg_names) != len(spec.signature):
            raise BuildError("kernel arguments differ from IMPARO_AOT.signature")
        for index, encoded in enumerate(spec.signature):
            base, separator, hint = encoded.partition(":")
            if base.lstrip("-").replace(".", "", 1).isdigit():
                parsed: int | float
                parsed = float(base) if "." in base else int(base)
                constants[kernel.arg_names[index]] = parsed
                signature[kernel.arg_names[index]] = "constexpr"
            else:
                signature[kernel.arg_names[index]] = base
            if separator:
                if hint not in {"1", "16"}:
                    raise BuildError("signature hints are restricted to :1 or :16")
                if hint == "1":
                    constants[kernel.arg_names[index]] = 1
                    signature[kernel.arg_names[index]] = "constexpr"
                else:
                    attributes[(index,)] = [["tt.divisibility", 16]]
        try:
            with tempfile.TemporaryDirectory(
                prefix="imparo-triton-clean-"
            ) as cache_root:
                root = Path(cache_root)
                isolated = {
                    "TRITON_CACHE_DIR": str(root / "cache"),
                    "TRITON_DUMP_DIR": str(root / "dump"),
                    "TRITON_OVERRIDE_DIR": str(root / "override"),
                    "XDG_CACHE_HOME": str(root / "xdg"),
                }
                previous = {name: os.environ.get(name) for name in isolated}
                os.environ.update(isolated)
                try:
                    # Do not call JITFunction.create_binder(): it consults the
                    # active driver and binds the host GPU, which breaks
                    # GPU-less CI and cross-compiling SM80 on an SM86 worker.
                    source_ir = ASTSource(
                        fn=kernel,
                        constexprs=constants,
                        signature=signature,
                        attrs=attributes,
                    )
                    gpu_target = GPUTarget("cuda", sm, 32)
                    backend = triton.compiler.make_backend(gpu_target)
                    options = backend.parse_options(
                        {
                            "num_warps": spec.num_warps,
                            "num_stages": spec.num_stages,
                        }
                    )
                    compiled = triton.compile(
                        source_ir,
                        target=gpu_target,
                        options=options.__dict__,
                    )
                    cubin = compiled.asm[backend.binary_ext]
                finally:
                    for name, old_value in previous.items():
                        if old_value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = old_value
        except Exception as error:
            raise BuildError(f"pinned Triton compile failed: {error}") from error
        metadata = compiled.metadata
        symbol = str(getattr(compiled, "name", spec.kernel_name))
        if symbol != spec.kernel_name:
            raise BuildError(
                f"compiled symbol {symbol!r} differs from opaque source symbol"
            )
        registers, local_memory = _resource_usage(bytes(cubin), symbol)
        return CompileResult(
            cubin=bytes(cubin),
            symbol=symbol,
            registers_per_thread=registers,
            static_shared_bytes=int(getattr(metadata, "shared", 0)),
            local_memory_bytes=local_memory,
            global_scratch_bytes=int(
                getattr(metadata, "global_scratch_size", 0)
            ),
            profile_scratch_bytes=int(
                getattr(metadata, "profile_scratch_size", 0)
            ),
        )


def _resource_usage(cubin: bytes, symbol: str) -> tuple[int, int]:
    tool = os.environ.get("IMPARO_CUOBJDUMP", "cuobjdump")
    with tempfile.TemporaryDirectory(prefix="imparo-cuobjdump-") as directory:
        path = Path(directory) / "module.cubin"
        path.write_bytes(cubin)
        try:
            process = subprocess.run(
                [tool, "--dump-resource-usage", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as error:
            raise BuildError(f"launch pinned cuobjdump: {error}") from error
    if process.returncode != 0:
        raise BuildError(
            "pinned cuobjdump resource query failed: " + process.stderr.strip()
        )
    matches = list(
        re.finditer(
            r"(?m)^\s*Function\s+([^:\r\n]+):\s*$", process.stdout
        )
    )
    selected = None
    for index, match in enumerate(matches):
        if match.group(1).strip() != symbol:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        selected = process.stdout[match.end() : end]
        break
    if selected is None:
        raise BuildError("cuobjdump did not report the compiled opaque symbol")

    def field(name: str) -> int:
        match = re.search(rf"\b{name}\s*:\s*(\d+)\b", selected)
        if match is None:
            raise BuildError(f"cuobjdump omitted resource field {name}")
        return int(match.group(1))

    registers = field("REG")
    local = field("LOCAL")
    stack = field("STACK")
    if not 0 <= registers <= 255:
        raise BuildError("cuobjdump reported invalid register usage")
    return registers, local + stack
