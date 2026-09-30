#!/usr/bin/env python3
"""Shared pieces of the dev harness. Cross-platform: pure python, per-OS branches only
where the OS owns the measurement (process footprint).

Defaults point at this machine's reference model and fork; every tool takes overrides:
    --model PATH        or env IMPARO_MODEL
    --ref-model PATH    or env IMPARO_REF_MODEL
    --ref-engine PATH   or env IMPARO_REF_ENGINE   (llama-server-compatible binary)
"""
import os, subprocess, sys, tempfile
from pathlib import Path

def abspath_model(p):
    """A MODEL PATH MUST BE ABSOLUTE, because not every engine runs in this directory.

    logit_agree launches llama-server with cwd=reference_working_directory(FORK) -- the
    reference's own build tree -- so a relative path resolves against the wrong root there.
    The server exits immediately with "gguf_init_from_file: failed to open GGUF file", the
    harness then waits out its whole 120 s readiness loop, and the only thing it reports is
    a bare ConnectionRefusedError that names neither the path nor the reason. Resolving here
    means no caller can reintroduce it.
    """
    return os.path.abspath(os.path.expanduser(p)) if p else p

DEFAULT_MODEL = abspath_model(os.environ.get("IMPARO_MODEL",
    "~/Library/Application Support/Zeraix/llama/models/"
    "unsloth_gemma-4-E4B-it-qat-GGUF/8c5a9e4f/UD-Q4_K_XL/"
    "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"))
DEFAULT_REF_MODEL = abspath_model(os.environ.get("IMPARO_REF_MODEL", DEFAULT_MODEL))
# The reference is llama.cpp UPSTREAM as shipped. ~/Github/llama.cpp is an older,
# modified checkout: it was this default until 2026-09-01, and one A/B that omitted
# --ref-engine recorded its short-leg speed (795 tok/s) as upstream's (925). A different
# binary is a different reference; when one is wanted, name it on the command line.
DEFAULT_REF = os.environ.get("IMPARO_REF_ENGINE",
    os.path.expanduser("~/Github/llama.cpp-upstream/build/bin/llama-server"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def bin_path(name):
    """target/release/<name>, or IMPARO_BIN_DIR/<name> when set -- an A/B against a saved
    build interleaves the two binaries under one harness instead of trusting two batches."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    return os.path.join(os.environ.get("IMPARO_BIN_DIR") or os.path.join(ROOT, "target", "release"), exe)

def toks(n):
    """The harness token pattern every gate and drift record was measured on."""
    return [2 if i == 0 else 1000 + i % 500 for i in range(n)]

def kv_sides(spec):
    """Return (K, V) for `type` or `k_type,v_type` harness syntax."""
    parts = spec.split(",")
    if len(parts) == 1 and parts[0]:
        return parts[0], parts[0]
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1]
    raise ValueError(f"invalid KV type specification: {spec!r}")

def reference_kv_args(spec):
    """llama.cpp arguments for a shared or per-side KV cache type."""
    ck, cv = kv_sides(spec)
    if ck == cv == "f16":
        return []
    return ["-ctk", ck, "-ctv", cv]

def reference_cuda_args(spec):
    """Canonical llama CUDA correctness arguments.

    Flash Attention is explicit even for f16. Relying on a binary's current default
    makes a recorded oracle ambiguous and previously let nominally identical gate runs
    exercise different kernels after a reference build/config change. Context
    checkpoints are a llama-server service policy, not model arithmetic: for hybrid
    recurrent models the default policy peels a four-token tail off the prompt. Disable
    it so an engine gate compares the same absolute prefill grid on both sides.
    """
    return ["-fa", "on", "-ctxcp", "0"] + reference_kv_args(spec)


def reference_working_directory(executable):
    """Return the authenticated directory used for llama backend discovery."""
    override = os.environ.get("IMPARO_REF_CWD")
    if override:
        return os.path.abspath(override)
    return os.path.dirname(os.path.abspath(executable))

def validate_correctness_gate_environment(model, reference):
    """Fail closed before a release correctness gate can exercise safe defaults.

    Ordinary diagnostic runs remain flexible.  In correctness-gate mode the runtime
    reads only ``IMPARO_HOST_CONFIG`` and the llama executable is meaningful only as
    part of its authenticated DLL bundle.  A misspelled config variable or a nearby
    ad-hoc bundle otherwise produces plausible output from the wrong numerical route.
    """
    if os.environ.get("IMPARO_CORRECTNESS_GATE") != "1":
        return
    if os.environ.get("IMPARO_TUNE_CONFIG"):
        raise RuntimeError(
            "IMPARO_TUNE_CONFIG is not a runtime input; use IMPARO_HOST_CONFIG"
        )
    config = os.environ.get("IMPARO_HOST_CONFIG")
    if not config:
        raise RuntimeError("IMPARO_HOST_CONFIG is required in correctness-gate mode")
    if not Path(config).is_file():
        raise RuntimeError(f"IMPARO_HOST_CONFIG does not exist: {config}")
    if not Path(model).is_file():
        raise RuntimeError(f"model does not exist: {model}")

    expected_manifest_sha256 = os.environ.get("IMPARO_REF_MANIFEST_SHA256", "")
    if len(expected_manifest_sha256) != 64:
        raise RuntimeError(
            "IMPARO_REF_MANIFEST_SHA256 is required in correctness-gate mode"
        )
    manifest_path = Path(os.environ.get(
        "IMPARO_REF_MANIFEST",
        Path(__file__).resolve().parent / "refs" / "oracles"
        / "llama-4695f001-windows-sm86.json",
    ))
    # Keep one authority for inventory, path and SHA validation.  seal_receipt does
    # not import this module, so the lazy import cannot form a cycle.
    from seal_receipt import _authenticate_oracle_bundle, _load_oracle_manifest
    manifest, manifest_sha256 = _load_oracle_manifest(manifest_path)
    if manifest_sha256 != expected_manifest_sha256.lower():
        raise RuntimeError(
            "oracle manifest SHA256 does not match IMPARO_REF_MANIFEST_SHA256"
        )
    _authenticate_oracle_bundle(Path(reference).resolve(), manifest)

def validate_correctness_gate_output(output, environment):
    """Reject evidence from safe defaults when an explicit CUDA gate config was requested."""
    if (environment.get("IMPARO_CORRECTNESS_GATE") != "1"
            or environment.get("IMPARO_GPU") == "0"):
        return
    if "using safe defaults" in output or "isolated CUDA gate candidate rejected" in output:
        raise RuntimeError("correctness gate config was rejected; refusing safe-default evidence\n"
                           + output[-8000:])
    expected = environment.get("IMPARO_HOST_CONFIG")
    marker = "[imparo] host config loaded from "
    loaded = [line[len(marker):].strip() for line in output.splitlines()
              if line.startswith(marker)]
    normalize = lambda path: os.path.normcase(os.path.abspath(path))
    if not expected or not loaded or any(normalize(path) != normalize(expected) for path in loaded):
        raise RuntimeError("correctness gate did not confirm the requested host config\n"
                           + output[-8000:])


def apply_imparo_kv_env(env, spec):
    """Apply a KV spec without inheriting stale per-side values from the parent."""
    env.pop("IMPARO_CTK", None)
    env.pop("IMPARO_CTV", None)
    ck, cv = kv_sides(spec)
    if ck != "f16":
        env["IMPARO_CTK"] = ck
    if cv != "f16":
        env["IMPARO_CTV"] = cv
    return env

def run_forward_process(prefix, model, n, **kwargs):
    """Run imparo-forward with one cross-platform token transport.

    Long contexts exceed Windows' process command-line limit when every token is an
    argv entry. imparo-forward already accepts -t FILE; keep the choice here so
    prefill and decode gates cannot silently diverge in how they launch the same
    artifact.
    """
    token_ids = toks(n)
    argv = [bin_path("imparo-forward")] + list(prefix) + [model]
    if n < 4096:
        return subprocess.run(argv + [str(token) for token in token_ids], **kwargs)

    path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", suffix=".tokens", delete=False
        ) as token_file:
            path = token_file.name
            token_file.write(" ".join(map(str, token_ids)))
        return subprocess.run(argv + ["-t", path], **kwargs)
    finally:
        if path is not None:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

def run_forward(model, n, env_extra=None, timeout=900):
    """One imparo-forward run over toks(n); returns (top10_line, full_stdout)."""
    env = dict(os.environ)
    env["IMPARO_GPU"] = env.get("IMPARO_GPU", "1")
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    process = run_forward_process(
        [], model, n, capture_output=True, text=True, env=env, timeout=timeout
    )
    out = process.stdout
    if process.returncode != 0:
        return None, out
    top = next((l for l in out.splitlines() if l.startswith("top10")), None)
    return top, out

def run_forward_reps(model, n, env_extra=None, reps=1, timeout=900):
    """imparo-forward --repeat: `reps` forwards in ONE process (Metal init amortized).
    Returns (list of top10 lines, full stdout). The det gate combines this with
    multiple processes so both determinism classes stay covered."""
    env = dict(os.environ)
    env["IMPARO_GPU"] = env.get("IMPARO_GPU", "1")
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    process = run_forward_process(
        ["--repeat", str(reps)], model, n,
        capture_output=True, text=True, env=env, timeout=timeout
    )
    out = process.stdout
    if process.returncode != 0:
        return [], out
    tops = [l for l in out.splitlines() if l.startswith("top10")]
    return tops, out

# KV DISK TIERS -- both engines keep one, and neither reaps it.
#
#     ~/.imparo/kv                    imparo's spill tier
#     ~/Library/Caches/llama.cpp/kv   the fork's
#
# A killed server leaves its directory behind, and a bracket leg kills a server every time.
# Across one working session that reached 4.5 GB for imparo and 2.5 GB for the fork, and it
# filled the disk -- which surfaced as `ld: write() failed, errno=28`, a LINKER error with
# no mention of space, on a build that had nothing wrong with it.
#
# Reaps only what a run CREATED: snapshot the entries first, remove the new ones after. The
# caches belong to whoever is using these engines, and a test harness has no business
# deleting a cache it did not make.
KV_TIERS = [os.path.expanduser("~/.imparo/kv"),
            os.path.expanduser("~/Library/Caches/llama.cpp/kv")]

def kv_tier_snapshot():
    """Entries present before a run, per tier."""
    seen = {}
    for d in KV_TIERS:
        try:
            seen[d] = set(os.listdir(d))
        except OSError:
            seen[d] = set()
    return seen

def kv_tier_reap(before, verbose=True):
    """Remove entries that appeared since `before`. Returns bytes reclaimed."""
    import shutil
    freed = 0
    for d, had in before.items():
        try:
            now = set(os.listdir(d))
        except OSError:
            continue
        for name in now - had:
            path = os.path.join(d, name)
            try:
                for root, _, files in os.walk(path):
                    for f in files:
                        try:
                            freed += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
                shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass
    if verbose and freed:
        print(f"  [harness] reaped {freed / 1e9:.2f} GB of KV disk tier", flush=True)
    return freed

def vm_stat_mib():
    """macOS vm_stat as MiB: wired, filebacked, anon, free. None elsewhere.

    The machine-level ruler, because phys_footprint misses most of what a GPU engine
    costs: on Apple unified memory the driver WIRES the pages the GPU uses, whether
    they are an engine's private buffer or a file mapping, and a mapping is never
    charged to the process -- measured 2026-09-11 on Qwen3.8-27B, each engine alone
    from a purged cache: llama.cpp wired +15204 MiB with a 2.3 GiB footprint, imparo
    wired +15409 MiB with a 16 GiB footprint. Absolute values are ambient -- only
    deltas between samples mean anything, and only against a SETTLED baseline
    (wired_settle): the previous engine's pages are still being unwired for seconds
    after it exits, and a baseline taken inside that window understated the reference
    by 9 GiB. `filebacked` is what Activity Monitor shows as Cached Files; an engine
    that mmaps leaves its whole model there after it exits."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return None
    import re
    page = re.search(r"page size of (\d+) bytes", out)
    if not page:
        return None
    scale = int(page.group(1)) / (1024 * 1024)
    fields = {"wired": r"Pages wired down:\s+(\d+)", "filebacked": r"File-backed pages:\s+(\d+)",
              "anon": r"Anonymous pages:\s+(\d+)", "free": r"Pages free:\s+(\d+)"}
    got = {}
    for k, pat in fields.items():
        m = re.search(pat, out)
        if not m:
            return None
        got[k] = int(m.group(1)) * scale
    return got

def wired_mib():
    """System-wide wired memory (MiB); see vm_stat_mib."""
    v = vm_stat_mib()
    return v["wired"] if v else None

def wired_settle(max_wait=30.0, tol_mib=64.0):
    """Wait until system wired memory stops moving (two readings a second apart within
    tol_mib), up to max_wait; returns the settled vm_stat_mib() dict, or None off macOS.
    An engine's pages are unwired for seconds after its process exits; a baseline taken
    before that finishes credits the next engine with a smaller delta than it costs."""
    v = vm_stat_mib()
    if v is None:
        return None
    import time
    deadline = time.time() + max_wait
    while time.time() < deadline:
        time.sleep(1.0)
        n = vm_stat_mib()
        if n is None:
            return v
        if abs(n["wired"] - v["wired"]) <= tol_mib:
            return n
        v = n
    return v

def purge_page_cache():
    """Drop the page cache (macOS `purge`) so a leg's baseline holds no other engine's
    file pages; never prompts (sudo -n), so it is a no-op without a NOPASSWD rule for
    /usr/sbin/purge. Returns True when it ran."""
    if sys.platform != "darwin":
        return False
    try:
        return subprocess.run(["sudo", "-n", "purge"], capture_output=True, text=True,
                              timeout=60).returncode == 0
    except Exception:
        return False

def gpu_mem_sample():
    '''Machine-level GPU memory as ("wired"|"vram", MiB), or None.

    Two rulers because the pools differ. Apple unified memory: the driver WIRES system
    pages while the GPU uses them, so the machine cost is vm_stat's wired delta. A
    discrete CUDA GPU has its own pool: the machine cost is device memory used, from
    nvidia-smi -- engine-agnostic, so imparo-cuda and a reference CUDA server read the
    same. Both are ambient absolutes: only deltas between samples mean anything.
    (nvidia-smi branch is written blind on a Mac -- unverified until a CUDA host runs it.)'''
    w = wired_mib()
    if w is not None:
        return ("wired", w)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        vals = [float(x) for x in out.split()]
        if vals:
            return ("vram", sum(vals))
    except Exception:
        pass
    return None

def footprint_mib(pid):
    """What the OS charges the process. Per-OS: macOS phys_footprint, Windows private
    bytes (ctypes psapi, no dependency), Linux VmRSS."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True,
                                 text=True, timeout=30).stdout
        except Exception:
            return None
        import re
        m = (re.search(r"([\d.]+)\s*([KMG])B\s+phys_footprint", out)
             or re.search(r"phys_footprint:\s*([\d.]+)\s*([KMG])", out))
        if not m:
            return None
        v, unit = float(m.group(1)), m.group(2)
        return v * {"K": 1 / 1024, "M": 1, "G": 1024}[unit]
    if sys.platform == "win32":
        import ctypes, ctypes.wintypes as wt
        class PMC_EX(ctypes.Structure):
            _fields_ = [(n, ctypes.c_size_t if "Size" in n or "Usage" in n else wt.DWORD)
                        for n in ("cb", "PageFaultCount")] + \
                       [(n, ctypes.c_size_t) for n in (
                        "PeakWorkingSetSize", "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED
        if not h:
            return None
        pmc = PMC_EX(); pmc.cb = ctypes.sizeof(PMC_EX)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        ctypes.windll.kernel32.CloseHandle(h)
        return pmc.PrivateUsage / (1024 * 1024) if ok else None
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None
