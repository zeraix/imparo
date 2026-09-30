#!/usr/bin/env python3
"""Interleaved speed + memory bracket: imparo vs a reference engine, same cache type,
cold prompts, one session. The ONLY comparison discipline this repo accepts for a
speed or memory claim.

    bracket.py                     # 2 long + 1 short + 1 DEEP pair, f16
    bracket.py --kv q4_0 --pairs 3
    bracket.py --engines imparo    # one engine only (A/B by env between runs)

THREE PROMPT REGIMES, because the answer differs by regime and quoting one as though it
were the whole picture is how a real advantage gets missed. Roughly 450 / 5.6k / 16k
tokens, matching the table in STATUS:

    short   dispatch efficiency dominates
    long    the mid-length turn; the NARROWEST cell, and the honest one to quote alone
    deep    attention dominates, and quantized-KV decode at depth is where this engine
            is furthest ahead -- +21% (q8_0) and +24% (q4_0) against the fork at 16310
            tokens, against +0.3% for f16 at 5651

A run that omits `deep` cannot see that, and one that omits `short` cannot see dispatch
cost. All three are on by default for that reason.

ENGINES: `imparo` and `ref` (a llama-server binary) as before, `omlx` (an oMLX server this
harness starts and stops like the others, with the model directory's name from
--omlx-model), and `NAME@URL` for an OpenAI-compatible server started elsewhere (rapid-mlx:
`rapid-mlx serve models/Qwen3.8-27B-MLX-4bit --port 8010`, then `rapid-mlx@http://127.0.0.1:8010`
with --url-model). One server per leg, so a 15 GB model is never resident twice.

TWO CLOCKS ON EVERY LINE. imparo and llama-server report their own `timings`, and those
server-side numbers are the kernel-attribution numbers every earlier record was taken on;
they stay in prefill_median/decode_median with clock=server. An MLX server reports none,
so its prefill_median/decode_median are the client clock (clock=client). And every engine,
whatever its primary clock, carries client_prefill/client_decode: the same streaming
request measured on this side of the socket, prefill = prompt_tokens / time-to-first-token
and decode = tokens / (last chunk - first chunk). Compare engines across the client
columns; compare imparo against its own history on the server ones.

THE FIRST PACKET IS NOT ALWAYS ONE TOKEN. oMLX streams two tokens per SSE frame on
Qwen3.8-27B; imparo and llama-server stream one. Both client-side rates assume one unless
corrected: the decode window then covers ct - c1 tokens, not ct - 1, and the first packet
arrives c1 - 1 decode steps late, which a naive prompt / TTFT charges to prefill. c1 is
estimated as the mean tokens per frame (tpc=, printed so a non-uniform stream is visible):
decode = (ct - c1) / window, prefill = prompt / (ttft - (c1 - 1) / decode). At c1 = 1 both
are the old formulas exactly. (2026-09-09: oMLX's 27B decode read 8.9 uncorrected, 8.6
corrected; its prefill 91.4 -> 95.1.)

Per leg it reports the medians on both clocks, the t+0 footprint and the RESTING footprint
after --rest seconds idle. Unique prompts per request so nothing is served from any cache;
a leg is rejected when a later run reports cached tokens AND prefills faster than the leg's
cold first run (a count alone is not enough: oMLX counts its own retried prefill there;
finish_leg says why). --thinking sends the same
enable_thinking to every engine (default off), because the servers' template defaults
differ and a thinking model's held-back opener tokens would be charged to one arm only.
"""
import argparse, os, re, shutil, subprocess, sys, time, urllib.error, urllib.request
from harness import (DEFAULT_MODEL, DEFAULT_REF, RESULTS, bin_path, footprint_mib,
                     gpu_mem_sample, kv_tier_reap, kv_tier_snapshot, purge_page_cache,
                     reference_kv_args, vm_stat_mib, wired_settle)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Server

# MEASURED tokens per prompt unit: 17.1 on gemma-4-E4B (5651/330), 18.1 on LFM2.5
# (5963/330). 20 is the CEILING used to size the context, not an estimate of the truth:
# undersizing it makes the server truncate, and a truncated prompt still returns a
# plausible number -- a measurement of a shorter prompt wearing a longer prompt's label.
#
# It is also not free to overshoot. At 24 the long regime derived a 9216-token context
# where every previously recorded long number was measured at 8192, which would have
# changed the KV allocation and made the new numbers incomparable to the old ones for a
# reason that has nothing to do with the engines. 20 keeps long at the 8192 floor and
# gives the deep regime 20480. `ctx_for` is bounded either way by the leg's own guard.
TOKENS_PER_UNIT_MAX = 20


def ctx_for(a, words):
    """Context this leg needs, rounded up to a whole 1024.

    DERIVED PER LEG rather than one --ctx for every regime, because the context size is
    itself a variable: a bigger KV allocation is a different amount of memory to touch.
    Sizing every leg for the deepest one would move the short and long numbers and make
    them incomparable to every result recorded before the deep regime existed.
    """
    need = words * TOKENS_PER_UNIT_MAX + a.max_tokens + 512
    return max(a.ctx, -(-need // 1024) * 1024)


# oMLX refuses every request without an API key (401 on /v1/models too); the harness's own
# server gets this one. Its prefix cache survives restarts under ~/.omlx/cache; every prompt
# here is unique, so it can serve nothing, but a run must leave the disk as it found it.
OMLX_API_KEY = os.environ.get("MLX_BRACKET_API_KEY", "bracket")
OMLX_CACHE = os.path.expanduser("~/.omlx/cache")
OMLX_BIN = os.path.expanduser("~/.omlx-build/venv/bin/omlx")
OMLX_MODELS = os.path.expanduser("~/.omlx/models")


def omlx_model_root_and_id(models_dir, model):
    """The --model-dir to serve and the model id inside it.

    oMLX discovers models as the subdirectories of ONE --model-dir, so a model that does not
    live under OMLX_MODELS is named by its path instead: the parent becomes the root and the
    last component the id. The 27B's MLX weights sit in the repo beside their GGUF, so both
    engines read one copy of those bytes and no second copy has to be kept in sync.
    """
    if model and (os.sep in model or os.path.isdir(os.path.expanduser(model))):
        path = os.path.abspath(os.path.expanduser(model.rstrip(os.sep)))
        return os.path.dirname(path), os.path.basename(path)
    return models_dir, model


def clear_omlx_cache(when):
    if os.path.isdir(OMLX_CACHE):
        shutil.rmtree(OMLX_CACHE, ignore_errors=True)
        print(f"  omlx cache cleared ({when}): {OMLX_CACHE}", flush=True)


def client_rates(r):
    """Prefill and decode tok/s on the CLIENT clock from one streamed reply `r`
    (Server.chat's dict), with the first-packet correction described in the module doc.

    Returns (prefill, decode, tokens_per_frame, ttft_ms); NaN where the reply cannot
    support the number (no content, one frame)."""
    u = r.get("usage") or {}
    pt = u.get("prompt_tokens", 0)
    ct = u.get("completion_tokens", r.get("chunks", 0))
    frames = r.get("chunks", 0)
    ttft, last = r.get("ttft_s"), r.get("last_s")
    if not pt or ttft is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    tpc = ct / frames if frames else 1.0
    c1 = max(1, round(tpc))
    window = (last - ttft) if last is not None else 0.0
    dec = (ct - c1) / window if ct > c1 and window > 0 else float("nan")
    ttft_1 = ttft - (c1 - 1) / dec if (c1 > 1 and dec == dec and dec > 0) else ttft
    return pt / ttft_1, dec, tpc, ttft * 1e3


def ref_speed_args(a):
    """Reference arguments for a SPEED comparison -- deliberately NOT
    `reference_cuda_args`, which is the CORRECTNESS set.

    The correctness set passes `-ctxcp 0`, and that is right for an oracle: the
    checkpoint policy peels a four-token tail off the prompt, so with it on, the two
    engines prefill different absolute grids and the comparison is not of the same work.

    IT IS WRONG HERE. Context checkpoints are how llama.cpp restores state outside the
    sliding window instead of re-prefilling -- the same job our KV pool checkpoints do,
    and ours are ON. Turning theirs off compares an engine carrying that machinery
    against one that is not:

        E4B 446-token prefill    -ctxcp 0   607 tok/s
                                 -ctxcp 1   506 tok/s      ~17%

    AND IT FLATTERS THE REFERENCE, so it was making imparo look worse, not better. That
    is the safe direction for a claim and still the wrong measurement. The default (32)
    is what llama.cpp users actually run, so it is what we compare against; `-fa on`
    stays, as a pin against the reference silently changing kernels between runs rather
    than a change of behaviour -- it is what `auto` resolves to on this hardware.

    `--ref-ctxcp 0` still reaches the old configuration, because both numbers are worth
    having as long as each is labelled with the configuration that produced it.
    """
    return ["-fa", "on", "-ctxcp", str(a.ref_ctxcp)] + reference_kv_args(a.kv)


def build_cmd(target, a, ctx):
    kv_ref = ref_speed_args(a)
    kv_imp = [] if a.kv == "f16" else ["--cache-type-k", a.kv, "--cache-type-v", a.kv]
    if target == "omlx":
        # Its server defaults, as a user runs it: no context flag (the model's own), the
        # prefix cache on (cleared around the leg, see clear_omlx_cache).
        return [a.omlx_bin, "serve", "--model-dir", a.omlx_models_dir,
                "--port", str(a.port), "--api-key", OMLX_API_KEY]
    if target == "ref":
        return [a.ref_engine, "-m", a.ref_model or a.model, "-c", str(ctx), "-ngl", "999",
                "--port", str(a.port), "--jinja", "--parallel", "1", "--no-mmproj",
                "-ub", str(a.ubatch), "-b", str(max(a.ubatch, 2048))] + kv_ref
    # --parallel 1 TO MATCH THE REFERENCE. llama.cpp is given `--parallel 1` above; imparo
    # defaults to 8 co-batched slots, and on an architecture where the co-batched step IS
    # built those slots carry their own rings and recurrent buffers. A one-client bracket
    # would then compare an engine holding eight slots' state against one holding one.
    # It is a no-op for lfm2moe (co-batched decode is not built for it, so the slot count
    # falls back to 1 either way, measured 2026-09-22), and it is not a no-op for E4B or the
    # 27B, which is why it belongs here rather than in a per-model note.
    return [bin_path("imparo-server"), "-m", a.model, "-c", str(ctx),
            "--port", str(a.port), "--parallel", "1"] + kv_imp

def ref_label(path):
    """The checkout and build directory of a llama-server binary, e.g.
    `llama.cpp-upstream/build` -- enough to tell two builds apart in a result line."""
    parts = os.path.abspath(path).split(os.sep)
    return "/".join(parts[-4:-2]) if len(parts) >= 4 else path


def other_gpu_clients():
    """Names of browser GPU processes alive right now (Chromium family `--type=gpu-process`,
    WebKit `com.apple.WebKit.GPU`). Not a benchmark check but the reason a leg reads 7%
    slower with nothing else changed: on 2026-09-02 a Chrome tab's GPU process took the
    GPU in bursts every couple of minutes (GPU at its top clock and 0% idle but +25% power,
    engines -7% prefill / -17% decode); with Chrome closed, 30 requests had no slow one.
    A browser that is open but idle can still be the one -- the field says whether it
    was there, the per-run numbers say whether it woke up."""
    try:
        out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return ["?"]
    names = set()
    for line in out.splitlines():
        if "--type=gpu-process" in line or "com.apple.WebKit.GPU" in line:
            m = re.search(r"/([^/]+)\.app/", line)
            names.add((m.group(1) if m else line.split()[0]).replace(" ", "_"))
    return sorted(names)


def mid(xs):
    """True median of a sorted list.

    `sorted[len//2]` is NOT a median on an even count: it returns the upper of the two
    middles, and both metrics here are tok/s, so upper = faster. Every even-repeat leg was
    reported at its better half.
    """
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def spread(xs):
    """Run-to-run dispersion, printed next to the median. A leg whose runs disagree by
    more than the effect under test cannot decide that test, and that has to be visible in
    the line itself rather than inferred by whoever reads the log."""
    return (xs[-1] - xs[0]) / max(xs[-1], 1e-9)


def measure(srv, text, max_tokens, conv, template_kwargs):
    """One streamed request, both clocks. Returns (prompt_tokens, cached_tokens,
    server_prefill, server_decode, client_prefill, client_decode, tokens_per_frame,
    ttft_ms); the server pair is NaN for an engine that sends no `timings`."""
    try:
        r = srv.chat([{"role": "user", "content": text}], conv, max_tokens=max_tokens,
                     template_kwargs=template_kwargs)
    except urllib.error.HTTPError as e:
        # A DIAGNOSTIC ARM PRODUCES GARBAGE TOKENS ON PURPOSE. With attention skipped the
        # model emits nonsense, and llama-server's chat parser answers 500 -- AFTER the
        # prefill it was asked to time is already done and logged. Raising there ends the
        # leg at one sample and, worse, makes the skipped arm impossible to repeat while
        # the unskipped arm repeats freely: two arms, two sample counts, one comparison.
        # The prefill number for those arms is read from the server log (pfill.py), so the
        # response body is not needed; refuse to swallow anything else.
        if e.code == 500 and os.environ.get("IMPARO_TOLERATE_GARBAGE") == "1":
            print(f"  [tolerated 500 on {conv}: skipped-arm output is garbage by design]",
                  flush=True)
            nan = float("nan")
            return 0, 0, nan, nan, nan, nan, nan, nan
        raise
    u = r.get("usage") or {}
    t = r.get("timings") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    spf = t.get("prompt_per_second", float("nan")) if t else float("nan")
    sdc = t.get("predicted_per_second", float("nan")) if t else float("nan")
    cpf, cdc, tpc, ttft_ms = client_rates(r)
    return u.get("prompt_tokens", 0), cached, spf, sdc, cpf, cdc, tpc, ttft_ms


def prompt_text(tag, i, words):
    """A unique prefix, then the repeated unit every recorded ptok was measured on."""
    return (f"{tag}{i}-{time.time_ns()} " +
            "The unified KV pool stores blocks and a conversation is an "
            "ordered list of block ids. " * words)


def parse_engine(spec):
    """'imparo' / 'ref' / 'omlx' -> (kind, name, url); 'NAME@URL' -> ('url', NAME, URL)."""
    if "@" in spec:
        name, url = spec.split("@", 1)
        return "url", name, url.rstrip("/")
    return spec, spec, None


def leg(target, a, words, tag):
    import tempfile
    env = dict(os.environ)
    # A FRESH KV STORE PER LEG, FOR BOTH ENGINES. Only imparo got one; the fork was left on
    # its default, which is the user's own cache directory (its disk tier defaults ON --
    # `--no-kv-disk` is the opt-out). So the legs were not symmetric: imparo started empty
    # every leg while the fork carried whatever 2.5 GB of state was already there, and the
    # harness wrote benchmark conversations into a directory it does not own.
    #
    # It also CRASHED the fork. On the second long request it erased a checkpoint reading
    # `n_tokens = -1, size = 9391.101 MiB` and the connection dropped. Against a fresh
    # directory the same three requests survive, so the stale state is what it cannot
    # digest -- a real fork bug, and one this harness was feeding.
    kind, name, url = parse_engine(target)
    own_kv = None
    if kind == "imparo":
        env["IMPARO_GPU"] = "1"
        env.setdefault("IMPARO_BATCH", str(a.ubatch))
        env.setdefault("IMPARO_KV_DIR", tempfile.mkdtemp(prefix="imparo-bracket-kv-"))
        # AND MUST NOT OUTLIVE THE LEG. This directory was created per leg and never
        # removed: 96 of them, 48 GB, filled the disk during one working session, and the
        # symptom was `ld: write() failed, errno=28` -- a LINKER error on an unrelated
        # build. Recorded in `own_kv` and removed in the finally below.
        own_kv = env["IMPARO_KV_DIR"] if "imparo-bracket-kv-" in env["IMPARO_KV_DIR"] else None
    elif kind == "ref":
        env["LLAMA_KV_DISK_DIR"] = tempfile.mkdtemp(prefix="llama-bracket-kv-")
        own_kv = env["LLAMA_KV_DISK_DIR"]
    elif kind == "omlx":
        clear_omlx_cache("before")
    template_kwargs = ({"enable_thinking": a.thinking == "on"} if a.thinking != "unset"
                       else None)
    # Both engines keep a KV disk tier and neither reaps it -- imparo at ~/.imparo/kv, the
    # fork at ~/Library/Caches/llama.cpp/kv. Snapshot now, remove only what THIS leg adds:
    # the caches belong to whoever uses these engines, and a harness has no business
    # deleting one it did not make.
    tiers_before = kv_tier_snapshot()
    os.makedirs(RESULTS, exist_ok=True)
    log = os.path.join(RESULTS, f"bracket-{name}.server.log")
    # AN ENGINE THIS HARNESS DID NOT START. Its memory is already resident, so the purged
    # baseline and the wired delta would describe nothing; it is measured on the client
    # clock alone and the line says lifecycle=external.
    if kind == "url":
        srv = Server(None, a.port, log, name, api_key=a.url_key, model=a.url_model,
                     base_url=url)
        rows = []
        for i in range(a.repeat):
            text = prompt_text(tag, i, words)
            rows.append(measure(srv, text, a.max_tokens, f"{tag}{i}", template_kwargs))
            print(f"  {name} run {i}: prompt={rows[-1][0]} cached={rows[-1][1]} "
                  f"client prefill={rows[-1][4]:.1f} decode={rows[-1][5]:.1f} "
                  f"tpc={rows[-1][6]:.2f} ttft={rows[-1][7]:.0f}ms", flush=True)
        return finish_leg(name, a, rows, ctx_for(a, words), f"url={url} model={a.url_model}",
                          clock="client", lifecycle="external")
    # THE BASELINE IS PURGED AND SETTLED. The previous leg's engine leaves its pages
    # behind twice over: wired for seconds after its exit, and, for an engine that
    # mmaps, as file cache for good. A baseline sampled inside either window charges
    # this leg less than it costs (the reference read +5.9 GiB wired on a 15 GiB model
    # until this settle). Both engines wire the model while serving; the wired delta
    # against this baseline is the number that compares them, not the per-process
    # footprint, which never charges a file mapping.
    purged = purge_page_cache()
    vm0 = wired_settle()
    g = gpu_mem_sample()
    gkind, gbase = g if g else (None, None)
    ctx = ctx_for(a, words)
    srv = Server(build_cmd(kind, a, ctx), a.port, log, name,
                 api_key=OMLX_API_KEY if kind == "omlx" else None,
                 model=a.omlx_model if kind == "omlx" else "local",
                 process_group=(kind == "omlx"))  # omlx serve forks omlx-server
    srv.cmd_env = env
    srv.start()
    rows = []
    gpeak = gbase
    try:
        for i in range(a.repeat):
            text = prompt_text(tag, i, words)
            rows.append(measure(srv, text, a.max_tokens, f"{tag}{i}", template_kwargs))
            p, cached, spf, sdc, cpf, cdc, tpc, ttft_ms = rows[-1]
            print(f"  {name} run {i}: prompt={p} cached={cached} "
                  f"server prefill={spf:.1f} decode={sdc:.1f} | "
                  f"client prefill={cpf:.1f} decode={cdc:.1f} tpc={tpc:.2f} "
                  f"ttft={ttft_ms:.0f}ms", flush=True)
            g = gpu_mem_sample()
            if g is not None and gpeak is not None:
                gpeak = max(gpeak, g[1])
        fp = footprint_mib(srv.proc.pid) or 0.0
        rest = 0.0
        grest = None
        if a.rest > 0:
            time.sleep(a.rest)
            rest = footprint_mib(srv.proc.pid) or 0.0
            g = gpu_mem_sample()
            grest = g[1] if g else None
    finally:
        srv.stop()
        if own_kv:
            shutil.rmtree(own_kv, ignore_errors=True)
        if kind == "omlx":
            clear_omlx_cache("after")
        kv_tier_reap(tiers_before)
    # What the engine leaves in the page cache after it is gone: an mmap engine's whole
    # model (Activity Monitor's Cached Files), a pread+F_NOCACHE engine's nothing.
    vm_after = wired_settle()
    cache_left = (vm_after["filebacked"] - vm0["filebacked"]) if (vm0 and vm_after) else None
    if kind == "omlx":
        cfg = f"omlx={omlx_version(a.omlx_bin)} model={a.omlx_model}"
    elif kind == "ref":
        # The reference BINARY is part of the result: two llama.cpp checkouts read 795 and
        # 925 tok/s on the same leg, and a line that only said "ref" let one stand in for
        # the other.
        cfg = f"ctxcp={a.ref_ctxcp} bin={ref_label(a.ref_engine)}"
    else:
        cfg = "pool=on"
    return finish_leg(name, a, rows, ctx, cfg, clock="client" if kind == "omlx" else "server",
                      lifecycle="leg", mem=(fp, rest, gkind, gbase, gpeak, grest, cache_left,
                                            purged))


def omlx_version(binary):
    try:
        return subprocess.run([binary, "--version"], capture_output=True, text=True,
                              timeout=30).stdout.strip().split()[-1]
    except Exception:
        return "?"


def finish_leg(name, a, rows, ctx, cfg, clock, lifecycle, mem=None):
    """Rejections, medians on both clocks and the ALL line for one leg's rows."""
    # A PROMPT SERVED FROM A CACHE returns a plausible tok/s for work that was not done. The
    # engine's cached_tokens is the first signal and on its own it is not enough: oMLX
    # counts the tokens of its OWN prefill that survived a memory-guard eviction retry
    # there (scheduler.py, the _PrefillEvictionNeeded handler), and reported 4096 of 5646
    # on a request whose time-to-first-token proved every token was computed. So the count
    # has to be confirmed by the clock: the first request of a leg is cold by construction
    # (a fresh server, its cache cleared), and a later one that reports cached tokens AND
    # prefills more than 15% faster than it was served from a cache -- a real hit skips
    # the cached share of the work, and the two host speed states are 7% apart.
    first = rows[0][4] if rows else float("nan")
    for i, r in enumerate(rows[1:], 1):
        if r[1] <= 0 or first != first:
            continue
        ratio = r[4] / first if first > 0 else float("nan")
        if ratio > 1.15:
            print(f"REJECTED {name}: run {i} reports {r[1]} of {r[0]} prompt tokens cached "
                  f"and prefilled {ratio:.2f}x faster than the leg's cold first run")
            return None
        print(f"  note: {name} run {i} reports {r[1]} cached tokens but prefilled at "
              f"{ratio:.2f}x the cold run's rate -- the engine's own bookkeeping, kept",
              flush=True)
    # AND A PROMPT THAT DID NOT FIT. A context too small for the regime makes the server
    # truncate, and a truncated prompt returns a perfectly plausible tok/s -- for a
    # shorter prompt than the one this leg claims to be measuring. Same failure shape as
    # the cache hit above, so it gets the same treatment.
    biggest = max((p for p, *_ in rows), default=0)
    if lifecycle == "leg" and biggest + a.max_tokens > ctx:
        print(f"REJECTED {name}: prompt {biggest} + {a.max_tokens} generated exceeds "
              f"the {ctx}-token context; the server truncated it")
        return None
    # EVERY RUN COUNTS. The first request used to be dropped as a warm-up because the
    # weights were wired lazily, so it paid the whole residency wire; the tier is wired at
    # load now and run 0 reads the same as the rest, so discarding it only threw away a
    # third of the samples. --drop-first restores the old behaviour for an engine that does
    # need a warm-up (the reference arm is unchanged by this: it never needed one either).
    keep = rows[1:] if (a.drop_first and len(rows) > 1) else rows
    # The primary pair is the engine's own clock where it has one (imparo, llama-server:
    # the numbers every earlier record was taken on) and the client clock where it does
    # not; the client pair is on every line.
    pf_i, dc_i = (2, 3) if clock == "server" else (4, 5)
    pf = sorted(r[pf_i] for r in keep)
    dw = sorted(r[dc_i] for r in keep)
    cpf = sorted(r[4] for r in keep)
    cdc = sorted(r[5] for r in keep)
    tpc = sorted(r[6] for r in keep)
    ttft = sorted(r[7] for r in keep)
    ptok = rows[-1][0]
    # Prefill tok/s is TILE-SENSITIVE: the prefill GEMM pads n_tok up to 64-token tiles,
    # so a prompt just over a boundary (e.g. 449 -> 512) does up to 14% more work than
    # one just under (447 -> 448). The ptok field makes the length explicit; compare
    # engines at the SAME ptok, and keep short prompts a multiple of 64 for a clean read.
    #
    # EVERY WARM RUN ON THE LINE, in request order, as prefill@decode. The host has two
    # speed states that flip every couple of minutes (2026-09-02: imparo 1011-1017 tok/s
    # fast vs 938-943 slow on the 5963 leg, decode 43 vs 36), and a two-sample median is
    # their mean: one slow request read as "-1.2% vs upstream" on a leg that is +3.8% in
    # either state. The median stays; the runs beside it say whether it mixed two states.
    runs = ",".join(f"{r[pf_i]:.1f}@{r[dc_i]:.1f}" for r in keep)
    clients = ",".join(other_gpu_clients()) or "none"
    line = (f"ALL {name} kv={a.kv} {cfg} ptok={ptok} prefill_median={mid(pf):.1f} "
            f"decode_median={mid(dw):.1f} clock={clock} client_prefill={mid(cpf):.1f} "
            f"client_decode={mid(cdc):.1f} tpc={mid(tpc):.2f} ttft_ms={mid(ttft):.0f} "
            f"decode_n={len(dw)} decode_spread={spread(dw):.1%} runs={runs} "
            f"think={a.thinking} gpu_clients={clients}")
    if mem is not None:
        fp, rest, gkind, gbase, gpeak, grest, cache_left, purged = mem
        line += f" t0={fp:.0f}MiB resting={rest:.0f}MiB"
        # Machine-level GPU memory deltas vs the pre-start baseline (wired on Apple
        # unified memory, device VRAM on CUDA). Ambient -- other work moves it too;
        # trust it across the interleaved pairs, not one leg in isolation.
        if gbase is not None and gpeak is not None:
            line += f" {gkind}_peak=+{gpeak - gbase:.0f}MiB"
            if grest is not None:
                line += f" {gkind}_rest={grest - gbase:+.0f}MiB {gkind}_abs={grest:.0f}MiB"
        if cache_left is not None:
            line += f" cache_left={cache_left:+.0f}MiB purge={'yes' if purged else 'no'}"
    else:
        line += f" lifecycle={lifecycle}"
    print(line, flush=True)
    return line


def vs_line(lines, regime, p):
    """The CROSS-ENGINE ratio, printed rather than left to the reader.

    Every ALL line carries two clocks, and the header says which to use: engines are compared
    on the CLIENT columns, imparo against its own history on the server ones. Leaving that to
    whoever reads the log is how a whole session got quoted from `decode_median` -- the server
    column, because it is the first one on the line. A harness that knows the answer should
    say it.

    Baseline is the FIRST engine named in --engines; every other engine is divided by it.
    Prints nothing for a one-engine run, which is the A/B-by-env case.
    """
    def col(l, k):
        m = re.search(rf"{k}=([0-9.]+)", l)
        return float(m.group(1)) if m else None
    # finish_leg returns None for a leg it REJECTED (a cached prompt, or one truncated by the
    # context) -- those legs print no ALL line and must not be compared against.
    lines = [l for l in lines if l]
    if len(lines) < 2:
        return
    base = lines[0]
    bname = base.split()[1]
    bpf, bdc = col(base, "client_prefill"), col(base, "client_decode")
    if not bpf or not bdc:
        return
    for other in lines[1:]:
        oname = other.split()[1]
        opf, odc = col(other, "client_prefill"), col(other, "client_decode")
        if not opf or not odc:
            continue
        print(f"VS {regime} pair {p} {bname}/{oname} CLIENT prefill={bpf / opf:.3f}x "
              f"decode={bdc / odc:.3f}x", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    # A converted model file (imparo-repack's Q8_0_TM) is imparo's alone: the reference
    # engine reads the ORIGINAL GGUF, so the two engines see the same numbers in different
    # byte orders. Defaults to --model.
    ap.add_argument("--ref-model", default=None)
    ap.add_argument("--ref-engine", default=DEFAULT_REF)
    ap.add_argument("--engines", nargs="*", default=["imparo", "ref"],
                    help="any of imparo, ref, omlx, NAME@URL (an OpenAI-compatible server "
                         "already running, measured on the client clock only)")
    ap.add_argument("--omlx-model", default=None,
                    help="model directory name under --omlx-models-dir, or a path to the model "
                         "directory (required for omlx)")
    ap.add_argument("--omlx-bin", default=OMLX_BIN)
    ap.add_argument("--omlx-models-dir", default=OMLX_MODELS)
    ap.add_argument("--url-model", default="default", help="model name sent to a NAME@URL engine")
    ap.add_argument("--url-key", default=None, help="bearer token for a NAME@URL engine")
    ap.add_argument("--thinking", choices=["on", "off", "unset"], default="off",
                    help="enable_thinking sent to EVERY engine as chat_template_kwargs; "
                         "unset sends nothing and each server applies its template's default")
    ap.add_argument("--kv", default="f16")
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--short-pairs", type=int, default=1)
    ap.add_argument("--long-words", type=int, default=330)
    ap.add_argument("--short-words", type=int, default=24)
    # ~16k tokens: the regime where attention dominates and quantized-KV decode is
    # furthest ahead. It has its own pair count so it can be switched off for a quick
    # run without losing the other two.
    ap.add_argument("--deep-pairs", type=int, default=1)
    ap.add_argument("--deep-words", type=int, default=950)
    # A FLOOR, not the context: each leg derives what its own regime needs (ctx_for).
    ap.add_argument("--ctx", type=int, default=8192)
    # MATCHED ON BOTH SIDES: llama takes it as -ub, imparo as IMPARO_BATCH ("the chunk
    # width this process will use"), set in leg(). Same prefill chunk, both engines.
    #
    # IT ALSO OVERRIDES imparo's OWN CHUNK. The compiled chunk is 512, but the tuned config
    # carries a per-model `batch` line (2048 for LFM2.5-8B-A1B, derived from its routed
    # GEMM), so a bracket run at the default --ubatch can measure a configuration imparo
    # does not ship. That is the right trade for attributing kernel differences and the
    # wrong one for "how fast is the thing we ship": for the latter set IMPARO_BATCH in the
    # environment (setdefault keeps it) or pass --ubatch, and say which you did.
    ap.add_argument("--ubatch", type=int, default=512)
    # llama.cpp's default. 0 disables checkpoints, which is the correctness harness's
    # setting and worth ~17% to the reference -- see ref_speed_args.
    ap.add_argument("--ref-ctxcp", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--drop-first", action="store_true",
                    help="discard run 0 as a warm-up (off: every run counts)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--rest", type=int, default=8)
    ap.add_argument("--port", type=int, default=8440)
    a = ap.parse_args()
    clients = other_gpu_clients()
    if clients:
        print(f"WARNING: Chromium/WebKit GPU process alive ({', '.join(clients)}); a browser tab "
              f"can take the GPU in bursts and every leg below may mix two speed states -- "
              f"close browsers first (an idle Electron app such as GitHub Desktop measured clean)")
    if "ref" in a.engines:
        print(f"ref engine: {a.ref_engine}", flush=True)
    if "omlx" in a.engines:
        if not a.omlx_model:
            ap.error("--engines omlx needs --omlx-model (a directory under --omlx-models-dir, "
                     "or a path to the model directory)")
        a.omlx_models_dir, a.omlx_model = omlx_model_root_and_id(a.omlx_models_dir, a.omlx_model)
        if not os.path.isdir(os.path.join(a.omlx_models_dir, a.omlx_model)):
            ap.error(f"no {a.omlx_model} under {a.omlx_models_dir}")
        print(f"omlx engine: {a.omlx_bin} ({omlx_version(a.omlx_bin)}) model {a.omlx_model}",
              flush=True)
    print(f"thinking={a.thinking} (chat_template_kwargs.enable_thinking on every engine)",
          flush=True)
    for p in range(a.pairs):
        got = []
        for eng in a.engines:
            print(f"== long pair {p+1} {eng} ==", flush=True)
            got.append(leg(eng, a, a.long_words, f"L{p}{eng[:1]}"))
        vs_line(got, "long", p + 1)
    for p in range(a.short_pairs):
        got = []
        for eng in a.engines:
            print(f"== short pair {p+1} {eng} ==", flush=True)
            got.append(leg(eng, a, a.short_words, f"S{p}{eng[:1]}"))
        vs_line(got, "short", p + 1)
    # DEEP LAST: it is the slowest regime and the one whose server needs the widest
    # context, so a run that is cut short still leaves the cheaper regimes measured.
    for p in range(a.deep_pairs):
        got = []
        for eng in a.engines:
            print(f"== deep pair {p+1} {eng} ==", flush=True)
            got.append(leg(eng, a, a.deep_words, f"D{p}{eng[:1]}"))
        vs_line(got, "deep", p + 1)
    return 0

if __name__ == "__main__":
    sys.exit(main())
