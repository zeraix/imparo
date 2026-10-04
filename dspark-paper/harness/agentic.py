#!/usr/bin/env python3
"""The paper's conversation workload through each engine's own server, on the client clock.

Conversations of several turns, each engine's own replies carried as history, each tool call
answered with the conversation's own recorded result. The conversations come from set files
in the test-set schema (../testset/build.py):

  --set=public                every paper set, in the order of PUBLIC below, on one server
  --set=specbench,toolace     named paper sets (../testset/sets/NAME.json)
  --set=PATH.json[:GROUP]     any set file, optionally one group of it
  --subset=smoke              the first conversation of each set, its first two turns: every
                              source shape (tool round trip, multi-turn, long document) once
  --subset=core               the evaluation subset: the first conversation of every category,
                              and the first half of a single-category source (42 conv., 87 turns)
  --subset=rest               its complement, the held-out conversations (31 conv., 61 turns)
  --only=NAME[,NAME]          the named conversations only

THINKING IS ALWAYS ON, whatever this asks for. Every request below sends
`enable_thinking: False`, and LFM2.5's chat template does not mention `enable_thinking`
at all -- its one thinking switch is `preserve_thinking`, which decides whether a past
turn's reasoning is sent back (the table in kv_gates.py). Measured: 475 chars of
reasoning with the flag absent, true and false alike. So the gate's replies are thinking
replies; do not read "thinking off" anywhere in these numbers.

Arms (one server at a time; an imparo arm gets a private HOME with the tune file copied in and no
learned store, so it starts cold and learns only from its own requests):

AN ARM IS NAMED FOR WHAT IT IS. The old two-letter codes (D0, T16, b, TN) keyed the logs
and every results.jsonl, and a reader had to look them up; agentic_report.py still reads
those files, so its label table carries both spellings.

  plain                      imparo, no drafter paired: its own plain decode
  chain3                     imparo, a chain of at most 3 draft tokens: llama.cpp's DSpark defaults
                             (--spec-draft-n-max 3, --spec-draft-p-min 0) on this engine
  chain                      imparo, the drafter's block verified as a CHAIN -- no tree, no learning
  tree16                     imparo, a FIXED 16-row tree, offset off, model off: tree verify, no learning
  budget                     imparo, the width CHOSEN BY THE COST MODEL each round (offset on, model off)
  budget-accept              budget plus the fitted acceptance model
  budget-accept-ngram        the above plus n-gram chains from the request's index
  budget-accept-ngram-table  the above plus the stored table (level 2) as each lookup's fallback
  tree-online                tree + both online models, n-gram off
  ngram-chain                online models + n-gram with ONE child per node (IMPARO_DSPARK_SHAPE=chain)
  complete                   THE COMPLETE FORM: tree + cost model + acceptance model + n-gram fill,
                             with the copy walk's resume and the conversation's earlier text pinned OFF
  complete-wN                the complete form with the width pinned to N rows (the width table):
                             every model and the n-gram fill as in complete, only the width fixed
  complete-table             complete plus the stored table
  complete-classes           the complete form with the 8-bit matrix kernel's row classes in place of
                             learned widths (IMPARO_DSPARK_CLASSES=declared: Table 9, read only by
                             the measurement build that still carries the declared classes)
  llama-dspark               llama-server upstream, --spec-type draft-dspark, the same drafter file,
                             upstream's default --spec-draft-n-max (3)
  llama-dspark9              as llama-dspark with --spec-draft-n-max 9, the drafter's trained block
  llama-ngram                llama-server upstream, --spec-type ngram-map-k4v,draft-dspark, k4v at
                             upstream's documented example for repeated text (8, 8, 2 hits)
  llama-plain                llama-server upstream, no speculation

`complete` is what the server runs by default with a drafter paired. (The copy walk's resume
and the earlier text were dropped as levers on 2026-09-18.)

usage: python3 -u dspark-paper/harness/agentic.py OUTDIR ARMS(comma) REPEATS MAX_TOKENS
                                                  [--set=...] [--subset=smoke] [--only=...]
env:   AGENTIC_TARGET (the target GGUF), AGENTIC_DRAFT (the DSpark drafter; both engines load
       this file), IMPARO_REF_ENGINE (llama-server), IMPARO_BIN_DIR (imparo-server), AGENTIC_CTX,
       AGENTIC_ENABLE_THINKING=1 (Qwen3), AGENTIC_KEEP_THINKING=1 (send each reply's reasoning
       back with `preserve_thinking`)
       Report: python3 dspark-paper/harness/agentic_report.py OUTDIR/results.jsonl
       Checks: python3 dspark-paper/harness/check.py OUTDIR
"""
import hashlib, json, os, re, shutil, subprocess, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "dev_harness"))
from engine import Server  # noqa: E402
from bracket import client_rates  # noqa: E402
from harness import DEFAULT_REF, bin_path  # noqa: E402

MODELS = f"{REPO}/models/LFM2.5-2.6B-GGUF"
TGT = os.environ.get("AGENTIC_TARGET", f"{MODELS}/LFM2.5-2.6B-Q8_0.gguf")
DRAFT = os.environ.get("AGENTIC_DRAFT",
                       f"{REPO}/models/LFM2.5-2.6B-DSpark-GGUF/LFM2.5-2.6B-DSpark-Q8_0.gguf")
LLAMA = DEFAULT_REF
# a reply that runs to its end needs room: prompt + max_tokens must fit (imparo refuses a request
# that does not, with 400, as the OpenAI API does; llama.cpp clamps)
CTX = int(os.environ.get("AGENTIC_CTX", "16384"))
# AGENTIC_KEEP_THINKING=1: the client sends each reply's reasoning back (`reasoning_content`) and
# asks the template to keep it (`preserve_thinking`), so the next prompt carries the reply as it
# was generated. Off, the client keeps only `content` -- what most OpenAI-style clients do -- and
# LFM2.5's template drops a past reply's thinking at the next user turn.
KEEP_THINKING = os.environ.get("AGENTIC_KEEP_THINKING") == "1"
# WHAT THE FLAG DOES IS THE MODEL'S ANSWER, NOT OURS. LFM2.5's template never mentions
# `enable_thinking`, so its replies are thinking replies whatever this says -- which is why
# the default is False and every LFM2.5 number here is a thinking-on number. Qwen3's
# template DOES honour it: sent False, its generation prompt writes a closed empty
# `<think>` block and the model answers without reasoning. Running the two models with one
# value would compare a reasoning workload against a non-reasoning one, so this is set per
# model: AGENTIC_ENABLE_THINKING=1 for Qwen3.
ENABLE_THINKING = os.environ.get("AGENTIC_ENABLE_THINKING") == "1"
PORT = 8475

SETS_DIR = os.path.join(os.path.dirname(HERE), "testset", "sets")
# The paper's workload, in the order a run serves it. Every file is built by
# ../testset/build.py from public sources; none of its turns is ours.
PUBLIC = ("specbench", "speedbench", "wildchat", "convfinqa", "toolace", "longbench")


def load_set(spec):
    """One `--set=` entry -> (label, file sha256, conversations). A conversation is a dict:
    name, kind, turns [{kind, type, text, result?}], tools (optional)."""
    path, group = spec, None
    if not spec.endswith(".json") and ":" in spec and spec.rsplit(":", 1)[0].endswith(".json"):
        path, group = spec.rsplit(":", 1)
    if not path.endswith(".json"):
        path = os.path.join(SETS_DIR, f"{spec}.json")
    raw = open(path, "rb").read()
    doc = json.loads(raw)
    convs = doc["conversations"]
    if group:
        want = set(doc["groups"][group])
        convs = [c for c in convs if c["name"] in want]
    label = doc["set"] + (f":{group}" if group else "")
    return label, hashlib.sha256(raw).hexdigest()[:16], doc.get("source_max_tokens"), convs


def arm_server(arm, outdir, tag):
    log = f"{outdir}/server_{tag}.log"
    env = dict(os.environ)
    # complete-wN: the complete form with its width pinned to N rows -- the same models, the same
    # n-gram fill, the same store learning from empty; only the width is not chosen.
    pinned = re.fullmatch(r"complete-w(\d+)", arm)
    classes = arm == "complete-classes"
    if pinned or classes:
        arm = "complete"
    if arm in (
        "tree16", "budget", "budget-accept", "budget-accept-ngram",
        "budget-accept-ngram-table", "chain", "chain3", "tree-online", "ngram-chain", "complete",
        "complete-table", "plain",
    ):
        home = f"{outdir}/home_{tag}"
        shutil.rmtree(home, ignore_errors=True)
        os.makedirs(f"{home}/.imparo")
        for f in os.listdir(os.path.expanduser("~/.imparo")):
            if f.startswith("device-"):
                shutil.copy2(os.path.expanduser(f"~/.imparo/{f}"), f"{home}/.imparo/")
            # The tuned configs only: learned stores (verify cost, dictionary) stay behind, so
            # every arm starts cold and learns only from its own requests.
            if f.startswith("tune-") and f.endswith(".txt"):
                shutil.copy2(os.path.expanduser(f"~/.imparo/{f}"), f"{home}/.imparo/")
        env.update({"HOME": home, "IMPARO_GPU": "1",
                    "IMPARO_DSPARK_ACCEPT": "1"
                    if arm in ("budget-accept", "budget-accept-ngram",
                               "budget-accept-ngram-table", "tree-online", "ngram-chain",
                               "complete", "complete-table")
                    else "0",
                    "IMPARO_DSPARK_NGRAM": "1"
                    if arm in ("budget-accept-ngram", "budget-accept-ngram-table",
                               "ngram-chain", "complete", "complete-table")
                    else "0",
                    "IMPARO_DSPARK_DICT": "1"
                    if arm in ("budget-accept-ngram-table", "complete-table")
                    else "0",
                    "IMPARO_DSPARK_SHAPE": "chain" if arm == "ngram-chain" else "tree"})
        env.pop("IMPARO_DSPARK_TOPK", None); env.pop("IMPARO_DSPARK_CHAIN_MAX", None)
        if arm == "tree16":
            env.update({"IMPARO_DSPARK_TREE": "16", "IMPARO_DSPARK_OFFSET": "0"})
        elif arm == "chain":
            env.update({"IMPARO_DSPARK_TREE": "chain", "IMPARO_DSPARK_OFFSET": "0"})
        elif arm == "chain3":
            # llama.cpp's DSpark configuration on this engine: a chain of at most 3 draft
            # tokens (--spec-draft-n-max 3), no confidence cut (--spec-draft-p-min 0)
            env.update({"IMPARO_DSPARK_TREE": "chain", "IMPARO_DSPARK_OFFSET": "0",
                        "IMPARO_DSPARK_CHAIN_MAX": "3"})
        else:
            env.pop("IMPARO_DSPARK_TREE", None); env.pop("IMPARO_DSPARK_OFFSET", None)
        if pinned:
            env["IMPARO_DSPARK_TREE"] = pinned.group(1)
        env.pop("IMPARO_DSPARK_CLASSES", None)
        if classes:
            env["IMPARO_DSPARK_CLASSES"] = "declared"
        cmd = [bin_path("imparo-server"), "-m", TGT, "-c", str(CTX), "--port", str(PORT)]
        if arm != "plain":
            cmd += ["--draft", DRAFT]
    else:
        cmd = [LLAMA, "-m", TGT, "-c", str(CTX), "-ngl", "999", "--port", str(PORT), "--jinja", "--parallel", "1",
               "--no-mmproj", "-fa", "on", "-ub", "512", "-b", "2048"]
        if arm == "llama-dspark":
            cmd += ["--spec-type", "draft-dspark", "-md", DRAFT]
        elif arm == "llama-dspark9":
            cmd += ["--spec-type", "draft-dspark", "-md", DRAFT, "--spec-draft-n-max", "9"]
        elif arm == "llama-ngram":
            # upstream's documented k4v example for repeated text (docs/speculative.md), DSpark at the
            # default draft length, which measured faster than 9 on this workload
            cmd += ["--spec-type", "ngram-map-k4v,draft-dspark", "-md", DRAFT,
                    "--spec-ngram-map-k4v-size-n", "8", "--spec-ngram-map-k4v-size-m", "8",
                    "--spec-ngram-map-k4v-min-hits", "2"]
    srv = Server(["caffeinate", "-dims"] + cmd, PORT, log, arm)
    srv.cmd_env = env
    return srv, log


def run_arm(arm, repeat, outdir, max_tokens, sets, results):
    tag = f"{arm}_r{repeat}"
    srv, log = arm_server(arm, outdir, tag)
    t_start = time.time()
    srv.start()
    print(f"  {tag}: ready in {time.time() - t_start:.1f}s", flush=True)
    try:
        def round_(sname, cname, messages, k, kind, ttype, tools):
            """One timed request. A tool turn runs this twice; everything else once."""
            before = os.path.getsize(log)
            kwargs = {"enable_thinking": ENABLE_THINKING}
            if KEEP_THINKING:
                kwargs["preserve_thinking"] = True
            r = srv.chat(messages, f"{tag}-{cname}", max_tokens=max_tokens, tools=tools,
                         template_kwargs=kwargs)
            time.sleep(0.3)  # let the server flush its per-request lines
            with open(log, errors="replace") as f:
                f.seek(before); lines = f.read()
            prefill, decode, tpc, ttft_ms = client_rates(r)
            u = r.get("usage") or {}; tm = r.get("timings") or {}
            row = {"arm": arm, "repeat": repeat, "set": sname, "conv": cname, "turn": k, "kind": kind, "type": ttype,
                   "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   "completion_tokens": u.get("completion_tokens"), "ttft_ms": ttft_ms, "decode_tok_s": decode,
                   "prefill_tok_s": prefill, "server_decode_tok_s": tm.get("predicted_per_second"),
                   "imparo_drafted": len(re.findall(r"\[imparo\] draft prompt_history=", lines)),
                   # imparo reports draft_verified / draft_accepted; llama.cpp draft_n / draft_n_accepted
                   "draft_n": tm.get("draft_verified", tm.get("draft_n")),
                   "draft_accepted": tm.get("draft_accepted", tm.get("draft_n_accepted")),
                   # imparo only: verify rounds, one-token steps, and the two counts by source
                   "draft_rounds": tm.get("draft_rounds"), "plain_rounds": tm.get("plain_rounds"),
                   "draft_sources": tm.get("draft_sources"),
                   "reasoning_chars": len(r.get("reasoning", "")), "text_chars": len(r.get("text", "")),
                   "tool_calls": len(r.get("tool_calls") or []),
                   "calls": r.get("tool_calls") or [],
                   "finished": (u.get("completion_tokens") or 0) < max_tokens,
                   "text_md5": hashlib.md5((r.get("reasoning", "") + "\n---\n" + r.get("text", "")).encode()).hexdigest()}
            results.append(row)
            print(f"    {cname} t{k} {ttype:<13} {kind:<7} pt={row['prompt_tokens']} cached={row['cached_tokens']} "
                  f"ct={row['completion_tokens']} ttft={ttft_ms:.0f}ms dec={decode:.1f} tok/s drafted={row['imparo_drafted']} "
                  f"draft={row['draft_accepted']}/{row['draft_n']} think={row['reasoning_chars']}ch answer={row['text_chars']}ch "
                  f"calls={row['tool_calls']} finished={row['finished']}", flush=True)
            for c in row["calls"]:
                print(f"      call {c.get('name')!r} arguments={c.get('arguments')!r}", flush=True)
            return r

        for sname, convs in sets:
            for conv in convs:
                cname, tools, messages = conv["name"], conv.get("tools"), []
                for k, turn in enumerate(conv["turns"], 1):
                    kind, ttype, result = turn["kind"], turn["type"], turn.get("result")
                    messages.append({"role": "user", "content": turn["text"]})
                    try:
                        r = round_(sname, cname, messages, k, kind, ttype + ("/call" if result else ""), tools)
                    except urllib.error.HTTPError as e:
                        # A SERVER THAT REFUSES THE REQUEST (llama.cpp's Qwen3 template path rejects a
                        # tool schema whose parameter type JSON Schema does not define, e.g. `float`).
                        # The reply does not exist, so the rest of this conversation cannot run on
                        # this arm: record the refusal and move to the next conversation. check.py and
                        # the tables leave the conversation out of every comparison with this arm.
                        body = e.read().decode(errors="replace")[:300]
                        results.append({"arm": arm, "repeat": repeat, "set": sname, "conv": cname, "turn": k,
                                        "kind": kind, "type": ttype + ("/call" if result else ""),
                                        "refused": f"HTTP {e.code}: {body}", "completion_tokens": 0,
                                        "decode_tok_s": None, "finished": True, "text_chars": 0,
                                        "reasoning_chars": 0, "tool_calls": 0, "calls": [],
                                        "text_md5": None})
                        print(f"    {cname} t{k} REFUSED by the server (HTTP {e.code}): {body}; "
                              f"the rest of this conversation is skipped on this arm", flush=True)
                        break
                    calls = r.get("tool_calls") or []
                    if calls and not result:
                        print(f"    {cname} t{k} UNEXPECTED TOOL CALL on a turn with no recorded result; "
                              f"the history keeps only its text", flush=True)
                    if result and calls:
                        # THE TOOL ROUND TRIP, as a client runs it: the assistant's call goes into the
                        # history, the result comes back as a `tool` message, and the model answers in a
                        # SECOND request. Both are timed; the second one is the round whose prompt now
                        # carries the tool's structured text.
                        ids = [f"call_{c['name']}_{i}" for i, c in enumerate(calls)]
                        call_msg = {"role": "assistant", "content": r.get("text") or None,
                                    "tool_calls": [{"id": i, "type": "function",
                                                    "function": {"name": c["name"], "arguments": c["arguments"]}}
                                                   for i, c in zip(ids, calls)]}
                        if KEEP_THINKING:
                            call_msg["reasoning_content"] = r.get("reasoning", "")
                        messages.append(call_msg)
                        for i in ids:
                            messages.append({"role": "tool", "tool_call_id": i, "content": result})
                        r = round_(sname, cname, messages, k, kind, ttype + "/result", tools)
                    elif result:
                        print(f"    {cname} t{k} NO TOOL CALL -- the model answered instead; "
                              f"the round trip did not run", flush=True)
                    reply = {"role": "assistant", "content": r.get("text", "")}
                    if KEEP_THINKING:
                        reply["reasoning_content"] = r.get("reasoning", "")
                    messages.append(reply)
    finally:
        srv.stop()
        # The arm's KV disk tier is 2-3 GB per full-reply pass and no analysis reads it; the learned
        # stores beside it (.imparo/*.txt) stay.
        kv = f"{outdir}/home_{tag}/.imparo/kv"
        if os.path.isdir(kv):
            freed = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(kv) for f in fs)
            shutil.rmtree(kv)
            print(f"  {tag}: removed its KV store ({freed / 2**30:.1f} GiB)", flush=True)


def opt(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def main():
    outdir, arms, repeats, max_tokens = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4])
    if len({a.lower() for a in arms}) != len(arms):
        sys.exit(f"arm names must differ in more than case (macOS file names ignore case): {arms}")
    chosen = opt("set", "public")
    specs = list(PUBLIC) if chosen == "public" else chosen.split(",")
    subset, only = opt("subset"), opt("only")
    if subset not in (None, "smoke", "core", "rest"):
        sys.exit(f"--subset={subset}: smoke, core or rest")
    sets, stamps = [], []
    for spec in specs:
        label, sha, cap, convs = load_set(spec)
        if subset == "smoke":
            convs = [{**convs[0], "turns": convs[0]["turns"][:2]}]
        elif subset in ("core", "rest"):
            # THE EVALUATION SUBSET, fixed by rule before any run: the first conversation of
            # every category of a multi-category source, and the first half of the
            # conversations of a single-category source. Every category stays; 87 of 148 turns.
            # `rest` is its complement: the held-out conversations no arm had run.
            kinds = list(dict.fromkeys(c["kind"] for c in convs))
            core = (convs[:len(convs) // 2] if len(kinds) == 1 else
                    [next(c for c in convs if c["kind"] == k) for k in kinds])
            names = {c["name"] for c in core}
            convs = core if subset == "core" else [c for c in convs if c["name"] not in names]
        if only:
            convs = [c for c in convs if c["name"] in only.split(",")]
        if not convs:
            continue
        if cap and cap != max_tokens:
            print(f"NOTE {label}: its source caps a generation at {cap} new tokens; this run uses "
                  f"{max_tokens}", flush=True)
        sets.append((label, convs))
        stamps.append(f"{label}={sha}")
    if not sets:
        sys.exit(f"--set={chosen} --only={only} selected no conversation")
    os.makedirs(outdir, exist_ok=True)
    rounds = sum(len(c["turns"]) for _, convs in sets for c in convs)
    results = []
    stamp = lambda: time.strftime("%H:%M:%S")
    print(f"AGENTIC START {stamp()} arms={arms} repeats={repeats} max_tokens={max_tokens} subset={subset} "
          f"turns={rounds} ctx={CTX} sets={','.join(stamps)} "
          f"imparo_md5={hashlib.md5(open(bin_path('imparo-server'),'rb').read()).hexdigest()}", flush=True)
    for repeat in range(1, repeats + 1):
        order = arms if repeat % 2 == 1 else list(reversed(arms))
        for arm in order:
            run_arm(arm, repeat, outdir, max_tokens, sets, results)
            with open(f"{outdir}/results.jsonl", "w") as f:
                for row in results:
                    f.write(json.dumps(row) + "\n")
    print(f"AGENTIC ALL DONE {stamp()} rows={len(results)}", flush=True)


if __name__ == "__main__":
    main()
