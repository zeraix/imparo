#!/usr/bin/env python3
"""Per-token decode cost by width policy and context (the microbenchmark behind the paper's Figure 5).

Arms: chain (the drafter's block verified as a chain), fixed widths, and the budget (width
chosen each round from the online cost model). Every other setting is the same in every arm --
the fitted acceptance model and its per-request terms on, n-gram OFF -- so the arms differ only in
how the width is chosen. Unmeasured budget runs learn the store first; every measured run starts
from a fresh copy of that warmed HOME, so no arm learns from another and all start from the same
state. With T4_WARM=model (the paper's width table) one store serves every (context, prompt) cell,
so a fixed width is never measured on a store fitted to the very text it then decodes. ms/token =
decode_ms / tokens from imparo-forward's `dspark decode` line.

env:   T4_WARM=model learns one store over every prompt before measuring (default: one per prompt);
       SURVEY_MODEL / SURVEY_DRAFT (default LFM2.5-2.6B and its drafter), PROMPT_SET (lfm25 for
       the LFM2.5 vocabulary, qwen3 for Qwen3's; ../testset/context/{set}_{ctx}{A,B,C}.ids)
usage: table4.py OUTDIR ARMS(comma: chain,10,12,16,budget) CTXS(comma) PROMPTS(comma: A,B,C)
                 DECODE [PASSES]
"""
import os, re, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# raw-continuation prompts built by ../testset/context.py
PROMPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testset", "context")
MODEL = os.environ.get("SURVEY_MODEL",
    f"{REPO}/models/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf")
DRAFT = os.environ.get("SURVEY_DRAFT",
    f"{REPO}/models/LFM2.5-2.6B-DSpark-GGUF/LFM2.5-2.6B-DSpark-Q8_0.gguf")
LINE = re.compile(r"dspark decode .*? tokens=(\d+) .*?draft_calls=(\d+) verified_blocks=(\d+) "
                  r"sequential_steps=(\d+) prefill_ms=([\d.]+) decode_ms=([\d.]+)")


def fresh_home(home):
    subprocess.run(["rm", "-rf", home]); os.makedirs(f"{home}/.imparo")
    for f in os.listdir(os.path.expanduser("~/.imparo")):
        if f.startswith("device-") or (f.startswith("tune-") and f.endswith(".txt")):
            subprocess.run(["cp", os.path.expanduser(f"~/.imparo/{f}"), f"{home}/.imparo/"])


def one(out, ctx, prompt, arm, decode, tag, home):
    ids = open(f"{PROMPTS}/{os.environ.get('PROMPT_SET', 'lfm25')}_{ctx}{prompt}.ids").read().split()
    env = dict(os.environ, HOME=home, IMPARO_GPU="1", IMPARO_KV_CAP=str(ctx + decode + 256),
               IMPARO_DSPARK_ACCEPT="1", IMPARO_DSPARK_NGRAM="0", IMPARO_DSPARK_SHAPE="tree",
               IMPARO_DSPARK_ROUND="1")
    env.pop("IMPARO_DSPARK_TREE", None); env.pop("IMPARO_DSPARK_OFFSET", None)
    if arm != "budget":
        env.update(IMPARO_DSPARK_TREE=arm)
    t0 = time.time()
    with open(f"{out}/{tag}.out", "w") as fo, open(f"{out}/{tag}.err", "w") as fe:
        rc = subprocess.run([f"{REPO}/target/release/imparo-forward", "--decode", str(decode),
                             "--dspark", DRAFT, MODEL, *ids], env=env, stdout=fo, stderr=fe).returncode
    m = next((LINE.search(l) for l in open(f"{out}/{tag}.out") if LINE.search(l)), None)
    if not m:
        return rc, None, None, time.time() - t0
    tokens, calls, blocks = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return rc, float(m.group(6)) / tokens, tokens / max(blocks, 1), time.time() - t0


def main():
    out, arms, ctxs, prompts = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4]
    decode = int(sys.argv[5])
    passes = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    ctxs = [int(c) for c in ctxs.split(",")]; prompts = prompts.split(",")
    os.makedirs(out, exist_ok=True)
    res = open(f"{out}/table4.tsv", "a")
    print(f"T4 START {time.strftime('%H:%M:%S')} arms={arms} ctxs={ctxs} prompts={prompts} "
          f"passes={passes}", flush=True)
    warm = {}
    # T4_WARM=model: ONE store learned over every (context, prompt) in turn, as a deployment learns
    # it once per model, and every measured run starts from a copy of it. Unset: one store per
    # (context, prompt), learned from that prompt alone.
    per_model = os.environ.get("T4_WARM") == "model"
    if per_model:
        shared = f"{out}/warm_model"; fresh_home(shared)
    for c in ctxs:
        for pr in prompts:
            if per_model:
                warm[(c, pr)] = shared
            else:
                warm[(c, pr)] = f"{out}/warm_c{c}{pr}"; fresh_home(warm[(c, pr)])
            rc, ms, tpr, secs = one(out, c, pr, "budget", decode, f"warm_c{c}{pr}", warm[(c, pr)])
            print(f"WARM ctx={c} prompt={pr} rc={rc} ms_per_token={ms} secs={secs:.1f}", flush=True)
    run_home = f"{out}/run_home"
    for p in range(1, passes + 1):
        for c in ctxs:
            for pr in prompts:
                order = arms if p % 2 == 1 else list(reversed(arms))
                for arm in order:
                    subprocess.run(["rm", "-rf", run_home])
                    subprocess.run(["cp", "-R", warm[(c, pr)], run_home])
                    rc, ms, tpr, secs = one(out, c, pr, arm, decode, f"p{p}_c{c}{pr}_{arm}", run_home)
                    line = (f"ROW pass={p} ctx={c} prompt={pr} arm={arm} rc={rc} ms_per_token="
                            f"{'nan' if ms is None else f'{ms:.4f}'} tokens_per_round="
                            f"{'nan' if tpr is None else f'{tpr:.3f}'} secs={secs:.1f}")
                    print(line, flush=True); res.write(line + "\n"); res.flush()
    print(f"T4 DONE {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
