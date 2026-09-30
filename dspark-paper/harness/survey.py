#!/usr/bin/env python3
"""Verify cost T(n) by pinned tree width n and context c, on the current build (paper Table 1).

One imparo-forward run per (c, n): DSpark decode with IMPARO_DSPARK_TREE=n pinned and the round
probe on; T is the median verify_us of the rounds that verified exactly n rows, the first two
rounds dropped (cold pipelines). Two passes, the second in reverse order.

usage: survey.py OUTDIR NS(comma) CTXS(comma) DECODE [PASSES]
"""
import os, re, statistics, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# raw-continuation prompts built by ../testset/context.py
PROMPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testset", "context")
MODEL = os.environ.get("SURVEY_MODEL",
    f"{REPO}/models/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf")
DRAFT = os.environ.get("SURVEY_DRAFT",
    f"{REPO}/models/LFM2.5-2.6B-DSpark-GGUF/LFM2.5-2.6B-DSpark-Q8_0.gguf")
ROUND = re.compile(r"dspark round start=(\d+) rows=(\d+) consumed=(\d+) draft_us=(\d+) "
                   r"tree_us=(\d+) verify_us=(\d+)")


def one(out, ctx, n, decode, tag):
    ids = open(f"{PROMPTS}/{os.environ.get('PROMPT_SET', 'lfm25')}_{ctx}A.ids").read().split()
    home = f"{out}/home"
    subprocess.run(["rm", "-rf", home]); os.makedirs(f"{home}/.imparo")
    for f in os.listdir(os.path.expanduser("~/.imparo")):
        if f.startswith("device-") or (f.startswith("tune-") and f.endswith(".txt")):
            subprocess.run(["cp", os.path.expanduser(f"~/.imparo/{f}"), f"{home}/.imparo/"])
    env = dict(os.environ, HOME=home, IMPARO_GPU="1", IMPARO_KV_CAP=str(ctx + decode + 256),
               IMPARO_DSPARK_TREE=str(n), IMPARO_DSPARK_OFFSET="0", IMPARO_DSPARK_ACCEPT="1",
               IMPARO_DSPARK_NGRAM="1", IMPARO_DSPARK_SHAPE="tree", IMPARO_DSPARK_ROUND="1")
    log = f"{out}/{tag}.err"
    t0 = time.time()
    with open(f"{out}/{tag}.out", "w") as fo, open(log, "w") as fe:
        rc = subprocess.run([f"{REPO}/target/release/imparo-forward", "--decode", str(decode),
                             "--dspark", DRAFT, MODEL, *ids], env=env, stdout=fo, stderr=fe).returncode
    rounds = [tuple(int(x) for x in m.groups()) for m in map(ROUND.match, open(log)) if m]
    kept = [r for r in rounds[2:] if r[1] == n]
    ver = [r[5] / 1000 for r in kept]
    med = statistics.median(ver) if ver else float("nan")
    lo = min(ver) if ver else float("nan")
    return rc, len(rounds), len(kept), med, lo, time.time() - t0


def main():
    out, ns, ctxs, decode = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    passes = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    os.makedirs(out, exist_ok=True)
    ns = [int(x) for x in ns.split(",")]; ctxs = [int(x) for x in ctxs.split(",")]
    cells = [(c, n) for c in ctxs for n in ns]
    res = open(f"{out}/survey.tsv", "a")
    print(f"SURVEY START {time.strftime('%H:%M:%S')} cells={len(cells)} passes={passes} "
          f"model={os.path.basename(MODEL)}", flush=True)
    for p in range(1, passes + 1):
        order = cells if p % 2 == 1 else list(reversed(cells))
        for c, n in order:
            rc, total, kept, med, lo, secs = one(out, c, n, decode, f"p{p}_c{c}_n{n}")
            line = (f"CELL pass={p} ctx={c} n={n} rc={rc} rounds={total} kept={kept} "
                    f"verify_ms_med={med:.3f} verify_ms_min={lo:.3f} secs={secs:.1f}")
            print(line, flush=True); res.write(line + "\n"); res.flush()
    print(f"SURVEY DONE {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
