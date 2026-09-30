#!/usr/bin/env python3
"""The paper's end-to-end numbers from one or more agentic.py results.jsonl files.

A unit is one timed round: (set, conversation, turn, round type). A tool turn is two units,
the call and the answer after its result. Repeats of an arm are averaged per unit first.

  decode      client-clock decode tok/s of the round
  ratio X/R   geometric mean over units of decode_X / decode_R, with a 95% interval from the
              t distribution of the per-unit log ratios; per source set and over all units
  throughput  total generated tokens / total decode seconds, per arm (a token-weighted view)

usage: tables.py RESULTS.jsonl [...] [--base ARM] [--json OUT.json]
"""
import json, math, sys
from collections import defaultdict

T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
        10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 25: 2.06, 30: 2.04, 40: 2.02, 60: 2.00}
SETS = ["specbench", "speedbench", "wildchat", "convfinqa", "toolace", "longbench"]
ORDER = ["llama-plain", "llama-dspark", "plain", "chain3", "chain", "tree16", "budget", "complete"]


def t975(df):
    for k in sorted(T975, reverse=True):
        if df >= k:
            return T975[k] if df < 60 else 1.96
    return T975[1]


def geo(logs):
    n = len(logs)
    if not n:
        return None
    m = sum(logs) / n
    if n < 2:
        return {"x": math.exp(m), "lo": None, "hi": None, "n": n}
    sd = math.sqrt(sum((v - m) ** 2 for v in logs) / (n - 1))
    h = t975(n - 1) * sd / math.sqrt(n)
    return {"x": math.exp(m), "lo": math.exp(m - h), "hi": math.exp(m + h), "n": n}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else None
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    paths = [a for a in args if a.endswith(".jsonl")]
    rows = [json.loads(l) for p in paths for l in open(p)]
    cell = defaultdict(list)
    for r in rows:
        cell[(r["arm"], (r["set"], r["conv"], r["turn"], r["type"]))].append(r)
    arms = sorted({a for a, _ in cell}, key=lambda a: ORDER.index(a) if a in ORDER else 99)
    units = sorted({u for _, u in cell})
    base = base or next((a for a in ("llama-dspark", "plain") if a in arms), arms[0])

    def dec(arm, u):
        xs = [r["decode_tok_s"] for r in cell.get((arm, u), []) if r.get("decode_tok_s")]
        return sum(xs) / len(xs) if xs else None

    report = {"base": base, "arms": {}, "units": len(units)}
    for a in arms:
        rs = [r for (arm, _), v in cell.items() if arm == a for r in v]
        toks = sum(r["completion_tokens"] or 0 for r in rs)
        secs = sum((r["completion_tokens"] - 1) / r["decode_tok_s"] for r in rs
                   if r.get("decode_tok_s") and (r["completion_tokens"] or 0) > 1)
        entry = {"rounds": len(rs), "tokens": toks, "throughput": toks / secs if secs else None,
                 "mean_decode": sum(d for d in (dec(a, u) for u in units) if d) /
                 max(1, sum(1 for u in units if dec(a, u)))}
        for ref in dict.fromkeys([base, "plain", "llama-dspark"]):
            if ref not in arms or ref == a:
                continue
            per = {}
            for s in SETS + ["all"]:
                us = [u for u in units if s == "all" or u[0] == s]
                logs = [math.log(dec(a, u) / dec(ref, u)) for u in us if dec(a, u) and dec(ref, u)]
                per[s] = geo(logs)
            entry[f"vs_{ref}"] = per
        report["arms"][a] = entry

    print(f"files {len(paths)}  rows {len(rows)}  units {len(units)}  arms {arms}  base {base}")
    print(f"{'arm':<14}{'rounds':>7}{'tokens':>9}{'tok/s(tw)':>11}{'mean dec':>10}")
    for a in arms:
        e = report["arms"][a]
        print(f"{a:<14}{e['rounds']:>7}{e['tokens']:>9}{e['throughput'] or 0:>11.2f}"
              f"{e['mean_decode']:>10.2f}")
    for ref in dict.fromkeys([base, "plain"]):
        if ref not in arms:
            continue
        print(f"\nratio over {ref} (geometric mean of per-round decode ratios, 95% CI)")
        print(f"{'arm':<14}" + "".join(f"{s:>20}" for s in SETS + ["all"]))
        for a in arms:
            per = report["arms"][a].get(f"vs_{ref}")
            if not per:
                continue
            cells = []
            for s in SETS + ["all"]:
                g = per[s]
                cells.append(f"{g['x']:.3f} [{g['lo']:.3f},{g['hi']:.3f}]" if g and g["lo"] else
                             (f"{g['x']:.3f}" if g else "-"))
            print(f"{a:<14}" + "".join(f"{c:>20}" for c in cells))
    if out_json:
        with open(out_json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
