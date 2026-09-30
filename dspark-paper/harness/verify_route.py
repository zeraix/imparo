#!/usr/bin/env python3
"""Compare the replies and speeds of two agentic.py runs of the same arm on the same subset.

A route change that is meant to move only time must leave every reply byte-identical (text_md5,
reasoning included). Prints, per reply, whether the text matches and the decode speed of both runs,
then IDENTICAL or DIFFERENT and the geometric mean of the speed ratio.

usage: verify_route.py OLD/results.jsonl OLD_ARM NEW/results.jsonl NEW_ARM
"""
import json, math, sys


def load(path, arm):
    rows = [json.loads(l) for l in open(path)]
    return {(r["set"], r["conv"], r["turn"], r["type"]): r for r in rows if r["arm"] == arm}


def main(old_path, old_arm, new_path, new_arm):
    old, new = load(old_path, old_arm), load(new_path, new_arm)
    units = [u for u in old if u in new]
    if not units:
        print("no common replies")
        return 2
    same, logs = 0, []
    for u in units:
        a, b = old[u], new[u]
        eq = a.get("text_md5") == b.get("text_md5")
        same += eq
        da, db = a.get("decode_tok_s"), b.get("decode_tok_s")
        ok = all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (da, db))
        if ok:
            logs.append(math.log(db / da))
        print(f"{u[1]:<18} t{u[2]} {u[3]:<16} {'same' if eq else 'DIFF'}  "
              f"old {da if da is None else round(da, 1)}  new {db if db is None else round(db, 1)}")
    gm = math.exp(sum(logs) / len(logs)) if logs else float("nan")
    print(f"{'IDENTICAL' if same == len(units) else 'DIFFERENT'} {same}/{len(units)} replies; "
          f"new/old decode speed x{gm:.3f} (geometric mean over {len(logs)})")
    return 0 if same == len(units) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
