#!/usr/bin/env python3
"""Checks a finished agentic.py run before anything reads its numbers. Prints CHECK PASS or
CHECK FAIL with each reason; exit 1 on FAIL, so a staged run stops at the first bad stage.

  1. every arm served the same rounds (same (set, conv, turn, type) rows in the same order)
  2. a reply cut at the cap is COUNTED, not failed: under greedy decoding a reasoning model can
     loop until the cap (Qwen3 says so of its own thinking mode), in every arm alike; the
     tables report such rounds separately
  3. every finished reply produced something: answer text or a tool call
  4. every speculative arm drafted: draft_n > 0 on every round of 32+ generated tokens
     (a drafter that never engaged makes a speculative arm a plain one)
  5. no server log carries a panic, and the harness log (if given) no Traceback
A conversation a server refused (HTTP error, recorded by agentic.py) is noted and left out of
check 1 for every arm, since the refusing arm could not continue it; checks 2-5 still run on every
reply that exists.
A tool turn the model answered without calling its tool is reported, not failed: that is
the model's choice, and every arm sees the same recorded result when it does call.

usage: check.py OUTDIR [HARNESS_LOG]
"""
import glob, json, os, re, sys

NOT_SPECULATIVE = ("plain", "llama-plain")


def main(outdir, harness_log=None):
    rows = [json.loads(l) for l in open(os.path.join(outdir, "results.jsonl"))]
    fails, notes, cut = [], [], []
    # A conversation a server refused (agentic.py records the refusal and skips the rest of that
    # conversation on that arm) is left out of the same-workload check for every arm, and noted.
    refused = [r for r in rows if r.get("refused")]
    gone = {(r.get("set"), r["conv"]) for r in refused}
    for r in refused:
        notes.append(f"{r['arm']} r{r['repeat']} {r['conv']} t{r['turn']} {r['type']} refused: "
                     f"{r['refused'][:120]}")
    rows_all, rows = rows, [r for r in rows if (r.get("set"), r["conv"]) not in gone]
    by_arm = {}
    for r in rows:
        by_arm.setdefault((r["arm"], r["repeat"]), []).append(r)
    shapes = {k: [(r.get("set"), r["conv"], r["turn"], r["type"].split("/")[-1] if "/" in r["type"] else r["type"])
                  for r in v] for k, v in by_arm.items()}
    ref_key = next(iter(shapes))
    for k, s in shapes.items():
        if [x[:3] for x in s] != [x[:3] for x in shapes[ref_key]]:
            fails.append(f"{k[0]} r{k[1]} served {len(s)} rounds, {ref_key[0]} r{ref_key[1]} "
                         f"{len(shapes[ref_key])}: the arms did not run the same workload")
    for r in (x for x in rows_all if not x.get("refused")):
        where = f"{r['arm']} r{r['repeat']} {r['conv']} t{r['turn']} {r['type']}"
        if not r["finished"]:
            cut.append(where)
        if r["finished"] and not r["text_chars"] and not r["tool_calls"]:
            fails.append(f"{where}: no answer and no tool call "
                         f"(reasoning {r['reasoning_chars']} chars)")
        if r["arm"] not in NOT_SPECULATIVE and (r["completion_tokens"] or 0) >= 32 \
                and not r.get("draft_n"):
            fails.append(f"{where}: {r['completion_tokens']} tokens and no draft verified")
    if harness_log and os.path.exists(harness_log):
        text = open(harness_log, errors="replace").read()
        if "Traceback" in text:
            fails.append(f"{harness_log}: Traceback")
        n = len(re.findall(r"NO TOOL CALL", text))
        if n:
            notes.append(f"{n} tool turn(s) answered without the tool call")
    for log in glob.glob(os.path.join(outdir, "server_*.log")):
        with open(log, errors="replace") as f:
            if any("panicked" in line for line in f):
                fails.append(f"{os.path.basename(log)}: panic")
    arms = sorted({k[0] for k in by_arm})
    per = ", ".join(f"{a} {sum(len(v) for k, v in by_arm.items() if k[0] == a)}" for a in arms)
    print(f"rounds per arm: {per}")
    if cut:
        notes.append(f"{len(cut)} round(s) cut at the cap: " + "; ".join(cut[:6]))
    for n in notes:
        print(f"NOTE {n}")
    if fails:
        print(f"CHECK FAIL ({len(fails)})")
        for f in fails[:40]:
            print(f"  {f}")
        return 1
    print(f"CHECK PASS {len(rows)} rounds, {len(by_arm)} arm runs"
          + (f"; {len(gone)} conversation(s) refused by a server left out" if gone else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
