#!/usr/bin/env python3
"""Read an agentic run's results.jsonl and report, per comparison and per turn type:

  unit       one (conversation, turn): the same messages in every arm (the history carries each engine's
             own replies, so later prompts differ between engines only by the replies' content)
  speed      client-clock decode tok/s, averaged over repeats first; a comparison X vs R is the geometric
             mean over turns of dec_X / dec_R, with a 95% interval from the t distribution of the log ratios
  latency    TTFT (ms) and turn time = TTFT + decode window, same pairing
  identity   imparo arms must emit the same text per turn (verify-exact); llama arms are not compared
  learning   for each imparo arm, dec / dec_T16 in the order the turns ran: does the ratio grow?

Fixed before the run (2026-09-17 14:36). gate2's arms and comparisons added before gate2's results were
read (2026-09-17 15:26). TC added to the arm order and identity list, and a paired TTFT ratio per comparison,
before gate_copy's results were read (2026-09-17 23:08). The paper ladder's two missing steps -- b/D0 and
TN/b -- were added 2026-09-21 01:55, before that run started. usage: python3 dspark-paper/harness/agentic_report.py RESULTS.jsonl"""
import json, math, sys
from collections import defaultdict

T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13, 16: 2.12, 17: 2.11, 18: 2.10, 19: 2.09, 20: 2.09}
GROUPS = {"first": ["first"], "same kind, continued": ["same"], "kind switch, continued": ["switch"],
          "new conversation, same kind": ["new-same"], "new conversation, other kind": ["new-different"]}
COMPARE = [
    ("tree16", "llama-dspark", "tree verify vs llama straight DSpark"),
    ("budget", "llama-dspark", "budget + offset vs llama DSpark"),
    ("budget-accept", "llama-dspark", "budget + offset + acceptance model vs llama DSpark"),
    ("budget", "tree16", "learning (budget + offset) vs fixed tree"),
    ("budget-accept", "budget", "acceptance model on top"),
    ("tree16", "plain", "imparo tree vs imparo plain"),
    ("llama-dspark", "llama-plain", "llama DSpark vs llama plain"),
    ("plain", "llama-plain", "imparo plain vs llama plain"),
    ("budget-accept", "tree16", "acceptance model + learning vs fixed tree"),
    ("budget-accept-ngram", "budget-accept", "n-gram chains on top (design 6.6, request index)"),
    ("budget-accept-ngram-table", "budget-accept-ngram", "the stored table on top (level 2)"),
    ("budget-accept-ngram", "tree16", "n-gram + learning vs fixed tree"),
    ("budget-accept-ngram", "llama-dspark", "n-gram + learning vs llama DSpark"),
    ("budget-accept-ngram-table", "llama-dspark", "n-gram + stored table + learning vs llama DSpark"),
    ("llama-ngram", "llama-dspark", "llama k4v n-gram + DSpark vs llama DSpark"),
    ("budget-accept-ngram", "llama-ngram", "imparo n-gram tree vs llama k4v + DSpark"),
    ("chain", "llama-dspark", "imparo plain DSpark vs llama DSpark"),
    ("tree-online", "chain", "+ online models + tree verify, over plain DSpark"),
    ("ngram-chain", "chain", "+ online models + n-gram (chain), over plain DSpark"),
    ("complete", "chain", "complete form over plain DSpark"),
    ("complete", "tree-online", "complete form vs tree verify only: what the n-gram adds"),
    ("complete", "ngram-chain", "complete form vs n-gram only: what the tree adds"),
    ("ngram-chain", "tree-online", "n-gram only vs tree verify only"),
    ("complete-table", "complete", "the stored table on top"),
    ("tree-online", "tree16", "online models vs fixed tree"),
    ("budget", "chain", "the cost model's tree over plain DSpark: what tree verify sized by cost adds"),
    ("complete", "budget", "n-gram + acceptance model over the cost model's tree"),
    ("complete", "llama-dspark", "complete form vs llama DSpark"),
    ("complete", "llama-ngram", "complete form vs llama k4v + DSpark"),
    ("complete", "tree16", "complete form vs tree-verify-only (fixed 16 rows, no learning): THE GOAL"),
    ("complete-copy", "complete", "copy walk resume + earlier text on top of the complete form"),
    ("complete-table", "tree16", "complete form + stored table vs tree-verify-only (fixed 16)"),
    ("ngram-chain", "tree16", "online models + n-gram (chain) vs tree-verify-only (fixed 16)"),
    # the same comparisons under the pre-2026-09-21 arm codes, so an old run still reports
    ("T16", "LD", "tree verify vs llama straight DSpark"),
    ("b", "LD", "budget + offset vs llama DSpark"),
    ("c", "LD", "budget + offset + acceptance model vs llama DSpark"),
    ("b", "T16", "learning (budget + offset) vs fixed tree"),
    ("c", "b", "acceptance model on top"),
    ("T16", "I0", "imparo tree vs imparo plain"),
    ("LD", "L0", "llama DSpark vs llama plain"),
    ("I0", "L0", "imparo plain vs llama plain"),
    ("c", "T16", "acceptance model + learning vs fixed tree"),
    ("B", "c", "n-gram chains on top (design 6.6, request index)"),
    ("C", "B", "the stored table on top (level 2)"),
    ("B", "T16", "n-gram + learning vs fixed tree"),
    ("B", "LD", "n-gram + learning vs llama DSpark"),
    ("C", "LD", "n-gram + stored table + learning vs llama DSpark"),
    ("LN", "LD", "llama k4v n-gram + DSpark vs llama DSpark"),
    ("B", "LN", "imparo n-gram tree vs llama k4v + DSpark"),
    ("D0", "LD", "imparo plain DSpark vs llama DSpark"),
    ("TM", "D0", "+ online models + tree verify, over plain DSpark"),
    ("CN", "D0", "+ online models + n-gram (chain), over plain DSpark"),
    ("TN", "D0", "complete form over plain DSpark"),
    ("TN", "TM", "complete form vs tree verify only: what the n-gram adds"),
    ("TN", "CN", "complete form vs n-gram only: what the tree adds"),
    ("CN", "TM", "n-gram only vs tree verify only"),
    ("TND", "TN", "the stored table on top"),
    ("TM", "T16", "online models vs fixed tree"),
    ("b", "D0", "the cost model's tree over plain DSpark: what tree verify sized by cost adds"),
    ("TN", "b", "n-gram + acceptance model over the cost model's tree"),
    ("TN", "LD", "complete form vs llama DSpark"),
    ("TN", "LN", "complete form vs llama k4v + DSpark"),
    ("TN", "T16", "complete form vs tree-verify-only (fixed 16 rows, no learning): THE GOAL"),
    ("TC", "TN", "copy walk resume + earlier text on top of the complete form"),
    ("TND", "T16", "complete form + stored table vs tree-verify-only (fixed 16)"),
    ("CN", "T16", "online models + n-gram (chain) vs tree-verify-only (fixed 16)"),
]


# Which workload kind each conversation belongs to, read from the harness itself so the two cannot
# drift apart. A run with --set=all carries every kind, and the kinds pull in different directions:
# tool results are structured repeated text (n-gram food), prose has nothing to copy.
# What each arm is, in words. The short codes key the logs, the home directories and
# every results.jsonl ever written, so they stay -- but nothing a person reads should
# make them look the codes up.
ARM_LABEL = {
    # what the arms are called now
    "llama-plain":               "llama.cpp, no speculation",
    "llama-dspark":              "llama.cpp DSpark, its default draft length 3",
    "llama-dspark9":             "llama.cpp DSpark, draft length 9",
    "llama-ngram":               "llama.cpp n-gram + DSpark",
    "plain":                     "imparo plain decode, no drafter",
    "chain":                     "chain: the drafter's block, no tree, no learning",
    "tree16":                    "fixed 16-row tree, no learning",
    "budget":                    "width from the cost model",
    "budget-accept":             "cost model + acceptance model",
    "budget-accept-ngram":       "cost + acceptance + n-gram",
    "budget-accept-ngram-table": "cost + acceptance + n-gram + stored table",
    "tree-online":               "tree + both online models, no n-gram",
    "ngram-chain":               "n-gram with one child per node, no tree",
    "complete":                  "the complete form",
    "complete-table":            "the complete form + stored table",
    "complete-copy":             "the complete form + copy walk",
    # and what they were called in every results.jsonl written before 2026-09-21
    "L0": "llama.cpp, no speculation", "LD": "llama.cpp DSpark, its default draft length 3",
    "LD9": "llama.cpp DSpark, draft length 9", "LN": "llama.cpp n-gram + DSpark",
    "I0": "imparo plain decode, no drafter", "D0": "chain: the drafter's block, no tree, no learning",
    "T16": "fixed 16-row tree, no learning", "b": "width from the cost model",
    "c": "cost model + acceptance model", "B": "cost + acceptance + n-gram",
    "C": "cost + acceptance + n-gram + stored table", "TM": "tree + both online models, no n-gram",
    "CN": "n-gram with one child per node, no tree", "TN": "the complete form",
    "TND": "the complete form + stored table", "TC": "the complete form + copy walk",
}


def label(arm):
    """The arm in words, with its code, for anything a person reads."""
    return f"{arm} ({ARM_LABEL.get(arm, 'unlabelled')})"


SET_ORDER = ["agentic", "usecases", "tools", "prose", "finance"]
SET_OF = {}
try:
    from agentic import SETS as _SETS  # noqa: E402
    for _name in SET_ORDER:
        for _c, _doc, _turns in _SETS.get(_name, []):
            SET_OF[_c] = _name
except Exception as e:  # a results file can still be read without the harness importable
    print(f"(set names unavailable: {e})")


def stat(logs):
    n = len(logs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = sum(logs) / n
    if n < 2:
        return math.exp(m), float("nan"), float("nan"), n
    sd = math.sqrt(sum((x - m) ** 2 for x in logs) / (n - 1))
    h = T975.get(n - 1, 1.96) * sd / math.sqrt(n)
    return math.exp(m), math.exp(m - h), math.exp(m + h), n


def main(path):
    rows = [json.loads(l) for l in open(path)]
    order = ["plain", "chain", "tree16", "budget", "budget-accept", "budget-accept-ngram",
             "budget-accept-ngram-table", "tree-online", "ngram-chain", "complete",
             "complete-copy", "complete-table", "llama-plain", "llama-dspark", "llama-ngram",
             # the pre-2026-09-21 spellings, so an old results.jsonl still sorts
             "I0", "D0", "T16", "b", "c", "B", "C", "TM", "CN", "TN", "TC", "TND", "L0", "LD", "LN"]
    arms = sorted({r["arm"] for r in rows}, key=lambda a: order.index(a) if a in order else 99)
    cell = defaultdict(list)  # (arm, conv, turn) -> rows
    for r in rows:
        cell[(r["arm"], r["conv"], r["turn"])].append(r)
    units = sorted({(r["conv"], r["turn"]) for r in rows})
    ttype = {(r["conv"], r["turn"]): r["type"] for r in rows}

    def mean(arm, unit, key):
        xs = [r[key] for r in cell.get((arm, *unit), []) if r.get(key) is not None and r[key] == r[key]]
        return sum(xs) / len(xs) if xs else None

    def turn_s(r):
        ct, dec = r.get("completion_tokens") or 0, r.get("decode_tok_s")
        if not dec or dec != dec or ct < 2:
            return None
        return r["ttft_ms"] / 1e3 + (ct - 1) / dec

    print(f"rows {len(rows)}  arms {arms}  units {len(units)}  repeats per arm "
          f"{sorted({len(cell[(a, *u)]) for a in arms for u in units})}")
    print("\n== per arm: decode tok/s (mean over turns), TTFT ms, draft acceptance")
    for a in arms:
        decs = [mean(a, u, "decode_tok_s") for u in units]
        tt = [mean(a, u, "ttft_ms") for u in units]
        acc = [(r.get("draft_accepted"), r.get("draft_n")) for u in units for r in cell.get((a, *u), [])]
        acc = [(x, y) for x, y in acc if x is not None and y]
        rate = f"{sum(x for x, _ in acc) / sum(y for _, y in acc):.3f}" if acc else "-"
        drafted = sum(r.get("imparo_drafted", 0) for u in units for r in cell.get((a, *u), []))
        print(f"  {label(a):<46} dec {sum(d for d in decs if d)/max(1, len([d for d in decs if d])):6.2f}  "
              f"ttft {sum(t for t in tt if t)/max(1, len([t for t in tt if t])):7.1f}  "
              f"llama accept {rate}  imparo drafted requests {drafted}")

    def geo(x, y, us):
        """geometric mean of dec_x / dec_y over the given units, with its 95% interval"""
        logs = []
        for u in us:
            dx, dy = mean(x, u, "decode_tok_s"), mean(y, u, "decode_tok_s")
            if dx and dy:
                logs.append(math.log(dx / dy))
        return stat(logs)

    # THE BASELINE BY NAME, never by position. `arms[0]` is whichever arm happens to be
    # first in the results file, and the harness REVERSES the arm order on the second
    # repeat -- so a run whose first repeat was interrupted would silently report every
    # ratio against a different arm.
    base = next(
        (a for a in ("llama-dspark", "LD", "llama-plain", "L0", "plain", "I0") if a in arms),
        arms[0],
    )
    kinds = [s for s in SET_ORDER if any(SET_OF.get(u[0]) == s for u in units)]
    if kinds:
        print(f"\n== per workload kind: decode tok/s per arm (mean over that kind's rounds), "
              f"then each arm over {base}")
        print("  " + f"{'kind':<10}{'rounds':>7}" + "".join(f"{a:>9}" for a in arms)
              + "   " + "".join(f"{a + '/' + base:>14}" for a in arms if a != base))
        for s in kinds + ["ALL"]:
            us = units if s == "ALL" else [u for u in units if SET_OF.get(u[0]) == s]
            decs = []
            for a in arms:
                d = [mean(a, u, "decode_tok_s") for u in us]
                d = [x for x in d if x]
                decs.append(sum(d) / len(d) if d else float("nan"))
            line = f"  {s:<10}{len(us):>7}" + "".join(f"{d:>9.1f}" for d in decs) + "   "
            for a in arms:
                if a == base:
                    continue
                g, lo, hi, n = geo(a, base, us)
                line += f"{('x%.3f' % g) if n else '-':>7}{('[%.2f,%.2f]' % (lo, hi)) if n > 1 else '':>7}"
            print(line)

        print("\n== per conversation: decode tok/s per arm")
        print("  " + f"{'conv':<8}{'kind':<10}{'turns':>6}" + "".join(f"{a:>9}" for a in arms))
        for cname in sorted({u[0] for u in units}, key=lambda c: (SET_ORDER.index(SET_OF.get(c, "agentic")), c)):
            us = [u for u in units if u[0] == cname]
            line = f"  {cname:<8}{SET_OF.get(cname, '?'):<10}{len(us):>6}"
            for a in arms:
                d = [mean(a, u, "decode_tok_s") for u in us]
                d = [x for x in d if x]
                line += f"{(sum(d) / len(d)) if d else float('nan'):>9.1f}"
            print(line)

    print("\n== comparisons: geometric mean of per-turn ratios, 95% interval")
    for x, y, description in COMPARE:
        if x not in arms or y not in arms:
            continue
        print(f"  {label(x)} vs {label(y)}\n    {description}")
        for gname, types in [("ALL", None)] + list(GROUPS.items()):
            logs_dec, logs_turn, logs_ttft = [], [], []
            for u in units:
                if types and ttype[u] not in types:
                    continue
                dx, dy = mean(x, u, "decode_tok_s"), mean(y, u, "decode_tok_s")
                if dx and dy:
                    logs_dec.append(math.log(dx / dy))
                fx, fy = mean(x, u, "ttft_ms"), mean(y, u, "ttft_ms")
                if fx and fy:
                    logs_ttft.append(math.log(fx / fy))
                tx = [turn_s(r) for r in cell.get((x, *u), [])]
                ty = [turn_s(r) for r in cell.get((y, *u), [])]
                tx, ty = [t for t in tx if t], [t for t in ty if t]
                if tx and ty:
                    logs_turn.append(math.log((sum(ty) / len(ty)) / (sum(tx) / len(tx))))
            g, lo, hi, n = stat(logs_dec)
            gt, lot, hit, _ = stat(logs_turn)
            gf, lof, hif, _ = stat(logs_ttft)
            if n:
                print(f"    {gname:<32} n={n:>2}  decode x{g:.3f} [{lo:.3f}, {hi:.3f}]   "
                      f"turn-time speedup x{gt:.3f} [{lot:.3f}, {hit:.3f}]   ttft ratio x{gf:.3f} [{lof:.3f}, {hif:.3f}]")

    print("\n== identity: imparo arms' text per turn and repeat")
    imp = [a for a in arms if a in ("T16", "b", "c", "B", "C", "D0", "TM", "CN", "TN", "TC", "TND", "I0")]
    same = total = 0
    diffs = []
    for u in units:
        for rep in (1, 2):
            md = {a: next((r["text_md5"] for r in cell.get((a, *u), []) if r["repeat"] == rep), None) for a in imp}
            md = {a: m for a, m in md.items() if m}
            if len(md) < 2:
                continue
            total += 1
            if len(set(md.values())) == 1:
                same += 1
            else:
                diffs.append((u, rep, md))
    print(f"  equal across {imp}: {same}/{total}")
    for u, rep, md in diffs[:10]:
        print(f"    differs {u} r{rep}: " + " ".join(f"{a}={m[:8]}" for a, m in md.items()))

    print("\n== learning: dec / dec_T16 in run order (repeat 1 then 2), imparo arms")
    for a in [a for a in ("b", "c", "B", "C", "TM", "CN", "TN", "TND") if a in arms and "T16" in arms]:
        for rep in (1, 2):
            seq = []
            for u in units:
                ra = next((r for r in cell.get((a, *u), []) if r["repeat"] == rep), None)
                rt = next((r for r in cell.get(("T16", *u), []) if r["repeat"] == rep), None)
                if ra and rt and ra["decode_tok_s"] and rt["decode_tok_s"]:
                    seq.append(ra["decode_tok_s"] / rt["decode_tok_s"])
            if seq:
                half = len(seq) // 2
                print(f"  {a} r{rep}: " + " ".join(f"{s:.2f}" for s in seq) +
                      f"   first half {sum(seq[:half])/max(1,half):.3f}  second half {sum(seq[half:])/max(1,len(seq)-half):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
