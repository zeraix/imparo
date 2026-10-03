#!/usr/bin/env python3
"""Every generated table and figure in paper.html, from one run directory of run.sh.

Each fragment is written between its markers in the paper, <!--FIG:name--> ... <!--/FIG:name-->
or <!--TAB:name--> ... <!--/TAB:name-->, so the paper can be regenerated from the data alone.
A fragment whose inputs are missing is left as it is, and the script says so.

  FIG verify_cost   T(n) against rows on each target        {m}_survey/survey.tsv
  TAB context_law   per-class context slope                 {m}_survey/survey.tsv
  TAB widths        AdaSpark against its pinned widths       {m}_learned(_narrow), and each _rep
  TAB classes       learned widths against kernel classes    {m}_classes, {m}_classes_rep results
  FIG marginal      what rows 9-16 cost and buy              {m}_widths/table4.tsv
  TAB e2e           four targets, speedups per engine        {m}_full/results.jsonl
  FIG e2e           the same as bars                          {m}_full/results.jsonl
  TAB ladder        ablation per source, both LFM2.5 targets  {m}_full/results.jsonl
  TAB sources       scheduler gain per source and target      {m}_full/results.jsonl
  TAB agree         text agreement between arms              {m}_smoke, {m}_full results
  TAB accept        fitted model against the head            {m}_full/server_complete_r1.log
  TAB ngram         rows and tokens by candidate source      {m}_figs results + server log
  FIG round         anatomy of a round                        {m}_figs/server_complete_r1.log
  FIG widths        rows chosen per round, per target         {m}_figs/server_complete_r1.log
  FIG learning      cost model estimate against measured      {m}_figs/server_complete_r1.log
  FIG calibration   realised against expected accepted nodes  {m}_figs/server_*_r1.log

usage: paper_figs.py RUNDIR [PAPER]      (PAPER defaults to ../paper.html)
"""
import math, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figs  # noqa: E402
import tables  # noqa: E402

# The learned-against-declared A/B, a sibling of the campaign's run directory.

MODELS = [("lfm26", "LFM2.5-2.6B"), ("moekm", "LFM2.5-8B-A1B"), ("q4b", "Qwen3-4B"),
          ("q8b", "Qwen3-8B")]
ABBR = {"specbench": "SB", "speedbench": "SP", "wildchat": "WC", "convfinqa": "CF",
        "toolace": "TA", "longbench": "LB"}
SOURCES = [("specbench", "Spec-Bench"), ("speedbench", "SPEED-Bench"), ("wildchat", "WildChat"),
           ("convfinqa", "ConvFinQA"), ("toolace", "ToolACE"), ("longbench", "LongBench")]
FONT = 'font-family="Newsreader, serif"'


def svg(w, h, body, label):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}">\n' + "\n".join(body) +
            "\n</svg>")


def figure(inner, caption):
    return (f'<figure class="fg">\n  <div class="plate">\n{inner}\n  </div>\n'
            f'  <figcaption>{caption}</figcaption>\n</figure>')


def txt(x, y, s, size=11, anchor="start", fill="var(--ink-3)", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}"{extra}>{s}</text>')


# ---------------------------------------------------------------- survey: T(n)
def survey(path):
    """{ctx: {n: median over passes of the per-pass median verify ms}}"""
    cells = {}
    for line in open(path):
        m = re.search(r"ctx=(\d+) n=(\d+) rc=0 .*verify_ms_med=([\d.]+)", line)
        if m and m.group(3) != "nan":
            cells.setdefault(int(m.group(1)), {}).setdefault(int(m.group(2)), []).append(
                float(m.group(3)))
    return {c: {n: statistics.median(v) for n, v in d.items()} for c, d in cells.items()}


def fig_verify_cost(data):
    """2 x 2 small multiples: T(n) at the middle and longest context, one panel per target."""
    W, H, pw, ph = 640, 470, 258, 165
    body, notes = [], []
    for k, (m, name) in enumerate(MODELS):
        d = data.get(m)
        ox, oy = 62 + (k % 2) * 315, 26 + (k // 2) * 222
        body.append(txt(ox, oy - 8, name, 12.5, fill="var(--ink)", extra=' font-weight="600"'))
        if not d:
            body.append(txt(ox + pw / 2, oy + ph / 2, "not measured", 12, "middle"))
            continue
        ctxs = sorted(d)
        mid, far = ctxs[len(ctxs) // 2], ctxs[-1]
        ns = sorted(d[mid])
        vals = [v for c in (mid, far) for v in d[c].values()]
        lo = 5 * math.floor(min(vals) * 0.95 / 5)
        hi = 5 * math.ceil(max(vals) * 1.03 / 5)
        nmax = 8 * math.ceil(max(ns) / 8)
        fx = lambda n: ox + pw * n / (nmax * 1.04)
        fy = lambda v: oy + ph - ph * (v - lo) / (hi - lo)
        body.append(f'<line x1="{ox}" y1="{oy + ph}" x2="{ox + pw}" y2="{oy + ph}" '
                    f'stroke="var(--ink)" stroke-width="1"/>')
        body.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy + ph}" stroke="var(--ink)" '
                    f'stroke-width="1"/>')
        step = 10 if hi - lo <= 60 else (20 if hi - lo <= 120 else 50)
        for v in range(int(lo), int(hi) + 1, step):
            if v < lo:
                continue
            body.append(f'<line x1="{ox}" y1="{fy(v):.1f}" x2="{ox + pw}" y2="{fy(v):.1f}" '
                        f'stroke="var(--grid)" stroke-width="0.5"/>')
            body.append(txt(ox - 5, fy(v) + 4, v, 10.5, "end"))
        for n in range(8, nmax + 1, 8 if nmax <= 48 else 16):
            body.append(txt(fx(n), oy + ph + 14, n, 10.5, "middle"))
        for c, style in ((far, 'stroke="var(--ink-3)" stroke-width="1.2" stroke-dasharray="4 3"'),
                         (mid, 'stroke="var(--ink)" stroke-width="1.7"')):
            pts = [(fx(n), fy(d[c][n])) for n in ns if n in d[c]]
            body.append('<path d="M ' + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts) +
                        f'" fill="none" {style}/>')
            if c == mid:
                for a, b in pts:
                    body.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="2.3" fill="var(--ink)"/>')
        t2, t16 = d[mid].get(2), d[mid].get(16)
        if t2 and t16:
            notes.append(f"{name} {t16 / t2:.2f}")
    body.append(txt(W / 2, H - 8, "rows verified <tspan font-style=\"italic\">n</tspan>", 12.5,
                    "middle", "var(--ink-2)"))
    body.append(txt(16, H / 2, "verify (ms)", 12.5, "middle", "var(--ink-2)",
                    f' transform="rotate(-90 16 {H / 2})"'))
    ctxs = sorted(next(iter(data.values())))
    body.append(f'<line x1="440" y1="{H - 30}" x2="466" y2="{H - 30}" stroke="var(--ink)" '
                f'stroke-width="1.7"/>')
    body.append(txt(472, H - 26, f"c = {ctxs[len(ctxs) // 2]:,}", 11, fill="var(--ink-2)"))
    body.append(f'<line x1="540" y1="{H - 30}" x2="566" y2="{H - 30}" stroke="var(--ink-3)" '
                f'stroke-width="1.2" stroke-dasharray="4 3"/>')
    body.append(txt(572, H - 26, f"c = {ctxs[-1]:,}", 11, fill="var(--ink-2)"))
    cap = ('<span class="lbl" id="fig-verify_cost">Figure 0:</span> Verify time against the number of rows verified, '
           'on each target, at two context lengths (median over two passes of the per-run median '
           'over rounds that verified exactly <i>n</i> rows). Each panel has its own vertical '
           'scale. T(16)/T(2) at 1,596 tokens: ' + ", ".join(notes) + ".")
    return figure(svg(W, H, body, "Verify time against rows verified on four targets"), cap)


def tab_context_law(data):
    rows = []
    for m, name in MODELS:
        d = data.get(m)
        if not d:
            continue
        ctxs = sorted(d)
        for n in (8, 16, 32):
            ys = [d[c].get(n) for c in ctxs]
            if None in ys:
                continue
            mx, my = sum(ctxs) / len(ctxs), sum(ys) / len(ys)
            b = sum((c - mx) * (y - my) for c, y in zip(ctxs, ys)) / \
                sum((c - mx) ** 2 for c in ctxs)
            a = my - b * mx
            res = max(abs(y - (a + b * c)) / y for c, y in zip(ctxs, ys))
            rows.append(f"<tr><td>{name}</td><td>{n}</td>" +
                        "".join(f"<td>{y:.1f}</td>" for y in ys) +
                        f"<td>{a:.1f}</td><td>{b * 1000:.2f}</td><td>{res * 100:.1f}%</td></tr>".replace("-", "&minus;"))
    if not rows:
        return None
    ctxs = sorted(next(iter(data.values())))
    head = ("<tr><th>target</th><th>rows</th>" +
            "".join(f"<th><i>T</i>({c:,})</th>" for c in ctxs) +
            "<th>&alpha; (ms)</th><th>&beta; (ms per 1k tokens)</th><th>max residual</th></tr>")
    return ('<figure class="tb">\n  <figcaption><span class="lbl" id="tab-context_law">Table 0:</span> Verify time '
            '(ms) at three contexts for three row counts on each target, and the fit '
            '<i>T</i>(<i>c</i>)&nbsp;=&nbsp;&alpha;&nbsp;+&nbsp;&beta;&middot;<i>c</i> of '
            'Eq.&nbsp;(3).</figcaption>\n  <div class="scroll"><table class="compact">\n    <thead>' + head +
            '</thead>\n    <tbody>\n      ' + "\n      ".join(rows) +
            '\n    </tbody>\n  </table></div>\n</figure>')


# ---------------------------------------------------------------- widths: fixed vs chosen
def widths(path):
    """{ctx: {arm: mean ms/token over prompts and passes}}"""
    acc = {}
    for line in open(path):
        m = re.search(r"ctx=(\d+) prompt=(\w) arm=(\w+) rc=0 ms_per_token=([\d.]+)", line)
        if m:
            acc.setdefault(int(m.group(1)), {}).setdefault(m.group(3), []).append(
                float(m.group(4)))
    return {c: {a: sum(v) / len(v) for a, v in d.items()} for c, d in acc.items()}


BAND_EDGE = 2048


def pinned_width(arm):
    """`complete-wN` -> N, the pinned width of an arm, or None for any other arm."""
    m = re.fullmatch(r"complete-w(\d+)", arm)
    return int(m.group(1)) if m else None


def tab_widths_agentic(stages):
    """THE WIDTH TABLE ON REAL USE: AdaSpark against the same scheduler with its width pinned, on the
    evaluation subset -- every model, the n-gram fill and the store learning from empty as in
    AdaSpark, only the width fixed. `stages` maps a target to its stages, each `(rows, repeat)`: one
    agentic run of AdaSpark and some pinned arms, then AdaSpark again after them. A pinned arm is
    compared with the AdaSpark runs of ITS stage, whose speed per reply is the mean of the two runs,
    so a drift across the stage cancels. Each cell is a pinned arm over AdaSpark, geometric mean of
    per-reply decode ratios (above 1 is faster than AdaSpark), for all replies and by the context
    the reply was decoded at; the last columns give AdaSpark over the best pinned width with its 95%
    interval, and AdaSpark's second run over its first in the first stage."""
    widths = sorted({pinned_width(r["arm"]) for st in stages.values() for rs, _ in st for r in rs}
                    - {None})
    rows = []
    for m, name in MODELS:
        st = [(rs, rep) for rs, rep in stages.get(m, []) if any(r["arm"] == "complete" for r in rs)]
        if not st:
            continue
        ctx = {(r["set"], r["conv"], r["turn"], r["type"]): r["prompt_tokens"] + (r["completion_tokens"] or 0) / 2
               for r in st[0][0] if r["arm"] == "complete"}
        bands = [("all replies", None), (f"under {BAND_EDGE:,}", lambda u: ctx.get(u, 0) < BAND_EDGE),
                 (f"{BAND_EDGE:,} and over", lambda u: ctx.get(u, 0) >= BAND_EDGE)]
        # Each stage's rows with AdaSpark's two runs together, averaged per reply by `ratios`.
        merged = [rs + [r for r in rep if r["arm"] == "complete"] for rs, rep in st]
        first_rs, first_rep = st[0]
        both = first_rs + [dict(r, arm="complete@2") for r in first_rep if r["arm"] == "complete"]
        for k, (label, keep) in enumerate(bands):
            fixed, home = {}, {}
            for rs in merged:
                for w in sorted({pinned_width(r["arm"]) for r in rs} - {None}):
                    g = ratios(rs, f"complete-w{w}", "complete", keep=keep)
                    if g:
                        fixed[w], home[w] = g, rs
            if not fixed:
                continue
            best_w = max(fixed, key=lambda w: fixed[w]["x"])
            cells = []
            for w in widths:
                g = fixed.get(w)
                if not g:
                    cells.append("<td>&ndash;</td>")
                    continue
                v = f"{g['x']:.3f}"
                cells.append(f"<td><b>{v}</b></td>" if w == best_w else f"<td>{v}</td>")
            best = ratios(home[best_w], "complete", f"complete-w{best_w}", keep=keep)
            again = ratios(both, "complete@2", "complete", keep=keep)
            again_s = f"{again['x']:.3f}" if again else "&ndash;"
            n = ratios(merged[0], "complete-w8", "complete", keep=keep) or fixed[best_w]
            rows.append(f"<tr><td>{name if k == 0 else ''}</td><td class=\"txt\">{label}</td>"
                        f"<td>{n['n']}</td>" + "".join(cells) +
                        f"<td>{ci(best, 3)}</td><td>{again_s}</td></tr>")
    if not rows:
        return None
    head = "".join(f"<th>{w}</th>" for w in widths)
    return ('<figure class="tb">\n  <figcaption><span class="lbl" id="tab-widths">Table 0:</span> The width on '
            'the evaluation subset: AdaSpark against the same scheduler with its width pinned, every other part unchanged '
            '(both online models, the n-gram fill, a store learning from empty). The width columns are each pinned arm\'s '
            'decode speed over AdaSpark\'s, geometric mean of per-reply ratios (above 1 is faster than AdaSpark), with the '
            'best pinned width in bold, for all replies and by the context a reply was decoded at (prompt plus half the '
            'reply). AdaSpark runs before the pinned arms and again after them, and its speed on a reply is the mean of the '
            'two runs. The routed target is also pinned at 4, 5 and 6 rows, in a second stage run the same way and compared '
            'with its own two AdaSpark runs; a dash marks a width not run (&sect;3.4). '
            '<i>AdaSpark / best</i> is AdaSpark over the best pinned width with its 95% bootstrap interval over '
            'conversations; <i>again</i> is AdaSpark\'s second run over its first. Looped and single-delta replies are left '
            'out as in <a class="xref" href="#tab-e2e">Table&nbsp;7</a>.</figcaption>\n'
            '  <div class="scroll"><table class="compact">\n'
            '    <thead><tr><th>target</th><th>replies</th><th>n</th>' + head +
            '<th>AdaSpark / best</th><th>again</th></tr></thead>\n'
            '    <tbody>\n      ' + "\n      ".join(rows) + '\n    </tbody>\n  </table></div>\n</figure>')


def widths_rows(path):
    """{(ctx, prompt): {arm: (ms_per_token, tokens_per_round)}}, averaged over passes."""
    acc = {}
    for line in open(path):
        m = re.search(r"ctx=(\d+) prompt=(\w) arm=(\w+) rc=0 ms_per_token=([\d.]+) "
                      r"tokens_per_round=([\d.]+)", line)
        if m:
            acc.setdefault((int(m.group(1)), m.group(2)), {}).setdefault(m.group(3), []).append(
                (float(m.group(4)), float(m.group(5))))
    return {k: {a: (sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v))
                for a, v in d.items()} for k, d in acc.items()}


def fig_marginal(paths):
    """What the second eight rows cost against what they buy: per target and (context, prompt),
    extra time per round of a 16-row tree over an 8-row one (x) against extra accepted tokens per
    round (y), both relative to the 8-row tree. Above the diagonal the wider tree is faster."""
    marks = {"lfm26": ("circle", "var(--ink)"), "moekm": ("square", "var(--ink)"),
             "q4b": ("circle", "none"), "q8b": ("square", "none")}
    pts = {}
    for m, name in MODELS:
        if m not in paths:
            continue
        for key, d in widths_rows(paths[m]).items():
            if "8" in d and "16" in d:
                (m8, k8), (m16, k16) = d["8"], d["16"]
                t8, t16 = m8 * k8, m16 * k16
                pts.setdefault(m, []).append((t16 / t8 - 1, k16 / k8 - 1))
    if not pts:
        return None
    W, H, x0, y0, x1, y1 = 620, 330, 64, 20, 600, 280
    allx = [x for v in pts.values() for x, _ in v]
    ally = [y for v in pts.values() for _, y in v]
    top = max(0.2, max(allx + ally) * 1.1)
    bot = min(0.0, min(ally) * 1.1)
    left = min(0.0, min(allx) * 1.3)
    fx = lambda v: x0 + (x1 - x0) * (v - left) / (top - left)
    fy = lambda v: y1 - (y1 - y0) * (v - bot) / (top - bot)
    body = [f'<line x1="{x0}" y1="{fy(0):.1f}" x2="{x1}" y2="{fy(0):.1f}" stroke="var(--ink)" stroke-width="1"/>',
            f'<line x1="{fx(0):.1f}" y1="{y0}" x2="{fx(0):.1f}" y2="{y1}" stroke="var(--ink)" stroke-width="1"/>']
    step = 0.1 if top <= 0.6 else (0.2 if top <= 1.2 else 0.5)
    v = 0.0
    while v <= top + 1e-9:
        body.append(f'<line x1="{fx(v):.1f}" y1="{y0}" x2="{fx(v):.1f}" y2="{y1}" stroke="var(--grid)" stroke-width="0.5"/>')
        body.append(txt(fx(v), y1 + 14, f"+{v * 100:.0f}%", 10.5, "middle"))
        if v >= bot:
            body.append(f'<line x1="{x0}" y1="{fy(v):.1f}" x2="{x1}" y2="{fy(v):.1f}" stroke="var(--grid)" stroke-width="0.5"/>')
            body.append(txt(x0 - 5, fy(v) + 4, f"+{v * 100:.0f}%", 10.5, "end"))
        v += step
    d = min(top, top)
    body.append(f'<line x1="{fx(0):.1f}" y1="{fy(0):.1f}" x2="{fx(d):.1f}" y2="{fy(d):.1f}" stroke="var(--ink-3)" '
                f'stroke-width="1" stroke-dasharray="4 3"/>')
    body.append(txt(fx(d * 0.72), fy(d * 0.72) - 8, "break-even", 11, "end", "var(--ink-3)", ' font-style="italic"'))
    for m, v in pts.items():
        shape, fill = marks[m]
        for x, y in v:
            if shape == "circle":
                body.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="3.6" fill="{fill}" stroke="var(--ink)" stroke-width="1"/>')
            else:
                body.append(f'<rect x="{fx(x) - 3.4:.1f}" y="{fy(y) - 3.4:.1f}" width="6.8" height="6.8" fill="{fill}" stroke="var(--ink)" stroke-width="1"/>')
    lx = x0 + 12
    for k, (m, name) in enumerate([(m, n) for m, n in MODELS if m in pts]):
        shape, fill = marks[m]
        yy = y0 + 12 + 16 * k
        if shape == "circle":
            body.append(f'<circle cx="{lx}" cy="{yy - 4}" r="3.6" fill="{fill}" stroke="var(--ink)" stroke-width="1"/>')
        else:
            body.append(f'<rect x="{lx - 3.4}" y="{yy - 7.4}" width="6.8" height="6.8" fill="{fill}" stroke="var(--ink)" stroke-width="1"/>')
        body.append(txt(lx + 10, yy, name, 11.5, fill="var(--ink-2)"))
    body.append(txt((x0 + x1) / 2, H - 6, "extra time per round, 16 rows over 8", 12.5, "middle", "var(--ink-2)"))
    body.append(txt(16, (y0 + y1) / 2, "extra accepted tokens per round", 12.5, "middle", "var(--ink-2)",
                    f' transform="rotate(-90 16 {(y0 + y1) / 2})"'))
    cap = ('<span class="lbl" id="fig-marginal">Figure 0:</span> What the second eight rows of a tree cost and what they buy. '
           'Each point is one target at one context (443, 1,596 or 8,444 tokens) and one of three prompts, from runs of 256 '
           'tokens at a fixed width of 8 and of 16 rows (acceptance model on, n-gram candidates off; setup notes): the extra time per round of a '
           '16-row tree over an 8-row tree (the drafter included), against the extra tokens it accepts per '
           'round, both relative to the 8-row tree. Above the diagonal the wider tree decodes faster. The '
           'diagonal is the break-even line for every target.')
    return figure(svg(W, H, body, "Marginal cost and marginal tokens of the second eight rows"), cap)


# ---------------------------------------------------------------- end to end
def e2e(path):
    import io, contextlib
    argv = sys.argv
    sys.argv = ["tables.py", path, "--json", "/dev/null"]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rows_ = [__import__("json").loads(l) for l in open(path)]
        return rows_
    finally:
        sys.argv = argv


def speed(r):
    """Client-clock decode tok/s of a reply, or None when the client clock has no interval: a
    reply streamed as one delta (a bare tool call) has no first-to-last token time."""
    v = r.get("decode_tok_s")
    return v if isinstance(v, (int, float)) and math.isfinite(v) and v > 0 else None


def ratios(rows, a, ref, sets=None, finished_only=True, keep=None):
    """Geometric mean over replies of decode_a / decode_ref, with a 95% percentile bootstrap
    interval that resamples whole CONVERSATIONS: replies of one conversation share their history
    and the arm's learned state, so they are not independent. A reply that reached the token cap
    in either arm is a repetition loop under greedy decoding: trivially predictable text, which
    would flatter every speculative arm, so by default it is left out (and counted by `cut`)."""
    import collections, random
    cell = collections.defaultdict(list)
    for r in rows:
        cell[(r["arm"], (r["set"], r["conv"], r["turn"], r["type"]))].append(r)
    units = {u for _, u in cell}
    by_conv = collections.defaultdict(list)
    for u in sorted(units):
        if sets and u[0] not in sets:
            continue
        if keep and not keep(u):
            continue
        x, y = cell.get((a, u)), cell.get((ref, u))
        if not (x and y) or not all(speed(r) for r in x + y):
            continue
        if finished_only and not all(r["finished"] for r in x + y):
            continue
        by_conv[(u[0], u[1])].append(math.log(sum(speed(r) for r in x) / len(x) /
                                              (sum(speed(r) for r in y) / len(y))))
    logs = [v for vs in by_conv.values() for v in vs]
    if not logs:
        return None
    g = {"x": math.exp(sum(logs) / len(logs)), "lo": None, "hi": None, "n": len(logs),
         "convs": len(by_conv)}
    convs = list(by_conv.values())
    if len(convs) >= 2:
        rng = random.Random(20260927)
        means = []
        for _ in range(2000):
            pick = [v for _ in convs for v in convs[rng.randrange(len(convs))]]
            means.append(sum(pick) / len(pick))
        means.sort()
        g["lo"], g["hi"] = math.exp(means[49]), math.exp(means[1949])
    return g


def unclocked(rows, arms):
    """Replies (set, conv, turn, type) whose client clock has no decode interval in some arm."""
    return {(r["set"], r["conv"], r["turn"], r["type"]) for r in rows
            if r["arm"] in arms and not speed(r) and not r.get("refused")}


def refused(rows, arms):
    """Conversations (set, conv) a server refused in some arm; the refusing arm has no reply there
    and skipped the rest of the conversation, so ratios against it cannot include them."""
    return {(r["set"], r["conv"]) for r in rows if r["arm"] in arms and r.get("refused")}


def cut(rows, arms):
    """Rounds (set, conv, turn, type) that reached the cap in any of `arms`."""
    return {(r["set"], r["conv"], r["turn"], r["type"]) for r in rows
            if r["arm"] in arms and not r["finished"]}


def throughput(rows, a, finished_only=True):
    rs = [r for r in rows if r["arm"] == a and speed(r) and r["completion_tokens"] > 1
          and (r["finished"] or not finished_only)]
    toks = sum(r["completion_tokens"] for r in rs)
    secs = sum((r["completion_tokens"] - 1) / speed(r) for r in rs)
    return toks / secs if secs else None


def signed(x):
    """A signed percentage with one decimal; a value that rounds to zero prints unsigned."""
    return "0.0%" if abs(x) < 0.05 else f"{x:+.1f}%".replace("-", "&minus;")


def ci(g, digits=2):
    if not g:
        return "&ndash;"
    if g["lo"] is None:
        return f"{g['x']:.{digits}f}&times;"
    return (f"{g['x']:.{digits}f}&times;<br><span class=\"ci\">"
            f"[{g['lo']:.{digits}f}, {g['hi']:.{digits}f}]</span>")


E2E_ARMS = {"llama-dspark", "chain3", "complete"}


def tab_e2e(runs, smoke):
    """Speed on the evaluation subset (the speculative arms), and each engine's speedup over its
    own autoregressive decode: its speculative tok/s on the evaluation subset over its autoregressive
    tok/s, which is measured on the short subset only (autoregressive speed barely depends on the
    text, and the short subset's shorter contexts make it, if anything, faster)."""
    rows, single, ar = [], [], []
    for m, name in MODELS:
        rs, sm = runs.get(m), smoke.get(m) or []
        if not rs or not E2E_ARMS <= {r["arm"] for r in rs}:
            continue
        k = len({(r['set'], r['conv'], r['turn'], r['type']) for r in rs if r['arm'] in E2E_ARMS and not speed(r)})
        single.append(f"{k or 'none'} on {name}")
        have, sh = {r["arm"] for r in rs}, {r["arm"] for r in sm}
        tp = lambda xs, a: f"{throughput(xs, a):.1f}" if throughput(xs, a) else "&ndash;"
        sp = lambda a, ref: (f"{throughput(rs, a) / throughput(sm, ref):.2f}&times;"
                             if a in have and ref in sh and throughput(sm, ref) else "&ndash;")
        if {"plain", "llama-plain"} <= sh:
            ar.append(f"{name} {throughput(sm, 'llama-plain'):.1f} and {throughput(sm, 'plain'):.1f}")
        rows.append(
            f"<tr><td>{name}</td><td>{tp(rs, 'llama-dspark')}</td><td>{tp(rs, 'chain3')}</td>"
            f"<td>{tp(rs, 'complete')}</td><td>{ci(ratios(rs, 'complete', 'llama-dspark'))}</td>"
            f"<td>{len(cut(rs, have))}</td>"
            f"<td>{sp('llama-dspark', 'llama-plain')}</td><td>{sp('complete', 'plain')}</td></tr>")
    if not rows:
        return None
    return ('<figure class="tb">\n  <figcaption><span class="lbl" id="tab-e2e">Table 0:</span> End to end. '
            'Left: decode tok/s on the evaluation subset (Appendix&nbsp;A; generated tokens over decode time, '
            'summed over replies) and AdaSpark over llama.cpp DSpark (geometric mean of per-reply '
            'ratios, 95% bootstrap interval over conversations). Replies that reached the 8,192-token cap in any arm are repetition '
            'loops under greedy decoding; they are left out of every column and counted. A reply streamed as a single delta (a bare tool call) has no client-clock decode interval and is left out as well (' + ', '.join(single) + '). Right: each '
            'engine\'s speculation speedup over its own autoregressive (AR) decode: its tok/s on the evaluation subset over '
            'its AR tok/s, which is measured on the short subset (setup notes) only (llama.cpp and imparo: ' + '; '.join(ar) +
            ' tok/s). AR speed barely depends on the text; the short subset\'s contexts are shorter, where AR decode is '
            'faster, so these speedups are if anything understated.</figcaption>\n  <div class="scroll"><table class="compact">\n    <thead>'
            '<tr><th rowspan="2">target</th><th colspan="5">evaluation subset: tok/s and ratio</th>'
            '<th colspan="2">over own AR</th></tr>'
            '<tr><th>llama.cpp<br>DSpark</th><th>imparo<br>DSpark</th><th>AdaSpark</th>'
            '<th>AdaSpark /<br>llama.cpp</th><th>looped</th>'
            '<th>llama.cpp<br>DSpark</th><th>AdaSpark</th></tr></thead>\n    <tbody>\n      ' +
            "\n      ".join(rows) + '\n    </tbody>\n  </table></div>\n</figure>')


def fig_e2e(runs):
    """Per target: the engine factor (imparo DSpark over llama.cpp DSpark, same configuration),
    the scheduler factor (AdaSpark over imparo DSpark, same engine) and their product."""
    names = [(m, n) for m, n in MODELS if runs.get(m) and E2E_ARMS <= {r["arm"] for r in runs[m]}]
    if not names:
        return None
    bars = [("imparo DSpark / llama.cpp DSpark", "chain3", "llama-dspark", "var(--band)"),
            ("AdaSpark / imparo DSpark", "complete", "chain3", "var(--ink-3)"),
            ("AdaSpark / llama.cpp DSpark", "complete", "llama-dspark", "var(--ink)")]
    W, bar, gap = 640, 12, 26
    H = 44 + len(names) * (len(bars) * bar + gap) + 40
    x0, x1 = 150, 600
    # Every factor over the SAME replies: those llama.cpp served, so the engine and scheduler factors
    # multiply to the total even where a server refuses a conversation the other serves.
    def served(rows):
        keep = {(r["set"], r["conv"], r["turn"], r["type"]) for r in rows
                if r["arm"] == "llama-dspark" and speed(r)}
        return [r for r in rows if (r["set"], r["conv"], r["turn"], r["type"]) in keep]
    vals = {m: [ratios(served(runs[m]), a, b) for _, a, b, _ in bars] for m, _ in names}
    top = max(g["hi"] or g["x"] for v in vals.values() for g in v if g)
    top = math.ceil(top * 2) / 2
    fx = lambda v: x0 + (x1 - x0) * v / top
    body = []
    v = 0.0
    while v <= top + 1e-9:
        body.append(f'<line x1="{fx(v):.1f}" y1="40" x2="{fx(v):.1f}" y2="{H - 34}" '
                    f'stroke="var(--grid)" stroke-width="0.5"/>')
        body.append(txt(fx(v), H - 20, f"{v:g}&times;", 10.5, "middle"))
        v += 0.5
    body.append(f'<line x1="{x0}" y1="40" x2="{x0}" y2="{H - 34}" stroke="var(--ink)" stroke-width="1"/>')
    body.append(f'<line x1="{fx(1):.1f}" y1="40" x2="{fx(1):.1f}" y2="{H - 34}" '
                f'stroke="var(--ink-3)" stroke-width="1" stroke-dasharray="3 3"/>')
    lx = 20
    for lab, _, _, fill in bars:
        body.append(f'<rect x="{lx}" y="6" width="12" height="9" fill="{fill}" stroke="var(--ink)" '
                    f'stroke-width="0.6"/>')
        body.append(txt(lx + 16, 14, lab, 10.5, fill="var(--ink-2)"))
        lx += 16 + 5.6 * len(lab) + 24  # 5.6 px a character at 10.5 px, measured on the render
    y = 46
    for m, name in names:
        body.append(txt(x0 - 10, y + len(bars) * bar / 2 + 4, name, 12, "end", "var(--ink-2)"))
        for k, ((_, _, _, fill), g) in enumerate(zip(bars, vals[m])):
            if not g:
                continue
            yy = y + k * bar
            body.append(f'<rect x="{x0}" y="{yy}" width="{fx(g["x"]) - x0:.1f}" height="{bar - 2}" '
                        f'fill="{fill}" stroke="var(--ink)" stroke-width="0.6"/>')
            if g["lo"]:
                body.append(f'<line x1="{fx(g["lo"]):.1f}" y1="{yy + (bar - 2) / 2:.1f}" '
                            f'x2="{fx(g["hi"]):.1f}" y2="{yy + (bar - 2) / 2:.1f}" '
                            f'stroke="var(--ink-2)" stroke-width="1"/>')
            body.append(txt(fx(g["hi"] or g["x"]) + 5, yy + bar - 3, f"{g['x']:.2f}&times;", 10.5,
                            fill="var(--ink-2)"))
        y += len(bars) * bar + gap
    body.append(txt((x0 + x1) / 2, H - 4, "ratio of decode speed, geometric mean over replies",
                    12, "middle", "var(--ink-2)"))
    cap = ('<span class="lbl" id="fig-e2e">Figure 0:</span> Where the gain over llama.cpp comes from. The engine '
           'factor is imparo DSpark over llama.cpp DSpark, the two engines running the same algorithm (a chain of at most three '
           'draft tokens); the scheduler factor is AdaSpark over imparo DSpark, on the same '
           'engine; the total is AdaSpark over llama.cpp DSpark, the product of the two, over the replies both engines served. Whiskers are 95% bootstrap '
           'intervals over conversations.')
    return figure(svg(W, H, body, "Engine and scheduler factors per target"), cap)


def tab_ladder(runs):
    """The ablation, one block of rows per target that ran it: each row adds to the row above."""
    arms = [("chain3", "imparo DSpark (3-token chain)"), ("chain", "full-block chain"),
            ("tree16", "fixed 16-row tree"), ("budget", "cost-model width"),
            ("complete", "AdaSpark (+ acceptance model, n-gram)")]
    out = []
    for m, name in MODELS:
        rows = runs.get(m)
        if not rows:
            continue
        have = {r["arm"] for r in rows}
        if not {"llama-dspark", "chain", "budget"} <= have:
            continue
        out.append(f'<tr class="grp"><td colspan="{len(SOURCES) + 2}">{name}</td></tr>')
        out.append("<tr><td>llama.cpp DSpark (tok/s)</td>" +
                   "".join(f"<td>{throughput([r for r in rows if r['set'] == s], 'llama-dspark'):.1f}</td>"
                           for s, _ in SOURCES) +
                   f"<td>{throughput(rows, 'llama-dspark'):.1f}</td></tr>")
        for a, lab in arms:
            if a not in have:
                continue
            cls = ' class="em"' if a == "complete" else ""
            cells = "".join(f"<td>{ratios(rows, a, 'llama-dspark', {s})['x']:.2f}</td>"
                            for s, _ in SOURCES)
            out.append(f"<tr{cls}><td>{lab}</td>{cells}<td>{ci(ratios(rows, a, 'llama-dspark'))}</td></tr>")
    if not out:
        return None
    return ('<figure class="tb">\n  <figcaption><span class="lbl" id="tab-ladder">Table 0:</span> Ablation on the '
            'dense and the routed LFM2.5 targets. The first row of each block is llama.cpp '
            'DSpark\'s tok/s per source; every other cell is the arm over it, the geometric mean of '
            'per-reply decode ratios (looped replies excluded), with the 95% bootstrap interval over conversations in the last column. From the fixed tree down, each row adds to the row above: the cost model\'s width, then the '
            'fitted acceptance model and n-gram candidates together. Sources: SB Spec-Bench, SP SPEED-Bench, WC WildChat, CF ConvFinQA, TA ToolACE, LB LongBench.</figcaption>\n  <div class="scroll"><table class="compact">\n'
            '    <thead><tr><th>arm</th>' + "".join(f"<th>{ABBR[k]}</th>" for k, _ in SOURCES) +
            '<th>all replies</th></tr></thead>\n    <tbody>\n      ' + "\n      ".join(out) +
            '\n    </tbody>\n  </table></div>\n</figure>')


def agreement(rows, a, b):
    """Rounds (same set, conversation, turn, type) where arms a and b emitted byte-identical text,
    reasoning included, over the rounds both ran; with the first turn only, where the two arms'
    histories cannot yet differ."""
    import collections
    cell = collections.defaultdict(dict)
    for r in rows:
        cell[(r["set"], r["conv"], r["turn"], r["type"])][r["arm"]] = r["text_md5"]
    both = [(u, v) for u, v in cell.items() if a in v and b in v]
    first = [(u, v) for u, v in both if u[2] == 1 and "/result" not in u[3]]
    same = lambda xs: sum(1 for _, v in xs if v[a] == v[b])
    return same(both), len(both), same(first), len(first)


def tab_agree(full):
    """AdaSpark's text against imparo's three-pick DSpark on the evaluation subset: the same verify
    rule, a different tree, so equal text means the tree's shape did not change what was emitted."""
    rows = []
    for m, name in MODELS:
        rs = full.get(m)
        if not rs or not {"complete", "chain3"} <= {r["arm"] for r in rs}:
            continue
        s, n, s1, n1 = agreement(rs, "complete", "chain3")
        rows.append(f"<tr><td>{name}</td><td>{s}/{n}</td><td>{s1}/{n1}</td></tr>")
    if not rows:
        return None
    return ('<figure class="tb">\n  <figcaption><span class="lbl" id="tab-agree">Table 0:</span> Replies of the '
            'evaluation subset whose text, reasoning included, is byte-identical between AdaSpark and imparo\'s '
            'three-pick DSpark. Later turns carry each arm\'s own earlier replies, so one early difference makes every '
            'later turn of that conversation differ; first turns share their history and isolate the verify.</figcaption>\n'
            '  <div class="scroll"><table class="compact">\n    <thead><tr><th>target</th><th>all replies</th>'
            '<th>first turns</th></tr></thead>\n    <tbody>\n      ' + "\n      ".join(rows) +
            '\n    </tbody>\n  </table></div>\n</figure>')


# The ToolACE conversations whose tool schemas build.py's type rename changed, in the evaluation
# subset and in the held-out part.
TOOLACE_RENAMED_CORE = {"ta-2", "ta-4"}
TOOLACE_RENAMED_REST = {"ta-5", "ta-6", "ta-7", "ta-8"}


def toolace_control(runs, toolace):
    """ta-1 and ta-3 did not change: each arm's speed on them in the ToolACE sub-run, where the
    learned state has seen only the ToolACE conversations before them, over its speed in the run
    they belong to. Printed for the text; it bounds what the shorter history cost the replaced rows."""
    for m, name in MODELS:
        if m not in toolace or m not in runs:
            continue
        out = []
        for a in ("llama-dspark", "complete"):
            new = [dict(r, arm=a + "@sub") for r in toolace[m] if r["arm"] == a and r["conv"] in ("ta-1", "ta-3")]
            old = [r for r in runs[m] if r["arm"] == a and r["set"] == "toolace" and r["conv"] in ("ta-1", "ta-3")]
            g = ratios(old + new, a + "@sub", a, finished_only=True)
            if g:
                out.append(f"{a} {g['x']:.3f} (n={g['n']})")
        print(f"  toolace control {name}: sub-run over run, ta-1 and ta-3: " + ", ".join(out))


def tab_heldout(runs, heldout):
    """AdaSpark over llama.cpp's DSpark on the evaluation subset, on the held-out conversations no
    arm had run when the subset was measured, and on the whole test set (both together)."""
    rows = []
    for m, name in MODELS:
        ev, ho = runs.get(m) or [], heldout.get(m) or []
        if not ho or not {"complete", "llama-dspark"} <= {r["arm"] for r in ho}:
            continue
        both = ev + ho
        cells = []
        for rs in (ev, ho, both):
            g = ratios(rs, "complete", "llama-dspark")
            cells.append(f"<td>{ci(g)}</td><td>{g['n'] if g else '&ndash;'}</td>")
        rows.append(f"<tr><td>{name}</td>{''.join(cells)}<td>{len(cut(ho, {'complete', 'llama-dspark'}))}</td></tr>")
    if not rows:
        return None
    return ('<figure class="tb">\n  <figcaption><span class="lbl" id="tab-heldout">Table 0:</span> AdaSpark over '
            'llama.cpp DSpark on the evaluation subset, on the held-out conversations (the 31 conversations, 61 turns, of '
            'the test set outside the subset, which no arm had run when the subset was measured) and on the whole test set '
            '(73 conversations, 148 turns): geometric mean of per-reply decode ratios with the 95% bootstrap interval over '
            'conversations, and the replies it rests on. Looped and single-delta replies are '
            'left out as in <a class="xref" href="#tab-e2e">Table&nbsp;7</a>; the last column counts the held-out replies '
            'that reached the cap.</figcaption>\n  <div class="scroll"><table class="compact">\n    <thead>'
            '<tr><th rowspan="2">target</th><th colspan="2">evaluation subset</th><th colspan="2">held out</th>'
            '<th colspan="2">whole test set</th><th rowspan="2">looped,<br>held out</th></tr>'
            '<tr><th>ratio</th><th>replies</th><th>ratio</th><th>replies</th><th>ratio</th><th>replies</th></tr></thead>\n'
            '    <tbody>\n      ' + "\n      ".join(rows) + '\n    </tbody>\n  </table></div>\n</figure>')


def tab_sources(runs):
    """The scheduler's gain by workload: AdaSpark over imparo DSpark (same engine, same rounds),
    per source and per target, cells shaded by size."""
    have = [(m, n) for m, n in MODELS if runs.get(m) and {"complete", "chain3"} <= {r["arm"] for r in runs[m]}]
    if not have:
        return None
    vals = {(m, s): ratios(runs[m], "complete", "chain3", {s}) for m, _ in have for s, _ in SOURCES}
    xs = [g["x"] for g in vals.values() if g]
    lo, hi = min(xs), max(xs)
    def cell(g):
        if not g:
            return "<td>&ndash;</td>"
        t = 0.0 if hi <= lo else (g["x"] - lo) / (hi - lo)
        shade = f"color-mix(in srgb, var(--ink) {int(8 + 30 * t)}%, var(--paper))"
        return f'<td style="background:{shade}">{g["x"]:.2f}</td>'
    head = "<tr><th>source</th>" + "".join(f"<th>{n}</th>" for _, n in have) + "</tr>"
    body = []
    for s, sn in SOURCES:
        body.append(f"<tr><td>{sn}</td>" + "".join(cell(vals[(m, s)]) for m, _ in have) + "</tr>")
    body.append('<tr class="em"><td>all replies</td>' +
                "".join(f"<td>{ratios(runs[m], 'complete', 'chain3')['x']:.2f}</td>" for m, _ in have) + "</tr>")
    return ('<figure class="tb">\n  <figcaption><span class="lbl" id="tab-sources">Table 0:</span> The scheduler\'s '
            'gain by workload: AdaSpark over imparo DSpark (same engine, a chain of at most three draft '
            'tokens), geometric mean of per-reply decode ratios per source, looped replies excluded. '
            'Darker cells are larger gains.</figcaption>\n  <div class="scroll"><table class="compact">\n    <thead>' +
            head + '</thead>\n    <tbody>\n      ' + "\n      ".join(body) +
            '\n    </tbody>\n  </table></div>\n</figure>')


# ---------------------------------------------------------------- acceptance verdicts
VERDICT = re.compile(r"dspark accept model_ll=([\d.]+) today_ll=([\d.]+) nodes=(\d+) "
                     r"priced=(\d+)/(\d+)")


def verdicts(log):
    """Per request: (model log loss per node, head-based log loss per node, nodes, priced, rounds)."""
    out = []
    for line in open(log, errors="replace"):
        m = VERDICT.search(line)
        if m:
            out.append((float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)),
                        int(m.group(5))))
    return out


def tab_accept(logs):
    rows = []
    for m, name in MODELS:
        v = verdicts(logs[m]) if m in logs else []
        if len(v) < 2:
            continue
        first, rest = v[0], v[1:]
        n = sum(x[2] for x in rest)
        lm = sum(x[0] * x[2] for x in rest) / n
        lt = sum(x[1] * x[2] for x in rest) / n
        wins = sum(1 for x in rest if x[0] < x[1])
        pr = sum(x[3] for x in rest) / max(1, sum(x[4] for x in rest))
        change = f"{(lm / lt - 1) * 100:+.1f}%".replace("-", "&minus;")
        rows.append(f"<tr><td>{name}</td><td>{len(rest)}</td><td>{n:,}</td><td>{lt:.4f}</td>"
                    f"<td>{lm:.4f}</td><td>{change}</td>"
                    f"<td>{wins}/{len(rest)}</td><td>{pr * 100:.0f}%</td>"
                    f"<td>{first[3]}/{first[4]}</td></tr>")
    if not rows:
        return None
    return ('<figure class="tb">\n  <figcaption><span class="lbl" id="tab-accept">Table 0:</span> Acceptance '
            'prediction on the evaluation subset: log loss per labelled node of the head-based estimate '
            '(the drafter\'s confidence head with its per-request offset) and of the fitted model, '
            'both scored on each round\'s labels before either learned from them, over every '
            'request after the first that did not reach the token cap. <i>priced</i> is the share of rounds whose tree the fitted '
            'model priced; the last column gives rounds priced over tree rounds for the first request, which starts '
            'from an empty store.</figcaption>\n  <div class="scroll"><table class="compact">\n    <thead><tr>'
            '<th>target</th><th>requests</th><th>nodes</th><th>head-based</th><th>fitted</th>'
            '<th>change</th><th>requests won</th><th>priced</th><th>first request</th></tr>'
            '</thead>\n    <tbody>\n      ' + "\n      ".join(rows) +
            '\n    </tbody>\n  </table></div>\n</figure>')


def fig_accept_curve(logs, fullruns):
    """Per request, in serving order: log loss per labelled node of the head-based estimate and
    of the fitted model, one panel per target, the source sets shaded. The online model's whole
    history on one axis: where it starts, when it overtakes the head, and whether a change of
    workload sets it back."""
    SHORT = {"specbench": "SB", "speedbench": "SP", "wildchat": "WC", "convfinqa": "CF",
             "toolace": "TA", "longbench": "LB"}
    W, H, pw, ph = 640, 470, 258, 160
    body, notes, switch_ranks = [], [], []
    have = [(m, n) for m, n in MODELS if m in logs]
    if not have:
        return None
    for k, (m, name) in enumerate(MODELS):
        ox, oy = 62 + (k % 2) * 315, 30 + (k // 2) * 222
        body.append(txt(ox, oy - 10, name, 12.5, fill="var(--ink)", extra=' font-weight="600"'))
        v = verdicts(logs[m]) if m in logs else []
        if len(v) < 2:
            body.append(txt(ox + pw / 2, oy + ph / 2, "not measured", 12, "middle"))
            continue
        sets = [r["set"] for r in fullruns.get(m, []) if r["arm"] == "complete" and r["finished"]]
        # the reduction in log loss per request, 1 - fitted / head-based: above zero, the fitted
        # model predicted that request's accept outcomes better
        red = [(1 - row[0] / row[1]) * 100 if row[1] > 0 else 0.0 for row in v]
        hi = max(5.0, 5 * math.ceil(max(red) * 1.1 / 5))
        lo = min(0.0, 5 * math.floor(min(red) * 1.1 / 5))
        fx = lambda i: ox + pw * (i + 0.5) / len(v)
        fy = lambda y: oy + ph - ph * (y - lo) / (hi - lo)
        # source bands: alternate shading, a short label on top
        if len(sets) == len(v):
            start = 0
            for i in range(1, len(sets) + 1):
                if i == len(sets) or sets[i] != sets[start]:
                    x0b, x1b = ox + pw * start / len(v), ox + pw * i / len(v)
                    shade = "var(--band-2)" if (list(dict.fromkeys(sets)).index(sets[start]) % 2) else "var(--paper)"
                    body.append(f'<rect x="{x0b:.1f}" y="{oy}" width="{x1b - x0b:.1f}" height="{ph}" '
                                f'fill="{shade}"/>')
                    body.append(txt((x0b + x1b) / 2, oy + 10, SHORT.get(sets[start], ""), 9.5, "middle"))
                    start = i
        body.append(f'<line x1="{ox}" y1="{oy + ph}" x2="{ox + pw}" y2="{oy + ph}" stroke="var(--ink)" stroke-width="1"/>')
        body.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy + ph}" stroke="var(--ink)" stroke-width="1"/>')
        step = 5 if hi - lo <= 30 else 10
        y = lo
        while y <= hi + 1e-9:
            body.append(f'<line x1="{ox}" y1="{fy(y):.1f}" x2="{ox + pw}" y2="{fy(y):.1f}" stroke="var(--grid)" stroke-width="0.5"/>')
            body.append(txt(ox - 5, fy(y) + 4, f"{y:+.0f}%".replace("+0%", "0%"), 10, "end"))
            y += step
        body.append(f'<line x1="{ox}" y1="{fy(0):.1f}" x2="{ox + pw}" y2="{fy(0):.1f}" stroke="var(--ink-3)" stroke-width="1" stroke-dasharray="3 2"/>')
        for i in range(0, len(v), 20):
            body.append(txt(fx(i), oy + ph + 14, i + 1, 10, "middle"))
        bw = max(1.0, pw / len(v) * 0.7)
        for i, r in enumerate(red):
            y0, y1 = sorted((fy(0), fy(r)))
            body.append(f'<rect x="{fx(i) - bw / 2:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(0.5, y1 - y0):.1f}" '
                        f'fill="{"var(--ink)" if r >= 0 else "var(--ink-3)"}"/>')
        wins = sum(1 for row in v[1:] if row[0] < row[1])
        notes.append(f"{wins} of {len(v) - 1}")
        # A change of source: where the first request of the new source ranks among that source's
        # other requests by the same reduction (0 lowest, 1 highest). A model that a change of
        # workload sets back would rank its first request low.
        if len(sets) == len(v):
            for s in list(dict.fromkeys(sets))[1:]:
                idx = [i for i, x in enumerate(sets) if x == s]
                rest = [red[i] for i in idx[1:]]
                if rest:
                    switch_ranks.append(sum(1 for x in rest if x < red[idx[0]]) / len(rest))
    body.append(txt(W / 2, H - 8, "request, in serving order", 12.5, "middle", "var(--ink-2)"))
    body.append(txt(16, H / 2, "log loss reduction, fitted vs head-based", 12.5, "middle", "var(--ink-2)",
                    f' transform="rotate(-90 16 {H / 2})"'))
    cap = ('<span class="lbl" id="fig-accept_curve">Figure 0:</span> The acceptance model while serving the evaluation '
           'subset from an empty store (looped replies left out): for each request, how much lower the fitted model\'s log loss per labelled node '
           'is than that of the head-based estimate it competes with (1 &minus; fitted / head-based), both scored before '
           'either learns from the request\'s labels. A bar above the dashed zero line is a request the fitted model predicted better. '
           'Shaded bands mark the six sources in serving order (SB Spec-Bench, SP SPEED-Bench, WC WildChat, '
           'CF ConvFinQA, TA ToolACE, LB LongBench). The fitted model is lower on ' + ", ".join(notes) + " requests after the first, in the order of the panels." +
           (f' At the {len(switch_ranks)} changes of source (five on each target), the first request of the new source ranks '
            f'{sum(switch_ranks) / len(switch_ranks):.2f} on average among that source\'s other requests '
            f'by this reduction (0 lowest, 1 highest) and below their median at '
            f'{sum(1 for x in switch_ranks if x < 0.5)} of them.' if switch_ranks else ""))
    return figure(svg(W, H, body, "Reduction in acceptance log loss per request, fitted model against head-based estimate"), cap)


def cold_start(log):
    """Cold-start rounds of a probed run: how many, at which widths, and their share of the first
    request's round time. A probe round is one whose `dspark cost probe=` line names a width; the
    first request ends at the first acceptance verdict line."""
    lines = open(log, errors="replace").read().splitlines()
    end = next((i for i, l in enumerate(lines) if "dspark accept model_ll=" in l), len(lines))
    probe_next, probes, widths, t_probe, t_first = False, 0, [], 0.0, 0.0
    for i, l in enumerate(lines):
        m = re.match(r"dspark cost probe=(\d+|-)", l)
        if m:
            probe_next = m.group(1) != "-"
            if probe_next:
                probes += 1
                widths.append(int(m.group(1)))
            continue
        r = ROUND_FULL.search(l)
        if r:
            us = sum(int(r.group(k)) for k in (4, 5, 6, 7))
            if i < end:
                t_first += us
                if probe_next:
                    t_probe += us
            probe_next = False
    return {"probes": probes, "widths": sorted(set(widths)), "probe_ms": t_probe / 1e3,
            "first_ms": t_first / 1e3}


RIDE = re.compile(r"^dspark ride kept=(\d+) drafter_only=(\d+)")


def rides(log):
    """Rounds with n-gram candidates: (same width as without them, wider, narrower)."""
    same = wider = narrower = 0
    for line in open(log, errors="replace"):
        m = RIDE.search(line)
        if m:
            k, d = int(m.group(1)), int(m.group(2))
            same += k == d
            wider += k > d
            narrower += k < d
    return same, wider, narrower


def tab_ngram(figrows, figlogs):
    """What n-gram fill contributes, from the probed AdaSpark run, one column per target: rows
    placed and draft tokens accepted by source (drafter only, both, n-gram only), each source's
    acceptance rate, and the rounds in which n-gram candidates moved the chooser to a wider class."""
    cols = {}
    for m, name in MODELS:
        rs = [r for r in figrows.get(m, []) if r["arm"] == "complete" and r.get("draft_sources")]
        if not rs or m not in figlogs:
            continue
        tot = {k: {"verified": 0, "accepted": 0} for k in ("drafter", "agreed", "ngram")}
        for r in rs:
            for k in tot:
                for f in ("verified", "accepted"):
                    tot[k][f] += r["draft_sources"][k][f]
        ver = sum(v["verified"] for v in tot.values())
        acc = sum(v["accepted"] for v in tot.values())
        same, wider, narrower = rides(figlogs[m])
        n = same + wider + narrower
        if not (ver and acc and n):
            continue
        rate = lambda k: (f"{tot[k]['accepted'] / tot[k]['verified'] * 100:.0f}%"
                          if tot[k]["verified"] else "&ndash;")
        cols[name] = [f"{tot['ngram']['verified'] / ver * 100:.1f}%",
                      f"{tot['ngram']['accepted'] / acc * 100:.1f}%",
                      f"{tot['agreed']['accepted'] / acc * 100:.1f}%",
                      rate("drafter"), rate("agreed"), rate("ngram"),
                      f"{n:,}", f"{wider / n * 100:.1f}%"]
    if not cols:
        return None
    labels = ["verified rows held by n-gram-only nodes",
              "accepted draft tokens from n-gram-only nodes",
              "accepted draft tokens from nodes both sources proposed",
              "acceptance rate, drafter-only nodes",
              "acceptance rate, nodes both proposed",
              "acceptance rate, n-gram-only nodes",
              "rounds with n-gram candidates",
              "of those, rounds the n-gram widened the verify"]
    names = list(cols)
    body = ["<tr><td>" + lab + "</td>" + "".join(f"<td>{cols[nm][k]}</td>" for nm in names) + "</tr>"
            for k, lab in enumerate(labels)]
    return ('<figure class="tb">\n  <figcaption><span class="lbl" id="tab-ngram">Table 0:</span> What n-gram fill '
            'contributes, from the instrumented AdaSpark run on the short subset. Rows are the verified nodes past each '
            'round\'s anchor; accepted tokens are the draft tokens committed. A round is widened when its n-gram '
            'candidates move the chooser to a wider class than it takes with them priced at zero.</figcaption>\n'
            '  <div class="scroll"><table class="compact">\n    <thead><tr><th></th>' +
            "".join(f"<th>{nm}</th>" for nm in names) + '</tr></thead>\n    <tbody>\n      ' +
            "\n      ".join(body) + '\n    </tbody>\n  </table></div>\n</figure>')


ROUND_FULL = re.compile(r"dspark round start=(\d+) rows=(\d+) consumed=(\d+) draft_us=(\d+) "
                        r"tree_us=(\d+) verify_us=(\d+) commit_us=(\d+)")


def round_split(log):
    """Median draft / tree / verify / commit ms and mean tokens per round, at the width the
    served rounds most often ran. Cold-start probe rounds (a `dspark cost probe=N` line names
    their width) are left out of the width share and the medians, as in the width-share figure;
    the all-rounds tokens and milliseconds per token include them, since they produced tokens."""
    import collections
    by = collections.defaultdict(list)
    every = []
    probe_next = False
    for line in open(log, errors="replace"):
        m = re.match(r"dspark cost probe=(\d+|-)", line)
        if m:
            probe_next = m.group(1) != "-"
            continue
        m = ROUND_FULL.search(line)
        if m:
            x = tuple(int(v) for v in m.groups())
            every.append(x)
            if not probe_next:
                by[x[1]].append(x)
            probe_next = False
    if not by:
        return None
    rows = max(by, key=lambda r: len(by[r]))
    v = by[rows]
    med = lambda k: statistics.median(x[k] for x in v) / 1000
    served = sum(len(xs) for xs in by.values())
    us_all = sum(x[3] + x[4] + x[5] + x[6] for x in every)
    tok_all = sum(x[2] for x in every)
    return {"rows": rows, "n": len(v), "share": len(v) / served,
            "draft": med(3), "tree": med(4), "verify": med(5), "commit": med(6),
            "tokens": sum(x[2] for x in v) / len(v),
            "tokens_all": tok_all / len(every), "ms_per_token_all": us_all / 1000 / tok_all}


def fig_round(logs, ar_ms):
    """Anatomy of a round per target: stacked draft / tree / verify / commit at the most common
    width, with tokens per round, ms per token and the same engine's autoregressive ms per token."""
    data = [(n, round_split(logs[m]), ar_ms.get(m)) for m, n in MODELS if m in logs]
    data = [(n, d, a) for n, d, a in data if d]
    if not data:
        return None
    W, bar, gap, x0, x1 = 640, 18, 30, 118, 330
    H = 40 + len(data) * (bar + gap) + 36
    top = max(d["draft"] + d["tree"] + d["verify"] + d["commit"] for _, d, _ in data) * 1.05
    fx = lambda v: x0 + (x1 - x0) * v / top
    parts = [("draft", "var(--ink-3)"), ("tree", "var(--grid)"), ("verify", "var(--ink)"),
             ("commit", "var(--band)")]
    body = []
    for k, (lab, fill) in enumerate(parts):
        lx = x0 + k * 80
        body.append(f'<rect x="{lx}" y="6" width="12" height="9" fill="{fill}" stroke="var(--ink)" stroke-width="0.6"/>')
        body.append(txt(lx + 16, 14, lab, 11, fill="var(--ink-2)"))
    step = 10 if top <= 80 else 20
    v = 0
    while v <= top:
        body.append(f'<line x1="{fx(v):.1f}" y1="30" x2="{fx(v):.1f}" y2="{H - 30}" stroke="var(--grid)" stroke-width="0.5"/>')
        body.append(txt(fx(v), H - 16, v, 10.5, "middle"))
        v += step
    y = 36
    notes = []
    for name, d, ar in data:
        body.append(txt(x0 - 10, y + bar - 5, name, 12, "end", "var(--ink-2)"))
        at = 0.0
        for lab, fill in parts:
            w = d[lab]
            body.append(f'<rect x="{fx(at):.1f}" y="{y}" width="{fx(at + w) - fx(at):.1f}" height="{bar}" '
                        f'fill="{fill}" stroke="var(--ink)" stroke-width="0.5"/>')
            at += w
        line2 = (f"all rounds: {d['tokens_all']:.1f} tok/round, "
                 f"{d['ms_per_token_all']:.1f} ms/tok")
        if ar:
            line2 += f" (AR {ar:.1f})"
        body.append(txt(fx(at) + 6, y + 8, f"{d['rows']} rows: {d['tokens']:.1f} tok/round", 10.5,
                        fill="var(--ink-2)"))
        body.append(txt(fx(at) + 6, y + 21, line2, 10.5, fill="var(--ink-3)"))
        notes.append(f"{name}: draft {d['draft']:.1f}, verify {d['verify']:.1f} ms at {d['rows']} rows "
                     f"({d['share'] * 100:.0f}% of rounds)")
        y += bar + gap
    body.append(txt((x0 + x1) / 2, H - 2, "milliseconds per round", 12, "middle", "var(--ink-2)"))
    cap = ('<span class="lbl" id="fig-round">Figure 0:</span> Anatomy of an AdaSpark round on each '
           'target, at the width its rounds most often ran: the drafter\'s forward, the tree build, '
           'the verify and the commit (medians), and the tokens committed per round at that width. The '
           'text beside each bar also gives the tokens per round and milliseconds per token over all rounds, '
           'wider ones included, beside the same engine\'s autoregressive (AR) milliseconds per token on the short subset. ' +
           "; ".join(notes) + ".")
    return figure(svg(W, H, body, "Anatomy of a round per target"), cap)


# ---------------------------------------------------------------- probe-run figures
def fig_from_figs(kind, logs):
    """Wrap figs.py's fragments: kind in widths / learning / calibration."""
    if kind == "widths":
        targets = {name: [x for x in figs.rounds(p) if not x["probe"]] for (m, name), p in logs}
        body, note = figs.fig_widths(targets)
        h = 20 + 34 * len(targets) + 10
        cap = ('<span class="lbl" id="fig-widths">Figure 0:</span> Share of rounds at each verified width, per '
               'target, from AdaSpark\'s rounds on the instrumented run over the short subset (setup notes), cold-start rounds excluded. ' + note + '.')
        return figure(svg(620, h, body, "Share of rounds at each width per target"), cap)
    if kind == "learning":
        (m, name), p = logs[0]
        body, note = figs.fig_learning(figs.rounds(p))
        cap = (f'<span class="lbl" id="fig-learning">Figure 0:</span> {name}: measured verify time of each round '
               '(dots; hollow dots are cold-start rounds at a width chosen for measurement) and the '
               'cost model\'s estimate for the width it chose (line), from an empty store, over '
               'the first 160 rounds. ' + note + '.')
        return figure(svg(620, 260, body, "Cost model estimate against measured verify time"), cap)
    if kind == "calibration":
        series = {lab: figs.rounds(p) for lab, p in logs}
        body, note = figs.fig_calibration(series)
        cap = ('<span class="lbl" id="fig-calibration">Figure 0:</span> Accepted nodes per round against the expected '
               'accepted nodes <i>S</i> at the chosen width, in ten equal-count bins of <i>S</i> per series, on the instrumented run over the short subset, over the rounds whose width the chooser set from <i>S</i> (cold-start rounds have none). The dashed '
               'diagonal is perfect calibration. Realised over expected accepted nodes: ' + note + '. The drafter-head series is the cost-model arm, which prices the tree with the head-based estimate.')
        return figure(svg(620, 260, body, "Acceptance calibration"), cap)


def number(paper):
    """Figures and tables numbered by order of appearance; every `xref` link takes its target's
    number. A caption is `<span class="lbl" id="fig-NAME">Figure N:</span>` (or tab-); a reference
    is `<a class="xref" href="#fig-NAME">Figure N</a>`."""
    nums, count = {}, {"fig": 0, "tab": 0}
    def lbl(m):
        kind = m.group(1)
        count[kind] += 1
        nums[f"{kind}-{m.group(2)}"] = count[kind]
        word = "Figure" if kind == "fig" else "Table"
        return f'<span class="lbl" id="{kind}-{m.group(2)}">{word} {count[kind]}:</span>'
    paper = re.sub(r'<span class="lbl" id="(fig|tab)-([\w-]+)">(?:Figure|Table) \d+:</span>', lbl, paper)
    missing = set()
    def xref(m):
        key = m.group(1)
        if key not in nums:
            missing.add(key)
            return m.group(0)
        word = "Figure" if key.startswith("fig") else "Table"
        return f'<a class="xref" href="#{key}">{word}&nbsp;{nums[key]}</a>'
    paper = re.sub(r'<a class="xref" href="#((?:fig|tab)-[\w-]+)">[^<]*</a>', xref, paper)
    if missing:
        print("  references to missing labels: " + ", ".join(sorted(missing)))
    return paper


# ---------------------------------------------------------------- learned width classes
def store_widths(home):
    """The widths a stored verify-cost table offers: the top row count of each `step` line."""
    import glob
    for f in glob.glob(os.path.join(home, ".imparo", "verify-*.txt")):
        tops = [int(l.split("=")[1].split(",")[1]) for l in open(f) if l.startswith("step=")]
        tops = sorted(set(t for t in tops if t > 0))  # the step at 0 rows is the round's fixed cost
        if tops:
            return tops
    return []


def spans(ws):
    """[2, 3, 4, 8, 9, 16] -> '2&ndash;4, 8, 9, 16': runs of three or more as a range."""
    out, i = [], 0
    while i < len(ws):
        j = i
        while j + 1 < len(ws) and ws[j + 1] == ws[j] + 1:
            j += 1
        out += [f"{ws[i]}&ndash;{ws[j]}"] if j - i >= 2 else [str(w) for w in ws[i:j + 1]]
        i = j + 1
    return ", ".join(out)


def tab_classes_agentic(stages, run):
    """Learned widths against the 8-bit matrix kernel's row classes (harness/run_classes.sh), on the
    evaluation subset: per target and context band, AdaSpark on the kernel classes over AdaSpark
    with learned widths, the learned arm's speed per reply the mean of its two runs (before and
    after the classes arm); and the widths each arm's table offered at the end of its run."""
    rows = []
    for m, name in MODELS:
        if m not in stages:
            continue
        rs, rep = stages[m]
        ctx = {(r["set"], r["conv"], r["turn"], r["type"]): r["prompt_tokens"] + (r["completion_tokens"] or 0) / 2
               for r in rs if r["arm"] == "complete"}
        merged = rs + [r for r in rep if r["arm"] == "complete"]
        cells = []
        for keep in (None, lambda u: ctx.get(u, 0) < BAND_EDGE, lambda u: ctx.get(u, 0) >= BAND_EDGE):
            cells.append(ci(ratios(merged, "complete", "complete-classes", keep=keep), 3))
        learned = store_widths(os.path.join(run, f"{m}_classes", "home_complete_r1"))
        kernel = store_widths(os.path.join(run, f"{m}_classes", "home_complete-classes_r1"))
        rows.append(f"<tr><td>{name}</td>" + "".join(f"<td>{c}</td>" for c in cells) +
                    f"<td class=\"txt\">{spans(learned)}</td>"
                    f"<td class=\"txt\">{spans(kernel)}</td></tr>")
    if not rows:
        return None
    return ('<figure class="tb">\n  <figcaption><span class="lbl" id="tab-classes">Table 0:</span> Learned against '
            'kernel width classes on the evaluation subset: AdaSpark\'s decode speed with learned widths (&sect;3.2) over '
            'its speed with the widths taken from the 8-bit matrix kernel\'s row classes, geometric mean of per-reply '
            'ratios with its 95% bootstrap interval over conversations, for all replies and by the context a reply was '
            'decoded at; above 1 the learned widths are faster. Both arms start from an empty store and ran on one build; '
            'the learned arm runs before the kernel-class arm and again after it, and its speed on a reply is the mean of '
            'the two runs. The last columns give the widths each arm\'s table offered at the end of its run (setup notes).'
            '</figcaption>\n  <div class="scroll"><table class="compact">\n    <thead><tr><th>target</th>'
            f'<th>all replies</th><th>under {BAND_EDGE:,}</th><th>{BAND_EDGE:,} and over</th>'
            '<th>learned widths</th><th>kernel-class widths</th></tr></thead>\n'
            '    <tbody>\n      ' + "\n      ".join(rows) + '\n    </tbody>\n  </table></div>\n</figure>')


# ---------------------------------------------------------------- looped replies
# A reply that reached the token cap is a repetition loop under greedy decoding: trivially
# predictable text that flatters every speculative method. It is left out of every speed ratio
# (`ratios`), and the per-round statistics must leave it out too, or one loop decides a column.
CAP = 8192
_UNLOOPED = {}


def unlooped(log):
    """A copy of a server log without the requests that reached the cap. Each request's lines end
    with its `[imparo] conv=... gen=N` summary line, so the log is cut there; a request with
    gen >= CAP is dropped whole. Returns the copy's path (cached per log)."""
    import tempfile
    if log in _UNLOOPED:
        return _UNLOOPED[log]
    keep, seg = [], []
    for line in open(log, errors="replace"):
        seg.append(line)
        m = re.search(r"\[imparo\] conv=\S+ prompt=\d+ reused=\d+ gen=(\d+)", line)
        if m:
            if int(m.group(1)) < CAP:
                keep += seg
            seg = []
    keep += seg
    fd, path = tempfile.mkstemp(suffix=".log", prefix="unlooped-")
    with os.fdopen(fd, "w") as f:
        f.writelines(keep)
    _UNLOOPED[log] = path
    return path


def put(paper, kind, name, frag):
    a, b = f"<!--{kind}:{name}-->", f"<!--/{kind}:{name}-->"
    i, j = paper.find(a), paper.find(b)
    if i < 0 or j < 0:
        print(f"  {kind}:{name}: markers missing in the paper")
        return paper
    return paper[: i + len(a)] + "\n" + frag + "\n" + paper[j:]


def main():
    run = sys.argv[1]
    paper_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(HERE),
                                                                    "paper.html")
    paper = open(paper_path).read()
    p = lambda *xs: os.path.join(run, *xs)
    sv = {m: survey(p(f"{m}_survey", "survey.tsv")) for m, _ in MODELS
          if os.path.exists(p(f"{m}_survey", "survey.tsv"))}
    wd = {m: widths(p(f"{m}_widths", "table4.tsv")) for m, _ in MODELS
          if os.path.exists(p(f"{m}_widths", "table4.tsv"))}
    import json
    runs = {m: [json.loads(l) for l in open(p(f"{m}_full", "results.jsonl"))] for m, _ in MODELS
            if os.path.exists(p(f"{m}_full", "results.jsonl"))}
    # Stages run after the campaign on the build that routes chains through the tree verify:
    # {m}_chain replaces the chain arms.
    for m in list(runs):
        if os.path.exists(p(f"{m}_chain", "results.jsonl")):
            new = [json.loads(l) for l in open(p(f"{m}_chain", "results.jsonl"))]
            arms = {r["arm"] for r in new}
            runs[m] = [r for r in runs[m] if r["arm"] not in arms] + new
        # run_learned.sh: every arm that runs the cost model, on the build whose cost model learns
        # its widths, replaces the same arm measured before
        if os.path.exists(p(f"{m}_learned", "results.jsonl")):
            new = [json.loads(l) for l in open(p(f"{m}_learned", "results.jsonl"))]
            new = [r for r in new if pinned_width(r["arm"]) is None]  # the pinned arms are Table 3's
            # AdaSpark ran twice on this subset, before the pinned arms and after them: its speed on
            # a reply is the mean of the two runs, one row per reply as for every other arm.
            if os.path.exists(p(f"{m}_learned_rep", "results.jsonl")):
                unit = lambda r: (r["set"], r["conv"], r["turn"], r["type"])
                again = {unit(r): r for r in map(json.loads, open(p(f"{m}_learned_rep", "results.jsonl")))
                         if r["arm"] == "complete"}
                for i, r in enumerate(new):
                    o = again.get(unit(r))
                    if r["arm"] == "complete" and o and r["finished"] and o["finished"] and speed(r) and speed(o):
                        new[i] = dict(r, decode_tok_s=(speed(r) + speed(o)) / 2, runs=2)
            arms = {r["arm"] for r in new}
            runs[m] = [r for r in runs[m] if r["arm"] not in arms] + new
    # THE WIDTH TABLE reads each stage that ran AdaSpark beside pinned widths, with AdaSpark's repeat
    # after them: the main stage on every target, and the narrow widths on the routed one.
    load = lambda st: [json.loads(l) for l in open(p(st, "results.jsonl"))] if os.path.exists(p(st, "results.jsonl")) else []
    # TABLE 10 reads the measurement build's learned and kernel-class arms (run_classes.sh).
    cstages = {m: (load(f"{m}_classes"), load(f"{m}_classes_rep")) for m, _ in MODELS
               if any(r["arm"] == "complete-classes" for r in load(f"{m}_classes"))}
    wstages = {}
    for m, _ in MODELS:
        for tag in (f"{m}_learned", f"{m}_learned_narrow"):
            rs = load(tag)
            if any(pinned_width(r["arm"]) for r in rs):
                wstages.setdefault(m, []).append((rs, load(f"{tag}_rep")))
    # run_toolace.sh: ToolACE with JSON Schema parameter types. Its rows replace every arm's rows
    # of the conversations whose schemas the rename changed; ta-1 and ta-3 keep the rows of the run
    # they belong to and are the control for the learned state (`toolace_control`).
    toolace = {m: [json.loads(l) for l in open(p(f"{m}_toolace", "results.jsonl"))] for m, _ in MODELS
               if os.path.exists(p(f"{m}_toolace", "results.jsonl"))}

    def relearned(rows, stage):
        """A run_learned.sh stage's arms replace the same arms of an earlier run: the scheduler
        changed, the llama.cpp and plain arms did not."""
        if not os.path.exists(p(stage, "results.jsonl")):
            return rows
        new = [json.loads(l) for l in open(p(stage, "results.jsonl"))]
        arms = {r["arm"] for r in new}
        return [r for r in rows if r["arm"] not in arms] + new

    toolace = {m: relearned(v, f"{m}_toolace_learned") for m, v in toolace.items()}

    def fixed(rows, m, convs):
        if m not in toolace:
            return rows
        arms = {r["arm"] for r in rows}
        return ([r for r in rows if not (r["set"] == "toolace" and r["conv"] in convs)] +
                [r for r in toolace[m] if r["conv"] in convs and r["arm"] in arms])

    for m in list(runs):
        runs[m] = fixed(runs[m], m, TOOLACE_RENAMED_CORE)
    heldout = {m: fixed(relearned([json.loads(l) for l in open(p(f"{m}_heldout", "results.jsonl"))],
                                  f"{m}_heldout_learned"), m,
                        TOOLACE_RENAMED_REST) for m, _ in MODELS
               if os.path.exists(p(f"{m}_heldout", "results.jsonl"))}
    toolace_control(runs, toolace)
    smoke = {m: [json.loads(l) for l in open(p(f"{m}_smoke", "results.jsonl"))] for m, _ in MODELS
             if os.path.exists(p(f"{m}_smoke", "results.jsonl"))}
    # the short subset serves only the autoregressive arms' speed (and the instrumented run below)
    # the instrumented run and the served log of AdaSpark: the learned-width stage where it exists
    figs_of = lambda m: (f"{m}_figs_learned" if os.path.exists(p(f"{m}_figs_learned", "results.jsonl"))
                         else f"{m}_figs")
    full_of = lambda m: (f"{m}_learned" if os.path.exists(p(f"{m}_learned", "server_complete_r1.log"))
                         else f"{m}_full")
    # every per-round and per-request statistic leaves the looped replies out (`unlooped`)
    figrows = {m: [r for r in (json.loads(l) for l in open(p(figs_of(m), "results.jsonl")))
                   if r["finished"]] for m, _ in MODELS
               if os.path.exists(p(figs_of(m), "results.jsonl"))}
    figlog = {m: unlooped(p(figs_of(m), "server_complete_r1.log")) for m, _ in MODELS
              if os.path.exists(p(figs_of(m), "server_complete_r1.log"))}
    fulllog = {m: unlooped(p(full_of(m), "server_complete_r1.log")) for m, _ in MODELS
               if os.path.exists(p(full_of(m), "server_complete_r1.log"))}
    done = []
    for kind, name, make in [
        ("FIG", "verify_cost", lambda: fig_verify_cost(sv) if sv else None),
        ("TAB", "context_law", lambda: tab_context_law(sv) if sv else None),
        ("TAB", "widths", lambda: tab_widths_agentic(wstages) if wstages else None),
        ("TAB", "classes", lambda: tab_classes_agentic(cstages, run) if cstages else None),
        ("FIG", "marginal", lambda: fig_marginal({m: p(f"{m}_widths", "table4.tsv") for m, _ in MODELS
                                                  if os.path.exists(p(f"{m}_widths", "table4.tsv"))})
         if wd else None),
        ("TAB", "e2e", lambda: tab_e2e(runs, smoke) if runs else None),
        ("FIG", "e2e", lambda: fig_e2e(runs) if runs else None),
        ("TAB", "ladder", lambda: tab_ladder(runs) if runs else None),
        ("TAB", "sources", lambda: tab_sources(runs) if runs else None),
        ("TAB", "heldout", lambda: tab_heldout(runs, heldout) if heldout else None),
        ("TAB", "agree", lambda: tab_agree(runs) if runs else None),
        ("TAB", "accept", lambda: tab_accept(fulllog)),
        ("FIG", "accept_curve", lambda: fig_accept_curve(fulllog, runs) if runs else None),
        ("TAB", "ngram", lambda: tab_ngram(figrows, figlog) if figlog else None),
        ("FIG", "round", lambda: fig_round(figlog, {m: 1000 / throughput(smoke[m], "plain")
                                                    for m in smoke if throughput(smoke[m], "plain")})
         if figlog else None),
        ("FIG", "widths", lambda: fig_from_figs("widths", [((m, n), figlog[m]) for m, n in MODELS
                                                          if m in figlog]) if figlog else None),
        ("FIG", "learning", lambda: fig_from_figs("learning", [((m, n), figlog[m])
                                                              for m, n in MODELS if m in figlog][:1])
         if figlog else None),
        ("FIG", "calibration", lambda: fig_from_figs("calibration", [
            (f"{n}, fitted", figlog[m]) for m, n in MODELS if m in figlog] + (
            [("LFM2.5-2.6B, drafter head", unlooped(p(figs_of("lfm26"), "server_budget_r1.log")))]
            if os.path.exists(p(figs_of("lfm26"), "server_budget_r1.log")) else [])) if figlog else None),
    ]:
        frag = make()
        if frag:
            paper = put(paper, kind, name, frag)
            done.append(f"{kind}:{name}")
        else:
            print(f"  {kind}:{name}: no data yet, left as it is")
    paper = number(paper)
    open(paper_path, "w").write(paper)
    print("written: " + ", ".join(done))


if __name__ == "__main__":
    main()
