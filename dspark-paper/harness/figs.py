#!/usr/bin/env python3
"""The paper's mechanism figures, from instrumented server logs (IMPARO_DSPARK_ROUND=1,
IMPARO_DSPARK_SCORE=1, IMPARO_DSPARK_TRACE=1). Prints each figure as an SVG fragment in the
paper's style (Newsreader, the page's colour tokens) to OUTDIR, plus the numbers it drew.

usage: figs.py OUTDIR LFM_COMPLETE_LOG LFM_BUDGET_LOG TARGET=LOG [TARGET=LOG ...]
  LFM_COMPLETE_LOG  server log of the complete form on LFM2.5-2.6B (figure A, and C's fitted arm)
  LFM_BUDGET_LOG    server log of the budget arm, acceptance from the drafter's head (C)
  TARGET=LOG        each target's complete-form server log (figure B)
"""
import math, os, re, sys

ROUND = re.compile(r'dspark round start=(\d+) rows=(\d+) consumed=(\d+) draft_us=(\d+) '
                   r'tree_us=(\d+) verify_us=(\d+)')
CHOOSE = re.compile(r'dspark choose kept=(\d+) hold=(\d+) fixed_us=([\d.]+) gain=([\d.-]+) '
                    r'gain_n=([\d.-]+) (?:gain_below=[\d.-]+ gain_n_below=[\d.-]+ )?rho=(\S+) order=(\d+) '
                    r'widths=(\S+)')
PATH = re.compile(r'dspark path fixed_us=[\d.]+ rows=(\d+) path=([\d,-]+) next=\d+')
CONV = re.compile(r'\[imparo\] conv=(\S+) prompt=(\d+)')


def rounds(log):
    """Rounds in order: rows, verify ms, context, and when the chooser ran, its estimate at the
    width it chose, the expected accepted nodes there (S) and the nodes actually accepted."""
    out, choose, path_rows, req = [], None, None, 0
    for line in open(log, errors='replace'):
        m = CHOOSE.search(line)
        if m:
            widths = {}
            for w in m.group(8).split(','):
                r, us, s, sn = w.split(':')
                widths[int(r)] = (float(us), float(s), float(sn))
            choose = (int(m.group(1)), widths)
            continue
        m = PATH.search(line)
        if m:
            accepted = len([p for p in m.group(2).split(',') if p != '']) - 1
            if out and out[-1]['accepted'] is None and out[-1]['rows'] == int(m.group(1)):
                out[-1]['accepted'] = accepted
            continue
        m = ROUND.search(line)
        if m:
            rec = {'req': req, 'ctx': int(m.group(1)), 'rows': int(m.group(2)),
                   'consumed': int(m.group(3)), 'verify_ms': int(m.group(6)) / 1000,
                   'est_ms': None, 'S': None, 'accepted': None, 'probe': choose is None}
            if choose is not None:
                kept, widths = choose
                w = widths.get(kept + 1)
                if w is not None:
                    rec['est_ms'] = w[0] / 1000
                    rec['S'] = w[1]
            out.append(rec)
            choose = None
            continue
        if CONV.search(line) and 'readiness' not in line:
            req += 1
    return out


def axis(x0, y0, x1, y1, xticks, yticks, xlabel, ylabel, fx, fy):
    s = [f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="var(--ink)" stroke-width="1"/>',
         f'<line x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}" stroke="var(--ink)" stroke-width="1"/>']
    for v in yticks:
        y = fy(v)
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="var(--grid)" '
                 f'stroke-width="0.5"/><text x="{x0-6}" y="{y+4:.1f}" text-anchor="end" '
                 f'font-family="Newsreader, serif" font-size="11" fill="var(--ink-3)">{v:g}</text>')
    for v in xticks:
        x = fx(v)
        s.append(f'<text x="{x:.1f}" y="{y1+16}" text-anchor="middle" font-family="Newsreader, '
                 f'serif" font-size="11" fill="var(--ink-3)">{v:g}</text>')
    s.append(f'<text x="{(x0+x1)/2}" y="{y1+34}" text-anchor="middle" font-family="Newsreader, '
             f'serif" font-size="12.5" fill="var(--ink-2)">{xlabel}</text>')
    s.append(f'<text x="16" y="{(y0+y1)/2}" text-anchor="middle" font-family="Newsreader, serif" '
             f'font-size="12.5" fill="var(--ink-2)" transform="rotate(-90 16 {(y0+y1)/2})">'
             f'{ylabel}</text>')
    return s


def fig_learning(recs, first=160):
    """A: measured verify time per round (dots, hollow = a probe round the chooser did not pick)
    against the cost model's estimate at the width it chose (line), from a cold start."""
    r = recs[:first]
    ymax = max(x['verify_ms'] for x in r) * 1.08
    x0, y0, x1, y1 = 58, 24, 600, 222
    fx = lambda i: x0 + (x1 - x0) * i / max(1, len(r) - 1)
    fy = lambda v: y1 - (y1 - y0) * v / ymax
    step = 20 if ymax > 60 else 10
    s = axis(x0, y0, x1, y1, list(range(0, len(r), 40)), list(range(0, int(ymax) + 1, step)),
             'round (one conversation after another, from an empty store)', 'verify (ms)', fx, fy)
    for i, x in enumerate(r):
        fill = 'none' if x['probe'] else 'var(--ink-3)'
        s.append(f'<circle cx="{fx(i):.1f}" cy="{fy(x["verify_ms"]):.1f}" r="2.4" fill="{fill}" '
                 f'stroke="var(--ink-3)" stroke-width="0.8"/>')
    pts = [(fx(i), fy(x['est_ms'])) for i, x in enumerate(r) if x['est_ms'] is not None]
    if pts:  # the estimate on top of the measurements, which it tracks closely
        s.append('<path d="M ' + ' L '.join(f'{a:.1f} {b:.1f}' for a, b in pts) +
                 '" stroke="var(--ink)" stroke-width="1.4" fill="none"/>')
    lx, ly = x1 - 230, y0 + 8
    s.append(f'<circle cx="{lx}" cy="{ly}" r="2.4" fill="var(--ink-3)" stroke="var(--ink-3)"/>')
    s.append(f'<text x="{lx + 8}" y="{ly + 4}" font-family="Newsreader, serif" font-size="11" fill="var(--ink-2)">measured</text>')
    s.append(f'<circle cx="{lx + 70}" cy="{ly}" r="2.4" fill="none" stroke="var(--ink-3)"/>')
    s.append(f'<text x="{lx + 78}" y="{ly + 4}" font-family="Newsreader, serif" font-size="11" fill="var(--ink-2)">cold start</text>')
    s.append(f'<line x1="{lx + 142}" y1="{ly}" x2="{lx + 164}" y2="{ly}" stroke="var(--ink)" stroke-width="1.4"/>')
    s.append(f'<text x="{lx + 169}" y="{ly + 4}" font-family="Newsreader, serif" font-size="11" fill="var(--ink-2)">estimate</text>')
    errs = [abs(x['verify_ms'] - x['est_ms']) / x['verify_ms'] for x in recs
            if x['est_ms'] is not None]
    late = errs[len(errs) // 2:]
    note = (f'Over all {len(recs):,} rounds, {sum(x["probe"] for x in recs)} of them cold-start rounds, '
            f'the median absolute error of the estimate over the second half of the run is '
            f'{sorted(late)[len(late)//2]*100:.1f}%' if late else '')
    return s, note


def fig_widths(targets):
    """B: the share of rounds at each width, per target: each target's cost and acceptance lead
    the chooser to its own width."""
    widths = sorted({x['rows'] for recs in targets.values() for x in recs})
    x0, x1, y0 = 170, 600, 20
    s, notes = [], []
    for k, (name, recs) in enumerate(targets.items()):
        y = y0 + k * 34
        s.append(f'<text x="{x0-8}" y="{y+15}" text-anchor="end" font-family="Newsreader, serif" '
                 f'font-size="12.5" fill="var(--ink-2)">{name}</text>')
        n = len(recs)
        at = x0
        shares = []
        for w in widths:
            c = sum(1 for x in recs if x['rows'] == w)
            if not c:
                continue
            wd = (x1 - x0) * c / n
            shade = 'var(--ink)' if c / n > 0.4 else ('var(--ink-3)' if c / n > 0.1 else 'var(--grid)')
            s.append(f'<rect x="{at:.1f}" y="{y}" width="{wd:.1f}" height="20" fill="{shade}" '
                     f'stroke="var(--paper)" stroke-width="1"/>')
            if wd > 26:
                txt = 'var(--paper)' if c / n > 0.1 else 'var(--ink)'
                s.append(f'<text x="{at + wd/2:.1f}" y="{y+14}" text-anchor="middle" '
                         f'font-family="IBM Plex Mono, monospace" font-size="10.5" fill="{txt}">'
                         f'{w}</text>')
            if c / n >= 0.01:
                shares.append(f'{w} rows {c/n*100:.0f}%')
            at += wd
        notes.append(f'{name}: ' + ', '.join(shares) + f', other widths under 1% ({n:,} rounds)')
    return s, '; '.join(notes)


def fig_calibration(series):
    """C: expected accepted nodes at the chosen width (binned) against the nodes accepted."""
    x0, y0, x1, y1 = 58, 24, 600, 222
    BINS = 10  # equal-count bins: each point is a tenth of the target's rounds
    lines, allpts = {}, {}
    for name, recs in series.items():
        pts = sorted((x['S'], x['accepted']) for x in recs
                     if x['S'] is not None and x['accepted'] is not None)
        allpts[name] = pts
        n = len(pts)
        groups = [pts[n * b // BINS:n * (b + 1) // BINS] for b in range(BINS)] if n >= BINS else []
        lines[name] = [(sum(p for p, _ in v) / len(v), sum(a for _, a in v) / len(v), len(v))
                       for v in groups if v]
    top = max([8] + [math.ceil(max(p, a)) for ln in lines.values() for p, a, _ in ln])
    top += top % 2
    fx = lambda v: x0 + (x1 - x0) * v / top
    fy = lambda v: y1 - (y1 - y0) * v / top
    s = axis(x0, y0, x1, y1, list(range(0, top + 1, 2)), list(range(0, top + 1, 2)),
             'expected accepted nodes S at the chosen width', 'accepted nodes', fx, fy)
    s.append(f'<line x1="{fx(0)}" y1="{fy(0)}" x2="{fx(top)}" y2="{fy(top)}" '
             f'stroke="var(--rule-2)" stroke-width="1" stroke-dasharray="3 3"/>')
    notes = []
    styles = [('', 'var(--ink)', 'circle', 'var(--ink)'),
              (' stroke-dasharray="6 3"', 'var(--ink-2)', 'square', 'var(--ink-2)'),
              (' stroke-dasharray="2 2"', 'var(--ink-2)', 'triangle', 'var(--ink-2)'),
              (' stroke-dasharray="6 2 2 2"', 'var(--ink-3)', 'circle', 'none'),
              (' stroke-dasharray="1 3"', 'var(--ink-3)', 'square', 'none')]
    def mark(kind, x, y, colour, fill):
        f = colour if fill != 'none' else 'var(--paper)'
        if kind == 'square':
            return f'<rect x="{x-2.6:.1f}" y="{y-2.6:.1f}" width="5.2" height="5.2" fill="{f}" stroke="{colour}" stroke-width="1"/>'
        if kind == 'triangle':
            return f'<path d="M {x:.1f} {y-3.2:.1f} L {x+3:.1f} {y+2.4:.1f} L {x-3:.1f} {y+2.4:.1f} Z" fill="{f}" stroke="{colour}" stroke-width="1"/>'
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" fill="{f}" stroke="{colour}" stroke-width="1"/>'
    for k, (name, recs) in enumerate(series.items()):
        pts, line = allpts[name], lines[name]
        dash, colour, kind, fill = styles[k % len(styles)]
        if line:
            s.append('<path d="M ' + ' L '.join(f'{fx(p):.1f} {fy(a):.1f}' for p, a, _ in line) +
                     f'" stroke="{colour}" stroke-width="1.5" fill="none"{dash}/>')
            for p, a, _ in line:
                s.append(mark(kind, fx(p), fy(a), colour, fill))
        tot_p = sum(p for p, _ in pts); tot_a = sum(a for _, a in pts)
        ly = y0 + 12 + 17 * k
        s.append(f'<line x1="{x0+14}" y1="{ly}" x2="{x0+40}" y2="{ly}" stroke="{colour}" stroke-width="1.5"{dash}/>')
        s.append(mark(kind, x0 + 27, ly, colour, fill))
        s.append(f'<text x="{x0+46}" y="{ly+4}" font-family="Newsreader, serif" '
                 f'font-size="11.5" fill="var(--ink-2)">{name}: realised / expected = '
                 f'{tot_a/tot_p:.2f}</text>' if tot_p else '')
        notes.append(f'{name} {tot_a/tot_p:.3f} ({len(pts):,} rounds)' if tot_p else name)
    return s, '; '.join(notes)


def main():
    out, lfm_c, lfm_b = sys.argv[1], sys.argv[2], sys.argv[3]
    targets = dict(a.split('=', 1) for a in sys.argv[4:])
    os.makedirs(out, exist_ok=True)
    rc = rounds(lfm_c)
    figs = {
        'learning': fig_learning(rc),
        'widths': fig_widths({k: [x for x in rounds(v) if not x['probe']] for k, v in targets.items()}),
        'calibration': fig_calibration({'fitted model': rc, "drafter's head": rounds(lfm_b)}),
    }
    for name, (svg, note) in figs.items():
        with open(os.path.join(out, f'{name}.svg'), 'w') as f:
            f.write('\n'.join(svg) + '\n')
        print(f'FIG {name}: {note}')


if __name__ == '__main__':
    main()
