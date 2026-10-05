"""HTML fragments -> LaTeX: the tables and figure captions paper_figs.py builds (via
latex_assets.py), and the inline text rules they need.

Inline rules: a variable is <i>x</i> (one letter, maybe a digit) or a Greek entity; <sub>/<sup>
attach to it; operators and brackets between two math pieces join them into one $...$. Citations
are [n, m] with whole numbers; cross-references are <a class="xref" href="#tab-..."> links and the
plain words Table/Figure N, Eq. (n), Proposition/Corollary n, Algorithm 1, Appendix X and §n.n.
"""
import html
import re
from html.parser import HTMLParser

MATH_ENT = {
    "times": r"\times", "minus": "-", "middot": r"\cdot", "le": r"\leq", "ge": r"\geq",
    "isin": r"\in", "asymp": r"\approx", "plusmn": r"\pm", "cup": r"\cup", "prod": r"\prod",
    "Sigma": r"\Sigma", "Delta": r"\Delta", "larr": r"\leftarrow", "rarr": r"\rightarrow",
    "lt": "<", "gt": ">", "#8467": r"\ell", "#8868": r"\top", "#8214": r"\|", "micro": r"\mu",
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma", "kappa": r"\kappa",
    "lambda": r"\lambda", "rho": r"\rho", "sigma": r"\sigma", "tau": r"\tau", "theta": r"\theta",
    "phi": r"\phi", "psi": r"\psi", "pi": r"\pi", "epsilon": r"\epsilon",
}
TEXT_ENT = {
    "nbsp": "~", "ensp": r"\enspace{}", "emsp": r"\quad{}", "ndash": "--", "mdash": "---",
    "sect": r"\S{}", "#8209": "-", "ldquo": "``", "rdquo": "''", "rsquo": "'", "lsquo": "`",
    "amp": r"\&", "hellip": r"\ldots{}", "quot": '"',
}
RADIC = "\x00RADIC"


def esc(t):
    """Escape plain text for LaTeX (the placeholders added by pre() survive untouched)."""
    out = []
    for ch in t:
        out.append({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}", "|": r"\textbar{}"}.get(ch, ch))
    return "".join(out)


class Node:
    def __init__(self, tag, attrs):
        self.tag, self.attrs, self.kids = tag, dict(attrs), []


class Tree(HTMLParser):
    VOID = {"br", "col", "img", "xref", "cite", "eqref", "secref", "appref", "propref", "algref"}

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.root = Node("root", []); self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs); self.stack[-1].kids.append(n)
        if tag not in self.VOID:
            self.stack.append(n)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].kids.append(Node(tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]; return

    def handle_data(self, d):
        self.stack[-1].kids.append(d)

    def handle_entityref(self, name):
        self.stack[-1].kids.append(("ent", name))

    def handle_charref(self, name):
        self.stack[-1].kids.append(("ent", "#" + name))


class Refs:
    """Number -> label maps for plain "Table N"/"Figure N" text ({"3": "tab:widths"}), and the
    citation keys by reference number."""

    def __init__(self, fig, tab, cite_keys):
        self.fig, self.tab, self.cite = fig, tab, cite_keys


def lab(html_id):
    kind, name = html_id.split("-", 1)
    return f"{kind}:{name.replace('_', '-')}"


def pre(frag, refs):
    """Replace cross-references and citations by placeholder tags before parsing."""
    f = frag
    f = re.sub(r'<a class="xref" href="#((?:tab|fig)-[\w-]+)">(Table|Figure)&nbsp;\d+</a>',
               lambda m: f'<xref k="{lab(m.group(1))}" w="{m.group(2)}"/>', f)
    f = re.sub(r'\b(Tables?|Figures?)(?:&nbsp;| )(\d+)(?: (to|and) (\d+))?\b(?![^<]*</a>)',
               lambda m: plain_ref(m, refs), f)
    f = re.sub(r'\bEq\.(?:&nbsp;| )\((\d)\)', r'<eqref n="\1"/>', f)
    f = re.sub(r'\b(Proposition|Corollary)(?:&nbsp;| )(\d)\b(?!\.<)', r'<propref w="\1" n="\2"/>', f)
    f = re.sub(r'\bAlgorithm(?:&nbsp;| )1\b(?!:)', '<algref/>', f)
    f = re.sub(r'\bAppendix(?:&nbsp;| )([ABC])\b(?!&ensp;)', r'<appref n="\1"/>', f)
    f = re.sub(r'&sect;(\d(?:\.\d)?)', r'<secref n="\1"/>', f)
    f = re.sub(r'\[(\d+(?:,\s*\d+)*)\]',
               lambda m: '<cite k="' + ",".join(refs.cite[int(x)] for x in re.split(r",\s*", m.group(1))) + '"/>', f)
    return f


def plain_ref(m, refs):
    word, a, conj, b = m.group(1), m.group(2), m.group(3), m.group(4)
    table = refs.tab if word.startswith("Table") else refs.fig
    if a not in table or (b and b not in table):
        return m.group(0)
    out = f'<xref k="{table[a]}" w="{word}"/>'
    if b:
        out += f' {conj} <xref k="{table[b]}" w=""/>'
    return out


def parse(frag, refs):
    t = Tree(); t.feed(pre(frag, refs)); t.close(); return t.root


def sub_math(node):
    """Content of <sub>/<sup>: single letters and digits stay math; words become \\mathrm."""
    toks = []
    for k in node.kids:
        if isinstance(k, str):
            s = html.unescape(k).strip()
            if s:
                toks.append(s if re.fullmatch(r"[A-Za-z]|[0-9*]+|[0-9]?[A-Za-z]", s) else rf"\mathrm{{{s}}}")
        elif isinstance(k, tuple):
            toks.append(MATH_ENT.get(k[1], ""))
        elif k.tag == "i":
            toks.append(text_of(k))
        elif k.tag in ("sub", "sup"):
            toks.append(("_" if k.tag == "sub" else "^") + "{" + sub_math(k) + "}")
    return "".join(toks)


def math_of(node):
    """<i> holding a variable -> its math, else None. A variable is one letter (maybe digits),
    a Greek entity, and <sub>/<sup> parts."""
    parts, letters = [], ""
    for k in node.kids:
        if isinstance(k, str):
            s = html.unescape(k)
            if not re.fullmatch(r"[A-Za-z][0-9]*", s.strip() or "x"):
                return None
            parts.append(s.strip()); letters += s.strip()
        elif isinstance(k, tuple):
            if k[1] not in MATH_ENT:
                return None
            parts.append(MATH_ENT[k[1]]); letters += "g"
        elif k.tag in ("sub", "sup"):
            parts.append(("_" if k.tag == "sub" else "^") + "{" + sub_math(k) + "}")
        else:
            return None
    return "".join(parts) if 0 < len(letters) <= 2 else None


def text_of(node):
    out = []
    for k in node.kids:
        if isinstance(k, str):
            out.append(html.unescape(k))
        elif isinstance(k, tuple):
            out.append(html.unescape(f"&{k[1]};"))
        else:
            out.append(text_of(k))
    return "".join(out)


def tokens(node, refs):
    """Flatten a node into [(kind, latex)]: kind 't' text, 'm' math, 'br' line break."""
    out = []
    for k in node.kids:
        if isinstance(k, str):
            if k:
                out.append(("t", esc(html.unescape(k))))
            continue
        if isinstance(k, tuple):
            name = k[1]
            if name == "radic":
                out.append(("m", RADIC))
            elif name in MATH_ENT:
                out.append(("m", MATH_ENT[name]))
            elif name in TEXT_ENT:
                out.append(("t", TEXT_ENT[name]))
            else:
                raise ValueError(f"entity &{name};")
            continue
        tag, cls = k.tag, k.attrs.get("class", "")
        if tag == "br":
            out.append(("br", ""))
        elif tag == "i" and cls == "st":
            out += tokens(k, refs)
        elif tag == "i":
            m = math_of(k)
            out.append(("m", m) if m is not None else ("t", r"\emph{" + join(tokens(k, refs)) + "}"))
        elif tag in ("sub", "sup"):
            mark = "_" if tag == "sub" else "^"
            if out and out[-1][0] == "m":
                out.append(("m", mark + "{" + sub_math(k) + "}"))
            else:
                cmd = r"\textsubscript" if tag == "sub" else r"\textsuperscript"
                out.append(("t", cmd + "{" + join(tokens(k, refs)) + "}"))
        elif tag == "b":
            out.append(("t", r"\textbf{" + join(tokens(k, refs)) + "}"))
        elif tag == "code":
            out.append(("t", r"\texttt{" + esc(text_of(k)) + "}"))
        elif tag == "a":
            href = k.attrs.get("href", "")
            body = join(tokens(k, refs))
            out.append(("t", rf"\href{{{href}}}{{{body}}}" if href.startswith("http") else body))
        elif tag == "span" and cls == "ci":
            out.append(("t", r"{\scriptsize " + join(tokens(k, refs)) + "}"))
        elif tag == "span" and cls == "lbl":
            continue
        elif tag in ("span", "p", "div"):
            out += tokens(k, refs)
        elif tag == "xref":
            w = k.attrs["w"]
            out.append(("t", (w + "~" if w else "") + r"\ref{" + k.attrs["k"] + "}"))
        elif tag == "eqref":
            out.append(("t", r"Eq.~\eqref{eq:" + k.attrs["n"] + "}"))
        elif tag == "propref":
            out.append(("t", k.attrs["w"] + r"~\ref{prop:" + k.attrs["n"] + "}"))
        elif tag == "algref":
            out.append(("t", r"Algorithm~\ref{alg:round}"))
        elif tag == "appref":
            out.append(("t", r"Appendix~\ref{app:" + k.attrs["n"] + "}"))
        elif tag == "secref":
            out.append(("t", r"\S\ref{sec:" + k.attrs["n"] + "}"))
        elif tag == "cite":
            out.append(("t", r"\cite{" + k.attrs["k"] + "}"))
        else:
            raise ValueError(f"tag <{tag}>")
    return out


# operators and brackets join two math pieces; a plain space does not (the HTML spaces formulas
# with &nbsp;), so a list like "geometry g, gamma the" stays apart
GLUE = re.compile(r"[~()\[\]=+\-,/|<>0-9.*:]*")


def merge(toks):
    """Join math pieces separated only by operators/brackets into one $...$ and wrap the rest."""
    toks = [list(t) for t in toks]
    # glue between two math tokens becomes math
    idx = [i for i, t in enumerate(toks) if t[0] == "m"]
    for a, b in zip(idx, idx[1:]):
        between = toks[a + 1:b]
        if between and all(t[0] == "t" and GLUE.fullmatch(t[1]) for t in between) and \
                sum(len(t[1]) for t in between) <= 12:
            for t in between:
                t[0] = "m"
    # group runs
    runs, cur = [], None
    for t in toks:
        if cur and cur[0] == t[0] and t[0] != "br":
            cur[1].append(t[1])
        else:
            cur = [t[0], [t[1]]]; runs.append(cur)
    # balance brackets: pull a following ')' / ']' (or a preceding '(') into the run
    for i, r in enumerate(runs):
        if r[0] != "m":
            continue
        s = "".join(r[1])
        for o, c in (("(", ")"), ("[", "]")):
            if s.count(o) > s.count(c) and i + 1 < len(runs) and runs[i + 1][0] == "t":
                nxt = "".join(runs[i + 1][1])
                if nxt.startswith(c):
                    r[1].append(c); runs[i + 1][1] = [nxt[1:]]; s += c
            if s.count(c) > s.count(o) and i > 0 and runs[i - 1][0] == "t":
                prv = "".join(runs[i - 1][1])
                if prv.endswith(o):
                    r[1].insert(0, o); runs[i - 1][1] = [prv[:-1]]; s = o + s
    out = []
    for kind, parts in runs:
        s = mjoin(parts) if kind == "m" else "".join(parts)
        if kind == "m":
            s = radic(s).replace("~", " ").strip()
            lead = " " if parts and parts[0].startswith((" ", "~")) else ""
            out.append(lead + "$" + s + "$")
        elif kind == "br":
            out.append("\x00BR")
        else:
            out.append(s)
    return "".join(out)


def mjoin(parts):
    """Concatenate math pieces; a control word followed by a letter needs a space (\\Delta S)."""
    s = ""
    for p in parts:
        if s and re.search(r"\\[A-Za-z]+$", s) and re.match(r"[A-Za-z]", p):
            s += " "
        s += p
    return s


OP = r"(?:-|\+|=|<|>|\\leq|\\geq|\\times|\\cdot)"
NUM = r"[0-9]+(?:\.[0-9]+)?"


def absorb(s):
    """Pull a number that an operator ties to a formula into its $...$: "$n -$~1" -> "$n - 1$",
    "$K$~=~8" -> "$K = 8$", "1~+~$S(n)$" -> "$1 + S(n)$", "$u = c$/8192", "3$\\sqrt{m}$"."""
    for _ in range(3):
        s = re.sub(r"\$([^$]*?)\s*(" + OP + r")\$\s*~\s*(" + NUM + r")(?![0-9,])", r"$\1 \2 \3$", s)
        s = re.sub(r"\$([^$]+)\$~(=|<|>)~(" + NUM + r")(?![0-9,])", r"$\1 \2 \3$", s)
        s = re.sub(r"(?<![\w.,])(" + NUM + r")~(\+|-|=|<)~\$", r"$\1 \2 ", s)
        s = re.sub(r"\$([^$]+)\$/(" + NUM + r")", r"$\1/\2$", s)
        s = re.sub(r"(?<![\w.])([0-9]+)\$\\sqrt", r"$\1\\sqrt", s)
    return s


def radic(s):
    return re.sub(re.escape(RADIC) + r"\s*([A-Za-z0-9]+|\\[a-z]+)", lambda m: r"\sqrt{" + m.group(1) + "}",
                  s).replace(RADIC, r"\surd")


def join(toks):
    return merge(toks)


def inline(frag, refs, br=" "):
    s = merge(tokens(parse(frag, refs), refs))
    s = s.replace("\x00BR", br)
    s = re.sub(r"(?<=[^\s~])\$\$(?=[^\s])", "", s)  # two adjacent math runs: join
    s = absorb(s)
    return re.sub(r"[ \t]+", " ", s).strip()


# ------------------------------------------------------------------- tables

def table(fig_html, refs, colspec=None, wide=True, x_cols=()):
    """<figure class="tb"> -> table/table* with booktabs, captions and notes."""
    cap = re.search(r"<figcaption>(.*?)</figcaption>", fig_html, re.S).group(1)
    tid = re.search(r'id="(tab-[\w-]+)"', fig_html).group(1)
    notes = re.findall(r'<p class="fn">(.*?)</p>', fig_html, re.S)
    thead = re.search(r"<thead>(.*?)</thead>", fig_html, re.S).group(1)
    tbody = re.search(r"<tbody>(.*?)</tbody>", fig_html, re.S).group(1)
    head_rows = rows(thead, refs, header=True)
    body_rows = rows(tbody, refs, header=False)
    ncol = max(sum(int(c["colspan"]) for c in r["cells"]) for r in body_rows)
    if colspec is None:
        colspec = "l" + "".join("X" if j in x_cols else "r" for j in range(1, ncol))
    env = "table*" if wide else "table"
    tab = "tabularx" if "X" in colspec else "tabular"
    width = "{\\textwidth}" if (tab == "tabularx" and wide) else ("{\\columnwidth}" if tab == "tabularx" else "")
    lines = [rf"\begin{{{env}}}[t]", r"\centering\small", rf"\caption{{{inline(cap, refs)}}}", rf"\label{{{lab(tid)}}}",
             r"\setlength{\tabcolsep}{4pt}", rf"\begin{{{tab}}}{width}{{{colspec}}}", r"\toprule"]
    pending = {}
    for ri, r in enumerate(head_rows):
        lines.append(r"\rowcolor{head}" + render_row(r, ri, pending, header=True, colspec=colspec, narrow=not wide))
        spans = cmidrules(r, ri, pending_snapshot=None)
        if ri < len(head_rows) - 1 and spans:
            lines.append(spans)
    lines.append(r"\midrule")
    pending = {}
    for ri, r in enumerate(body_rows):
        lines.append(render_row(r, ri, pending, header=False, colspec=colspec, narrow=not wide))
    lines += [r"\bottomrule", rf"\end{{{tab}}}"]
    for n in notes:
        lines.append(r"\par\smallskip\parbox{\linewidth}{\footnotesize " + inline(n, refs) + "}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines) + "\n"


def rows(part, refs, header):
    out = []
    for tr in re.findall(r"<tr([^>]*)>(.*?)</tr>", part, re.S):
        cls = re.search(r'class="([^"]*)"', tr[0])
        cells = []
        for tag, attrs, body in re.findall(r"<(th|td)([^>]*)>(.*?)</\1>", tr[1], re.S):
            cs = re.search(r'colspan="(\d+)"', attrs); rs = re.search(r'rowspan="(\d+)"', attrs)
            st = re.search(r'style="([^"]*)"', attrs); cc = re.search(r'class="([^"]*)"', attrs)
            cells.append({"colspan": cs.group(1) if cs else "1", "rowspan": rs.group(1) if rs else "1",
                          "style": st.group(1) if st else "", "cls": cc.group(1) if cc else "",
                          # a line break inside a cell is "\\ " (with a space): the public export's
                          # share-path scan reads "\\{\\scriptsize" as \\server\\share
                          "body": body, "tex": inline(body, refs, br="\\\\ ")})
        out.append(cells)
        out[-1] = {"cls": cls.group(1) if cls else "", "cells": cells}
    return [r for r in out] if not header else [r["cells"] for r in out]


def cell_tex(c, align, header, xcol=False, narrow=False):
    t = c["tex"]
    if xcol:                    # a paragraph column wraps by itself; a box would not
        t = t.replace("\\\\", r"\newline ")
    if header and narrow and "\\\\" not in t and len(t) > 8 and "-" in t[3:]:
        i = t.index("-", 3)     # one-column table: a long model name breaks after its first hyphen
        t = t[:i + 1] + "\\\\" + t[i + 1:]
    if "\\\\" in t:
        t = r"\makecell[" + align + "]{" + t + "}"
    if header:
        t = r"\textbf{" + t + "}" if "makecell" not in t else t.replace(r"\makecell[" + align + "]{", r"\makecell[" + align + r"]{\bfseries ")
    m = re.search(r"var\(--c1\)\s*(\d+)%", c.get("style", ""))
    if m:
        t = r"\cellcolor{c1!" + m.group(1) + "}" + t
    return t


def render_row(r, ri, pending, header, colspec, narrow=False):
    cells = r if header else r["cells"]
    cls = "" if header else r["cls"]
    out, col, it = [], 0, iter(cells)
    aligns = [c for c in colspec if c in "lrcXp"]
    ncol = len(aligns)
    while col < ncol:
        if col in pending:
            left, n, t = pending[col]
            if left == 1:
                del pending[col]; out.append(rf"\multirow{{-{n}}}{{*}}{{{t}}}" if t is not None else "")
            else:
                pending[col] = (left - 1, n, t); out.append("")
            col += 1; continue
        try:
            c = next(it)
        except StopIteration:
            out.append(""); col += 1; continue
        span, rspan = int(c["colspan"]), int(c["rowspan"])
        align = "l" if aligns[col] in "lXp" else "r"
        t = cell_tex(c, "c" if (header and span > 1) else align, header,
                     xcol=aligns[col] == "X" and span == 1, narrow=narrow)
        if t and cls == "em":       # a row's style goes inside the cell, before any span wrapper
            t = r"\textbf{" + t + "}"
        elif t and cls == "grp":
            t = r"\textit{" + t + "}"
        if rspan > 1:
            for j in range(col, col + span):
                pending[j] = (rspan - 1, rspan, t if j == col else None)
            out.extend([""] * span); col += span; continue
        if span > 1:
            t = rf"\multicolumn{{{span}}}{{c}}{{{t}}}"
        out.append(t); col += span
    line = " & ".join(out) + r" \\"
    return {"em": r"\rowcolor{tint1}", "grp": r"\rowcolor{band2}"}.get(cls, "") + line


def cmidrules(r, ri, pending_snapshot):
    col, rules = 1, []
    for c in r:
        span = int(c["colspan"])
        if span > 1:
            rules.append(rf"\cmidrule(lr){{{col}-{col + span - 1}}}")
        col += span
    return " ".join(rules)
