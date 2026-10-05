"""paper.html's figures and tables -> the LaTeX paper's generated parts (latex/gen/).

Every table and figure of the paper, its caption and its numbers, is produced once, by
paper_figs.py (generated blocks) or by hand (Tables 11-15, Figures 1-2), in paper.html. This
script carries them into the LaTeX version so the two cannot disagree:
  gen/tab-<name>.tex   the table environment (booktabs), caption and label
  gen/fig-<name>.pdf   the figure, printed from its SVG by Chrome with TrueType fonts
  gen/fig-<name>.tex   the figure environment, caption and label
  --bib                (re)writes latex/refs.bib from the reference list; overwrites hand edits

usage: latex_assets.py [--bib]      (needs network for the fonts, like print_pdf.py)
"""
import html
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import html2tex as h2t  # noqa: E402
import print_pdf  # noqa: E402

PAPER = os.path.join(HERE, "..", "paper.html")
LATEX = os.path.join(HERE, "..", "latex")
GEN = os.path.join(LATEX, "gen")
FIG_WIDTH = r"0.8\textwidth"  # one scale for every figure: SVG text keeps one size (~6.6 pt)


# ------------------------------------------------------------------ references

def references(page):
    items = re.findall(r"<li>(.*?)</li>", re.search(r'<ol class="refs">(.*?)</ol>', page, re.S).group(1), re.S)
    out, used = [], set()
    for it in items:
        m = re.match(r"\s*(.*?)\s*<i>(.*?)</i>\s*(.*)$", it, re.S)
        authors, title, rest = m.group(1).rstrip("."), m.group(2).strip(), m.group(3).strip()
        alias = re.match(r"\((\w[\w.-]*)\)\.?\s*", rest)
        if alias:
            title = title.rstrip(".") + f" ({alias.group(1)})"; rest = rest[alias.end():]
        title = title.rstrip(".")
        rest = rest.lstrip(". ").strip()
        plain = html.unescape(re.sub(r"<[^>]+>", "", authors))
        first = re.split(r",| et al| \(", plain)[0].strip()
        last = re.sub(r"[^a-z]", "", (first.split()[-1] if " " in first and "." in first else first).lower())
        year = (re.findall(r"\b(19\d\d|20\d\d)\b", rest) or ["nd"])[-1]
        word = next((w for w in re.findall(r"[A-Za-z0-9]+", html.unescape(re.sub(r"<[^>]+>", "", title))).__iter__()
                     if w.lower() not in {"a", "an", "the", "on", "to", "from", "via", "with", "for", "of"}), "x").lower()
        key = f"{last}{year}{word}"
        while key in used:
            key += "b"
        used.add(key)
        out.append({"key": key, "authors": plain, "title": title, "rest": rest})
    return out


def bib_authors(a):
    if "(" in a or not re.search(r"[A-Z]\.", a):   # an organisation or a named team: print as written
        return "{" + a + "}"
    parts = [p.strip() for p in re.split(r",\s*", a.replace(" et al.", ", others").replace(" et al", ", others"))]
    return " and ".join(p for p in parts if p)


def write_bib(page, refs_tex):
    lines = []
    for r in references(page):
        title = h2t.inline(r["title"], refs_tex)
        rest = h2t.inline(r["rest"], refs_tex)
        rest = re.sub(r"\b(github\.com/\S+?)([;.,]?)(?=\s|$)", lambda m: r"\url{" + m.group(1) + "}" + m.group(2), rest)
        lines.append(f"@misc{{{r['key']},\n  author = {{{bib_authors(r['authors'])}}},\n"
                     f"  title = {{{{{title}}}}},\n  howpublished = {{{rest.rstrip('.')}}}\n}}\n")
    open(os.path.join(LATEX, "refs.bib"), "w").write("\n".join(lines))
    print(f"wrote refs.bib: {len(lines)} entries")


# ------------------------------------------------------------------ tables and figures

def figures_of(page, cls):
    return re.findall(rf'<figure class="{cls}[^"]*">.*?</figure>', page, re.S)


def write_tables(page, refs):
    names = []
    for f in figures_of(page, "tb"):
        tid = re.search(r'id="(tab-[\w-]+)"', f).group(1)
        col = 'class="tb col"' in f
        body = re.search(r"<tbody>(.*?)</tbody>", f, re.S).group(1)
        # a column of long text cells becomes a paragraph column (X); short text stays left-aligned
        widths = {}
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
            j = 0
            for attrs, cell in re.findall(r"<td([^>]*)>(.*?)</td>", tr, re.S):
                span = int((re.search(r'colspan="(\d+)"', attrs) or [None, "1"])[1])
                if 'class="txt"' in attrs:
                    widths[j] = max(widths.get(j, 0), len(re.sub(r"<[^>]+>|&\w+;", "x", cell)))
                j += span
        ncol = max(sum(int((re.search(r'colspan="(\d+)"', a) or [None, "1"])[1]) for a in re.findall(r"<td([^>]*)>", tr))
                   for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S))
        spec = ""
        for j in range(ncol):
            if widths.get(j, 0) > 30:
                spec += "X"
            elif j == 0 or j in widths:
                spec += "l"
            else:
                spec += "r"
        if tid == "tab-constants":    # long names and long values: both wrap
            spec = "lXXl"
        # one column only for tables of short cells; a paragraph column needs the full width
        tex = h2t.table(f, refs, colspec=spec, wide=not col or "X" in spec)
        name = h2t.lab(tid).replace(":", "-")
        open(os.path.join(GEN, name + ".tex"), "w").write(tex)
        names.append(name)
    print(f"wrote {len(names)} tables: " + " ".join(names))


def write_figures(page, refs, work):
    printed, title, _ = print_pdf.print_copy(PAPER, work)
    head = open(printed, encoding="utf-8").read()
    head = head[:head.index('<div class="page">')]
    names = []
    for f in figures_of(page, "fg"):
        fid = re.search(r'id="(fig-[\w-]+)"', f).group(1)
        name = h2t.lab(fid).replace(":", "-")
        svg = re.search(r"<svg.*?</svg>", f, re.S).group(0)
        w, h = (float(x) for x in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
        doc = (head + f"<style>@page{{size:{w}px {h}px;margin:0}} html,body{{margin:0;padding:0;"
               f"background:#fff;overflow:hidden}} svg{{display:block;width:{w}px;height:{h}px;max-width:none}}</style>"
               + svg + "</body></html>")
        src = os.path.join(work, name + ".html")
        open(src, "w", encoding="utf-8").write(doc)
        pdf = os.path.join(GEN, name + ".pdf")
        subprocess.run([print_pdf.CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=10000", f"--print-to-pdf={pdf}", "file://" + src],
                       check=True, capture_output=True)
        cap = h2t.inline(re.search(r"<figcaption>(.*?)</figcaption>", f, re.S).group(1), refs)
        env = [r"\begin{figure*}[t]", r"\centering",                 # one line per item: the export's
               r"\includegraphics[width=" + FIG_WIDTH + "]{gen/" + name + ".pdf}",  # path scan reads a
               r"\caption{" + cap + "}", r"\label{" + h2t.lab(fid) + "}",          # backslash pair before
               r"\end{figure*}"]                                                    # text as a share path
        open(os.path.join(GEN, name + ".tex"), "w").write("\n".join(env) + "\n")
        names.append(name)
    print(f"wrote {len(names)} figures: " + " ".join(names))
    return names


def check_figures(names):
    """Every figure PDF: one page, fonts embedded, none of them Type 3."""
    from pypdf import PdfReader
    for n in names:
        r = PdfReader(os.path.join(GEN, n + ".pdf"))
        fonts = print_pdf.fonts_of(r.pages[0].get("/Resources"), set())
        bad = [f for f in fonts if f[1] == "/Type3" or not f[2]]
        if len(r.pages) != 1 or bad:
            sys.exit(f"latex_assets: {n}.pdf has {len(r.pages)} pages, bad fonts {bad}")
    print(f"checked {len(names)} figure PDFs: one page each, fonts embedded, no Type 3")


def main():
    args = sys.argv[1:]
    page = open(PAPER, encoding="utf-8").read()
    refs = h2t.Refs(page, {i: r["key"] for i, r in enumerate(references(page), 1)})
    os.makedirs(GEN, exist_ok=True)
    if "--bib" in args:
        write_bib(page, refs)
    write_tables(page, refs)
    with tempfile.TemporaryDirectory() as work:
        names = write_figures(page, refs, work)
    check_figures(names)


if __name__ == "__main__":
    main()
