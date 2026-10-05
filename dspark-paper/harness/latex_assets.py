"""The LaTeX paper's tables and figures, written from HTML fragments.

paper_figs.py builds every generated table and figure from the run data as an HTML fragment (a
<figure> with its caption and, for a figure, an SVG); this module turns each one into
latex/gen/: a table into gen/tab-<name>.tex (booktabs), a figure into gen/fig-<name>-img.pdf
(the SVG printed by Chrome with embedded TrueType fonts) and gen/fig-<name>.tex (the figure
environment). The image has its own name because arXiv deletes a PDF that shares its name
with a .tex file. The two hand-drawn figures live in latex/figures/*.svg; running this file
renders each to the PDF beside it when the SVG is newer.

usage: latex_assets.py [--force]      (needs Chrome and network for the fonts)
"""
import glob
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import html2tex as h2t  # noqa: E402

LATEX = os.path.normpath(os.path.join(HERE, "..", "latex"))
GEN = os.path.join(LATEX, "gen")
FIGURES = os.path.join(LATEX, "figures")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FIG_WIDTH = r"0.8\textwidth"  # one scale for every figure: SVG text keeps one size (~6.6 pt)
# Static TrueType files: Google Fonts serves these to a plain client (a browser gets variable
# WOFF2, which Chrome would embed as Type 3).
FONTS = ("https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;"
         "0,6..72,600;0,6..72,700;1,6..72,400;1,6..72,600&family=IBM+Plex+Mono:wght@400;500;700&display=swap")
# The figures' colours: the SVGs name them as CSS variables. --c1..--c6 are an Okabe-Ito palette.
TOKENS = """:root{
  --paper:#FFFFFF; --ink:#111111; --ink-2:#333333; --ink-3:#5C5C5C;
  --rule:#111111; --rule-2:#BBBBBB; --grid:#D8D8D8; --band:#F0F0F0; --band-2:#F7F7F7;
  --link:#1A4E8A; --mark:#000000;
  --c1:#0072B2; --c2:#E69F00; --c3:#009E73; --c4:#CC79A7; --c5:#D55E00; --c6:#56B4E9;
  --tint-1:#E3EEF7; --tint-2:#FBEBCC; --tint-3:#DDF1EA; --head:#EAF1F7;}"""


# ------------------------------------------------------------------ rendering

def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "latex_assets"})) as r:
        return r.read()


def font_css(work):
    """@font-face rules pointing at downloaded TrueType files."""
    css = fetch(FONTS).decode()
    urls = re.findall(r"url\((https://fonts\.gstatic\.com/[^)]+)\)", css)
    if not urls or any(not u.endswith(".ttf") for u in urls):
        sys.exit("latex_assets: Google Fonts did not serve TrueType files; the PDFs would carry Type 3")
    for i, u in enumerate(urls):
        path = os.path.join(work, f"font{i}.ttf")
        with open(path, "wb") as f:
            f.write(fetch(u))
        css = css.replace(u, "file://" + path)
    return css


def render_svg(svg, pdf, work, css):
    """One SVG -> a one-page PDF of the SVG's own size."""
    w, h = (float(x) for x in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    page = ("<!doctype html><html><head><meta charset='utf-8'><style>" + css + TOKENS +
            f"@page{{size:{w}px {h}px;margin:0}} html,body{{margin:0;padding:0;background:#fff;overflow:hidden}}"
            f"svg{{display:block;width:{w}px;height:{h}px}}</style></head><body>" + svg + "</body></html>")
    src = os.path.join(work, os.path.basename(pdf) + ".html")
    open(src, "w", encoding="utf-8").write(page)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=10000", f"--print-to-pdf={pdf}", "file://" + src],
                   check=True, capture_output=True)
    check_pdf(pdf)


def fonts_of(resources, seen):
    """(name, subtype, embedded) for every font a page draws, through form XObjects too."""
    out = []
    if resources is None:
        return out
    resources = resources.get_object()
    for _, ref in (resources.get("/Font") or {}).items():
        font = ref.get_object()
        if id(font) in seen:
            continue
        seen.add(id(font))
        sub = font.get("/Subtype")
        desc = font["/DescendantFonts"][0].get_object() if sub == "/Type0" else font
        fd = desc.get("/FontDescriptor")
        fd = fd.get_object() if fd is not None else {}
        out.append((str(font.get("/BaseFont", "(none)")), str(sub),
                    any(k in fd for k in ("/FontFile", "/FontFile2", "/FontFile3"))))
    for _, ref in (resources.get("/XObject") or {}).items():
        x = ref.get_object()
        if x.get("/Subtype") == "/Form":
            out += fonts_of(x.get("/Resources"), seen)
    return out


def check_pdf(pdf):
    """A figure PDF: one page, every font embedded, none of them Type 3."""
    from pypdf import PdfReader
    r = PdfReader(pdf)
    bad = [f for f in fonts_of(r.pages[0].get("/Resources"), set()) if f[1] == "/Type3" or not f[2]]
    if len(r.pages) != 1 or bad:
        sys.exit(f"latex_assets: {pdf} has {len(r.pages)} pages, bad fonts {bad}")


# ------------------------------------------------------------------ fragments -> LaTeX

def refs():
    """Citation keys in refs.bib order (a caption's [n] is the n-th entry). Fragments name other
    tables and figures by link (class xref), so no number maps are needed."""
    keys = re.findall(r"@\w+\{([^,\s]+),", open(os.path.join(LATEX, "refs.bib")).read())
    return h2t.Refs({}, {}, {i: k for i, k in enumerate(keys, 1)})


def table_tex(frag, r):
    """<figure class="tb"> -> (name, LaTeX table). A column of long text cells becomes a
    paragraph column (X); a table with one stays full width."""
    tid = re.search(r'id="(tab-[\w-]+)"', frag).group(1)
    body = re.search(r"<tbody>(.*?)</tbody>", frag, re.S).group(1)
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S)
    widths = {}
    for tr in trs:
        j = 0
        for attrs, cell in re.findall(r"<td([^>]*)>(.*?)</td>", tr, re.S):
            span = int((re.search(r'colspan="(\d+)"', attrs) or [None, "1"])[1])
            if 'class="txt"' in attrs:
                widths[j] = max(widths.get(j, 0), len(re.sub(r"<[^>]+>|&\w+;", "x", cell)))
            j += span
    ncol = max(sum(int((re.search(r'colspan="(\d+)"', a) or [None, "1"])[1]) for a in re.findall(r"<td([^>]*)>", tr))
               for tr in trs)
    spec = "".join("X" if widths.get(j, 0) > 30 else ("l" if j == 0 or j in widths else "r") for j in range(ncol))
    col = 'class="tb col"' in frag
    return h2t.lab(tid).replace(":", "-"), h2t.table(frag, r, colspec=spec, wide=not col or "X" in spec)


def figure_tex(frag, r, work, css):
    """<figure class="fg"> -> (name, figure environment), its image rendered to gen/<name>-img.pdf."""
    fid = re.search(r'id="(fig-[\w-]+)"', frag).group(1)
    name = h2t.lab(fid).replace(":", "-")
    render_svg(re.search(r"<svg.*?</svg>", frag, re.S).group(0), os.path.join(GEN, name + "-img.pdf"), work, css)
    cap = h2t.inline(re.search(r"<figcaption>(.*?)</figcaption>", frag, re.S).group(1), r)
    env = [r"\begin{figure*}[t]", r"\centering",                 # one line per item: the export's
           r"\includegraphics[width=" + FIG_WIDTH + "]{gen/" + name + "-img.pdf}",  # path scan reads a
           r"\caption{" + cap + "}", r"\label{" + h2t.lab(fid) + "}",          # backslash pair before
           r"\end{figure*}"]                                                    # text as a share path
    return name, "\n".join(env) + "\n"


class Writer:
    """Writes fragments into latex/gen/ (fonts fetched once)."""

    def __init__(self):
        os.makedirs(GEN, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory()
        self.css = font_css(self.tmp.name)
        self.refs = refs()

    def write(self, frag):
        if '<figure class="fg"' in frag:
            name, tex = figure_tex(frag, self.refs, self.tmp.name, self.css)
        else:
            name, tex = table_tex(frag, self.refs)
        open(os.path.join(GEN, name + ".tex"), "w").write(tex)
        return name


def main():
    """Render each hand-drawn figure (latex/figures/*.svg) to the PDF beside it when stale."""
    force = "--force" in sys.argv[1:]
    todo = [s for s in sorted(glob.glob(os.path.join(FIGURES, "*.svg")))
            if force or not os.path.exists(s[:-4] + ".pdf") or os.path.getmtime(s[:-4] + ".pdf") < os.path.getmtime(s)]
    if not todo:
        print("latex_assets: hand-drawn figures up to date")
        return
    with tempfile.TemporaryDirectory() as work:
        css = font_css(work)
        for s in todo:
            render_svg(open(s, encoding="utf-8").read(), s[:-4] + ".pdf", work, css)
    print("latex_assets: rendered " + ", ".join(os.path.basename(s) for s in todo))


if __name__ == "__main__":
    main()
