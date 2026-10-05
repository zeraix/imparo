"""latex/AdaSpark.tex -> AdaSpark.pdf and the arXiv source bundle.

Steps: render the hand-drawn figures if their SVGs changed (latex_assets.py); pdflatex, bibtex,
pdflatex twice; fail on
any undefined reference or citation, overfull line or BibTeX warning; check every PDF font is
embedded and none is Type 3; move the PDF to dspark-paper/AdaSpark.pdf; pack latex/arxiv-source.tar.gz
(AdaSpark.tex, AdaSpark.bbl, gen/, figures/*.pdf) and compile that bundle alone in an empty directory,
as arXiv does. latex/gen/ is tracked: paper_figs.py RUNDIR rewrites it from the run data.

usage: build_latex.py       TeX from $TEXBIN, else ~/Library/TinyTeX/bin/*, else PATH
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LATEX = os.path.normpath(os.path.join(HERE, "..", "latex"))
OUT = os.path.normpath(os.path.join(HERE, "..", "AdaSpark.pdf"))
BUNDLE = os.path.join(LATEX, "arxiv-source.tar.gz")


def texbin():
    for d in [os.environ.get("TEXBIN", "")] + glob.glob(os.path.expanduser("~/Library/TinyTeX/bin/*")):
        if d and os.path.exists(os.path.join(d, "pdflatex")):
            return d
    p = shutil.which("pdflatex")
    if not p:
        sys.exit("build_latex: no pdflatex (set TEXBIN or install TinyTeX)")
    return os.path.dirname(p)


def run(tb, cwd, *cmd):
    r = subprocess.run([os.path.join(tb, cmd[0]), *cmd[1:]], cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def compile_in(tb, cwd, bibtex=True):
    code, out = run(tb, cwd, "pdflatex", "-interaction=nonstopmode", "-halt-on-error", "AdaSpark.tex")
    if code:
        sys.exit("build_latex: pdflatex failed\n" + "\n".join(l for l in out.splitlines() if l.startswith("!"))[:2000])
    if bibtex:
        code, out = run(tb, cwd, "bibtex", "AdaSpark")
        warn = [l for l in out.splitlines() if "Warning" in l or "error" in l.lower()]
        if code or warn:
            sys.exit("build_latex: bibtex: " + "; ".join(warn[:5]))
    for _ in range(2):
        code, out = run(tb, cwd, "pdflatex", "-interaction=nonstopmode", "-halt-on-error", "AdaSpark.tex")
        if code:
            sys.exit("build_latex: pdflatex failed on a later pass")
    log = open(os.path.join(cwd, "AdaSpark.log"), errors="replace").read()
    bad = re.findall(r"(?:Reference|Citation) `[^']+' on page \d+ undefined|Overfull \\hbox \([\d.]+pt too wide\)", log)
    if bad:
        sys.exit("build_latex: " + "; ".join(sorted(set(bad))[:8]))
    return os.path.join(cwd, "AdaSpark.pdf")


def check_fonts(pdf):
    sys.path.insert(0, HERE)
    import latex_assets
    from pypdf import PdfReader
    r = PdfReader(pdf)
    fonts, seen = [], set()
    for p in r.pages:
        fonts += latex_assets.fonts_of(p.get("/Resources"), seen)
    type3 = [f for f in fonts if f[1] == "/Type3"]
    bare = [f for f in fonts if f[1] != "/Type3" and not f[2]]
    if type3 or bare:
        sys.exit(f"build_latex: {len(type3)} Type 3 fonts, not embedded: {[f[0] for f in bare]}")
    return len(r.pages), sorted({f[1] for f in fonts})


def main():
    if not glob.glob(os.path.join(LATEX, "gen", "*.tex")):
        sys.exit("build_latex: latex/gen/ is empty; it is tracked (git checkout) or rewritten by paper_figs.py RUNDIR")
    subprocess.run([sys.executable, os.path.join(HERE, "latex_assets.py")], check=True)
    tb = texbin()
    pdf = compile_in(tb, LATEX)
    pages, kinds = check_fonts(pdf)
    shutil.move(pdf, OUT)   # one copy of the paper, outside the build folder
    # arXiv removes x.pdf when x.tex exists ("name conflict"): it takes the PDF for that file's output
    stems = {}
    for p in glob.glob(os.path.join(LATEX, "gen", "*")) + glob.glob(os.path.join(LATEX, "figures", "*")):
        stems.setdefault(os.path.splitext(p)[0], set()).add(os.path.splitext(p)[1])
    clash = sorted(os.path.basename(s) for s, ext in stems.items() if {".tex", ".pdf"} <= ext)
    if clash:
        sys.exit("build_latex: arXiv would delete these PDFs, which share a name with a .tex: " + ", ".join(clash))
    with tarfile.open(BUNDLE, "w:gz") as t:
        for name in ["AdaSpark.tex", "AdaSpark.bbl"] + sorted(
                os.path.relpath(p, LATEX) for p in glob.glob(os.path.join(LATEX, "gen", "*"))
                + glob.glob(os.path.join(LATEX, "figures", "*.pdf"))):
            t.add(os.path.join(LATEX, name), arcname=name)
    with tempfile.TemporaryDirectory() as d:          # the bundle alone, the way arXiv builds it
        with tarfile.open(BUNDLE) as t:
            t.extractall(d)
        alone = compile_in(tb, d, bibtex=False)
        alone_pages, _ = check_fonts(alone)
    if alone_pages != pages:
        sys.exit(f"build_latex: the bundle builds {alone_pages} pages, the source {pages}")
    print(f"ok  {OUT}: {pages} pages, {os.path.getsize(OUT) / 2**20:.2f} MiB, fonts {', '.join(k[1:] for k in kinds)}")
    print(f"ok  {BUNDLE}: {os.path.getsize(BUNDLE) / 2**10:.0f} KiB, compiles alone to the same {pages} pages")


if __name__ == "__main__":
    main()
