#!/usr/bin/env python3
"""paper.html -> AdaSpark-html.pdf (the web page printed) for arXiv's PDF route (https://info.arxiv.org/help/submit_pdf.html),
then a check of that PDF against arXiv's rules.

arXiv takes one PDF whose fonts are all embedded outline fonts (TrueType or Type 1, never Type 3),
with no JavaScript, not produced from TeX. The paper prints with headless Chrome, whose PDF writer
embeds a variable font as Type 3. Google Fonts serves a browser the variable Newsreader; served to
a plain HTTP client, the same stylesheet names static TrueType files. So the print copy carries
that stylesheet inline, pointing at the downloaded files, and the check fails the build on any
Type 3 or unembedded font.

The print copy also takes the paper's full title as <title>, which becomes the PDF's Title (the web
page keeps its short one). Chrome writes a tagged PDF, and here also its outline (bookmarks).

usage: print_pdf.py [OUT.pdf]          needs network (the fonts) and pypdf (the check)
"""
import html, os, re, subprocess, sys, tempfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper.html")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LETTER = (612, 792)


def fetch(url):
    # urllib's own User-Agent: Google Fonts then answers with static TrueType, not WOFF2.
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def print_copy(src, work):
    """The page as printed: fonts inline from local static files, the full title as <title>."""
    page = open(src, encoding="utf-8").read()
    link = re.search(r'<link rel="stylesheet" href="(https://fonts\.googleapis\.com/css2\?[^"]+)">', page)
    if not link:
        sys.exit("print_pdf: no Google Fonts stylesheet in paper.html")
    css = fetch(html.unescape(link.group(1))).decode()
    urls = re.findall(r"url\((https://fonts\.gstatic\.com/[^)]+)\)", css)
    if not urls or any(not u.endswith(".ttf") for u in urls):
        sys.exit("print_pdf: Google Fonts did not serve TrueType files; the PDF would carry Type 3")
    for i, u in enumerate(urls):
        path = os.path.join(work, f"font{i}.ttf")
        with open(path, "wb") as f:
            f.write(fetch(u))
        css = css.replace(u, "file://" + path)
    page = page.replace(link.group(0), "<style>\n" + css + "</style>")
    page = re.sub(r'<link rel="preconnect"[^>]*>\n?', "", page)
    h1 = re.search(r'<h1 class="title">(.*?)</h1>', page, re.S)
    title = html.unescape(re.sub(r"<br\s*/?>", " ", h1.group(1)))
    title = re.sub(r"<[^>]+>", "", title).replace("‑", "-")
    page = re.sub(r"<title>.*?</title>", "<title>" + html.escape(title) + "</title>", page, count=1)
    out = os.path.join(work, "paper.html")
    open(out, "w", encoding="utf-8").write(page)
    return out, title, len(urls)


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
        desc = font
        if sub == "/Type0":
            desc = font["/DescendantFonts"][0].get_object()
        fd = desc.get("/FontDescriptor")
        fd = fd.get_object() if fd is not None else {}
        embedded = any(k in fd for k in ("/FontFile", "/FontFile2", "/FontFile3"))
        out.append((str(font.get("/BaseFont", "(none)")), str(sub), embedded))
    for _, ref in (resources.get("/XObject") or {}).items():
        x = ref.get_object()
        if x.get("/Subtype") == "/Form":
            out += fonts_of(x.get("/Resources"), seen)
    return out


def check(pdf, title):
    """arXiv's rules, each one a line; returns the problems."""
    from pypdf import PdfReader
    r = PdfReader(pdf)
    raw = open(pdf, "rb").read()
    root = r.trailer["/Root"].get_object()
    names = root.get("/Names")
    names = names.get_object() if names is not None else {}
    problems = []
    fonts, seen = [], set()
    for page in r.pages:
        fonts += fonts_of(page.get("/Resources"), seen)
    type3 = [f for f in fonts if f[1] == "/Type3"]
    bare = [f for f in fonts if f[1] != "/Type3" and not f[2]]
    sizes = {(round(float(p.mediabox.width)), round(float(p.mediabox.height))) for p in r.pages}
    producer = (r.metadata or {}).get("/Producer", "")
    rules = [
        ("one PDF, not encrypted", not r.is_encrypted),
        ("no Type 3 fonts", not type3),
        ("every font embedded", not bare),
        ("no JavaScript", b"/JavaScript" not in raw and b"/JS" not in raw and "/JavaScript" not in names),
        ("no forms, attachments or open actions",
         all(k not in root for k in ("/AcroForm", "/OpenAction", "/AA")) and "/EmbeddedFiles" not in names),
        ("not produced by TeX", "tex" not in str(producer).lower()),
        ("text is extractable", "AdaSpark" in (r.pages[0].extract_text() or "")),
    ]
    for name, ok in rules:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            problems.append(name)
    if type3:
        print(f"       {len(type3)} Type 3 fonts")
    for f in bare:
        print(f"       not embedded: {f[0]}")
    print(f"  {'ok  ' if sizes == {LETTER} else 'note'} page size {sorted(sizes)} (US Letter is 612x792)")
    tagged = "/StructTreeRoot" in root and "/MarkInfo" in root
    print(f"  {'ok  ' if tagged else 'note'} tagged for screen readers (recommended)")
    print(f"  {'ok  ' if '/Outlines' in root else 'note'} outline (bookmarks)")
    got = (r.metadata or {}).get("/Title", "")
    print(f"  {'ok  ' if got == title else 'note'} title: {got!r}")
    print(f"  {len(r.pages)} pages, {os.path.getsize(pdf) / 2**20:.2f} MiB, producer {producer!r}")
    kinds = sorted({(f[0].split("+")[-1], f[1]) for f in fonts})
    print("  fonts: " + ", ".join(f"{n} {t[1:]}" for n, t in kinds))
    return problems


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "AdaSpark-html.pdf"))
    with tempfile.TemporaryDirectory() as work:
        page, title, n = print_copy(PAPER, work)
        print(f"print copy: {n} static font files inline, title {title!r}")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--generate-pdf-document-outline", "--virtual-time-budget=15000",
                        f"--print-to-pdf={out}", "file://" + page],
                       check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    print(f"wrote {out}")
    problems = check(out, title)
    if problems:
        sys.exit("print_pdf: the PDF breaks arXiv's rules: " + "; ".join(problems))


if __name__ == "__main__":
    main()
