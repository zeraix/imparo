#!/usr/bin/env python3
"""Raw-continuation prompts at fixed context lengths, for the paper's microbenchmarks (the
verify-cost survey and the width tables). Built from LongBench text by a fixed rule.

These runs measure what a verify costs at a given number of keys and which width a policy
picks there, so the prompt is token ids with no chat template: the model continues the text.
Three kinds of text, because acceptance (and so the best width) depends on what is being
continued:

  A  report   LongBench gov_report
  B  news     LongBench multi_news
  C  code     LongBench lcc

RULE. For each (kind, context length c): take the task's contexts in the data file's order,
whole, joined by a blank line, until the text tokenises to at least c tokens (the model's
own vocabulary: llama-tokenize --ids --no-bos), and keep the first c tokens. A prefix, not
LongBench's middle cut: the middle cut protects a question at the end, and a continuation
prompt has none -- cut from the middle, a 443-token prompt would splice a document's first
paragraphs onto its last. No item is chosen by hand, and none is repeated: a repeated
passage is trivially predictable and would flatter every drafter.

The LongBench data comes from build.py's pinned download (.cache/longbench). The output
names a vocabulary, not a model: every target with that vocabulary reads the same ids.

usage: context.py VOCAB=MODEL.gguf [VOCAB=MODEL.gguf ...] [--ctx 443,1596,8444]
       writes context/{VOCAB}_{c}{A,B,C}.ids and context/manifest.json
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, ".cache", "longbench")
OUT = os.path.join(HERE, "context")
TOKENIZE = os.environ.get("LLAMA_TOKENIZE", "llama-tokenize")  # llama.cpp's tokenizer tool, on PATH by default
KINDS = {"A": "gov_report", "B": "multi_news", "C": "lcc"}


def tokenize(model, text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    try:
        out = subprocess.run([TOKENIZE, "-m", model, "-f", path, "--ids", "--no-bos",
                              "--log-disable"], capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    line = [l for l in out.splitlines() if l.startswith("[")][-1]
    return json.loads(line)


def items(task):
    with open(os.path.join(DATA, f"{task}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            yield task, r.get("_id"), r["context"]


def build(model, c, task):
    """Pack whole items until the text reaches c tokens; keep the first c."""
    texts, used = [], []
    for task, rid, ctx in items(task):
        texts.append(ctx)
        used.append(f"{task}:{rid}")
        # a character is at most one token in these vocabularies, so tokenise only once the
        # text could possibly be long enough
        if sum(len(t) for t in texts) < c:
            continue
        ids = tokenize(model, "\n\n".join(texts))
        if len(ids) >= c:
            return ids[:c], used, len(ids)
    sys.exit(f"{task}: the data ran out before {c} tokens")


def main():
    args = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    ctxs = [int(x) for x in (sys.argv[sys.argv.index("--ctx") + 1] if "--ctx" in sys.argv
                             else "443,1596,8444").split(",")]
    if not args:
        sys.exit(__doc__)
    os.makedirs(OUT, exist_ok=True)
    mpath = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for a in args:
        vocab, model = a.split("=", 1)
        for c in ctxs:
            for kind, task in KINDS.items():
                ids, used, packed = build(model, c, task)
                name = f"{vocab}_{c}{kind}"
                body = " ".join(map(str, ids))
                with open(os.path.join(OUT, f"{name}.ids"), "w") as f:
                    f.write(body + "\n")
                manifest[name] = {"tokens": len(ids), "packed_tokens": packed, "items": used,
                                  "tokenizer_model": os.path.basename(model),
                                  "sha256": hashlib.sha256(body.encode()).hexdigest()[:16]}
                print(f"{name:<18} {len(ids):>5} tokens from {len(used)} item(s), "
                      f"packed {packed}: {', '.join(used)}")
    with open(mpath, "w") as f:
        json.dump(dict(sorted(manifest.items())), f, indent=1)


if __name__ == "__main__":
    main()
