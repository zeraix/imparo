# Adaptive DSpark: the paper, its test set, and its harness

AdaSpark: Adaptive DSpark with Online Learning for Tree Verification and N-gram Fill.
Liquan Liu, Yifan Zhang, Bowei Xu. arXiv:2610.05774 (cs.LG), https://arxiv.org/abs/2610.05774

Everything a number in the paper depends on, apart from the engine itself.

```
latex/
  AdaSpark.tex        the paper (pdfLaTeX); refs.bib its references
  gen/                its generated tables and figures, written by harness/paper_figs.py
  figures/            the two hand-drawn figures (SVG, rendered to PDF)
manifest.md           planning notes behind the paper (claims, open questions)
testset/
  build.py            builds sets/ from six public datasets; rules in its docstring and paper §6.1
  sets/*.json         the built conversation sets (committed; source revision + hash per item)
  context.py          builds context/: raw-continuation prompts for the microbenchmarks (§3, §5)
  context/*.ids       prompt token ids per vocabulary (lfm25, qwen3) and context; manifest.json
harness/
  run.sh              every paper measurement, staged model by model (below)
  agentic.py          serves the conversation sets through each engine's own server
  check.py            PASS/FAIL gate on a finished agentic run
  agentic_report.py   per-arm and per-set tables from results.jsonl
  table4.py           width policies at fixed contexts (chain / fixed widths / budget)
  survey.py           verify cost by pinned width and context (Table 1)
  figs.py             the mechanism figures, from a probe run's server logs
  paper_figs.py       every generated table and figure, from one run directory, into latex/gen/
  build_latex.py      AdaSpark.pdf and the arXiv source bundle
runs/                 run outputs (not committed)
```

No prompt in `testset/` is written by us.

## Rebuild the test set

```
python3 dspark-paper/testset/build.py        # sets/, ~1 min, needs network
python3 dspark-paper/testset/context.py \
  "lfm25=<LFM2.5 GGUF>" "qwen3=<Qwen3 GGUF>"  # context/, needs llama-tokenize
git diff --stat dspark-paper/testset          # empty when the sources have not moved
```

## Run the measurements

```
dspark-paper/harness/run.sh dspark-paper/runs/<commit> [lfm26 moekm q4b q8b]
```

Per model, in order, stopping at the first failure:

```
smoke  -> check -> full set -> check -> width table -> (Table 1 survey, LFM2.5-2.6B) -> figure run
```

The smoke stage runs every arm on one conversation per set (two turns each) and must pass
`check.py` before the full set starts. `RUN.log` in the output folder records the commit, the
binaries' md5 and every stage's start, exit and check result. `REPEATS=2` runs the full set
twice, the second time with the arms in reverse order.
