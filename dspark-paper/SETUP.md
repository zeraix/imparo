# AdaSpark paper: setup and reproduction

The experimental setup of the AdaSpark paper ([latex/AdaSpark.tex](latex/AdaSpark.tex)): the engine, the baseline, the targets and the
machine; the prompts; the arms and the metric; and how to reproduce every table and figure. Section, table,
figure and equation numbers refer to the paper.

## Engine, baseline, targets and hardware

AdaSpark is implemented in imparo, an inference engine with Metal and CUDA backends. A verify is one batched forward over the tree's nodes, with a per-row ancestor mask in attention; the widest verify it runs is 64 rows. The two LFM2.5 targets are hybrids whose blocks mix attention with a short convolution. A tree over such a layer needs per-row convolution state rather than a mask, and the engine carries it.

All measurements were taken on one Apple M3 Pro (12 CPU cores, 18 GPU cores, 36 GB of unified memory with 150 GB/s nominal bandwidth, macOS 15.7.9) with imparo's Metal backend. The step boundaries in particular are a property of that machine's kernels: we expect the shape (nearly flat classes, steps, per-class slopes in context) to transfer to another device and the breakpoints not to, but have not verified this on a second device.

The baseline is llama.cpp ([github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)), build 11201, commit `2145525a4`, run on the same machine with the same GGUF files, with `--spec-type draft-dspark` and the same drafter file, at its defaults: at most three draft tokens per round (`--spec-draft-n-max 3`) and no minimum draft probability (`--spec-draft-p-min 0`). Its verification is a chain. Both engines run with a 32,768-token context and serve one request at a time.

The table below lists the four targets and their drafters: three dense targets in two families and two weight formats, and one mixture-of-experts target that shares its family with a dense one.

**Targets and drafters** (Qwen Team, 2025; Liquid AI, 2026). The weights are GGUF files (llama.cpp's model file format), which imparo and the llama.cpp baseline both read. The token mixer of the LFM2.5 targets combines short convolutions (conv.) with attention. Q8_0 stores 8-bit weights and Q4_K_M mostly 4-bit weights, both in small blocks with a scale per block. A published drafter is the developer's GGUF file, loaded as released; a converted one is made from a Hugging Face checkpoint (below). Each drafter uses its target's weight format. *Block* is the drafter's block size *B*.

| target | weights | token mixer | feed-forward | size | drafter | block |
|---|---|---|---|---|---|---|
| LFM2.5-2.6B | Q8_0 | conv. + attention | dense | 2.87 GB | published | 9 |
| LFM2.5-8B-A1B | Q4_K_M | conv. + attention | 32 experts, 4 active | 5.16 GB | published | 9 |
| Qwen3-4B | Q4_K_M | attention | dense | 2.50 GB | converted | 7 |
| Qwen3-8B | Q4_K_M | attention | dense | 5.03 GB | converted | 7 |

**Where the files come from.** Every file is on Hugging Face. A link pins the repository revision we used; the SHA-256 is of the file both engines loaded.

| file | Hugging Face | SHA-256 |
|---|---|---|
| LFM2.5-2.6B target | [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/blob/b421ad1d549afeda6a0fb2ad3a697cb5a7879adc/LFM2.5-2.6B-Q8_0.gguf), `LFM2.5-2.6B-Q8_0.gguf` at revision `b421ad1d` | `36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757` |
| LFM2.5-2.6B drafter | [LiquidAI/LFM2.5-2.6B-DSpark-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF/blob/7bc2896af56d82ccc7e156800197408db464d63b/LFM2.5-2.6B-DSpark-Q8_0.gguf), `LFM2.5-2.6B-DSpark-Q8_0.gguf` | `85a98fafdaf1328b6876fd1360d7ed69e74c6cefc14dd07fb6306e1940386c87` |
| LFM2.5-8B-A1B target | [LiquidAI/LFM2.5-8B-A1B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/blob/49c14831707011e64d70b2ebd8462ba08d608434/LFM2.5-8B-A1B-Q4_K_M.gguf), `LFM2.5-8B-A1B-Q4_K_M.gguf` | `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0` |
| LFM2.5-8B-A1B drafter | [LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF/blob/7ba04ee5ff05a4baf2681fe5ddda6d736581ccdf/LFM2.5-8B-A1B-DSpark-Q4_K_M.gguf), `LFM2.5-8B-A1B-DSpark-Q4_K_M.gguf` | `017278ec4409671890f2722a690a52dba59f5f02eff604f0c1e79cf52f7c4ae8` |
| Qwen3-4B target | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/22c9fc8a8c7700b76a1789366280a6a5a1ad1120/Qwen3-4B-Q4_K_M.gguf), `Qwen3-4B-Q4_K_M.gguf` | `f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a` |
| Qwen3-4B drafter | [deepseek-ai/dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7/tree/3457dff1417cb84927f6098a5fcb7cee85c934b7), converted | `df42b34eab26698fdc24e6dd20319a19147ccc3eb82b1141648bb0361c4eecbe` |
| Qwen3-8B target | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf), `Qwen3-8B-Q4_K_M.gguf` | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |
| Qwen3-8B drafter | [deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7/tree/03326e5043815da1f81b109078b2889737c26017), converted | `ed0dd07229286e344a88fb69b03d789fd4c72d516a79dd3e0513388d201d140d` |

The LFM2.5-2.6B repository replaced its `Q8_0` file on 2026-08-19; the file on its main branch now has a different hash, so use the revision link.

**Why the Qwen3 drafters are converted.** Liquid AI publishes its drafters as GGUF files, which both engines load as released. DeepSeek publishes its Qwen3 drafters only as Hugging Face checkpoints (`model.safetensors` in bf16, with a `config.json` that declares `Qwen3DSparkModel`), and neither engine reads that format. llama.cpp's `convert_hf_to_gguf.py` has a converter for this architecture. It writes the drafter as a GGUF file of the architecture both engines' DSpark loaders read (`dflash` in their sources), takes the tokenizer from the target's checkpoint (`--target-model-dir`, [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) or [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)), and drops the checkpoint's token embedding and output head, since the drafter uses the target's (the 8B checkpoint carries a copy of Qwen3-8B's output head; we checked that the two are the same). We write it as F16 and quantise it to Q4_K_M with `llama-quantize`, the target's format:

```
python convert_hf_to_gguf.py <drafter checkpoint> --target-model-dir <target checkpoint> \
    --outtype f16 --outfile Qwen3-8B-DSpark-F16.gguf
llama-quantize Qwen3-8B-DSpark-F16.gguf Qwen3-8B-DSpark-Q4_K_M.gguf Q4_K_M
```

A different version of the converter or the quantiser may write different bytes from the SHA-256 above; we have not checked.

## Test set

The test set (six public sources, the selection rule, the evaluation subset and the departures from
the sources' protocols) is described in the paper's Appendix A; `testset/build.py` rebuilds it.

## Microbenchmark and development prompts

The measurements of §3.2 and §3.4 continue text at a fixed context rather than hold a conversation: a prompt is token ids with no chat template, at 443, 1,596 or 8,444 tokens. The text is LongBench's government reports, news articles and source code, taken whole in the data file's order until the context is reached and then cut to its first tokens, one prompt set per vocabulary. No text is repeated, since repeated text is trivially predictable and would favour every drafter.

**Development corpus.** The constants of §3 and the design comparisons of §3.3, §3.4 and §3.6 used a separate corpus of 489 prompts from 17 public suites: HumanEval, GSM8K, Alpaca, MT-Bench, Natural Questions, CNN/DailyMail and Alpaca-zh at contexts of 128 to 2,048 tokens, and ten LongBench subsets at 1,024 to 16,384 tokens. It is split by suite and context band into 408 prompts for fitting and 81 held out, and a 16-prompt subset takes two held-out prompts per context band. Long items are cut from the middle and short ones reach a long band by packing distinct items; no text is repeated. The corpus shares two sources with the test set, MT-Bench (inside Spec-Bench) and LongBench; we did not check the two for shared items.

## Arms and metric

Each speculative arm serves the evaluation subset in the order of the test-set table (the paper's Appendix A) on one server. Every imparo arm starts from an empty learned store and learns only from its own requests, so its models see the workload change five times. Two arms separate the engine from the scheduler: imparo runs DSpark in llama.cpp's configuration, so comparing the two engines holds the algorithm fixed, and comparing that arm with AdaSpark holds the engine fixed. The arms are:

- **autoregressive**, in each engine: no drafter.
- **llama.cpp DSpark**: the baseline above, a chain of at most three draft tokens per round.
- **imparo DSpark**: the same configuration in imparo: a chain of at most three of the drafter's picks, no minimum probability, no tree, no learned models, no n-gram. The chain is verified on the same route as AdaSpark's trees, laid out as a tree with one child per node, so that the two arms differ only in scheduling.
- **full-block chain**: the drafter's whole block (9 picks for LFM2.5, 7 for Qwen3) verified as a chain.
- **fixed tree**: a 16-row tree, best-first on the head-based estimate; no learned models.
- **cost model**: the width chosen by Eq. (7) with the online cost model; acceptance from the head-based estimate; no n-gram.
- **AdaSpark**: the cost model, the acceptance model and n-gram candidates together.
- **AdaSpark at a pinned width** (8, 12 or 16 rows, and on the routed target also 4, 5 and 6; Table 3): AdaSpark with its width fixed by `IMPARO_DSPARK_TREE=N`. On the dense targets a verify of 2 to 8 rows takes within 4% of the 8-row time, so a narrower pinned width would give up tokens for little saving and is not run there. The acceptance model, its per-request offset and the n-gram candidates run as in AdaSpark, from an empty store, so the arm differs from AdaSpark only in how the width is set.

The engine's batch path for causal rows does not split the keys across threadgroups as the tree path does: at 8,444 tokens of context an 8-row chain on it takes 74.7 ms on Qwen3-4B against 35.0 ms for an 8-row tree, so a comparison with a chain on that path would credit the scheduler with a faster kernel. The three-pick chain gives the same text on both routes in 10 of 11 replies of the short subset on each of LFM2.5-2.6B and Qwen3-4B; we expect, but have not traced, that each differing reply is a near tie that the key split's summation order resolves differently.

The autoregressive arms, the two DSpark arms and AdaSpark run on all four targets. The ablation arms run on LFM2.5-2.6B and LFM2.5-8B-A1B, the dense and routed members of one family. The autoregressive arms measure only each engine's autoregressive speed, the denominator of its speedup over itself (Table 4); they run on a *short subset* fixed the same way: the first two turns of the first conversation of each source (11 timed replies). Table 7, Figure 7 to Figure 10 and §4.6 come from a separate, instrumented run of AdaSpark and the cost-model arm on the short subset with per-round logging on; the logging costs time, so those runs are not used for speed. Replies that reached the 8,192-token cap are left out of every statistic, per-round and per-request ones included, as they are out of the speed ratios.

The metric is decode speed on the client's clock: generated tokens over the time from the first to the last token of each reply. A reply that arrives as a single streamed delta, such as a bare tool call, has no such interval and is left out of the ratios. llama.cpp builds a tool-call parser from the Qwen3 chat template and refuses a request whose tool schema uses a type that JSON Schema does not define; ToolACE's schemas name Python's types, which the set builder renames (the test-set table (the paper's Appendix A)), so every engine serves every conversation. A comparison between two arms is the geometric mean over replies of the per-reply ratio. Replies within one conversation share their history and their arm's learned state, so they are not independent; the 95% interval is a percentile bootstrap that resamples whole conversations (2,000 resamples).

## Reproduction

The test set, its builder, the harness and the scripts that produce every table and figure are in the `dspark-paper/` directory of the engine's repository. `testset/build.py` rebuilds the conversation sets from the pinned source revisions and `testset/context.py` rebuilds the microbenchmark prompts. `harness/run.sh` runs, per target, the short subset with every arm, which must pass automatic checks; then the speculative arms on the evaluation subset, which must pass the same checks; then the microbenchmark of fixed widths that Figure 5 reads, the cost survey (Table 1, Figure 3) and the instrumented run for the figures. `harness/run_chain.sh` measures the chain arms. `harness/run_learned.sh` measures every arm that runs the cost model, whose rows are the ones reported for those arms: per target, AdaSpark, the cost-model arm on the LFM2.5 pair and the three pinned widths on the evaluation subset (Table 3), then AdaSpark a second time on the same conversations, after the pinned arms; on the routed target a second such stage with the widths pinned to 4, 5 and 6 rows; Table 3 compares each pinned arm with the mean of the two AdaSpark runs of its stage; then the instrumented run, the held-out conversations and the ToolACE conversations. `harness/run_heldout.sh` and `harness/run_toolace.sh` run the other arms on the held-out and the ToolACE conversations. Each run log records the checksums of the binaries it ran, and the harness reuses a completed stage only if it was measured on the same server binary.

The ToolACE conversations were run by every arm of the evaluation run, over all eight in one run per arm, and those rows replace every arm's rows of the six conversations whose schemas the type rename of the test-set table (the paper's Appendix A) changed. Each arm of that run starts from an empty store and serves ToolACE first, where in the evaluation run it had already served four other sources. On the two conversations the rename did not change, AdaSpark ran 4–7% slower in that run than in the evaluation run, and llama.cpp's DSpark within 0.6% on three targets and 6% faster on LFM2.5-2.6B; we did not test the store as the cause, but the replaced rows, if anything, understate AdaSpark.

Table 9's kernel classes are not part of the released engine, which learns its widths only. `harness/run_classes.sh` runs it on a measurement build: the engine with `harness/kernel_classes.patch` applied, which restores the kernel classes, built into its own directory, where `IMPARO_DSPARK_CLASSES=declared` prices the 8-bit kernel's row classes and the default learns widths as the released engine does. Per target it runs AdaSpark with learned widths, AdaSpark on the kernel classes, and AdaSpark with learned widths again, on the evaluation subset. The width-invariance check of §3.5 is `imparo-forward --decode-vs-verify`.

The paper is `latex/AdaSpark.tex`, with its references in `latex/refs.bib`. Its generated tables and figures are in `latex/gen/`, written from the run data by `harness/paper_figs.py RUNDIR`, which builds each one as an HTML fragment and passes it to `harness/latex_assets.py`: a table becomes booktabs LaTeX, a figure an SVG printed to PDF by Chrome with embedded TrueType fonts, named `fig-<name>-img.pdf` because arXiv deletes a PDF that shares its name with a `.tex` file. `latex/gen/` is tracked, since the run data is not. Tables 11-15 are written in `AdaSpark.tex` and Figures 1-2 are drawn by hand in `latex/figures/*.svg`; `latex_assets.py` renders each SVG to the PDF beside it when the SVG changes.

`harness/build_latex.py` compiles the LaTeX paper with pdfLaTeX and BibTeX (from `$TEXBIN`, else a TinyTeX installation in `~/Library/TinyTeX`, else the `PATH`) into `AdaSpark.pdf`, fails on any undefined reference or citation, overfull line, BibTeX warning, unembedded or Type 3 font, and packs the arXiv source bundle `latex/arxiv-source.tar.gz` (the source, its `.bbl`, `gen/` and the figure PDFs), which it then compiles alone in an empty directory, as arXiv does, and compares page counts; it also refuses a bundle in which a PDF shares its name with a `.tex` file. The packages it needs beyond TinyTeX's base are newtx, txfonts, fontaxes, xpatch, makecell, multirow, colortbl, algorithms, algorithmicx, enumitem and caption.
