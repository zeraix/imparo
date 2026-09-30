# DSpark on Apple GPUs: paper manifest

A working list of what we can claim, what backs it, and what is still missing. Every number
here is measured on this machine (Apple M3 Pro, 12 cores, macOS 15.7.9) unless it says
otherwise; nothing in this file is an estimate presented as a result.

Status keys: **SHIPPED** (in the engine, gated), **MEASURED** (a number exists),
**OPEN** (not built).

---

## 1. The claim we are actually in a position to make

> Speculative decoding on an Apple GPU, where the verify is BIT-IDENTICAL to plain decode,
> the verify's cost turns out to be a STEP FUNCTION IN ROWS WITH AN INVERSION and AFFINE IN
> CONTEXT WITH A DIFFERENT SLOPE PER CLASS -- which no linear cost model can express -- and
> the tree is sized from that, on a hybrid recurrent/attention model that almost no
> speculative-decoding work handles.

The three parts of that sentence are the three things to defend.

THE EMPHASIS MOVED on 2026-09-16, deliberately. It was going to be "we learn the cost online";
the literature check found that is substantially CAST's claim (2.7a). The measurement is what
is ours; the estimator is how we get it.

---

## 2. What is done

### 2.1 Chain verify, then tree verify (SHIPPED)

- One masked attention entry serves the drafter's block, a tree, and a chain, at every head
  dim the models use (64 / 128 / 256). One 64-bit mask per row; positions per row; the entry
  must not depend on where the batch starts in a tile. Rejected: a tree kernel per
  architecture / head dim / node count, which is what the CUDA lab has.
- Rows go depth-first, most likely child first, so an accepted prefix of the drafter's chain
  needs no KV copy; only rows past a branch point are moved.
- **Tasks #241-#248. Design sections 6.4, 6.5, 7.**

### 2.2 The verify is bit-identical (SHIPPED)

- The tree path produces the same logits as the chain path. `det_gate` pins EXACT at
  128..16384 on LFM2 and E4B, decode hashes equal.
- This is stronger than the usual "distribution-preserving" claim and is rarely shown on a
  real engine. It is what lets us say speculation costs nothing in quality, full stop.
- **Memory `verify-rows-near-ties-lfm2`; the near-tie rows and the gate order are recorded.**

### 2.3 Recurrent layers verify and commit as a tree (SHIPPED)

- LFM2's short convolution and Qwen3.8's gated delta rule are not attention: a tree forward
  over them has to carry per-row state. The gated-delta tree verify + commit is a REGISTER
  WALK, bit-identical, and 11-18x faster than the masked-solve form we first built.
- Almost no speculative-decoding work handles recurrent / SSM layers at all. Both models we
  serve are hybrids, so this is not optional for us and is a real contribution.
- **Tasks #243, #245, #246. Memory `gdn-tree-verify-register-walk`.**

### 2.4 The drafter reads the target's weights and activations (SHIPPED)

- No separate draft model resident in memory; the drafter's forward taps the target's
  residual stream. One-row Q8 projections take the simdgroups their blocks need (W2 0.78 ->
  0.26 ms, DSpark decode -7.05%, same bits).
- **Tasks #250-#253. Design 5.3.**

### 2.5 Candidate trees, best-first on Q (SHIPPED)

- Candidates are the chain's picks and the siblings of each biased column; the tree is
  best-first on Q, the product of q along a node's path. Every prefix of that order is
  itself a tree, which is what makes a chosen width well defined.
- **Task #255: N=16 read -12.3% ms/token against the chain.**

### 2.6 The verify's cost is a STEP function, and we measured it (MEASURED, SHIPPED)

This is the systems finding most papers get wrong by assumption.

```
n      T(443)   T(1596)  T(8444)      the tile the GEMM picks
2..8   28.9     29.2     31.0         64x8    1 walk
9..16  32.4     32.9     35.6         64x16   1 walk
17..32 44.6     45.3     50.7         64x32   1 walk
33..47 79.5     80.1     85.1         64x64   1 walk
48..64 67.8     69.1     74.7         seat    2 walks
```

- Flat inside a tile to under 1 ms on 29-45, and NOT rising with n inside it: a verify pays
  for the tile it lands in, not for its rows.
- 40 rows cost MORE than 48. A wider verify is sometimes cheaper, which no linear or convex
  cost model can express.
- Context moves it by +7-14% and cannot reorder the classes.
- **78 engine runs, 17 min 43 s. Evidence `2026-09-16-step4c-verify-cost-table.md`.**

### 2.7 The cost table is learned online, by the engine (SHIPPED)

- The tuner cannot do it: `imparo-tune` never loads a model's tensors, so it has no verify to
  run. The engine measures T from the rounds it runs anyway -- the probe IS the verify, so it
  produces the round's tokens whatever width it runs at.
- The engine learns WHERE the cost steps as well: a ladder of widths (2, 4, 8, ... 64), split
  only where a width inside an interval could be chosen. It started as a backend declaration
  (`matmat_row_class`); only the 8-bit kernel declared one, and the learned widths replaced it
  (Table 11, 2026-09-29).
- **Cost is a FUNCTION of the context, not a number**, and the classes do not move together:
  `T_k(c) = a_k + b_k*c` to 0.25% across a 19x range, with `b` growing 4x from 8 rows to 64.
  A table of constants cannot express that, and the error is systematic. The engine learns it
  with online elastic-net QUANTILE regression (pinball loss, because GPU interference is
  one-sided), FTRL-Proximal, 399 ns per round, on three timescales: the context law (slow,
  a kernel property), each class's own residual (fast), and the machine's drift measured on
  the one thing every round pays.
- Stored beside the device profile. **Not a convenience**: a 256-token answer moves an
  8444-token context by 3%, so the slope is not identifiable inside one request at all.
- **Evidence `2026-09-16-online-cost-law.md`.** Read section 6 before writing about this.

### 2.7a WHAT THIS IS NOT (read before framing the paper)

Checked 2026-09-16, CAST read in full 2026-09-16. Learned cost models for kernels are
established but OFFLINE (TVM/Ansor/Halide/TpuGraphs). Online-learned latency models inside a
serving loop already ship: llm-d trains XGBoost in real time on a sliding window; OmniPilot
already uses quantile loss for the same multiplicative-error reason.

**"We learn the verify cost online and choose the tree from it" is substantially CAST's
claim** (arXiv 2510.26577, ICLR 2026). Do not frame the paper around the estimator.

What CAST actually does, from the paper rather than a summary:

- Its cost model is a **precomputed lookup table of measured wall-clock times**,
  `f(B, c, n)` over batch size, a bucketed context index, and tokens fed. Profiled per device
  and per batch size, ahead of time. Appendix E names the precompute as the method's own
  limitation.
- **Nothing is fitted online.** Its only runtime state is a FIFO mean of drafter
  confidence-gain ratios, used for depth pruning -- not a cost model.
- Its context axis exists but is never measured: **no values of the bucket width or count, no
  slope, no figure, no statement that cost rises with context.** In the expansion stage the
  index is the accumulated draft-node count, not the prompt at all.
- It uses **raw draft softmax path-products as acceptance proxies, uncalibrated**. Three
  hand-set thresholds (4 / 3 / 2.5) are retuned per model, which is where the miscalibration
  is absorbed.
- A800 / H20 / RTX 4090; six dense Transformers; distribution-preserving verification with no
  output-equality test; context lengths never stated.
- **Sequoia is not cited** -- all 38 references checked. CAST offers no positioning language
  against the one prior method that also profiles hardware to size trees.

So what stays ours:

1. **The cost STRUCTURE.** A step function in rows from GEMM tile padding, WITH AN INVERSION
   (40 rows cost more than 48), and affine in context with a different slope per row class
   (4x from 8 rows to 64). CAST's Algorithm 1 takes "Arrays u[1..n], c[1..n] strictly
   increasing" as a precondition and Theorem 4.1's closed form is affine -- an inversion
   makes its marginal-ratio denominator negative and flips the test's sense. It cannot
   represent this and reports nothing like it.
2. **What the inversion costs an engine that does not measure.** A hand-set top-k of 10 or 12
   is DOMINATED on this GPU: +12.1% / +5.2% at ctx 1596, +8.6% / +6.5% at 8444, while
   drafting fewer tokens than the cheaper 16. This is the envelope's win and it is the number
   to lead with -- not per-round adaptation, which is worth 0.19% (2.8).
3. **Online elastic-net quantile regression** (pinball, one-sided GPU interference),
   two-timescale, persisted across requests. CAST is offline and says so.
4. **Bit-identical verification on a hybrid recurrent/attention model.** CAST claims lossless
   only in the distribution-preserving sense and never tests output identity; no SSM/hybrid.
5. **Apple GPU / Metal.**
6. **Censored feedback as a structural property** of adaptive tree sizing (2.8). Not addressed
   by CAST, Sequoia or AdaptiveSD.

**Do not claim calibration as a distinction.** CAST does not calibrate -- but neither do we,
and we measured that we do not need to: our Q is already calibrated (2.8).

### 2.7c THE SUBJECT IS TEST-TIME TRAINING; THE ESTIMATOR IS ITS FIRST PART

**The paper is about TEST-TIME TRAINING -- the four lifetimes of design 11.2 -- not about the
cost estimator.** The estimator is one piece of it: the part that learns, from the traffic the
engine is already serving, HOW MANY TREE NODES TO VERIFY. Framed that way it is level 0/1 of
the same programme that continues into the dictionary (level 2) and the drafter's own state
(level 3), and it is the first piece SHIPPED.

Writing it as "an online cost model" invites the CAST comparison of 2.7a and loses the frame.
Writing it as "the first lifetime of test-time training, the one that picks the tree" keeps it,
and makes the negative results below part of a programme rather than a failed feature.

**Then, within that part: state its value as REACHING the best width cheaply. Not BEATING the
best width.** The second claim is the one the measurements do not support, and chasing it has cost
this project more than once.

The point is that the best width is not a constant of the method:

```
"16" is a property of   the Q8 GEMM tile boundary on an M3 Pro, for Q8_0_TM weights, at this
                        drafter's block size, for this model -- with st_gemm_narrow_max TUNED
                        and an inversion inside it (40 rows 79.5 ms, 48 rows 67.8)
change any of           device, weight kind, drafter block, model
and it moves, with no way to know where without measuring.
```

So a shipped constant is a shipped *measurement of someone else's machine*. The question a
serving engine actually faces is not "is 16 beatable" but "what is the 16 here, and what does
finding it cost".

**What finding it costs offline.** A sweep over widths x contexts x prompts, per device, per
quant, per drafter. On this corpus, computing the hindsight-best constant took 81 held-out
prompts x 3 fixed widths = 243 runs, about 40 minutes of exclusive GPU, and it answers for one
device, one quant, one model. CAST's Appendix E names precisely this precompute as its own
limitation.

**What it costs online.** Nothing. The table is fitted from the rounds the engine was going to
run anyway and persists across requests. The evidence that it converges to the right answer is
that it reproduced an independent 78-run survey without seeing it: learned bar 1.0849 -> 1.1230
against 1.0935 -> 1.1218 predicted, across 128..16384.

**So the benchmark is not budget-vs-best-fixed.** It is two questions:

```
agreement    does the ONLINE-derived constant equal the HINDSIGHT-best constant?
             on this corpus the hindsight answer is 16 at six of eight bands, 8 at 8192/16384.
cost         online: zero extra runs.  offline: 243 runs per (device, quant, model).
```

The agreement question is FALSIFIABLE AND NOT YET RUN. It is the experiment the paper needs,
and it is independent of whether per-round adaptation ever pays.

**Per-round adaptation is a separate, weaker claim.** It is worth 0.19% where it wins (2.8) and
on the public corpus it loses to the best constant. Do not lead with it, do not let it stand in
for the estimator's value, and do not let its failure be read as the estimator failing -- the
cost side and the per-round controller are different claims resting on different evidence.

### 2.7b THE ESTIMATOR, STATED (for the methods section)

**The problem class.** Not reinforcement learning, and saying so precisely matters because it
determines what machinery is needed.

```
  RL              action changes the state; return is delayed; credit must be assigned
  contextual      action changes only THIS round's payoff; payoff observed immediately
  bandit
  this problem    one round's payoff is observed immediately, but the action DOES move the
                  state: the width decides how far the round advances, so where the next
                  round starts -- an average-reward semi-Markov problem
```

One round is one action (a width) with one immediately observed outcome: `(rows -> ms)` from
the clock and `(rows -> accepted)` from the verify. The exact average-reward rule adds a
relative-value term for the state the width leads to (Puterman 1994); the design drops it, and
the measured lag-1 autocorrelation of per-round acceptance (+0.055 / +0.285) makes that error
second order. Nothing is bootstrapped or backpropagated.

But it is a bandit with an unusual feedback shape, and THAT is the structural contribution:

```
  ordinary bandit      pull arm k          -> learn arm k's payoff, nothing else
  this problem         run at n rows       -> learn accept(m) for EVERY m <= n
                                           -> learn NOTHING about any m > n
```

Full information below the chosen action, zero above it: **one-sided (right) censoring**. It
follows from the prefix property -- every prefix of the best-first order is itself a valid
tree -- so one wide round yields the counterfactual payoff of every narrower width for free.

Two consequences the policy must respect, and both are measured:

- **Run wide to learn.** A policy that settles narrow can never observe the evidence that
  would widen it. The incumbent therefore seeds at the WIDEST offered width, not at the
  argmax of one noisy round -- that seeding defect cost the policy its floor (commit
  6d60d3de).
- **Exploration is not free.** Each observation costs exactly what it measures: a real verify
  at a real width. This is the one place the RL intuition transfers.

**There is a SECOND censoring, on the price rather than the value, and it has the opposite
cure.** `T_k(c)` is measured only at the (class, context) pairs the policy actually ran, so a
class that runs at one end of the range gets a law whose slope is noise, held at zero only when
the noise is negative (the >= 0 clamp; no span check gates the fit) -- which the budget then
applies across the whole range.

```
  accept(m)   censored ABOVE the width that ran        -> the cure is to run WIDE
  T_k(c)      censored to the (class, ctx) pairs run   -> the cure is to run VARIED
```

**No explorer is needed for it, and that is a measurement, not a hope.** Mid-training, eight
rows had 493 observations all at ctx 12289..16425 and priced 11.1% high at 128; a hundred
prompts later it spanned 129..16427 and priced 1.2% high. The overprice makes the narrow class
TIE with the wider one rather than excluding it, so it still runs sometimes and each run widens
the span. Traffic whose context varies breaks this without any policy change -- so the corpus
is the cure, and a deliberate explorer stays a design note rather than a build. It is warranted
only for classes the envelope NEVER offers, which stay frozen at three observations for good.
Here those are frozen at prices that correctly exclude them; nothing guarantees that on another
device. Design: `speculator-design.md` 9.1.

**The cost objective.** Per row class k, over weights w in R^5:

```
  min_w  E[ rho_tau( T - w . phi(u) ) ]  +  L1*||w||_1  +  L2*||w||^2

    phi(u) = [1, u, u^2, sqrt(u), ln(1+u)]        u = ctx / 8192
    rho_tau(r) = r * (tau - 1[r < 0])             tau low
```

Three choices, each forced by a measured property rather than taste:

- **Pinball, not squared.** GPU interference is ONE-SIDED -- a browser on the GPU, a cold
  pipeline, a thermal drop make a round slower and nothing makes it faster. Squared loss fits
  the mean of a right-skewed distribution and ratchets upward forever. Pinball at a low
  quantile fits the lower envelope.
- **A redundant basis.** Affine in context fits this model on this machine to 0.25%, but that
  is a fit, not a law: another kernel regime (a flash-decoding slice boundary, a paged-KV page
  count, an attention route change past some key count) can put curvature in it. The
  regulariser decides which terms earn their place instead of the author asserting the shape.
- **Elastic net, not lasso.** Over a short run the context barely moves -- a 256-token answer
  spans 3% of an 8k context -- and the basis terms go nearly collinear. Pure L1 picks one of a
  collinear group arbitrarily and flips between them, which makes the price jump for no
  physical reason. L2 shares the weight and holds it still. L1 does NOT zero a slope that is
  updated every round: its |z| grows like the gradient's scale times sqrt(rounds) and passes
  L1 on the first fold. A class that has only ever run at ONE context prices as a constant
  because the >= 0 clamp catches the slope noise's negative side, not because of L1.

**The solver.** FTRL-Proximal (McMahan et al.), per observation, O(d) flops and O(d) state --
no matrix, no inversion, nothing that grows with the number of rounds:

```
  g_i  = -phi_i * (tau - 1[r < 0])
  n_i += g_i^2
  z_i += g_i - (sqrt(n_i') - sqrt(n_i))/ALPHA * w_i
  w_i  = 0                                                      if |z_i| <= L1
       = -(z_i - sgn(z_i)*L1) / ((BETA + sqrt(n_i))/ALPHA + L2)  otherwise
```

The weights are DERIVED from `(z, n)` rather than stored, so a coordinate reads exactly zero
while its |z| is under L1 -- lasting for a rarely visited feature, not for one visited every
round. A standardized rule, zero unless |z_i| > c sqrt(n_i), is what makes "zero until the
evidence says otherwise" true for the latter. The per-coordinate adaptive rate is what lets an
intercept of ~30 and a feature of ~1 train under one update.

**The value side.** The budget compares widths by `1 + S(n)`, and what the choice turns on is
the MARGINAL `S(hi) - S(lo)`, which is mis-scaled. That scale is itself a function of context
and CHANGES SIGN across the range -- 1.69x at ctx 1596, 0.885x at 8444 -- so a pooled scalar
averages to ~1.0, which is the identity, and corrects nothing at either end. A trained session
measured exactly that: g = 1.032 over 888 wide rounds. So the gain is a line, fitted by the
same kind of online least squares, ridge-shrunk toward "no correction":

```
  actual marginal  ~  predicted marginal * g(u),   g(u) = a + b*(u - 0.5)
```

carried in five decayed scalars (a 2x2 normal system), solved where it is read.

**The schedule.** Three rates, and only one of them is on the hot path:

```
  per round (~11 ms)   set_context(ctx) -> price every class -> pick n -> verify -> commit
                       -> one FTRL step                                        399 ns
                       -> the gain's five scalars, from the accepted walk
                       -> push (ctx, ms) into a 128-sample ring

  per request          refit(): per law, train a candidate FROM SCRATCH, 24 epochs over the
                       ring's first 96; score incumbent and candidate on the last 32 the
                       candidate never saw; adopt ONLY if it wins.      0.175 ms / law

  per process          seed anchors -> laws -> gain from the stored table. Arrives trained.
```

Persistence is **not a convenience**. A 256-token answer moves an 8444-token context by 3%,
so the slope is not identifiable inside one request at all.

Two honest notes for the methods section:

- **The batch refit's benefit is not established.** A joint fit over a window is the better
  estimator in principle and the held-out gate means adopting one can never make the table
  worse -- but on clean data with enough samples the online pass already lands within 4% of
  the truth and the refit is no closer. It is there because the machinery is sound, cheap and
  gated, not because a measurement asked for it.
- **Asynchrony is not warranted at this size** and should not be claimed. 399 ns on an 11 ms
  round is 0.004%; the whole refit is ~1.2 ms once per multi-second request. A background
  thread would buy nothing measurable and would add a race on the table. The shape is already
  right for it if the fit ever grows -- the refit produces a CANDIDATE adopted only after it
  wins on held-out data, so it can move off the hot path and swap in atomically.

### 2.8 The row count is chosen per round (SHIPPED)

```
n* = argmax over the measured widths of (1 + S(n)) / (D + T(n))
```

```
ctx     chain    fixed 16   budget    vs the best fixed width
443     11.131   10.271      9.782     -4.8%
1596    13.042   10.592     10.506     -0.8%
8444    14.629   12.123     12.132     +0.1%  (inside the spread)
```

- D, the round's n-independent cost, belongs in the denominator and the design first had it
  out. Built without it the chooser takes 8 rows in 50 of 64 rounds at 1596 and LOSES 2.2% to
  a fixed 16. That correction is worth writing up: it is a sunk-cost error that is easy to
  make and that measurement caught.
- **Evidence `2026-09-16-step4d-budget.md`.**
- **What the per-round choice is WORTH, measured 2026-09-16: 0.19%.** Six probe runs, 244
  rounds. The same context measured five times gives a bar stable to four decimals
  (1.1051..1.1059) and a value ratio that straddles it (1.0977..1.1314), so the winner flips
  between runs. Averaged over the six, always-16 loses 0.19% to perfect adaptation and
  always-8 loses 1.79%. Rows 8..15 are worth 0.35..0.61 accepted tokens and cost 3.7..4.6 ms
  = 0.34..0.42 tokens: marginal value equals marginal cost to within the noise.
- **Three independent tests say no pre-verify statistic does better.** Q is calibrated
  (recalibration buys 1.5% log loss at 1596, none at 8444); the average-reward gain rule ties
  the per-round ratio rule; sweeping the decision bar gives a best of -0.15% / -1.26% with
  the optimum in OPPOSITE directions at the two contexts. The oracle's -4.28% is clairvoyance.
- **The feedback is censored.** A round at n rows reveals accept(m) for every m <= n and
  nothing above it, so a learner locks on its seed (8 in 56/56 rounds, 16 in 47/47). Stepping
  down is free; stepping up costs an explore round. This is a real property of adaptive tree
  sizing and we have not found it named in the prior art.
- **Evidence `2026-09-16-value-model-and-the-width-prize.md`.**

### 2.8a THE BENCHMARK RULE (without it the width result is worthless)

A width policy can be made to "win" by choosing the test mix. Measured on the held-out set:

```
  category        8      10      12      16   best fixed
      code    11.15   11.95   11.64   11.45   8
    config    11.04   11.26   11.07   10.80   16
  dialogue     7.59    7.57    7.53    7.25   16
     prose    14.71   15.51   15.09   14.11   16
 technical    10.50   10.83   10.67   10.41   16
```

Weight the mix toward code and fixed 8 becomes the best constant; balance it and fixed 16
does. So "the budget beats fixed 16" is not a claim -- it is a statement about the mix.

**THE BASELINE IS `min` OVER FIXED WIDTHS, COMPUTED ON THE SAME MIX, WITH HINDSIGHT.** That
is a strictly harder baseline than anything shippable -- no engine can know in advance which
constant its users' mix will favour -- and it cannot be gamed by reweighting, because
reweighting only changes which constant the policy must beat.

And the same table is why adaptation has any value at all: THE CATEGORIES DISAGREE. If every
prompt wanted 16, no policy could beat the constant and the honest answer would be to ship
16. The test mix must therefore contain prompts that disagree, and the baseline must be the
best single constant for the whole of it. Both conditions, or the number means nothing.

Acceptance is what drives the disagreement -- measured over the 80 training prompts of the
first corpus, accepted tokens per round:

```
  dialogue 7.878   technical 4.664   config 4.191   prose 4.095   code 3.700
```

Low acceptance wants 8 rows, high acceptance wants 16, and that ordering matches the table
above exactly.

THAT CORPUS IS RETIRED, for a reason worth stating. Its `dialogue` prompts cycled five canned
turns, so the model was continuing text already in its own context and 7.878 is inflated by
the corpus construction, not by the workload. Whoever writes the prompts chooses the answer.
The replacement is section 2.8b.

### 2.8c WHAT THE RESULT DECIDES (write this down before reading it)

Two numbers come out of the held-out test, and the pair decides the next step. Fixing the rule
in advance is what stops a disappointing number from being re-read as a reason to keep going.

```
  budget vs best FIXED width, min over the mix with hindsight     does it win?
  budget vs a PER-PROMPT ORACLE over those widths                 was there anything to win?

  wins by enough                          ship it; add nothing
  ties, ceiling ~0                        FINISHED, not failed -- no policy can win on this
                                          mix, and the honest answer is to ship the constant
  ties or wins thinly, ceiling real and
  largely uncaptured                      the value model needs more than context: design 11.8
```

The middle row is the one to guard. A tie with no headroom is a result, and reporting it as a
shortfall would be the same error as reporting a win on a mix chosen to produce one.

### 2.8b THE BENCHMARK CORPUS: public datasets only

A claim about a MIX has to be checked on a mix nobody involved controls. So every prompt now
comes from a published benchmark, and the construction is a sampling procedure with a seed
rather than an author's judgement.

```
  CAST's own list      MT-Bench, HumanEval, GSM8K, Alpaca, CNN/DailyMail, Natural Questions
  long context         LongBench (Bai et al. 2023) -- the standard long-context suite
  Chinese              LongBench's Chinese subsets + alpaca-zh for short prompts
```

Seventeen suites, 489 prompts, stratified 80/20 into 408 train and 81 held out:

```
  suite            lang  kind      bands
  humaneval        en    code      128 .. 2048
  gsm8k            en    math      128 .. 2048
  alpaca           en    instruct  128 .. 2048
  mtbench          en    chat      128 .. 2048
  nq               en    qa        128 .. 2048
  cnndm            en    summary   128 .. 2048
  alpaca_zh        zh    instruct  128 .. 2048
  qasper           en    qa        1024 .. 16384
  gov_report       en    summary   1024 .. 16384
  multi_news       en    summary   1024 .. 16384
  hotpotqa         en    qa        1024 .. 16384
  lcc              en    code      1024 .. 16384
  repobench-p      en    code      1024 .. 16384
  multifieldqa_zh  zh    qa        1024 .. 16384
  dureader         zh    qa        1024 .. 16384
  vcsum            zh    summary   1024 .. 16384
  lsht             zh    classify  1024 .. 16384

  en 336 / zh 153      code 96, qa 156, summary 132, classify 36, instruct 32, math 24, chat 18
  bands 128, 512, 1024, 2048, 4096, 8192, 12288, 16384
```

Four rules the construction follows, each because the alternative would rig the number:

- **Stratify the split on the (suite, band) CELL, not at random.** The claim is that one
  policy handles every situation; both halves must therefore contain every situation. A
  random split leaves whole cells in train, and the test set then silently stops testing
  them. All 83 cells are covered in train, 81 of 83 in test.
- **Cut long items head-and-tail, never from the end.** LongBench's own evaluation does this.
  A long-context task's question sits at the end, so cutting from the end deletes the task.
- **A short set reaches a long band by PACKING DISTINCT items**, which is what a few-shot
  prompt is. Repeating text would be the one construction that flatters any drafter: long
  repeats are trivially predictable, acceptance rises, and wide trees look better than they
  are.
- **A KNOWN GAP: no free-generation kind.** Every long-context suite here is a READING task,
  so at 4096..16384 every prompt hands the drafter something to copy. Free generation belongs
  at the low-acceptance end and is present only accidentally (MT-Bench's own `writing` and
  `roleplay` rows were collapsed into `chat`; Alpaca's 45 creative instructions into
  `instruct`). The bias runs AGAINST the policy -- it shifts the mix toward wide and so toward
  fixed 16 being the right constant. It also removes the one cell where the bar (low at short
  context, favouring wide) and acceptance (lowest under free generation, favouring narrow)
  disagree. Next corpus; evidence section 6.
- **The table is restored before every test run**, so no held-out prompt teaches another --
  but it is NOT frozen inside a run, and that is deliberate. The cost law is effectively
  frozen (11500 seeded vs ~90 added, ~0.4%); the GAIN is not (decay 1/256, so ~30% of its
  mass is the current prompt's after 90 rounds). That is the shipped behaviour -- a served
  request adapts within itself -- so the claim is "trained on 408, deployed on 81 unseen,
  still adapting per request", NOT "a frozen model on held-out data".

### 2.9 Apple-GPU kernel work that may be new to a Metal engine

Not all of it is DSpark, but it is what makes the DSpark numbers possible, and some of it we
have not seen elsewhere:

- **Mega-kernel decode**: one persistent dispatch per token's layers on LFM2 (34 -> 5
  dispatches/token), with a timeout failsafe that rolls a region back and re-runs it on the
  dispatch path. Found and recorded: Apple GPUs violate occupancy-bound execution at two
  threadgroups per core (OOPSLA 2021 s6.3.2), so the task-queue form is 12% slower than the
  barrier form at one per core. **#141, #149-#156.**
- **Q8_0 tile-major weight file** (type 1000), zero-copy, bit-identical to the GGUF. **#102,
  #112, #113.**
- **One Q8 GEMM tile rule, two regimes, one tuned bound** -- and the measurement that a
  64-row tile is slower and the 33-row verify step is tile padding. **#256, #257.**
- **Flash-decoding at hd <= 128** and one uniform page load per paged-KV block (-7%
  attention). **#96, #106.**
- **Block-quant decode GEMV at the 136 GB/s wall** for Q4_K/Q5_K/Q6_K. **#173.**
- **Residency as a route precondition**, which removed a class of cold-start stalls. **#154.**

---

## 3. What is missing

| | status | note |
|---|---|---|
| The guard (design 10) | OPEN | cool down when drafting loses; the fork's guard fired at ~10K prefix |
| Test-time training (design 11.6) | OPEN | Draft-OPD recipe; the user's next step |
| n-gram dictionary (11.5), request index (11.4) | OPEN | free candidates, should raise S where it binds |
| Levels 2-3 of the candidate tree (6.1) | OPEN | level 1 only today; S is the binding constraint at depth |
| Qwen3.8-27B end to end | OPEN | kernels prototyped (#245, #246); #166 is the frame |
| vs llama.cpp / oMLX / uzu WITH speculation | OPEN | today's comparisons are all plain decode |
| Acceptance-rate curves by context and by source | OPEN | we have ms/token, not the acceptance profile |
| A second device (M4 / M1 Max) | OPEN | every number here is one M3 Pro |

---

## 3a. The model set and the comparison engine (settled 2026-09-21)

THREE SMALL MODELS, the user's call -- the 27B is too slow to sweep and every claim here should
be reproducible on a laptop:

| target | drafter | GGUF | note |
|---|---|---|---|
| LFM2.5-2.6B | LiquidAI/LFM2.5-2.6B-DSpark | have it | 5 attention layers, rank-256 Markov head, confidence head, block 9; a sidecar, paired with the target GGUF |
| Qwen3-8B | deepseek-ai/dspark_qwen3_8b_block7 | ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF, 1.53 GB | ready to download |
| Qwen3-4B | deepseek-ai/dspark_qwen3_4b_block7 | none found; convert ourselves | llama.cpp LLM_ARCH_DSPARK |

CONVERSION GOTCHA: a DSpark draft ships NO TOKENIZER and reuses the target's, so
`--target-model-dir` is required when converting to GGUF. Runtime: `--spec-type draft-dspark
--spec-draft-n-max 7`.

THE FILES, and why these ones (checked on Hugging Face 2026-09-21):

```
8B target    Qwen/Qwen3-8B-GGUF  Qwen3-8B-Q4_K_M.gguf                       downloading 2026-09-21
8B drafter   ankk98/dspark-qwen3-8b-block7-Q4_K_M-GGUF                      downloading 2026-09-21
             (a conversion of deepseek-ai/dspark_qwen3_8b_block7)
4B target    Qwen/Qwen3-4B-GGUF  Q4_K_M, to match                           not fetched
4B drafter   deepseek-ai/dspark_qwen3_4b_block7, no GGUF found -- convert   not fetched
LFM2.5       LiquidAI/LFM2.5-2.6B-DSpark (+GGUF)                            have it
```

THE QUANT IS HELD AT Q4_K_M ACROSS THE QWEN PAIRS, target and drafter alike -- the drafter's card
pairs it with exactly this target file. unsloth also publishes both targets (UD-Q4_K_XL and a
fuller UD ladder) and the 27B we serve is an unsloth UD-Q4_K_S; using Qwen's own Q4_K_M for the 4B
and 8B instead keeps the two benchmarked Qwen models on ONE recipe, which is what the comparison
between them needs. The 27B is not in the benchmark set, so its different recipe costs nothing.

FOR THE 4B, one choice is still open: take a community GGUF if one appears, or convert
deepseek-ai/dspark_qwen3_4b_block7 ourselves. Converting is the safer default -- a drafter's
quantisation moves its acceptance rate, so the 4B and 8B drafters should be quantised the same
way or the comparison between those two models is partly a comparison of quantisers. Whichever is
used, record its provenance and the llama.cpp commit in the results table, as
[[bracket-reference-binary-is-part-of-the-result]] requires of any reference binary.

What the set does and does not buy: LFM2.5 is a hybrid (short convolution + attention), Qwen3-4B
and Qwen3-8B are dense attention. So the set tests SIZE and a second architecture family, and the
recurrent-tree claim (2.3) still rests on LFM2.5 alone among the benchmarked models -- the gated
delta rule work is prototyped on the 27B but will not be swept. Say that in the paper rather than
implying three independent tests of it.

### The comparison engine is llama.cpp, and only llama.cpp

Checked 2026-09-21. To compare on the SAME FILE the other engine must read GGUF:

```
llama.cpp    GGUF + DSpark (PR #25173; PR #27383 for LFM2; a separate PR adds shape-optimised
             METAL kernels for DSpark) -- a fair opponent on this hardware, not a strawman
LM Studio    supports DSpark by wrapping llama.cpp; measuring it measures llama.cpp
Ollama       does not expose draft-model flags
mlx-dspark   a native MLX port (ARahim3/mlx-dspark) covering Gemma-4, Qwen3.8, LFM2.5, Bonsai --
             the closest prior art, but MLX safetensors, so a comparison would be a different
             quantisation of different files. That is the same weights problem that made the uzu
             comparison meaningless (memory uzu-joins-the-comparison).
```

mlx-dspark is still PRIOR ART the paper must position against even though it cannot be
benchmarked head to head: section 2.7a did this for CAST and the same is owed here.

---

## 4. The long-context gap, stated honestly

At 8444 tokens of context the chooser TIES the best fixed width. That is not a failure of the
rule -- it is the rule reporting that the options are close. The round, attributed:

```
form  rows  round   draft   tree  verify  append   draft%  verify%
tree     8  43.36   10.89   0.03   31.21    1.24    25.1%    72.0%
tree    16  47.81   10.91   0.03   35.60    1.23    22.8%    74.5%
```

- A marginal row costs 0.55 ms inside a tile; a token is worth ~11.5 ms. A row pays for
  itself at ~5% acceptance, so the chooser stopping at 16 says rows 17-24 are below that --
  **S is the binding constraint, not T and not the rule.**
- The drafter costs 10.9 ms EVERY round, 23-25% of it, independent of n, and has never been
  attributed. It is the largest n-independent term and the obvious next lever.
- The formulation that makes the drafter a DECISION rather than a constant is an
  average-reward SMDP over `u = (draft block length, verify rows)`; the current rule is that
  problem with the bias function held constant. The prize is bounded by the drafter's share:
  ~12% at depth if half of it can be amortised across rounds.

**Nothing in this section is built or measured beyond the attribution above.**

---

## 5. Where the numbers live

```
docs/speculator-design.md                              the design, section by section
docs/evidence/dspark/2026-09-15-step4b-*.md            level-1 tree, the 33-row step
docs/evidence/dspark/2026-09-16-step4c-*.md            the cost table and its estimator
docs/evidence/dspark/2026-09-16-step4d-budget.md       the budget, and the D correction
docs/megakernel-decode.md                              the mega-kernel design + evidence
docs/evidence/bracket/2026-09-03-omlx-comparison.md    standing vs oMLX / llama.cpp
```
