//! A good tuner understands model, hardware and kernel, and only by that it provides value for all users including developers.
//!
//! Three requirements and a claim. The three: know the MODEL (tensor shapes, head
//! geometry, quantization), the HARDWARE (limits that can be queried and limits that
//! must be measured), and the KERNEL (what each knob selects, and which knobs move
//! together). The claim is that value follows only from all three -- and the developer
//! is a user here too: the instrument that picks a shape for a stranger's Mac is the
//! same one the kernel gets developed with.
//!
//! Stage 1: per-kernel micro-benchmarks. No model weights, no forward pass.
//!
//! BACKEND-GENERIC: everything here speaks the `Backend` trait and the backend's own
//! `KnobDecl` registry. The registry declares WHAT to measure (stage, sweep kind,
//! workload); this file owns HOW to measure honestly (warm-up, noise floors,
//! interleaving, screens, the drift control). A backend developer adds a knob in
//! their crate's registry and never touches this file.
//!
//! Most knobs are properties of ONE kernel at ONE shape. None of them depend on the
//! weight VALUES, only on the shapes and the hardware -- so none of them need the
//! model mapped. The end-to-end trial harness is stage 2 (main.rs), opt-in and small.

use imparo_backend::{
    Backend, BatchGeometry, BatchPhase, BufId, Epilogue, KnobDecl, ModelFacts,
    SweepKind, TunerScratchRegion, Workload, WorkloadEffects,
};

/// One real dense gated-FFN projection triple, mapped onto independent regions of the
/// tuner's synthetic weight blob. Names are resolved by the model bench extension;
/// consumers never infer gate/up/down from list position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FfnTransactionShape {
    pub gate: (u64, u32, u32, u32),
    pub up: (u64, u32, u32, u32),
    pub down: (u64, u32, u32, u32),
}

/// One real per-layer-embedding gate/projection pair and its token-major per-layer
/// multiplier layout. Together with `FfnTransactionShape` this is the complete
/// exact-128 transaction; neither half is sufficient evidence for the coupled route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PleTransactionShape {
    pub gate: (u64, u32, u32, u32),
    pub proj: (u64, u32, u32, u32),
    pub per_layer_stride: u32,
}

/// One model-validated short-convolution state transition. The tensor offset comes
/// from GGUF metadata; width, taps, and state size must also agree with the model plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShortconvTransactionShape {
    pub weight_off: u64,
    pub width: u32,
    pub kernel: u32,
    pub state_elements: u32,
}

/// The shapes the kernels are measured at. Read from the GGUF header, which is
/// kilobytes; the tensor DATA is never mapped.
#[derive(Clone, Debug)]
pub struct Shapes {
    /// The model's one process-wide gated activation, resolved from every layer.
    pub activation: Epilogue,
    /// The decode-step matmuls as `(weight offset, n_in, n_out, weight kind)`, at the
    /// offsets AND THE QUANT TYPES the model actually stores them with.
    ///
    /// THE KIND IS READ, NOT ASSUMED. A GGUF carries a type per TENSOR, and mixed-quant
    /// files are the norm rather than the exception -- the K-quant families put different
    /// widths on different projections on purpose. This list used to be dimensions only
    /// and every entry was measured as q4_0, which is a different kernel and a different
    /// bytes-per-row from what several of those tensors would really run.
    ///
    /// That guard is what made adding Q8_0 weights a one-line change here: the list
    /// carries kind 2 for a Q8_0 tensor and the workload dispatches the Q8 routes,
    /// without a Q8-specific workload variant. Before the kind was read, every entry was
    /// measured as q4_0 -- a different kernel over a different bytes-per-row.
    ///
    /// Every shape used to be read from offset 0, so all seven hit the SAME bytes and
    /// the memory system saw a locality the engine never has -- on a kernel whose whole
    /// cost is weight traffic. Empty falls back to shapes derived from the config.
    pub decode_mix: Vec<(u64, u32, u32, u32)>,
    /// The lm-head GEMV (`output.weight`, or the tied `token_embd.weight`) at its real
    /// dimensions and post-load kind: (offset, n_in, n_out, kind). It runs ONCE per token
    /// where the layer mix runs n_layers times, so DecodeMix counts it at 1/n_layers --
    /// the largest single GEMV per token (LFM2 ~278 MB, E4B ~377 MB, 10-15% of the decode
    /// bytes) was in no workload before 2026-09-04, and the row-major decode pair was
    /// being ranked on layer shapes the transformed model no longer dispatches.
    pub lm_head: Option<(u64, u32, u32, u32)>,
    /// The majority exact-128 attention-Q projection. Kept explicit because the
    /// exact-128 route may change its MMQ token tile while leaving every other
    /// projection unchanged; the atomic tuner transaction must price that work at
    /// a real model offset instead of inferring it from a dimension-only mix.
    pub exact128_local_q: Option<(u64, u32, u32, u32)>,
    /// Complete gate/up/down transaction, or None when the model bench cannot prove
    /// three supported tensors with compatible dimensions and independent spans.
    pub ffn_transaction: Option<FfnTransactionShape>,
    /// Exact PLE transaction, or None unless the named tensors and model layout were
    /// validated. The exact-128 knob is inapplicable without both this and the FFN triple.
    pub ple_transaction: Option<PleTransactionShape>,
    /// Short-convolution shape, or None unless tensor and recurrent-state metadata agree.
    pub shortconv_transaction: Option<ShortconvTransactionShape>,
    /// Weight kinds present on the layer projections, as a bitmask over the wire values
    /// (bit 1 = Q4_0, bit 2 = Q8_0). Derived from `decode_mix`, so it says what the
    /// workloads will actually dispatch.
    pub weight_kinds: u32,
    /// The kind the WIDE prefill matmuls carry (n_embd->n_ff, n_ff->n_embd). PrefillGemm
    /// used a hardcoded Q4_0 and would otherwise rank a Q8 tile knob against a Q4 kernel.
    pub wide_kind: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_head: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    /// 0 for a dense model; the MoE knobs key off this.
    pub n_experts: u32,
    /// Head dim of the FULL-attention layers -- the only geometry whose span grows
    /// with the conversation, so the only one a decode-routing boundary can be
    /// measured on. In Gemma4 E4B the majority geometry is the 256-dim WINDOW
    /// layers, whose span never exceeds their window.
    pub deep_head_dim: u32,
    /// Blocks in the model, from the GGUF header.
    pub n_layers: u32,
    /// Compute dispatches ONE layer encodes in a decode step, counted from the model's
    /// forward code (the ModelBench extension). See `ModelFacts::layer_dispatches`.
    pub layer_dispatches: u32,
    /// The token width that exercises EVERY candidate for the prefill pair's second tile:
    /// the largest token tile in the backend's shape table (128 on Metal). The dispatch's
    /// pad rule selects a candidate only where it pads no worse than the first tile, so a
    /// narrower width would leave the wider candidates unmeasured -- they would read as
    /// the first tile's time, a tie, and never be picked.
    pub pair_tile_tokens: u32,
    /// THE MODEL'S REAL ATTENTION GEOMETRY, one entry per layer: `(head_dim, window)`,
    /// window 0 meaning full attention.
    ///
    /// Here because a decode step is a MIX and ranking a knob on one member of it gets the
    /// answer wrong. gemma4 E4B runs 7 full-attention layers at 512 dims among 35 windowed
    /// ones at 256 whose spans never exceed their window -- so a workload that times one
    /// deep dispatch is measuring 7/42 of the attention a step does, with none of the
    /// interleaving.
    pub attn_layers: Vec<(u32, u32)>,
}

/// What stage 1 decides: one value per Micro-stage Benched knob, in registry order,
/// plus the screen survivors of every `screened` knob for stage 2 to judge end to end.
#[derive(Clone, Debug)]
pub struct Picks {
    pub picks: Vec<(&'static str, u32)>,
    /// Screen survivors per screened knob. A LIST, not a pick: micro-benchmarking once
    /// ranked rt_shape 6 as 1.6% faster and it was 11% SLOWER across a real prefill.
    /// The screen only throws out candidates that cannot run on this device.
    pub screened: Vec<(&'static str, Vec<u32>)>,
}

/// Interleaved rounds per candidate. Three is enough for a median to reject one bad
/// sample; more rounds cost linearly and this has to stay light.
const ROUNDS: usize = 3;

/// Largest cartesian product a coupled tuple may be swept over. Past this the joint
/// search costs more than it can be worth and the tuple falls back to one knob at a
/// time, with the fallback logged -- a silent truncation would read as "measured
/// jointly" when it was not.
const JOINT_MAX: usize = 32;

/// Every stage of the exact-128 coupled route must be observed in one completed
/// submission. The low bits prove the route's commit stages; the three six-bit
/// counters prove that every expected model layer reached its final controlled
/// stage. A stage-only OR mask can hide one fallback behind another successful
/// layer and is therefore not benchmarkable/promotable evidence.
const EXACT128_STAGE_MASK: u32 = 0x3f;
const EXACT128_COUNT_MASK: u32 = 0x3f;
const EXACT128_ATTN_COUNT_SHIFT: u32 = 6;
const EXACT128_FFN_COUNT_SHIFT: u32 = 12;
const EXACT128_PLE_COUNT_SHIFT: u32 = 18;
const EXACT128_TOKEN64_COUNT_SHIFT: u32 = 24;

fn exact128_route_evidence(
    attn_layers: usize,
    total_layers: usize,
    token64_layers: usize,
) -> Option<u32> {
    let attn_layers = u32::try_from(attn_layers).ok()?;
    let total_layers = u32::try_from(total_layers).ok()?;
    let token64_layers = u32::try_from(token64_layers).ok()?;
    if attn_layers == 0
        || attn_layers > EXACT128_COUNT_MASK
        || total_layers > EXACT128_COUNT_MASK
        || token64_layers > EXACT128_COUNT_MASK
    {
        return None;
    }
    Some(
        EXACT128_STAGE_MASK
            | (attn_layers << EXACT128_ATTN_COUNT_SHIFT)
            | (total_layers << EXACT128_FFN_COUNT_SHIFT)
            | (total_layers << EXACT128_PLE_COUNT_SHIFT)
            | (token64_layers << EXACT128_TOKEN64_COUNT_SHIFT),
    )
}

/// Tokens a prefix-matched turn actually prefills. A continuation re-sends the whole
/// conversation and the cache matches almost all of it; STATUS.md records a two-turn
/// case reusing 640 of 721. The tail is what the engine runs, and it is neither a full
/// chunk nor a single token.
const REUSE_TAIL: u32 = 64;

/// KV layers the span-crossing sweep rotates through. Four puts ~130 MB between
/// two reads of the same layer at the deepest rung, well past the SLC.
const SPAN_LAYERS: usize = 4;

/// The longest a single command buffer may hold the GPU, in microseconds.
///
/// One display frame at 120 Hz. A command buffer runs to completion without being
/// preempted between its kernels, so its duration is a window in which the compositor
/// cannot present. This is what made the machine unusable, not the total amount of GPU
/// work: sleeping between buffers does not help -- the buffer already queued still runs
/// to the end. Making the buffer shorter does.
const FRAME_US: f64 = 2_000.0;

/// The GPU share this tuner is allowed to take while it runs, as 1/DUTY. The machine
/// belongs to the user; a tuner that makes the UI stutter is not usable regardless of how
/// good its answers are.
///
/// Two things had to change together to stop it freezing the machine. A command buffer
/// could hold FRAME_US of GPU work, and that was 8300 us -- a whole display frame, so the
/// compositor missed one every buffer. And the sleep between buffers was a fixed 8300 us,
/// which against an 8300 us buffer is a 50% duty cycle: half the GPU, taken in
/// frame-length bursts. Now a buffer holds at most 2 ms and the sleep is proportional to
/// what was just submitted, so the share is bounded whatever the kernel under test costs.
///
/// Wall-clock cost is DUTY x the GPU time, which is the right trade for a background
/// citizen. IMPARO_TUNE_FULL=1 removes the pacing for a dedicated run.
const DUTY: f64 = 4.0;

fn kv_scratch_bytes(kv_tag: &str, slots: u32, n_kv: u32, head_dim: u32) -> u64 {
    if kv_tag == "f16" {
        return 0;
    }
    u64::from(slots)
        * u64::from(n_kv)
        * u64::from(head_dim)
        * u64::try_from(std::mem::size_of::<u16>()).unwrap_or(2)
}

/// Output rows a GEMM micro-bench computes. Sized so ONE dispatch stays around a
/// millisecond -- a micro-bench that holds the GPU for a display frame at a time makes
/// the machine stutter no matter how the gaps between buffers are paced, because a
/// dispatch is indivisible. Large enough that the grid still fills the GPU.
/// A noise floor this wide cannot resolve a knob-sized difference, so "every candidate
/// landed within it" says the BENCH is too coarse rather than the knob unwired. Measured
/// case: the prefill attention tuple on a q4_0 cache spreads its candidates by ~3% on a
/// floor of 6.5%, and the kernels are demonstrably reached -- the same workload takes 9.8
/// ms on f16 and 79 ms on q4_0.
const UNRESOLVABLE_FLOOR: f64 = 0.05;

/// The position every DEEP workload is measured at. 16k is a long agentic context and the
/// depth this repo's end-to-end brackets use, so a knob ranked here is a knob verified
/// there. Kept as a module constant because the KV allocation has to honour it too -- the
/// two used to disagree, and the workloads quietly ran shallow.
/// The score-tile path's threadgroup score capacity, the same 8192 the model clamps to
/// (`max_scores()` in workflow_gpu.rs, 32 KiB of floats). It is a CAP, not a request: the
/// engine passes `min(span, window)` CLAMPED to it, and the kernel sizes threadgroup memory
/// from whatever it is handed.
///
/// The workloads here were passing `pos + 1` unclamped, so a 16k-deep decode asked for
/// 16385 scores -- twice the cap, 64 KiB of threadgroup memory against a 32 KiB device
/// limit -- and a 64k one asked for eight times it. That also changes `score_tile`, and
/// with it how many slices the score-tile path splits into, which is exactly the thing the
/// streaming boundary is racing.
const MAX_SCORES: u32 = 8192;

/// What the engine asks for at a given span and window: the span it will actually scan,
/// capped. Mirrors `scores_needed` on the model side.
fn scores_needed(span: u32, window: u32) -> u32 {
    let needed = if window > 0 { span.min(window) } else { span };
    needed.clamp(1, MAX_SCORES)
}

const DEEP_DECODE_POS: u32 = 16384;
/// Queries per dispatch for the deep prefill attention workload. See its arm: per-key
/// ranking, an eighth of the 512-query dispatch, four 8-row tiles.
const ATTN_DEEP_QUERIES: u32 = 64;

/// The token counts the tile-ranking workload sends, and why they are SMALL.
///
/// What the tile knobs trade is weight re-reads against padding, and neither term needs
/// many tokens to show up: the weight working set does not depend on the token count at
/// all, and the padding branch is decided by `n_tok % toks`. So the widths are chosen for
/// what they EXERCISE, not for how much work they carry --
///
///   256   a multiple of both 32 and 64, so the rule takes the WIDE tile
///   224   a multiple of 32 but not 64, so 64 pads to 256 and the rule takes the NARROW
///
/// -- which keeps each dispatch under a millisecond at the light output slice this
/// workload uses. A dispatch cannot be preempted, and that, not total GPU time, is what
/// makes a tuning run visible to whoever is using the machine: `time_us` already scales
/// the REP count to a fixed 4 ms of GPU per measurement, so a bigger rep buys fewer reps
/// rather than a longer sweep. The pair's SECOND tile is not ranked here: it needs the
/// full projection, which is `PrefillGemmUbatch`.
const RANK_WIDTHS: [u32; 2] = [256, 224];

/// Output width for the WORK-TERM prefill GEMM, and the fallback when the device profile
/// carries no measured knee.
///
/// Two different jobs used to share one workload. Measuring how much MATMUL a prefill
/// layer carries -- the command-buffer trade's work term -- only needs the right amount
/// of arithmetic, so a narrow slice is right and cheap. RANKING A TILE needs the weight
/// tile to miss cache, which a narrow slice cannot do. They are separate workloads now:
/// `PrefillGemm` keeps this width, `PrefillGemmWidths` derives a wider one.
const GEMM_WORK_SLICE: u32 = 1024;

/// How far past the measured cache knee the two-width ranking slice reaches.
///
/// Four times the knee keeps every dispatch of `PrefillGemmWidths` under a millisecond,
/// which is the tuner's first rule: a dispatch cannot be preempted, and long ones freeze
/// the machine for whoever is using it (measured: the full 23 MB projection at 512 tokens
/// is ~5 ms per dispatch and did exactly that). What this slice CANNOT do is rank the
/// pair's SECOND tile: that trade is weight re-reads against padding, and a 4 MB tile
/// stays inside the last-level cache -- the 1 MB knee the profile measures is a different
/// cache -- so the re-reads are free here and expensive in the engine. Measured on LFM2
/// (Q8, M3 Pro), 64x64 against the 64x32 default at the whole-ubatch width:
///
///     slice  4x knee  (4 MB)    +0.1%      end to end, by env pin:  +3.1% / +3.0%
///            8x knee  (8 MB)    -0.1%                                (5963 / 17123 tokens)
///           16x knee (16 MB)    +2.2%   on a 2.4% floor
///           full n_ff (23 MB)   +4.0%   on a 2.0% floor
///
/// So the second tile has its own workload, `PrefillGemmUbatch`, which dispatches the full
/// projection at the fewest tokens that exercise every candidate; the first tile and the
/// other two knobs on this workload keep the light slice.
const GEMM_SLICE_PAST_KNEE: u64 = 4;

/// Output width for the two-width TILE-RANKING prefill GEMM: the smallest multiple of 64
/// whose weight bytes clear the knee by `GEMM_SLICE_PAST_KNEE`, capped at the model's
/// `n_ff`. Sliced because the full pair is ~8.9 ms of GPU in ONE dispatch; slicing changes
/// nothing about what is ranked -- identical kernel, tile and per-threadgroup work, fewer
/// threadgroups -- for the knobs whose trade this slice can see (see the constant above
/// for the one whose trade it cannot).
fn gemm_slice(profile: &imparo_backend::DeviceProfile, s: &Shapes) -> u32 {
    if profile.cache_knee_bytes == 0 {
        return s.n_ff.min(GEMM_WORK_SLICE);
    }
    // Bytes per weight element for the kind the wide projections actually carry: Q4_0 is
    // an 18-byte block of 32, Q8_0 a 34-byte block of 32, anything else raw f32.
    let (num, den): (u64, u64) = match s.wide_kind {
        1 => (18, 32),
        2 => (34, 32),
        _ => (4, 1),
    };
    let want_bytes = profile
        .cache_knee_bytes
        .saturating_mul(GEMM_SLICE_PAST_KNEE);
    let per_row = u64::from(s.n_embd).saturating_mul(num) / den.max(1);
    let rows = want_bytes.div_ceil(per_row.max(1));
    let rows = u32::try_from(rows.next_multiple_of(64)).unwrap_or(u32::MAX);
    s.n_ff.min(rows.max(64))
}

/// GPU busy time a single timed region should reach, in microseconds. Short
/// regions are all noise: the same kernel timed twice at 0.3 ms varies by tens
/// of percent, which is wider than any candidate's real difference.
const TARGET_BUSY_US: f64 = 4_000.0;
/// One untimed rep longer than this is a catastrophe, not a candidate (see `time_us`).
const REP_CATASTROPHE_US: f64 = 200_000.0;

fn reps_per_buffer(reps: usize, one_us: f64, restore_each_rep: bool) -> usize {
    if restore_each_rep {
        1
    } else {
        ((FRAME_US / one_us.max(1.0)) as usize).clamp(1, reps)
    }
}

/// How far off the fastest a screened candidate may read and still be swept.
///
/// Four times is far wider than any real winner needs -- the seven viable rt shapes sit
/// within 1.3x of each other -- and far below the 60x a register-spilling one shows.
const SCREEN_MAX: f64 = 4.0;

/// How much faster one side of a crossing has to be before the crossing is called.
/// The curve is steep either side of it, so a margin this size moves the answer by at
/// most one rung.
const CROSSOVER_MARGIN: f64 = 0.05;

/// First reachable rung in the single loss->win suffix. More than one transition means
/// a scalar minimum cannot represent the observed route map and must fail closed.
fn stable_token_min_threshold(rows: &[(u32, bool, bool)]) -> Option<u32> {
    let statuses: Vec<bool> = rows
        .iter()
        .filter(|(_, reachable, _)| *reachable)
        .map(|(_, _, wins)| *wins)
        .collect();
    if statuses
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count()
        > 1
    {
        return None;
    }
    rows.iter().enumerate().find_map(|(index, row)| {
        (row.1 && row.2 && rows[index..].iter().all(|later| later.1 && later.2))
            .then_some(row.0)
    })
}

fn settle_token_min_scans(scans: &[Option<u32>], ladder: &[u32], default: u32) -> u32 {
    let mut positions: Vec<usize> = scans
        .iter()
        .filter_map(|threshold| {
            threshold.and_then(|value| ladder.iter().position(|n| *n == value))
        })
        .collect();
    positions.sort_unstable();
    if positions.len() == 3 && positions[2].saturating_sub(positions[0]) <= 1 {
        ladder[positions[1]]
    } else {
        default
    }
}

/// One display frame of GPU idle between the tuner's command buffers, so the window
/// server and every other app stay smooth while a sweep runs. Background-citizen mode
/// is the DEFAULT; IMPARO_TUNE_FULL=1 removes the gaps for an unattended/CI machine.
/// OFF by default, and that is the fix rather than a regression.
///
/// This slept between command buffers so the machine would stay usable. It did the
/// opposite. The evidence, all on one machine:
///
/// ```text
///                        freezes?  GPU/buffer  longest dispatch  sleeps between
///   server, 16k prefill    no        ~1 s          ~34 ms            never
///   llama-bench            no      whole graph     model-sized       never
///   this tuner, paced      YES      1-4 ms          ~1 ms          every buffer
/// ```
///
/// The two that do far MORE work, in longer buffers, with longer dispatches, are fine.
/// The one that does least, in short bursts with a sleep after each, is the one that
/// stalls the display. Sustained load is not the problem; chopping it into bursts is,
/// because the GPU never settles -- the warm-up in this same file already records the
/// device ramping (the same reference read 28% faster after warming up), and pacing
/// forces that ramp continuously.
///
/// Two wrong diagnoses preceded this, both mine: "it is just pacing, not a bug" and then
/// "a single indivisible dispatch is too long". The second is disproved by the server's
/// 34 ms attention dispatches, which do not stutter.
///
/// IMPARO_TUNE_PACE=1 restores the sleeps if a machine is ever found that wants them.
fn pace(busy_us: f64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var("IMPARO_TUNE_PACE").is_ok_and(|v| v == "1")) {
        let idle = (busy_us * (DUTY - 1.0)).clamp(200.0, 50_000.0);
        std::thread::sleep(std::time::Duration::from_micros(idle as u64));
    }
}

fn device_or_wall_us(b: &dyn Backend, wall_us: f64, context: &str) -> f64 {
    let gpu = b.last_gpu_us();
    if gpu.is_finite() && gpu > 0.0 {
        gpu
    } else if b.tuner_requires_device_timing() {
        panic!(
            "backend requires device-event timing, but {context} produced no valid device time"
        )
    } else {
        wall_us
    }
}

struct RestoreGuard<'a>(&'a dyn Fn());

impl Drop for RestoreGuard<'_> {
    fn drop(&mut self) {
        (self.0)();
    }
}

/// Median of `samples` timings of `f`, in microseconds; `reps` per sample so a single
/// dispatch's fixed cost is amortised the way a real forward pass amortises it. Repeats
/// are packed into command buffers no longer than `FRAME_US` (see there).
fn time_us(b: &dyn Backend, samples: usize, reps: usize, f: &dyn Fn()) -> f64 {
    time_us_reset(b, samples, reps, false, &|| {}, f)
}

fn time_us_checked(
    b: &dyn Backend,
    samples: usize,
    reps: usize,
    evidence: Option<(Workload, u32)>,
    f: &dyn Fn(),
) -> f64 {
    time_us_reset_checked(b, samples, reps, false, &|| {}, evidence, f)
}

fn time_us_reset(
    b: &dyn Backend,
    samples: usize,
    reps: usize,
    restore_each_rep: bool,
    reset: &dyn Fn(),
    f: &dyn Fn(),
) -> f64 {
    time_us_reset_checked(b, samples, reps, restore_each_rep, reset, None, f)
}

fn time_us_reset_checked(
    b: &dyn Backend,
    samples: usize,
    reps: usize,
    restore_each_rep: bool,
    reset: &dyn Fn(),
    evidence: Option<(Workload, u32)>,
    f: &dyn Fn(),
) -> f64 {
    // One untimed rep: it pays for the pipeline switch, and its duration says how many
    // reps may share one command buffer.
    let t0 = std::time::Instant::now();
    reset();
    let restore = RestoreGuard(reset);
    b.reset_tuner_dispatch_proof();
    b.begin();
    f();
    b.end().unwrap_or_else(|rc| {
        panic!("tuner warm-up GPU submission failed with backend code {rc}")
    });
    b.validate_tuner_dispatch_proof()
        .unwrap_or_else(|error| panic!("tuner warm-up dispatch proof failed: {error}"));
    if let Some((workload, expected)) = evidence {
        assert_eq!(
            b.tuner_route_evidence(workload),
            expected,
            "tuner candidate silently fell back during warm-up"
        );
    }
    let one = device_or_wall_us(b, t0.elapsed().as_secs_f64() * 1e6, "the warm-up");
    drop(restore);
    pace(one);
    // A CATASTROPHE IS REJECTED ON ITS FIRST REP, NEVER TIMED. The heaviest legitimate
    // rep here is ~10 ms (PrefillGemmWidths, four dispatches); a candidate whose one
    // untimed rep takes 20x that is a spiller or a broken route, and timing it means
    // `samples x reps` more of the same -- rt_gemm<Q8> shape 7 read 12 s a rep, and the
    // LFM2 tune sat in waitUntilCompleted for three minutes before it was killed (#111).
    // Infinity loses every comparison below; the screen is meant to catch these first.
    if one > REP_CATASTROPHE_US {
        println!(
            "  one rep took {:.0} ms (> {:.0} ms): candidate rejected without timing",
            one / 1e3,
            REP_CATASTROPHE_US / 1e3
        );
        return f64::INFINITY;
    }
    // Scale the rep count so a timed region is milliseconds of GPU, not
    // microseconds. The declared reps were tuned when a frame of pacing sat
    // inside the measurement and hid the variance; with that gone, a 0.3 ms
    // region shows a 44% noise floor and no candidate can ever beat its
    // margin. Cheap: the work is short by construction.
    let reps = reps.max(((TARGET_BUSY_US / one.max(1.0)) as usize).clamp(1, 4096));
    // A mutable operation must see the same baseline on every invocation, including
    // repetitions packed into one command buffer. Since restoring is a host operation,
    // do not pack multiple mutable reps behind one begin/end boundary. A coupled route
    // is likewise verified once per submission so one successful repetition cannot
    // hide a later fallback by OR-ing its evidence bits.
    let per_buf = reps_per_buffer(reps, one, restore_each_rep || evidence.is_some());

    // Time the COMMAND BUFFERS, not the gaps between them. `pace` sleeps a
    // frame after each buffer so the machine stays usable while a sweep runs;
    // inside the timed region that sleep was charged as kernel time, and since
    // it is constant it compressed every ratio toward 1.0 -- at 12 reps of
    // ~300 us, 3.6 ms of work was being read as 11.9 ms. A span sweep whose
    // kernels differ by 32% reported 1.08x until this moved out.
    let run = || -> f64 {
        let mut busy = 0.0;
        let mut done = 0;
        while done < reps {
            let n = per_buf.min(reps - done);
            reset();
            let restore = RestoreGuard(reset);
            b.reset_tuner_dispatch_proof();
            let t = std::time::Instant::now();
            b.begin();
            for _ in 0..n {
                f();
            }
            b.end().unwrap_or_else(|rc| {
                panic!("tuner timed GPU submission failed with backend code {rc}")
            });
            b.validate_tuner_dispatch_proof()
                .unwrap_or_else(|error| panic!("tuner dispatch proof failed: {error}"));
            if let Some((workload, expected)) = evidence {
                assert_eq!(
                    b.tuner_route_evidence(workload),
                    expected,
                    "tuner candidate silently fell back during a timed submission"
                );
            }
            // The DEVICE's own busy time where it can report it. A wall clock here also
            // charges commit and waitUntilCompleted, which is variable and comparable to
            // the work when a region holds one short buffer -- that is what held this
            // bench's noise floor above the differences it was meant to resolve.
            let this =
                device_or_wall_us(b, t.elapsed().as_secs_f64() * 1e6, "a timed region");
            drop(restore);
            busy += this;
            done += n;
            pace(this);
        }
        busy / reps as f64
    };

    let mut v = Vec::with_capacity(samples);
    for _ in 0..samples {
        v.push(run());
    }
    // THE MINIMUM, NOT THE MEDIAN. Kernel-timing noise is ONE-SIDED: clock ramp,
    // contention, scheduling and thermal throttling can only ever make a run slower,
    // never faster. So the fastest sample is the best estimate of what this kernel does
    // on an unobstructed machine, and every slower one is that plus interference.
    //
    // The median treats interference as if it were symmetric error, which made this
    // harness hostage to whatever else the machine was doing: three tuning runs were
    // REFUSED in one afternoon because the control "drifted" 30-33%, and the drift was
    // the GPU ramping its clocks upward -- 793 -> 546 us, 946 -> 661, 1558 -> 1049.
    // Under a minimum those are non-events; the fast sample is simply the true one.
    // tinygrad's BEAM search takes min of 3 for the same reason.
    //
    // It also fixes a smaller bug: `v[v.len()/2]` on an EVEN sample count returns the
    // upper of the two middles, so every even-sampled measurement was reported at its
    // slower half.
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

enum ScratchBytes {
    F32(Vec<f32>),
    Bytes(Vec<u8>),
}

struct ScratchImage {
    regions: Vec<(TunerScratchRegion, ScratchBytes)>,
    mutable: bool,
}

struct KnobRestore<'a> {
    knob: &'a KnobDecl,
    value: u32,
}

impl Drop for KnobRestore<'_> {
    fn drop(&mut self) {
        (self.knob.apply)(self.value);
    }
}

impl ScratchImage {
    fn capture(b: &dyn Backend, effects: WorkloadEffects) -> Self {
        Self::capture_with(
            effects,
            |id, off, values| b.read(id, off, values),
            |layer, is_v, off, values| b.read_kv_bytes(layer, is_v, off, values),
        )
    }

    fn capture_regions(b: &dyn Backend, regions: &[TunerScratchRegion]) -> Self {
        let regions = regions
            .iter()
            .map(|region| match *region {
                TunerScratchRegion::BufferF32 { id, off, elements } => {
                    let mut data = vec![0.0; elements];
                    b.read(id, off, &mut data);
                    (*region, ScratchBytes::F32(data))
                }
                TunerScratchRegion::KvBytes {
                    layer,
                    is_v,
                    off,
                    bytes,
                } => {
                    let mut data = vec![0; bytes];
                    b.read_kv_bytes(layer, is_v, off, &mut data);
                    (*region, ScratchBytes::Bytes(data))
                }
            })
            .collect();
        Self {
            regions,
            mutable: true,
        }
    }

    fn capture_with(
        effects: WorkloadEffects,
        mut read_f32: impl FnMut(BufId, u64, &mut [f32]),
        mut read_bytes: impl FnMut(u32, bool, u64, &mut [u8]),
    ) -> Self {
        let WorkloadEffects::Mutable(regions) = effects else {
            return Self {
                regions: Vec::new(),
                mutable: false,
            };
        };
        let regions = regions
            .iter()
            .map(|region| match *region {
                TunerScratchRegion::BufferF32 { id, off, elements } => {
                    let mut data = vec![0.0; elements];
                    read_f32(id, off, &mut data);
                    (*region, ScratchBytes::F32(data))
                }
                TunerScratchRegion::KvBytes {
                    layer,
                    is_v,
                    off,
                    bytes,
                } => {
                    let mut data = vec![0; bytes];
                    read_bytes(layer, is_v, off, &mut data);
                    (*region, ScratchBytes::Bytes(data))
                }
            })
            .collect();
        Self {
            regions,
            mutable: true,
        }
    }

    fn restore(&self, b: &dyn Backend) {
        self.restore_with(
            |id, off, values| b.write(id, off, values),
            |layer, is_v, off, values| b.write_kv_bytes(layer, is_v, off, values),
        );
    }

    fn restore_with(
        &self,
        mut write_f32: impl FnMut(BufId, u64, &[f32]),
        mut write_bytes: impl FnMut(u32, bool, u64, &[u8]),
    ) {
        for (region, data) in &self.regions {
            match (region, data) {
                (
                    TunerScratchRegion::BufferF32 { id, off, .. },
                    ScratchBytes::F32(values),
                ) => write_f32(*id, *off, values),
                (
                    TunerScratchRegion::KvBytes {
                        layer, is_v, off, ..
                    },
                    ScratchBytes::Bytes(values),
                ) => write_bytes(*layer, *is_v, *off, values),
                _ => unreachable!("scratch region and captured representation agree"),
            }
        }
    }
}

/// The Q4_0 weight kind on the wire (imparo-cpu's WeightKind discriminant).
const WK_Q4: u32 = 1;

/// Measures each Micro-stage knob against its declared workload, and screens every
/// `screened` knob. The backend must already be initialised (synthetic buffer or real
/// model -- weight VALUES do not affect timing).
///
/// # Errors
/// Returns a message when a buffer cannot be allocated or the machine drifts too much
/// for the measurements to be comparable to each other.
#[allow(clippy::too_many_lines)]
pub fn measure(
    b: &dyn Backend,
    reg: &'static [KnobDecl],
    s: &Shapes,
    verbose: bool,
    // Whether the caller accepts knobs that can change output BITS. Off by default: the
    // tuner ranks on time, so a bit-affecting knob would otherwise trade numerics for
    // speed with nothing noticing.
    allow_bits: bool,
    // Measured ONCE by the caller and passed in. Sweeping the spill cliff and the cache
    // knee costs seconds; doing it twice in one run is seconds of the user's GPU spent
    // re-establishing something that cannot have changed since the last call.
    profile: &imparo_backend::DeviceProfile,
    // Measure ONE knob instead of the whole stage. The working pattern is to optimise a
    // kernel and bench that kernel: a full stage-1 pass touches every knob and every
    // screened candidate, which is minutes of paced GPU to answer one question.
    only: Option<&str>,
    // EVERY KNOB'S SWEEP STARTS FROM ITS SEAT, in registry order: the value the caller
    // captured after backend init (the stored config's, or the compiled default). Not the
    // live value -- an earlier knob's sweep may have moved it. The first prefill tile's
    // setter resets the pair to one shape, so without this the second tile swept against
    // its compiled default and lost its stored, end-to-end-verified value.
    seats: &[u32],
) -> Result<Picks, String> {
    // Phase timing, always on. A tuner that takes minutes must be able to say WHERE,
    // or every diagnosis of it is a guess -- which is how a KV staging loop and a
    // measurement loop got conflated for an entire session.
    let t_phase = std::time::Instant::now();
    let mark = |what: &str| {
        if verbose {
            println!(
                "  [phase] {what} at {:.2}s",
                t_phase.elapsed().as_secs_f64()
            );
        }
    };
    // GIVEN A MODEL, ONLY THE APPLICABLE SUBSET OF KNOBS RUNS. A knob that selects a MoE
    // routing path has nothing to say about a dense model; one that picks between
    // head-dim-512 kernels is irrelevant to a model without such layers. Checked BEFORE
    // the screen, so an inapplicable knob costs no measurement and gets no recorded pick
    // -- and the skips are printed, because a knob silently absent from a config file is
    // indistinguishable from one that was measured and left alone.
    let facts = ModelFacts {
        n_embd: s.n_embd,
        n_ff: s.n_ff,
        n_head: s.n_head,
        n_kv: s.n_kv,
        head_dim: s.head_dim,
        deep_head_dim: s.deep_head_dim,
        n_experts: s.n_experts,
        n_layers: s.n_layers,
        layer_dispatches: s.layer_dispatches,
        weight_kinds: s.weight_kinds,
    };
    let shortconv_available = s.shortconv_transaction.is_some();

    // DEPENDENCY ORDER, checked rather than assumed. The tuner drives the registry in
    // declaration order; a knob that must be settled first has to sit earlier in the file,
    // and until now nothing said so or noticed when it stopped being true.
    for (i, d) in reg.iter().enumerate() {
        for want in d.after {
            let at = reg.iter().position(|o| o.name == *want);
            match at {
                Some(j) if j < i => {}
                Some(_) => {
                    return Err(format!(
                        "{} declares after={want}, but {want} is declared LATER in the \
                         registry -- it would be measured against an untuned {want}",
                        d.name
                    ));
                }
                None => {
                    return Err(format!(
                        "{} declares after={want}, which is not in the registry",
                        d.name
                    ));
                }
            }
        }
    }

    let applicable = |d: &KnobDecl| {
        d.applies.is_none_or(|f| f(&facts))
            && (allow_bits || !d.bit_affecting)
            && (!matches!(
                d.workload,
                Workload::PrefillFfnTransaction | Workload::DecodeFfnTransaction
            ) || s.ffn_transaction.is_some())
            && (d.workload != Workload::DecodeShortconvTransaction
                || shortconv_available)
    };
    let selected_tuple = if let Some(name) = only {
        let target = reg
            .iter()
            .find(|d| d.name == name)
            .ok_or_else(|| format!("requested knob {name} is not in this registry"))?;
        if target.applies.is_some_and(|f| !f(&facts)) {
            return Err(format!(
                "requested knob {name} does not apply to this model's shape/weight kinds"
            ));
        }
        if matches!(
            target.workload,
            Workload::PrefillFfnTransaction | Workload::DecodeFfnTransaction
        ) && s.ffn_transaction.is_none()
        {
            return Err(format!(
                "requested knob {name} needs a validated dense FFN tensor triple"
            ));
        }
        if target.workload == Workload::DecodeShortconvTransaction
            && !shortconv_available
        {
            return Err(format!(
                "requested knob {name} needs a validated short-convolution tensor and state shape"
            ));
        }
        if target.bit_affecting && !allow_bits {
            return Err(format!(
                "requested knob {name} can change output bits; rerun with \
                 --allow-bit-changes and re-run numerical gates"
            ));
        }
        if let Some(group) = target.tuple {
            let held: Vec<&str> = reg
                .iter()
                .filter(|d| {
                    d.tuple == Some(group)
                        && d.applies.is_none_or(|f| f(&facts))
                        && d.bit_affecting
                        && !allow_bits
                })
                .map(|d| d.name)
                .collect();
            if !held.is_empty() {
                return Err(format!(
                    "requested knob {name} belongs to coupled tuple {group}, whose \
                     bit-affecting members are held: {}; rerun with --allow-bit-changes \
                     and re-run numerical gates",
                    held.join(" ")
                ));
            }
        }
        target.tuple
    } else {
        None
    };
    let selected = |d: &KnobDecl| {
        only.is_none_or(|name| {
            d.name == name || selected_tuple.is_some_and(|g| d.tuple == Some(g))
        })
    };
    let expect = |decls: &[&KnobDecl]| {
        let names: Vec<&str> = decls.iter().map(|d| d.name).collect();
        b.set_tuner_dispatch_expectation(&names)
            .unwrap_or_else(|error| {
                panic!("failed to arm dispatch expectation: {error}")
            });
    };
    let expect_one = |decl: &KnobDecl| expect(&[decl]);
    if verbose {
        let held: Vec<&str> = reg
            .iter()
            .filter(|d| d.bit_affecting)
            .map(|d| d.name)
            .collect();
        if !held.is_empty() {
            if allow_bits {
                println!(
                    "  bits: sweeping {} which CAN change output bits -- regenerate the \
                     pins and re-check agreement before trusting the result",
                    held.join(" ")
                );
            } else {
                println!(
                    "  bits: holding {} at their incumbent (they change output bits); \
                     --allow-bit-changes to sweep them",
                    held.join(" ")
                );
            }
        }
    }
    // LEGAL values only, decided by arithmetic before anything is dispatched. The screen
    // exists for candidates that are merely bad; a candidate that provably cannot run
    // should not cost a dispatch to find out -- a register-spilling shape measured 292x
    // off the pace and saturated the memory bus while it did.
    let legal_values = |d: &KnobDecl| -> Vec<u32> {
        // Candidates come from the device when the knob says so, and from the compiled
        // list otherwise. Either way the legality filter is the same.
        let base = match d.candidates {
            Some(f) => f(&facts, profile),
            None => d.values.to_vec(),
        };
        base.into_iter()
            .filter(|v| d.legal.is_none_or(|f| f(*v, &facts, profile)))
            .collect()
    };
    if verbose {
        let skipped: Vec<&str> = reg
            .iter()
            .filter(|d| !applicable(d))
            .map(|d| d.name)
            .collect();
        if !skipped.is_empty() {
            println!(
                "  model: {} of {} knobs do not apply to this model, skipped: {}",
                skipped.len(),
                reg.len(),
                skipped.join(" ")
            );
        }
    }
    let hidden = s.n_embd.max(s.n_ff);
    let tile = 512_u32; // prefill tokens per measurement
    for (id, n) in [
        (BufId::Cur, hidden * tile),
        (BufId::Logits, hidden * tile),
        (BufId::X, hidden * tile),
        (BufId::G, hidden * tile),
        (BufId::U, hidden * tile),
        (BufId::Q, s.head_dim * s.n_head * tile),
        (BufId::K, s.head_dim * s.n_kv * tile),
        (BufId::V, s.head_dim * s.n_kv * tile),
        (BufId::Attn, s.head_dim * s.n_head * tile),
        // The sliced decode kernels write one partial per (head, slice) HERE.
        // Without it they bound a nil buffer and their "timings" measured
        // writes that went nowhere -- which is why the span sweep read ~1.0x
        // at spans the standalone probe separates by 15-32%. Must match the
        // backend's slice cap (ATTN_MAX_SLICES) and the deep head dim.
        (
            BufId::AttnPart,
            s.n_head * 256 * (s.head_dim.max(s.deep_head_dim) + 2),
        ),
    ] {
        b.alloc(id, u64::from(n) * 4)
            .map_err(|rc| format!("alloc {id:?} rc={rc}"))?;
    }
    if let Some(ple) = s.ple_transaction {
        let ple_width = ple.gate.2;
        for (id, n) in [
            (BufId::Model0, ple_width * tile),
            (BufId::Model1, s.n_embd * tile),
            (BufId::Model2, ple.per_layer_stride * tile),
        ] {
            b.alloc(id, u64::from(n) * 4)
                .map_err(|rc| format!("alloc {id:?} rc={rc}"))?;
        }
        b.write(
            BufId::Model2,
            0,
            &vec![0.01_f32; (ple.per_layer_stride * tile) as usize],
        );
    }
    if let Some(shortconv) = s.shortconv_transaction {
        for (id, n) in [
            (BufId::Model3, shortconv.width * 3),
            (BufId::Model4, shortconv.width),
            (BufId::Recur, shortconv.state_elements),
        ] {
            b.alloc(id, u64::from(n) * 4)
                .map_err(|rc| format!("alloc {id:?} rc={rc}"))?;
        }
        b.write(
            BufId::Model3,
            0,
            &vec![0.01_f32; (shortconv.width * 3) as usize],
        );
        b.write(
            BufId::Recur,
            0,
            &vec![0.02_f32; shortconv.state_elements as usize],
        );
    }
    mark("buffers allocated");
    let kv_len = 1024_u32;
    // Layers for the span-crossing sweep: a decode-attention rep against ONE
    // layer is served from the SLC and stops measuring memory (a same-buffer
    // bench once read above this machine's DRAM peak), so the sweep rotates
    // layers and each needs its own span-sized cache.
    let span_ladder: Option<&[u32]> = reg.iter().find_map(|d| match d.sweep {
        SweepKind::SpanCrossing { ladder, .. } => Some(ladder),
        _ => None,
    });
    let deep_hd = s.head_dim.max(s.deep_head_dim);
    // KV DEPTH IS THE MAX OF WHAT ANY DEEP CONSUMER NEEDS, not whatever the span ladder
    // happens to want. Sizing it from the ladder alone coupled two unrelated things: when
    // the span-crossing knob stopped being swept, `deep_len` fell back to 1024 and every
    // "deep" workload silently became a shallow one -- attn_min_tgs went on reporting a
    // winner, measured at a depth 16x short of the one it is meant to rank.
    //
    // The ladder still decides how many LAYERS to allocate, because only the span sweep
    // needs to rotate them.
    //
    // AND IT MUST EXCEED THE READ POSITION BY A MARGIN, not by one. `attn_hd` passes this
    // length through as the cache's RING size, so sizing it to DEEP_DECODE_POS + 1 put the
    // wrap exactly at the position being scanned. attn_min_tgs then measured 432 us where
    // it had measured 603 -- a 1.4x swing on an unchanged knob at an unchanged depth,
    // which is the shape of a harness defect, not of noise.
    let (kv_layers, deep_len) =
        span_ladder.map_or((SPAN_LAYERS, DEEP_DECODE_POS * 2), |l| {
            (
                SPAN_LAYERS,
                l.iter()
                    .copied()
                    .max()
                    .unwrap_or(kv_len)
                    .max(DEEP_DECODE_POS * 2),
            )
        });
    // PER-LAYER SIZES, from the model's own geometry. A windowed layer's span can never
    // exceed its window, so giving it a full-depth cache buys nothing and costs hundreds of
    // megabytes; a full-attention layer needs the whole depth. `alloc_kv` already takes a
    // size per layer -- this used to hand it the same number 42 times.
    //
    // Each is sized a quarter past the span it will be read at, so the RING never wraps at
    // the read position. That wrap is what made attn_min_tgs swing 1.4x.
    let slots = |span: u32| span + span / 4;
    let kv_tag = b.kv_tag();
    let row_bytes = |width: u32| -> u64 {
        match kv_tag.as_str() {
            "f16" => u64::from(width) * 2,
            "q4_0" => u64::from(width / 32) * 18,
            "q8_0" => u64::from(width / 32) * 34,
            _ => unreachable!("tuner configures one canonical K/V type"),
        }
    };
    if !matches!(kv_tag.as_str(), "f16" | "q4_0" | "q8_0") {
        return Err(format!("unsupported tuner KV tag: {kv_tag}"));
    }
    // Quantized prefill dequantizes every initialized row into backend-owned half
    // scratch. The production workflows allocate that scratch from their KV capacity;
    // the synthetic tuner must uphold the same contract. Omitting these buffers let the
    // matmul-only warm-up pass and then failed the first real attention candidate with
    // CUDA_RC_INVALID, so no candidate config could ever be written or receipted.
    let scratch_slots = slots(deep_len.max(kv_len));
    let scratch_bytes =
        kv_scratch_bytes(kv_tag.as_str(), scratch_slots, s.n_kv, deep_hd);
    if scratch_bytes != 0 {
        for id in [BufId::Kdq, BufId::Vdq] {
            b.alloc(id, scratch_bytes)
                .map_err(|rc| format!("alloc {id:?} rc={rc}"))?;
        }
    }
    let kv_sizes: Vec<u64> = if s.attn_layers.is_empty() {
        vec![
            row_bytes(s.n_kv * deep_hd) * u64::from(slots(deep_len.max(kv_len)));
            kv_layers
        ]
    } else {
        s.attn_layers
            .iter()
            .map(|&(hd, window)| {
                // Full-attention layers must reach the deepest rung the boundary scan
                // walks; windowed ones can never be read past their window.
                let span = if window == 0 {
                    deep_len.max(DEEP_DECODE_POS)
                } else {
                    window.min(DEEP_DECODE_POS)
                };
                row_bytes(s.n_kv * hd) * u64::from(slots(span.max(kv_len)))
            })
            .collect()
    };
    let per_layer = *kv_sizes.iter().max().unwrap_or(&0);
    if verbose {
        let full = s.attn_layers.iter().filter(|(_, w)| *w == 0).count();
        println!(
            "  kv: {} layers ({full} full-attention, {} windowed), {:.0} MB total, \
             largest layer {:.0} MB",
            kv_sizes.len(),
            s.attn_layers.len() - full,
            kv_sizes.iter().sum::<u64>() as f64 / 1e6,
            per_layer as f64 / 1e6
        );
    }
    b.alloc_kv(&kv_sizes)
        .map_err(|rc| format!("alloc_kv rc={rc}"))?;
    // ALWAYS filled. This used to run only `if span_ladder.is_some()`, and no Metal knob is a
    // SpanCrossing any more, so every attention knob (attn_min_tgs, attn_stream_*,
    // attn_fd_chunk, attn_fa_nsg, attn_qcomb_*) was ranked on an all-zero K/V -- exactly the
    // data the note below explains is not neutral (review #116, D10).
    {
        // VALUE fill, once. Zeros make every score 0 (exp(0)=1, no softmax tail);
        // scores past f32 exp underflow give weights of exactly 0 and let the
        // streaming kernel skip V rows its rival does not; rows of one repeated
        // value are served cheaper than real data. Every element varies and the
        // magnitudes stay moderate.
        let f16 = |x: f32| -> u16 {
            let b = x.to_bits();
            let sign = ((b >> 16) & 0x8000) as u16;
            let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
            if x == 0.0 || exp <= 0 {
                return sign;
            }
            sign | ((exp as u16) << 10) | ((b >> 13) & 0x3ff) as u16
        };
        for layer in 0..kv_layers as u32 {
            let (layer_hd, window) = s
                .attn_layers
                .get(layer as usize)
                .copied()
                .unwrap_or((deep_hd, 0));
            let span = if window == 0 {
                deep_len.max(DEEP_DECODE_POS)
            } else {
                window.min(DEEP_DECODE_POS)
            };
            let staged_rows = slots(span.max(kv_len));
            let mut row = vec![0_u8; row_bytes(s.n_kv * layer_hd) as usize];
            for j in 0..staged_rows {
                let a =
                    ((j.wrapping_mul(2_654_435_761) >> 20) & 63) as f32 / 32.0 - 1.0;
                let c = ((j.wrapping_mul(40_503) >> 8) & 63) as f32 / 64.0;
                match kv_tag.as_str() {
                    "f16" => {
                        for (e, pair) in row.chunks_mut(2).enumerate() {
                            let value = if e % 2 == 0 { a } else { c - a * 0.5 };
                            pair.copy_from_slice(&f16(value).to_le_bytes());
                        }
                    }
                    "q4_0" => {
                        for (block, encoded) in row.chunks_exact_mut(18).enumerate() {
                            encoded[..2].copy_from_slice(&f16(0.125).to_le_bytes());
                            for (pair, packed) in encoded[2..].iter_mut().enumerate() {
                                let base = 2 * pair + block + j as usize;
                                let low = (base % 16) as u8;
                                let high = ((base + 1) % 16) as u8;
                                *packed = low | (high << 4);
                            }
                        }
                    }
                    "q8_0" => {
                        for (block, encoded) in row.chunks_exact_mut(34).enumerate() {
                            encoded[..2]
                                .copy_from_slice(&f16(1.0 / 64.0).to_le_bytes());
                            for (index, quant) in encoded[2..].iter_mut().enumerate() {
                                let value =
                                    ((index + block + j as usize) % 63) as i8 - 31;
                                *quant = value as u8;
                            }
                        }
                    }
                    _ => unreachable!("validated above"),
                }
                let off = u64::from(j) * row.len() as u64;
                b.write_kv_bytes(layer, false, off, &row);
                b.write_kv_bytes(layer, true, off, &row);
            }
        }
        let qn = (s.n_head * deep_hd) as usize;
        let q: Vec<f32> = (0..qn)
            .map(|i| ((i % 61) as f32 / 61.0 - 0.5) * 0.2)
            .collect();
        b.write(BufId::Q, 0, &q);
    }
    mark("kv staged");
    b.write(BufId::Cur, 0, &vec![0.01_f32; (hidden * tile) as usize]);

    // The widest matmul under test, which is the one the GEMM knobs should be judged
    // on -- a tile that wins on a narrow row can lose on a wide one -- at the WORK slice,
    // not the wider one the tile-ranking workload needs. This is a control for machine
    // drift, so what it must be is CHEAP and repeatable; it ranks nothing, and cache
    // residency does not change what it reports. Unsliced it is ~12.5 ms in one dispatch
    // and the warm-up runs it up to thirteen times until the clocks settle, which on its
    // own stalled the compositor for a dozen frames a run. A dispatch cannot be
    // preempted, so the only lever on that is how much work one carries.
    let slice = gemm_slice(profile, s);
    let (n_in, n_out) = (s.n_embd, s.n_ff.min(GEMM_WORK_SLICE));
    // THE MODEL'S KIND, not WK_Q4: the control exists to detect the machine drifting, and a
    // kernel the model never runs measures a different machine state (Q4 dequant on a Q8
    // model reads the Q4 path's occupancy, not the one every ranked workload is subject to).
    let reference = || {
        time_us(b, 3, 4, &|| {
            b.matmat(s.wide_kind, 0, n_in, n_out, BufId::Cur, BufId::Logits, tile);
        })
    };

    // Let the GPU reach steady clocks BEFORE anything is compared. GPUs ramp under
    // sustained load (the same reference read 28% faster across one warm-up here);
    // candidates measured early are charged for a cold device.
    let mut prev = reference();
    let mut warm = 1;
    for _ in 0..12 {
        let cur = reference();
        let moved = (cur - prev).abs() / prev.max(1e-9);
        prev = cur;
        warm += 1;
        if moved < 0.02 {
            break;
        }
    }
    if verbose {
        println!("  warm-up {warm} runs, settled at {prev:.0} us");
    }
    let ref_before = reference();

    // --- workload builders, from the registry's declared vocabulary.
    // The decode mix: the decode-step matmuls at the model's real offsets, summed --
    // one `lanes`/`sgs` value serves all of them, so no single one may choose it.
    let derived: Vec<(u64, u32, u32, u32)> = [
        (s.n_embd, s.n_head * s.head_dim),
        (s.n_embd, s.n_kv * s.head_dim),
        (s.n_embd, s.n_kv * s.head_dim),
        (s.n_head * s.head_dim, s.n_embd),
        (s.n_embd, s.n_ff),
        (s.n_embd, s.n_ff),
        (s.n_ff, s.n_embd),
    ]
    .into_iter()
    // The fallback shapes carry q4_0 explicitly. That is a GUESS, and it stopped being
    // a safe one when Q8_0 kernels landed: this path runs only when the gguf gave no
    // tensor list, so there is no kind to read and a Q8 model reaching here would be
    // measured on the wrong kernel. It is the same hazard `wide_kind` exists for, with
    // no information available to resolve it.
    .map(|(i, o)| (0, i, o, WK_Q4))
    .collect();
    let mix = if s.decode_mix.is_empty() {
        derived
    } else {
        s.decode_mix.clone()
    };
    // The complete transaction must be timed as the workflow invokes it. Every rep
    // dirties Cur with a semantic no-op so both routes pay their real activation
    // conversion instead of benchmarking an immortal Q8 cache entry.
    let run_ffn_transaction = |n_tok: u32, candidate_must_run: bool| {
        let shape = s
            .ffn_transaction
            .expect("FFN transaction workload requires a validated tensor triple");
        let (gate_off, gate_in, gate_out, gate_kind) = shape.gate;
        let (up_off, up_in, up_out, up_kind) = shape.up;
        let (down_off, down_in, down_out, down_kind) = shape.down;
        assert_eq!((gate_in, gate_out), (up_in, up_out));
        assert_eq!((down_in, down_out), (gate_out, gate_in));
        b.scale(BufId::Cur, 1.0, n_tok * gate_in);
        b.set_activation(s.activation);
        let fused = b.ffn_gated_down(
            gate_kind,
            gate_off,
            up_kind,
            up_off,
            down_kind,
            down_off,
            gate_in,
            gate_out,
            down_out,
            BufId::Cur,
            BufId::G,
            BufId::X,
            n_tok,
        );
        if candidate_must_run {
            assert!(
                fused,
                "FFN sidecar candidate became unreachable after its preflight probe"
            );
        }
        if !fused {
            let fused_pair = b.matmat_gated(
                gate_kind,
                gate_off,
                up_kind,
                up_off,
                gate_in,
                gate_out,
                BufId::Cur,
                BufId::G,
                BufId::U,
                n_tok,
            );
            if !fused_pair {
                b.matmat(
                    gate_kind,
                    gate_off,
                    gate_in,
                    gate_out,
                    BufId::Cur,
                    BufId::G,
                    n_tok,
                );
                let fused_epilogue = b.supports_epilogue(s.activation);
                if fused_epilogue {
                    b.set_epilogue(s.activation);
                }
                b.matmat(
                    up_kind,
                    up_off,
                    up_in,
                    up_out,
                    BufId::Cur,
                    if fused_epilogue { BufId::G } else { BufId::U },
                    n_tok,
                );
                if fused_epilogue {
                    b.set_epilogue(Epilogue::None);
                } else {
                    b.act_mul(BufId::G, BufId::U, n_tok * gate_out);
                }
            }
            b.matmat(
                down_kind,
                down_off,
                down_in,
                down_out,
                BufId::G,
                BufId::X,
                n_tok,
            );
        }
    };
    // Slot 39 controls D256 Attention, FFN, PLE and (for value 2) the majority local-Q
    // projection's exact-128 token tile. Ranking it on FFN+PLE
    // alone can admit a route whose attention loss is larger than both wins. Running
    // one of each is still the wrong weight for Gemma4 E4B: 35 of 42 layers use the
    // controlled D256 attention route, while all 42 execute FFN and PLE. Require the
    // complete layer geometry and replay those controlled stages in model order.
    let exact128_d256_layers = s
        .attn_layers
        .iter()
        .filter(|&&(head_dim, _)| head_dim == 256)
        .count();
    let exact128_base_evidence =
        exact128_route_evidence(exact128_d256_layers, s.attn_layers.len(), 0);
    let exact128_facts_complete = usize::try_from(s.n_layers).ok()
        == Some(s.attn_layers.len())
        && exact128_base_evidence.is_some()
        && s.exact128_local_q.is_some();
    let exact128_route_value = || {
        reg.iter()
            .find(|decl| decl.name == "prefill_exact128_sm86_route")
            .map_or(0, |decl| (decl.current)())
    };
    let exact128_expected_evidence = || {
        exact128_route_evidence(
            exact128_d256_layers,
            s.attn_layers.len(),
            if exact128_route_value() == 2 {
                exact128_d256_layers
            } else {
                0
            },
        )
    };
    let run_exact128_transaction = |candidate_must_run: bool| {
        assert!(
            exact128_facts_complete,
            "exact-128 route requires complete per-layer attention facts"
        );
        let ple = s
            .ple_transaction
            .expect("exact-128 route requires a validated PLE tensor pair");
        let (q_off, q_in, q_out, q_kind) = s
            .exact128_local_q
            .expect("exact-128 route requires a validated majority local-Q tensor");
        assert_eq!(q_in, s.n_embd);
        let (gate_off, gate_in, gate_out, gate_kind) = ple.gate;
        let (proj_off, proj_in, proj_out, proj_kind) = ple.proj;
        assert_eq!((gate_in, gate_out), (s.n_embd, proj_in));
        assert_eq!(proj_out, s.n_embd);
        for (layer, &(head_dim, window)) in s.attn_layers.iter().enumerate() {
            let layer = u32::try_from(layer).expect("attention layer index fits u32");
            if head_dim == 256 {
                b.matmat(q_kind, q_off, q_in, q_out, BufId::Cur, BufId::X, 128);
                b.attention(
                    layer,
                    head_dim,
                    s.n_head,
                    s.n_kv,
                    s.n_kv * head_dim,
                    0,
                    1.0,
                    window,
                    128,
                    scores_needed(128, window),
                    if window == 0 { 0 } else { kv_len - 1 },
                );
            }
            run_ffn_transaction(128, candidate_must_run);
            b.ple_project(
                gate_kind,
                gate_off,
                proj_kind,
                proj_off,
                s.n_embd,
                gate_out,
                BufId::Cur,
                BufId::Model0,
                BufId::Model2,
                layer
                    .checked_mul(gate_out)
                    .expect("per-layer embedding offset fits u32"),
                ple.per_layer_stride,
                BufId::Model1,
                128,
            );
        }
    };
    // Reachability is part of the evidence, not inferred from equal timings. The probe
    // also pays lazy sidecar packing/scratch setup outside the steady-state interval.
    let probe_ffn_candidate = |n_tok: u32| -> bool {
        let Some(shape) = s.ffn_transaction else {
            return false;
        };
        let geometry = BatchGeometry::try_new(0, n_tok, BatchPhase::Prefill).unwrap();
        b.set_batch_geometry(geometry)
            .expect("set FFN candidate probe geometry");
        b.set_activation(Epilogue::Gelu);
        b.begin();
        b.scale(BufId::Cur, 1.0, n_tok * shape.gate.1);
        let reached = b.ffn_gated_down(
            shape.gate.3,
            shape.gate.0,
            shape.up.3,
            shape.up.0,
            shape.down.3,
            shape.down.0,
            shape.gate.1,
            shape.gate.2,
            shape.down.2,
            BufId::Cur,
            BufId::G,
            BufId::X,
            n_tok,
        );
        b.end()
            .unwrap_or_else(|rc| panic!("FFN candidate preflight failed rc={rc}"));
        reached
    };
    let probe_exact128_candidate = || -> bool {
        if !exact128_facts_complete
            || s.ple_transaction.is_none()
            || !probe_ffn_candidate(128)
        {
            return false;
        }
        let geometry = BatchGeometry::try_new(0, 128, BatchPhase::Prefill).unwrap();
        b.set_batch_geometry(geometry)
            .expect("set exact-128 full-route probe geometry");
        b.begin();
        run_exact128_transaction(true);
        if b.end().is_err() {
            return false;
        }
        b.tuner_route_evidence(Workload::PrefillFfnExact128)
            == exact128_expected_evidence()
                .expect("complete exact-128 facts have countable route evidence")
    };
    // HEAD DIM IS A PARAMETER, because this model has two geometries and they do not
    // reach the same kernels. gemma4 E4B runs 256-dim WINDOWED layers in the majority and
    // 512-dim FULL-attention layers for the rest, and only the full-attention span grows
    // with the conversation -- a windowed layer's span never exceeds its window.
    //
    // Every DEEP workload here used s.head_dim, the majority, so "deep" meant deep in the
    // geometry that cannot go deep. IMPARO_ATTN_WHICH counted the dispatches: 6948 on the
    // 256-dim score-tile path against a few hundred streaming ones. That is why the
    // decode_stream tuple read INERT -- its kernels were a rounding error in its own
    // workload.
    // WINDOW AND RING ARE PART OF THE THING BEING MEASURED, not setup. Three engine
    // predicates test `window == 0` literally, and this passed window = len:
    //
    //     want_stream  needs window == 0 && ring == 0   -> streaming could NEVER run here
    //     want_gqa     needs window == 0                -> grouping could NEVER engage
    //
    // window = len is NUMERICALLY full attention -- lo clamps to 0 -- so the arithmetic
    // looked right and the kernel selection was silently wrong. Every knob measured on
    // these workloads was therefore ranked on the score-tile path alone, whatever the
    // engine would have picked. span_crossing already had this fixed; attn_hd did not, and
    // it feeds four workloads.
    //
    // A FULL-ATTENTION layer takes window 0 and ring 0, exactly as the engine dispatches
    // it; a WINDOWED layer takes its real window and a ring sized to hold it.
    let attn_hd = |n_tok: u32, pos: u32, len: u32, hd: u32, window: u32| {
        move || {
            b.attention(
                0,
                hd,
                s.n_head,
                s.n_kv,
                s.n_kv * hd,
                pos,
                1.0,
                window,
                n_tok,
                scores_needed(pos + n_tok, window),
                if window == 0 { 0 } else { len - 1 },
            );
        }
    };
    // The shallow workloads stand in for the WINDOWED layers, which is the geometry
    // s.head_dim describes, so they keep a window and the ring that serves it.
    let attn_at =
        |n_tok: u32, pos: u32, len: u32| attn_hd(n_tok, pos, len, s.head_dim, len);
    // The deep workloads run the geometry whose span actually grows -- the FULL-attention
    // layers -- so window 0 and ring 0, which is what makes the streaming and grouped
    // kernels reachable here at all.
    let attn_deep =
        |n_tok: u32, pos: u32, len: u32| attn_hd(n_tok, pos, len, s.deep_head_dim, 0);
    // THE GEOMETRY MOST LAYERS RUN (head dim, window), for the prefill-attention
    // cross-check. E4B: 512/0 on its few global layers is what the deep workload ranks;
    // 256 with a sliding window is what most layers are. A model with one geometry gets
    // that geometry at the SHORT position instead, so the second regime is never the first
    // one measured twice.
    let major = {
        let mut counts: Vec<((u32, u32), usize)> = Vec::new();
        for &g in &s.attn_layers {
            match counts.iter_mut().find(|(k, _)| *k == g) {
                Some((_, n)) => *n += 1,
                None => counts.push((g, 1)),
            }
        }
        counts
            .iter()
            .max_by_key(|(_, n)| *n)
            .map_or((s.deep_head_dim, 0), |(g, _)| *g)
    };
    let attn_major = |n_tok: u32| {
        let (hd, window) = major;
        if window == 0 && hd == s.deep_head_dim {
            attn_hd(n_tok, 512, kv_len, hd, 0)
        } else {
            let len = window.max(kv_len);
            attn_hd(n_tok, DEEP_DECODE_POS.max(len), len, hd, window)
        }
    };
    let attn = |n_tok: u32, pos: u32| attn_at(n_tok, pos, kv_len);
    // Deep enough that the streaming decode kernel engages, and CHOSEN rather than
    // inherited. This used to be the span ladder's top rung, on the reasoning that its KV
    // is allocated anyway -- convenient, and it made the depth of every deep workload a
    // side effect of an unrelated knob's ladder.
    //
    // It bit as soon as that ladder was extended to reach the depths the streaming kernel
    // exists for. deep_pos went from 8191 to 32767 and attn_min_tgs changed its answer from
    // 48 to 72, because the knob's optimum genuinely moves with depth. That is worth
    // knowing -- it is the evidence that this knob wants VARIANTS -- but it must not arrive
    // as an accident of someone editing a different knob's ladder.
    //
    // 16k is the depth to rank a decode knob at: it is a long agentic context, it is where
    // the end-to-end brackets in this repo are run, and it is what the shipped value is
    // therefore verified against.
    let deep_pos = DEEP_DECODE_POS.min(deep_len.saturating_sub(1)).max(kv_len);
    let scratch_by_workload = std::cell::RefCell::new(std::collections::HashMap::<
        Workload,
        ScratchImage,
    >::new());
    let run_workload = |w: Workload| -> f64 {
        if !scratch_by_workload.borrow().contains_key(&w) {
            let image = match w.effects() {
                WorkloadEffects::ModelRecurrentState => {
                    let shape = s.shortconv_transaction.expect(
                        "short-convolution workload requires a validated state shape",
                    );
                    ScratchImage::capture_regions(
                        b,
                        &[TunerScratchRegion::BufferF32 {
                            id: BufId::Recur,
                            off: 0,
                            elements: shape.state_elements as usize,
                        }],
                    )
                }
                effects => ScratchImage::capture(b, effects),
            };
            scratch_by_workload.borrow_mut().insert(w, image);
        }
        let images = scratch_by_workload.borrow();
        let image = images.get(&w).expect("workload scratch was captured");
        let timed = |samples: usize, reps: usize, f: &dyn Fn()| {
            time_us_reset(b, samples, reps, image.mutable, &|| image.restore(b), f)
        };
        match w {
            Workload::DecodeMix => {
                let layer: f64 = mix
                    .iter()
                    .map(|&(off, i, o, wk)| {
                        timed(3, 16, &move || {
                            b.matmat(wk, off, i, o, BufId::Cur, BufId::Logits, 1);
                        })
                    })
                    .sum();
                // The lm-head at its per-layer share: one dispatch per token against
                // n_layers layer mixes.
                let head = s.lm_head.map_or(0.0, |(off, i, o, wk)| {
                    timed(3, 16, &move || {
                        b.matmat(wk, off, i, o, BufId::Cur, BufId::Logits, 1);
                    }) / f64::from(s.n_layers.max(1))
                });
                layer + head
            }
            Workload::DecodeFfnTransaction => {
                assert!(
                    s.ffn_transaction.is_some(),
                    "Decode FFN transaction requires a named tensor triple"
                );
                timed(5, 8, &|| run_ffn_transaction(1, false))
            }
            Workload::DecodeShortconvTransaction => {
                let shape = s
                    .shortconv_transaction
                    .expect("short-convolution workload requires a validated shape");
                timed(5, 64, &|| {
                    b.shortconv(
                        BufId::Model3,
                        shape.weight_off,
                        BufId::Recur,
                        0,
                        BufId::Model4,
                        shape.width,
                        shape.kernel,
                        1,
                    );
                })
            }
            Workload::NarrowMix(n) => timed(3, 8, &|| {
                // THE MODEL'S OWN KIND, not a hardcoded Q4_0: this workload ranks the
                // prefill tile knobs, and each quant has its own tile family, so a Q8
                // shape knob measured against the Q4 GEMM is measuring a kernel it
                // cannot move.
                b.matmat(
                    s.wide_kind,
                    0,
                    s.n_embd,
                    s.n_ff,
                    BufId::Cur,
                    BufId::Logits,
                    n,
                );
                b.matmat(
                    s.wide_kind,
                    0,
                    s.n_ff,
                    s.n_embd,
                    BufId::Cur,
                    BufId::Logits,
                    n,
                );
            }),
            // 256 reps, not 16: this kernel is ~43 us and a 0.7 ms timed region is
            // dominated by command-buffer overhead -- its noise floor read 28-34%,
            // wider than any candidate's difference, so the knob could never switch.
            Workload::AttentionDecode => timed(3, 256, &attn(1, 512)),
            // TIMED WITHOUT THE WEIGHT STREAM, DELIBERATELY, and it was tried the other
            // way. Adding the mix per layer makes the region faithful in KIND and useless
            // in RESOLUTION: attention is ~19% of a decode step, so an 8.4% attention
            // difference compresses to ~1%, under this knob's 3% switch margin, and it
            // stops being able to switch at all.
            //
            //   contended    sweep read 48 -> 806.5, 72 -> 812.1, 96 -> 801.3  (~1% apart)
            //                so it KEPT 72
            //   end to end   score-tile @16k f16: min_tgs 48 -> 34.9, 72 -> 32.2
            //                48 is 8.4% FASTER, i.e. keeping 72 was wrong
            //
            // span_crossing can afford the mix because it fits LINES -- the marginal
            // isolates the span-dependent part and the fixed term absorbs the weights.
            // A single-point ranking like this one cannot.
            Workload::AttentionDecodeDeep => {
                // FORCE THE SCORE-TILE PATH. This workload ranks attn_min_tgs, which
                // governs score-tile's slice count and has NO effect on the streaming
                // kernel. The engine derives the boundary from the cache type (f16 8192,
                // quantized 512), so at deep_pos = 16384 the dispatch streams on every
                // cache type -- and the sweep was ranking a knob against a kernel it
                // cannot move. It picked 8 that way, against 48 measured end to end
                // (score-tile @16k f16: 48 -> 34.9, 72 -> 32.2).
                let boundary = reg.iter().find(|d| d.name == "attn_stream_min_pos");
                let restore = boundary.map(|d| KnobRestore {
                    knob: d,
                    value: (d.current)(),
                });
                if let Some(d) = boundary {
                    // 1 << 30, NOT u32::MAX: UINT32_MAX is the engine's "derive the
                    // boundary from the cache type" sentinel, so forcing it here would
                    // hand the dispatch 8192 on f16 and 512 on a quantized cache -- both
                    // below deep_pos, so it would stream, the exact opposite of what this
                    // forces. No span reaches 1 << 30, so score-tile is what runs.
                    (d.apply)(1 << 30);
                }
                let t = timed(3, 64, &attn_deep(1, deep_pos, deep_len.max(kv_len)));
                drop(restore);
                t
            }
            Workload::AttentionPrefill => timed(3, 2, &attn(tile, 512)),
            // LIGHT FIRST: the ranking is per KEY (the loop over the context is what a
            // prefill attention knob changes), so 64 queries rank it as 512 do, at an
            // eighth of the dispatch: 512 queries at 16k keys was 18 ms for the winning
            // simdgroup count and 186 ms for a spilling one -- unpreemptible, in a
            // sweep. 64 is still four 8-row query tiles, enough for a QT-16 variant.
            Workload::AttentionPrefillDeep => timed(
                3,
                2,
                &attn_deep(ATTN_DEEP_QUERIES, deep_pos, deep_len.max(kv_len)),
            ),
            // The tail a prefix match leaves. Small enough to miss the wide tile, large
            // enough to miss the decode path.
            // ONE DECODE STEP'S ATTENTION, layer by layer at each layer's own geometry.
            // Fewer reps because a step is 42 dispatches, not one.
            Workload::DecodeAttentionStep => timed(3, 4, &|| {
                for (li, &(hd, window)) in s.attn_layers.iter().enumerate() {
                    let span = if window == 0 {
                        DEEP_DECODE_POS
                    } else {
                        window.min(DEEP_DECODE_POS)
                    }
                    .max(kv_len);
                    let pos = span - 1;
                    // RING 0 FOR FULL ATTENTION, and it is not cosmetic: the streaming
                    // decode kernel refuses any dispatch with a ring
                    //
                    //   want_stream = ... && !gqa && hdi < 3 && window == 0 && ring == 0
                    //
                    // so handing every layer a ring made both sides of this race run the
                    // SAME score-tile kernel, and the ratio duly read 1.00 at every depth.
                    // IMPARO_ATTN_WHICH counted it: 26 604 dispatches, every one "split",
                    // none streaming. The engine passes 0 for full-attention layers and a
                    // ring only for windowed ones, which is also why only full-attention
                    // layers can ever take this path.
                    let ring = if window == 0 { 0 } else { slots(span) - 1 };
                    b.attention(
                        u32::try_from(li).unwrap_or(0),
                        hd,
                        s.n_head,
                        s.n_kv,
                        s.n_kv * hd,
                        pos,
                        1.0,
                        window,
                        1,
                        scores_needed(pos + 1, window),
                        ring,
                    );
                }
            }),
            Workload::AttentionPrefillReuse => {
                timed(3, 8, &attn_deep(REUSE_TAIL, deep_pos, deep_len.max(kv_len)))
            }
            Workload::AttentionPrefillMajor => {
                timed(3, 2, &attn_major(ATTN_DEEP_QUERIES))
            }
            // Both directions, because they load the tile differently: n_embd->n_ff is
            // wide-out and n_ff->n_embd is wide-in, and a shape can favour one.
            //
            // SLICED to `gemm_slice` outputs, which keeps the machine usable AND keeps
            // the weight tile past the measured cache knee.
            // The full pair is 2 x 26.8 GFLOP = ~8.9 ms of GPU in ONE dispatch, and a
            // dispatch cannot be split: `per_buf` bounds how many REPS share a command
            // buffer, so an 8.9 ms rep lands as an 8.9 ms burst however the gaps are
            // paced, and the compositor drops a frame every burst. Slicing the output
            // makes each dispatch ~0.9 ms while changing nothing about what is being
            // ranked: identical kernel, identical tile, identical per-threadgroup work,
            // only fewer threadgroups. The grid stays at 128 of them, comfortably above
            // the fill point, so the shapes are still compared on a full machine. Measure
            // the smallest problem that still answers the question.
            //
            // "the ~72 that fill this GPU" is what this comment used to say, borrowing
            // attn_min_tgs's literal. Both fill points are measured now and both read 16
            // here; 128 clears either. The borrowed number was never measured.
            Workload::PrefillGemm => {
                let out = s.n_ff.min(GEMM_WORK_SLICE);
                time_us(b, 3, 4, &move || {
                    // `wide_kind`, not WK_Q4: this arm ranks st_gemm_shape and
                    // q8_full_tiles, and its doc has claimed since the field was added
                    // that it reads the model's kind here. It did not.
                    b.matmat(
                        s.wide_kind,
                        0,
                        s.n_embd,
                        out,
                        BufId::Cur,
                        BufId::Logits,
                        tile,
                    );
                    b.matmat(
                        s.wide_kind,
                        0,
                        s.n_ff,
                        out.min(s.n_embd),
                        BufId::Cur,
                        BufId::Logits,
                        tile,
                    );
                })
            }
            // THE WIDTHS A REAL PREFILL SENDS, not one of them. A request of N tokens is
            // chunked into whole ubatches plus a remainder, so a knob whose answer
            // depends on the dispatch's token count has to be judged on both. The
            // remainder is deliberately NOT a multiple of the wide tile: that is the case
            // where the engine's padding rule takes the narrow tile, and a workload that
            // only ever sends exact ubatches would never see it.
            Workload::PrefillGemmWidths => {
                let out = slice;
                time_us(b, 3, 2, &move || {
                    for w in RANK_WIDTHS {
                        b.matmat(
                            s.wide_kind,
                            0,
                            s.n_embd,
                            out,
                            BufId::Cur,
                            BufId::Logits,
                            w,
                        );
                        b.matmat(
                            s.wide_kind,
                            0,
                            s.n_ff,
                            out.min(s.n_embd),
                            BufId::Cur,
                            BufId::Logits,
                            w,
                        );
                    }
                })
            }
            // THE PAIR'S SECOND TILE: the FULL projection (the tile must leave the
            // last-level cache for the re-read trade to exist -- see GEMM_SLICE_PAST_KNEE
            // for the measured ladder) at the FEWEST tokens that exercise every candidate
            // (`pair_tile_tokens`, the widest token tile in the table). Fewer tokens is
            // what keeps this light: the weight-read share per token does not depend on
            // the token count, so 128 tokens ranks what 512 ranks (+10.4% on a 2.3% floor
            // against +4.0% on 2.0%; same order) at ~1.4 ms per dispatch instead of ~5 ms
            // -- and 5 ms dispatches back to back froze the machine. Both directions, as
            // above.
            // The pair projection at the chunk width the engine dispatches (`tile` = 512
            // tokens): the regime of the LARGE shape. The 128-token pair tile below is the
            // regime of the first tile only.
            Workload::PrefillGemmChunk => {
                let out = s.n_ff;
                time_us(b, 3, 2, &move || {
                    b.matmat(
                        s.wide_kind,
                        0,
                        s.n_embd,
                        out,
                        BufId::Cur,
                        BufId::Logits,
                        tile,
                    );
                    b.matmat(
                        s.wide_kind,
                        0,
                        s.n_ff,
                        out.min(s.n_embd),
                        BufId::Cur,
                        BufId::Logits,
                        tile,
                    );
                })
            }
            Workload::PrefillGemmUbatch => {
                let out = s.n_ff;
                let w = s.pair_tile_tokens.max(64);
                // Five samples of four reps: this region is ~2.8 ms per rep and a short
                // region's floor swings (2.3% and 8.3% in two runs of the same knob at
                // 3 x 2); more samples per measurement tighten the floor without
                // lengthening any dispatch.
                time_us(b, 5, 4, &move || {
                    b.matmat(
                        s.wide_kind,
                        0,
                        s.n_embd,
                        out,
                        BufId::Cur,
                        BufId::Logits,
                        w,
                    );
                    b.matmat(
                        s.wide_kind,
                        0,
                        s.n_ff,
                        out.min(s.n_embd),
                        BufId::Cur,
                        BufId::Logits,
                        w,
                    );
                })
            }
            Workload::PrefillFfnTransaction => {
                assert!(
                    s.ffn_transaction.is_some(),
                    "FFN transaction workload is inapplicable without a named tensor triple"
                );
                let geometry =
                    BatchGeometry::try_new(0, tile, BatchPhase::Prefill).unwrap();
                b.set_batch_geometry(geometry)
                    .expect("set FFN transaction batch geometry");
                time_us(b, 3, 2, &|| run_ffn_transaction(tile, false))
            }
            Workload::PrefillFfnExact128 => {
                assert!(
                    exact128_facts_complete
                        && s.ffn_transaction.is_some()
                        && s.ple_transaction.is_some(),
                    "exact-128 route requires complete layer facts and validated PLE/FFN tensors"
                );
                let geometry =
                    BatchGeometry::try_new(0, 128, BatchPhase::Prefill).unwrap();
                b.set_batch_geometry(geometry)
                    .expect("set exact-128 FFN batch geometry");
                time_us(b, 3, 2, &|| run_exact128_transaction(false))
            }
        }
    };

    // ONE LAYER'S DECODE WORK, timed here rather than in discovery because it needs the
    // staged weights, and discovery runs before them. It is the missing term in the
    // command-buffer trade: the fixed submission cost is only worth paying when a buffer
    // carries enough work to cover it, and "enough" is measured in layers.
    let mut owned = *profile;
    if owned.layer_work_ns == 0 {
        let _ = run_workload(Workload::DecodeMix); // discarded: first read carries setup
        let mut t: Vec<f64> =
            (0..5).map(|_| run_workload(Workload::DecodeMix)).collect();
        t.sort_by(f64::total_cmp);
        owned.layer_work_ns = (t[2] * 1000.0) as u32;
        // DecodeMix is the layer's MATMULS -- the projections and the FFN, at the model's
        // real dimensions. It excludes the norms, rope and attention that also dispatch in
        // a real layer, so it is a floor on a layer's work. The DISPATCH count comes from
        // the model's forward code instead (ModelFacts::layer_dispatches), because that
        // one cannot be measured from a matmul-only workload.
        // The same term for PREFILL. PrefillGemm is SLICED to `gemm_slice` outputs to keep
        // the machine usable, so scale it back to the layer's real width; and it is the
        // matmuls only, so like the decode term it is a FLOOR on a layer's work. A floor
        // is the safe direction here: it makes the encode lag look relatively larger, so
        // the derivation errs toward flushing rather than away from it.
        let _ = run_workload(Workload::PrefillGemm);
        let mut tp: Vec<f64> = (0..3)
            .map(|_| run_workload(Workload::PrefillGemm))
            .collect();
        tp.sort_by(f64::total_cmp);
        // `slice` is what the workload actually dispatched, and gemm_slice never returns
        // zero, so the ratio is the real scale-up with no clamp needed.
        // PrefillGemm's own width, not the ranking workload's: this scales its measured
        // matmul time back up to the layer's real width, and it is the work term that
        // feeds the command-buffer trade.
        let work_slice = s.n_ff.min(GEMM_WORK_SLICE);
        let scale = f64::from(s.n_ff.max(1)) / f64::from(work_slice.max(1));
        owned.layer_work_prefill_ns = (tp[1] * scale * 1000.0) as u32;
        if verbose {
            println!(
                "  layer: {:.1} us of decode matmul, {:.0} us of prefill matmul (the \
                 command-buffer trade's work terms)",
                f64::from(owned.layer_work_ns) / 1000.0,
                f64::from(owned.layer_work_prefill_ns) / 1000.0
            );
        }
    }
    let profile = &owned;
    b.set_tuner_dispatch_expectation(&[])?;

    // --- SCREEN every `screened` knob's candidates on a tiny matmul first.
    //
    // A candidate whose registers spill runs orders of magnitude slower, and A DISPATCH
    // CANNOT BE INTERRUPTED: one spilling rt shape is a multi-second command buffer
    // that freezes every app on the machine. 512x512 at 32 tokens tells candidates
    // apart for ~22 ms total; a candidate 60x off the pace does not go on to win, so
    // rejecting it costs no accuracy. The rule is a ratio, not a list: which candidate
    // falls out is for the DEVICE to say, not this file.
    //
    // The probe must actually RUN the candidate under test, so every Crossing knob is
    // forced to its `lo` (routing off) first -- with the narrow-N routing live, an
    // n_tok=32 matmat takes the nb8 tile and every rt shape screens identical, which
    // is exactly how the spilling shape once reached stage 2 and froze the machine.
    // SMALL on purpose. The screen decides by RATIO -- a register-spilling candidate
    // reads ~80x the best -- and a ratio is preserved at any problem size, so the size
    // should be chosen for what it costs the machine when the candidate IS the bad one.
    // At 512x512x32 a spiller took 21.12 ms in a single unpreemptible dispatch and froze
    // the UI for several frames; the screen was doing exactly its job, at the user's
    // expense. A sixteenth of the work keeps the same ratio and the same verdict.
    // THE PROBE RUNS THE MODEL'S WEIGHT KIND, not Q4 for everyone. `q8_design` chooses the
    // Q8 GEMM's design; a Q4 matmat never reads it, so all eight of its candidates screened
    // identical and rt_gemm<Q8> shape 7 -- 12 s per buffer at the workload's size, ~1000x
    // the others -- reached the sweep and held the machine for minutes (task #111).
    let screen_probe = || {
        let t = std::time::Instant::now();
        b.reset_tuner_dispatch_proof();
        b.begin();
        b.matmat(s.wide_kind, 0, 256, 256, BufId::Cur, BufId::Logits, 8);
        b.end().unwrap_or_else(|rc| {
            panic!("tuner screen GPU submission failed with backend code {rc}")
        });
        b.validate_tuner_dispatch_proof().unwrap_or_else(|error| {
            panic!("tuner screen dispatch proof failed: {error}")
        });
        device_or_wall_us(b, t.elapsed().as_secs_f64() * 1e6, "a GEMM screen probe")
    };
    // AN ATTENTION KNOB IS SCREENED ON AN ATTENTION DISPATCH. The matmul above cannot
    // rank a simdgroup count of the attention kernel (every candidate reads identical),
    // and a spilling one -- attn_fa_nsg=1 is 10x the winner -- would then go on to the
    // deep workload unscreened. Same rule, same ratio: tiny (64 queries over 1024 keys,
    // ~0.1 ms for the winner), decided by the DEVICE.
    let screen_probe_attn = || {
        let t = std::time::Instant::now();
        b.reset_tuner_dispatch_proof();
        b.begin();
        attn_deep(64, 1024, deep_len.max(kv_len))();
        b.end().unwrap_or_else(|rc| {
            panic!("tuner screen GPU submission failed with backend code {rc}")
        });
        b.validate_tuner_dispatch_proof().unwrap_or_else(|error| {
            panic!("tuner screen dispatch proof failed: {error}")
        });
        device_or_wall_us(
            b,
            t.elapsed().as_secs_f64() * 1e6,
            "an Attention screen probe",
        )
    };
    let screen_probe_attn_decode = || {
        let t = std::time::Instant::now();
        b.reset_tuner_dispatch_proof();
        b.begin();
        attn(1, 512)();
        b.end().unwrap_or_else(|rc| {
            panic!("tuner screen GPU submission failed with backend code {rc}")
        });
        b.validate_tuner_dispatch_proof().unwrap_or_else(|error| {
            panic!("tuner screen dispatch proof failed: {error}")
        });
        device_or_wall_us(
            b,
            t.elapsed().as_secs_f64() * 1e6,
            "an Attention Decode screen probe",
        )
    };
    mark("warm-up done");
    let mut screened_out: Vec<(&'static str, Vec<u32>)> = Vec::new();
    for d in reg
        .iter()
        .filter(|d| d.screened && applicable(d) && selected(d))
    {
        expect_one(d);
        let legal_here = legal_values(d);
        if verbose && legal_here.len() < d.values.len() {
            let bad: Vec<String> = d
                .values
                .iter()
                .filter(|v| !legal_here.contains(v))
                .map(ToString::to_string)
                .collect();
            println!(
                "  {}: {} illegal on this device, never dispatched: {}",
                d.name,
                bad.len(),
                bad.join(" ")
            );
        }
        let saved_routing: Vec<(&KnobDecl, u32)> = reg
            .iter()
            .filter_map(|r| match r.sweep {
                // Only the CROSSING kinds are pinned to their low end here; a derived
                // knob has no end to pin and a Values knob is swept, not crossed.
                SweepKind::Crossing { lo, .. }
                | SweepKind::TokenMinCrossing { lo, .. }
                | SweepKind::SpanCrossing { lo, .. } => {
                    let cur = (r.current)();
                    (r.apply)(lo);
                    Some((r, cur))
                }
                SweepKind::Derived | SweepKind::External | SweepKind::Values => None,
            })
            .collect();
        // Read the incumbent BEFORE probing: `apply` on an unavailable candidate leaves
        // the current value in place, which is also the availability test.
        let incumbent = (d.current)();
        let mut readings: Vec<(u32, f64)> = Vec::new();
        for &v in &legal_here {
            (d.apply)(v);
            if (d.current)() != v {
                continue;
            } // this device cannot run it
            let probe: &dyn Fn() -> f64 = match d.workload {
                Workload::AttentionPrefillDeep
                | Workload::AttentionPrefillReuse
                | Workload::AttentionPrefillMajor => &screen_probe_attn,
                Workload::AttentionDecode => &screen_probe_attn_decode,
                _ => &screen_probe,
            };
            let _ = probe(); // untimed: pays the switch
            // MIN of five, not one sample: on a host that is being USED, a single
            // wall-clock probe measures whoever else had the GPU that millisecond.
            // The min is the uncontended estimate; a real spiller reads ~80x even at min.
            let best = (0..5)
                .map(|_| {
                    let v = probe();
                    pace(v);
                    v
                })
                .fold(f64::MAX, f64::min);
            readings.push((v, best));
        }
        (d.apply)(incumbent);
        for (r, v) in saved_routing {
            (r.apply)(v);
        }
        let best = readings.iter().map(|(_, us)| *us).fold(f64::MAX, f64::min);
        let survivors: Vec<u32> = readings
            .iter()
            .filter(|(v, us)| {
                let keep = *us <= best * SCREEN_MAX;
                if !keep && verbose {
                    println!(
                        "  {}={v} rejected: {:.2} ms on the screen against \
                              {:.2} ms best",
                        d.name,
                        us / 1e3,
                        best / 1e3
                    );
                }
                keep
            })
            .map(|(v, _)| *v)
            .collect();
        screened_out.push((d.name, survivors));
    }
    // The screen's verdict, as a lookup the sweep below MUST consult. It did not, and
    // that is the whole reason a tuner run took minutes and stalled the display: the
    // screen correctly identified rt_shape=7 as 292x off the pace and printed
    // "rejected", then the sweep measured it anyway -- three rounds of a
    // register-spilling kernel hammering device memory, about 30 seconds of it, while
    // every other candidate took 0.05 s. The list was collected, named `screened_out`
    // when it actually holds SURVIVORS, and only ever reported.
    let survivors_by_knob: std::collections::HashMap<&'static str, Vec<u32>> =
        screened_out.iter().cloned().collect();

    // --- Values sweep: incumbent-holds-the-seat, interleaved rounds, per-axis noise
    // floor measured by the same function that judges the axis (floors do not transfer
    // between workloads: the 512-token GEMM repeats to 0.4% while the decode GEMV
    // beside it moves 19%).
    let sweep = |d: &KnobDecl| -> u32 {
        expect_one(d);
        let mut current = reg
            .iter()
            .position(|x| x.name == d.name)
            .and_then(|i| seats.get(i).copied())
            .unwrap_or_else(|| (d.current)());
        if d.workload == Workload::PrefillFfnExact128
            && current != 0
            && !probe_exact128_candidate()
        {
            (d.apply)(0);
            current = 0;
        }
        (d.apply)(current);
        let time_one = || {
            if d.workload == Workload::PrefillFfnExact128 {
                let geometry =
                    BatchGeometry::try_new(0, 128, BatchPhase::Prefill).unwrap();
                b.set_batch_geometry(geometry)
                    .expect("set exact-128 sweep geometry");
                let enabled = (d.current)() != 0;
                time_us_checked(
                    b,
                    3,
                    2,
                    Some((
                        Workload::PrefillFfnExact128,
                        if enabled {
                            exact128_expected_evidence().expect(
                                "exact-128 sweep requires complete countable model facts",
                            )
                        } else {
                            0
                        },
                    )),
                    &|| run_exact128_transaction(enabled),
                )
            } else {
                run_workload(d.workload)
            }
        };
        let _ = time_one(); // discarded: the first read carries the state change
        let mut probe: Vec<f64> = (0..5).map(|_| time_one()).collect();
        probe.sort_by(f64::total_cmp);
        let noise = (probe[4] - probe[0]) / probe[2].max(1e-9);
        // BEAT THE OBSERVED SPREAD, not a written 3%. `noise` is the full range of five
        // probes over their median -- already a conservative estimate, since a candidate
        // has to clear the worst-to-best swing of the incumbent. The floor here used to
        // be 0.03, with nothing saying why, and on a quiet axis that is 30x the noise the
        // same function just measured.
        //
        // It cost a real win. Tuning space v15 restarted every knob from its compiled
        // default, because a knob-space bump discards the stored file, and then:
        //     q8_full_tiles noise floor 0.1%, margin to switch 3.0%
        //     q8_full_tiles=0   3280.2 us vs 3379.1      2.9% faster
        //     q8_full_tiles=1   kept (no candidate beat it by 3%)
        // The 3% floor protected a value that had never been measured, and the shipped
        // config regressed E4B's deep prefill 535.0 -> 530.8 tok/s.
        //
        // The 0.005 floor remains so a pathologically quiet probe cannot make every
        // difference look significant. Noisy axes are unchanged: the decode GEMV moves
        // 19% and still demands 19%.
        let margin = noise.max(0.005);
        if verbose {
            println!(
                "  {} noise floor {:.1}%, margin to switch {:.1}%",
                d.name,
                noise * 100.0,
                margin * 100.0
            );
        }
        // Only what survived the screen. The incumbent always keeps its seat even if the
        // screen disliked it -- refusing to measure the value we are actually shipping
        // would leave nothing to compare against.
        let legal_all = legal_values(d);
        let allowed: &[u32] = survivors_by_knob
            .get(d.name)
            .map_or(legal_all.as_slice(), |v| v.as_slice());
        let mut cands: Vec<u32> = vec![current];
        for &value in allowed.iter().filter(|value| **value != current) {
            (d.apply)(value);
            if d.workload == Workload::PrefillFfnExact128
                && value != 0
                && !probe_exact128_candidate()
            {
                continue;
            }
            cands.push(value);
        }
        (d.apply)(current);
        if verbose && allowed.len() < d.values.len() {
            println!(
                "  {}: {} of {} values survived the screen",
                d.name,
                allowed.len(),
                d.values.len()
            );
        }
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); cands.len()];
        for _ in 0..ROUNDS {
            for (i, &v) in cands.iter().enumerate() {
                (d.apply)(v);
                samples[i].push(time_one());
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let base = med(&mut samples[0]);
        let mut best = (base, current);
        for (i, &v) in cands.iter().enumerate().skip(1) {
            let us = med(&mut samples[i]);
            let win = us < best.0 * (1.0 - margin);
            if verbose {
                println!(
                    "  {}={v:<5} {us:>9.1} us vs {base:>9.1}{}",
                    d.name,
                    if win { "  <-" } else { "" }
                );
            }
            if win {
                best = (us, v);
            }
        }
        if verbose && best.1 == current {
            println!(
                "  {}={current:<5} kept (no candidate beat it by {:.0}%)",
                d.name,
                margin * 100.0
            );
        }
        // INERT CHECK. If no candidate moved the clock beyond the noise floor, this knob
        // does not govern the workload it is declared on -- and that is the defect this
        // registry produced five times: rt_shape, attn_blk, nr0, attn_min_tgs and the
        // three streaming knobs were each declared against a workload that never reached
        // the kernel they select. Every one of them duly ranked its candidates within
        // noise of each other and picked the incumbent, which reads exactly like "the
        // default was already best".
        //
        // A knob whose candidates all measure the same is not a knob that is well tuned.
        // It is a knob being measured on the wrong thing, or a knob that no longer wires
        // to anything. Say which two possibilities they are, because the fix differs.
        if verbose && cands.len() >= 2 {
            let mut times: Vec<f64> =
                (0..cands.len()).map(|i| med(&mut samples[i])).collect();
            times.sort_by(f64::total_cmp);
            let (lo, hi) = (times[0], times[times.len() - 1]);
            if hi < lo * (1.0 + margin) {
                if margin > UNRESOLVABLE_FLOOR {
                    println!(
                        "  {} UNRESOLVED on {:?}: all {} candidates within a {:.1}% noise \
                         floor, which is too wide to decide anything. Not evidence the \
                         knob is unwired -- evidence this workload is too noisy to rank it.",
                        d.name,
                        d.workload,
                        cands.len(),
                        margin * 100.0
                    );
                } else {
                    println!(
                        "  {} INERT on {:?}: all {} candidates within {:.1}%, on a floor \
                         tight enough to have seen a difference -- either this workload \
                         never reaches the kernel it selects, or the knob no longer wires \
                         to anything",
                        d.name,
                        d.workload,
                        cands.len(),
                        margin * 100.0
                    );
                }
            }
        }
        // CROSS-CHECK, the same question the tuple path asks and for the same reason: race
        // the winner against the best alternative on a SECOND regime, and if the order
        // flips then no single value serves both. A CONST knob is where this matters most
        // -- it ships one value for every regime by definition, so a flip is the signal
        // that it should not be a CONST at all.
        // THE SEAT DEFENDS ITSELF ON THE SECOND REGIME. A winner on this knob's own workload
        // replaces the stored value only if the value it replaces is not faster than it by
        // the margin on the cross-check regime -- a knob one value cannot serve in both
        // regimes keeps what the engine already runs. Until 2026-09-02 this block only
        // PRINTED the flip and applied the winner anyway (attn_min_tgs=8 was written that
        // way against a short-regime cross-check that disagreed).
        // CONFIRM THE SWITCH HEAD TO HEAD. The seat's `base` is one median taken before the
        // candidates, and one outlier there makes every candidate a winner: the q8_narrow
        // seat read 916 us in one run and 991 us in the next while the candidates sat at
        // 895-995 (2026-09-02), and the E4B prefill-attention seat read 6930 then 6403 us
        // the same way -- both wrote a switch the engine measured as a loss. A winner is
        // written only if it still beats the seat by the margin when the two are timed
        // together, interleaved, in alternating order.
        if best.1 != current {
            let (mut su, mut wu) = (Vec::new(), Vec::new());
            for r in 0..3 {
                let order = if r % 2 == 0 {
                    [current, best.1]
                } else {
                    [best.1, current]
                };
                for v in order {
                    (d.apply)(v);
                    let t = time_one();
                    if v == current { su.push(t) } else { wu.push(t) }
                }
            }
            let (su, wu) = (med(&mut su), med(&mut wu));
            if wu < su * (1.0 - margin) {
                if verbose {
                    println!(
                        "  {}={} confirmed head to head ({wu:.1} vs the seat {current}'s \
                         {su:.1} us)",
                        d.name, best.1
                    );
                }
            } else {
                if verbose {
                    println!(
                        "  {}={} NOT CONFIRMED head to head ({wu:.1} vs the seat {current}'s \
                         {su:.1} us, margin {:.1}%): the base sample was the outlier; \
                         keeping {current}",
                        d.name,
                        best.1,
                        margin * 100.0
                    );
                }
                best = (base, current);
            }
        }
        if let (Some(cw), true) = (d.cross_check, best.1 != current) {
            let on = |v: u32| {
                (d.apply)(v);
                let _ = run_workload(cw);
                let mut t: Vec<f64> = (0..3).map(|_| run_workload(cw)).collect();
                t.sort_by(f64::total_cmp);
                t[1]
            };
            let (wv, wu, iu) = (best.1, on(best.1), on(current));
            if iu < wu * (1.0 - margin) {
                if verbose {
                    println!(
                        "  {} CROSS-CHECK FLIP on {cw:?}: {wv} wins its own regime but the \
                         seat {current} is {:.0}% faster here ({iu:.1} vs {wu:.1} us) -- one \
                         value does not serve both; keeping {current}",
                        d.name,
                        (wu / iu - 1.0) * 100.0
                    );
                }
                best = (base, current);
            } else if verbose {
                println!(
                    "  {} cross-check on {cw:?}: {wv} holds ({wu:.1} vs the seat {current}'s \
                     {iu:.1} us)",
                    d.name
                );
            }
        }
        (d.apply)(best.1);
        best.1
    };

    // --- Crossing scan: at each ladder rung n, the knob's own kernel (forced `hi`)
    // races the alternative (forced `lo`) on an n-token matmat. The pick is the last
    // width where the hi-side keeps winning; three scans must agree within one rung or
    // the compiled default stands. (The old min_tok scan recorded that its crossing
    // "never reproduced" -- that was the task-#5 wobble perturbing one side, root-caused
    // since as the engine's arena aliasing and fixed; both sides are honest to time now.)
    let crossing = |d: &KnobDecl, ladder: &[u32], hi: u32, lo: u32| -> u32 {
        expect_one(d);
        let default = (d.current)();
        let scan = || {
            let mut ratio = vec![0.0_f64; ladder.len()];
            for (i, &n) in ladder.iter().enumerate() {
                let one = |boundary: u32| {
                    (d.apply)(boundary);
                    time_us(b, 3, 8, &move || {
                        // THE MODEL'S OWN KIND, for the same reason `NarrowMix` gives:
                        // each quant has its own route family, so a Q8 boundary forced
                        // against the Q4 GEMM moves nothing. Measured on LFM2.5 (Q8_0)
                        // with the hardcoded Q4: `q8_gemv_max_tok` read hi 1.00x at every
                        // rung of [2,4,8,16,32] -- three scans found [32, 32, 1], the
                        // agreement guard fired, and the default stood. End to end the
                        // same routing choice was 89 ms against 60 ms.
                        b.matmat(
                            s.wide_kind,
                            0,
                            s.n_embd,
                            s.n_ff,
                            BufId::Cur,
                            BufId::Logits,
                            n,
                        );
                    })
                };
                let hi_us = one(hi);
                let lo_us = one(lo);
                ratio[i] = lo_us / hi_us.max(1e-9); // > 1: the hi-side kernel wins
            }
            let found = ladder
                .iter()
                .enumerate()
                .find(|(i, _)| {
                    ratio[*i] < 1.0 - CROSSOVER_MARGIN
                        && ratio.get(i + 1).is_none_or(|r| *r < 1.0)
                })
                .map(|(_, &n)| n - 1);
            (found, ratio)
        };
        let mut found: Vec<u32> = Vec::new();
        let mut last = Vec::new();
        for _ in 0..3 {
            let (f, r) = scan();
            // No crossing on the ladder. Two different situations: the hi side WON at every
            // rung (take the top rung), or the two sides TIED at every rung -- no information,
            // e.g. a Q4-only knob on a Q8 model, where both arms dispatch the same kernel.
            // The tie used to be read as a win and wrote the top rung (gemv_max_tok=16 on
            // LFM2, review #116 D11); it keeps the default now.
            let no_information = r.iter().all(|x| (x - 1.0).abs() < CROSSOVER_MARGIN);
            last = r;
            found.push(f.unwrap_or_else(|| {
                if no_information {
                    default
                } else {
                    *ladder.last().unwrap()
                }
            }));
        }
        let mut sorted = found.clone();
        sorted.sort_unstable();
        let pos = |v: u32| {
            ladder
                .iter()
                .position(|n| n.saturating_sub(1) == v || *n == v)
        };
        let agrees = matches!((pos(sorted[0]), pos(sorted[2])),
                              (Some(a), Some(b)) if b - a <= 1)
            || sorted[0] == sorted[2];
        let pick = if agrees { sorted[1] } else { default };
        if verbose {
            for (i, &n) in ladder.iter().enumerate() {
                print!("  n_tok={n:<3} hi {:.2}x", last[i]);
            }
            println!();
            println!("  {}: scans found {found:?} -> {pick}", d.name);
        }
        (d.apply)(pick);
        pick
    };

    // --- Minimum-token crossing for a complete semantic operation. Unlike Crossing,
    // the candidate owns the LARGE side. Every rung first proves that the candidate
    // actually dispatches, then races candidate/control in h/l/l/h/h/l order. A single
    // threshold is accepted only when three scans agree and each scan contains at most
    // one loss->win transition; a non-monotonic route map is not representable by this
    // knob and therefore fails closed.
    let token_min_crossing = |d: &KnobDecl, ladder: &[u32], hi: u32, lo: u32| -> u32 {
        let default = (d.current)();
        if d.workload != Workload::PrefillFfnTransaction || s.ffn_transaction.is_none()
        {
            if verbose {
                println!(
                    "  {}: no validated full FFN tensor triple; keeping safe {}",
                    d.name, default
                );
            }
            (d.apply)(default);
            return default;
        }
        let med3 = |mut values: [f64; 3]| {
            values.sort_by(f64::total_cmp);
            values[1]
        };
        let spread = |values: [f64; 3], median: f64| {
            let lo = values.iter().copied().fold(f64::MAX, f64::min);
            let hi = values.iter().copied().fold(0.0_f64, f64::max);
            (hi - lo) / median.max(1e-9)
        };
        let scan = || {
            let mut rows: Vec<(u32, bool, f64, f64, f64, bool)> =
                Vec::with_capacity(ladder.len());
            for &n in ladder {
                (d.apply)(hi);
                expect_one(d);
                let reachable = probe_ffn_candidate(n);
                if !reachable {
                    rows.push((n, false, 0.0, 0.0, 0.0, false));
                    continue;
                }
                let one = |value: u32| {
                    (d.apply)(value);
                    if value == hi {
                        expect_one(d);
                    } else {
                        expect(&[]);
                    }
                    b.set_batch_geometry(
                        BatchGeometry::try_new(0, n, BatchPhase::Prefill).unwrap(),
                    )
                    .expect("set FFN crossing geometry");
                    time_us(b, 1, 2, &|| {
                        run_ffn_transaction(n, value == hi);
                    })
                };
                // Contemporary, order-reversed legs cancel slow thermal drift.
                let h1 = one(hi);
                let l1 = one(lo);
                let l2 = one(lo);
                let h2 = one(hi);
                let h3 = one(hi);
                let l3 = one(lo);
                let hv = [h1, h2, h3];
                let lv = [l1, l2, l3];
                let h = med3(hv);
                let l = med3(lv);
                let noise = spread(hv, h).max(spread(lv, l));
                let gain = (l - h) / l.max(1e-9);
                let wins = gain > CROSSOVER_MARGIN.max(noise);
                rows.push((n, true, h, l, noise, wins));
            }
            let decisions: Vec<(u32, bool, bool)> =
                rows.iter().map(|row| (row.0, row.1, row.5)).collect();
            let threshold = stable_token_min_threshold(&decisions);
            (threshold, rows)
        };

        let mut scans = Vec::with_capacity(3);
        let mut last = Vec::new();
        for _ in 0..3 {
            let (threshold, rows) = scan();
            scans.push(threshold);
            last = rows;
        }
        let pick = settle_token_min_scans(&scans, ladder, default);
        if verbose {
            for (n, reachable, h, l, noise, wins) in &last {
                if *reachable {
                    println!(
                        "    n_tok={n:<3} sidecar={h:>8.1} us native={l:>8.1} us +                             gain={:+.2}% noise={:.2}% {}",
                        (l / h.max(1e-9) - 1.0) * 100.0,
                        noise * 100.0,
                        if *wins { "win" } else { "hold" }
                    );
                } else {
                    println!("    n_tok={n:<3} sidecar unreachable; native only");
                }
            }
            println!("  {}: scans {scans:?} -> {pick}", d.name);
        }
        (d.apply)(pick);
        pick
    };

    // --- Span crossing: which decode-attention kernel is cheaper at each span.
    //
    // Per DISPATCH, not end to end. Only the full-attention layers' span grows
    // with a conversation (7 of 42 on this model), so end to end the choice is
    // under 1% at shallow spans -- far inside decode's run-to-run spread -- and
    // measuring it there produces a boundary that disagrees with itself. The
    // dispatch cost is the whole question: the rest of the step is identical
    // whichever kernel runs.
    /// How many times each full-attention dispatch is issued inside one timed decode step.
    ///
    /// ONE, BECAUSE THE ENGINE ISSUES ONE. Raising it was tried, to lift attention's share of
    /// the clock above the scatter, and it works as arithmetic and fails as a measurement --
    /// see the note at the dispatch. Left as a named constant because the next person to hit
    /// the noise floor here will reach for exactly this, and it is worth their knowing it was
    /// measured and rejected rather than never tried.
    const ATTN_REPS: usize = 1;

    let span_crossing = |d: &KnobDecl, ladder: &[u32], hi: u32, lo: u32| -> u32 {
        expect_one(d);
        let default = (d.current)();
        // RACES ONE DECODE STEP'S ATTENTION, not one dispatch, at depth `n`.
        //
        // Not because a single dispatch measured the wrong SIZE -- because it measures a
        // different machine. See the bandwidth note below: the same kernel costs 0.0510 us
        // per position per layer alone against 0.0363 inside a decode step, a 40% gap, and
        // a boundary is exactly where a systematic 40% error on one side moves the answer.
        //
        // A decode step's attention is 7 full-attention dispatches at 512 dims among 35
        // windowed ones at 256 whose spans never reach this boundary at all. Timing the
        // deep one alone measures a sixth of the work with none of the interleaving, and
        // this knob is the one place that difference decided the answer.
        let one = |v: u32, n: u32| -> f64 {
            (d.apply)(v);
            time_us(b, 3, 4, &|| {
                // A WHOLE DECODE STEP'S DISPATCHES, weights included -- because attention
                // in the engine never runs alone, and running it alone measures a
                // different machine.
                //
                //     engine, per layer   ~100 MB of weights streamed + the KV read
                //     attention alone     the KV read only
                //
                // With concurrent dispatch on, the engine keeps the memory system busy
                // across both, so the same kernel achieves a higher effective bandwidth
                // there. Measured, the gap is 40%: score-tile costs 0.0510 us per position
                // per layer in an attention-only workload against 0.0363 in the engine.
                // The weight SHAPES are model ground truth the tuner already reads
                // (decode_mix, at their real dimensions); they just were not being used
                // here.
                //
                // ADDING THEM COSTS THE ESTIMATE NOTHING, which is the property that makes
                // this safe and which the old ratio did not have. Common work W enters both
                // sides equally:
                //
                //     ratio       (W + t_lo) / (W + t_hi)   -> W DILUTES, ninefold measured
                //     difference  t_lo - t_hi               -> W cancels exactly
                //     crossing    (F_hi - F_lo)/(m_lo - m_hi) -> W cancels in the numerator
                //
                // So the two-line model can afford fidelity where the ratio could not.
                // The full-attention layers, in order: the only ones this boundary governs,
                // and the pool the repeats round-robin over so no dispatch re-reads the KV
                // the one before it just pulled in.
                let full: Vec<(usize, u32)> = s
                    .attn_layers
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, window))| *window == 0)
                    .map(|(li, &(hd, _))| (li, hd))
                    .collect();
                if full.is_empty() {
                    return;
                }
                let mut fi = 0usize;
                for &(_hd, window) in &s.attn_layers {
                    // Every layer streams its weights, governed or not: that is the
                    // contention attention actually runs against.
                    for &(off, i_n, o_n, wk) in &mix {
                        b.matmat(wk, off, i_n, o_n, BufId::Cur, BufId::Logits, 1);
                    }
                    if window != 0 {
                        continue;
                    }
                    fi += 1;
                    let span = n.max(kv_len);
                    let pos = span - 1;
                    // AMPLIFY THE SIGNAL, NOT THE FIDELITY. A decode step runs ~7 full
                    // attention dispatches among 42 layers of weight streaming, so the
                    // quantity this ladder exists to measure is a few percent of what the
                    // clock sees -- at the 128 rung, 26 us of signal on an 18344 us
                    // measurement, 0.14%. Measured scatter on that workload was 0.68% to
                    // 6.13% per rung against effects of 0.4% to 2.5%, which is why the
                    // ladder returned 16384 / 16384 / 4096 where the end-to-end bracket
                    // measures 8192 / ~512 / ~512.
                    //
                    // The repeats walk different layers so no dispatch re-reads the KV its
                    // predecessor just pulled in.
                    //
                    // WHAT AMPLIFICATION COSTS, and KV reuse is NOT the answer -- that was
                    // tested and refuted. f16 at the 128 rung, per-rung difference:
                    //
                    //     1x                       -0.08%  (inside a 0.94% band: a TIE)
                    //     8x, same layer 8 times   +2.21%  streaming ahead
                    //     8x, round-robin          +2.31%  streaming ahead, unchanged
                    //
                    // Round-robin removes the reuse and moves nothing, so the shift is not
                    // cache residency. What remains is CONCURRENCY: 8 reps turn 7 attention
                    // dispatches per step into 56, and the two kernels do not scale alike
                    // when the machine is saturated with them -- streaming runs 128 threads
                    // per threadgroup against score-tile's 512. The engine issues 7.
                    //
                    // So this workload is faithful at 1x and precise at 8x, and not both:
                    // at 1x the scatter (0.68-6.13%) swamps the effect, and at 8x the answer
                    // is measured in a dispatch regime the engine never enters. The end-to-end
                    // bracket has f16 score-tile ahead by 0.7-1.0% at 551-5651, where this
                    // says streaming by 2.1-2.3%. That contradiction is unresolved.
                    for r in 0..ATTN_REPS {
                        let (rli, rhd) = full[(fi + r) % full.len()];
                        b.attention(
                            u32::try_from(rli).unwrap_or(0),
                            rhd,
                            s.n_head,
                            s.n_kv,
                            s.n_kv * rhd,
                            pos,
                            1.0,
                            window,
                            1,
                            scores_needed(pos + 1, window),
                            // See the note in DecodeAttentionStep: a ring locks out the
                            // streaming kernel, so a race that hands one to every layer
                            // compares the score-tile path against itself.
                            if window == 0 {
                                0
                            } else {
                                slots(window.min(DEEP_DECODE_POS).max(kv_len)) - 1
                            },
                        );
                    }
                }
            })
        };
        // ONE PASS, HI AND LO INTERLEAVED A/B/B/A AT EACH RUNG. Three scans used to walk
        // the ladder so a median could paper over a threshold rule that flipped on a single
        // rung; the fit averages instead, so the repetition is gone and this is faster.
        //
        // The A/B/B/A order is not decoration. A DIFFERENCE IS ONLY VALID IF ITS LEGS ARE
        // CONTEMPORANEOUS: deriving one from separate runs produced a confident 7.3% that a
        // same-session comparison could not reproduce, because this machine drifts about 1%
        // between sessions -- measured, on a prefill number that this decode-only knob
        // cannot affect at all. Pairing the legs back to back and reversing the order
        // cancels drift within a rung instead of banking it.
        let mut samples: Vec<crate::crossing::Sample> =
            Vec::with_capacity(ladder.len());
        let mut raw: Vec<(f64, f64)> = Vec::with_capacity(ladder.len());
        // THREE LEGS A SIDE, MEDIAN, AND THE SCATTER KEPT. Two legs averaged cannot tell a
        // 0.5% effect from a 0.5% wobble, and that is the size of everything this ladder
        // decides: the common weight stream under both sides is ~18 ms while the shallow
        // rungs' whole signal is ~26 us. The median rejects a single excursion; the spread
        // becomes the rung's noise band, so the estimator compares each difference against
        // the scatter of the legs that produced it instead of a typed tolerance.
        let med3 = |mut v: [f64; 3]| -> f64 {
            v.sort_by(f64::total_cmp);
            v[1]
        };
        let spread = |v: [f64; 3], m: f64| -> f64 {
            let lo = v.iter().copied().fold(f64::MAX, f64::min);
            let hi = v.iter().copied().fold(0.0_f64, f64::max);
            (hi - lo) / m.max(1e-9)
        };
        for &n in ladder {
            // Interleaved and order-reversed within the rung, so drift cannot line up with
            // one side: h l l h h l.
            let h1 = one(hi, n);
            let l1 = one(lo, n);
            let l2 = one(lo, n);
            let h2 = one(hi, n);
            let h3 = one(hi, n);
            let l3 = one(lo, n);
            let hv = [h1, h2, h3];
            let lv = [l1, l2, l3];
            let h = med3(hv);
            let l = med3(lv);
            raw.push((h, l));
            samples.push(crate::crossing::Sample {
                span: n,
                hi_us: h,
                lo_us: l,
                noise: spread(hv, h).max(spread(lv, l)),
            });
        }
        if verbose {
            // BOTH SIDES, and their cost PER POSITION. If both paths are bandwidth-bound
            // they move the same bytes and each must be linear in span, so cost/position is
            // flat and the ratio cannot trend. Whichever column bends is where the
            // mechanism is -- printing only the ratio hides which side moved.
            for (sm, (h, l)) in samples.iter().zip(&raw) {
                println!(
                    "    span={:<6} stream {:>8.1} us ({:.4} us/pos)  score {:>8.1} us \
                     ({:.4} us/pos)  ratio {:.2}",
                    sm.span,
                    h,
                    h / f64::from(sm.span),
                    l,
                    l / f64::from(sm.span),
                    l / h.max(1e-9)
                );
            }
        }
        // SIGNS FIRST, FIT SECOND. The fit needs intercepts, and on this workload both
        // intercepts carry ~18 ms of weight streaming that belongs to neither kernel. See
        // `threshold_from_signs`: the same three cache types gave 1587, a degenerate zero
        // line, and -5388 through the fit, and give the right rung through the signs.
        if let Some(t) = crate::crossing::threshold_from_signs(&samples) {
            if verbose {
                let band: Vec<String> = samples
                    .iter()
                    .map(|s| {
                        let relative_percent =
                            (s.lo_us - s.hi_us) / s.lo_us.max(1e-9) * 100.0;
                        let mark = if relative_percent > s.noise * 100.0 {
                            "hi"
                        } else if relative_percent < -s.noise * 100.0 {
                            "lo"
                        } else {
                            "tie"
                        };
                        format!(
                            "{}:{relative_percent:+.2}%/{:.2}%{mark}",
                            s.span,
                            s.noise * 100.0
                        )
                    })
                    .collect();
                println!(
                    "  {}: per-rung sign vs that rung's own scatter -> {} -> hi wins from \
                     {t} onward",
                    d.name,
                    band.join("  ")
                );
            }
            // The ladder left the knob on whichever leg ran last; every exit from this
            // closure owes it the chosen value.
            (d.apply)(t);
            return t;
        }
        let pick = match crate::crossing::estimate(&samples) {
            crate::crossing::Verdict::Crossing { span, hi, lo } => {
                let settled = crate::crossing::settle_to_ladder(span, default, ladder);
                if verbose {
                    println!(
                        "  {}: hi = {:.0} us + {:.4}/pos, lo = {:.0} us + {:.4}/pos -> they \
                         meet at span {:.0} -> {}",
                        d.name,
                        hi.fixed,
                        hi.marginal,
                        lo.fixed,
                        lo.marginal,
                        span,
                        settled
                    );
                }
                settled
            }
            crate::crossing::Verdict::NearParallel {
                hi_marginal,
                lo_marginal,
            } => {
                if verbose {
                    println!(
                        "  {}: {:.4} against {:.4} us per position -- the two kernels cost \
                         the SAME per position, so neither overtakes and there is no \
                         crossing to place. Keeping {}",
                        d.name, hi_marginal, lo_marginal, default
                    );
                }
                default
            }
            crate::crossing::Verdict::NotAffine { which, worst_rel } => {
                if verbose {
                    println!(
                        "  {}: the {} kernel is not affine in span ({:.0}% off its own \
                         line). Something SWITCHES rather than scaling, and that is worth \
                         finding. Keeping {}",
                        d.name,
                        which,
                        worst_rel * 100.0,
                        default
                    );
                }
                default
            }
            crate::crossing::Verdict::OutsideRange { span } => {
                if verbose {
                    println!(
                        "  {}: the lines meet at {:.0}, outside every span measured -- that \
                         is extrapolation, not measurement. Keeping {}",
                        d.name, span, default
                    );
                }
                default
            }
            crate::crossing::Verdict::TooFew => default,
        };
        (d.apply)(pick);
        pick
    };

    // --- A TUPLE is swept JOINTLY. Everything else is swept one knob at a time.
    //
    // Coordinate descent -- hold the rest, move one, keep it if it wins -- is only valid
    // when the knobs are independent. Where they meet inside one selection they are not,
    // and a coordinate sweep then reports "nothing beat the incumbent" and is BELIEVED,
    // because every single step really is worse. Only the joint move wins. The prefill
    // attention shape cost about ten attempts across two sessions to exactly this: NSG,
    // BLK and PT are each forced by a different limit, so no one of them moves alone.
    //
    // Within a tuple: sweep the Values members over their cartesian product, apply the
    // winner, and only THEN run any boundary searches (Crossing / SpanCrossing) that
    // belong to the tuple -- so a boundary is found against the shape that won, not
    // against whatever happened to be current. That is the whole of the nb8 coupling:
    // nb8_max is the batch size at which the narrow tile engages, and the right answer
    // depends on which tile nb8_shape put behind it.
    let joint = |members: &[&'static KnobDecl]| -> Vec<(&'static str, u32)> {
        expect(members);
        let vals: Vec<&&KnobDecl> = members
            .iter()
            .filter(|d| matches!(d.sweep, SweepKind::Values))
            .collect();
        let rest: Vec<&&KnobDecl> = members
            .iter()
            .filter(|d| !matches!(d.sweep, SweepKind::Values))
            .collect();
        let mut out: Vec<(&'static str, u32)> = Vec::new();
        if !vals.is_empty() {
            // Only screen survivors, same as the single-knob sweep.
            let axes: Vec<Vec<u32>> = vals
                .iter()
                .map(|d| {
                    survivors_by_knob
                        .get(d.name)
                        .cloned()
                        .unwrap_or_else(|| legal_values(d))
                })
                .collect();
            let total: usize = axes.iter().map(Vec::len).product();
            if total > JOINT_MAX {
                if verbose {
                    println!(
                        "  tuple [{}] UNRESOLVED: {total} combinations exceeds the \
                         exhaustive limit {JOINT_MAX}; keeping the incumbent instead \
                         of misreporting coordinate descent as a joint winner",
                        vals.iter().map(|d| d.name).collect::<Vec<_>>().join(" ")
                    );
                }
                for d in &vals {
                    let current = (d.current)();
                    (d.apply)(current);
                    out.push((d.name, current));
                }
            } else {
                // Every combination, incumbent first so it holds the seat on a tie.
                let cur: Vec<u32> = vals.iter().map(|d| (d.current)()).collect();
                let mut combos: Vec<Vec<u32>> = vec![Vec::new()];
                for ax in &axes {
                    combos = combos
                        .iter()
                        .flat_map(|c| {
                            ax.iter().map(move |v| {
                                let mut n = c.clone();
                                n.push(*v);
                                n
                            })
                        })
                        .collect();
                }
                combos.retain(|c| *c != cur);
                combos.insert(0, cur.clone());
                let apply = |c: &[u32]| {
                    for (d, v) in vals.iter().zip(c) {
                        (d.apply)(*v);
                    }
                };
                let wl = vals[0].workload;
                let time_one = || run_workload(wl);
                apply(&cur);
                let _ = time_one();
                let mut probe: Vec<f64> = (0..5).map(|_| time_one()).collect();
                probe.sort_by(f64::total_cmp);
                // 3 % FLOOR, KEPT ON PURPOSE (2026-09-02): lowered to 0.005 to match the
                // single-knob sweep, the gemv tuple picked sgs 8 / lanes 32 on E4B at
                // 396.9 vs 402.1 us (1.3 %) and the engine read decode 40.7 vs 40.7 and
                // 35.0 vs 35.4 tok/s -- flat to -1 %. The DecodeMix axis has a recorded 19 %
                // noise floor between runs; a 1.3 % micro margin on it is not a ranking.
                let margin = ((probe[4] - probe[0]) / probe[2].max(1e-9)).max(0.03);
                let mut samples: Vec<Vec<f64>> = vec![Vec::new(); combos.len()];
                for _ in 0..ROUNDS {
                    for (i, c) in combos.iter().enumerate() {
                        apply(c);
                        samples[i].push(time_one());
                    }
                }
                let med = |v: &mut Vec<f64>| {
                    v.sort_by(f64::total_cmp);
                    v[v.len() / 2]
                };
                let base = med(&mut samples[0]);
                let mut best = (base, cur.clone());
                for (i, c) in combos.iter().enumerate().skip(1) {
                    let us = med(&mut samples[i]);
                    let win = us < best.0 * (1.0 - margin);
                    if verbose {
                        let line: Vec<String> = vals
                            .iter()
                            .zip(c)
                            .map(|(d, v)| format!("{}={v}", d.name))
                            .collect();
                        println!(
                            "  [{}]  {us:>9.1} us vs {base:>9.1}{}",
                            line.join(" "),
                            if win { "  <-" } else { "" }
                        );
                    }
                    if win && us < best.0 {
                        best = (us, c.clone());
                    }
                }
                // CROSS-CHECK: one value has to serve every regime the engine runs, but
                // the sweep only saw the one its kernel spends most time in. Re-measure
                // the winner against the best alternative on a second regime; if the
                // order FLIPS, no single value serves both and the knob wants a
                // per-regime value. Say so loudly -- a flip that goes unnoticed is a
                // knob tuned for one half of the engine's work.
                // The tuple gets the same INERT check as a single knob, on the same
                // reasoning: a whole tuple whose combinations all land within noise is a
                // tuple measured on a workload its kernels never run on.
                if verbose && combos.len() >= 2 {
                    let mut t: Vec<f64> =
                        (0..combos.len()).map(|i| med(&mut samples[i])).collect();
                    t.sort_by(f64::total_cmp);
                    if t[t.len() - 1] < t[0] * (1.0 + margin) {
                        if margin > UNRESOLVABLE_FLOOR {
                            println!(
                                "  tuple {:?} UNRESOLVED: all {} combinations within a \
                                 {:.1}% noise floor, too wide to decide anything. Not \
                                 evidence they are unwired.",
                                vals[0].tuple,
                                combos.len(),
                                margin * 100.0
                            );
                        } else {
                            println!(
                                "  tuple {:?} INERT: all {} combinations within {:.1}%, on \
                                 a floor tight enough to have seen a difference -- either \
                                 this workload never reaches the kernels these select, or \
                                 they no longer wire to anything",
                                vals[0].tuple,
                                combos.len(),
                                margin * 100.0
                            );
                        }
                    }
                }
                // CONFIRM THE SWITCH HEAD TO HEAD (see the single-knob sweep): the seat's
                // base is one median, and an outlier there elects every combination.
                if best.1 != cur {
                    let (mut su, mut wu) = (Vec::new(), Vec::new());
                    for r in 0..3 {
                        let order: [&Vec<u32>; 2] = if r % 2 == 0 {
                            [&cur, &best.1]
                        } else {
                            [&best.1, &cur]
                        };
                        for c in order {
                            apply(c);
                            let t = time_one();
                            if *c == cur { su.push(t) } else { wu.push(t) }
                        }
                    }
                    let (su, wu) = (med(&mut su), med(&mut wu));
                    if wu < su * (1.0 - margin) {
                        if verbose {
                            println!(
                                "  tuple winner confirmed head to head ({wu:.1} vs the \
                                 seat's {su:.1} us)"
                            );
                        }
                    } else {
                        if verbose {
                            println!(
                                "  tuple winner NOT CONFIRMED head to head ({wu:.1} vs the \
                                 seat's {su:.1} us, margin {:.1}%): the base sample was the \
                                 outlier; keeping the seat",
                                margin * 100.0
                            );
                        }
                        best = (base, cur.clone());
                    }
                }
                // Same rule as a single knob: the seat (the combination the engine runs)
                // defends itself on the cross-check regime, and a winner it beats there by
                // the margin is not written.
                if let (Some(cw), true) =
                    (vals.iter().find_map(|d| d.cross_check), best.1 != cur)
                {
                    let on = |c: &Vec<u32>| {
                        apply(c);
                        let _ = run_workload(cw);
                        let mut v: Vec<f64> =
                            (0..3).map(|_| run_workload(cw)).collect();
                        v.sort_by(f64::total_cmp);
                        v[1]
                    };
                    let name = |c: &Vec<u32>| {
                        vals.iter()
                            .zip(c)
                            .map(|(d, v)| format!("{}={v}", d.name))
                            .collect::<Vec<_>>()
                            .join(" ")
                    };
                    let (wu, iu) = (on(&best.1), on(&cur));
                    let flipped = iu < wu * (1.0 - margin);
                    if verbose {
                        println!(
                            "  cross-check on {cw:?}: winner [{}] {wu:.1} us, seat [{}] \
                             {iu:.1} us{}",
                            name(&best.1),
                            name(&cur),
                            if flipped {
                                "   <- FLIPPED: no single value serves both regimes; \
                                 keeping the seat"
                            } else {
                                "   (agrees)"
                            }
                        );
                    }
                    if flipped {
                        best = (base, cur.clone());
                    }
                }
                apply(&best.1);
                if verbose {
                    println!(
                        "  tuple winner: {}  ({} combinations, margin {:.1}%)",
                        vals.iter()
                            .zip(&best.1)
                            .map(|(d, v)| format!("{}={v}", d.name))
                            .collect::<Vec<_>>()
                            .join(" "),
                        combos.len(),
                        margin * 100.0
                    );
                }
                for (d, v) in vals.iter().zip(&best.1) {
                    out.push((d.name, *v));
                }
            }
        }
        // Boundaries last, against the shape that just won.
        for d in &rest {
            let v = match d.sweep {
                SweepKind::Crossing { ladder, hi, lo } => crossing(d, ladder, hi, lo),
                SweepKind::TokenMinCrossing { ladder, hi, lo } => {
                    token_min_crossing(d, ladder, hi, lo)
                }
                SweepKind::SpanCrossing { ladder, hi, lo } => {
                    span_crossing(d, ladder, hi, lo)
                }
                // A derived member is computed, not searched, whichever tuple it is in.
                SweepKind::Derived => d
                    .derive
                    .map_or_else(|| (d.current)(), |f| f(&facts, profile)),
                SweepKind::External => (d.current)(),
                SweepKind::Values => unreachable!("filtered above"),
            };
            out.push((d.name, v));
        }
        out
    };

    // --- Drive the registry, in declaration order (= dependency order).
    // `--knob NAME` on a tupled knob measures the WHOLE tuple: a member of a coupled
    // tuple cannot be judged on its own, which is the reason the tuple exists.
    let mut picks: Vec<(&'static str, u32)> = Vec::new();
    mark("screen done");
    let wanted = |d: &KnobDecl| -> bool { applicable(d) && selected(d) };
    let mut done_tuples: Vec<&'static str> = Vec::new();
    for d in reg.iter().filter(|d| wanted(d)) {
        if let Some(g) = d.tuple {
            if done_tuples.contains(&g) {
                continue;
            }
            done_tuples.push(g);
            let members: Vec<&'static KnobDecl> = reg
                .iter()
                .filter(|o| o.tuple == Some(g) && applicable(o))
                .collect();
            if verbose {
                println!("  tuple \"{g}\": {} knobs swept jointly", members.len());
            }
            picks.extend(joint(&members));
            continue;
        }
        let v = match d.sweep {
            // COMPUTED, never swept. The value is a function of ground truth, so
            // searching for it would be searching for something already known -- and a
            // measurement that disagreed with the arithmetic would mean the arithmetic
            // is wrong, not that the knob wants a different value.
            SweepKind::Derived => {
                let raw = d
                    .derive
                    .map_or_else(|| (d.current)(), |f| f(&facts, profile));
                // HYSTERESIS, the same mechanism the boundary estimator uses, and here for
                // the same reason: a derived number that moves between runs is worse than
                // the literal it replaced, because the literal at least answers twice.
                //
                // flush_layers reads 6, 7, 7, 7, 8 across runs. Its inputs move together --
                // one run showed a 3.5x global slowdown in BOTH the submission and encode
                // costs -- so the ratio holds and only the rounding of a continuous square
                // root to an integer wobbles. That wobble is worth about 1 us on a 28 000 us
                // token, so what it costs is not speed, it is a config that rewrites itself.
                //
                // A knob that derives EXACTLY, like pt_512x from a queried limit, lands on
                // its current value and passes through untouched.
                // ZERO IS AN ANSWER, NOT AN ESTIMATE. `settle` snaps on a log scale and
                // floors at 1, so a derivation that returns 0 ("never flush": the GPU is
                // not waiting on the host) was written as 1 in every stored file while
                // the compiled default and the running backend held 0 -- the config said
                // what the engine did not do (task #110). A derived 0 passes through.
                let v = if raw == 0 {
                    0
                } else {
                    crate::crossing::settle(f64::from(raw), (d.current)(), 1.5)
                };
                if verbose {
                    let live = (d.current)();
                    println!(
                        "  {} = {v} (derived{}){}",
                        d.name,
                        if raw == v {
                            String::new()
                        } else {
                            format!(" {raw}, held")
                        },
                        if live == v {
                            String::new()
                        } else {
                            format!("   <- MISMATCH: backend has {live}")
                        }
                    );
                }
                v
            }
            SweepKind::External => {
                let value = (d.current)();
                if verbose {
                    println!(
                        "  {} = {value} (external whole-engine admission; micro tuner preserves incumbent)",
                        d.name
                    );
                }
                value
            }
            SweepKind::Values => sweep(d),
            SweepKind::Crossing { ladder, hi, lo } => crossing(d, ladder, hi, lo),
            SweepKind::TokenMinCrossing { ladder, hi, lo } => {
                token_min_crossing(d, ladder, hi, lo)
            }
            SweepKind::SpanCrossing { ladder, hi, lo } => {
                span_crossing(d, ladder, hi, lo)
            }
        };
        picks.push((d.name, v));
    }

    // Restore the reference settings and re-time. Same work, same knobs, so any change
    // is the machine, not the measurement.
    b.set_tuner_dispatch_expectation(&[])?;
    let ref_after = reference();
    let drift = (ref_after - ref_before).abs() / ref_before.max(1e-9);
    if verbose {
        println!(
            "  control {ref_before:.1} -> {ref_after:.1} us  drift={:.0}%",
            drift * 100.0
        );
    }
    // WARN, DO NOT REFUSE. With a minimum rather than a median, a machine that sped up
    // mid-run no longer corrupts the comparison -- each candidate still reports its own
    // best. Drift remains worth SAYING, because a machine that genuinely changed state
    // is worth knowing about, but it is no longer a reason to throw the run away.
    if drift > 0.20 {
        eprintln!(
            "[tune] the machine moved {:.0}% during stage 1 ({ref_before:.0} us -> \
             {ref_after:.0} us); with min-of-N the picks still stand, but re-run on a \
             quiet machine if a margin looks close",
            drift * 100.0
        );
    }

    mark("knobs done");
    Ok(Picks {
        picks,
        screened: screened_out,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        RestoreGuard, ScratchImage, exact128_route_evidence, kv_scratch_bytes,
        reps_per_buffer, settle_token_min_scans, stable_token_min_threshold,
    };
    use imparo_backend::{BufId, TunerScratchRegion, WorkloadEffects};
    use std::cell::{Cell, RefCell};

    #[test]
    fn exact128_evidence_binds_every_controlled_model_layer() {
        assert_eq!(
            exact128_route_evidence(35, 42, 0),
            Some(0x3f | (35 << 6) | (42 << 12) | (42 << 18))
        );
        assert_eq!(
            exact128_route_evidence(35, 42, 35),
            Some(0x3f | (35 << 6) | (42 << 12) | (42 << 18) | (35 << 24))
        );
        assert_eq!(exact128_route_evidence(0, 42, 0), None);
        assert_eq!(exact128_route_evidence(64, 64, 0), None);
    }

    #[test]
    fn tuner_quantized_kv_scratch_covers_the_full_synthetic_capacity() {
        let expected = 40_960_u64 * 8 * 64 * 2;
        assert_eq!(kv_scratch_bytes("q4_0", 40_960, 8, 64), expected);
        assert_eq!(kv_scratch_bytes("q8_0", 40_960, 8, 64), expected);
        assert_eq!(kv_scratch_bytes("f16", 40_960, 8, 64), 0);
    }

    #[test]
    fn mutable_work_is_never_packed_without_a_restore_between_reps() {
        assert_eq!(reps_per_buffer(64, 1.0, true), 1);
        assert_eq!(reps_per_buffer(64, 500.0, true), 1);
        assert!(reps_per_buffer(64, 1.0, false) > 1);
    }

    #[test]
    fn mutable_scratch_captures_and_restores_buffer_and_kv_bytes() {
        static REGIONS: [TunerScratchRegion; 2] = [
            TunerScratchRegion::BufferF32 {
                id: BufId::Cur,
                off: 3,
                elements: 3,
            },
            TunerScratchRegion::KvBytes {
                layer: 2,
                is_v: true,
                off: 5,
                bytes: 4,
            },
        ];
        let image = ScratchImage::capture_with(
            WorkloadEffects::Mutable(&REGIONS),
            |id, off, dst| {
                assert_eq!((id, off), (BufId::Cur, 3));
                dst.copy_from_slice(&[1.0, 2.0, 3.0]);
            },
            |layer, is_v, off, dst| {
                assert_eq!((layer, is_v, off), (2, true, 5));
                dst.copy_from_slice(&[7, 8, 9, 10]);
            },
        );
        let f32_seen = RefCell::new(Vec::new());
        let bytes_seen = RefCell::new(Vec::new());
        image.restore_with(
            |id, off, src| f32_seen.borrow_mut().push((id, off, src.to_vec())),
            |layer, is_v, off, src| {
                bytes_seen
                    .borrow_mut()
                    .push((layer, is_v, off, src.to_vec()));
            },
        );
        assert_eq!(
            *f32_seen.borrow(),
            vec![(BufId::Cur, 3, vec![1.0, 2.0, 3.0])]
        );
        assert_eq!(*bytes_seen.borrow(), vec![(2, true, 5, vec![7, 8, 9, 10])]);
    }

    #[test]
    fn restore_guard_runs_during_unwind() {
        let restored = Cell::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let restore = || restored.set(true);
            let _guard = RestoreGuard(&restore);
            panic!("simulated backend failure");
        }));
        assert!(result.is_err());
        assert!(restored.get());
    }

    #[test]
    fn token_min_crossing_accepts_one_stable_winning_suffix() {
        let rows = [
            (128, false, false),
            (129, true, false),
            (256, true, false),
            (384, true, true),
            (449, true, true),
            (512, true, true),
        ];
        assert_eq!(stable_token_min_threshold(&rows), Some(384));
    }

    #[test]
    fn token_min_crossing_rejects_no_win_and_non_monotonic_routes() {
        assert_eq!(
            stable_token_min_threshold(&[(128, false, false), (256, true, false)]),
            None
        );
        assert_eq!(
            stable_token_min_threshold(&[
                (128, true, false),
                (256, true, true),
                (384, true, false),
                (512, true, true),
            ]),
            None
        );
    }

    #[test]
    fn token_min_scan_consensus_requires_three_neighbouring_answers() {
        let ladder = [128, 256, 384, 449, 512];
        assert_eq!(
            settle_token_min_scans(&[Some(384), Some(449), Some(449)], &ladder, 0,),
            449
        );
        assert_eq!(
            settle_token_min_scans(&[Some(256), None, Some(384)], &ladder, 0),
            0
        );
        assert_eq!(
            settle_token_min_scans(&[Some(128), Some(449), Some(512)], &ladder, 0,),
            0
        );
    }
}
