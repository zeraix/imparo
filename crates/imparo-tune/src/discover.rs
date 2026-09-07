//! DISCOVERY: the device's own limits, measured because nothing reports them.
//!
//! Queried limits are easy -- Metal will state its threadgroup budget and thread count.
//! The ones that decide whether a shape is legal or a duplication is free are not exposed
//! by any API, and a literal in their place is a number read off one machine:
//!
//!   register spill cliff   how many accumulators a thread may hold before the compiler
//!                          spills them to device memory. Was a constant, 2048, guessed
//!                          from three working shapes -- and wrong, because it implied
//!                          the budget divides among threads when the cliff is per
//!                          thread and flat.
//!   cache knee             the working set past which reads fall from cache rate to
//!                          DRAM rate. This is what says whether a re-read is free.
//!   DRAM read rate         what a memory-bound kernel should be judged against.
//!
//! These run once per machine and belong to the tuner, not to a folder of examples a
//! human has to remember to run. A tuner that measures its own ground truth is one that
//! works on a Mac nobody here has seen.
use imparo_backend::{Backend, DeviceProfile};

/// Live-accumulator counts the spill sweep walks, ascending.
const NACC: [u32; 8] = [4, 8, 12, 16, 24, 32, 48, 64];

/// A rate this far below its predecessor is a cliff, not noise. A spill is a 90x
/// collapse here, so the threshold only has to separate "fell off" from "wobbled".
const CLIFF_DROP: f64 = 0.25;

/// Threadgroup counts the MEMORY-BOUND fill sweep walks. Wider and denser than TGS
/// because the answer is expected in the tens, not the single digits, and a ladder that
/// jumps 16 -> 32 -> 64 cannot tell 40 from 72.
const TGS_MEM: [u32; 10] = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192];

/// Threadgroup counts the fill sweep walks. A GPU absorbs threadgroups in parallel until
/// its cores are full; past that, more of them cost time linearly.
const TGS: [u32; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Working sets the cache sweep walks, MB.
const SETS: [u64; 9] = [1, 2, 4, 6, 8, 12, 24, 64, 256];

pub fn profile(b: &dyn Backend, verbose: bool) -> DeviceProfile {
    let mut p = b.device_profile(); // queried limits
    p.max_accumulators = spill_cliff(b, verbose);
    p.fill_threadgroups = fill_point(b, verbose);
    p.attn_score_ceiling_gflops = score_ceiling(b, verbose);
    p.fill_threads_membound = fill_point_membound(b, verbose);
    let (commit_ns, encode_ns) = submission_costs(b, verbose);
    p.commit_overhead_ns = commit_ns;
    p.encode_cost_ns = encode_ns;
    let (knee, dram) = cache_knee(b, verbose);
    p.cache_knee_bytes = knee;
    p.dram_read_mbs = dram;
    if verbose {
        println!(
            "  device: threadgroup {} B, max threads {}, accumulators {}, \
             cache knee {} MB, DRAM {} MB/s",
            p.threadgroup_bytes,
            p.max_threads,
            p.max_accumulators,
            p.cache_knee_bytes >> 20,
            p.dram_read_mbs
        );
    }
    p
}

/// Accumulators a thread may hold before the rate collapses. Swept at several thread
/// counts because the whole question is whether the budget divides among them -- it does
/// not, and assuming it did is what produced a bound that rejected a legal shape.
fn spill_cliff(b: &dyn Backend, verbose: bool) -> u32 {
    let mut worst = u32::MAX;
    for tpg in [128_u32, 256, 512] {
        let mut prev = 0.0_f64;
        let mut here = *NACC.last().unwrap_or(&0);
        for (i, &_n) in NACC.iter().enumerate() {
            let r = b.spill_rate(i as u32, 128, tpg, 4096);
            if i == 0 && r <= 0.0 {
                if verbose {
                    println!("    spill probe unsupported; keeping unknown sentinel");
                }
                return 0;
            }
            if prev > 0.0 && r < prev * (1.0 - CLIFF_DROP) {
                here = NACC[i.saturating_sub(1)];
                break;
            }
            prev = r;
        }
        if verbose {
            println!("    spill cliff at {tpg} threads: {here} accumulators");
        }
        worst = worst.min(here);
    }
    worst
}

/// THREADGROUPS TO FILL THE GPU. Dispatch a fixed amount of work per threadgroup and
/// raise the count: while the machine is not full the aggregate rate climbs roughly
/// linearly, and once it is full the rate flattens because the extra threadgroups queue.
/// The count at which climbing stops is the fill point.
///
/// IT DOES NOT YET REPLACE attn_min_tgs, and the measurement is what says so. This probe
/// is ALU-dense -- 8 accumulators multiplied 4096 times, no memory traffic -- and it
/// saturates at 16 threadgroups of 256 threads, which is 128 simdgroups. attn_min_tgs is
/// 72, and it gates a DECODE ATTENTION dispatch, which is memory-latency-bound. A
/// latency-bound kernel needs more threadgroups in flight to hide its loads than a
/// compute-dense one needs to saturate the ALUs, so the two numbers are not the same
/// quantity and 16 must not be substituted for 72.
///
/// So this is real device ground truth and the wrong ground truth for that knob. Deriving
/// attn_min_tgs needs a fill probe with the attention kernel's ACCESS PATTERN, which is
/// the same lesson as re-probing the matrix-op ceilings with the calling kernel's operand
/// mix (#41): a ceiling measured on the wrong access pattern is a ceiling for a different
/// kernel.
fn fill_point(b: &dyn Backend, verbose: bool) -> u32 {
    let mut prev = 0.0_f64;
    let mut fill = *TGS.last().unwrap_or(&0);
    for &t in &TGS {
        // Fixed work per threadgroup, so the only thing changing is how many there are.
        let r = b.spill_rate(1, t, 256, 4096);
        if r <= 0.0 {
            if verbose {
                println!(
                    "    compute-fill probe unsupported; keeping unknown sentinel"
                );
            }
            return 0;
        }
        // Still climbing means still filling. A rate that stops climbing by a quarter of
        // what perfect scaling would give is a machine that has run out of cores.
        if prev > 0.0 && r < prev * 1.25 {
            fill = t / 2;
            break;
        }
        prev = r;
    }
    if verbose {
        println!("    fill point: {fill} threadgroups");
    }
    fill.max(1)
}

/// WHAT A COMMAND BUFFER COSTS, both halves of it.
///
/// `flush_layers` decides how many layers are encoded before a command buffer is
/// committed, and its comment already said what it is: "a tuned value like `lanes` or
/// `sgs` -- the balance it strikes between encode/execute overlap and live command buffers
/// is a property of the machine, so the tuner has to be able to set it." It had a setter
/// and no declaration, so nothing set it; the compiled 7 is one machine's balance.
///
/// It cannot be micro-benched. Every workload here is a single op, and this knob governs
/// how a MULTI-LAYER encode loop is submitted -- there is nothing for a one-op benchmark
/// to see. But both costs in the trade are directly measurable:
///
///   commit  microseconds per commit+wait of an EMPTY command buffer
///   encode  microseconds the host spends encoding ONE dispatch, never committed
///
/// and with the per-layer GPU time (which the DecodeMix workload already measures: it is
/// one layer's matmuls at their real dimensions), the balance is arithmetic. See
/// `derive_flush_layers`.
fn submission_costs(b: &dyn Backend, verbose: bool) -> (u32, u32) {
    // MEDIAN OF FIVE, not one reading. A single batch put the submission cost at 8.83 us
    // on one run and 9.48 on the next, and `derive_flush` takes a square root of it, so the
    // derived layer count came out 7 and then 9 -- a config that changes when nothing about
    // the machine changed. A derivation has to be reproducible or it is a sweep with extra
    // steps.
    let med = |f: &dyn Fn() -> f64| {
        let mut v: Vec<f64> = (0..5).map(|_| f()).collect();
        v.sort_by(f64::total_cmp);
        v[2]
    };
    // The commit term is a GPU-side GAP between back-to-back buffers and the encode term
    // is host time per dispatch; both are floors whose noise is one-sided (a scheduling
    // hiccup only lengthens a gap, contention only lengthens an encode). The median of
    // five read 0.42 / 0.50 / 3.67 us for the gap across runs and derived flush 1 / 2 / 5
    // (2026-09-04); the minimum of five holds at 1-2, the flat region of the trade.
    let mn = |f: &dyn Fn() -> f64| (0..5).map(|_| f()).fold(f64::INFINITY, f64::min);
    let commit = mn(&|| b.commit_overhead(64));
    let sync = med(&|| b.sync_overhead(32));
    let encode = mn(&|| b.encode_cost(256));
    if verbose && commit > 0.0 {
        println!(
            "    command buffer: {commit:.2} us to submit, {sync:.1} us to submit AND WAIT, \
             {encode:.3} us to encode a dispatch"
        );
    }
    ((commit * 1000.0) as u32, (encode * 1000.0) as u32)
}

/// THREADGROUPS TO FILL A MEMORY-BOUND DISPATCH -- the other fill point, and the one
/// `attn_min_tgs` actually describes.
///
/// `fill_point` above is ALU-dense and saturates at 16 threadgroups. `attn_min_tgs` is 72
/// and gates the DECODE ATTENTION dispatch, which streams K and V with almost no
/// arithmetic. A latency-bound kernel needs far more threadgroups in flight to hide its
/// loads than a compute-dense one needs to saturate the ALUs, so 16 was never the number
/// that knob wanted -- substituting it would have been a measurement used for a quantity
/// it does not describe.
///
/// This measures the right quantity with the right access pattern, and it needs no new
/// kernel: the bandwidth probe already splits a FIXED total working set across the grid,
/// so its aggregate rate climbs while threadgroups are still buying parallelism and
/// flattens once the memory system is saturated. The knee is the fill point.
fn fill_point_membound(b: &dyn Backend, verbose: bool) -> u32 {
    // 32 MB: past the knee by more than 10x, so every read is a real miss and the sweep
    // measures the memory system rather than the cache.
    const SET: u64 = 32 << 20;
    // Threads per threadgroup the sweep runs at. The answer is REPORTED IN THREADS, so
    // this choice does not leak into whatever consumes it.
    const TPG: u32 = 256;
    // THE WHOLE CURVE, then the knee -- not a crossing found while walking it. A
    // step-to-step test ("still climbing by a tenth?") answers differently depending on
    // how wide the ladder's step is: 16 -> 24 is a 1.5x jump in threadgroups and 32 -> 48
    // is the same ratio, but 8 -> 16 is 2x, so the same threshold reads saturation at
    // different places. It jittered between 24 and 32 across runs, and the candidate list
    // built from it jittered with it.
    //
    // Against the plateau instead: the first count within 90% of the best rate seen. That
    // is the same question -- where does adding threadgroups stop buying bandwidth -- with
    // an answer that does not depend on the ladder's spacing or on one noisy sample.
    // MEDIAN OF THREE SWEEPS PER RUNG. One reading put the knee at 16 threadgroups on one
    // run and 24 on the next, and this number is not consumed directly -- it SCALES a
    // candidate ladder (attn_min_tgs offers base x {1,2,3,6,12}). So a 1.5x wobble here
    // moved the ladder from [8,16,24,48,96] to [12,24,36,72,144] and the shipped value
    // from 48 to 36 with nothing about the machine having changed.
    //
    // A derived quantity that moves between runs is worse than a literal: the literal at
    // least gives the same answer twice.
    let mut rates: Vec<(u32, f64)> = Vec::new();
    for &t in &TGS_MEM {
        let mut r: Vec<f64> = (0..3).map(|_| b.bw_read(SET, 2, t, TPG)).collect();
        r.sort_by(f64::total_cmp);
        let gbs = r[1];
        if verbose {
            println!("      membound tgs={t}: {gbs:.1} GB/s");
        }
        rates.push((t, gbs));
    }
    let peak = rates.iter().map(|&(_, g)| g).fold(0.0_f64, f64::max);
    if peak <= 0.0 {
        if verbose {
            println!("    memory-fill probe unsupported; keeping unknown sentinel");
        }
        return 0;
    }
    let fill = rates
        .iter()
        .find(|&&(_, g)| g >= peak * 0.90)
        .map_or(*TGS_MEM.last().unwrap_or(&1), |&(t, _)| t);
    if verbose {
        println!(
            "    fill point (memory-bound): {fill} threadgroups x {TPG} = {} threads",
            fill * TPG
        );
    }
    fill.max(1) * TPG
}

/// THE SCORE PHASE'S OWN CEILING, with that loop's operand mix rather than a generic one,
/// measured twice: K in cache, and K streaming from DRAM.
///
/// WHY IT EXISTS. The prefill attention score phase measures ~2.7 TFLOPS and was compared
/// against ~7.3 from `mma_peak`, which reads both operands from registers with nothing
/// streaming. That comparison said the score phase reached 37% of the machine, and the
/// missing 63% drove real work. But the score loop re-reads staged Q from threadgroup
/// memory and streams K from device on every step, and a matrix rate is set by how the
/// operands ARRIVE as much as by the multiplies.
///
/// WHAT IT MEASURES on an M3 Pro: ~4.1 TFLOPS. So the score phase sits at about 65% of a
/// ceiling it can actually reach, not 37% of one it cannot. The headroom is real but it is
/// roughly 1.5x, not 2.7x, and a plan sized against 2.7x is sized against a kernel that
/// does not exist.
///
/// AND THE TWO SPANS AGREE, within noise -- 4.13 streaming against 4.20 cached. Both come
/// from the SAME kernel with the same multiplies; only K's working set differs, and 8 MB
/// of it is walked past a 2 MB knee. So K's arrival is NOT what bounds this loop: the
/// operand mix is. That also refutes the reading these numbers first suggested, and it
/// says raising the score phase means changing the mix -- more multiplies per operand
/// load, i.e. more query rows per K read -- not feeding K faster.
///
/// Sweeps the simdgroup count because the ceiling depends on how many are resident: the
/// shipping kernel runs 16, and a ceiling read at 2 would flatter it.
fn score_ceiling(b: &dyn Backend, verbose: bool) -> u32 {
    // Position-tile masks, not byte counts: a tile is 16 positions of `stride` halfs, so
    // 512-wide K makes a tile 16 KB. 4 tiles = 64 KB sits in cache; 2048 tiles = 32 MB is
    // past the 2 MB knee by more than 10x.
    const CACHED: u32 = 3; // mask -> 4 tiles
    const STREAM: u32 = 2047; // mask -> 2048 tiles, the whole probe buffer
    let tgs = b.device_profile().max_threads / 32;
    let mut cached = 0.0_f64;
    let mut stream = 0.0_f64;
    for sgs in [4_u32, 8, 16] {
        // stride 512: the deep head dim, so the device-side walk strides like real K.
        cached = cached.max(b.scoremix_rate(tgs, sgs, 4096, 512, CACHED));
        stream = stream.max(b.scoremix_rate(tgs, sgs, 4096, 512, STREAM));
    }
    if verbose && cached > 0.0 {
        println!(
            "    score-mix ceiling: {stream:.2} TFLOPS with K streaming, {cached:.2} with \
             K cached (agreeing: the bound is the OPERAND MIX, not K's arrival)"
        );
    }
    (stream * 1000.0) as u32
}

/// The knee, and the rate past it. Sweeps the working set and watches the read rate fall
/// from cache speed to DRAM speed; the last size before the fall is the cache size.
fn cache_knee(b: &dyn Backend, verbose: bool) -> (u64, u32) {
    let mut prev = 0.0_f64;
    let mut knee = 0_u64;
    let mut tail: Vec<f64> = Vec::new();
    for &mb in &SETS {
        let bytes = mb << 20;
        let reps = u32::try_from((256 / mb).max(2)).unwrap_or(2);
        let gbs = b.bw_read(bytes, reps, 128, 128);
        if prev > 0.0 && gbs < prev * (1.0 - CLIFF_DROP) && knee == 0 {
            knee = bytes >> 1; // the last size that still held the cache rate
        }
        if knee != 0 {
            tail.push(gbs);
        }
        prev = gbs;
    }
    tail.sort_by(f64::total_cmp);
    let dram = tail.get(tail.len() / 2).copied().unwrap_or(0.0);
    if verbose {
        println!("    cache knee {} MB, DRAM {:.0} GB/s", knee >> 20, dram);
    }
    (knee, (dram * 1000.0) as u32)
}
