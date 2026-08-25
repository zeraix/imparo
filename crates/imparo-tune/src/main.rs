//! A good tuner understands model, hardware and kernel, and only by that it provides value for all users including developers.
//!
//! Three requirements and a claim. The three: know the MODEL (tensor shapes, head
//! geometry, quantization), the HARDWARE (limits that can be queried and limits that
//! must be measured), and the KERNEL (what each knob selects, and which knobs move
//! together). The claim is that value follows only from all three -- and the developer
//! is a user here too: the instrument that picks a shape for a stranger's Mac is the
//! same one the kernel gets developed with.
//!
//! Host tuner: measures the declared shape/dispatch search space on THIS machine and
//! stores the winner.
//!
//! BACKEND-GENERIC: the backend under test contributes its `Backend` ops handle and its
//! `KnobDecl` registry (name, stage, sweep, workload, hooks, space version); everything
//! below iterates those. The ONLY backend-specific code is `space()` -- the composition
//! root, where the cfg rule allows it -- so bringing the tuner to CUDA is one arm there
//! plus the CUDA crate's own registry.
//!
//! The values it picks cannot be predicted from specifications, so they are measured
//! rather than hardcoded. Hardcoding them is how an engine ends up tuned to one
//! developer's laptop.
//!
//! Fitting decisions (KV quantisation, layer placement, pool capacity) are NOT here:
//! those are arithmetic against declared device capacity and belong in a solver, not a
//! benchmark.

mod knobs;
mod crossing;
mod discover;
mod micro;

use std::path::PathBuf;
use std::time::Instant;

use imparo_backend::{Backend, BackendKnobs, KnobDecl};
use imparo_model::build_plan;
use imparo_model::PREFILL_BATCH;

/// The backend under test, chosen at the composition root -- the one place the cfg
/// rule allows a target check. A CUDA host adds its arm here; nothing else changes.
struct Space {
    ops: &'static dyn Backend,
    reg: &'static [KnobDecl],
    version: u32,
    tag: &'static str,
    /// Backend-specific process preparation (pipeline pre-builds), run before init.
    prepare: fn(),
}

#[cfg(all(target_os = "macos", not(feature = "cuda")))]
fn space() -> Space {
    use imparo_model::imparo_metal_reexport as imparo_metal;
    static BE: imparo_metal::MetalBackend = imparo_metal::MetalBackend;
    Space {
        ops: &BE,
        reg: BE.knob_registry(),
        version: BE.space_version(),
        tag: "metal",
        // Every candidate pipeline has to exist BEFORE init, or a sweep finds a nil
        // pipeline and silently measures whichever candidate was selected last.
        prepare: || unsafe {
            std::env::set_var("IMPARO_GPU", "1");
            std::env::set_var("IMPARO_RT_ALL", "1");
            std::env::set_var("IMPARO_NR0_ALL", "1");
        },
    }
}

/// The CUDA arm mirrors backend.rs: the `cuda` feature wins on every OS. Written
/// blind on a Mac -- compile-verified, UNVERIFIED on real hardware until a CUDA box
/// runs stage 1 and the gates.
#[cfg(feature = "cuda")]
fn space() -> Space {
    static BE: imparo_cuda::CudaBackend = imparo_cuda::CudaBackend;
    Space {
        ops: &BE,
        reg: BE.knob_registry(),
        version: BE.space_version(),
        tag: "cuda",
        // Opt into the GPU path (backend::enable_gpu); CUDA needs no pipeline
        // pre-builds, so that is the whole preparation.
        prepare: || unsafe { std::env::set_var("IMPARO_GPU", "1") },
    }
}

#[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
fn space() -> Space {
    eprintln!(
        "imparo-tune: no GPU backend in this build -- this platform needs the \
               `cuda` feature (cargo build -p imparo-tune --features cuda, CUDA \
               toolkit required)"
    );
    std::process::exit(2);
}





/// One candidate configuration: every Benched knob's value, in REGISTRY ORDER, plus
/// `batch` -- which is not a backend knob (the model layer chunks prefill; the engine
/// reads it before any backend exists) and so stays typed.
///
/// Generic on purpose: this struct once hardcoded the metal knob list, making five
/// copies of the same enumeration (here, apply, the printed line, the stored file,
/// hostconfig's parser), and `lanes` and later `gemv_max_tok` were each lost in one
/// of the copies. The registry is now the only enumeration.
#[derive(Clone, PartialEq)]
struct Candidate {
    batch: usize,
    vals: Vec<u32>,
}

impl Candidate {
    fn line(&self, reg: &[KnobDecl]) -> String {
        let mut s: Vec<String> = reg
            .iter()
            .zip(&self.vals)
            .map(|(d, v)| format!("{}={v}", d.name))
            .collect();
        s.push(format!("batch={}", self.batch));
        s.join(" ")
    }
    fn set(&mut self, reg: &[KnobDecl], name: &str, v: u32) {
        if let Some(i) = reg.iter().position(|d| d.name == name) {
            self.vals[i] = v;
        }
    }
}









#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sp = space();
    let mut args = std::env::args().skip(1);
    let first = args.next().ok_or(
        "usage: imparo-tune MODEL.gguf [--kv f16|q8_0|q4_0] [--out FILE] [--knob NAME] \
         [--allow-bit-changes] [--explain] | --print-fingerprint",
    )?;
    // --kv: which cache type to tune FOR. It must be set before the backend initialises,
    // because the quantized attention pipelines are built at init from it -- a process
    // configured for f16 never compiles the QT-8 prefill kernels that a q4 cache runs, so
    // one process can only ever measure one cache type. That is also why the stored
    // config is keyed by it: an f16 tuning applied to a q4 server is a measurement from
    // one set of kernels steering another.
    let mut kv = "f16".to_string();
    {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        if let Some(i) = raw.iter().position(|a| a == "--kv") {
            if let Some(v) = raw.get(i + 1) {
                if !["f16", "q8_0", "q4_0"].contains(&v.as_str()) {
                    return Err(format!("--kv {v}: expected f16, q8_0 or q4_0").into());
                }
                kv.clone_from(v);
                unsafe {
                    std::env::set_var("IMPARO_CTK", v);
                    std::env::set_var("IMPARO_CTV", v);
                }
            }
        }
    }
    if first == "--print-fingerprint" {
        println!("{}", imparo_host::fingerprint_for(sp.tag, sp.version, &kv));
        return Ok(());
    }
    let path = PathBuf::from(first);
    let mut out_path: Option<PathBuf> = None;
    // --knob NAME: measure just that knob. Implies --micro.
    let mut only_knob: Option<String> = None;
    // Knobs that can change output BITS are held at their incumbent unless asked for.
    // The tuner ranks on time; it cannot see accuracy, so trading it must be a choice.
    let mut allow_bits = false;
    // --explain: print the knowledge the declarations carry, and stop. If the registry
    // really holds what decides what, against which evidence, it should be readable
    // without reading the source.
    let mut explain = false;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_path = args.next().map(PathBuf::from),
            "--knob" => only_knob = args.next(),
            // Already parsed above (it has to be applied before init); skip its value.
            "--kv" => {
                let _ = args.next();
            }
            "--allow-bit-changes" => allow_bits = true,
            "--explain" => explain = true,
            _ => {}
        }
    }
    if let Some(k) = &only_knob {
        if !sp.reg.iter().any(|d| d.name == k) {
            let names: Vec<&str> = sp.reg.iter().map(|d| d.name).collect();
            return Err(format!("--knob {k}: not in the registry. known: {}", names.join(" ")).into());
        }
    }
    let fp = imparo_host::fingerprint_for(sp.tag, sp.version, &kv);
    let out = out_path.unwrap_or_else(|| {
        let dir = std::env::var("HOME")
            .map_or_else(|_| ".".into(), PathBuf::from)
            .join(".imparo");
        let _ = std::fs::create_dir_all(&dir);
        dir.join(format!("tune-{}.txt", imparo_host::slug(&fp)))
    });
    println!("host  {fp}");
    println!("out   {}", out.display());

    // Shapes come from the GGUF HEADER. `read` parses metadata; the tensor DATA is only
    // mapped by `Weights::open_with`.
    let document = imparo_gguf::read(&path)?;
    let plan = build_plan(&document, &path)?;
    let c = plan.config.clone();
    // Two attention geometries exist in this model; measure the one most layers use.
    let mut hd_count: std::collections::BTreeMap<u32, usize> =
        std::collections::BTreeMap::new();
    for l in &plan.layers.iter().filter(|l| l.attention.is_attention()).collect::<Vec<_>>()
    {
        *hd_count.entry(l.attention.head_dim()).or_default() += 1;
    }
    let head_dim = hd_count
        .iter()
        .max_by_key(|(_, n)| **n)
        .map_or(256, |(d, _)| *d);
    // The decode-routing sweep runs at the FULL-attention geometry: window
    // layers never attend past their window, so their span cannot cross.
    let mut full_hd: std::collections::BTreeMap<u32, usize> =
        std::collections::BTreeMap::new();
    for l in &plan.layers {
        if let imparo_model::Attention::Full { head_dim, .. } = l.attention {
            *full_hd.entry(head_dim).or_default() += 1;
        }
    }
    let deep_head_dim = full_hd
        .iter()
        .max_by_key(|(_, n)| **n)
        .map_or(head_dim, |(d, _)| *d);
    // From the GGUF header: 0 or absent means a dense model, and the MoE knobs are then
    // never offered. Keyed by the architecture's own prefix, which differs per model.
    let arch_name = document
        .string_value("general.architecture")
        .unwrap_or("gemma4")
        .to_string();
    let n_experts = u32::try_from(
        document
            .unsigned_value(&format!("{arch_name}.expert_count"))
            .unwrap_or(0),
    )
    .unwrap_or(0);
    let shapes = micro::Shapes {
        n_experts,
        n_embd: c.n_embd,
        n_ff: c.n_ff,
        n_head: c.n_heads,
        n_kv: c.n_kv_heads,
        head_dim,
        deep_head_dim,
        n_layers: c.n_layers,
        // Filled below from the model's bench extension, once the architecture is known.
        layer_dispatches: 0,
        // The real per-layer attention geometry, in layer order: (head_dim, window), with
        // window 0 for full attention. A decode step is this MIX, not one deep dispatch.
        attn_layers: plan
            .layers
            .iter()
            .filter(|l| l.attention.is_attention())
            .map(|l| match l.attention {
                imparo_model::Attention::Full { head_dim, .. } => (head_dim, 0u32),
                imparo_model::Attention::Window { head_dim, window, .. } => (head_dim, window),
                // Filtered above: a conv block has no attention dispatch to time, and
                // handing this list a (0, 0) entry would put a zero-width attention in
                // the decode-step mix.
                imparo_model::Attention::Recurrent { .. } => unreachable!(),
            })
            .collect(),
        decode_mix: Vec::new(),
        // Both refilled below from decode_mix, once the tensors have been read.
        weight_kinds: 0,
        wide_kind: 1,
    };
    println!(
        "shape n_embd={} n_ff={} heads={} kv={} head_dim={head_dim}",
        c.n_embd, c.n_ff, c.n_heads, c.n_kv_heads
    );

    // Start from compiled defaults, not from whatever a previous run stored: a tuner
    // that seeds itself from its own last answer gives a different answer each run.
    unsafe { std::env::set_var("IMPARO_NO_HOSTCONFIG", "1") };
    let t_micro = Instant::now();
    (sp.prepare)();

    // The incumbent is the COMPILED DEFAULTS, not whatever a previous run stored: a
    // tuner that seeds itself from its own last answer gives a different answer each
    // run. IMPARO_NO_HOSTCONFIG=1 above guarantees the backend is at its defaults.
    let incumbent = Candidate {
        batch: PREFILL_BATCH,
        vals: sp.reg.iter().map(|d| (d.current)()).collect(),
    };
    let model_bytes = std::fs::metadata(&path).map_or(0, |m| m.len());

    // THE MODEL'S TENSOR DATA IS NEVER MAPPED -- but every piece of the model the tuner
    // reasons about IS read. Those are different things, and the distinction matters
    // because this component only earns its keep by understanding the model:
    //
    //   SHAPES        n_embd, n_ff, head counts, head dims, and the per-tensor
    //                 dimensions of the decode matmuls -- all from the GGUF HEADER,
    //                 which is a metadata read that maps no tensor data. Every knob
    //                 whose value is computed rather than searched is computed from
    //                 these.
    //   WEIGHT BYTES  needed only so there is something to multiply. A matmul's timing
    //                 does not depend on what is in the matrix, so a synthetic blob
    //                 serves, and mapping 2.5 GB to read 15 MB of it is a large memory
    //                 event for nothing.
    //
    // THE BLOB MUST EXCEED CACHE. Sized to the widest matmul alone it is ~15 MB, which
    // sits inside this machine's measured ~8 MB knee plus its neighbours and would be
    // served far faster than real weights ever are -- a decode GEMV is memory-bound, and
    // measuring it against a cache-resident blob flatters it and can pick the wrong
    // knob. Sized well past the knee, the memory system behaves as it will in
    // production. Per-tensor offsets are spread across it for the same reason.
    let widest = shapes.n_embd as usize * shapes.n_ff as usize / 32 * 18;
    let wbytes = widest.max(256 << 20);
    let layout = std::alloc::Layout::from_size_align(wbytes, 16384)
        .map_err(|e| format!("synthetic weight layout: {e}"))?;
    // SAFETY: freshly allocated, 16 KB aligned, and it outlives every use below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    if base.is_null() {
        return Err("synthetic weight blob allocation failed".into());
    }
    unsafe { sp.ops.init_weights(base, wbytes as u64) }
        .map_err(|rc| format!("backend init_weights rc={rc}"))?;

    // WHAT THE BACKEND ACTUALLY HOLDS, checked against what --kv asked for, and checked
    // HERE because this is the first moment it can be: the cache type is applied when the
    // backend initialises, so asking before this returns the default no matter what was
    // requested. The config is keyed by the cache type, so a flag that never arrived would
    // MISLABEL every answer in the file rather than fail.
    {
        let live = sp.ops.kv_tag();
        assert!(
            live == kv,
            "--kv {kv} did not reach the backend: it reports {live}. The config would be \
             labelled {kv} and hold measurements taken on {live}."
        );
    }

    // The decode matmuls at their REAL per-tensor dimensions, read from the header. The
    // tensor list comes from the model's bench extension (knobs.rs); the tuner core stays
    // model-agnostic. Offsets are spread across the blob rather than the model's own,
    // which would point past it -- the dimensions are what set the work, the spread is
    // what keeps the reads off one cache-hot page.
    let mbench = knobs::bench_for(&arch_name);
    let names = mbench.decode_tensors;
    // THE QUANT TYPE IS READ PER TENSOR, not assumed. A GGUF carries a type per tensor and
    // mixed-quant files are normal, so a list that recorded dimensions and then measured
    // every entry as q4_0 would be timing a different kernel over a different number of
    // bytes for any tensor that is not q4_0.
    //
    // A tensor whose type this engine cannot run is DROPPED and counted, not silently
    // measured as something else -- the print below says how many made it.
    let mut unsupported: Vec<String> = Vec::new();
    let decode_mix: Vec<(u64, u32, u32, u32)> = names
        .iter()
        .enumerate()
        .filter_map(|(i, n)| {
            document.tensor(&format!("blk.0.{n}.weight")).and_then(|t| {
                let (ne0, ne1) = (*t.dimensions.first()?, *t.dimensions.get(1)?);
                let Some(kind) = imparo_gguf::weights::weight_kind(t.ggml_type) else {
                    unsupported.push(format!("{n}(ggml={})", t.ggml_type));
                    return None;
                };
                let span = (wbytes as u64) / (names.len().max(1) as u64);
                let off = (i as u64) * span;
                // BYTES PER BLOCK COMES FROM THE KIND. This was a literal 18 (Q4_0),
                // which under-counts a Q8_0 tensor by 89% -- 34 bytes per 32 values --
                // so the bounds check below would admit an offset whose kernel then
                // reads past the synthetic weight buffer.
                let bpb = match kind {
                    imparo_gguf::weights::WeightKind::F32 => 4 * 32,
                    imparo_gguf::weights::WeightKind::Q4_0 => 18,
                    imparo_gguf::weights::WeightKind::Q8_0 => 34,
                };
                let need = ne0 * ne1 / 32 * bpb;
                (off + need <= wbytes as u64)
                    .then_some((off, ne0 as u32, ne1 as u32, kind as u32))
            })
        })
        .collect();
    if !unsupported.is_empty() {
        println!(
            "  decode_mix: {} tensor(s) dropped, no kernel for their quant type: {}",
            unsupported.len(),
            unsupported.join(" ")
        );
    }
    println!(
        "synthetic weights {} MB (past the cache knee), model tensor data NOT mapped; \
         decode_mix {} of {} tensors at their real dimensions AND quant types",
        wbytes >> 20,
        decode_mix.len(),
        names.len()
    );
    // WHAT THE WORKLOADS WILL ACTUALLY DISPATCH, read from the tensors rather than
    // assumed. `weight_kinds` gates the per-quant knob families through `applies`;
    // `wide_kind` is what the two widest projections carry, which is the kind the prefill
    // tile knobs are ranked against.
    let weight_kinds = decode_mix.iter().fold(0u32, |m, &(_, _, _, k)| m | (1u32 << k));
    let wide_kind = decode_mix
        .iter()
        .filter(|&&(_, i, o, _)| i.max(o) >= shapes.n_ff)
        .map(|&(_, _, _, k)| k)
        .next_back()
        .unwrap_or(1);
    println!("  weight kinds present: mask {weight_kinds:#05b}, wide projections {wide_kind}");
    let shapes = micro::Shapes {
        decode_mix,
        layer_dispatches: mbench.layer_dispatches,
        weight_kinds,
        wide_kind,
        ..shapes
    };

    if explain {
        let facts = imparo_backend::ModelFacts {
            n_embd: shapes.n_embd,
            n_ff: shapes.n_ff,
            n_head: shapes.n_head,
            n_kv: shapes.n_kv,
            head_dim: shapes.head_dim,
            deep_head_dim: shapes.deep_head_dim,
            n_experts: shapes.n_experts,
            n_layers: shapes.n_layers,
            layer_dispatches: shapes.layer_dispatches,
            weight_kinds: shapes.weight_kinds,
        };
        let profile = discover::profile(sp.ops, false);
        explain_registry(sp.reg, &facts, &profile);
        return Ok(());
    }

    // ALWAYS profile: the point of running the tuner is to refresh what this machine is,
    // not to trust what it was. Measured once here and handed to the sweep.
    let measured = discover::profile(sp.ops, true);
    let picks = micro::measure(sp.ops, sp.reg, &shapes, true, allow_bits, &measured,
                               only_knob.as_deref())
        .map_err(|e| format!("micro: {e}"))?;
    let mut cur = incumbent.clone();
    for (name, v) in &picks.picks {
        cur.set(sp.reg, name, *v);
    }
    println!(
        "picks {}  survived_screen={:?} in {:.2}s",
        picks
            .picks
            .iter()
            .map(|(n, v)| format!("{n}={v}"))
            .collect::<Vec<_>>()
            .join(" "),
        picks.screened,
        t_micro.elapsed().as_secs_f64()
    );
    // A single-knob run must not write the file: it measured one knob and would commit
    // every other knob at whatever the incumbent happened to be.
    if only_knob.is_some() {
        println!("--knob: one knob measured, config NOT written");
        return Ok(());
    }
    let against = "micro-benches only (no end-to-end search)";
    let knob_lines =
        sp.reg
            .iter()
            .zip(&cur.vals)
            .fold(String::new(), |mut s, (d, v)| {
                use std::fmt::Write as _;
                let _ = writeln!(s, "{}={v}", d.name);
                s
            });
    // The measured profile goes in the file so the ENGINE can read it before it compiles
    // its kernel library. Measuring costs seconds the tuner can afford once and a process
    // start cannot afford at all -- and without it a derivation falls back to a literal,
    // which is what deriving was for.
    let dev = format!(
        "device_max_accumulators={}\ndevice_cache_knee_bytes={}\ndevice_dram_read_mbs={}\n\
         device_threadgroup_bytes={}\ndevice_max_threads={}\n",
        measured.max_accumulators,
        measured.cache_knee_bytes,
        measured.dram_read_mbs,
        measured.threadgroup_bytes,
        measured.max_threads,
    );
    let body = format!(
        "# imparo host tuning, {} search space v{}\n\
         # every knob chosen by math + micro-bench; no end-to-end search\n\
         # verified against: {against}\n\
         fingerprint={fp}\n\
         model={}\nmodel_bytes={model_bytes}\n\
         {dev}{knob_lines}batch={}\n",
        sp.tag,
        sp.version,
        path.file_name()
            .map_or_else(|| "?".into(), |n| n.to_string_lossy().into_owned()),
        cur.batch
    );
    std::fs::write(&out, body)?;
    println!("ALL {}  -> {}", cur.line(sp.reg), out.display());
    Ok(())
}

/// Print what the registry KNOWS: for each knob, what decides it, against which regime,
/// what it is coupled to, what it may not do. The point of putting these in the
/// declaration rather than in a comment or a conditional is that they can be read back.
fn explain_registry(
    reg: &'static [KnobDecl],
    facts: &imparo_backend::ModelFacts,
    profile: &imparo_backend::DeviceProfile,
) {
    use imparo_backend::SweepKind as Sw;
    println!("\nMODEL   n_embd={} n_ff={} heads={} kv={} head_dim={} deep={} experts={}",
        facts.n_embd, facts.n_ff, facts.n_head, facts.n_kv, facts.head_dim,
        facts.deep_head_dim, facts.n_experts);
    println!("DEVICE  threadgroup={} B  threads={}  accumulators={}  cache_knee={} MB  dram={} MB/s",
        profile.threadgroup_bytes, profile.max_threads, profile.max_accumulators,
        profile.cache_knee_bytes >> 20, profile.dram_read_mbs);
    println!(
        "\n{:<22} {:<10} {:<22} {:<14} NOTES",
        "KNOB", "DECIDED BY", "MEASURED IN", "TUPLE"
    );
    for d in reg {
        let how = match d.sweep {
            Sw::Derived => "computed",
            Sw::Values => "swept",
            Sw::Crossing { .. } | Sw::SpanCrossing { .. } => "boundary",
        };
        let regime = match d.sweep {
            Sw::Derived => "-".to_string(),
            // A boundary drives its own ladder and never reads `workload`; printing it
            // would state something the code does not do.
            Sw::Crossing { .. } | Sw::SpanCrossing { .. } => "own ladder".to_string(),
            Sw::Values => format!("{:?}", d.workload),
        };
        let mut notes: Vec<String> = Vec::new();
        if !d.applies.is_none_or(|f| f(facts)) {
            notes.push("N/A to this model".into());
        }
        if d.bit_affecting {
            notes.push("CHANGES BITS".into());
        }
        if d.legal.is_some() {
            let bad: Vec<String> = d.values.iter()
                .filter(|v| !d.legal.is_some_and(|f| f(**v, facts, profile)))
                .map(ToString::to_string).collect();
            if !bad.is_empty() {
                notes.push(format!("illegal here: {}", bad.join(",")));
            }
        }
        if let Some(cw) = d.cross_check {
            notes.push(format!("cross-checked in {cw:?}"));
        }
        if matches!(d.sweep, Sw::Crossing { .. } | Sw::SpanCrossing { .. }) {
            notes.push("defines a regime".into());
        }
        println!("{:<22} {:<10} {:<22} {:<14} {}",
            d.name, how, regime, d.tuple.unwrap_or("-"), notes.join("; "));
    }
    println!("\nA boundary knob PARTITIONS the runtime state into regimes, which is why it is");
    println!("searched after the knobs measured within them. A tuple is several knobs that are");
    println!("really one decision. A computed knob is never searched -- ground truth already");
    println!("determines it.");
}
