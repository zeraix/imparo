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

mod crossing;
mod discover;
mod knobs;
mod micro;

use std::path::PathBuf;
use std::time::Instant;

#[cfg(any(feature = "cuda", feature = "cuda-dynamic", target_os = "macos"))]
use imparo_backend::BackendKnobs;
use imparo_backend::{Backend, KnobDecl, SweepKind};
use imparo_model::PREFILL_BATCH;
use imparo_model::build_plan;

/// The backend under test, chosen at the composition root -- the one place the cfg
/// rule allows a target check. A CUDA host adds its arm here; nothing else changes.
struct Space {
    ops: &'static dyn Backend,
    reg: &'static [KnobDecl],
    version: u32,
    tag: &'static str,
    /// Backend-specific process preparation (pipeline pre-builds), run before init.
    prepare: fn(),
    /// The widest token tile in the backend's prefill GEMM shape table: the width that
    /// exercises every candidate for the pair's second tile (`Shapes::pair_tile_tokens`).
    pair_tile_tokens: fn() -> u32,
}

struct TunerModeGuard {
    backend: &'static dyn Backend,
    active: bool,
}

impl TunerModeGuard {
    fn enter(backend: &'static dyn Backend) -> Result<Self, String> {
        backend
            .set_tuner_mode(true)
            .map_err(|rc| format!("failed to enter backend tuner mode: rc={rc}"))?;
        Ok(Self {
            backend,
            active: true,
        })
    }

    /// Leave tuner mode before publishing any result. `Drop` remains the unwind/error
    /// safety net, but a failed transition must be observable on the successful path.
    fn close(&mut self) -> Result<(), String> {
        if !self.active {
            return Ok(());
        }
        self.backend
            .set_tuner_mode(false)
            .map_err(|rc| format!("failed to leave backend tuner mode: rc={rc}"))?;
        self.active = false;
        Ok(())
    }
}

impl Drop for TunerModeGuard {
    fn drop(&mut self) {
        if self.active {
            if let Err(rc) = self.backend.set_tuner_mode(false) {
                eprintln!("imparo-tune: failed to leave backend tuner mode: rc={rc}");
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "cuda-dynamic", test))]
fn cuda_tuner_env_is_polluting(name: &str) -> bool {
    name == "IMPARO_GPU_PROBE"
        || (name.starts_with("IMPARO_CUDA_")
            && ["_LAB", "_PROFILE", "_TRACE", "_DUMP", "_FAIL", "FORCE_OOM"]
                .iter()
                .any(|fragment| name.contains(fragment)))
}

#[cfg(any(feature = "cuda", feature = "cuda-dynamic"))]
fn reject_polluting_cuda_tuner_environment() -> Result<(), String> {
    let mut rejected: Vec<String> = std::env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .filter(|name| cuda_tuner_env_is_polluting(name))
        .collect();
    rejected.sort_unstable();
    rejected.dedup();
    if rejected.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "refusing CUDA tuning with route/diagnostic environment: {}",
            rejected.join(", ")
        ))
    }
}

#[cfg(all(
    target_os = "macos",
    not(any(feature = "cuda", feature = "cuda-dynamic"))
))]
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
            // The Q8 wide-prefill shapes obey the same rule as the Q4 ones above. Omitted
            // when that family landed, so its sweep compared nil pipelines.
            std::env::set_var("IMPARO_Q8_ALL", "1");
            // The FA op's simdgroup count sizes register arrays, so each value is its own
            // kernel; the sweep selects them at dispatch, so every legal one must exist.
            std::env::set_var("IMPARO_FA_ALL", "1");
        },
        pair_tile_tokens: || {
            (0..imparo_metal::st_gemm_shapes())
                .map(imparo_metal::q8_shape_tokens)
                .max()
                .unwrap_or(64)
        },
    }
}

/// The CUDA arm mirrors backend.rs: the `cuda` feature wins on every OS. Written
/// blind on a Mac -- compile-verified, UNVERIFIED on real hardware until a CUDA box
/// runs stage 1 and the gates.
#[cfg(any(feature = "cuda", feature = "cuda-dynamic"))]
fn space() -> Space {
    static BE: imparo_cuda::CudaBackend = imparo_cuda::CudaBackend;
    Space {
        ops: &BE,
        reg: BE.knob_registry(),
        version: BE.space_version(),
        tag: "cuda",
        // Opt into the GPU path (backend::enable_gpu); CUDA needs no pipeline
        // pre-builds, so that is the whole preparation.
        prepare: || unsafe {
            std::env::set_var("IMPARO_GPU", "1");
            // Cross the Rust/MSVC CRT boundary with a static lab setter, not a process
            // environment mutation: Windows C getenv is not required to observe an env
            // value Rust added after startup. Ordinary engine processes never call it.
            imparo_cuda::prepare_tuner_lab();
        },
        pair_tile_tokens: || 64,
    }
}

#[cfg(all(
    not(target_os = "macos"),
    not(any(feature = "cuda", feature = "cuda-dynamic"))
))]
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

/// Batch is a model-level setting, not a searched backend knob. Retuning another
/// coordinate must keep its stored effective value instead of resetting it to the
/// compiled default. Match `imparo_model::prefill_batch`'s minimum of one.
fn seated_prefill_batch(stored: Option<usize>) -> usize {
    stored.unwrap_or(PREFILL_BATCH).max(1)
}

/// A dynamic ladder needs the loaded model and measured device. Preflight can
/// reject static-list mistakes immediately; dynamic membership is checked again
/// after discovery, before a candidate is applied or written.
fn external_candidate_membership(
    declaration: &KnobDecl,
    value: u32,
    context: Option<(&imparo_backend::ModelFacts, &imparo_backend::DeviceProfile)>,
) -> Result<(), String> {
    let values = match declaration.candidates {
        Some(candidates) => {
            let Some((facts, profile)) = context else {
                return Ok(());
            };
            candidates(facts, profile)
        }
        None => declaration.values.to_vec(),
    };
    if values.contains(&value) {
        Ok(())
    } else {
        Err(format!(
            "--external-candidate {}={value} is not declared; candidates: {values:?}",
            declaration.name
        ))
    }
}

/// A stored request can become illegal when init discovers the device's limits.
/// Accept a backend adjustment only when the existing legality hook explains BOTH
/// sides: the request cannot run here, and the readback can. A legal value lost by
/// init/setter remains an error, as does any unexplained or illegal replacement.
fn stored_seat_adjusted(
    declaration: &KnobDecl,
    requested: u32,
    actual: u32,
    facts: &imparo_backend::ModelFacts,
    profile: &imparo_backend::DeviceProfile,
) -> Result<bool, String> {
    if actual == requested {
        return Ok(false);
    }
    if declaration.legal.is_some_and(|legal| {
        !legal(requested, facts, profile) && legal(actual, facts, profile)
    }) {
        return Ok(true);
    }
    Err(format!(
        "stored seat {} requested={requested} actual={actual} after backend initialization; \
         the registry does not explain this as an illegal request adjusted to a legal value",
        declaration.name
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExternalCandidate {
    name: String,
    value: u32,
}

fn parse_external_candidate(raw: &str) -> Result<ExternalCandidate, String> {
    let (name, value) = raw
        .split_once('=')
        .ok_or("--external-candidate expects NAME=VALUE")?;
    if name.is_empty() || value.is_empty() || value.contains('=') {
        return Err("--external-candidate expects one non-empty NAME=VALUE".into());
    }
    let value = value
        .parse::<u32>()
        .map_err(|_| "--external-candidate VALUE must be an unsigned integer")?;
    Ok(ExternalCandidate {
        name: name.into(),
        value,
    })
}

#[allow(clippy::too_many_lines)]
/// Writes the measured device profile to ITS OWN file, keyed by host and backend only.
///
/// It used to live in the tune file, which is also keyed by the knob space, the KV type
/// and the model -- so bumping any of those threw away a measurement that was still true
/// and every derived shape fell back to its compiled literal. Nothing here depends on a
/// model or a knob.
fn store_device_profile(
    tag: &str,
    m: &imparo_backend::DeviceProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    let vals = [
        ("device_max_accumulators", u64::from(m.max_accumulators)),
        ("device_cache_knee_bytes", m.cache_knee_bytes),
        ("device_dram_read_mbs", u64::from(m.dram_read_mbs)),
        ("device_threadgroup_bytes", m.threadgroup_bytes),
        ("device_max_threads", u64::from(m.max_threads)),
    ];
    imparo_host::write_device_profile(tag, &vals)?;
    println!("device {}", imparo_host::device_path_for(tag).display());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(any(feature = "cuda", feature = "cuda-dynamic"))]
    reject_polluting_cuda_tuner_environment()?;
    let sp = space();
    let mut args = std::env::args().skip(1);
    let first = args.next().ok_or(
        "usage: imparo-tune MODEL.gguf [--kv f16|q8_0|q4_0] [--out FILE] [--knob NAME] \
         [--emit-candidate] [--external-candidate NAME=VALUE] \
         [--allow-bit-changes] [--explain] [--discover-only] | --print-fingerprint",
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
    // A single-knob run remains non-writing unless the caller explicitly requests a
    // complete candidate at an explicit path. This is for correctness sealing and A/B;
    // it never replaces the host's default tuning file by accident.
    let mut emit_candidate = false;
    // End-to-end routes are intentionally not ranked by the micro tuner. This option
    // materializes one registry-validated candidate selected by an external whole-model
    // experiment; it still has no runtime authority until receipt sealing succeeds.
    let mut external_candidates: Vec<ExternalCandidate> = Vec::new();
    // Knobs that can change output BITS are held at their incumbent unless asked for.
    // The tuner ranks on time; it cannot see accuracy, so trading it must be a choice.
    let mut allow_bits = false;
    // --explain: print the knowledge the declarations carry, and stop. If the registry
    // really holds what decides what, against which evidence, it should be readable
    // without reading the source.
    let mut explain = false;
    // --discover-only: measure this machine and write the device profile, nothing else.
    // The engine derives its kernel shapes from these numbers at library-compile time, so
    // a machine needs them before it needs any tuning -- and measuring them takes seconds
    // where a full tune takes many minutes.
    let mut discover_only = false;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_path = args.next().map(PathBuf::from),
            "--knob" => only_knob = args.next(),
            "--emit-candidate" => emit_candidate = true,
            "--external-candidate" => {
                let candidate = parse_external_candidate(
                    &args.next().ok_or("--external-candidate needs NAME=VALUE")?,
                )?;
                if external_candidates
                    .iter()
                    .any(|existing| existing.name == candidate.name)
                {
                    return Err(format!(
                        "--external-candidate {} may be specified only once",
                        candidate.name
                    )
                    .into());
                }
                external_candidates.push(candidate);
            }
            // Already parsed above (it has to be applied before init); skip its value.
            "--kv" => {
                let _ = args.next();
            }
            "--allow-bit-changes" => allow_bits = true,
            "--explain" => explain = true,
            "--discover-only" => discover_only = true,
            _ => {}
        }
    }
    for external in &external_candidates {
        let declaration = sp
            .reg
            .iter()
            .find(|declaration| declaration.name == external.name)
            .ok_or_else(|| {
                format!("external knob {} is not in the registry", external.name)
            })?;
        if declaration.sweep != SweepKind::External
            || declaration.category != imparo_backend::KnobCategory::EndToEnd
        {
            return Err(format!(
                "--external-candidate {} is not an External/EndToEnd knob",
                external.name
            )
            .into());
        }
        external_candidate_membership(declaration, external.value, None)?;
        if declaration.bit_affecting && !allow_bits {
            return Err(format!(
                "--external-candidate {} changes output bits; add --allow-bit-changes",
                external.name
            )
            .into());
        }
    }
    if !external_candidates.is_empty() {
        if let Some(name) = &only_knob {
            if external_candidates.len() != 1 || external_candidates[0].name != *name {
                return Err(
                    "--knob may accompany only one matching --external-candidate"
                        .into(),
                );
            }
        } else {
            // External knobs are never micro-ranked. Selecting the first one only
            // constrains micro::measure to its incumbent while all explicit external
            // values are applied below as one whole-model candidate tuple.
            only_knob = Some(external_candidates[0].name.clone());
        }
        if !emit_candidate || out_path.is_none() {
            return Err(
                "--external-candidate requires --emit-candidate and an explicit --out FILE"
                    .into(),
            );
        }
    }
    if let Some(k) = &only_knob {
        let Some(target) = sp.reg.iter().find(|d| d.name == k) else {
            let names: Vec<&str> = sp.reg.iter().map(|d| d.name).collect();
            return Err(format!(
                "--knob {k}: not in the registry. known: {}",
                names.join(" ")
            )
            .into());
        };
        if target.bit_affecting && !allow_bits {
            return Err(format!(
                "--knob {k} can change output bits; rerun with --allow-bit-changes and \
                 re-run numerical gates"
            )
            .into());
        }
    }
    if emit_candidate && (only_knob.is_none() || out_path.is_none()) {
        return Err("--emit-candidate requires both --knob NAME and --out FILE".into());
    }
    let fp = imparo_host::fingerprint_for(sp.tag, sp.version, &kv);
    let model_bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
    let out = out_path.unwrap_or_else(|| imparo_host::path_for(&fp, model_bytes));
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
    for l in &plan
        .layers
        .iter()
        .filter(|l| l.attention.is_attention())
        .collect::<Vec<_>>()
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
        activation: imparo_model::backend::model_activation(&plan)?.epilogue(),
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
        pair_tile_tokens: (sp.pair_tile_tokens)(),
        // The real per-layer attention geometry, in layer order: (head_dim, window), with
        // window 0 for full attention. A decode step is this MIX, not one deep dispatch.
        attn_layers: plan
            .layers
            .iter()
            .filter(|l| l.attention.is_attention())
            .map(|l| match l.attention {
                imparo_model::Attention::Full { head_dim, .. } => (head_dim, 0u32),
                imparo_model::Attention::Window {
                    head_dim, window, ..
                } => (head_dim, window),
                // Filtered above: a conv block has no attention dispatch to time, and
                // handing this list a (0, 0) entry would put a zero-width attention in
                // the decode-step mix.
                imparo_model::Attention::Recurrent { .. } => unreachable!(),
            })
            .collect(),
        decode_mix: Vec::new(),
        lm_head: None,
        exact128_local_q: None,
        ffn_transaction: None,
        ple_transaction: None,
        shortconv_transaction: None,
        // Both refilled below from decode_mix, once the tensors have been read.
        weight_kinds: 0,
        wide_kind: 1,
    };
    println!(
        "shape n_embd={} n_ff={} heads={} kv={} head_dim={head_dim}",
        c.n_embd, c.n_ff, c.n_heads, c.n_kv_heads
    );

    // THE SEAT IS THE STORED VALUE, when this machine has one. Backend init applies the
    // host config exactly as the engine does, so every knob a previous run MEASURED holds
    // its seat and a candidate must beat it by the noise floor; a knob the file does not
    // name starts from its compiled default, as before.
    //
    // This used to set IMPARO_NO_HOSTCONFIG=1 so that "a tuner that seeds itself from its
    // own last answer" could not "give a different answer each run". Measured, the
    // opposite held. From compiled defaults, two runs minutes apart on the same idle
    // machine disagreed on three knobs (nb8_max 48/47, attn_min_tgs 72/144, flush_layers
    // 7/5), and st_gemm_large_shape lost a stored, end-to-end-verified 7 (+3.1% / +3.0%
    // at 5963 / 17123 tokens) to its never-measured default 3 on a 4.7% floor. Under the
    // incumbent-holds-the-seat rule a seat that is itself a measurement converges -- the
    // value stays until a candidate beats it by more than the run can resolve -- while a
    // seat that is a literal makes every noisy run a coin toss against the file. The
    // space version still keys the file, so a config measured against different kernels
    // never seeds a new build; and derived knobs are recomputed, not seated.
    let t_micro = Instant::now();
    (sp.prepare)();
    // WHICH GGML TYPE EACH WIRE KIND IS -- the same handover `backend::enable_gpu` makes at
    // model load, made here too because the tuner never loads a model's tensors. It is a
    // static table with no model input, so there is nothing to derive and nothing to get
    // wrong; what WAS wrong is that the tuner never handed it over at all. Without it the
    // Metal matmat cannot map a kind to a decode brick and REFUSES every dispatch above
    // Q8_0_TM, which no earlier model reached:
    //
    //   gemma4 Q4_0, LFM2 Q8_0_TM   kinds 1 and 3   dispatched      every sweep real
    //   qwen35 UD mix               kinds 22..32    all refused     38005 empty dispatches
    //
    // and the sweep still ranked twelve prefill tiles against each other on nothing.
    sp.ops
        .set_weight_kind_types(&imparo_gguf::weights::wire_kind_types());
    let mut tuner_mode = TunerModeGuard::enter(sp.ops)?;

    // Apply stored seats before init for selectors that influence native preparation.
    // CUDA installs architecture defaults during init, so the exact same seats are
    // reapplied below before the incumbent is captured. Without the second application,
    // a dependent knob is benchmarked against compiled defaults instead of the stored
    // route named by its `after` contract.
    let stored_config =
        imparo_host::read_for_device(model_bytes, true, (sp.tag, sp.version, &kv));
    match stored_config.as_ref() {
        Some(stored) => {
            let mut seated: Vec<String> = Vec::new();
            for (k, v) in &stored.knobs {
                match sp.reg.iter().find(|d| d.name == k.as_str()) {
                    Some(d) => {
                        (d.apply)(*v);
                        seated.push(format!("{k}={v}"));
                    }
                    None => {
                        println!("stored key '{k}' unknown to this build; not seated");
                    }
                }
            }
            println!(
                "seats applied from the stored config ({} of {} knobs): {}",
                seated.len(),
                sp.reg.len(),
                seated.join(" ")
            );
        }
        None => {
            println!(
                "seats: compiled defaults (no stored config for this fingerprint)"
            );
        }
    }

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
    // The layer tensors spread over `blk_bytes`; the lm-head, the largest GEMV of a
    // token, gets its own region after them at its real size (it does not fit in a
    // layer's slot: 278 MB on LFM2, 377 MB on E4B).
    let blk_bytes = widest.max(256 << 20);
    let lm_head_info: Option<(u64, u32, u32, u32, usize)> =
        ["output.weight", "token_embd.weight"]
            .iter()
            .find_map(|name| document.tensor(name))
            .and_then(|t| {
                let (ne0, ne1) = (*t.dimensions.first()?, *t.dimensions.get(1)?);
                let runtime_type = imparo_model::backend::runtime_weight_type(
                    &t.name,
                    t.ggml_type,
                    &t.dimensions,
                );
                let kind = imparo_gguf::weights::weight_kind(runtime_type)?;
                let layout = imparo_gguf::tensor_layout(runtime_type).ok()?;
                let bytes = (ne0 * ne1) as usize / layout.block_elements as usize
                    * layout.block_bytes as usize;
                Some((blk_bytes as u64, ne0 as u32, ne1 as u32, kind as u32, bytes))
            });
    let wbytes = blk_bytes + lm_head_info.map_or(0, |h| (h.4 + 16383) & !16383);
    let layout = std::alloc::Layout::from_size_align(wbytes, 16384)
        .map_err(|e| format!("synthetic weight layout: {e}"))?;
    // SAFETY: freshly allocated, 16 KB aligned, and it outlives every use below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    if base.is_null() {
        return Err("synthetic weight blob allocation failed".into());
    }
    // THE SAME HEAD DIMS THE ENGINE INJECTS, from the same place, before the same call.
    // The kernel library is compiled inside init_weights and specialises on these: one
    // attention kernel slot per dim the model uses. Skipping them here compiled a library
    // with NO qcomb slots at all, so the tuner's prefill attention workload dispatched
    // qtile while the engine dispatched qcomb, and every qcomb knob read "N/A to this
    // model" -- a whole kernel family the tuner could neither see nor rank.
    //
    // This is the rule that a workload must dispatch what the engine dispatches, one
    // level down: it applies to the LIBRARY, not only to the tensor shapes.
    let mut hds: Vec<u32> = plan
        .layers
        .iter()
        .filter(|l| l.attention.is_attention())
        .map(|l| l.attention.head_dim())
        .collect();
    hds.sort_unstable();
    hds.dedup();
    sp.ops.set_attention_head_dims(&hds);
    // The K/V row width goes with each dim, as the engine passes it (imparo-model
    // backend.rs): the prefill attention kernels compile the stride as a constant, and a
    // library compiled without it is a different kernel from the one the engine runs.
    let kvws: Vec<u32> = hds.iter().map(|hd| plan.config.n_kv_heads * hd).collect();
    sp.ops.set_attention_kv_widths(&kvws);
    sp.ops.set_activation(shapes.activation);
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

    if let Some(stored) = stored_config.as_ref() {
        for (name, value) in &stored.knobs {
            if let Some(declaration) = sp.reg.iter().find(|d| d.name == name.as_str()) {
                (declaration.apply)(*value);
            }
        }
    }
    // Validate the completed tuple below, once model facts are available. Checking
    // during the apply loop would miss a later coupled setter changing an earlier seat.

    // The decode matmuls at their REAL per-tensor dimensions, read from the header. The
    // tensor list comes from the model's bench extension (knobs.rs); the tuner core stays
    // model-agnostic. Offsets are spread across the blob rather than the model's own,
    // which would point past it -- the dimensions are what set the work, the spread is
    // what keeps the reads off one cache-hot page.
    let mbench = knobs::bench_for(&arch_name).ok_or_else(|| {
        format!(
            "no tuner workload description for GGUF architecture {arch_name}; refusing to apply another model's tensor mix"
        )
    })?;
    let decode_names = mbench.decode_tensors;
    let mut names: Vec<&'static str> = decode_names.to_vec();
    if let Some((gate, proj)) = mbench.ple_tensors {
        for name in [gate, proj] {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    if let Some(shortconv) = mbench.shortconv_tensor {
        if !names.contains(&shortconv) {
            names.push(shortconv);
        }
    }
    // THE QUANT TYPE IS READ PER TENSOR, not assumed. A GGUF carries a type per tensor and
    // mixed-quant files are normal, so a list that recorded dimensions and then measured
    // every entry as q4_0 would be timing a different kernel over a different number of
    // bytes for any tensor that is not q4_0.
    //
    // A tensor whose type this engine cannot run is DROPPED and counted, not silently
    // measured as something else -- the print below says how many made it.
    let mut unsupported: Vec<String> = Vec::new();
    // Keep one slot per declared tensor even when a tensor is unsupported. Flattening
    // only after named FFN extraction prevents one dropped projection from shifting all
    // later names onto the wrong offsets.
    let mapped: Vec<Option<(u64, u32, u32, u32)>> = names
        .iter()
        .enumerate()
        .map(|(i, n)| {
            document.tensor(&format!("blk.0.{n}.weight")).and_then(|t| {
                let (ne0, ne1) = (*t.dimensions.first()?, *t.dimensions.get(1)?);
                // THE KIND THE ENGINE DISPATCHES AFTER LOAD, not the file's: an original
                // Q8_0 file is transformed to tile-major at load on a backend with the
                // readers, and the knobs must be ranked on the kernels that will run.
                let runtime_type = imparo_model::backend::runtime_weight_type(
                    &t.name,
                    t.ggml_type,
                    &t.dimensions,
                );
                let Some(kind) = imparo_gguf::weights::weight_kind(runtime_type) else {
                    unsupported.push(format!("{n}(ggml={})", t.ggml_type));
                    return None;
                };
                let span = (blk_bytes as u64) / (names.len().max(1) as u64);
                // cudaMalloc supplies at least 256-byte alignment and the packed MMQ
                // kernels preserve that tensor-base contract. Spreading synthetic
                // tensors at arbitrary byte offsets violated the contract for nonzero
                // prefill projections (decode MMVQ happened to tolerate it).
                const SYNTHETIC_TENSOR_ALIGNMENT: u64 = 256;
                let off = ((i as u64) * span) & !(SYNTHETIC_TENSOR_ALIGNMENT - 1);
                // THE SIZE COMES FROM THE LAYOUT TABLE, not from a match on the kind.
                // It was a literal 18 (Q4_0), which under-counts a Q8_0 tensor by 89% --
                // 34 bytes per 32 values -- so the bounds check below admitted an offset
                // whose kernel then read past the synthetic weight buffer. A match here
                // fixed that but kept two mistakes: it repeated the geometry a third
                // time, and it hardcoded /32, which a k-quant's 256-element super-block
                // breaks silently. tensor_layout(runtime_type) answers both.
                let layout = imparo_gguf::tensor_layout(runtime_type).ok()?;
                let need = ne0 * ne1 / layout.block_elements * layout.block_bytes;
                (off + need <= blk_bytes as u64).then_some((
                    off,
                    ne0 as u32,
                    ne1 as u32,
                    kind as u32,
                ))
            })
        })
        .collect();
    let decode_mix: Vec<(u64, u32, u32, u32)> = mapped
        .iter()
        .take(decode_names.len())
        .flatten()
        .copied()
        .collect();
    let lookup_tensor = |wanted: &str| {
        names
            .iter()
            .position(|name| *name == wanted)
            .and_then(|index| mapped.get(index).copied().flatten())
    };
    let exact128_local_q = lookup_tensor("attn_q").filter(|&(_, n_in, n_out, kind)| {
        kind == 1
            && n_in == shapes.n_embd
            && n_out == shapes.n_head.saturating_mul(shapes.head_dim)
            && n_out != 0
    });
    let ffn_transaction = match (
        lookup_tensor("ffn_gate"),
        lookup_tensor("ffn_up"),
        lookup_tensor("ffn_down"),
    ) {
        (Some(gate), Some(up), Some(down))
            if matches!(gate.3, 1 | 2 | 3)
                && gate.3 == up.3
                && up.3 == down.3
                && gate.1 == shapes.n_embd
                && gate.2 == shapes.n_ff
                && up.1 == shapes.n_embd
                && up.2 == shapes.n_ff
                && down.1 == shapes.n_ff
                && down.2 == shapes.n_embd
                && gate.0 != up.0
                && gate.0 != down.0
                && up.0 != down.0 =>
        {
            Some(micro::FfnTransactionShape { gate, up, down })
        }
        _ => None,
    };
    let ple_transaction = mbench.ple_tensors.and_then(|(gate_name, proj_name)| {
        let gate = lookup_tensor(gate_name)?;
        let proj = lookup_tensor(proj_name)?;
        let per_layer_stride = gate.2.checked_mul(shapes.n_layers)?;
        (gate.3 == 1
            && proj.3 == 1
            && gate.1 == shapes.n_embd
            && gate.2 != 0
            && proj.1 == gate.2
            && proj.2 == shapes.n_embd
            && gate.0 != proj.0)
            .then_some(micro::PleTransactionShape {
                gate,
                proj,
                per_layer_stride,
            })
    });
    let mut recurrent_state = None;
    let mut recurrent_state_consistent = true;
    for state in plan
        .layers
        .iter()
        .filter_map(|layer| match layer.attention {
            imparo_model::Attention::Recurrent { r_elems, s_elems, .. } => {
                Some((r_elems, s_elems))
            }
            _ => None,
        })
    {
        match recurrent_state {
            None => recurrent_state = Some(state),
            Some(previous) if previous == state => {}
            Some(_) => {
                recurrent_state_consistent = false;
                break;
            }
        }
    }
    let shortconv_transaction = if recurrent_state_consistent {
        mbench.shortconv_tensor.and_then(|name| {
            let weight = lookup_tensor(name)?;
            let (state_elements, matrix_state_elements) = recurrent_state?;
            let kernel = weight.1;
            let width = weight.2;
            let expected_state = width.checked_mul(kernel.checked_sub(1)?)?;
            (weight.3 == 0
                && width == shapes.n_embd
                && matrix_state_elements == 0
                && state_elements == expected_state)
                .then_some(micro::ShortconvTransactionShape {
                    weight_off: weight.0,
                    width,
                    kernel,
                    state_elements,
                })
        })
    } else {
        None
    };
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
        decode_names.len()
    );
    // WHAT THE WORKLOADS WILL ACTUALLY DISPATCH, read from the tensors rather than
    // assumed. `weight_kinds` gates the per-quant knob families through `applies`;
    // `wide_kind` is what the two widest projections carry, which is the kind the prefill
    // tile knobs are ranked against.
    let lm_head = lm_head_info.map(|(off, i, o, k, _)| (off, i, o, k));
    // The lm-head's kind counts too: on a model transformed to tile-major at load it is
    // the one row-major GEMV left per token, and the row-major decode pair must apply.
    let weight_kinds = decode_mix
        .iter()
        .map(|&(_, _, _, k)| k)
        .chain(lm_head.map(|(_, _, _, k)| k))
        .fold(0u64, |m, k| {
            // The shift width is checked, not assumed: a kind past the mask's width would
            // wrap to another kind's bit and gate the wrong knob family, silently.
            assert!(k < 64, "weight kind {k} does not fit the weight_kinds mask");
            m | (1u64 << k)
        });
    if let Some((_, i, o, k)) = lm_head {
        println!(
            "  lm-head in the decode mix: {i} -> {o} kind {k}, weighted 1/{} (n_layers)",
            shapes.n_layers.max(1)
        );
    }
    let wide_kind = decode_mix
        .iter()
        .filter(|&&(_, i, o, _)| i.max(o) >= shapes.n_ff)
        .map(|&(_, _, _, k)| k)
        .next_back()
        .unwrap_or(1);
    println!(
        "  weight kinds present: mask {weight_kinds:#05b}, wide projections {wide_kind}"
    );
    let shapes = micro::Shapes {
        decode_mix,
        lm_head,
        exact128_local_q,
        ffn_transaction,
        ple_transaction,
        shortconv_transaction,
        layer_dispatches: mbench.layer_dispatches,
        weight_kinds,
        wide_kind,
        ..shapes
    };

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
    let limits = sp.ops.device_profile();
    if let Some(stored) = stored_config.as_ref() {
        for (name, requested) in &stored.knobs {
            if let Some(declaration) = sp.reg.iter().find(|d| d.name == name.as_str()) {
                let actual = (declaration.current)();
                if stored_seat_adjusted(
                    declaration,
                    *requested,
                    actual,
                    &facts,
                    &limits,
                )? {
                    println!(
                        "stored seat {name} requested={requested} adjusted to legal actual={actual} \
                         after device initialization; using actual as the incumbent{}",
                        if declaration.bit_affecting {
                            " (can change output bits; re-run numerical gates)"
                        } else {
                            ""
                        }
                    );
                }
            }
        }
    }
    // Capture only the final, checked readback -- never publish the stale request or
    // rank candidates against it. No kernel workload has run since stored-seat apply.
    let incumbent = Candidate {
        batch: seated_prefill_batch(stored_config.as_ref().and_then(|c| c.batch)),
        vals: sp.reg.iter().map(|d| (d.current)()).collect(),
    };
    println!(
        "prefill batch={} retained from the stored config or compiled default; batch is not swept",
        incumbent.batch
    );

    if explain {
        let profile = discover::profile(sp.ops, false);
        explain_registry(sp.reg, &facts, &profile);
        tuner_mode.close()?;
        return Ok(());
    }

    // ALWAYS profile: the point of running the tuner is to refresh what this machine is,
    // not to trust what it was. Measured once here and handed to the sweep.
    // Was this machine already measured BEFORE this process compiled its kernel library?
    // The library is built at init, from these numbers; discover runs after init. So on a
    // machine with no profile the sweep would rank knobs against a library built from
    // fallbacks, and the engine would then compile a different one. Measure, write, stop.
    let had_profile = imparo_host::read_device_profile(sp.tag).is_some();
    let measured = discover::profile(sp.ops, true);
    sp.ops.validate_tuner_profile(&measured)?;
    store_device_profile(sp.tag, &measured)?;
    if discover_only || !had_profile {
        if had_profile {
            println!("ALL device profile written; no knob swept (--discover-only)");
        } else {
            println!(
                "ALL device profile written -- this machine had none, and the kernel \
                 library in THIS process was compiled before these numbers existed. \
                 Re-run to tune against the library the engine will use."
            );
        }
        tuner_mode.close()?;
        return Ok(());
    }
    for external in &external_candidates {
        let declaration = sp
            .reg
            .iter()
            .find(|declaration| declaration.name == external.name)
            .expect("external candidate was registry-validated before model init");
        external_candidate_membership(
            declaration,
            external.value,
            Some((&facts, &measured)),
        )?;
        if !declaration.applies.is_none_or(|applies| applies(&facts)) {
            return Err(format!(
                "--external-candidate {} does not apply to this model",
                external.name
            )
            .into());
        }
        if declaration
            .legal
            .is_some_and(|legal| !legal(external.value, &facts, &measured))
        {
            return Err(format!(
                "--external-candidate {}={} is illegal for this model/device",
                external.name, external.value
            )
            .into());
        }
    }
    let picks = micro::measure(
        sp.ops,
        sp.reg,
        &shapes,
        true,
        allow_bits,
        &measured,
        only_knob.as_deref(),
        &incumbent.vals,
    )
    .map_err(|e| format!("micro: {e}"))?;
    let mut cur = incumbent.clone();
    for (name, v) in &picks.picks {
        cur.set(sp.reg, name, *v);
    }
    for external in &external_candidates {
        cur.set(sp.reg, &external.name, external.value);
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
    if only_knob.is_some() && !emit_candidate {
        tuner_mode.close()?;
        println!("--knob: one knob measured, config NOT written");
        return Ok(());
    }
    let against = if external_candidates.is_empty() {
        only_knob.as_deref().map_or(
            "micro-benches only (no end-to-end search)".to_string(),
            |name| {
                format!("single {name} micro-bench; all other knobs compiled defaults")
            },
        )
    } else {
        let tuple = external_candidates
            .iter()
            .map(|external| format!("{}={}", external.name, external.value))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "external whole-model candidate tuple {tuple}; NOT admitted until \
                 end-to-end timing and the fixed correctness receipt both pass"
        )
    };
    let knob_lines =
        sp.reg
            .iter()
            .zip(&cur.vals)
            .fold(String::new(), |mut s, (d, v)| {
                use std::fmt::Write as _;
                let _ = writeln!(s, "{}={v}", d.name);
                s
            });
    let body = format!(
        "# imparo host tuning, {} search space v{}\n\
         # complete candidate; selection scope is recorded below\n\
         # verified against: {against}\n\
         fingerprint={fp}\n\
         toolchain={}\n\
         model={}\nmodel_bytes={model_bytes}\n\
         {knob_lines}batch={}\n",
        sp.tag,
        sp.version,
        imparo_host::toolchain(),
        path.file_name()
            .map_or_else(|| "?".into(), |n| n.to_string_lossy().into_owned()),
        cur.batch
    );
    // A CONFIG IS A CLAIM THAT THE SWEEP DISPATCHED. If the backend refused any dispatch,
    // some candidate was ranked against work that never ran, and the file would record that
    // as a measurement. This is the same rule #74 gave speed_gate -- an untuned or unproven
    // baseline cannot be recorded -- applied to the tuner's own output.
    //
    // It caught nothing on gemma4 or LFM2 because their weight kinds are 1 and 3, the two
    // the Metal matmat serves without the wire-kind table. A qwen35 tune refused 38005
    // dispatches, printed 38005 lines, exited 0 and wrote a config in which twelve prefill
    // tiles had each measured the same empty dispatch.
    let refused = sp.ops.refused_dispatches();
    if refused > 0 {
        return Err(format!(
            "the backend REFUSED {refused} dispatch(es) during this tune -- some candidate \
             was ranked against work that never ran, so nothing is written. The refusals are \
             logged with their weight kind and n_out; a kind above Q8_0_TM usually means the \
             wire-kind table never reached the backend."
        )
        .into());
    }
    // Publish only after production stream/graph state was restored successfully.
    tuner_mode.close()?;
    imparo_host::receipted_config::write_atomic(&out, body.as_bytes())?;
    println!("ALL {}  -> {}", cur.line(sp.reg), out.display());
    Ok(())
}

#[cfg(test)]
// These tests stay beside the environment parsing/apply path they protect; the
// registry explainer below is production-only presentation code.
#[allow(clippy::items_after_test_module)]
mod environment_tests {
    use super::{
        ExternalCandidate, cuda_tuner_env_is_polluting, parse_external_candidate,
    };

    #[test]
    fn parses_explicit_external_candidate() {
        assert_eq!(
            parse_external_candidate("prefill_exact128_sm86_route=5"),
            Ok(ExternalCandidate {
                name: "prefill_exact128_sm86_route".into(),
                value: 5,
            })
        );
    }

    #[test]
    fn rejects_malformed_external_candidate() {
        for raw in ["", "route", "=5", "route=", "route=5=6", "route=-1"] {
            assert!(parse_external_candidate(raw).is_err(), "{raw}");
        }
    }

    #[test]
    fn cuda_tuner_rejects_route_and_diagnostic_injection() {
        for name in [
            "IMPARO_GPU_PROBE",
            "IMPARO_CUDA_PREFILL_FFN_SIDECAR_ONLY_LAB",
            "IMPARO_CUDA_PREFILL_FFN_SIDECAR_FAIL_PRECOMMIT_LAB",
            "IMPARO_CUDA_PROFILE_OPS",
            "IMPARO_CUDA_PREFILL_FFN_SIDECAR_TRACE",
            "IMPARO_CUDA_ROUTE_DUMP",
            "IMPARO_CUDA_FORCE_OOM",
        ] {
            assert!(cuda_tuner_env_is_polluting(name), "{name}");
        }
    }

    #[test]
    fn cuda_tuner_keeps_identity_and_device_environment() {
        for name in [
            "IMPARO_CUDA_BACKEND",
            "IMPARO_CUDA_DEVICE",
            "IMPARO_CUDA_SM",
            "IMPARO_CUDA_ARCHS",
            "IMPARO_CUDA_RESERVE_MIB",
        ] {
            assert!(!cuda_tuner_env_is_polluting(name), "{name}");
        }
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod stored_seat_tests {
    use super::{
        external_candidate_membership, seated_prefill_batch, stored_seat_adjusted,
    };
    use imparo_backend::{
        DeviceProfile, KnobCategory, KnobDecl, ModelFacts, SweepKind, Workload,
    };

    fn fixture() -> (KnobDecl, ModelFacts, DeviceProfile) {
        (
            KnobDecl {
                name: "device_limited_seat",
                legal: Some(|value, _, device| {
                    value > 0 && value <= device.max_threads
                }),
                applies: None,
                bit_affecting: true,
                derive: None,
                candidates: None,
                after: &[],
                cross_check: None,
                tuple: None,
                category: KnobCategory::EndToEnd,
                values: &[],
                apply: |_| {},
                current: || 0,
                screened: false,
                sweep: SweepKind::External,
                workload: Workload::DecodeMix,
            },
            ModelFacts {
                n_embd: 2048,
                n_ff: 10752,
                n_head: 32,
                n_kv: 8,
                head_dim: 64,
                deep_head_dim: 64,
                n_experts: 0,
                n_layers: 30,
                layer_dispatches: 0,
                weight_kinds: 4,
            },
            DeviceProfile {
                max_threads: 20,
                ..DeviceProfile::default()
            },
        )
    }

    #[test]
    fn preserves_model_level_batch_while_tuning_backend_knobs() {
        for batch in [1, 512, 1024, 2048] {
            assert_eq!(seated_prefill_batch(Some(batch)), batch);
        }
    }

    #[test]
    fn batch_without_a_stored_value_uses_the_shared_default() {
        assert_eq!(seated_prefill_batch(None), imparo_model::PREFILL_BATCH);
    }

    #[test]
    fn zero_batch_matches_the_runtime_effective_minimum() {
        assert_eq!(seated_prefill_batch(Some(0)), 1);
    }

    #[test]
    fn external_dynamic_ladder_is_resolved_after_discovery() {
        let (mut decl, facts, mut device) = fixture();
        decl.candidates = Some(|_, d| {
            [8, 16, 24, 32]
                .into_iter()
                .filter(|v| v * 32 <= d.max_threads)
                .collect()
        });
        // Before discovery, an empty static values list is NOT the dynamic ladder.
        assert!(external_candidate_membership(&decl, 24, None).is_ok());
        device.max_threads = 1024;
        assert!(
            external_candidate_membership(&decl, 24, Some((&facts, &device))).is_ok()
        );
        assert!(
            external_candidate_membership(&decl, 23, Some((&facts, &device))).is_err()
        );
        device.max_threads = 512;
        assert!(
            external_candidate_membership(&decl, 24, Some((&facts, &device))).is_err()
        );
        assert!(
            external_candidate_membership(&decl, 16, Some((&facts, &device))).is_ok()
        );
    }

    #[test]
    fn external_static_ladder_still_rejects_undeclared_values_at_preflight() {
        let (mut decl, facts, device) = fixture();
        decl.values = &[1, 2];
        for context in [None, Some((&facts, &device))] {
            assert!(external_candidate_membership(&decl, 2, context).is_ok());
            assert!(external_candidate_membership(&decl, 3, context).is_err());
        }
    }

    #[test]
    fn external_dynamic_ladder_takes_precedence_over_static_values() {
        let (mut decl, facts, device) = fixture();
        decl.values = &[32];
        decl.candidates = Some(|_, _| vec![24]);
        assert!(
            external_candidate_membership(&decl, 24, Some((&facts, &device))).is_ok()
        );
        assert!(
            external_candidate_membership(&decl, 32, Some((&facts, &device))).is_err()
        );
    }

    #[test]
    fn accepts_device_clamping_of_stale_stored_requests() {
        let (decl, facts, mut device) = fixture();
        assert_eq!(
            stored_seat_adjusted(&decl, 36, 20, &facts, &device),
            Ok(true)
        );
        for limit in [1, 8, 18, 20, 32, 64] {
            device.max_threads = limit;
            assert_eq!(
                stored_seat_adjusted(&decl, limit * 2, limit, &facts, &device),
                Ok(true)
            );
        }
    }

    #[test]
    fn unchanged_readback_is_not_reported_as_an_adjustment() {
        let (decl, facts, device) = fixture();
        assert_eq!(
            stored_seat_adjusted(&decl, 16, 16, &facts, &device),
            Ok(false)
        );
    }

    #[test]
    fn a_legal_value_lost_by_the_setter_still_fails() {
        let (decl, facts, device) = fixture();
        let error = stored_seat_adjusted(&decl, 16, 8, &facts, &device).unwrap_err();
        assert!(error.contains("device_limited_seat requested=16 actual=8"));
    }

    #[test]
    fn an_illegal_replacement_is_not_accepted() {
        let (decl, facts, device) = fixture();
        assert!(stored_seat_adjusted(&decl, 36, 24, &facts, &device).is_err());
        assert!(stored_seat_adjusted(&decl, 36, 0, &facts, &device).is_err());
    }

    #[test]
    fn a_mismatch_without_a_legality_rule_still_fails() {
        let (mut decl, facts, device) = fixture();
        decl.legal = None;
        assert!(stored_seat_adjusted(&decl, 36, 20, &facts, &device).is_err());
    }
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
    println!(
        "\nMODEL   n_embd={} n_ff={} heads={} kv={} head_dim={} deep={} experts={}",
        facts.n_embd,
        facts.n_ff,
        facts.n_head,
        facts.n_kv,
        facts.head_dim,
        facts.deep_head_dim,
        facts.n_experts
    );
    println!(
        "DEVICE  threadgroup={} B  threads={}  accumulators={}  cache_knee={} MB  dram={} MB/s",
        profile.threadgroup_bytes,
        profile.max_threads,
        profile.max_accumulators,
        profile.cache_knee_bytes >> 20,
        profile.dram_read_mbs
    );
    println!(
        "\n{:<22} {:<10} {:<22} {:<14} {:<26} NOTES",
        "KNOB", "DECIDED BY", "MEASURED IN", "TUPLE", "LADDER"
    );
    // The ladder a knob will actually be swept on, resolved the same way the sweep
    // resolves it. A derived ladder cannot be read from the source at all -- that is the
    // point of deriving it -- so the one command whose job is to make the registry
    // readable has to compute it too.
    let ladder = |d: &KnobDecl| -> Vec<u32> {
        match d.candidates {
            Some(f) => f(facts, profile),
            None => d.values.to_vec(),
        }
    };
    for d in reg {
        let how = match d.sweep {
            Sw::Derived => "computed",
            Sw::External => "end-to-end",
            Sw::Values => "swept",
            Sw::Crossing { .. }
            | Sw::TokenMinCrossing { .. }
            | Sw::SpanCrossing { .. } => "boundary",
        };
        let regime = match d.sweep {
            Sw::Derived => "-".to_string(),
            Sw::External => "full model".to_string(),
            // A boundary drives its own ladder and never reads `workload`; printing it
            // would state something the code does not do.
            Sw::Crossing { .. }
            | Sw::TokenMinCrossing { .. }
            | Sw::SpanCrossing { .. } => "own ladder".to_string(),
            Sw::Values => format!("{:?}", d.workload),
        };
        let mut notes: Vec<String> = Vec::new();
        if !d.applies.is_none_or(|f| f(facts)) {
            notes.push("N/A to this model".into());
        }
        if d.bit_affecting {
            notes.push("CHANGES BITS".into());
        }
        if d.category == imparo_backend::KnobCategory::EndToEnd {
            notes.push("external performance admission required".into());
        }
        if d.legal.is_some() {
            // AGAINST THE RESOLVED LADDER, not `values`: every knob that derives its
            // candidates leaves `values` empty, so this note was silently empty for
            // exactly the knobs whose ladder nobody can read off the source.
            let bad: Vec<String> = ladder(d)
                .iter()
                .filter(|v| !d.legal.is_some_and(|f| f(**v, facts, profile)))
                .map(ToString::to_string)
                .collect();
            if !bad.is_empty() {
                notes.push(format!("illegal here: {}", bad.join(",")));
            }
        }
        if let Some(cw) = d.cross_check {
            notes.push(format!("cross-checked in {cw:?}"));
        }
        if matches!(
            d.sweep,
            Sw::Crossing { .. } | Sw::TokenMinCrossing { .. } | Sw::SpanCrossing { .. }
        ) {
            notes.push("defines a regime".into());
        }
        let rungs = match d.sweep {
            Sw::Crossing { .. }
            | Sw::TokenMinCrossing { .. }
            | Sw::SpanCrossing { .. } => "own ladder".to_string(),
            Sw::Derived => "-".to_string(),
            Sw::External | Sw::Values => {
                let v = ladder(d);
                let text = v
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                if d.candidates.is_some() {
                    format!("{text} (derived)")
                } else {
                    text
                }
            }
        };
        println!(
            "{:<22} {:<10} {:<22} {:<14} {:<26} {}",
            d.name,
            how,
            regime,
            d.tuple.unwrap_or("-"),
            rungs,
            notes.join("; ")
        );
    }
    println!(
        "\nA boundary knob PARTITIONS the runtime state into regimes, which is why it is"
    );
    println!(
        "searched after the knobs measured within them. A tuple is several knobs that are"
    );
    println!(
        "really one decision. A computed knob is never searched -- ground truth already"
    );
    println!("determines it.");
}
