//! One fixed, non-timing check of the actual production W4A16 FFN.
//! Include only in the Windows cuda-speculative imparo-forward diagnostic.
//! The caller must supply the already frozen real layer/input fixture.
use std::io::Read as _;
use std::path::Path;

use imparo_backend::{Backend, BatchGeometry, BatchPhase, BufId, Epilogue};
use imparo_cuda::correctness::{read_f32_checked, write_f32_checked};
use imparo_cuda::knobs::CUDA_KNOBS;
use imparo_model::{ops, weights::Weights};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const H: usize = 2560;
const F: usize = 10240;
const M: usize = 3;
const MAX_REL_L2: f64 = 0.01; // Existing complete-FFN FP32 reference ceiling.

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}
fn sha(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn exact(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn rc<T>(result: Result<T, i32>, where_: &str) -> Result<T, String> {
    result.map_err(|code| format!("{where_} failed rc={code}"))
}
fn registry() -> Value {
    json!(
        CUDA_KNOBS
            .iter()
            .map(|d| (d.name, (d.current)()))
            .collect::<Vec<_>>()
    )
}

fn checked_environment() -> Result<bool, String> {
    if std::env::var("IMPARO_CORRECTNESS_GATE").as_deref() != Ok("1") {
        return Err("W4A16 FFN check requires IMPARO_CORRECTNESS_GATE=1".into());
    }
    // Do not inherit unbound kernel, capture, scheduler or numeric overrides.
    const ALLOWED: &[&str] = &[
        "IMPARO_CORRECTNESS_GATE",
        "IMPARO_HOST_CONFIG",
        "IMPARO_GPU",
        "IMPARO_BACKEND",
        "IMPARO_CTK",
        "IMPARO_CTV",
        "IMPARO_KV_CAP",
        "IMPARO_BACKEND_CACHE",
        "IMPARO_CUDA_ARCHS",
        "IMPARO_CUDA_BACKEND",
        "IMPARO_CUDA_DEVICE",
        "IMPARO_CUDA_RESERVE_MIB",
        "IMPARO_CUDA_SM",
        "IMPARO_CUDA_WEIGHT_CACHE_MIB",
        "IMPARO_MODEL",
        "IMPARO_REF_MODEL",
        "IMPARO_ENGINE",
        "IMPARO_REF_ENGINE",
        "IMPARO_REF_MANIFEST",
        "IMPARO_REF_MANIFEST_SHA256",
        "IMPARO_REF_CWD",
        "IMPARO_REF_ARGS",
    ];
    for (key, _) in std::env::vars_os() {
        let key = key.to_string_lossy();
        if key.to_ascii_uppercase().starts_with("IMPARO_")
            && !ALLOWED.contains(&key.as_ref())
        {
            return Err(format!(
                "unbound environment is forbidden in W4A16 check: {key}"
            ));
        }
    }
    if std::env::var("IMPARO_BACKEND").as_deref() != Ok("cuda")
        || std::env::var("IMPARO_GPU").as_deref() != Ok("1")
    {
        return Err(
            "W4A16 check requires explicit IMPARO_BACKEND=cuda and IMPARO_GPU=1".into(),
        );
    }
    Ok(std::env::var_os("IMPARO_HOST_CONFIG").is_some())
}

struct Projection {
    offset: u64,
    kind: u32,
    bytes: Vec<u8>,
}
struct Fixture {
    gate: Projection,
    up: Projection,
    down: Projection,
    norm_offset: u64,
    post_offset: u64,
    norm: Vec<f32>,
    post: Vec<f32>,
    eps: f32,
    projection_follows: bool,
}
fn projection(
    weights: &Weights,
    name: &str,
    ni: usize,
    no: usize,
) -> Result<Projection, String> {
    let t = weights
        .get(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if t.ne0() != ni || t.ne1() != no || t.ggml_type != imparo_gguf::weights::GGML_Q4_0
    {
        return Err(format!(
            "{name} does not match the frozen Q4_0 {ni}x{no} domain"
        ));
    }
    Ok(Projection {
        offset: t.offset as u64,
        kind: t.ggml_type,
        bytes: weights.raw(t).to_vec(),
    })
}
fn load_norm(weights: &Weights, name: &str) -> Result<(u64, Vec<f32>), String> {
    let t = weights
        .get(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if t.ggml_type != imparo_gguf::weights::GGML_F32 || t.elements() != H {
        return Err(format!("{name} is not the frozen F32 norm"));
    }
    let values = weights.f32s(t).to_vec();
    if !values.iter().all(|v| v.is_finite()) {
        return Err(format!("nonfinite norm {name}"));
    }
    Ok((t.offset as u64, values))
}
fn cpu_reference(f: &Fixture, input: &[f32]) -> Vec<f32> {
    let n = input.len() / H;
    let mut normalized = input.to_vec();
    for row in normalized.chunks_exact_mut(H) {
        ops::rms_norm(row, Some(&f.norm), f.eps);
    }
    let mut gate = vec![0.0; n * F];
    let mut up = vec![0.0; n * F];
    ops::mul_mat_bytes(&f.gate.bytes, f.gate.kind, H, F, &normalized, n, &mut gate);
    ops::mul_mat_bytes(&f.up.bytes, f.up.kind, H, F, &normalized, n, &mut up);
    ops::gelu(&mut gate);
    ops::mul_into(&mut gate, &up);
    let mut output = vec![0.0; n * H];
    ops::mul_mat_bytes(&f.down.bytes, f.down.kind, F, H, &gate, n, &mut output);
    for (row, residual) in output.chunks_exact_mut(H).zip(input.chunks_exact(H)) {
        ops::rms_norm(row, Some(&f.post), f.eps);
        for (v, add) in row.iter_mut().zip(residual) {
            *v += *add;
        }
    }
    output
}
fn rel_l2(actual: &[f32], reference: &[f32]) -> f64 {
    let mut diff = 0.0f64;
    let mut norm = 0.0f64;
    for (&a, &r) in actual.iter().zip(reference) {
        diff += (f64::from(a) - f64::from(r)).powi(2);
        norm += f64::from(r).powi(2);
    }
    (diff / norm.max(1e-30)).sqrt()
}

fn run_ffn(
    be: &dyn Backend,
    f: &Fixture,
    input: &[f32],
    w4: bool,
) -> Result<Vec<f32>, String> {
    let n = u32::try_from(input.len() / H).map_err(|_| "token count overflow")?;
    if !matches!(n, 1 | 3) || input.len() != n as usize * H {
        return Err("invalid fixed FFN input".into());
    }
    be.begin_forward(n == 1);
    let encoded = (|| {
        let phase = if n == 1 {
            BatchPhase::Decode
        } else {
            BatchPhase::Prefill
        };
        let geometry = BatchGeometry::try_new(512, n, phase).map_err(str::to_string)?;
        rc(be.set_batch_geometry(geometry), "set real FFN geometry")?;
        rc(write_f32_checked(BufId::O, 0, input), "upload FFN input")?;
        let poison = vec![f32::from_bits(0x7fc12345); input.len()];
        rc(write_f32_checked(BufId::X, 0, &poison), "poison FFN output")?;
        be.set_activation(Epilogue::Gelu);
        be.set_epilogue(Epilogue::None);
        be.rms_norm_projection(
            BufId::Cur,
            BufId::O,
            f.norm_offset,
            H as u32,
            f.eps,
            n,
            H as u32,
            0,
        );
        if w4 {
            if !be.ffn_gated_down(
                1,
                f.gate.offset,
                1,
                f.up.offset,
                1,
                f.down.offset,
                H as u32,
                F as u32,
                H as u32,
                BufId::Cur,
                BufId::G,
                BufId::X,
                n,
            ) {
                return Err("selected production W4A16 FFN was not dispatched".into());
            }
        } else {
            // Only the restore witness: read the original canonical matrices before/after packing.
            be.matmat(
                1,
                f.gate.offset,
                H as u32,
                F as u32,
                BufId::Cur,
                BufId::G,
                n,
            );
            be.matmat(1, f.up.offset, H as u32, F as u32, BufId::Cur, BufId::U, n);
            be.act_mul(BufId::G, BufId::U, n * F as u32);
            be.matmat(1, f.down.offset, F as u32, H as u32, BufId::G, BufId::X, n);
        }
        let prepared = f.projection_follows
            && be.rms_norm_add_projection(
                BufId::X,
                BufId::X,
                f.post_offset,
                H as u32,
                f.eps,
                n,
                H as u32,
                0,
                BufId::O,
            );
        let fused = prepared
            || be.rms_norm_add(
                BufId::X,
                BufId::X,
                f.post_offset,
                H as u32,
                f.eps,
                n,
                H as u32,
                0,
                BufId::O,
                1.0,
            );
        if !fused {
            be.rms_norm(BufId::X, f.post_offset, H as u32, f.eps, n, H as u32, 0);
            be.add(BufId::X, BufId::O, n * H as u32);
        }
        Ok(())
    })();
    // Preserve both the original encode failure and any asynchronous/pending error.
    let completed = rc(be.end(), "complete FFN transaction");
    match (encoded, completed) {
        (Err(a), Err(b)) => return Err(format!("{a}; {b}")),
        (Err(a), _) | (_, Err(a)) => return Err(a),
        _ => {}
    }
    let mut output = vec![f32::from_bits(0x7fc54321); input.len()];
    rc(
        read_f32_checked(BufId::X, 0, &mut output),
        "read FFN output",
    )?;
    if !output.iter().all(|v| v.is_finite()) {
        return Err("nonfinite/incomplete FFN output".into());
    }
    Ok(output)
}

/// Returns one evidence object; callers must print it and preserve nonzero exit on Err.
/// This is a fixed check, not a tuner workload and not a sealed receipt.
pub fn run(
    model_path: &Path,
    input_path: &Path,
    layer: usize,
) -> Result<Value, String> {
    let has_config = checked_environment()?;
    let document = imparo_gguf::read(model_path).map_err(|e| e.to_string())?;
    let plan =
        imparo_model::build_plan(&document, model_path).map_err(|e| e.to_string())?;
    if plan.config.architecture != "gemma4"
        || plan.config.n_layers != 42
        || plan.config.n_embd != H as u32
        || plan.config.n_ff != F as u32
        || layer >= 42
    {
        return Err("W4A16 check requires Gemma4 E4B H2560/F10240/L42".into());
    }
    let mut weights = Weights::open(model_path).map_err(|e| e.to_string())?;
    let (norm_offset, norm) =
        load_norm(&weights, &format!("blk.{layer}.ffn_norm.weight"))?;
    let (post_offset, post) =
        load_norm(&weights, &format!("blk.{layer}.post_ffw_norm.weight"))?;
    let fixture = Fixture {
        gate: projection(&weights, &format!("blk.{layer}.ffn_gate.weight"), H, F)?,
        up: projection(&weights, &format!("blk.{layer}.ffn_up.weight"), H, F)?,
        down: projection(&weights, &format!("blk.{layer}.ffn_down.weight"), F, H)?,
        norm_offset,
        post_offset,
        norm,
        post,
        eps: plan.config.norm_eps,
        projection_follows: plan.embed.per_layer_dim.unwrap_or(0) > 0,
    };
    let mut raw = Vec::with_capacity(M * H * 4 + 1);
    std::fs::File::open(input_path)
        .map_err(|e| e.to_string())?
        .take((M * H * 4 + 1) as u64)
        .read_to_end(&mut raw)
        .map_err(|e| e.to_string())?;
    if raw.len() != M * H * 4 {
        return Err("fixture must be exactly 3x2560 little-endian F32".into());
    }
    let input: Vec<f32> = raw
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    if !input.iter().all(|v| v.is_finite()) {
        return Err("fixture contains nonfinite input".into());
    }
    let model_sha = hex(&weights.full_file_sha256());
    let reference = cpu_reference(&fixture, &input);
    if !reference.iter().all(|v| v.is_finite()) {
        return Err("nonfinite CPU FP32 reference".into());
    }
    let capacity = std::env::var("IMPARO_KV_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    imparo_model::backend::enable_gpu(
        &mut weights,
        &plan,
        capacity,
    )?;
    let choice = CUDA_KNOBS
        .iter()
        .find(|d| d.name == "e4b_ffn_w4a16")
        .ok_or("missing W4A16 registry entry")?;
    if !has_config {
        (choice.apply)(1);
    }
    let selected_policy = (choice.current)();
    if !matches!(selected_policy, 1 | 2) {
        return Err(
            "actual registry W4A16 policy is not 1 or 2; no config override permitted"
                .into(),
        );
    }
    let snapshot = registry();
    let mut model = imparo_model::load(weights, plan, capacity)?;
    if !model.has_device_workflow() {
        return Err("model has no CUDA workflow".into());
    }
    model.ensure_gpu_ready()?; // Existing Gemma admission registers every one of the 42 FFN spans.
    let be = imparo_model::backend::active().ok_or("CUDA backend absent")?;
    for (id, n) in [
        (BufId::Cur, M * H),
        (BufId::O, M * H),
        (BufId::X, M * H),
        (BufId::G, M * F),
        (BufId::U, M * F),
    ] {
        rc(be.alloc(id, (n * 4) as u64), "ensure fixed FFN buffer")?;
    }
    let mut untouched = [42.0_f32];
    if read_f32_checked(BufId::Cur, u64::MAX, &mut untouched).is_ok()
        || write_f32_checked(BufId::Cur, u64::MAX, &untouched).is_ok()
        || untouched != [42.0_f32]
    {
        return Err("checked transfer did not reject an overflowing offset".into());
    }
    rc(
        be.prepare_projection_phase(false),
        "canonical phase before witness",
    )?;
    let canonical_before = run_ffn(be, &fixture, &input, false)?;
    rc(
        be.prepare_projection_phase(true),
        "load and pack all FFN spans",
    )?;
    let m3 = run_ffn(be, &fixture, &input, true)?;
    let m1 = run_ffn(be, &fixture, &input[H..2 * H], true)?;
    rc(
        be.prepare_projection_phase(false),
        "restore canonical FFN weights",
    )?;
    let canonical_after = run_ffn(be, &fixture, &input, false)?;
    rc(be.prepare_projection_phase(true), "repack FFN weights")?;
    let repacked = run_ffn(be, &fixture, &input, true)?;
    rc(
        be.prepare_projection_phase(false),
        "leave canonical weight phase",
    )?;
    rc(be.end(), "final gate synchronization")?;
    let error = rel_l2(&m3, &reference);
    let row_errors: Vec<f64> = m3
        .chunks_exact(H)
        .zip(reference.chunks_exact(H))
        .map(|(a, b)| rel_l2(a, b))
        .collect();
    let canonical_error = rel_l2(&canonical_before, &reference);
    let row_exact = exact(&m1, &m3[H..2 * H]);
    let restore_exact = exact(&canonical_before, &canonical_after);
    let repack_exact = exact(&m3, &repacked);
    let snapshot_after = registry();
    let passed = error <= MAX_REL_L2
        && error <= canonical_error + 0.001
        && row_errors.iter().all(|v| *v <= MAX_REL_L2)
        && row_exact
        && restore_exact
        && repack_exact
        && snapshot == snapshot_after;
    let result = json!({
        "gate":"w4a16_full_ffn_fp32_reference","gate_version":1,"passed":passed,
        "model_sha256":model_sha,"input_sha256":sha(&raw),"layer":layer,"shape":[M,H,F],
        "gate_weight_sha256":sha(&fixture.gate.bytes),"up_weight_sha256":sha(&fixture.up.bytes),
        "down_weight_sha256":sha(&fixture.down.bytes),"norm_sha256":sha(&f32_bytes(&fixture.norm)),
        "post_norm_sha256":sha(&f32_bytes(&fixture.post)),"norm_eps":fixture.eps,
        "registry":snapshot,"registry_after":snapshot_after,"host_config_present":has_config,
        "w4a16_policy":selected_policy,
        "sealed_receipt":false,"timing_admissible":false,"candidate_math_relL2":error,
        "canonical_math_relL2":canonical_error,"per_row_relL2":row_errors,"max_relL2":MAX_REL_L2,
        "m1_matches_m3_middle_row_exact":row_exact,"canonical_restore_exact":restore_exact,
        "repack_exact":repack_exact,"output_sha256":sha(&f32_bytes(&m3)),
        "reference":"existing CPU Q4 dequant + FP32 dot/RMS/GELU; no Q8 activation quantization",
        "readback":"checked upload/download return codes and stream synchronization; finite poisoned output",
        "restore_scope":"all42 phase transitions, selected layer complete FFN behavior; not an all-weight byte proof"
    });
    if !passed {
        return Err(format!("W4A16 production FFN gate failed: {result}"));
    }
    Ok(result)
}
