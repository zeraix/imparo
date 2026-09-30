//! Explicit activation bases carried by a rotated-weight GGUF.
//!
//! This is model metadata, not a tuner option. The backend must apply a forward
//! transform before each named matmul and the inverse after embedding lookup, or
//! refuse the model. PTQ1_0 payloads must never be treated as ordinary Qwen weights.
//! Contract checked against PrismML llama.cpp d8f26ee (llama-graph.cpp,
//! build_lora_mm/build_inp_embd; llama-model.cpp Hadamard metadata loading).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use imparo_backend::{GroupedHeads, WeightInputTransform};
use imparo_gguf::{Document, MetadataValue, Scalar};

const PREFIX: &str = "prism.hadamard.";
const PTQ1_0: u32 = 143;
const MAX_NAMES: u64 = 65_536;
const MAX_SIGNS: u64 = 4_194_304;
const ARRAY_BUDGET: u64 = 256 * 1024 * 1024;

/// Checks scalar conventions without reading tensor payloads.
/// Returns false only for a model with no rotation metadata and no PTQ1_0 weights.
pub fn validate_header(doc: &Document) -> Result<bool, String> {
    let present = doc.metadata.keys().any(|key| key.starts_with(PREFIX));
    if !present {
        if doc.tensors.iter().any(|t| t.ggml_type == PTQ1_0) {
            return Err(
                "PTQ1_0 weights require explicit prism.hadamard metadata".into()
            );
        }
        return Ok(false);
    }
    if doc.string_value("general.architecture") != Some("qwen35") {
        return Err(
            "Prism activation bases currently require the qwen35 workflow".into(),
        );
    }
    if doc.unsigned_value("prism.hadamard.version") != Some(1) {
        return Err("unsupported or missing prism.hadamard.version (expected 1)".into());
    }
    if doc.unsigned_value("prism.hadamard.block_size") != Some(1024) {
        return Err("Prism activation bases currently require block_size 1024".into());
    }
    for (key, expected) in [
        (
            "prism.hadamard.transform",
            "normalized-sylvester-walsh-hadamard",
        ),
        ("prism.hadamard.axis", "input-last-dimension"),
        ("prism.hadamard.sign_mode", "explicit"),
    ] {
        if doc.string_value(key) != Some(expected) {
            return Err(format!("unsupported or missing {key}: expected {expected}"));
        }
    }
    match doc.metadata.get("prism.hadamard.gdn_v_grouped") {
        None | Some(MetadataValue::Scalar(Scalar::Bool(_))) => {}
        _ => return Err("prism.hadamard.gdn_v_grouped must be a boolean".into()),
    }
    Ok(true)
}

fn array(
    doc: &Document,
    path: &Path,
    key: &str,
    max: u64,
) -> Result<Vec<Scalar>, String> {
    let values = imparo_gguf::read_metadata_array(path, key, max, ARRAY_BUDGET)
        .map_err(|e| format!("{key}: {e}"))?;
    let Some(MetadataValue::Array(summary)) = doc.metadata.get(key) else {
        return Err(format!("missing array {key}"));
    };
    if summary.element_type != values.element_type
        || summary.element_count != values.values.len() as u64
        || !values.values.starts_with(&summary.preview)
    {
        return Err(format!("{key} changed since the model header was read"));
    }
    Ok(values.values)
}

/// Resolve names to file offsets, the same offsets stored in `Weights::Tensor`.
/// Repacking can change a weight kind but does not change this address identity.
pub fn from_document(
    doc: &Document,
    path: &Path,
) -> Result<Vec<WeightInputTransform>, String> {
    if !validate_header(doc)? {
        return Ok(Vec::new());
    }
    let mut arrays = BTreeMap::new();
    for (key, max) in [
        ("prism.hadamard.weight_names", MAX_NAMES),
        ("prism.hadamard.inverse_weight_names", MAX_NAMES),
        ("prism.hadamard.sign_widths", MAX_NAMES),
        ("prism.hadamard.sign_values", MAX_SIGNS),
    ] {
        arrays.insert(key, array(doc, path, key, max)?);
    }
    resolve(doc, &arrays)
}

fn integer(value: &Scalar) -> Option<i64> {
    match value {
        Scalar::Signed(n) => Some(*n),
        Scalar::Unsigned(n) => i64::try_from(*n).ok(),
        _ => None,
    }
}

fn resolve(
    doc: &Document,
    arrays: &BTreeMap<&str, Vec<Scalar>>,
) -> Result<Vec<WeightInputTransform>, String> {
    let mut signs = BTreeMap::new();
    let values = &arrays["prism.hadamard.sign_values"];
    let mut offset = 0_usize;
    for raw in &arrays["prism.hadamard.sign_widths"] {
        let width = integer(raw)
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| *n > 0 && n % 1024 == 0)
            .ok_or("Prism sign widths must be positive multiples of 1024")?;
        let end = offset
            .checked_add(width as usize)
            .ok_or("Prism sign width overflow")?;
        let row = values
            .get(offset..end)
            .ok_or("Prism sign values are shorter than sign widths")?;
        let row: Vec<i8> = row
            .iter()
            .map(|v| match integer(v) {
                Some(-1) => Ok(-1),
                Some(1) => Ok(1),
                _ => Err("Prism signs must be integer -1 or +1"),
            })
            .collect::<Result<_, _>>()?;
        if signs.insert(width, row).is_some() {
            return Err(format!("duplicate Prism sign width {width}"));
        }
        offset = end;
    }
    if signs.is_empty() || offset != values.len() {
        return Err("empty Prism sign table or trailing sign values".into());
    }
    let grouped = matches!(
        doc.metadata.get("prism.hadamard.gdn_v_grouped"),
        Some(MetadataValue::Scalar(Scalar::Bool(true)))
    );
    let mut names = BTreeSet::new();
    let mut transforms = Vec::new();
    for (key, inverse) in [
        ("prism.hadamard.weight_names", false),
        ("prism.hadamard.inverse_weight_names", true),
    ] {
        if arrays[key].is_empty() {
            return Err(format!("{key} is empty"));
        }
        for raw in &arrays[key] {
            let Scalar::String(name) = raw else {
                return Err(format!("{key} must contain strings"));
            };
            if !names.insert(name.as_str()) {
                return Err(format!("duplicate Prism transformed weight {name}"));
            }
            if inverse && name != "token_embd.weight" {
                return Err(format!(
                    "inverse basis is supported only after token_embd lookup: {name}"
                ));
            }
            if !inverse && !forward_role(name) {
                return Err(format!(
                    "{name} is not a verified transformed qwen35 matmul"
                ));
            }
            let tensor = doc
                .tensor(name)
                .ok_or_else(|| format!("missing Prism weight {name}"))?;
            if tensor.dimensions.len() != 2 || tensor.ggml_type != PTQ1_0 {
                return Err(format!(
                    "Prism weight {name} must be a two-dimensional PTQ1_0 tensor"
                ));
            }
            let width = u32::try_from(tensor.dimensions[0])
                .map_err(|_| "Prism input width overflow")?;
            let sign = signs
                .get(&width)
                .ok_or_else(|| format!("no Prism signs for {name} width {width}"))?;
            let permutation = if grouped && name.ends_with(".ssm_out.weight") {
                let kh = doc
                    .unsigned_value("qwen35.ssm.group_count")
                    .and_then(|v| u32::try_from(v).ok())
                    .filter(|v| *v > 0)
                    .ok_or("invalid Prism key head count")?;
                let vh = doc
                    .unsigned_value("qwen35.ssm.time_step_rank")
                    .and_then(|v| u32::try_from(v).ok())
                    .filter(|v| *v > 0)
                    .ok_or("invalid Prism value head count")?;
                if vh % kh != 0 || width % vh != 0 {
                    return Err(format!(
                        "invalid Prism grouped-head geometry for {name}"
                    ));
                }
                // Existing DeltaNet retains tiled h % key_heads semantics. Only
                // ssm_out's input changes order: [head_dim,key_heads,repeat]
                // -> [head_dim,repeat,key_heads], then signs and normalized H.
                Some(GroupedHeads {
                    head_dim: width / vh,
                    key_heads: kh,
                    value_heads: vh,
                })
            } else {
                None
            };
            transforms.push(WeightInputTransform {
                weight_offset: tensor.absolute_offset,
                width,
                block_size: 1024,
                signs: sign.clone(),
                inverse,
                permutation,
            });
        }
    }
    for tensor in &doc.tensors {
        if tensor.ggml_type == PTQ1_0 && !names.contains(tensor.name.as_str()) {
            return Err(format!(
                "PTQ1_0 tensor {} has no declared activation basis",
                tensor.name
            ));
        }
    }
    Ok(transforms)
}

fn forward_role(name: &str) -> bool {
    if name == "output.weight" {
        return true;
    }
    let mut parts = name.split('.');
    if parts.next() != Some("blk")
        || parts.next().and_then(|n| n.parse::<u32>().ok()).is_none()
    {
        return false;
    }
    matches!(
        (parts.next(), parts.next(), parts.next()),
        (
            Some(
                "attn_qkv"
                    | "attn_gate"
                    | "ssm_out"
                    | "ffn_down"
                    | "ffn_gate"
                    | "ffn_up"
                    | "attn_q"
                    | "attn_k"
                    | "attn_v"
                    | "attn_output"
            ),
            Some("weight"),
            None
        )
    )
}

/// The scalar workflow has no registered transform seam yet. Refuse explicitly
/// even when a caller bypasses `backend::enable_gpu` and constructs the model.
pub fn ensure_workflow_basis(
    weights: &imparo_gguf::weights::Weights,
) -> Result<(), String> {
    if weights.gpu_enabled() {
        return Ok(());
    }
    let doc = imparo_gguf::read(weights.source_path()).map_err(|e| e.to_string())?;
    if validate_header(&doc)? {
        return Err("Prism rotated weights require a backend with registered activation transforms; CPU reference workflow is not yet supported".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use imparo_gguf::{ArraySummary, TensorInfo, ValueType};

    fn fixture() -> (Document, BTreeMap<&'static str, Vec<Scalar>>) {
        let mut doc = Document {
            version: 3,
            alignment: 32,
            data_offset: 4096,
            file_size: 16384,
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
        };
        for (key, val) in [
            ("general.architecture", Scalar::String("qwen35".into())),
            ("prism.hadamard.version", Scalar::Unsigned(1)),
            ("prism.hadamard.block_size", Scalar::Unsigned(1024)),
            (
                "prism.hadamard.transform",
                Scalar::String("normalized-sylvester-walsh-hadamard".into()),
            ),
            (
                "prism.hadamard.axis",
                Scalar::String("input-last-dimension".into()),
            ),
            (
                "prism.hadamard.sign_mode",
                Scalar::String("explicit".into()),
            ),
            ("prism.hadamard.gdn_v_grouped", Scalar::Bool(true)),
            ("qwen35.ssm.group_count", Scalar::Unsigned(2)),
            ("qwen35.ssm.time_step_rank", Scalar::Unsigned(8)),
        ] {
            doc.metadata.insert(key.into(), MetadataValue::Scalar(val));
        }
        for (i, name) in ["blk.0.ssm_out.weight", "token_embd.weight"]
            .iter()
            .enumerate()
        {
            doc.tensors.push(TensorInfo {
                name: (*name).into(),
                dimensions: vec![1024, 8],
                ggml_type: PTQ1_0,
                relative_offset: i as u64 * 2048,
                absolute_offset: 4096 + i as u64 * 2048,
                byte_size: 1792,
                type_field_offset: 0,
            });
        }
        let arrays = BTreeMap::from([
            (
                "prism.hadamard.weight_names",
                vec![Scalar::String("blk.0.ssm_out.weight".into())],
            ),
            (
                "prism.hadamard.inverse_weight_names",
                vec![Scalar::String("token_embd.weight".into())],
            ),
            ("prism.hadamard.sign_widths", vec![Scalar::Signed(1024)]),
            ("prism.hadamard.sign_values", vec![Scalar::Signed(-1); 1024]),
        ]);
        for (key, values) in &arrays {
            doc.metadata.insert(
                (*key).into(),
                MetadataValue::Array(ArraySummary {
                    element_type: if key.contains("names") {
                        ValueType::String
                    } else {
                        ValueType::Int32
                    },
                    element_count: values.len() as u64,
                    preview: values.iter().take(8).cloned().collect(),
                }),
            );
        }
        (doc, arrays)
    }

    #[test]
    fn basis_preserves_delta_contract_and_marks_inverse_embedding() {
        let (doc, arrays) = fixture();
        assert!(validate_header(&doc).unwrap());
        let tx = resolve(&doc, &arrays).unwrap();
        assert_eq!(tx.len(), 2);
        assert_eq!(tx[0].weight_offset, 4096);
        let perm = tx[0].permutation.as_ref().unwrap();
        assert_eq!(
            (perm.head_dim, perm.key_heads, perm.value_heads),
            (128, 2, 8)
        );
        assert!(!tx[0].inverse);
        assert!(tx[1].inverse);
        assert!(tx[1].permutation.is_none());
        assert_eq!(tx[1].signs, vec![-1; 1024]);
    }

    #[test]
    fn basis_rejects_missing_coverage_bad_signs_and_duplicate_names() {
        let (doc, mut arrays) = fixture();
        arrays.get_mut("prism.hadamard.sign_values").unwrap()[900] = Scalar::Signed(0);
        assert!(resolve(&doc, &arrays).unwrap_err().contains("signs"));
        let (doc, mut arrays) = fixture();
        arrays
            .get_mut("prism.hadamard.weight_names")
            .unwrap()
            .push(Scalar::String("blk.0.ssm_out.weight".into()));
        assert!(resolve(&doc, &arrays).unwrap_err().contains("duplicate"));
        let (mut doc, arrays) = fixture();
        let mut uncovered = doc.tensors[0].clone();
        uncovered.name = "blk.0.ffn_down.weight".into();
        doc.tensors.push(uncovered);
        assert!(resolve(&doc, &arrays).unwrap_err().contains("no declared"));
    }

    #[test]
    fn basis_rejects_missing_header_and_unknown_conventions() {
        let (mut doc, _) = fixture();
        doc.metadata.clear();
        assert!(
            validate_header(&doc)
                .unwrap_err()
                .contains("require explicit")
        );
        let (mut doc, _) = fixture();
        doc.metadata.insert(
            "prism.hadamard.sign_mode".into(),
            MetadataValue::Scalar(Scalar::String("identity".into())),
        );
        assert!(validate_header(&doc).unwrap_err().contains("sign_mode"));
    }

    #[test]
    #[ignore = "requires IMPARO_BONSAI_MODEL pointing at the real GGUF; metadata only"]
    fn real_model_basis_and_plan_admission() {
        let path = std::path::PathBuf::from(
            std::env::var_os("IMPARO_BONSAI_MODEL").expect("set IMPARO_BONSAI_MODEL"),
        );
        let document = imparo_gguf::read(&path).unwrap();
        let plan = crate::build_plan(&document, &path).unwrap();
        let transforms = from_document(&document, &path).unwrap();
        assert_eq!(plan.config.architecture, "qwen35");
        assert_eq!(plan.config.n_layers, 64);
        assert_eq!(transforms.len(), 402);
        assert_eq!(transforms.iter().filter(|t| t.inverse).count(), 1);
        assert_eq!(
            transforms
                .iter()
                .filter(|t| t.permutation.is_some())
                .count(),
            48
        );
        assert_eq!(plan.weight_residency.row_gathered, &["token_embd.weight"]);
        let weights =
            imparo_gguf::weights::Weights::open_with(&document, &path).unwrap();
        assert!(
            ensure_workflow_basis(&weights)
                .unwrap_err()
                .contains("CPU reference")
        );
        println!(
            "Bonsai metadata admitted: {} transforms, 48 grouped outputs, 1 inverse embedding; CPU reference refused",
            transforms.len()
        );
    }
}
