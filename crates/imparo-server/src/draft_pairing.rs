//! Drafter admission for the ordinary server. Model providers own identity and lifecycle checks.
//!
//! A DSpark drafter is named by its file (`--draft`): the server opens it at start and maps it
//! after the target's -- no manifest and no offline pairing step. The Gemma4 MTP drafter (CUDA)
//! still comes through its pairing manifest (`--draft-pairing`).
use imparo_model::speculative::DraftSpec;
use std::path::{Path, PathBuf};

/// The drafter for `target`, whose header `document` is: its file's path (the caller maps it
/// after the target's, `Weights::open_with_appended`) and its spec. `draft` is a DSpark drafter
/// file, `mtp_pairing` a Gemma4 MTP manifest; at most one of them.
pub(super) fn load(
    target: &Path,
    document: &imparo_gguf::Document,
    architecture: &str,
    draft: Option<&Path>,
    mtp_pairing: Option<&Path>,
    requested_mask: Option<u32>,
) -> Result<Option<(PathBuf, DraftSpec)>, String> {
    match select(draft, mtp_pairing, requested_mask)? {
        Source::None => Ok(None),
        Source::Mtp(path) => {
            if architecture != "gemma4" {
                return Err("gemma4-mtp requires a Gemma4 target".into());
            }
            load_mtp(path, target)
        }
        Source::Dspark(path) => {
            // The drafter reads the target's layer outputs and borrows its embedding
            // table and head; what it needs from the target is those, not a particular
            // mixer -- nor a particular feed-forward, which is why a routed target pairs
            // with the same drafter shape a dense one does.
            if !matches!(architecture, "lfm2" | "lfm2moe" | "qwen3") {
                return Err(format!("DSpark has no pairing for a {architecture} target"));
            }
            let (draft, pairing) = imparo_model::dspark::Pairing::open(path, requested_mask)?;
            let mask = pairing.mask_token();
            let vocab = document
                .tensor("token_embd.weight")
                .and_then(|t| t.dimensions.get(1))
                .copied()
                .ok_or("target embedding vocabulary missing")?;
            if u64::from(mask) >= vocab {
                return Err("DSpark mask token is outside target vocabulary".into());
            }
            eprintln!(
                "[imparo] drafter admitted kind=dspark mask_token={mask} file={}",
                draft.display()
            );
            Ok(Some((draft, DraftSpec::Dspark(pairing))))
        }
    }
}

// The Gemma4 MTP drafter exists only as CUDA operators.
#[cfg(feature = "cuda-speculative")]
fn load_mtp(
    path: &Path,
    target: &Path,
) -> Result<Option<(PathBuf, DraftSpec)>, String> {
    let (draft, pairing) = imparo_model::gemma4_mtp::Pairing::load(path, target)?;
    eprintln!("[imparo] draft pairing admitted kind=gemma4-mtp");
    Ok(Some((draft, DraftSpec::GemmaMtp(pairing))))
}
#[cfg(not(feature = "cuda-speculative"))]
fn load_mtp(
    _path: &Path,
    _target: &Path,
) -> Result<Option<(PathBuf, DraftSpec)>, String> {
    Err("gemma4-mtp requires the cuda-speculative build feature".into())
}

#[cfg(feature = "cuda-speculative")]
pub(super) fn apply_knobs() -> Result<(), String> {
    imparo_model::backend::apply_lab_knobs_from_env()
}
pub(super) fn request_enabled(body: &serde_json::Value) -> bool {
    body.get("imparo_draft")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true)
}

/// Which drafter the options name.
#[derive(Debug, PartialEq)]
enum Source<'a> {
    None,
    Dspark(&'a Path),
    Mtp(&'a Path),
}

// Pure option resolution, separated from model IO so admission errors are testable.
fn select<'a>(
    draft: Option<&'a Path>,
    mtp_pairing: Option<&'a Path>,
    mask: Option<u32>,
) -> Result<Source<'a>, String> {
    match (draft, mtp_pairing) {
        (Some(_), Some(_)) => Err("--draft and --draft-pairing name two drafters".into()),
        (Some(d), None) => Ok(Source::Dspark(d)),
        (None, Some(m)) => {
            if mask.is_some() {
                return Err("--draft-mask-token only applies to a DSpark --draft".into());
            }
            Ok(Source::Mtp(m))
        }
        (None, None) => {
            if mask.is_some() {
                return Err("--draft-mask-token requires --draft".into());
            }
            Ok(Source::None)
        }
    }
}

#[cfg(test)]
mod option_tests {
    use super::{Source, select};
    use std::path::Path;

    #[test]
    fn no_drafter_keeps_ordinary_path_and_orphan_options_fail() {
        assert_eq!(select(None, None, None).unwrap(), Source::None);
        assert!(select(None, None, Some(7)).is_err());
    }

    #[test]
    fn a_drafter_file_is_dspark_and_a_manifest_is_mtp() {
        let d = Path::new("models/drafter.gguf");
        let m = Path::new("models/mtp.json");
        assert_eq!(select(Some(d), None, Some(7)).unwrap(), Source::Dspark(d));
        assert_eq!(select(None, Some(m), None).unwrap(), Source::Mtp(m));
        assert!(select(None, Some(m), Some(7)).is_err());
        assert!(select(Some(d), Some(m), None).is_err());
    }
}
