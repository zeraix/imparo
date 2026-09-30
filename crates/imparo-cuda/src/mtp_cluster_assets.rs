//! Fixed cluster assets for the retained E4B assistant policy.
//! Native retains only these immutable static bytes; upload remains owner-local.

use imparo_program_pack::identity::{encode_hex, sha256};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

const CENTROIDS_BYTES: usize = 2_097_152;
const ORDERING_BYTES: usize = 1_048_576;
const CENTROIDS_SHA256: &str =
    "d293fc2fc2b68dea9716cc6cad81c4847084640d393962c637ef593415aa68c7";
const ORDERING_SHA256: &str =
    "2d4a619b6fdf687972daaf298bf5ce341f4e2f1d05c20de10c77684e20481b07";
static CENTROIDS: &[u8] = include_bytes!("../native/gemma4_mtp_assets/centroids.f32");
static ORDERING: &[u8] = include_bytes!("../native/gemma4_mtp_assets/ordering.u32");
static VERIFIED: OnceLock<Result<(), String>> = OnceLock::new();
static LOGGED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    fn imparo_cuda_gemma4_mtp_cluster_assets(
        centroids: *const u8,
        centroids_bytes: u64,
        ordering: *const u8,
        ordering_bytes: u64,
    ) -> i32;
}

fn validate(
    name: &str,
    bytes: &[u8],
    size: usize,
    expected: &str,
) -> Result<(), String> {
    if bytes.len() != size {
        return Err(format!("MTP cluster {name} size mismatch"));
    }
    if encode_hex(&sha256(bytes)) != expected {
        return Err(format!("MTP cluster {name} sha256 mismatch"));
    }
    Ok(())
}

pub(crate) fn prepare() -> Result<(), String> {
    VERIFIED
        .get_or_init(|| {
            validate("centroids", CENTROIDS, CENTROIDS_BYTES, CENTROIDS_SHA256)?;
            validate("ordering", ORDERING, ORDERING_BYTES, ORDERING_SHA256)
        })
        .as_ref()
        .map_err(|error| error.clone())?;
    // SAFETY: both authenticated slices have static lifetime. Native freezes their
    // addresses at a closed boundary and copies bytes into aligned host vectors
    // before the existing private-owner upload; it never reopens a filesystem path.
    let rc = unsafe {
        imparo_cuda_gemma4_mtp_cluster_assets(
            CENTROIDS.as_ptr(),
            CENTROIDS.len() as u64,
            ORDERING.as_ptr(),
            ORDERING.len() as u64,
        )
    };
    if rc != 0 {
        return Err(format!(
            "MTP cluster native asset registration failed rc={rc}"
        ));
    }
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[mtp-cluster-assets] centroids_sha256={CENTROIDS_SHA256} centroids_bytes={CENTROIDS_BYTES} ordering_sha256={ORDERING_SHA256} ordering_bytes={ORDERING_BYTES} source=embedded"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_assets_match_fixed_identity() {
        validate("centroids", CENTROIDS, CENTROIDS_BYTES, CENTROIDS_SHA256).unwrap();
        validate("ordering", ORDERING, ORDERING_BYTES, ORDERING_SHA256).unwrap();
    }

    #[test]
    fn altered_asset_is_rejected() {
        let mut bytes = ORDERING.to_vec();
        bytes[0] ^= 1;
        assert!(
            validate("ordering", &bytes, ORDERING_BYTES, ORDERING_SHA256)
                .unwrap_err()
                .contains("sha256 mismatch")
        );
    }
}
