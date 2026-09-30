//! Frozen W4A16 image selection and exact-byte native loading.
//!
//! The caller selects the target owner and invokes this only for an enabled
//! W4A16 Decode/verification phase, before Graph capture or weight conversion.

use crate::ffi::{
    imparo_cuda_e4b_ffn_w4a16_module_identity, imparo_cuda_load_e4b_ffn_w4a16,
};
use imparo_program_pack::identity::{encode_hex, sha256};
use std::borrow::Cow;
use std::io::Read as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

const EXPECTED_BYTES: usize = 4_348_832;
const EXPECTED_SHA256: &str =
    "9521406b46b918c23450e6199ed973149c88fcb2e33e5a7d8348212f7694cf37";
const INVALID: i32 = 3;
static EMBEDDED: &[u8] =
    include_bytes!("../native/sm86/w4a16_marlin/marlin-sm80.cubin");

struct VerifiedImage {
    bytes: Cow<'static, [u8]>,
    digest: [u8; 32],
    source: &'static str,
}

// Freeze only immutable source bytes, never native owner/module readiness.
// A failed first choice stays failed instead of silently changing the source.
static IMAGE: OnceLock<Result<VerifiedImage, String>> = OnceLock::new();
static LOGGED: AtomicBool = AtomicBool::new(false);

fn validate_image(bytes: &[u8]) -> Result<[u8; 32], String> {
    if bytes.len() != EXPECTED_BYTES {
        return Err(format!(
            "W4A16 cubin size mismatch: expected {EXPECTED_BYTES}, got {}",
            bytes.len()
        ));
    }
    let digest = sha256(bytes);
    let actual = encode_hex(&digest);
    if actual != EXPECTED_SHA256 {
        return Err(format!(
            "W4A16 cubin sha256 mismatch: expected {EXPECTED_SHA256}, got {actual}"
        ));
    }
    Ok(digest)
}

fn select_image() -> Result<VerifiedImage, String> {
    let (bytes, source): (Cow<'static, [u8]>, &'static str) =
        match std::env::var_os("IMPARO_LAB_E4B_W4A16_CUBIN") {
            Some(path) => {
                let file = std::fs::File::open(&path).map_err(|error| {
                    format!("cannot open IMPARO_LAB_E4B_W4A16_CUBIN: {error}")
                })?;
                // Read from one handle, with one extra byte to reject oversize
                // inputs without trusting metadata or allocating unboundedly.
                let mut bytes = Vec::with_capacity(EXPECTED_BYTES + 1);
                file.take((EXPECTED_BYTES + 1) as u64)
                    .read_to_end(&mut bytes)
                    .map_err(|error| {
                        format!("cannot read IMPARO_LAB_E4B_W4A16_CUBIN: {error}")
                    })?;
                (Cow::Owned(bytes), "lab")
            }
            None => (Cow::Borrowed(EMBEDDED), "embedded"),
        };
    let digest = validate_image(bytes.as_ref())?;
    Ok(VerifiedImage {
        bytes,
        digest,
        source,
    })
}

pub(crate) fn prepare() -> Result<(), i32> {
    let image = IMAGE.get_or_init(select_image).as_ref().map_err(|error| {
        eprintln!("[w4a16-module] {error}");
        INVALID
    })?;
    // SAFETY: both pointers refer to immutable owned/static bytes retained in
    // IMAGE for the process lifetime. Native synchronously loads this exact
    // image and must not reopen a path; no native owner success is cached here.
    let rc = unsafe {
        imparo_cuda_load_e4b_ffn_w4a16(
            image.bytes.as_ptr(),
            image.bytes.len() as u64,
            image.digest.as_ptr(),
        )
    };
    if rc != 0 {
        eprintln!("[w4a16-module] native image load rejected rc={rc}");
        return Err(rc);
    }
    let mut actual = [0_u8; 32];
    // SAFETY: actual is writable for exactly the supplied 32-byte capacity.
    let rc =
        unsafe { imparo_cuda_e4b_ffn_w4a16_module_identity(actual.as_mut_ptr(), 32) };
    if rc != 0 {
        eprintln!("[w4a16-module] native loaded identity unavailable rc={rc}");
        return Err(rc);
    }
    if actual != image.digest {
        eprintln!(
            "[w4a16-module] loaded sha256 mismatch: expected {}, got {}",
            encode_hex(&image.digest),
            encode_hex(&actual)
        );
        return Err(INVALID);
    }
    if !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[w4a16-module] sha256={} bytes={} source={}",
            encode_hex(&actual),
            image.bytes.len(),
            image.source
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_image_has_frozen_length_and_digest() {
        assert_eq!(
            encode_hex(&validate_image(EMBEDDED).unwrap()),
            EXPECTED_SHA256
        );
    }

    #[test]
    fn same_length_corruption_is_rejected() {
        let mut damaged = EMBEDDED.to_vec();
        damaged[EXPECTED_BYTES / 2] ^= 1;
        assert_eq!(damaged.len(), EXPECTED_BYTES);
        assert!(
            validate_image(&damaged)
                .unwrap_err()
                .contains("sha256 mismatch")
        );
    }

    #[test]
    fn truncated_image_is_rejected() {
        assert!(
            validate_image(&EMBEDDED[..EMBEDDED.len() - 1])
                .unwrap_err()
                .contains("size mismatch")
        );
    }
}
