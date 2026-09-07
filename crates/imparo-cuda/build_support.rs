//! Shared semantic build identity for CUDA native artifacts.
//!
//! This file is compiled by both `build.rs` (static source builds) and the release
//! signing tool (per-SM plugins). Keep it independent of the Imparo crate graph.

use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn collect_native(directory: &Path, root: &Path, files: &mut Vec<PathBuf>) {
    for entry in
        std::fs::read_dir(directory).expect("read CUDA native source directory")
    {
        let entry = entry.expect("read CUDA native source entry");
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "tests") {
                continue;
            }
            collect_native(&path, root, files);
        } else if path.extension().is_some_and(|extension| {
            matches!(extension.to_str(), Some("cu" | "cuh" | "h" | "inc" | "def"))
        }) {
            files.push(
                path.strip_prefix(root)
                    .expect("native source under root")
                    .into(),
            );
        }
    }
}

fn field(hash: &mut Sha256, name: &str, value: &[u8]) {
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name.as_bytes());
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}

/// SHA-256 of the complete native source/build contract, not a host executable.
///
/// Target, SM list, math mode and nvcc identity are inputs because they can change the
/// emitted backend. Paths are repository-relative and sorted, so checkout location and
/// directory enumeration order cannot change the result.
pub fn native_build_sha256(
    manifest_dir: &Path,
    backend_abi: u32,
    target: &str,
    archs: &str,
    math_mode: &str,
    compiler_identity: &str,
) -> String {
    let mut files = Vec::new();
    collect_native(&manifest_dir.join("native"), manifest_dir, &mut files);
    files.extend([PathBuf::from("build.rs"), PathBuf::from("cuda-sm.json")]);
    files.sort();

    let mut hash = Sha256::new();
    field(&mut hash, "domain", b"imparo-cuda-native-build-v1");
    field(&mut hash, "backend_abi", &backend_abi.to_le_bytes());
    field(&mut hash, "target", target.as_bytes());
    field(&mut hash, "archs", archs.as_bytes());
    field(&mut hash, "math_mode", math_mode.as_bytes());
    field(&mut hash, "compiler", compiler_identity.as_bytes());
    for relative in files {
        let canonical = relative.to_string_lossy().replace('\\', "/");
        field(&mut hash, "path", canonical.as_bytes());
        field(
            &mut hash,
            "content",
            &std::fs::read(manifest_dir.join(&relative))
                .expect("read CUDA build input"),
        );
    }
    let mut output = String::with_capacity(64);
    for byte in hash.finalize() {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Stable nvcc identity used by both Cargo and release workflows.
pub fn compiler_identity(nvcc: &str) -> Result<String, String> {
    let output = std::process::Command::new(nvcc)
        .arg("--version")
        .output()
        .map_err(|error| format!("run {nvcc} --version: {error}"))?;
    if !output.status.success() {
        return Err(format!("{nvcc} --version failed with {}", output.status));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|_| format!("{nvcc} --version produced non-UTF-8 output"))?;
    Ok(stdout.replace("\r\n", "\n").trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_a_lowercase_sha256_and_build_inputs_matter() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let base =
            native_build_sha256(root, 21, "windows-x86_64", "86", "fast", "nvcc-x");
        assert_eq!(
            base,
            native_build_sha256(root, 21, "windows-x86_64", "86", "fast", "nvcc-x")
        );
        assert_eq!(base.len(), 64);
        assert!(
            base.bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
        assert_ne!(
            base,
            native_build_sha256(root, 21, "windows-x86_64", "89", "fast", "nvcc-x")
        );
        assert_ne!(
            base,
            native_build_sha256(root, 21, "windows-x86_64", "86", "precise", "nvcc-x")
        );
    }
}
