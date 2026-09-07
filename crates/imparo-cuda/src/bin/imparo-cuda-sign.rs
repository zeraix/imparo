//! Release-only manifest/signature generator for per-SM CUDA plugins.

#[path = "../../build_support.rs"]
mod build_support;

use ed25519_dalek::{Signer as _, SigningKey};
use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn decode_key(value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("IMPARO_CUDA_RELEASE_PRIVATE_KEY must be a 32-byte hex seed".into());
    }
    let mut out = [0_u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[i * 2..i * 2 + 2], 16)
            .map_err(|e| e.to_string())?;
    }
    Ok(out)
}

fn sm_from_filename(path: &Path) -> Result<u32, String> {
    let name = path
        .file_name()
        .and_then(|v| v.to_str())
        .ok_or("plugin has no UTF-8 filename")?;
    let start = name
        .find("-sm")
        .ok_or_else(|| format!("plugin filename has no -sm: {name}"))?
        + 3;
    let digits: String = name[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits
        .parse()
        .map_err(|_| format!("plugin filename has invalid SM: {name}"))
}

fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let first = args
        .next()
        .and_then(|v| v.into_string().ok())
        .ok_or("usage: imparo-cuda-sign build-id TARGET ARCHS fast|precise [NVCC] | PLATFORM BASE_URL OUTPUT DLL...")?;
    if first == "build-id" {
        let target = args
            .next()
            .and_then(|v| v.into_string().ok())
            .ok_or("build-id requires TARGET")?;
        let archs = args
            .next()
            .and_then(|v| v.into_string().ok())
            .ok_or("build-id requires ARCHS")?;
        let math = args
            .next()
            .and_then(|v| v.into_string().ok())
            .ok_or("build-id requires fast or precise")?;
        if !matches!(math.as_str(), "fast" | "precise") {
            return Err("build-id math mode must be fast or precise".into());
        }
        let nvcc = args
            .next()
            .and_then(|v| v.into_string().ok())
            .unwrap_or_else(|| "nvcc".into());
        let compiler = build_support::compiler_identity(&nvcc)?;
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        println!(
            "{}",
            build_support::native_build_sha256(
                root,
                imparo_cuda::CUDA_BACKEND_ABI,
                &target,
                &archs,
                &math,
                &compiler
            )
        );
        return Ok(());
    }
    let platform = first;
    let base_url = args
        .next()
        .and_then(|v| v.into_string().ok())
        .ok_or("missing BASE_URL")?;
    let output = args.next().map(PathBuf::from).ok_or("missing OUTPUT")?;
    if !base_url.starts_with("https://") {
        return Err("BASE_URL must use HTTPS".into());
    }
    let paths: Vec<PathBuf> = args.map(PathBuf::from).collect();
    if paths.is_empty() {
        return Err("no CUDA plugin DLLs supplied".into());
    }

    let private = std::env::var("IMPARO_CUDA_RELEASE_PRIVATE_KEY")
        .map_err(|_| "IMPARO_CUDA_RELEASE_PRIVATE_KEY is required")?;
    let key = SigningKey::from_bytes(&decode_key(&private)?);
    let public = hex(key.verifying_key().as_bytes());
    if let Ok(expected) = std::env::var("IMPARO_CUDA_RELEASE_PUBLIC_KEY") {
        if !expected.eq_ignore_ascii_case(&public) {
            return Err("release public key does not match private signing key".into());
        }
    }

    let mut backends = Vec::new();
    for path in paths {
        let file_name = path
            .file_name()
            .and_then(|v| v.to_str())
            .ok_or("plugin has no filename")?
            .to_string();
        let sm = sm_from_filename(&path)?;
        let bytes = fs::metadata(&path)
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len();
        let mut file = fs::File::open(&path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut digest = Sha256::new();
        let mut buf = vec![0_u8; 1024 * 1024];
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|e| format!("hash {}: {e}", path.display()))?;
            if n == 0 {
                break;
            }
            digest.update(&buf[..n]);
        }
        let sha256 = hex(&digest.finalize());
        let message = imparo_cuda::cuda_backend_signed_message(
            &platform,
            sm,
            imparo_cuda::CUDA_BACKEND_ABI,
            bytes,
            &sha256,
        );
        let signature = hex(&key.sign(&message).to_bytes());
        backends.push(serde_json::json!({
            "platform": platform, "sm": sm, "abi": imparo_cuda::CUDA_BACKEND_ABI, "file": file_name,
            "bytes": bytes, "sha256": sha256, "signature": signature,
            "url": format!("{}/{}", base_url.trim_end_matches('/'), file_name),
        }));
    }
    backends.sort_by_key(|v| v["sm"].as_u64().unwrap_or_default());
    let document = serde_json::json!({ "schema": 1, "backends": backends });
    fs::write(&output, serde_json::to_string_pretty(&document).unwrap())
        .map_err(|e| format!("write {}: {e}", output.display()))?;
    eprintln!("CUDA release public key: {public}");
    Ok(())
}

fn main() -> Result<(), String> {
    std::thread::Builder::new()
        .name("imparo-cuda-sign".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .map_err(|e| e.to_string())?
        .join()
        .map_err(|_| "CUDA signing worker panicked".to_string())?
}
