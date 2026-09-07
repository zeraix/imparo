//! Resolves, authenticates and caches the one backend matching the active NVIDIA SM.

use ed25519_dalek::{Signature, VerifyingKey};
use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use crate::dylib::Library;

const MANIFEST: &str =
    include_str!(concat!(env!("OUT_DIR"), "/imparo_cuda_manifest.json"));
const PUBLIC_KEY_HEX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/imparo_cuda_public_key.txt"));

#[derive(Clone, Debug, Eq, PartialEq)]
struct Entry {
    platform: String,
    sm: u32,
    abi: u32,
    file: String,
    bytes: u64,
    sha256: String,
    signature: String,
    url: String,
}

pub(crate) fn resolve_backend() -> Result<PathBuf, String> {
    // Explicit local path is a developer override and never downloads code.
    if let Some(path) = std::env::var_os("IMPARO_CUDA_BACKEND") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path.clone()).ok_or_else(|| {
            format!("IMPARO_CUDA_BACKEND does not exist: {}", path.display())
        });
    }

    let release_key = decode_hex::<32>(PUBLIC_KEY_HEX.trim())?;
    if release_key == [0; 32] {
        return Err("this development runtime has no release public key; automatic CUDA backend download is disabled".into());
    }

    let sm = detect_sm()?;
    let entries = parse_manifest(MANIFEST)?;
    let entry = select_entry(&entries, sm, platform()).ok_or_else(|| {
        format!(
            "this Imparo release has no CUDA backend for {}/sm{sm}",
            platform()
        )
    })?;
    let destination = cache_root()
        .join(format!("abi{}", entry.abi))
        .join(format!("sm{}", entry.sm))
        .join(&entry.file);
    if destination.is_file() && authenticate(&destination, entry)? {
        return Ok(destination);
    }
    download(entry, &destination)?;
    Ok(destination)
}

fn platform() -> &'static str {
    #[cfg(all(windows, target_arch = "x86_64"))]
    {
        "windows-x86_64"
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "linux-x86_64"
    }
    #[cfg(not(any(
        all(windows, target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "x86_64")
    )))]
    {
        "unsupported"
    }
}

fn cache_root() -> PathBuf {
    if let Some(path) = std::env::var_os("IMPARO_BACKEND_CACHE") {
        return PathBuf::from(path).join("cuda");
    }
    #[cfg(windows)]
    if let Some(path) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(path)
            .join("Imparo")
            .join("backends")
            .join("cuda");
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(path)
            .join("imparo")
            .join("backends")
            .join("cuda");
    }
    std::env::var_os("HOME")
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
        .join(".cache")
        .join("imparo")
        .join("backends")
        .join("cuda")
}

fn parse_manifest(text: &str) -> Result<Vec<Entry>, String> {
    let root: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| format!("parse embedded CUDA backend manifest: {e}"))?;
    if root.get("schema").and_then(serde_json::Value::as_u64) != Some(1) {
        return Err("unsupported embedded CUDA backend manifest schema".into());
    }
    let values = root
        .get("backends")
        .and_then(serde_json::Value::as_array)
        .ok_or("embedded CUDA manifest has no backends array")?;
    values
        .iter()
        .map(|v| {
            let get_s = |name| {
                v.get(name)
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| format!("CUDA backend entry missing {name}"))
            };
            let file = get_s("file")?;
            if Path::new(&file).file_name().and_then(|s| s.to_str()) != Some(&file) {
                return Err(format!("CUDA backend file must be a basename: {file}"));
            }
            let sha256 = get_s("sha256")?.to_ascii_lowercase();
            let signature = get_s("signature")?.to_ascii_lowercase();
            if !valid_hex(&sha256, 64) {
                return Err(format!("invalid SHA-256 for {file}"));
            }
            if !valid_hex(&signature, 128) {
                return Err(format!("invalid Ed25519 signature for {file}"));
            }
            Ok(Entry {
                platform: get_s("platform")?,
                sm: u32::try_from(
                    v.get("sm")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or("CUDA backend entry missing sm")?,
                )
                .map_err(|_| "CUDA backend sm is too large")?,
                abi: u32::try_from(
                    v.get("abi")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or("CUDA backend entry missing abi")?,
                )
                .map_err(|_| "CUDA backend abi is too large")?,
                file,
                bytes: v
                    .get("bytes")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or("CUDA backend entry missing bytes")?,
                sha256,
                signature,
                url: get_s("url")?,
            })
        })
        .collect()
}

fn valid_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

/// Hash the selected backend artifact itself. This is deliberately the plugin file,
/// never the common server/tuner executable which happens to load it.
pub(crate) fn backend_artifact_sha256(path: &Path) -> Result<[u8; 32], String> {
    let mut file = fs::File::open(path)
        .map_err(|error| format!("open CUDA backend {}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            format!("hash CUDA backend {}: {error}", path.display())
        })?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hash.finalize().into())
}

fn select_entry<'a>(
    entries: &'a [Entry],
    device_sm: u32,
    wanted_platform: &str,
) -> Option<&'a Entry> {
    entries
        .iter()
        .filter(|e| e.platform == wanted_platform && e.abi == crate::CUDA_BACKEND_ABI)
        .filter(|e| e.sm / 10 == device_sm / 10 && e.sm <= device_sm)
        .max_by_key(|e| e.sm)
}

fn signed_message(entry: &Entry) -> Vec<u8> {
    crate::cuda_backend_signed_message(
        &entry.platform,
        entry.sm,
        entry.abi,
        entry.bytes,
        &entry.sha256,
    )
}

fn decode_hex<const N: usize>(value: &str) -> Result<[u8; N], String> {
    if !valid_hex(value, N * 2) {
        return Err(format!("expected {} hex characters", N * 2));
    }
    let mut out = [0_u8; N];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[i * 2..i * 2 + 2], 16)
            .map_err(|e| e.to_string())?;
    }
    Ok(out)
}

fn authenticate(path: &Path, entry: &Entry) -> Result<bool, String> {
    authenticate_with_key(path, entry, PUBLIC_KEY_HEX.trim())
}

fn authenticate_with_key(
    path: &Path,
    entry: &Entry,
    public_key_hex: &str,
) -> Result<bool, String> {
    let meta =
        fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
    if meta.len() != entry.bytes {
        return Ok(false);
    }
    let mut file =
        fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buf = vec![0_u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("hash {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    if encode_hex(&hash.finalize()) != entry.sha256 {
        return Ok(false);
    }
    let key_bytes = decode_hex::<32>(public_key_hex)?;
    if key_bytes == [0; 32] {
        return Err("this development runtime has no release public key; automatic CUDA backend download is disabled".into());
    }
    let key = VerifyingKey::from_bytes(&key_bytes)
        .map_err(|e| format!("invalid embedded CUDA release public key: {e}"))?;
    let signature = Signature::from_bytes(&decode_hex::<64>(&entry.signature)?);
    Ok(key
        .verify_strict(&signed_message(entry), &signature)
        .is_ok())
}

fn download(entry: &Entry, destination: &Path) -> Result<(), String> {
    if !entry.url.starts_with("https://") {
        return Err(format!(
            "refusing non-HTTPS CUDA backend URL: {}",
            entry.url
        ));
    }
    let parent = destination
        .parent()
        .ok_or("CUDA backend destination has no parent")?;
    fs::create_dir_all(parent)
        .map_err(|e| format!("create CUDA cache {}: {e}", parent.display()))?;
    let temporary = parent.join(format!(".{}.{}.part", entry.file, std::process::id()));
    let max_bytes = entry.bytes.to_string();
    let status = std::process::Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--connect-timeout",
            "15",
            "--retry",
            "2",
            "--max-filesize",
            &max_bytes,
            "--output",
        ])
        .arg(&temporary)
        .arg(&entry.url)
        .status()
        .map_err(|e| format!("start curl for CUDA backend: {e}"))?;
    if !status.success() {
        let _ = fs::remove_file(&temporary);
        return Err(format!(
            "download CUDA backend failed with {status}: {}",
            entry.url
        ));
    }
    match authenticate(&temporary, entry) {
        Ok(true) => {}
        Ok(false) => {
            let _ = fs::remove_file(&temporary);
            return Err(format!(
                "CUDA backend failed hash/signature verification: {}",
                entry.file
            ));
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
    }
    // Two processes may request the same SM concurrently. Keep a valid winner and
    // treat it as success; never replace authenticated code unnecessarily.
    if destination.is_file() && authenticate(destination, entry)? {
        let _ = fs::remove_file(&temporary);
        return Ok(());
    }
    if destination.exists() {
        fs::remove_file(destination).map_err(|e| {
            format!("remove invalid CUDA backend {}: {e}", destination.display())
        })?;
    }
    if let Err(error) = fs::rename(&temporary, destination) {
        // A competing process can win between the checks above and rename. Accept
        // only an authenticated winner; otherwise preserve the installation error.
        if destination.is_file() && authenticate(destination, entry)? {
            let _ = fs::remove_file(&temporary);
            return Ok(());
        }
        let _ = fs::remove_file(&temporary);
        return Err(format!(
            "install CUDA backend {}: {error}",
            destination.display()
        ));
    }
    eprintln!(
        "[imparo] downloaded and verified CUDA sm{} backend",
        entry.sm
    );
    Ok(())
}

fn detect_sm() -> Result<u32, String> {
    if let Ok(value) = std::env::var("IMPARO_CUDA_SM") {
        return value
            .parse::<u32>()
            .map_err(|_| format!("invalid IMPARO_CUDA_SM={value}"));
    }
    #[cfg(windows)]
    let driver = Library::open_system32("nvcuda.dll")?;
    #[cfg(not(windows))]
    let driver = Library::open(Path::new("libcuda.so.1"))?;
    unsafe {
        type Init = unsafe extern "C" fn(u32) -> i32;
        type Get = unsafe extern "C" fn(*mut i32, i32) -> i32;
        type Attr = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
        let init: Init = std::mem::transmute(driver.symbol(b"cuInit\0")?);
        let get: Get = std::mem::transmute(driver.symbol(b"cuDeviceGet\0")?);
        let attr: Attr = std::mem::transmute(driver.symbol(b"cuDeviceGetAttribute\0")?);
        if init(0) != 0 {
            return Err("cuInit failed while detecting CUDA SM".into());
        }
        let ordinal = std::env::var("IMPARO_CUDA_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mut device = 0;
        if get(&raw mut device, ordinal) != 0 {
            return Err(format!("cuDeviceGet({ordinal}) failed"));
        }
        let (mut major, mut minor) = (0, 0);
        if attr(&raw mut major, 75, device) != 0
            || attr(&raw mut minor, 76, device) != 0
        {
            return Err("cuDeviceGetAttribute(compute capability) failed".into());
        }
        Ok((major as u32) * 10 + minor as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer as _, SigningKey};
    fn entry(sm: u32) -> Entry {
        Entry {
            platform: "windows-x86_64".into(),
            sm,
            abi: crate::CUDA_BACKEND_ABI,
            file: format!("x{sm}.dll"),
            bytes: 1,
            sha256: "0".repeat(64),
            signature: "0".repeat(128),
            url: "https://example.invalid/x".into(),
        }
    }
    #[test]
    fn selection_never_crosses_sm_major() {
        let entries = [entry(80), entry(86), entry(89), entry(90)];
        assert_eq!(select_entry(&entries, 87, "windows-x86_64").unwrap().sm, 86);
        assert_eq!(select_entry(&entries, 90, "windows-x86_64").unwrap().sm, 90);
        assert!(select_entry(&entries, 75, "windows-x86_64").is_none());
    }

    #[test]
    fn authenticates_hash_and_ed25519_signature() {
        let path = std::env::temp_dir()
            .join(format!("imparo-cuda-signature-{}.dll", std::process::id()));
        fs::write(&path, b"test cuda backend").unwrap();
        let expected_artifact: [u8; 32] = Sha256::digest(b"test cuda backend").into();
        assert_eq!(backend_artifact_sha256(&path).unwrap(), expected_artifact);
        let key = SigningKey::from_bytes(&[7_u8; 32]);
        let mut value = entry(86);
        value.bytes = fs::metadata(&path).unwrap().len();
        value.sha256 = encode_hex(&Sha256::digest(fs::read(&path).unwrap()));
        value.signature = encode_hex(&key.sign(&signed_message(&value)).to_bytes());
        let public = encode_hex(key.verifying_key().as_bytes());
        assert!(authenticate_with_key(&path, &value, &public).unwrap());
        fs::write(&path, b"tampered cuda backend").unwrap();
        assert!(!authenticate_with_key(&path, &value, &public).unwrap());
        let _ = fs::remove_file(path);
    }
}
