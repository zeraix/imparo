//! Domain-separated identities over exact Program Pack bytes.

use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;

const SIGNATURE_DOMAIN: &[u8] = b"imparo-program-pack-v1";
const CONTENT_DOMAIN: &[u8] = b"imparo-program-pack-content-v1";
const INSTALL_DOMAIN: &[u8] = b"imparo-program-pack-install-v1";
const CONTRACT_DOMAIN: &[u8] = b"imparo-program-contract-v1";
const CHOICE_GROUP_DOMAIN: &[u8] = b"imparo-program-choice-group-v1";

/// One immutable file included in the content identity.
#[derive(Clone, Copy, Debug)]
pub struct ContentEntry<'a> {
    pub path: &'a str,
    pub bytes: &'a [u8],
}

/// Pre-hashed immutable entry used by the streaming installer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContentIdentityEntry {
    pub path: String,
    pub bytes: u64,
    pub sha256: [u8; 32],
}

#[must_use]
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

#[must_use]
pub fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

pub fn decode_hex_32(value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("expected 64 lowercase hexadecimal characters".into());
    }
    if value.bytes().any(|b| b.is_ascii_uppercase()) {
        return Err("hexadecimal identity must be lowercase".into());
    }
    let mut out = [0_u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|error| format!("decode SHA-256: {error}"))?;
    }
    Ok(out)
}

fn length_prefixed(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update([0]);
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
    hash.finalize().into()
}

/// Exact bytes authenticated by an Ed25519 Program Pack signature.
#[must_use]
pub fn signature_message(raw_manifest: &[u8]) -> Vec<u8> {
    let mut message =
        Vec::with_capacity(SIGNATURE_DOMAIN.len() + 1 + 8 + 32 + raw_manifest.len());
    message.extend_from_slice(SIGNATURE_DOMAIN);
    message.push(0);
    message.extend_from_slice(&(raw_manifest.len() as u64).to_le_bytes());
    message.extend_from_slice(&sha256(raw_manifest));
    message.extend_from_slice(raw_manifest);
    message
}

#[must_use]
pub fn contract_digest(raw_contract: &[u8]) -> [u8; 32] {
    length_prefixed(CONTRACT_DOMAIN, raw_contract)
}

#[must_use]
pub fn choice_group_digest(group_id: &str) -> [u8; 32] {
    length_prefixed(CHOICE_GROUP_DOMAIN, group_id.as_bytes())
}

/// Hash the complete immutable object independently of its signing key.
pub fn content_digest(entries: &[ContentEntry<'_>]) -> Result<[u8; 32], String> {
    let metadata: Vec<_> = entries
        .iter()
        .map(|entry| ContentIdentityEntry {
            path: entry.path.to_owned(),
            bytes: entry.bytes.len() as u64,
            sha256: sha256(entry.bytes),
        })
        .collect();
    content_digest_from_metadata(&metadata)
}

/// Hash already verified streaming entries without loading modules into memory.
pub fn content_digest_from_metadata(
    entries: &[ContentIdentityEntry],
) -> Result<[u8; 32], String> {
    let mut ordered = entries.to_vec();
    ordered.sort_unstable_by(|left, right| {
        left.path.as_bytes().cmp(right.path.as_bytes())
    });
    let mut seen = BTreeSet::new();
    for entry in &ordered {
        if entry.path.is_empty() || !seen.insert(entry.path.clone()) {
            return Err("content identity contains an empty or duplicate path".into());
        }
    }
    let count = u32::try_from(ordered.len()).map_err(|_| "too many content entries")?;
    let mut hash = Sha256::new();
    hash.update(CONTENT_DOMAIN);
    hash.update([0]);
    hash.update(count.to_le_bytes());
    for entry in ordered {
        let path = entry.path.as_bytes();
        let path_len =
            u32::try_from(path.len()).map_err(|_| "content path is too long")?;
        hash.update(path_len.to_le_bytes());
        hash.update(path);
        hash.update(entry.bytes.to_le_bytes());
        hash.update(entry.sha256);
    }
    Ok(hash.finalize().into())
}

#[must_use]
pub fn install_digest(content: &[u8; 32], raw_signature: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(INSTALL_DOMAIN);
    hash.update([0]);
    hash.update(content);
    hash.update((raw_signature.len() as u64).to_le_bytes());
    hash.update(sha256(raw_signature));
    hash.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_identity_is_order_independent_but_path_sensitive() {
        let a = ContentEntry {
            path: "a",
            bytes: b"one",
        };
        let b = ContentEntry {
            path: "b",
            bytes: b"two",
        };
        assert_eq!(
            content_digest(&[a, b]).unwrap(),
            content_digest(&[b, a]).unwrap()
        );
        let changed = ContentEntry {
            path: "c",
            bytes: b"one",
        };
        assert_ne!(
            content_digest(&[a, b]).unwrap(),
            content_digest(&[changed, b]).unwrap()
        );
        assert!(content_digest(&[a, a]).is_err());
    }

    #[test]
    fn signature_and_install_domains_do_not_alias() {
        let raw = b"{}";
        let content = content_digest(&[ContentEntry {
            path: "manifest.json",
            bytes: raw,
        }])
        .unwrap();
        assert_ne!(
            sha256(&signature_message(raw)),
            install_digest(&content, raw)
        );
    }
}
