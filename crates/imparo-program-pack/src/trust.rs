//! Delegated Ed25519 trust roots, rotation, and revocation.

use crate::identity::{decode_hex_32, sha256, signature_message};
use base64::Engine as _;
use ed25519_dalek::{Signature, VerifyingKey};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DistributionScope {
    Community,
    Commercial,
    Custom,
}

#[derive(Clone, Debug)]
pub struct TrustedKey {
    pub key_id: String,
    pub public_key: [u8; 32],
    pub scope: DistributionScope,
    pub release_channel: String,
}

#[derive(Clone, Debug, Default)]
pub struct TrustStore {
    keys: BTreeMap<String, TrustedKey>,
    revoked: BTreeSet<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SignatureEnvelope {
    schema: u32,
    domain: String,
    algorithm: String,
    key_id: String,
    manifest_bytes: u64,
    manifest_sha256: String,
    signature: String,
}

impl TrustStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, key: TrustedKey) -> Result<(), String> {
        let key_max = crate::Policy::embedded()?.limits.trust_key_max_count;
        if self.keys.len() >= key_max {
            return Err("Program Pack trust store exceeds policy key limit".into());
        }
        validate_namespaced_id(&key.key_id)?;
        validate_namespaced_id(&key.release_channel)?;
        VerifyingKey::from_bytes(&key.public_key).map_err(|error| {
            format!("invalid Ed25519 public key {}: {error}", key.key_id)
        })?;
        if self
            .keys
            .values()
            .any(|existing| existing.public_key == key.public_key)
        {
            return Err(
                "one Program Pack public key cannot have multiple delegated identities"
                    .into(),
            );
        }
        if self.keys.contains_key(&key.key_id) {
            return Err("duplicate Program Pack trust key id".into());
        }
        self.keys.insert(key.key_id.clone(), key);
        Ok(())
    }

    pub fn revoke(&mut self, key_id: &str) -> Result<(), String> {
        if !self.keys.contains_key(key_id) {
            return Err(format!("cannot revoke unknown Program Pack key {key_id}"));
        }
        self.revoked.insert(key_id.to_owned());
        Ok(())
    }

    /// Authenticate exact manifest bytes. This does not grant correctness or entitlement.
    pub fn verify(
        &self,
        manifest: &[u8],
        signature_envelope: &[u8],
        distribution: DistributionScope,
        release_channel: &str,
    ) -> Result<String, String> {
        if signature_envelope.is_empty() {
            return Err("unsigned Program Pack is rejected".into());
        }
        let signature_max = usize::try_from(
            crate::Policy::embedded()?
                .limits
                .signature_envelope_max_bytes,
        )
        .map_err(|_| "Program Pack signature ceiling does not fit this host")?;
        if signature_envelope.len() > signature_max {
            return Err("Program Pack signature envelope exceeds limit".into());
        }
        let text = std::str::from_utf8(signature_envelope)
            .map_err(|_| "Program Pack signature envelope is not UTF-8")?;
        let envelope: SignatureEnvelope =
            serde_json::from_str(text).map_err(|error| {
                format!("parse strict Program Pack signature envelope: {error}")
            })?;
        if envelope.schema != 1
            || envelope.domain != "imparo-program-pack-v1"
            || envelope.algorithm != "ed25519"
        {
            return Err("unsupported Program Pack signature envelope".into());
        }
        validate_namespaced_id(&envelope.key_id)?;
        validate_namespaced_id(release_channel)?;
        if signature_envelope != canonical_envelope(&envelope)?.as_slice() {
            return Err("Program Pack signature envelope is not canonical ASCII".into());
        }
        if envelope.manifest_bytes != manifest.len() as u64 {
            return Err("Program Pack signature manifest length mismatch".into());
        }
        if decode_hex_32(&envelope.manifest_sha256)? != sha256(manifest) {
            return Err("Program Pack signature manifest digest mismatch".into());
        }
        let key = self.keys.get(&envelope.key_id).ok_or_else(|| {
            format!("unknown Program Pack signing key {}", envelope.key_id)
        })?;
        if self.revoked.contains(&envelope.key_id) {
            return Err(format!(
                "revoked Program Pack signing key {}",
                envelope.key_id
            ));
        }
        if !scope_allows(key.scope, distribution) {
            return Err(
                "Program Pack signing key scope does not match distribution".into()
            );
        }
        if key.release_channel != release_channel {
            return Err(
                "Program Pack signing key is not delegated for this release channel"
                    .into(),
            );
        }
        let raw_signature = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(envelope.signature.as_bytes())
            .map_err(|_| "Program Pack signature is not canonical base64url")?;
        if raw_signature.len() != 64
            || base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&raw_signature)
                != envelope.signature
        {
            return Err(
                "Program Pack signature is not canonical Ed25519 base64url".into()
            );
        }
        let signature_bytes: [u8; 64] = raw_signature
            .try_into()
            .map_err(|_| "Program Pack signature is not 64 bytes")?;
        let signature = Signature::from_bytes(&signature_bytes);
        let verifying_key = VerifyingKey::from_bytes(&key.public_key)
            .map_err(|error| format!("invalid trusted Ed25519 key: {error}"))?;
        verifying_key
            .verify_strict(&signature_message(manifest), &signature)
            .map_err(|_| "Program Pack signature verification failed".to_owned())?;
        Ok(envelope.key_id)
    }
}

fn scope_allows(key: DistributionScope, manifest: DistributionScope) -> bool {
    key == DistributionScope::Custom || key == manifest
}

fn canonical_envelope(envelope: &SignatureEnvelope) -> Result<Vec<u8>, String> {
    validate_namespaced_id(&envelope.key_id)?;
    let text = format!(
        "{{\"schema\":{},\"domain\":\"{}\",\"algorithm\":\"{}\",\"key_id\":\"{}\",\"manifest_bytes\":{},\"manifest_sha256\":\"{}\",\"signature\":\"{}\"}}",
        envelope.schema,
        envelope.domain,
        envelope.algorithm,
        envelope.key_id,
        envelope.manifest_bytes,
        envelope.manifest_sha256,
        envelope.signature
    );
    Ok(text.into_bytes())
}

pub(crate) fn validate_namespaced_id(value: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    let is_word = |byte: u8| byte.is_ascii_lowercase() || byte.is_ascii_digit();
    let is_separator = |byte: u8| matches!(byte, b'.' | b'_' | b'-');
    let mut saw_separator = false;
    let mut previous_separator = false;
    let valid = (3..=128).contains(&bytes.len())
        && bytes.first().is_some_and(|byte| is_word(*byte))
        && bytes.last().is_some_and(|byte| is_word(*byte))
        && bytes.iter().all(|byte| {
            if is_separator(*byte) {
                saw_separator = true;
                if previous_separator {
                    return false;
                }
                previous_separator = true;
                true
            } else {
                previous_separator = false;
                is_word(*byte)
            }
        })
        && saw_separator;
    if !valid {
        return Err("invalid namespaced key id".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ContentEntry, content_digest, encode_hex, install_digest};
    use ed25519_dalek::{Signer as _, SigningKey};
    use serde_json::json;

    fn envelope(raw: &[u8], key_id: &str, key: &SigningKey) -> Vec<u8> {
        let signature = key.sign(&signature_message(raw));
        let value = json!({
            "schema": 1,
            "domain": "imparo-program-pack-v1",
            "algorithm": "ed25519",
            "key_id": key_id,
            "manifest_bytes": raw.len(),
            "manifest_sha256": encode_hex(&sha256(raw)),
            "signature": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(signature.to_bytes()),
        });
        let parsed: SignatureEnvelope = serde_json::from_value(value).unwrap();
        canonical_envelope(&parsed).unwrap()
    }

    #[test]
    fn unknown_wrong_scope_revoked_and_tampered_signatures_fail_closed() {
        let key = SigningKey::from_bytes(&[7; 32]);
        let raw = b"{}";
        let sig = envelope(raw, "imparo.test.community", &key);
        let mut store = TrustStore::new();
        assert!(
            store
                .verify(
                    raw,
                    &sig,
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .is_err()
        );
        store
            .add(TrustedKey {
                key_id: "imparo.test.community".into(),
                public_key: key.verifying_key().to_bytes(),
                scope: DistributionScope::Community,
                release_channel: "imparo.community.stable".into(),
            })
            .unwrap();
        assert!(
            store
                .verify(
                    raw,
                    &sig,
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .is_ok()
        );
        assert!(
            store
                .verify(
                    raw,
                    &sig,
                    DistributionScope::Commercial,
                    "imparo.community.stable"
                )
                .is_err()
        );
        assert!(
            store
                .verify(
                    raw,
                    &sig,
                    DistributionScope::Community,
                    "imparo.community.beta"
                )
                .is_err()
        );
        assert!(
            store
                .verify(
                    b"{ }",
                    &sig,
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .is_err()
        );
        store.revoke("imparo.test.community").unwrap();
        assert!(
            store
                .verify(
                    raw,
                    &sig,
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .is_err()
        );
    }

    #[test]
    fn duplicate_unknown_and_noncanonical_envelopes_are_rejected() {
        let store = TrustStore::new();
        let duplicate = br#"{"schema":1,"schema":1,"domain":"imparo-program-pack-v1","algorithm":"ed25519","key_id":"imparo.test.key","manifest_bytes":2,"manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","signature":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}"#;
        assert!(
            store
                .verify(
                    b"{}",
                    duplicate,
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .is_err()
        );
        assert!(
            store
                .verify(
                    b"{}",
                    b"{}",
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .is_err()
        );
    }

    #[test]
    fn canonical_bytes_key_aliases_and_channel_delegation_are_enforced() {
        let key = SigningKey::from_bytes(&[8; 32]);
        let raw = b"{}";
        let signature = envelope(raw, "imparo.test.primary", &key);
        let mut store = TrustStore::new();
        store
            .add(TrustedKey {
                key_id: "imparo.test.primary".into(),
                public_key: key.verifying_key().to_bytes(),
                scope: DistributionScope::Community,
                release_channel: "imparo.community.stable".into(),
            })
            .unwrap();
        assert!(
            store
                .verify(
                    raw,
                    &signature,
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .is_ok()
        );
        assert!(
            store
                .verify(
                    raw,
                    &signature,
                    DistributionScope::Community,
                    "imparo.community.beta"
                )
                .unwrap_err()
                .contains("release channel")
        );
        let mut whitespace = vec![b' '];
        whitespace.extend(&signature);
        assert!(
            store
                .verify(
                    raw,
                    &whitespace,
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .unwrap_err()
                .contains("canonical ASCII")
        );
        let reordered = String::from_utf8(signature.clone()).unwrap().replacen(
            "{\"schema\":1,\"domain\":\"imparo-program-pack-v1\"",
            "{\"domain\":\"imparo-program-pack-v1\",\"schema\":1",
            1,
        );
        assert!(
            store
                .verify(
                    raw,
                    reordered.as_bytes(),
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .unwrap_err()
                .contains("canonical ASCII")
        );
        assert!(
            store
                .add(TrustedKey {
                    key_id: "imparo.test.alias".into(),
                    public_key: key.verifying_key().to_bytes(),
                    scope: DistributionScope::Custom,
                    release_channel: "imparo.custom.stable".into(),
                })
                .unwrap_err()
                .contains("multiple delegated identities")
        );
        assert!(
            store
                .verify(
                    raw,
                    &signature,
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .is_ok()
        );
    }

    #[test]
    fn key_rotation_preserves_content_but_creates_new_admission() {
        let raw = b"{}";
        let content = content_digest(&[ContentEntry {
            path: "manifest.json",
            bytes: raw,
        }])
        .unwrap();
        let old = SigningKey::from_bytes(&[9; 32]);
        let new = SigningKey::from_bytes(&[10; 32]);
        let old_signature = envelope(raw, "imparo.test.old", &old);
        let new_signature = envelope(raw, "imparo.test.new", &new);
        assert_ne!(
            install_digest(&content, &old_signature),
            install_digest(&content, &new_signature)
        );
        let mut store = TrustStore::new();
        for (id, key) in [("imparo.test.old", &old), ("imparo.test.new", &new)] {
            store
                .add(TrustedKey {
                    key_id: id.into(),
                    public_key: key.verifying_key().to_bytes(),
                    scope: DistributionScope::Community,
                    release_channel: "imparo.community.stable".into(),
                })
                .unwrap();
        }
        store.revoke("imparo.test.old").unwrap();
        assert!(
            store
                .verify(
                    raw,
                    &old_signature,
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .is_err()
        );
        assert!(
            store
                .verify(
                    raw,
                    &new_signature,
                    DistributionScope::Community,
                    "imparo.community.stable"
                )
                .is_ok()
        );
    }
}
