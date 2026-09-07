//! Fail-closed, content-addressed Program Pack installation.
//!
//! The installer accepts an already unpacked directory but never trusts or renames it.
//! Every admitted byte is copied through a no-follow handle into an engine-owned stage,
//! checked while streaming, made durable, and only then published by atomic rename.

use crate::extensions::{ExtensionRegistry, ExtensionResolution};
use crate::identity::{
    ContentIdentityEntry, content_digest_from_metadata, decode_hex_32, encode_hex,
    install_digest, sha256,
};
use crate::manifest::{Distribution, Manifest, Module};
use crate::policy::Policy;
use crate::trust::{DistributionScope, TrustStore};
use fs2::FileExt as _;
use serde::{Deserialize, Serialize};
use sha2::Digest as _;
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MANIFEST_FILE: &str = "manifest.json";
const SIGNATURE_FILE: &str = "manifest.sig";
const CONTENT_DIGEST_FILE: &str = "content.digest";
const SBOM_FILE: &str = "SBOM.spdx.json";
const NOTICES_FILE: &str = "THIRD_PARTY_NOTICES";
const PROVENANCE_FILE: &str = "provenance.json";
const CHANNEL_SCHEMA: u32 = 1;
const TRANSACTION_FILE: &str = "transaction.json";

#[derive(Clone, Debug)]
pub struct Admission {
    content_digest: String,
    install_digest: String,
    key_id: String,
    disabled_variants: BTreeSet<String>,
    release_channel: String,
    distribution: Distribution,
}

/// Immutable, revalidated module bytes suitable for a synchronous data-only loader.
/// No cache path escapes this crate, so callers cannot accidentally reopen mutable
/// transport state after admission.
#[derive(Clone, Debug)]
pub struct AdmittedModule {
    id: String,
    sha256: String,
    bytes: Vec<u8>,
}

impl AdmittedModule {
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// A trust-current admission plus the exact manifest and module bytes it authenticated.
#[derive(Clone, Debug)]
pub struct AdmittedPack {
    admission: Admission,
    manifest: Manifest,
    modules: Vec<AdmittedModule>,
}

impl AdmittedPack {
    #[must_use]
    pub fn admission(&self) -> &Admission {
        &self.admission
    }
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    #[must_use]
    pub fn modules(&self) -> &[AdmittedModule] {
        &self.modules
    }
}

impl Admission {
    #[must_use]
    pub fn content_digest(&self) -> &str {
        &self.content_digest
    }
    #[must_use]
    pub fn install_digest(&self) -> &str {
        &self.install_digest
    }
    #[must_use]
    pub fn key_id(&self) -> &str {
        &self.key_id
    }
    #[must_use]
    pub fn disabled_variants(&self) -> &BTreeSet<String> {
        &self.disabled_variants
    }
    #[must_use]
    pub fn release_channel(&self) -> &str {
        &self.release_channel
    }
    #[must_use]
    pub fn distribution(&self) -> Distribution {
        self.distribution
    }
}

/// A durable channel view. `known_good` is the sole retained N-1 admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelState {
    pub generation: u64,
    pub current: Option<String>,
    pub known_good: Option<String>,
}

#[derive(Debug)]
pub struct Installer {
    cache_root: PathBuf,
    trust: TrustStore,
    extensions: ExtensionRegistry,
    policy: Policy,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ChannelPointer {
    schema: u32,
    generation: u64,
    install_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ChannelTransaction {
    schema: u32,
    committed: bool,
    before_current: Option<ChannelPointer>,
    before_known_good: Option<ChannelPointer>,
    after_current: Option<ChannelPointer>,
    after_known_good: Option<ChannelPointer>,
}

struct ChannelLock(File);

impl Drop for ChannelLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

impl Installer {
    pub fn new(
        cache_root: impl Into<PathBuf>,
        trust: TrustStore,
        extensions: ExtensionRegistry,
    ) -> Result<Self, String> {
        Ok(Self {
            cache_root: cache_root.into(),
            trust,
            extensions,
            policy: Policy::embedded()?,
        })
    }

    pub fn trust_mut(&mut self) -> &mut TrustStore {
        &mut self.trust
    }

    /// Install from a transport-owned staging directory. No archive parsing or GPU
    /// loading occurs here, and the caller's directory is never renamed or mutated.
    pub fn install_from_staged_dir(&self, source: &Path) -> Result<Admission, String> {
        ensure_safe_directory(source)?;
        let raw_manifest = read_bounded_secure(
            &source.join(MANIFEST_FILE),
            self.policy.limits.manifest_max_bytes,
        )?;
        let raw_signature = read_bounded_secure(
            &source.join(SIGNATURE_FILE),
            self.policy.limits.signature_envelope_max_bytes,
        )?;
        let manifest = Manifest::parse(&raw_manifest).map_err(|e| e.to_string())?;
        let distribution = distribution_scope(manifest.distribution);
        let key_id = self.trust.verify(
            &raw_manifest,
            &raw_signature,
            distribution,
            &manifest.release_channel,
        )?;
        let resolution = self.extensions.resolve(&manifest)?;
        validate_source_inventory(source, &manifest)?;

        let declared_total = manifest.modules.iter().try_fold(
            raw_manifest.len() as u64 + raw_signature.len() as u64,
            |total, module| {
                total
                    .checked_add(module.bytes)
                    .ok_or("Program Pack size overflow")
            },
        )?;
        if declared_total > self.policy.limits.pack_max_bytes {
            return Err("Program Pack exceeds policy size ceiling".into());
        }

        self.prepare_cache()?;
        let _install_lock = lock_exclusive(&self.v1_root().join("install.lock"))?;
        let nonce = unique_nonce()?;
        let stage_root = self
            .v1_root()
            .join("staging")
            .join(format!("install-{nonce}.part"));
        fs::create_dir(&stage_root)
            .map_err(io_error("create private Program Pack stage"))?;
        let result = self.stage_and_publish(
            source,
            &stage_root,
            &manifest,
            &raw_manifest,
            &raw_signature,
            &key_id,
            &resolution,
        );
        if stage_root.exists() {
            let _ = fs::remove_dir_all(&stage_root);
        }
        result
    }

    /// Revalidate immutable bytes against current trust, revocation, extension and
    /// integrity policy. Admission is authentication only, never correctness or license.
    pub fn load_admission(&self, install: &str) -> Result<Admission, String> {
        decode_hex_32(install)?;
        let admission_dir = self.v1_root().join("admissions").join(install);
        ensure_safe_directory(&admission_dir)?;
        let content_raw =
            read_bounded_secure(&admission_dir.join(CONTENT_DIGEST_FILE), 64)?;
        if content_raw.len() != 64 {
            return Err(
                "Program Pack content.digest must be exactly 64 ASCII bytes".into()
            );
        }
        let content_text = std::str::from_utf8(&content_raw)
            .map_err(|_| "Program Pack content.digest is not ASCII")?;
        let content = decode_hex_32(content_text)?;
        let raw_signature = read_bounded_secure(
            &admission_dir.join(SIGNATURE_FILE),
            self.policy.limits.signature_envelope_max_bytes,
        )?;
        if encode_hex(&install_digest(&content, &raw_signature)) != install {
            return Err("Program Pack admission identity mismatch".into());
        }
        let object_dir = self.v1_root().join("objects").join(content_text);
        let (manifest, raw_manifest) =
            self.verify_object(&object_dir, content_text, raw_signature.len() as u64)?;
        let key_id = self.trust.verify(
            &raw_manifest,
            &raw_signature,
            distribution_scope(manifest.distribution),
            &manifest.release_channel,
        )?;
        let resolution = self.extensions.resolve(&manifest)?;
        verify_exact_inventory(&admission_dir, &[CONTENT_DIGEST_FILE, SIGNATURE_FILE])?;
        Ok(Admission {
            content_digest: content_text.to_owned(),
            install_digest: install.to_owned(),
            key_id,
            disabled_variants: resolution.disabled_variants,
            release_channel: manifest.release_channel,
            distribution: manifest.distribution,
        })
    }

    /// Revalidate an admission against current trust and return owned module bytes.
    /// `cuModuleLoadDataEx` is synchronous, so the CUDA bridge only needs these bytes
    /// for the duration of its install call and never receives a filesystem path.
    pub fn load_admitted_pack(&self, install: &str) -> Result<AdmittedPack, String> {
        let admission = self.load_admission(install)?;
        let object_dir = self
            .v1_root()
            .join("objects")
            .join(admission.content_digest());
        let raw_manifest = read_bounded_secure(
            &object_dir.join(MANIFEST_FILE),
            self.policy.limits.manifest_max_bytes,
        )?;
        let manifest =
            Manifest::parse(&raw_manifest).map_err(|error| error.to_string())?;
        let mut modules = Vec::with_capacity(manifest.modules().len());
        for module in manifest.modules() {
            let raw = read_bounded_secure(
                &object_dir.join(&module.file),
                self.policy.limits.module_max_bytes,
            )?;
            let bytes = u64::try_from(raw.len())
                .map_err(|_| "Program Pack module length does not fit u64")?;
            if bytes != module.bytes || encode_hex(&sha256(&raw)) != module.sha256 {
                return Err(
                    "installed Program Pack module changed after admission".into()
                );
            }
            modules.push(AdmittedModule {
                id: module.id.clone(),
                sha256: module.sha256.clone(),
                bytes: raw,
            });
        }
        Ok(AdmittedPack {
            admission,
            manifest,
            modules,
        })
    }

    fn v1_root(&self) -> PathBuf {
        self.cache_root.join("v1")
    }

    fn prepare_cache(&self) -> Result<(), String> {
        let root = self.v1_root();
        fs::create_dir_all(root.join("objects"))
            .map_err(io_error("create object cache"))?;
        fs::create_dir_all(root.join("admissions"))
            .map_err(io_error("create admission cache"))?;
        fs::create_dir_all(root.join("channels"))
            .map_err(io_error("create channel cache"))?;
        fs::create_dir_all(root.join("staging"))
            .map_err(io_error("create staging cache"))?;
        for directory in [
            root.as_path(),
            root.join("objects").as_path(),
            root.join("admissions").as_path(),
            root.join("channels").as_path(),
            root.join("staging").as_path(),
        ] {
            ensure_safe_directory(directory)?;
        }
        sync_directory(&root)
    }

    fn stage_and_publish(
        &self,
        source: &Path,
        stage_root: &Path,
        manifest: &Manifest,
        raw_manifest: &[u8],
        raw_signature: &[u8],
        key_id: &str,
        resolution: &ExtensionResolution,
    ) -> Result<Admission, String> {
        let object_stage = stage_root.join("object");
        fs::create_dir(&object_stage).map_err(io_error("create object stage"))?;
        fs::create_dir(object_stage.join("modules"))
            .map_err(io_error("create module stage"))?;
        let mut entries = Vec::with_capacity(manifest.modules.len() + 4);
        write_new_file(&object_stage.join(MANIFEST_FILE), raw_manifest)?;
        entries.push(identity_entry(MANIFEST_FILE, raw_manifest));
        let mut copied_total = raw_manifest.len() as u64 + raw_signature.len() as u64;
        for module in &manifest.modules {
            let entry =
                copy_declared_module(source, &object_stage, module, &self.policy)?;
            copied_total = copied_total
                .checked_add(entry.bytes)
                .ok_or("Program Pack size overflow")?;
            entries.push(entry);
        }
        for (path, declared_hash, ceiling) in [
            (
                SBOM_FILE,
                manifest.sbom_sha256.as_str(),
                self.policy.limits.sbom_max_bytes,
            ),
            (
                NOTICES_FILE,
                manifest.notices_sha256.as_str(),
                self.policy.limits.notices_max_bytes,
            ),
            (
                PROVENANCE_FILE,
                manifest.provenance_sha256.as_str(),
                self.policy.limits.provenance_max_bytes,
            ),
        ] {
            let entry = copy_checked(
                source,
                &object_stage,
                path,
                None,
                declared_hash,
                ceiling,
            )?;
            copied_total = copied_total
                .checked_add(entry.bytes)
                .ok_or("Program Pack size overflow")?;
            entries.push(entry);
        }
        if copied_total > self.policy.limits.pack_max_bytes {
            return Err("Program Pack exceeds policy size ceiling".into());
        }
        sync_directory(&object_stage.join("modules"))?;
        sync_directory(&object_stage)?;
        let content_bytes = content_digest_from_metadata(&entries)?;
        let content = encode_hex(&content_bytes);
        let object_dir = self.v1_root().join("objects").join(&content);
        publish_directory(&object_stage, &object_dir, || {
            self.verify_object(&object_dir, &content, raw_signature.len() as u64)
                .map(|_| ())
        })?;
        let install = encode_hex(&install_digest(&content_bytes, raw_signature));
        let admission_stage = stage_root.join("admission");
        fs::create_dir(&admission_stage).map_err(io_error("create admission stage"))?;
        write_new_file(
            &admission_stage.join(CONTENT_DIGEST_FILE),
            content.as_bytes(),
        )?;
        write_new_file(&admission_stage.join(SIGNATURE_FILE), raw_signature)?;
        sync_directory(&admission_stage)?;
        let admission_dir = self.v1_root().join("admissions").join(&install);
        publish_directory(&admission_stage, &admission_dir, || {
            verify_exact_inventory(
                &admission_dir,
                &[CONTENT_DIGEST_FILE, SIGNATURE_FILE],
            )?;
            let old_content =
                read_bounded_secure(&admission_dir.join(CONTENT_DIGEST_FILE), 64)?;
            let old_signature = read_bounded_secure(
                &admission_dir.join(SIGNATURE_FILE),
                self.policy.limits.signature_envelope_max_bytes,
            )?;
            if old_content != content.as_bytes() || old_signature != raw_signature {
                return Err("immutable Program Pack admission collision".into());
            }
            Ok(())
        })?;
        sync_directory(&self.v1_root().join("objects"))?;
        sync_directory(&self.v1_root().join("admissions"))?;
        Ok(Admission {
            content_digest: content,
            install_digest: install,
            key_id: key_id.to_owned(),
            disabled_variants: resolution.disabled_variants.clone(),
            release_channel: manifest.release_channel.clone(),
            distribution: manifest.distribution,
        })
    }

    fn verify_object(
        &self,
        object_dir: &Path,
        expected_content: &str,
        signature_bytes: u64,
    ) -> Result<(Manifest, Vec<u8>), String> {
        ensure_safe_directory(object_dir)?;
        let raw_manifest = read_bounded_secure(
            &object_dir.join(MANIFEST_FILE),
            self.policy.limits.manifest_max_bytes,
        )?;
        let manifest = Manifest::parse(&raw_manifest).map_err(|e| e.to_string())?;
        verify_object_inventory(object_dir, &manifest)?;
        let mut entries = Vec::with_capacity(manifest.modules.len() + 4);
        entries.push(identity_entry(MANIFEST_FILE, &raw_manifest));
        for module in &manifest.modules {
            entries.push(hash_checked_file(
                &object_dir.join(path_from_manifest(&module.file)?),
                &module.file,
                Some(module.bytes),
                &module.sha256,
                self.policy.limits.module_max_bytes,
            )?);
        }
        for (path, declared_hash, ceiling) in [
            (
                SBOM_FILE,
                manifest.sbom_sha256.as_str(),
                self.policy.limits.sbom_max_bytes,
            ),
            (
                NOTICES_FILE,
                manifest.notices_sha256.as_str(),
                self.policy.limits.notices_max_bytes,
            ),
            (
                PROVENANCE_FILE,
                manifest.provenance_sha256.as_str(),
                self.policy.limits.provenance_max_bytes,
            ),
        ] {
            entries.push(hash_checked_file(
                &object_dir.join(path),
                path,
                None,
                declared_hash,
                ceiling,
            )?);
        }
        if encode_hex(&content_digest_from_metadata(&entries)?) != expected_content {
            return Err("Program Pack content identity mismatch".into());
        }
        let total = entries.iter().try_fold(signature_bytes, |sum, entry| {
            sum.checked_add(entry.bytes)
                .ok_or("Program Pack size overflow")
        })?;
        if total > self.policy.limits.pack_max_bytes {
            return Err("installed Program Pack exceeds policy size ceiling".into());
        }
        Ok((manifest, raw_manifest))
    }

    /// Atomically advance current and N-1 through a durable transaction journal.
    pub fn advance_channel(
        &self,
        channel: &str,
        admission: &Admission,
    ) -> Result<ChannelState, String> {
        validate_channel_id(channel)?;
        let refreshed = self.load_admission(admission.install_digest())?;
        if refreshed.install_digest() != admission.install_digest() {
            return Err("Program Pack admission changed during channel advance".into());
        }
        if refreshed.release_channel() != channel {
            return Err(
                "Program Pack admission is delegated for a different channel".into(),
            );
        }
        self.prepare_cache()?;
        let channel_dir = self.v1_root().join("channels").join(channel);
        fs::create_dir_all(&channel_dir)
            .map_err(io_error("create channel directory"))?;
        let _lock = lock_exclusive(&channel_dir.join("channel.lock"))?;
        recover_channel_transaction(&channel_dir, &self.policy)?;
        let before_current =
            read_pointer_optional(&channel_dir.join("current.json"), &self.policy)?;
        let before_known =
            read_pointer_optional(&channel_dir.join("known-good.json"), &self.policy)?;
        ensure_matching_generations(before_current.as_ref(), before_known.as_ref())?;
        if let Some(pointer) = &before_current {
            if self
                .load_admission(&pointer.install_digest)?
                .release_channel()
                != channel
            {
                return Err(
                    "Program Pack prior current belongs to another channel".into()
                );
            }
        }
        if let Some(pointer) = &before_known {
            if self
                .load_admission(&pointer.install_digest)?
                .release_channel()
                != channel
            {
                return Err(
                    "Program Pack prior known-good belongs to another channel".into()
                );
            }
        }
        let generation = before_current
            .as_ref()
            .map_or(1, |p| p.generation.saturating_add(1));
        if generation == u64::MAX {
            return Err("Program Pack channel generation exhausted".into());
        }
        let after_current = Some(ChannelPointer {
            schema: CHANNEL_SCHEMA,
            generation,
            install_digest: admission.install_digest().to_owned(),
        });
        let after_known = before_current.as_ref().map(|previous| ChannelPointer {
            schema: CHANNEL_SCHEMA,
            generation,
            install_digest: previous.install_digest.clone(),
        });
        let mut transaction = ChannelTransaction {
            schema: CHANNEL_SCHEMA,
            committed: false,
            before_current: before_current.clone(),
            before_known_good: before_known.clone(),
            after_current: after_current.clone(),
            after_known_good: after_known.clone(),
        };
        write_transaction(&channel_dir, &transaction, &self.policy)?;
        write_pointer_optional(
            &channel_dir.join("known-good.json"),
            after_known.as_ref(),
        )?;
        write_pointer_optional(
            &channel_dir.join("current.json"),
            after_current.as_ref(),
        )?;
        sync_directory(&channel_dir)?;
        transaction.committed = true;
        write_transaction(&channel_dir, &transaction, &self.policy)?;
        remove_if_exists(&channel_dir.join(TRANSACTION_FILE))?;
        sync_directory(&channel_dir)?;
        Ok(state_from_pointers(after_current, after_known))
    }

    /// Swap current and known-good after both admissions pass current policy.
    pub fn rollback_channel(&self, channel: &str) -> Result<ChannelState, String> {
        validate_channel_id(channel)?;
        let channel_dir = self.v1_root().join("channels").join(channel);
        let _lock = lock_exclusive(&channel_dir.join("channel.lock"))?;
        recover_channel_transaction(&channel_dir, &self.policy)?;
        let current =
            read_pointer_optional(&channel_dir.join("current.json"), &self.policy)?
                .ok_or("Program Pack channel has no current admission")?;
        let known =
            read_pointer_optional(&channel_dir.join("known-good.json"), &self.policy)?
                .ok_or("Program Pack channel has no known-good admission")?;
        ensure_matching_generations(Some(&current), Some(&known))?;
        if self
            .load_admission(&current.install_digest)?
            .release_channel()
            != channel
            || self
                .load_admission(&known.install_digest)?
                .release_channel()
                != channel
        {
            return Err(
                "Program Pack rollback admission is delegated for a different channel"
                    .into(),
            );
        }
        let generation = current
            .generation
            .checked_add(1)
            .ok_or("Program Pack channel generation exhausted")?;
        let after_current = Some(ChannelPointer {
            schema: CHANNEL_SCHEMA,
            generation,
            install_digest: known.install_digest.clone(),
        });
        let after_known = Some(ChannelPointer {
            schema: CHANNEL_SCHEMA,
            generation,
            install_digest: current.install_digest.clone(),
        });
        let mut transaction = ChannelTransaction {
            schema: CHANNEL_SCHEMA,
            committed: false,
            before_current: Some(current),
            before_known_good: Some(known),
            after_current: after_current.clone(),
            after_known_good: after_known.clone(),
        };
        write_transaction(&channel_dir, &transaction, &self.policy)?;
        write_pointer_optional(
            &channel_dir.join("known-good.json"),
            after_known.as_ref(),
        )?;
        write_pointer_optional(
            &channel_dir.join("current.json"),
            after_current.as_ref(),
        )?;
        sync_directory(&channel_dir)?;
        transaction.committed = true;
        write_transaction(&channel_dir, &transaction, &self.policy)?;
        remove_if_exists(&channel_dir.join(TRANSACTION_FILE))?;
        sync_directory(&channel_dir)?;
        Ok(state_from_pointers(after_current, after_known))
    }

    pub fn channel_state(&self, channel: &str) -> Result<ChannelState, String> {
        validate_channel_id(channel)?;
        let channel_dir = self.v1_root().join("channels").join(channel);
        if !channel_dir.exists() {
            return Ok(ChannelState {
                generation: 0,
                current: None,
                known_good: None,
            });
        }
        let _lock = lock_exclusive(&channel_dir.join("channel.lock"))?;
        recover_channel_transaction(&channel_dir, &self.policy)?;
        let current =
            read_pointer_optional(&channel_dir.join("current.json"), &self.policy)?;
        let known =
            read_pointer_optional(&channel_dir.join("known-good.json"), &self.policy)?;
        ensure_matching_generations(current.as_ref(), known.as_ref())?;
        if let Some(pointer) = &current {
            if self
                .load_admission(&pointer.install_digest)?
                .release_channel()
                != channel
            {
                return Err("Program Pack current admission is delegated for a different channel".into());
            }
        }
        if let Some(pointer) = &known {
            if self
                .load_admission(&pointer.install_digest)?
                .release_channel()
                != channel
            {
                return Err("Program Pack known-good admission is delegated for a different channel".into());
            }
        }
        Ok(state_from_pointers(current, known))
    }
}

fn distribution_scope(distribution: Distribution) -> DistributionScope {
    match distribution {
        Distribution::Community => DistributionScope::Community,
        Distribution::Commercial => DistributionScope::Commercial,
    }
}

fn state_from_pointers(
    current: Option<ChannelPointer>,
    known: Option<ChannelPointer>,
) -> ChannelState {
    let generation = current.as_ref().map_or(0, |p| p.generation);
    ChannelState {
        generation,
        current: current.map(|p| p.install_digest),
        known_good: known.map(|p| p.install_digest),
    }
}

fn ensure_matching_generations(
    current: Option<&ChannelPointer>,
    known: Option<&ChannelPointer>,
) -> Result<(), String> {
    if let Some(pointer) = current {
        validate_pointer(pointer)?;
    }
    if let Some(pointer) = known {
        validate_pointer(pointer)?;
    }
    if let (Some(current), Some(known)) = (current, known) {
        if current.generation != known.generation {
            return Err("Program Pack channel pointers have split generations".into());
        }
    }
    if current.is_none() && known.is_some() {
        return Err("Program Pack channel has known-good without current".into());
    }
    Ok(())
}

fn validate_pointer(pointer: &ChannelPointer) -> Result<(), String> {
    if pointer.schema != CHANNEL_SCHEMA || pointer.generation == 0 {
        return Err("unsupported Program Pack channel pointer".into());
    }
    decode_hex_32(&pointer.install_digest)?;
    Ok(())
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|e| format!("serialize Program Pack state: {e}"))
}

fn read_pointer_optional(
    path: &Path,
    policy: &Policy,
) -> Result<Option<ChannelPointer>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = read_bounded_secure(path, policy.limits.channel_pointer_max_bytes)?;
    let pointer: ChannelPointer = serde_json::from_slice(&raw)
        .map_err(|e| format!("parse Program Pack channel pointer: {e}"))?;
    validate_pointer(&pointer)?;
    if canonical_json(&pointer)? != raw {
        return Err("Program Pack channel pointer is not canonical JSON".into());
    }
    Ok(Some(pointer))
}

fn write_pointer_optional(
    path: &Path,
    pointer: Option<&ChannelPointer>,
) -> Result<(), String> {
    match pointer {
        Some(pointer) => atomic_replace_file(path, &canonical_json(pointer)?),
        None => remove_if_exists(path),
    }
}

fn write_transaction(
    dir: &Path,
    tx: &ChannelTransaction,
    policy: &Policy,
) -> Result<(), String> {
    let raw = canonical_json(tx)?;
    if raw.len() as u64 > policy.limits.channel_transaction_max_bytes {
        return Err("Program Pack channel transaction exceeds policy limit".into());
    }
    atomic_replace_file(&dir.join(TRANSACTION_FILE), &raw)?;
    sync_directory(dir)
}

fn recover_channel_transaction(dir: &Path, policy: &Policy) -> Result<(), String> {
    let path = dir.join(TRANSACTION_FILE);
    if !path.exists() {
        return Ok(());
    }
    let raw = read_bounded_secure(&path, policy.limits.channel_transaction_max_bytes)?;
    let tx: ChannelTransaction = serde_json::from_slice(&raw)
        .map_err(|e| format!("parse Program Pack channel transaction: {e}"))?;
    if tx.schema != CHANNEL_SCHEMA || canonical_json(&tx)? != raw {
        return Err("Program Pack channel transaction is not canonical".into());
    }
    ensure_matching_generations(
        tx.before_current.as_ref(),
        tx.before_known_good.as_ref(),
    )?;
    ensure_matching_generations(
        tx.after_current.as_ref(),
        tx.after_known_good.as_ref(),
    )?;
    validate_transaction_step(&tx)?;
    let (current, known) = if tx.committed {
        (tx.after_current.as_ref(), tx.after_known_good.as_ref())
    } else {
        (tx.before_current.as_ref(), tx.before_known_good.as_ref())
    };
    write_pointer_optional(&dir.join("known-good.json"), known)?;
    write_pointer_optional(&dir.join("current.json"), current)?;
    sync_directory(dir)?;
    remove_if_exists(&path)?;
    sync_directory(dir)
}

fn validate_transaction_step(tx: &ChannelTransaction) -> Result<(), String> {
    let before_generation = tx.before_current.as_ref().map_or(0, |p| p.generation);
    let after = tx
        .after_current
        .as_ref()
        .ok_or("Program Pack transaction has no after-current")?;
    if after.generation
        != before_generation
            .checked_add(1)
            .ok_or("Program Pack channel generation exhausted")?
    {
        return Err(
            "Program Pack transaction does not advance exactly one generation".into(),
        );
    }
    match (&tx.before_current, &tx.after_known_good) {
        (None, None) => {}
        (Some(before), Some(after_known))
            if before.install_digest == after_known.install_digest => {}
        _ => return Err("Program Pack transaction does not retain exact N-1".into()),
    }
    Ok(())
}

fn identity_entry(path: &str, bytes: &[u8]) -> ContentIdentityEntry {
    ContentIdentityEntry {
        path: path.to_owned(),
        bytes: bytes.len() as u64,
        sha256: sha256(bytes),
    }
}

fn copy_declared_module(
    source: &Path,
    destination: &Path,
    module: &Module,
    policy: &Policy,
) -> Result<ContentIdentityEntry, String> {
    let relative = path_from_manifest(&module.file)?;
    copy_checked(
        source,
        destination,
        relative.to_str().ok_or("module path is not UTF-8")?,
        Some(module.bytes),
        &module.sha256,
        policy.limits.module_max_bytes,
    )
}

fn copy_checked(
    source_root: &Path,
    destination_root: &Path,
    relative: &str,
    expected_bytes: Option<u64>,
    expected_hash: &str,
    ceiling: u64,
) -> Result<ContentIdentityEntry, String> {
    let relative_path = path_from_manifest(relative)?;
    let source = source_root.join(&relative_path);
    let destination = destination_root.join(&relative_path);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(io_error("create Program Pack destination directory"))?;
    }
    let mut input = open_regular_nofollow(&source)?;
    let metadata = input
        .metadata()
        .map_err(io_error("inspect Program Pack source file"))?;
    ensure_regular_metadata(&metadata)?;
    let bytes = metadata.len();
    if bytes > ceiling || expected_bytes.is_some_and(|value| value != bytes) {
        return Err("Program Pack file length violates manifest or policy".into());
    }
    let expected_digest = decode_hex_32(expected_hash)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&destination)
        .map_err(io_error("create Program Pack destination file"))?;
    let mut hash = sha2::Sha256::new();
    let mut copied = 0_u64;
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(io_error("read Program Pack source file"))?;
        if count == 0 {
            break;
        }
        copied = copied
            .checked_add(count as u64)
            .ok_or("Program Pack file length overflow")?;
        if copied > ceiling {
            return Err("Program Pack file grew beyond policy limit".into());
        }
        sha2::Digest::update(&mut hash, &buffer[..count]);
        output
            .write_all(&buffer[..count])
            .map_err(io_error("write Program Pack staged file"))?;
    }
    if copied != bytes || expected_bytes.is_some_and(|value| value != copied) {
        return Err("Program Pack file changed during copy".into());
    }
    let digest: [u8; 32] = sha2::Digest::finalize(hash).into();
    if digest != expected_digest {
        return Err("Program Pack file digest mismatch".into());
    }
    output
        .sync_all()
        .map_err(io_error("sync Program Pack staged file"))?;
    drop(output);
    let staged = open_regular_nofollow(&destination)?;
    let staged_metadata = staged
        .metadata()
        .map_err(io_error("inspect staged Program Pack file"))?;
    ensure_regular_metadata(&staged_metadata)?;
    if staged_metadata.len() != copied {
        return Err("staged Program Pack file length mismatch".into());
    }
    Ok(ContentIdentityEntry {
        path: relative.to_owned(),
        bytes: copied,
        sha256: digest,
    })
}

fn hash_checked_file(
    path: &Path,
    identity_path: &str,
    expected_bytes: Option<u64>,
    expected_hash: &str,
    ceiling: u64,
) -> Result<ContentIdentityEntry, String> {
    let mut file = open_regular_nofollow(path)?;
    let metadata = file
        .metadata()
        .map_err(io_error("inspect installed Program Pack file"))?;
    ensure_regular_metadata(&metadata)?;
    if metadata.len() > ceiling
        || expected_bytes.is_some_and(|value| value != metadata.len())
    {
        return Err(
            "installed Program Pack file length violates manifest or policy".into(),
        );
    }
    let mut hash = sha2::Sha256::new();
    let mut seen = 0_u64;
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(io_error("read installed Program Pack file"))?;
        if count == 0 {
            break;
        }
        seen = seen
            .checked_add(count as u64)
            .ok_or("Program Pack file length overflow")?;
        if seen > ceiling {
            return Err("installed Program Pack file exceeds policy limit".into());
        }
        sha2::Digest::update(&mut hash, &buffer[..count]);
    }
    if seen != metadata.len() {
        return Err("installed Program Pack file changed during verification".into());
    }
    let digest: [u8; 32] = sha2::Digest::finalize(hash).into();
    if digest != decode_hex_32(expected_hash)? {
        return Err("installed Program Pack file digest mismatch".into());
    }
    Ok(ContentIdentityEntry {
        path: identity_path.to_owned(),
        bytes: seen,
        sha256: digest,
    })
}

fn validate_source_inventory(root: &Path, manifest: &Manifest) -> Result<(), String> {
    let mut expected = object_inventory(manifest)?;
    expected.insert(SIGNATURE_FILE.to_owned());
    verify_inventory(root, &expected)
}

fn verify_object_inventory(root: &Path, manifest: &Manifest) -> Result<(), String> {
    verify_inventory(root, &object_inventory(manifest)?)
}

fn object_inventory(manifest: &Manifest) -> Result<BTreeSet<String>, String> {
    let mut expected = BTreeSet::from([
        MANIFEST_FILE.to_owned(),
        SBOM_FILE.to_owned(),
        NOTICES_FILE.to_owned(),
        PROVENANCE_FILE.to_owned(),
    ]);
    for module in &manifest.modules {
        path_from_manifest(&module.file)?;
        if !expected.insert(module.file.clone()) {
            return Err("duplicate Program Pack inventory path".into());
        }
    }
    Ok(expected)
}

fn verify_exact_inventory(root: &Path, expected: &[&str]) -> Result<(), String> {
    verify_inventory(
        root,
        &expected.iter().map(|value| (*value).to_owned()).collect(),
    )
}

fn verify_inventory(root: &Path, expected: &BTreeSet<String>) -> Result<(), String> {
    ensure_safe_directory(root)?;
    let expected_root: BTreeSet<_> = expected
        .iter()
        .filter(|path| !path.contains('/'))
        .cloned()
        .collect();
    let expected_modules: BTreeSet<_> = expected
        .iter()
        .filter_map(|path| path.strip_prefix("modules/").map(str::to_owned))
        .collect();
    let mut found_root = BTreeSet::new();
    let mut found_modules = BTreeSet::new();
    let mut saw_modules = false;
    for entry in fs::read_dir(root).map_err(io_error("read Program Pack directory"))? {
        let entry = entry.map_err(io_error("read Program Pack directory entry"))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "Program Pack filename is not UTF-8")?;
        let file_type = entry
            .file_type()
            .map_err(io_error("inspect Program Pack directory entry"))?;
        if file_type.is_symlink() {
            return Err("Program Pack inventory contains a link".into());
        }
        if file_type.is_dir() {
            if name != "modules" || saw_modules {
                return Err(
                    "Program Pack inventory contains an unexpected directory".into()
                );
            }
            saw_modules = true;
            ensure_safe_directory(&entry.path())?;
            for module in fs::read_dir(entry.path())
                .map_err(io_error("read Program Pack modules directory"))?
            {
                let module =
                    module.map_err(io_error("read Program Pack module entry"))?;
                let module_type = module
                    .file_type()
                    .map_err(io_error("inspect Program Pack module entry"))?;
                if !module_type.is_file() || module_type.is_symlink() {
                    return Err("Program Pack modules contains a non-file entry".into());
                }
                let module_name = module
                    .file_name()
                    .into_string()
                    .map_err(|_| "Program Pack module filename is not UTF-8")?;
                if !expected_modules.contains(&module_name)
                    || !found_modules.insert(module_name)
                {
                    return Err(
                        "Program Pack inventory has an extra or duplicate module"
                            .into(),
                    );
                }
            }
        } else if file_type.is_file() {
            if !expected_root.contains(&name) || !found_root.insert(name) {
                return Err(
                    "Program Pack inventory has an extra or duplicate root file".into(),
                );
            }
        } else {
            return Err("Program Pack inventory contains a special file".into());
        }
    }
    if found_root != expected_root
        || found_modules != expected_modules
        || saw_modules == expected_modules.is_empty()
    {
        return Err("Program Pack inventory has missing or extra entries".into());
    }
    Ok(())
}

fn path_from_manifest(value: &str) -> Result<PathBuf, String> {
    if value.is_empty()
        || value.contains('\\')
        || value.contains(':')
        || value.starts_with('/')
    {
        return Err("invalid Program Pack relative path".into());
    }
    let path = Path::new(value);
    if path
        .components()
        .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err("invalid Program Pack relative path component".into());
    }
    Ok(path.to_path_buf())
}

fn read_bounded_secure(path: &Path, ceiling: u64) -> Result<Vec<u8>, String> {
    let mut file = open_regular_nofollow(path)?;
    let metadata = file
        .metadata()
        .map_err(io_error("inspect bounded Program Pack file"))?;
    ensure_regular_metadata(&metadata)?;
    if metadata.len() > ceiling {
        return Err("Program Pack file exceeds policy limit".into());
    }
    let capacity = usize::try_from(metadata.len())
        .map_err(|_| "Program Pack file is too large for host")?;
    let mut bytes = Vec::with_capacity(capacity);
    Read::by_ref(&mut file)
        .take(ceiling.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(io_error("read bounded Program Pack file"))?;
    if bytes.len() as u64 > ceiling || bytes.len() as u64 != metadata.len() {
        return Err("Program Pack file changed or exceeded limit during read".into());
    }
    Ok(bytes)
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(io_error("create immutable Program Pack file"))?;
    file.write_all(bytes)
        .map_err(io_error("write immutable Program Pack file"))?;
    file.sync_all()
        .map_err(io_error("sync immutable Program Pack file"))
}

fn publish_directory<F>(
    stage: &Path,
    destination: &Path,
    verify_existing: F,
) -> Result<(), String>
where
    F: FnOnce() -> Result<(), String>,
{
    match rename_new_directory(stage, destination) {
        Ok(()) => {
            if let Some(parent) = destination.parent() {
                sync_directory(parent)?;
            }
            Ok(())
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => verify_existing(),
        Err(error) => {
            // Windows reports PermissionDenied when the destination directory exists.
            if destination.is_dir() {
                verify_existing()
            } else {
                Err(format!("publish immutable Program Pack directory: {error}"))
            }
        }
    }
}

#[cfg(not(windows))]
fn rename_new_directory(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn rename_new_directory(source: &Path, destination: &Path) -> io::Result<()> {
    move_file_ex(source, destination, false)
}

fn atomic_replace_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or("Program Pack state path has no parent")?;
    fs::create_dir_all(parent)
        .map_err(io_error("create Program Pack state directory"))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("Program Pack state filename is not UTF-8")?;
    let temporary = parent.join(format!(".{file_name}.{}.part", unique_nonce()?));
    write_new_file(&temporary, bytes)?;
    replace_file(&temporary, path)?;
    sync_directory(parent)
}

fn remove_if_exists(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("remove Program Pack state file: {error}")),
    }
}

fn lock_exclusive(path: &Path) -> Result<ChannelLock, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(io_error("create Program Pack lock directory"))?;
    }
    let file = open_lock_nofollow(path)?;
    file.lock_exclusive()
        .map_err(io_error("lock Program Pack state"))?;
    let check = open_regular_nofollow(path)?;
    if file_identity(&file)? != file_identity(&check)? {
        return Err("Program Pack lock path changed during acquisition".into());
    }
    Ok(ChannelLock(file))
}

fn validate_channel_id(value: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    let word = |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit();
    let separator = |b: u8| matches!(b, b'.' | b'_' | b'-');
    let mut prior_separator = false;
    let mut saw_separator = false;
    if !(3..=128).contains(&bytes.len())
        || !bytes.first().is_some_and(|b| word(*b))
        || !bytes.last().is_some_and(|b| word(*b))
    {
        return Err("invalid Program Pack channel id".into());
    }
    for byte in bytes {
        if separator(*byte) {
            if prior_separator {
                return Err("invalid Program Pack channel id".into());
            }
            prior_separator = true;
            saw_separator = true;
        } else {
            if !word(*byte) {
                return Err("invalid Program Pack channel id".into());
            }
            prior_separator = false;
        }
    }
    if !saw_separator {
        return Err("invalid Program Pack channel id".into());
    }
    Ok(())
}

fn unique_nonce() -> Result<String, String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "system clock precedes Unix epoch")?
        .as_nanos();
    Ok(format!("{}-{nanos}", std::process::id()))
}

fn io_error(context: &'static str) -> impl FnOnce(io::Error) -> String {
    move |error| format!("{context}: {error}")
}

#[cfg(unix)]
fn open_regular_nofollow(path: &Path) -> Result<File, String> {
    use std::os::unix::fs::OpenOptionsExt as _;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(io_error("open no-follow Program Pack file"))?;
    ensure_file_handle(&file)?;
    Ok(file)
}

#[cfg(windows)]
fn open_regular_nofollow(path: &Path) -> Result<File, String> {
    use std::os::windows::fs::OpenOptionsExt as _;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
        .map_err(io_error("open no-follow Program Pack file"))?;
    ensure_file_handle(&file)?;
    Ok(file)
}

#[cfg(unix)]
fn open_lock_nofollow(path: &Path) -> Result<File, String> {
    use std::os::unix::fs::OpenOptionsExt as _;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(io_error("open no-follow Program Pack lock"))?;
    ensure_file_handle(&file)?;
    Ok(file)
}

#[cfg(windows)]
fn open_lock_nofollow(path: &Path) -> Result<File, String> {
    use std::os::windows::fs::OpenOptionsExt as _;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
        .map_err(io_error("open no-follow Program Pack lock"))?;
    ensure_file_handle(&file)?;
    Ok(file)
}

fn ensure_file_handle(file: &File) -> Result<(), String> {
    let metadata = file
        .metadata()
        .map_err(io_error("inspect Program Pack file handle"))?;
    ensure_regular_metadata(&metadata)?;
    #[cfg(windows)]
    if windows_file_identity(file)?.2 != 1 {
        return Err("Program Pack file has multiple hard links".into());
    }
    Ok(())
}

#[cfg(unix)]
fn file_identity(file: &File) -> Result<(u64, u64), String> {
    use std::os::unix::fs::MetadataExt as _;
    let metadata = file
        .metadata()
        .map_err(io_error("inspect Program Pack file identity"))?;
    Ok((metadata.dev(), metadata.ino()))
}

#[cfg(windows)]
fn file_identity(file: &File) -> Result<(u32, u64), String> {
    let (volume, index, _) = windows_file_identity(file)?;
    Ok((volume, index))
}

fn ensure_regular_metadata(metadata: &fs::Metadata) -> Result<(), String> {
    if !metadata.file_type().is_file() {
        return Err("Program Pack entry is not a regular file".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if metadata.nlink() != 1 {
            return Err("Program Pack file has multiple hard links".into());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err("Program Pack file is a reparse point".into());
        }
    }
    Ok(())
}

#[cfg(windows)]
fn windows_file_identity(file: &File) -> Result<(u32, u64, u32), String> {
    use std::ffi::c_void;
    use std::mem::MaybeUninit;
    use std::os::windows::io::AsRawHandle as _;
    #[repr(C)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    #[repr(C)]
    struct ByHandleFileInformation {
        attributes: u32,
        creation: FileTime,
        access: FileTime,
        write: FileTime,
        volume_serial: u32,
        size_high: u32,
        size_low: u32,
        link_count: u32,
        index_high: u32,
        index_low: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetFileInformationByHandle(
            file: *mut c_void,
            information: *mut ByHandleFileInformation,
        ) -> i32;
    }
    let mut information = MaybeUninit::<ByHandleFileInformation>::uninit();
    // SAFETY: the OS writes one initialized structure on success; the handle stays alive.
    let result = unsafe {
        GetFileInformationByHandle(file.as_raw_handle(), information.as_mut_ptr())
    };
    if result == 0 {
        return Err(format!(
            "inspect Program Pack Windows file identity: {}",
            io::Error::last_os_error()
        ));
    }
    // SAFETY: success guarantees the output structure was initialized.
    let information = unsafe { information.assume_init() };
    let index =
        (u64::from(information.index_high) << 32) | u64::from(information.index_low);
    Ok((information.volume_serial, index, information.link_count))
}

fn ensure_safe_directory(path: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(io_error("inspect Program Pack directory"))?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err("Program Pack directory is not a real directory".into());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err("Program Pack directory is a reparse point".into());
        }
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(io_error("sync Program Pack directory"))
}

#[cfg(windows)]
fn sync_directory(path: &Path) -> Result<(), String> {
    // Windows does not support FlushFileBuffers on directory handles. File bytes are
    // flushed individually and state-file renames use MOVEFILE_WRITE_THROUGH.
    ensure_safe_directory(path)
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> Result<(), String> {
    fs::rename(source, destination).map_err(io_error("replace Program Pack state file"))
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> Result<(), String> {
    move_file_ex(source, destination, true)
        .map_err(io_error("replace Program Pack state file"))
}

#[cfg(windows)]
fn move_file_ex(source: &Path, destination: &Path, replace: bool) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt as _;
    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(
            existing: *const u16,
            replacement: *const u16,
            flags: u32,
        ) -> i32;
    }
    let existing: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let replacement: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    // SAFETY: both pointers reference NUL-terminated buffers alive for the call.
    let flags = MOVEFILE_WRITE_THROUGH
        | if replace {
            MOVEFILE_REPLACE_EXISTING
        } else {
            0
        };
    let result = unsafe { MoveFileExW(existing.as_ptr(), replacement.as_ptr(), flags) };
    if result == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use ed25519_dalek::{Signer as _, SigningKey};
    use serde_json::Value;
    use tempfile::TempDir;

    const KEY_ID: &str = "imparo.test.community";
    const CHANNEL: &str = "imparo.community.stable";
    const MODULE: &str = "modules/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.cubin";

    fn trusted_installer(cache: &Path, key: &SigningKey) -> Installer {
        let mut trust = TrustStore::new();
        trust
            .add(crate::TrustedKey {
                key_id: KEY_ID.into(),
                public_key: key.verifying_key().to_bytes(),
                scope: DistributionScope::Community,
                release_channel: CHANNEL.into(),
            })
            .unwrap();
        Installer::new(cache, trust, ExtensionRegistry::new()).unwrap()
    }

    fn make_pack(root: &Path, key: &SigningKey, version: &str) {
        let module = format!("fake-cubin-{version}").into_bytes();
        let sbom = br#"{"spdxVersion":"SPDX-2.3"}"#;
        let notices = b"test notices";
        let provenance = br#"{"builder":"test"}"#;
        fs::create_dir_all(root.join("modules")).unwrap();
        fs::write(root.join(MODULE), &module).unwrap();
        fs::write(root.join(SBOM_FILE), sbom).unwrap();
        fs::write(root.join(NOTICES_FILE), notices).unwrap();
        fs::write(root.join(PROVENANCE_FILE), provenance).unwrap();
        let mut value: Value = serde_json::from_slice(include_bytes!(
            "../../../schemas/examples/program-pack-v1.minimal.json"
        ))
        .unwrap();
        value["pack_version"] = Value::String(version.into());
        value["modules"][0]["bytes"] = Value::from(module.len() as u64);
        value["modules"][0]["sha256"] = Value::String(encode_hex(&sha256(&module)));
        value["sbom_sha256"] = Value::String(encode_hex(&sha256(sbom)));
        value["notices_sha256"] = Value::String(encode_hex(&sha256(notices)));
        value["provenance_sha256"] = Value::String(encode_hex(&sha256(provenance)));
        let manifest = serde_json::to_vec(&value).unwrap();
        Manifest::parse(&manifest).unwrap();
        fs::write(root.join(MANIFEST_FILE), &manifest).unwrap();
        let signature = key.sign(&crate::identity::signature_message(&manifest));
        let envelope = format!(
            "{{\"schema\":1,\"domain\":\"imparo-program-pack-v1\",\"algorithm\":\"ed25519\",\"key_id\":\"{KEY_ID}\",\"manifest_bytes\":{},\"manifest_sha256\":\"{}\",\"signature\":\"{}\"}}",
            manifest.len(),
            encode_hex(&sha256(&manifest)),
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(signature.to_bytes())
        );
        fs::write(root.join(SIGNATURE_FILE), envelope).unwrap();
    }

    #[test]
    fn install_reloads_exact_bytes_and_rejects_tamper_or_extra_files() {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join("source");
        fs::create_dir(&source).unwrap();
        let key = SigningKey::from_bytes(&[11; 32]);
        make_pack(&source, &key, "1.0.0");
        let installer = trusted_installer(&temp.path().join("cache"), &key);
        let admission = installer.install_from_staged_dir(&source).unwrap();
        assert_eq!(
            installer
                .load_admission(admission.install_digest())
                .unwrap()
                .key_id(),
            KEY_ID
        );
        let admission_path = installer
            .v1_root()
            .join("admissions")
            .join(admission.install_digest());
        assert_eq!(
            fs::read(admission_path.join(CONTENT_DIGEST_FILE))
                .unwrap()
                .len(),
            64
        );
        let object_path = installer
            .v1_root()
            .join("objects")
            .join(admission.content_digest());
        fs::write(object_path.join(MODULE), b"tampered!!").unwrap();
        assert!(
            installer
                .load_admission(admission.install_digest())
                .is_err()
        );
        fs::write(source.join("unexpected"), b"x").unwrap();
        assert!(installer.install_from_staged_dir(&source).is_err());
    }

    #[test]
    fn load_admitted_pack_returns_verified_owned_bytes_without_cache_paths() {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join("source");
        let cache = temp.path().join("cache");
        fs::create_dir(&source).unwrap();
        let key = SigningKey::from_bytes(&[17; 32]);
        make_pack(&source, &key, "1.0.0");
        let expected_bytes = fs::read(source.join(MODULE)).unwrap();
        let installer = trusted_installer(&cache, &key);
        let admission = installer.install_from_staged_dir(&source).unwrap();

        let pack = installer
            .load_admitted_pack(admission.install_digest())
            .unwrap();
        assert_eq!(
            pack.admission().install_digest(),
            admission.install_digest()
        );
        assert_eq!(
            pack.admission().content_digest(),
            admission.content_digest()
        );
        assert_eq!(pack.manifest().modules().len(), 1);
        assert_eq!(pack.modules().len(), 1);
        assert_eq!(pack.modules()[0].id(), pack.manifest().modules()[0].id);
        assert_eq!(
            pack.modules()[0].sha256(),
            pack.manifest().modules()[0].sha256
        );
        assert_eq!(pack.modules()[0].bytes(), expected_bytes);

        // The admitted value is safe to hand to a backend: neither its public API nor
        // its Debug representation contains the source/cache/object filesystem path.
        let debug = format!("{pack:?}");
        for path in [&source, &cache] {
            assert!(
                !debug.contains(path.to_string_lossy().as_ref()),
                "AdmittedPack leaked a filesystem path"
            );
        }
    }

    #[test]
    fn load_admitted_pack_rejects_installed_module_length_and_digest_tamper() {
        fn install_case(
            temp: &TempDir,
            name: &str,
            seed: u8,
        ) -> (Installer, Admission, PathBuf) {
            let source = temp.path().join(format!("{name}-source"));
            let cache = temp.path().join(format!("{name}-cache"));
            fs::create_dir(&source).unwrap();
            let key = SigningKey::from_bytes(&[seed; 32]);
            make_pack(&source, &key, "1.0.0");
            let installer = trusted_installer(&cache, &key);
            let admission = installer.install_from_staged_dir(&source).unwrap();
            let module = installer
                .v1_root()
                .join("objects")
                .join(admission.content_digest())
                .join(MODULE);
            (installer, admission, module)
        }

        let temp = TempDir::new().unwrap();
        let (length_installer, length_admission, length_module) =
            install_case(&temp, "length", 18);
        let mut longer = fs::read(&length_module).unwrap();
        longer.push(0);
        fs::write(&length_module, longer).unwrap();
        assert!(
            length_installer
                .load_admitted_pack(length_admission.install_digest())
                .unwrap_err()
                .contains("length violates manifest or policy")
        );

        let (hash_installer, hash_admission, hash_module) =
            install_case(&temp, "digest", 19);
        let mut changed = fs::read(&hash_module).unwrap();
        changed[0] ^= 0xff;
        fs::write(&hash_module, changed).unwrap();
        assert!(
            hash_installer
                .load_admitted_pack(hash_admission.install_digest())
                .unwrap_err()
                .contains("digest mismatch")
        );
    }

    #[test]
    fn load_admitted_pack_rechecks_current_trust_and_revocation() {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join("source");
        let cache = temp.path().join("cache");
        fs::create_dir(&source).unwrap();
        let key = SigningKey::from_bytes(&[20; 32]);
        make_pack(&source, &key, "1.0.0");
        let mut installer = trusted_installer(&cache, &key);
        let admission = installer.install_from_staged_dir(&source).unwrap();
        assert!(
            installer
                .load_admitted_pack(admission.install_digest())
                .is_ok()
        );

        let untrusted =
            Installer::new(&cache, TrustStore::new(), ExtensionRegistry::new())
                .unwrap();
        assert!(
            untrusted
                .load_admitted_pack(admission.install_digest())
                .unwrap_err()
                .contains("unknown Program Pack signing key")
        );

        installer.trust_mut().revoke(KEY_ID).unwrap();
        assert!(
            installer
                .load_admitted_pack(admission.install_digest())
                .unwrap_err()
                .contains("revoked Program Pack signing key")
        );
    }

    #[test]
    fn unsigned_truncated_wrong_hash_and_noncanonical_signature_fail_closed() {
        let temp = TempDir::new().unwrap();
        let key = SigningKey::from_bytes(&[12; 32]);
        let installer = trusted_installer(&temp.path().join("cache"), &key);

        let unsigned = temp.path().join("unsigned");
        fs::create_dir(&unsigned).unwrap();
        make_pack(&unsigned, &key, "1.0.0");
        fs::write(unsigned.join(SIGNATURE_FILE), b"").unwrap();
        assert!(installer.install_from_staged_dir(&unsigned).is_err());

        let truncated = temp.path().join("truncated");
        fs::create_dir(&truncated).unwrap();
        make_pack(&truncated, &key, "1.0.1");
        fs::write(truncated.join(MANIFEST_FILE), b"{").unwrap();
        assert!(installer.install_from_staged_dir(&truncated).is_err());

        let wrong_hash = temp.path().join("wrong-hash");
        fs::create_dir(&wrong_hash).unwrap();
        make_pack(&wrong_hash, &key, "1.0.2");
        fs::write(wrong_hash.join(MODULE), b"wrong-bytes").unwrap();
        assert!(installer.install_from_staged_dir(&wrong_hash).is_err());

        let noncanonical = temp.path().join("noncanonical");
        fs::create_dir(&noncanonical).unwrap();
        make_pack(&noncanonical, &key, "1.0.3");
        let raw = fs::read(noncanonical.join(SIGNATURE_FILE)).unwrap();
        let mut padded = vec![b' '];
        padded.extend(raw);
        fs::write(noncanonical.join(SIGNATURE_FILE), padded).unwrap();
        assert!(installer.install_from_staged_dir(&noncanonical).is_err());
    }

    #[test]
    fn channel_retains_one_known_good_and_rolls_back() {
        let temp = TempDir::new().unwrap();
        let key = SigningKey::from_bytes(&[13; 32]);
        let mut installer = trusted_installer(&temp.path().join("cache"), &key);
        let first_dir = temp.path().join("first");
        let second_dir = temp.path().join("second");
        fs::create_dir(&first_dir).unwrap();
        fs::create_dir(&second_dir).unwrap();
        make_pack(&first_dir, &key, "1.0.0");
        make_pack(&second_dir, &key, "1.0.1");
        let first = installer.install_from_staged_dir(&first_dir).unwrap();
        let second = installer.install_from_staged_dir(&second_dir).unwrap();
        assert!(
            installer
                .advance_channel("imparo.community.beta", &first)
                .is_err()
        );
        let one = installer.advance_channel(CHANNEL, &first).unwrap();
        assert_eq!(one.generation, 1);
        assert_eq!(one.current.as_deref(), Some(first.install_digest()));
        assert_eq!(one.known_good, None);
        let two = installer.advance_channel(CHANNEL, &second).unwrap();
        assert_eq!(two.generation, 2);
        assert_eq!(two.current.as_deref(), Some(second.install_digest()));
        assert_eq!(two.known_good.as_deref(), Some(first.install_digest()));
        let three = installer.rollback_channel(CHANNEL).unwrap();
        assert_eq!(three.generation, 3);
        assert_eq!(three.current.as_deref(), Some(first.install_digest()));
        assert_eq!(three.known_good.as_deref(), Some(second.install_digest()));
        assert_eq!(installer.channel_state(CHANNEL).unwrap(), three);
        installer.trust_mut().revoke(KEY_ID).unwrap();
        assert!(installer.channel_state(CHANNEL).is_err());
    }

    #[test]
    fn traversal_oversize_and_concurrent_installs_fail_or_converge_safely() {
        for path in [
            "../escape.cubin",
            "/absolute.cubin",
            "modules\\escape.cubin",
            "C:escape",
        ] {
            assert!(path_from_manifest(path).is_err());
        }
        let temp = TempDir::new().unwrap();
        let too_large = temp.path().join("large");
        fs::write(&too_large, b"12345").unwrap();
        assert!(read_bounded_secure(&too_large, 4).is_err());

        let extra_dir = temp.path().join("extra-dir");
        fs::create_dir(&extra_dir).unwrap();
        let extra_key = SigningKey::from_bytes(&[16; 32]);
        make_pack(&extra_dir, &extra_key, "1.0.0");
        fs::create_dir(extra_dir.join("unexpected-empty-directory")).unwrap();
        assert!(
            trusted_installer(&temp.path().join("extra-cache"), &extra_key)
                .install_from_staged_dir(&extra_dir)
                .is_err()
        );

        let source = temp.path().join("race-source");
        fs::create_dir(&source).unwrap();
        let key = SigningKey::from_bytes(&[14; 32]);
        make_pack(&source, &key, "1.0.0");
        let installer =
            std::sync::Arc::new(trusted_installer(&temp.path().join("cache"), &key));
        let left = std::sync::Arc::clone(&installer);
        let right = std::sync::Arc::clone(&installer);
        let left_source = source.clone();
        let right_source = source.clone();
        let a = std::thread::spawn(move || {
            left.install_from_staged_dir(&left_source).unwrap()
        });
        let b = std::thread::spawn(move || {
            right.install_from_staged_dir(&right_source).unwrap()
        });
        let a = a.join().unwrap();
        let b = b.join().unwrap();
        assert_eq!(a.content_digest(), b.content_digest());
        assert_eq!(a.install_digest(), b.install_digest());
        assert!(installer.load_admission(a.install_digest()).is_ok());
    }

    #[test]
    fn incomplete_channel_transaction_recovers_both_pointers_to_before_state() {
        let temp = TempDir::new().unwrap();
        let key = SigningKey::from_bytes(&[15; 32]);
        let installer = trusted_installer(&temp.path().join("cache"), &key);
        let first_dir = temp.path().join("journal-first");
        let second_dir = temp.path().join("journal-second");
        fs::create_dir(&first_dir).unwrap();
        fs::create_dir(&second_dir).unwrap();
        make_pack(&first_dir, &key, "1.0.0");
        make_pack(&second_dir, &key, "1.0.1");
        let first = installer.install_from_staged_dir(&first_dir).unwrap();
        let second = installer.install_from_staged_dir(&second_dir).unwrap();
        installer.advance_channel(CHANNEL, &first).unwrap();
        let before = installer.advance_channel(CHANNEL, &second).unwrap();
        let channel_dir = installer.v1_root().join("channels").join(CHANNEL);
        let before_current =
            read_pointer_optional(&channel_dir.join("current.json"), &installer.policy)
                .unwrap();
        let before_known = read_pointer_optional(
            &channel_dir.join("known-good.json"),
            &installer.policy,
        )
        .unwrap();
        let interrupted = ChannelPointer {
            schema: CHANNEL_SCHEMA,
            generation: before.generation + 1,
            install_digest: first.install_digest().to_owned(),
        };
        let interrupted_known = ChannelPointer {
            schema: CHANNEL_SCHEMA,
            generation: before.generation + 1,
            install_digest: second.install_digest().to_owned(),
        };
        let tx = ChannelTransaction {
            schema: CHANNEL_SCHEMA,
            committed: false,
            before_current: before_current.clone(),
            before_known_good: before_known.clone(),
            after_current: Some(interrupted.clone()),
            after_known_good: Some(interrupted_known),
        };
        write_transaction(&channel_dir, &tx, &installer.policy).unwrap();
        write_pointer_optional(&channel_dir.join("current.json"), Some(&interrupted))
            .unwrap();
        let recovered = installer.channel_state(CHANNEL).unwrap();
        assert_eq!(recovered, before);
        assert!(!channel_dir.join(TRANSACTION_FILE).exists());

        let committed_current = ChannelPointer {
            schema: CHANNEL_SCHEMA,
            generation: before.generation + 1,
            install_digest: first.install_digest().to_owned(),
        };
        let committed_known = ChannelPointer {
            schema: CHANNEL_SCHEMA,
            generation: before.generation + 1,
            install_digest: second.install_digest().to_owned(),
        };
        let committed = ChannelTransaction {
            schema: CHANNEL_SCHEMA,
            committed: true,
            before_current: read_pointer_optional(
                &channel_dir.join("current.json"),
                &installer.policy,
            )
            .unwrap(),
            before_known_good: read_pointer_optional(
                &channel_dir.join("known-good.json"),
                &installer.policy,
            )
            .unwrap(),
            after_current: Some(committed_current),
            after_known_good: Some(committed_known),
        };
        write_transaction(&channel_dir, &committed, &installer.policy).unwrap();
        let recovered_committed = installer.channel_state(CHANNEL).unwrap();
        assert_eq!(recovered_committed.generation, before.generation + 1);
        assert_eq!(
            recovered_committed.current.as_deref(),
            Some(first.install_digest())
        );
        assert_eq!(
            recovered_committed.known_good.as_deref(),
            Some(second.install_digest())
        );
    }
}
