//! Explicit extension admission. Unknown optional extensions disable only their variants.

use crate::manifest::Manifest;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default)]
pub struct ExtensionRegistry {
    revisions: BTreeMap<String, BTreeSet<u32>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExtensionResolution {
    pub disabled_variants: BTreeSet<String>,
}

impl ExtensionRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, id: impl Into<String>, revision: u32) -> Result<(), String> {
        let id = id.into();
        crate::trust::validate_namespaced_id(&id)?;
        if revision == 0 {
            return Err("extension revision must be nonzero".into());
        }
        let limit = crate::Policy::embedded()?.limits.extension_max_count;
        let total: usize = self.revisions.values().map(BTreeSet::len).sum();
        if total >= limit {
            return Err("extension registry exceeds policy limit".into());
        }
        if !self.revisions.entry(id).or_default().insert(revision) {
            return Err("duplicate extension registry entry".into());
        }
        Ok(())
    }

    pub fn resolve(&self, manifest: &Manifest) -> Result<ExtensionResolution, String> {
        manifest.validate().map_err(|error| error.to_string())?;
        for extension in &manifest.required_extensions {
            if !self.supports(&extension.id, extension.revision) {
                return Err(format!(
                    "unknown required Program Pack extension {} revision {}",
                    extension.id, extension.revision
                ));
            }
        }
        let optional: BTreeMap<_, _> = manifest
            .optional_extensions
            .iter()
            .map(|extension| (extension.id.as_str(), extension.revision))
            .collect();
        let disabled_variants = manifest
            .variants
            .iter()
            .filter(|variant| {
                variant.extension.as_ref().is_some_and(|extension| {
                    optional.get(extension.id.as_str()) == Some(&extension.revision)
                        && !self.supports(&extension.id, extension.revision)
                })
            })
            .map(|variant| variant.variant_id.clone())
            .collect();
        Ok(ExtensionResolution { disabled_variants })
    }

    fn supports(&self, id: &str, revision: u32) -> bool {
        self.revisions
            .get(id)
            .is_some_and(|revisions| revisions.contains(&revision))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Manifest {
        Manifest::parse(include_bytes!(
            "../../../schemas/examples/program-pack-v1.minimal.json"
        ))
        .unwrap()
    }

    #[test]
    fn unknown_required_fails_while_unknown_optional_disables_only_its_variant() {
        let mut required = fixture();
        required
            .required_extensions
            .push(crate::manifest::Extension {
                id: "imparo.extension.required".into(),
                revision: 1,
            });
        assert!(ExtensionRegistry::new().resolve(&required).is_err());

        let mut optional = fixture();
        let extension = crate::manifest::Extension {
            id: "imparo.extension.optional".into(),
            revision: 1,
        };
        optional.optional_extensions.push(extension.clone());
        optional.variants[0].extension = Some(extension.clone());
        optional.validate().unwrap();
        let disabled = ExtensionRegistry::new().resolve(&optional).unwrap();
        assert_eq!(disabled.disabled_variants.len(), 1);
        let mut known = ExtensionRegistry::new();
        known.add(extension.id, extension.revision).unwrap();
        assert!(
            known
                .resolve(&optional)
                .unwrap()
                .disabled_variants
                .is_empty()
        );
    }

    #[test]
    fn registry_rejects_non_namespaced_or_ambiguous_ids() {
        for id in ["plain", "Imparo.extension", "imparo..extension"] {
            assert!(ExtensionRegistry::new().add(id, 1).is_err());
        }
    }
}
