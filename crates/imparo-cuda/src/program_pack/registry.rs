//! Transactional process catalog and backend-generic built-in fallback exposure.

use super::constraints::{
    AdapterAuthority, RuntimeEligibility, compiled_adapters, decode_hex_32,
    validate_pack_runtime,
};
use super::loader::{choice_group_identity, eligible_variants, install_native};
use crate::ffi::{
    imparo_cuda_program_bind, imparo_cuda_program_freeze, imparo_cuda_program_reset,
};
use imparo_backend::{
    DeviceProfile, ModelFacts, ProgramCandidateDecl, ProgramChoiceDecl,
    ProgramProvider, Workload,
};
use imparo_program_pack::Installer;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};

#[derive(Clone)]
struct ChoiceState {
    decl: ProgramChoiceDecl,
    current: [u8; 32],
    builtin: [u8; 32],
    variants: BTreeMap<[u8; 32], VariantState>,
}

#[derive(Clone)]
struct VariantState {
    manifest: imparo_program_pack::manifest::Variant,
    authority: AdapterAuthority,
}

#[allow(dead_code)] // Materialized by the first compiled adapter and gated smoke.
pub(crate) struct SelectedVariant {
    pub manifest: imparo_program_pack::manifest::Variant,
    pub authority: AdapterAuthority,
}

#[derive(Clone)]
struct State {
    frozen: bool,
    native_installed: bool,
    catalog_healthy: bool,
    installed: BTreeSet<String>,
    choices: BTreeMap<String, ChoiceState>,
    last_query_eligible: BTreeSet<(String, [u8; 32])>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            frozen: false,
            native_installed: false,
            catalog_healthy: true,
            installed: BTreeSet::new(),
            choices: BTreeMap::new(),
            last_query_eligible: BTreeSet::new(),
        }
    }
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

fn state() -> &'static Mutex<State> {
    STATE.get_or_init(|| Mutex::new(State::default()))
}

pub struct InstallReport {
    pub eligible_variants: usize,
    pub native_fallback_groups: usize,
}

pub fn install(
    installer: &Installer,
    install_digest: &str,
) -> Result<InstallReport, String> {
    let identity = crate::runtime_identity()?;
    let runtime = RuntimeEligibility {
        device_sm: identity.device_sm,
        driver_version: identity.driver_version,
        backend_abi: identity.backend_abi,
        math_mode: match crate::CUDA_MATH_MODE {
            "precise" => imparo_program_pack::manifest::MathMode::Strict,
            _ => imparo_program_pack::manifest::MathMode::Fast,
        },
    };
    install_with_authorities(installer, install_digest, &runtime, compiled_adapters())
}

#[cfg(test)]
#[allow(dead_code)] // Used by the separately cfg-gated real SM86 smoke.
pub(crate) fn install_for_test(
    installer: &Installer,
    install_digest: &str,
    runtime: &RuntimeEligibility,
    authorities: &[AdapterAuthority],
) -> Result<InstallReport, String> {
    install_with_authorities(installer, install_digest, runtime, authorities)
}

fn install_with_authorities(
    installer: &Installer,
    install_digest: &str,
    runtime: &RuntimeEligibility,
    authorities: &[AdapterAuthority],
) -> Result<InstallReport, String> {
    let pack = installer.load_admitted_pack(install_digest)?;
    validate_pack_runtime(pack.manifest(), runtime)?;
    let eligible = eligible_variants(&pack, authorities)?;

    let mut guard = state()
        .lock()
        .map_err(|_| "Program registry lock poisoned")?;
    if guard.frozen {
        return Err("Program catalog is frozen".into());
    }
    if !guard.catalog_healthy {
        return Err("Program catalog health check failed; reset is required".into());
    }
    if guard.installed.contains(install_digest) {
        return Ok(InstallReport {
            eligible_variants: eligible.len(),
            native_fallback_groups: eligible
                .iter()
                .map(|item| item.variant.choice_group_id.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
        });
    }
    let mut staged = guard.clone();

    let eligible_by_group =
        eligible
            .iter()
            .fold(BTreeMap::<&str, Vec<_>>::new(), |mut map, item| {
                map.entry(item.variant.choice_group_id.as_str())
                    .or_default()
                    .push(item);
                map
            });
    for group in pack.manifest().choice_groups() {
        let Some(candidates) = eligible_by_group.get(group.choice_group_id.as_str())
        else {
            continue;
        };
        let authority = candidates[0].authority;
        if candidates.iter().any(|candidate| {
            candidate.authority.builtin_variant_id != authority.builtin_variant_id
                || candidate.authority.builtin_config_id != authority.builtin_config_id
        }) {
            return Err(
                "Program choice group has inconsistent built-in authority".into()
            );
        }
        let builtin = authority.builtin_variant_id;
        let mut declarations = vec![builtin_candidate(authority)];
        let mut variants = BTreeMap::new();
        for item in candidates {
            let variant_id = decode_hex_32(&item.variant.variant_id)?;
            declarations.push(ProgramCandidateDecl {
                variant_id,
                config_id: decode_hex_32(&item.variant.config_id)?,
                provider: ProgramProvider::ProgramPack,
            });
            variants.insert(
                variant_id,
                VariantState {
                    manifest: item.variant.clone(),
                    authority: item.authority.clone(),
                },
            );
        }
        let decl = ProgramChoiceDecl {
            choice_group_id: group.choice_group_id.clone(),
            candidates: declarations,
            workload: workload(&group.workload)?,
            cross_check: group.cross_check.as_ref().map(workload).transpose()?,
            screened: group.screened,
            bit_affecting: group.bit_affecting,
            joint_with: group.joint_with.clone(),
        };
        match staged.choices.get_mut(&group.choice_group_id) {
            Some(existing) => {
                let same_contract = existing.decl.choice_group_id
                    == decl.choice_group_id
                    && existing.decl.workload == decl.workload
                    && existing.decl.cross_check == decl.cross_check
                    && existing.decl.screened == decl.screened
                    && existing.decl.bit_affecting == decl.bit_affecting
                    && existing.decl.joint_with == decl.joint_with;
                if !same_contract {
                    return Err(
                        "Program choice group collides with a different declaration"
                            .into(),
                    );
                }
                if existing.builtin != builtin {
                    return Err(
                        "Program choice group changes built-in authority".into()
                    );
                }
                for candidate in decl.candidates.into_iter().skip(1) {
                    if existing
                        .decl
                        .candidates
                        .iter()
                        .any(|prior| prior.variant_id == candidate.variant_id)
                    {
                        return Err(
                            "Program variant identity collides across packs".into()
                        );
                    }
                    existing.decl.candidates.push(candidate);
                }
                existing.variants.extend(variants);
            }
            None => {
                staged.choices.insert(
                    group.choice_group_id.clone(),
                    ChoiceState {
                        decl,
                        current: builtin,
                        builtin,
                        variants,
                    },
                );
            }
        }
    }
    // Native install is the final fallible operation and is itself transactional.
    // Publish the staged Rust view only after the Driver registry accepted it.
    if !eligible.is_empty() {
        install_native(&pack, &eligible)?;
    }
    staged.native_installed |= !eligible.is_empty();
    staged.installed.insert(install_digest.to_owned());
    *guard = staged;
    Ok(InstallReport {
        eligible_variants: eligible.len(),
        native_fallback_groups: eligible_by_group.len(),
    })
}

pub(crate) fn choices(
    facts: &ModelFacts,
    profile: &DeviceProfile,
) -> Vec<ProgramChoiceDecl> {
    state()
        .lock()
        .map(|mut guard| {
            if !guard.catalog_healthy {
                return Vec::new();
            }
            let mut eligible = BTreeSet::new();
            let declarations = guard
                .choices
                .iter()
                .map(|(group, choice)| {
                    let mut decl = choice.decl.clone();
                    decl.candidates.retain(|candidate| {
                        if candidate.provider == ProgramProvider::BuiltIn {
                            return true;
                        }
                        let applies = choice
                            .variants
                            .get(&candidate.variant_id)
                            .map(|variant| {
                                candidate_applicable(variant, facts, profile)
                            })
                            .unwrap_or(false);
                        if applies {
                            eligible.insert((group.clone(), candidate.variant_id));
                        }
                        applies
                    });
                    decl
                })
                .collect();
            guard.last_query_eligible = eligible;
            declarations
        })
        .unwrap_or_default()
}

pub(crate) fn freeze() -> Result<(), String> {
    let mut guard = state()
        .lock()
        .map_err(|_| "Program registry lock poisoned")?;
    if guard.frozen {
        return Ok(());
    }
    if !guard.catalog_healthy {
        return Err("Program catalog health check failed; reset is required".into());
    }
    if guard.native_installed {
        // SAFETY: no arguments; native performs its own lifecycle checks.
        let rc = unsafe { imparo_cuda_program_freeze() };
        if rc != 0 {
            return Err(format!(
                "native Program catalog freeze failed with code {rc}"
            ));
        }
    }
    guard.frozen = true;
    Ok(())
}

pub(crate) fn bind(group: &str, variant: &[u8; 32]) -> Result<(), String> {
    let mut guard = state()
        .lock()
        .map_err(|_| "Program registry lock poisoned")?;
    if !guard.frozen {
        return Err("Program catalog must be frozen before binding".into());
    }
    if !guard.catalog_healthy {
        return Err("Program catalog health check failed; reset is required".into());
    }
    let applicable = guard
        .last_query_eligible
        .contains(&(group.to_owned(), *variant));
    let choice = guard
        .choices
        .get_mut(group)
        .ok_or_else(|| "unknown Program choice group".to_string())?;
    if !choice
        .decl
        .candidates
        .iter()
        .any(|candidate| &candidate.variant_id == variant)
    {
        return Err("Program variant is not eligible for this choice group".into());
    }
    if variant != &choice.builtin && !applicable {
        return Err(
            "Program variant is not applicable to the current model/device".into(),
        );
    }
    if &choice.current == variant {
        return Ok(());
    }
    // Native accepts an unregistered, nonzero variant only as the safe built-in
    // fallback marker; it can never be launched as a Program function.
    let group_id = choice_group_identity(group);
    let rc = unsafe { imparo_cuda_program_bind(group_id.as_ptr(), variant.as_ptr()) };
    if rc != 0 {
        return Err(format!("native Program bind failed with code {rc}"));
    }
    choice.current = *variant;
    Ok(())
}

pub(crate) fn current(group: &str) -> Option<[u8; 32]> {
    let guard = state().lock().ok()?;
    let choice = guard.choices.get(group)?;
    Some(visible_current(&guard, choice))
}

#[allow(dead_code)] // Called only by compiled adapters and the gated SM86 smoke today.
pub(crate) fn selected_variant(group: &str) -> Result<Option<SelectedVariant>, String> {
    let guard = state()
        .lock()
        .map_err(|_| "Program registry lock poisoned")?;
    let choice = guard
        .choices
        .get(group)
        .ok_or_else(|| "unknown Program choice group".to_string())?;
    if !guard.catalog_healthy {
        return Err("Program catalog health check failed; reset is required".into());
    }
    if choice.current == choice.builtin {
        Ok(None)
    } else {
        choice
            .variants
            .get(&choice.current)
            .cloned()
            .map(|variant| {
                Some(SelectedVariant {
                    manifest: variant.manifest,
                    authority: variant.authority,
                })
            })
            .ok_or_else(|| "bound Program variant is absent".into())
    }
}

pub(crate) fn catalog_identity() -> imparo_backend::ProgramCatalogIdentity {
    let Ok(mut guard) = state().lock() else {
        return super::profile_identity::pending_identity();
    };
    if guard.native_installed
        && super::profile_identity::query_native_catalog().is_err()
    {
        guard.catalog_healthy = false;
        guard.last_query_eligible.clear();
        for choice in guard.choices.values_mut() {
            choice.current = choice.builtin;
        }
    }
    super::profile_identity::pending_identity()
}

pub(crate) fn shutdown() -> Result<(), String> {
    let mut guard = state()
        .lock()
        .map_err(|_| "Program registry lock poisoned")?;
    // Holding the Rust registry lock prevents a concurrent bind/install from observing
    // native reset without the matching Rust reset.
    let rc = unsafe { imparo_cuda_program_reset() };
    if rc != 0 {
        return Err(format!("native Program shutdown failed with code {rc}"));
    }
    reset_state(&mut guard);
    Ok(())
}

fn builtin_candidate(authority: &AdapterAuthority) -> ProgramCandidateDecl {
    ProgramCandidateDecl {
        variant_id: authority.builtin_variant_id,
        config_id: authority.builtin_config_id,
        provider: ProgramProvider::BuiltIn,
    }
}

fn candidate_applicable(
    variant: &VariantState,
    facts: &ModelFacts,
    profile: &DeviceProfile,
) -> bool {
    let facts_complete = facts.n_embd != 0
        && facts.n_ff != 0
        && facts.n_head != 0
        && facts.n_kv != 0
        && facts.head_dim != 0
        && facts.n_layers != 0
        && facts.weight_kinds != 0;
    let resources = &variant.manifest.resources;
    let profile_complete = profile.max_threads != 0
        && profile.threadgroup_bytes != 0
        && resources.threads_per_block_max <= profile.max_threads
        && u64::from(resources.static_shared_bytes_max)
            + u64::from(resources.dynamic_shared_bytes_max)
            <= profile.threadgroup_bytes;
    facts_complete
        && profile_complete
        && (variant.authority.applies)(&variant.manifest, facts, profile)
}

fn visible_current(state: &State, choice: &ChoiceState) -> [u8; 32] {
    if state.catalog_healthy {
        choice.current
    } else {
        choice.builtin
    }
}

fn reset_state(state: &mut State) {
    *state = State::default();
}

fn workload(
    value: &imparo_program_pack::manifest::Workload,
) -> Result<Workload, String> {
    use imparo_program_pack::manifest::WorkloadId as Id;
    Ok(match value.workload_id {
        Id::DecodeMix => Workload::DecodeMix,
        Id::NarrowMix => Workload::NarrowMix(
            value
                .parameters
                .narrow_tokens
                .ok_or("narrow_mix has no token count")?,
        ),
        Id::AttentionDecode => Workload::AttentionDecode,
        Id::AttentionDecodeDeep => Workload::AttentionDecodeDeep,
        Id::AttentionPrefill => Workload::AttentionPrefill,
        Id::AttentionPrefillDeep => Workload::AttentionPrefillDeep,
        Id::DecodeAttentionStep => Workload::DecodeAttentionStep,
        Id::AttentionPrefillReuse => Workload::AttentionPrefillReuse,
        Id::PrefillGemm => Workload::PrefillGemm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> ModelFacts {
        ModelFacts {
            n_embd: 1024,
            n_ff: 4096,
            n_head: 8,
            n_kv: 4,
            head_dim: 128,
            deep_head_dim: 128,
            n_experts: 0,
            n_layers: 16,
            layer_dispatches: 8,
            weight_kinds: 1,
        }
    }

    fn profile() -> DeviceProfile {
        DeviceProfile {
            threadgroup_bytes: 64 * 1024,
            max_threads: 1024,
            ..DeviceProfile::default()
        }
    }

    fn only_a(
        variant: &imparo_program_pack::manifest::Variant,
        _: &ModelFacts,
        _: &DeviceProfile,
    ) -> bool {
        variant.variant_id.starts_with('a')
    }

    fn choice(authority: &AdapterAuthority) -> ChoiceState {
        let manifest = super::super::constraints::tests::manifest();
        let mut variant = manifest.variants()[0].clone();
        variant.variant_id =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        let variant_id = decode_hex_32(&variant.variant_id).unwrap();
        let mut authority = authority.clone();
        authority.applies = only_a;
        ChoiceState {
            decl: ProgramChoiceDecl {
                choice_group_id: authority.choice_group_id.clone(),
                candidates: vec![
                    builtin_candidate(&authority),
                    ProgramCandidateDecl {
                        variant_id,
                        config_id: [5; 32],
                        provider: ProgramProvider::ProgramPack,
                    },
                ],
                workload: Workload::AttentionDecode,
                cross_check: None,
                screened: true,
                bit_affecting: false,
                joint_with: vec![],
            },
            current: variant_id,
            builtin: authority.builtin_variant_id,
            variants: BTreeMap::from([(
                variant_id,
                VariantState {
                    manifest: variant,
                    authority,
                },
            )]),
        }
    }

    #[test]
    fn builtin_identity_is_exactly_the_compiled_authority() {
        let manifest = super::super::constraints::tests::manifest();
        let authority = super::super::constraints::tests::authority(&manifest);
        let candidate = builtin_candidate(&authority);
        assert_eq!(candidate.variant_id, [8; 32]);
        assert_eq!(candidate.config_id, [9; 32]);
        assert_eq!(candidate.provider, ProgramProvider::BuiltIn);
    }

    #[test]
    fn applicability_requires_complete_facts_profile_and_the_exact_variant() {
        let manifest = super::super::constraints::tests::manifest();
        let authority = super::super::constraints::tests::authority(&manifest);
        let choice = choice(&authority);
        let variant = choice.variants.values().next().unwrap();
        assert!(candidate_applicable(variant, &facts(), &profile()));
        let mut missing = facts();
        missing.n_embd = 0;
        assert!(!candidate_applicable(variant, &missing, &profile()));
        let mut wrong = variant.clone();
        wrong.manifest.variant_id =
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into();
        assert!(!candidate_applicable(&wrong, &facts(), &profile()));
        assert!(!candidate_applicable(
            variant,
            &facts(),
            &DeviceProfile::default()
        ));
    }

    #[test]
    fn unhealthy_catalog_hides_stale_external_route() {
        let manifest = super::super::constraints::tests::manifest();
        let authority = super::super::constraints::tests::authority(&manifest);
        let choice = choice(&authority);
        let mut state = State::default();
        assert_ne!(choice.current, choice.builtin);
        assert_eq!(visible_current(&state, &choice), choice.current);
        state.catalog_healthy = false;
        assert_eq!(visible_current(&state, &choice), choice.builtin);
    }

    #[test]
    fn rust_registry_reset_is_repeatable_for_teardown_smoke() {
        let manifest = super::super::constraints::tests::manifest();
        let authority = super::super::constraints::tests::authority(&manifest);
        let group = authority.choice_group_id.clone();
        let mut state = State::default();
        for iteration in 0..1_000 {
            state.frozen = true;
            state.native_installed = true;
            state.installed.insert(format!("pack-{iteration}"));
            state.choices.insert(group.clone(), choice(&authority));
            reset_state(&mut state);
            assert!(!state.frozen);
            assert!(!state.native_installed);
            assert!(state.catalog_healthy);
            assert!(state.installed.is_empty());
            assert!(state.choices.is_empty());
            assert!(state.last_query_eligible.is_empty());
        }
    }
}
