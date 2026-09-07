//! Centralized, versioned safety ceilings.

use serde::Deserialize;

const DEFAULT_POLICY: &str = include_str!("../../../program-pack-policy.toml");

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub format: FormatPolicy,
    pub limits: Limits,
    pub search: SearchPolicy,
    pub fallback: FallbackPolicy,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormatPolicy {
    pub policy_version: u32,
    pub program_pack_abi: u32,
    pub kernel_contract_abi: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub manifest_max_bytes: u64,
    pub signature_envelope_max_bytes: u64,
    pub pack_max_bytes: u64,
    pub module_max_bytes: u64,
    pub sbom_max_bytes: u64,
    pub notices_max_bytes: u64,
    pub provenance_max_bytes: u64,
    pub channel_pointer_max_bytes: u64,
    pub channel_transaction_max_bytes: u64,
    pub json_depth_max: usize,
    pub json_node_max: usize,
    pub trust_key_max_count: usize,
    pub module_max_count: usize,
    pub choice_group_max_count: usize,
    pub variant_max_count: usize,
    pub extension_max_count: usize,
    pub string_max_bytes: usize,
    pub symbol_max_bytes: usize,
    pub launch_argument_max_count: usize,
    pub constraint_max_count: usize,
    pub effect_max_count: usize,
    pub scratch_max_bytes: u64,
    pub dynamic_shared_memory_max_bytes: u64,
    pub block_threads_max: u32,
    pub grid_x_max: u64,
    pub grid_y_max: u64,
    pub grid_z_max: u64,
    pub launch_expression_depth_max: usize,
    pub launch_expression_nodes_max: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchPolicy {
    pub screen_top_k: usize,
    pub benchmark_warmup_iterations: usize,
    pub benchmark_measure_iterations: usize,
    pub minimum_relative_improvement: f64,
    pub maximum_regression_relative: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)]
pub struct FallbackPolicy {
    pub native_required: bool,
    pub fail_closed: bool,
    pub retain_known_good_versions: usize,
    pub allow_forward_hot_swap: bool,
    pub telemetry_enabled: bool,
    pub network_during_inference: bool,
}

impl Policy {
    #[cfg(test)]
    fn parse(text: &str) -> Result<Self, String> {
        let policy = Self::parse_unbounded(text)?;
        policy.require_within(&Self::parse_unbounded(DEFAULT_POLICY)?)?;
        Ok(policy)
    }

    pub fn embedded() -> Result<Self, String> {
        Self::parse_unbounded(DEFAULT_POLICY)
    }

    fn parse_unbounded(text: &str) -> Result<Self, String> {
        let policy: Self = toml::from_str(text)
            .map_err(|error| format!("parse Program Pack policy: {error}"))?;
        policy.validate()?;
        Ok(policy)
    }

    #[cfg(test)]
    fn require_within(&self, hard: &Self) -> Result<(), String> {
        let c = &self.limits;
        let h = &hard.limits;
        let candidate = [
            c.manifest_max_bytes,
            c.signature_envelope_max_bytes,
            c.pack_max_bytes,
            c.module_max_bytes,
            c.sbom_max_bytes,
            c.notices_max_bytes,
            c.provenance_max_bytes,
            c.channel_pointer_max_bytes,
            c.channel_transaction_max_bytes,
            c.json_depth_max as u64,
            c.json_node_max as u64,
            c.trust_key_max_count as u64,
            c.module_max_count as u64,
            c.choice_group_max_count as u64,
            c.variant_max_count as u64,
            c.extension_max_count as u64,
            c.string_max_bytes as u64,
            c.symbol_max_bytes as u64,
            c.launch_argument_max_count as u64,
            c.constraint_max_count as u64,
            c.effect_max_count as u64,
            c.scratch_max_bytes,
            c.dynamic_shared_memory_max_bytes,
            c.block_threads_max as u64,
            c.grid_x_max,
            c.grid_y_max,
            c.grid_z_max,
            c.launch_expression_depth_max as u64,
            c.launch_expression_nodes_max as u64,
        ];
        let ceiling = [
            h.manifest_max_bytes,
            h.signature_envelope_max_bytes,
            h.pack_max_bytes,
            h.module_max_bytes,
            h.sbom_max_bytes,
            h.notices_max_bytes,
            h.provenance_max_bytes,
            h.channel_pointer_max_bytes,
            h.channel_transaction_max_bytes,
            h.json_depth_max as u64,
            h.json_node_max as u64,
            h.trust_key_max_count as u64,
            h.module_max_count as u64,
            h.choice_group_max_count as u64,
            h.variant_max_count as u64,
            h.extension_max_count as u64,
            h.string_max_bytes as u64,
            h.symbol_max_bytes as u64,
            h.launch_argument_max_count as u64,
            h.constraint_max_count as u64,
            h.effect_max_count as u64,
            h.scratch_max_bytes,
            h.dynamic_shared_memory_max_bytes,
            h.block_threads_max as u64,
            h.grid_x_max,
            h.grid_y_max,
            h.grid_z_max,
            h.launch_expression_depth_max as u64,
            h.launch_expression_nodes_max as u64,
        ];
        if candidate
            .iter()
            .zip(ceiling)
            .any(|(value, ceiling)| *value > ceiling)
        {
            return Err(
                "Program Pack policy cannot raise a built-in hard ceiling".into()
            );
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), String> {
        if self.format.policy_version != 1
            || self.format.program_pack_abi != crate::PROGRAM_PACK_ABI
            || self.format.kernel_contract_abi != 1
        {
            return Err("unsupported Program Pack policy format or ABI".into());
        }
        if self.limits.manifest_max_bytes == 0
            || self.limits.module_max_bytes == 0
            || self.limits.signature_envelope_max_bytes == 0
            || self.limits.channel_pointer_max_bytes == 0
            || self.limits.channel_transaction_max_bytes
                < self.limits.channel_pointer_max_bytes
            || self.limits.json_depth_max == 0
            || self.limits.json_node_max == 0
            || self.limits.trust_key_max_count == 0
            || self.limits.pack_max_bytes < self.limits.module_max_bytes
            || self.limits.module_max_count == 0
            || self.limits.variant_max_count == 0
            || self.limits.launch_expression_depth_max == 0
            || self.limits.launch_expression_nodes_max == 0
        {
            return Err(
                "Program Pack policy contains an empty or inconsistent safety ceiling"
                    .into(),
            );
        }
        if !self.fallback.native_required
            || !self.fallback.fail_closed
            || self.fallback.retain_known_good_versions != 1
            || self.fallback.allow_forward_hot_swap
            || self.fallback.telemetry_enabled
            || self.fallback.network_during_inference
        {
            return Err(
                "Program Pack v1 fallback/privacy invariants were weakened".into()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracked_policy_is_strict_and_fail_closed() {
        let policy = Policy::embedded().unwrap();
        assert_eq!(policy.limits.module_max_count, 64);
        assert_eq!(policy.limits.channel_transaction_max_bytes, 4096);
        assert!(
            Policy::parse(
                &DEFAULT_POLICY.replace("fail_closed = true", "fail_closed = false")
            )
            .is_err()
        );
        assert!(Policy::parse(&format!("{DEFAULT_POLICY}\nunknown = 1\n")).is_err());
        for replacement in [
            "channel_transaction_max_bytes = 0",
            "channel_transaction_max_bytes = 512",
        ] {
            assert!(
                Policy::parse(
                    &DEFAULT_POLICY
                        .replace("channel_transaction_max_bytes = 4096", replacement)
                )
                .is_err()
            );
        }
        assert!(
            Policy::parse(&DEFAULT_POLICY.replace(
                "manifest_max_bytes = 1048576",
                "manifest_max_bytes = 1048577"
            ))
            .is_err()
        );
    }
}
