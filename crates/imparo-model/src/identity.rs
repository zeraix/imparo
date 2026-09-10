//! Stable semantic identity for a fully built [`crate::ModelPlan`].
//!
//! The encoding is deliberately independent of Rust's `Debug`, struct layout,
//! endianness, and serde defaults. Every semantic field is tagged and serialized in a
//! fixed order before SHA-256. Changing the encoding requires a domain-version bump.

use std::fmt::{Display, Formatter};

use sha2::{Digest, Sha256};

use crate::{Activation, Attention, Ffn, KvSource, ModelPlan};

const DOMAIN: &[u8] = b"imparo-model-plan-sha256-v2\0";

/// SHA-256 of the complete canonical semantic encoding of a [`ModelPlan`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModelPlanIdentity([u8; 32]);

impl ModelPlanIdentity {
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    #[must_use]
    pub fn to_hex(self) -> String {
        self.0
            .iter()
            .fold(String::with_capacity(64), |mut out, byte| {
                use std::fmt::Write as _;
                let _ = write!(out, "{byte:02x}");
                out
            })
    }
}

impl Display for ModelPlanIdentity {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

struct Canonical(Sha256);

impl Canonical {
    fn new() -> Self {
        let mut hash = Sha256::new();
        hash.update(DOMAIN);
        Self(hash)
    }

    fn u8(&mut self, value: u8) {
        self.0.update([value]);
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u32(&mut self, value: u32) {
        self.0.update(value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    fn string(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.0.update(value.as_bytes());
    }

    fn option_u32(&mut self, value: Option<u32>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.u32(value);
            }
            None => self.u8(0),
        }
    }

    fn option_f32(&mut self, value: Option<f32>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.f32(value);
            }
            None => self.u8(0),
        }
    }

    fn attention(&mut self, attention: Attention) {
        match attention {
            Attention::Full {
                head_dim,
                rope_base,
                rope_dim,
            } => {
                self.u8(0);
                self.u32(head_dim);
                self.f32(rope_base);
                self.u32(rope_dim);
            }
            Attention::Window {
                head_dim,
                rope_base,
                rope_dim,
                window,
            } => {
                self.u8(1);
                self.u32(head_dim);
                self.f32(rope_base);
                self.u32(rope_dim);
                self.u32(window);
            }
            Attention::Recurrent { r_elems, s_elems, key_dim, value_dim } => {
                self.u8(2);
                self.u32(r_elems);
                self.u32(s_elems);
                // The matrix's SHAPE, not just its size: a kernel is compiled for these,
                // so two files with the same s_elems and different coordinates run
                // different code and are different configurations.
                self.u32(key_dim);
                self.u32(value_dim);
            }
        }
    }

    fn activation(&mut self, activation: Activation) {
        self.u8(match activation {
            Activation::Gelu => 0,
            Activation::Silu => 1,
        });
    }

    fn ffn(&mut self, ffn: Ffn) {
        match ffn {
            Ffn::Dense { activation, hidden } => {
                self.u8(0);
                self.activation(activation);
                self.u32(hidden);
            }
            Ffn::Moe {
                activation,
                expert_hidden,
                experts,
                experts_used,
                shared_hidden,
            } => {
                self.u8(1);
                self.activation(activation);
                self.u32(expert_hidden);
                self.u32(experts);
                self.u32(experts_used);
                self.u32(shared_hidden);
            }
        }
    }

    fn kv_source(&mut self, source: KvSource) {
        match source {
            KvSource::Own => self.u8(0),
            KvSource::SharedWith(layer) => {
                self.u8(1);
                self.u32(layer);
            }
        }
    }

    fn finish(self) -> ModelPlanIdentity {
        ModelPlanIdentity(self.0.finalize().into())
    }
}

fn encode_hadamard_width(out: &mut Canonical, width: imparo_backend::HadamardWidth) {
    match width {
        imparo_backend::HadamardWidth::Disabled => out.u32(0),
        imparo_backend::HadamardWidth::FullHead => out.u32(1),
        imparo_backend::HadamardWidth::Fixed(value) => {
            out.u32(2);
            out.u32(value);
        }
    }
}

impl ModelPlan {
    /// Hashes every semantic plan field using the canonical versioned encoding above.
    ///
    /// This identifies execution geometry and policy, not weight bytes. Pair it with
    /// `Weights::full_file_sha256()` when a tuned configuration must bind to both.
    #[must_use]
    pub fn sha256_identity(&self) -> ModelPlanIdentity {
        let mut out = Canonical::new();
        out.string(&self.config.architecture);
        out.u32(self.config.n_layers);
        out.u32(self.config.n_embd);
        out.u32(self.config.n_ff);
        out.u32(self.config.n_heads);
        out.u32(self.config.n_kv_heads);
        out.u32(self.config.context_length);
        out.u32(self.config.vocab_size);
        out.f32(self.config.norm_eps);

        out.bool(self.embed.scale_by_sqrt_embd);
        out.option_u32(self.embed.per_layer_dim);
        match self.kv_storage_basis {
            crate::KvStorageBasisPolicy::Canonical => out.u32(0),
            crate::KvStorageBasisPolicy::BackendRoute => out.u32(1),
            crate::KvStorageBasisPolicy::BackendOverrideOrCanonical => out.u32(3),
            crate::KvStorageBasisPolicy::ExplicitRoute(route) => {
                out.u32(2);
                encode_hadamard_width(&mut out, route.key);
                encode_hadamard_width(&mut out, route.value);
            }
        }

        out.u64(self.weight_residency.row_gathered.len() as u64);
        for name in self.weight_residency.row_gathered {
            out.string(name);
        }

        out.u64(self.layers.len() as u64);
        for layer in &self.layers {
            out.u32(layer.index);
            out.attention(layer.attention);
            out.ffn(layer.ffn);
            out.kv_source(layer.kv_source);
        }

        out.bool(self.output.final_norm);
        out.option_f32(self.output.logit_softcap);
        out.bool(self.output.tied_embeddings);
        out.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EmbedPlan, LayerPlan, ModelConfig, OutputPlan, WeightResidencyPlan};

    static TABLES: &[&str] = &["per_layer_model_proj.weight"];

    fn fixture() -> ModelPlan {
        ModelPlan {
            config: ModelConfig {
                architecture: "fixture".into(),
                n_layers: 2,
                n_embd: 64,
                n_ff: 192,
                n_heads: 4,
                n_kv_heads: 2,
                context_length: 4096,
                vocab_size: 32000,
                norm_eps: 1e-5,
            },
            embed: EmbedPlan {
                scale_by_sqrt_embd: true,
                per_layer_dim: Some(16),
                per_layer_row_bytes: None,
            },
            kv_storage_basis: crate::KvStorageBasisPolicy::BackendRoute,
            weight_residency: WeightResidencyPlan {
                row_gathered: TABLES,
            },
            layers: vec![
                LayerPlan {
                    index: 0,
                    attention: Attention::Window {
                        head_dim: 32,
                        rope_base: 10_000.0,
                        rope_dim: 32,
                        window: 512,
                    },
                    ffn: Ffn::Dense {
                        activation: Activation::Gelu,
                        hidden: 192,
                    },
                    kv_source: KvSource::Own,
                },
                LayerPlan {
                    index: 1,
                    attention: Attention::Recurrent {
                        key_dim: 0,
                        value_dim: 0,
                        r_elems: 128,
                        s_elems: 64,
                    },
                    ffn: Ffn::Moe {
                        activation: Activation::Silu,
                        expert_hidden: 96,
                        experts: 8,
                        experts_used: 2,
                        shared_hidden: 32,
                    },
                    kv_source: KvSource::SharedWith(0),
                },
            ],
            output: OutputPlan {
                final_norm: true,
                logit_softcap: Some(30.0),
                tied_embeddings: false,
            },
        }
    }

    #[test]
    fn identity_is_deterministic_and_hex_is_canonical() {
        let plan = fixture();
        let identity = plan.sha256_identity();
        assert_eq!(identity, plan.clone().sha256_identity());
        assert_eq!(identity.to_hex(), identity.to_string());
        assert_eq!(identity.to_string().len(), 64);
        assert!(identity.to_string().bytes().all(|b| b.is_ascii_hexdigit()));
    }

    #[test]
    fn identity_covers_each_plan_section_and_variant() {
        let base = fixture().sha256_identity();
        let mut changes = Vec::new();

        let mut plan = fixture();
        plan.config.norm_eps = -0.0;
        changes.push(plan.sha256_identity());

        let mut plan = fixture();
        plan.embed.per_layer_dim = None;
        changes.push(plan.sha256_identity());

        let mut plan = fixture();
        plan.kv_storage_basis = crate::KvStorageBasisPolicy::Canonical;
        changes.push(plan.sha256_identity());

        let mut plan = fixture();
        plan.kv_storage_basis = crate::KvStorageBasisPolicy::ExplicitRoute(
            imparo_backend::KvQuantizationRoute {
                key: imparo_backend::HadamardWidth::FullHead,
                value: imparo_backend::HadamardWidth::Fixed(64),
            },
        );
        changes.push(plan.sha256_identity());

        let mut plan = fixture();
        plan.weight_residency = WeightResidencyPlan::default();
        changes.push(plan.sha256_identity());

        let mut plan = fixture();
        plan.layers[0].attention = Attention::Full {
            head_dim: 32,
            rope_base: 10_000.0,
            rope_dim: 32,
        };
        changes.push(plan.sha256_identity());

        let mut plan = fixture();
        plan.layers[1].ffn = Ffn::Dense {
            activation: Activation::Silu,
            hidden: 96,
        };
        changes.push(plan.sha256_identity());

        let mut plan = fixture();
        plan.layers[1].kv_source = KvSource::Own;
        changes.push(plan.sha256_identity());

        let mut plan = fixture();
        plan.output.logit_softcap = None;
        changes.push(plan.sha256_identity());

        assert!(changes.into_iter().all(|identity| identity != base));
    }

    #[test]
    fn layer_order_is_part_of_the_execution_plan() {
        let mut reordered = fixture();
        reordered.layers.swap(0, 1);
        assert_ne!(fixture().sha256_identity(), reordered.sha256_identity());
    }
}
