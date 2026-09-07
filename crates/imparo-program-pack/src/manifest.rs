//! Strict, bounded parser for Program Pack v1 manifests.

use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::error;
use std::fmt::{self, Write as _};

/// Hard ceiling applied before UTF-8 decoding or JSON allocation.
pub const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;

fn deserialize_non_null_option<'de, D, T>(
    deserializer: D,
) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl error::Error for Error {}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Manifest {
    pub(crate) schema: u32,
    pub(crate) program_pack_abi: u32,
    pub(crate) pack_id: String,
    pub(crate) pack_version: String,
    pub(crate) distribution: Distribution,
    pub(crate) release_channel: String,
    pub(crate) required_entitlement_features: Vec<String>,
    pub(crate) backend: String,
    pub(crate) engine_api: VersionRange,
    pub(crate) backend_abi: VersionRange,
    pub(crate) target: Target,
    pub(crate) required_extensions: Vec<Extension>,
    pub(crate) optional_extensions: Vec<Extension>,
    pub(crate) toolchain: Toolchain,
    pub(crate) modules: Vec<Module>,
    pub(crate) choice_groups: Vec<ChoiceGroup>,
    pub(crate) variants: Vec<Variant>,
    pub(crate) notices_sha256: String,
    pub(crate) sbom_sha256: String,
    pub(crate) provenance_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireManifest {
    schema: u32,
    program_pack_abi: u32,
    pack_id: String,
    pack_version: String,
    distribution: Distribution,
    release_channel: String,
    required_entitlement_features: Vec<String>,
    backend: String,
    engine_api: VersionRange,
    backend_abi: VersionRange,
    target: Target,
    required_extensions: Vec<Extension>,
    optional_extensions: Vec<Extension>,
    toolchain: Toolchain,
    modules: Vec<Module>,
    choice_groups: Vec<ChoiceGroup>,
    variants: Vec<Variant>,
    notices_sha256: String,
    sbom_sha256: String,
    provenance_sha256: String,
}

impl From<WireManifest> for Manifest {
    fn from(wire: WireManifest) -> Self {
        Self {
            schema: wire.schema,
            program_pack_abi: wire.program_pack_abi,
            pack_id: wire.pack_id,
            pack_version: wire.pack_version,
            distribution: wire.distribution,
            release_channel: wire.release_channel,
            required_entitlement_features: wire.required_entitlement_features,
            backend: wire.backend,
            engine_api: wire.engine_api,
            backend_abi: wire.backend_abi,
            target: wire.target,
            required_extensions: wire.required_extensions,
            optional_extensions: wire.optional_extensions,
            toolchain: wire.toolchain,
            modules: wire.modules,
            choice_groups: wire.choice_groups,
            variants: wire.variants,
            notices_sha256: wire.notices_sha256,
            sbom_sha256: wire.sbom_sha256,
            provenance_sha256: wire.provenance_sha256,
        }
    }
}

impl Manifest {
    pub fn parse(raw: &[u8]) -> Result<Self, Error> {
        if raw.len() > MAX_MANIFEST_BYTES {
            return Err(Error::new(format!(
                "manifest exceeds {MAX_MANIFEST_BYTES}-byte limit"
            )));
        }
        let text = std::str::from_utf8(raw)
            .map_err(|error| Error::new(format!("manifest is not UTF-8: {error}")))?;
        let wire: WireManifest = serde_json::from_str(text)
            .map_err(|error| Error::new(format!("invalid manifest JSON: {error}")))?;
        let manifest = Self::from(wire);
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), Error> {
        self.validate_schema()?;
        self.validate_semantics()
    }

    #[must_use]
    pub fn distribution(&self) -> Distribution {
        self.distribution
    }
    #[must_use]
    pub fn program_pack_abi(&self) -> u32 {
        self.program_pack_abi
    }
    #[must_use]
    pub fn pack_id(&self) -> &str {
        &self.pack_id
    }
    #[must_use]
    pub fn pack_version(&self) -> &str {
        &self.pack_version
    }
    #[must_use]
    pub fn backend(&self) -> &str {
        &self.backend
    }
    #[must_use]
    pub fn engine_api(&self) -> &VersionRange {
        &self.engine_api
    }
    #[must_use]
    pub fn backend_abi(&self) -> &VersionRange {
        &self.backend_abi
    }
    #[must_use]
    pub fn target(&self) -> &Target {
        &self.target
    }
    #[must_use]
    pub fn release_channel(&self) -> &str {
        &self.release_channel
    }
    #[must_use]
    pub fn modules(&self) -> &[Module] {
        &self.modules
    }
    #[must_use]
    pub fn choice_groups(&self) -> &[ChoiceGroup] {
        &self.choice_groups
    }
    #[must_use]
    pub fn variants(&self) -> &[Variant] {
        &self.variants
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Distribution {
    Community,
    Commercial,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VersionRange {
    pub min: u32,
    pub max_exclusive: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct Extension {
    pub id: String,
    pub revision: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub sm: u16,
    pub warp_size: u16,
    pub driver_min: u32,
    pub math_mode: MathMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MathMode {
    Strict,
    Fast,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Toolchain {
    pub builder_image_sha256: String,
    pub build_recipe_sha256: String,
    pub producer: AotProducer,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AotProducer {
    Triton {
        triton_revision: String,
        triton_version: String,
        python: String,
        cuda_toolkit: String,
    },
    CudaCpp {
        compiler_id: CompilerId,
        compiler_version: String,
        cuda_toolkit: String,
    },
    ExternalAot {
        producer_id: String,
        producer_revision: String,
        source_sha256: String,
        toolchain_sha256: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompilerId {
    Nvcc,
    ClangCuda,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Module {
    pub id: String,
    pub file: String,
    pub format: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContractRef {
    pub id: String,
    pub revision: u32,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct WorkloadParameters {
    #[serde(
        default,
        deserialize_with = "deserialize_non_null_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub narrow_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub workload_id: WorkloadId,
    pub revision: u32,
    pub parameters: WorkloadParameters,
    pub parameters_sha256: String,
    pub fixture_sha256: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WorkloadId {
    #[serde(rename = "imparo.workload.decode_mix")]
    DecodeMix,
    #[serde(rename = "imparo.workload.narrow_mix")]
    NarrowMix,
    #[serde(rename = "imparo.workload.attention_decode")]
    AttentionDecode,
    #[serde(rename = "imparo.workload.attention_decode_deep")]
    AttentionDecodeDeep,
    #[serde(rename = "imparo.workload.attention_prefill")]
    AttentionPrefill,
    #[serde(rename = "imparo.workload.attention_prefill_deep")]
    AttentionPrefillDeep,
    #[serde(rename = "imparo.workload.decode_attention_step")]
    DecodeAttentionStep,
    #[serde(rename = "imparo.workload.attention_prefill_reuse")]
    AttentionPrefillReuse,
    #[serde(rename = "imparo.workload.prefill_gemm")]
    PrefillGemm,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChoiceGroup {
    pub choice_group_id: String,
    pub contract: ContractRef,
    pub workload: Workload,
    #[serde(default, deserialize_with = "deserialize_non_null_option")]
    pub cross_check: Option<Workload>,
    pub screened: bool,
    pub bit_affecting: bool,
    pub joint_with: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ShapeConstraint {
    pub slot: String,
    pub axis: u8,
    pub min: u64,
    pub max: u64,
    pub multiple_of: u64,
    #[serde(default, deserialize_with = "deserialize_non_null_option")]
    pub one_of: Option<Vec<u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DtypeConstraint {
    pub slot: String,
    pub allowed: Vec<Dtype>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    F16,
    Bf16,
    F32,
    I32,
    U32,
    U64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct QuantizationConstraint {
    pub slot: String,
    pub allowed: Vec<Quantization>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    None,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LayoutConstraint {
    pub slot: String,
    pub allowed: Vec<Layout>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    Contiguous,
    RowMajor,
    ColumnMajor,
    PagedKv,
    HeadMajor,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AlignmentConstraint {
    pub slot: String,
    pub bytes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Constraints {
    pub shapes: Vec<ShapeConstraint>,
    pub dtypes: Vec<DtypeConstraint>,
    pub quantizations: Vec<QuantizationConstraint>,
    pub layouts: Vec<LayoutConstraint>,
    pub alignments: Vec<AlignmentConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AliasRule {
    pub mode: AliasMode,
    pub target_slot: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AliasMode {
    Forbidden,
    MayAlias,
    MustAlias,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Effect {
    pub slot: String,
    pub access: Access,
    pub aliasing: Vec<AliasRule>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Scratch {
    pub max_bytes: u64,
    pub alignment: u32,
    pub zero_initialized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LaunchArgument {
    Slot { slot: String, wire_type: WireType },
    ManifestU32 { name: String, value: u32 },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WireType {
    TensorPtr,
    StatePtr,
    ScratchPtr,
    ScalarI32,
    ScalarU32,
    ScalarU64,
    ScalarF32,
    ManifestU32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum IntegerExpression {
    Const {
        value: u64,
    },
    Slot {
        slot: String,
    },
    Op {
        op: IntegerOp,
        args: Box<[IntegerExpression; 2]>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IntegerOp {
    Add,
    Mul,
    CeilDiv,
    Min,
    Max,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Grid {
    pub x: IntegerExpression,
    pub y: IntegerExpression,
    pub z: IntegerExpression,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Block {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    pub arguments: Vec<LaunchArgument>,
    pub grid: Grid,
    pub block: Block,
    pub dynamic_shared_bytes: IntegerExpression,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Resources {
    pub registers_per_thread_max: u16,
    pub static_shared_bytes_max: u32,
    pub dynamic_shared_bytes_max: u32,
    pub local_memory_bytes_max: u64,
    pub threads_per_block_max: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NumericalClass {
    BitExact {
        reference: String,
        contract_version: u32,
    },
    GateBounded {
        gate_suite: String,
        contract_version: u32,
    },
    DiagnosticOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TypedReference {
    Variant { id: String },
    ChoiceGroup { id: String },
    Feature { id: String },
}

impl TypedReference {
    fn key(&self) -> (&'static str, &str) {
        match self {
            Self::Variant { id } => ("variant", id),
            Self::ChoiceGroup { id } => ("choice_group", id),
            Self::Feature { id } => ("feature", id),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeatureReference {
    Feature { id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub variant_id: String,
    pub config_id: String,
    pub choice_group_id: String,
    #[serde(default, deserialize_with = "deserialize_non_null_option")]
    pub extension: Option<Extension>,
    pub contract: ContractRef,
    pub module_id: String,
    pub symbol: String,
    pub constraints: Constraints,
    pub effects: Vec<Effect>,
    pub scratch: Scratch,
    pub launch: Launch,
    pub resources: Resources,
    pub graph_capture: GraphCapture,
    pub graph_update_slots: Vec<String>,
    pub numerical_class: NumericalClass,
    pub determinism: Determinism,
    pub bit_affecting: bool,
    pub required_entitlement_features: Vec<String>,
    pub requires: Vec<TypedReference>,
    pub conflicts: Vec<TypedReference>,
    pub provides: Vec<FeatureReference>,
    pub joint_with: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GraphCapture {
    Forbidden,
    CaptureOnly,
    ReplayUpdateSafe,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Determinism {
    Required,
    NotRequired,
}

impl Manifest {
    fn validate_schema(&self) -> Result<(), Error> {
        ensure(self.schema == 1, "schema must be 1")?;
        ensure(self.program_pack_abi == 1, "program_pack_abi must be 1")?;
        namespaced_id(&self.pack_id, "pack_id")?;
        stable_semver(&self.pack_version, "pack_version")?;
        namespaced_id(&self.release_channel, "release_channel")?;
        ensure(self.backend == "cuda", "backend must be cuda")?;
        validate_string_list(
            &self.required_entitlement_features,
            64,
            "required_entitlement_features",
            namespaced_id,
        )?;
        self.engine_api.validate_schema("engine_api")?;
        self.backend_abi.validate_schema("backend_abi")?;
        self.target.validate_schema()?;
        validate_extension_list(&self.required_extensions, "required_extensions")?;
        validate_extension_list(&self.optional_extensions, "optional_extensions")?;
        self.toolchain.validate_schema()?;
        bounded_len(&self.modules, 1, 64, "modules")?;
        for module in &self.modules {
            module.validate_schema()?;
        }
        bounded_len(&self.choice_groups, 1, 256, "choice_groups")?;
        for group in &self.choice_groups {
            group.validate_schema()?;
        }
        bounded_len(&self.variants, 1, 4096, "variants")?;
        for variant in &self.variants {
            variant.validate_schema()?;
        }
        sha256(&self.notices_sha256, "notices_sha256")?;
        sha256(&self.sbom_sha256, "sbom_sha256")?;
        sha256(&self.provenance_sha256, "provenance_sha256")?;
        Ok(())
    }
}

impl VersionRange {
    fn validate_schema(&self, label: &str) -> Result<(), Error> {
        ensure(self.min >= 1, format!("{label}.min is below 1"))?;
        ensure(
            self.max_exclusive >= 2,
            format!("{label}.max_exclusive is below 2"),
        )
    }
}

impl Extension {
    fn validate_schema(&self, label: &str) -> Result<(), Error> {
        namespaced_id(&self.id, &format!("{label}.id"))?;
        ensure(self.revision >= 1, format!("{label}.revision is below 1"))
    }
}

impl Target {
    fn validate_schema(&self) -> Result<(), Error> {
        ensure(
            (50..=999).contains(&self.sm),
            "target.sm is outside 50..=999",
        )?;
        ensure(self.warp_size == 32, "target.warp_size must be 32")
    }
}

impl Toolchain {
    fn validate_schema(&self) -> Result<(), Error> {
        sha256(&self.builder_image_sha256, "toolchain.builder_image_sha256")?;
        sha256(&self.build_recipe_sha256, "toolchain.build_recipe_sha256")?;
        match &self.producer {
            AotProducer::Triton {
                triton_revision,
                triton_version,
                python,
                cuda_toolkit,
            } => {
                git_revision(triton_revision, "toolchain.producer.triton_revision")?;
                stable_semver(triton_version, "toolchain.producer.triton_version")?;
                python_version(python, "toolchain.producer.python")?;
                tool_version(cuda_toolkit, "toolchain.producer.cuda_toolkit")
            }
            AotProducer::CudaCpp {
                compiler_version,
                cuda_toolkit,
                ..
            } => {
                tool_version(compiler_version, "toolchain.producer.compiler_version")?;
                tool_version(cuda_toolkit, "toolchain.producer.cuda_toolkit")
            }
            AotProducer::ExternalAot {
                producer_id,
                producer_revision,
                source_sha256,
                toolchain_sha256,
            } => {
                namespaced_id(producer_id, "toolchain.producer.producer_id")?;
                git_revision(
                    producer_revision,
                    "toolchain.producer.producer_revision",
                )?;
                sha256(source_sha256, "toolchain.producer.source_sha256")?;
                sha256(toolchain_sha256, "toolchain.producer.toolchain_sha256")
            }
        }
    }
}

impl Module {
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    fn validate_schema(&self) -> Result<(), Error> {
        opaque_id(&self.id, "module.id")?;
        ensure(
            self.file.len() == 78
                && self.file.starts_with("modules/")
                && self.file.ends_with(".cubin")
                && is_lower_hex(&self.file.as_bytes()[8..72]),
            "module.file has invalid content-addressed path",
        )?;
        ensure(self.format == "cubin", "module.format must be cubin")?;
        ensure(
            (1..=268_435_456).contains(&self.bytes),
            "module.bytes is outside 1..=268435456",
        )?;
        sha256(&self.sha256, "module.sha256")
    }
}

impl ContractRef {
    fn validate_schema(&self, label: &str) -> Result<(), Error> {
        namespaced_id(&self.id, &format!("{label}.id"))?;
        ensure(self.revision >= 1, format!("{label}.revision is below 1"))?;
        sha256(&self.sha256, &format!("{label}.sha256"))
    }
}

impl Workload {
    fn validate_schema(&self, label: &str) -> Result<(), Error> {
        ensure(self.revision == 1, format!("{label}.revision must be 1"))?;
        if let Some(tokens) = self.parameters.narrow_tokens {
            ensure(
                (1..=1_048_576).contains(&tokens),
                format!("{label}.parameters.narrow_tokens is outside bounds"),
            )?;
        }
        sha256(
            &self.parameters_sha256,
            &format!("{label}.parameters_sha256"),
        )?;
        sha256(&self.fixture_sha256, &format!("{label}.fixture_sha256"))
    }
}

impl ChoiceGroup {
    fn validate_schema(&self) -> Result<(), Error> {
        namespaced_id(&self.choice_group_id, "choice_group.choice_group_id")?;
        self.contract.validate_schema("choice_group.contract")?;
        self.workload.validate_schema("choice_group.workload")?;
        if let Some(cross_check) = &self.cross_check {
            cross_check.validate_schema("choice_group.cross_check")?;
        }
        validate_string_list(
            &self.joint_with,
            32,
            "choice_group.joint_with",
            namespaced_id,
        )
    }
}

impl Variant {
    fn validate_schema(&self) -> Result<(), Error> {
        opaque_id(&self.variant_id, "variant.variant_id")?;
        opaque_id(&self.config_id, "variant.config_id")?;
        namespaced_id(&self.choice_group_id, "variant.choice_group_id")?;
        if let Some(extension) = &self.extension {
            extension.validate_schema("variant.extension")?;
        }
        self.contract.validate_schema("variant.contract")?;
        opaque_id(&self.module_id, "variant.module_id")?;
        ensure(
            self.symbol.len() == 67
                && self.symbol.starts_with("ip_")
                && is_lower_hex(&self.symbol.as_bytes()[3..]),
            "variant.symbol has invalid form",
        )?;
        self.constraints.validate_schema()?;
        bounded_len(&self.effects, 1, 128, "variant.effects")?;
        for effect in &self.effects {
            effect.validate_schema()?;
        }
        self.scratch.validate_schema()?;
        self.launch.validate_schema()?;
        self.resources.validate_schema()?;
        validate_string_list(
            &self.graph_update_slots,
            64,
            "variant.graph_update_slots",
            slot_id,
        )?;
        self.numerical_class.validate_schema()?;
        validate_string_list(
            &self.required_entitlement_features,
            64,
            "variant.required_entitlement_features",
            namespaced_id,
        )?;
        validate_ref_list(&self.requires, "variant.requires")?;
        validate_ref_list(&self.conflicts, "variant.conflicts")?;
        bounded_len(&self.provides, 0, 64, "variant.provides")?;
        ensure_unique(self.provides.iter(), "variant.provides")?;
        for reference in &self.provides {
            match reference {
                FeatureReference::Feature { id } => {
                    namespaced_id(id, "variant.provides.id")?;
                }
            }
        }
        validate_string_list(&self.joint_with, 32, "variant.joint_with", namespaced_id)
    }
}

impl Constraints {
    fn validate_schema(&self) -> Result<(), Error> {
        bounded_len(&self.shapes, 0, 128, "constraints.shapes")?;
        for item in &self.shapes {
            slot_id(&item.slot, "shape.slot")?;
            ensure(item.axis <= 31, "shape.axis exceeds 31")?;
            ensure(
                item.min <= MAX_SAFE_JSON_INTEGER && item.max <= MAX_SAFE_JSON_INTEGER,
                "shape bound exceeds safe JSON integer",
            )?;
            ensure(
                (1..=MAX_SAFE_JSON_INTEGER).contains(&item.multiple_of),
                "shape.multiple_of is outside bounds",
            )?;
            if let Some(one_of) = &item.one_of {
                bounded_len(one_of, 1, 128, "shape.one_of")?;
                ensure_unique(one_of.iter(), "shape.one_of")?;
                ensure(
                    one_of.iter().all(|value| *value <= MAX_SAFE_JSON_INTEGER),
                    "shape.one_of exceeds safe JSON integer",
                )?;
            }
        }
        bounded_len(&self.dtypes, 0, 128, "constraints.dtypes")?;
        for item in &self.dtypes {
            slot_id(&item.slot, "dtype.slot")?;
            bounded_len(&item.allowed, 1, 8, "dtype.allowed")?;
            ensure_unique(item.allowed.iter(), "dtype.allowed")?;
        }
        bounded_len(&self.quantizations, 0, 128, "constraints.quantizations")?;
        for item in &self.quantizations {
            slot_id(&item.slot, "quantization.slot")?;
            bounded_len(&item.allowed, 1, 8, "quantization.allowed")?;
            ensure_unique(item.allowed.iter(), "quantization.allowed")?;
        }
        bounded_len(&self.layouts, 0, 128, "constraints.layouts")?;
        for item in &self.layouts {
            slot_id(&item.slot, "layout.slot")?;
            bounded_len(&item.allowed, 1, 8, "layout.allowed")?;
            ensure_unique(item.allowed.iter(), "layout.allowed")?;
        }
        bounded_len(&self.alignments, 0, 128, "constraints.alignments")?;
        for item in &self.alignments {
            slot_id(&item.slot, "alignment.slot")?;
            alignment(item.bytes, "alignment.bytes")?;
        }
        Ok(())
    }
}

impl Effect {
    fn validate_schema(&self) -> Result<(), Error> {
        slot_id(&self.slot, "effect.slot")?;
        bounded_len(&self.aliasing, 0, 64, "effect.aliasing")?;
        for alias in &self.aliasing {
            slot_id(&alias.target_slot, "effect.aliasing.target_slot")?;
        }
        Ok(())
    }
}

impl Scratch {
    fn validate_schema(&self) -> Result<(), Error> {
        ensure(
            self.max_bytes <= 2_147_483_648,
            "scratch.max_bytes exceeds ceiling",
        )?;
        alignment(self.alignment, "scratch.alignment")
    }
}

impl Launch {
    fn validate_schema(&self) -> Result<(), Error> {
        bounded_len(&self.arguments, 1, 64, "launch.arguments")?;
        for argument in &self.arguments {
            match argument {
                LaunchArgument::Slot { slot, .. } => {
                    slot_id(slot, "launch.argument.slot")?;
                }
                LaunchArgument::ManifestU32 { name, .. } => {
                    slot_id(name, "launch.argument.name")?;
                }
            }
        }
        ensure(
            (1..=1024).contains(&self.block.x),
            "launch.block.x outside bounds",
        )?;
        ensure(
            (1..=1024).contains(&self.block.y),
            "launch.block.y outside bounds",
        )?;
        ensure(
            (1..=64).contains(&self.block.z),
            "launch.block.z outside bounds",
        )?;
        self.grid.x.validate_schema()?;
        self.grid.y.validate_schema()?;
        self.grid.z.validate_schema()?;
        self.dynamic_shared_bytes.validate_schema()
    }
}

impl IntegerExpression {
    fn validate_schema(&self) -> Result<(), Error> {
        match self {
            Self::Const { value } => ensure(
                *value <= MAX_SAFE_JSON_INTEGER,
                "launch constant exceeds safe JSON integer",
            ),
            Self::Slot { slot } => slot_id(slot, "launch expression slot"),
            Self::Op { args, .. } => {
                args[0].validate_schema()?;
                args[1].validate_schema()
            }
        }
    }
}

impl Resources {
    fn validate_schema(&self) -> Result<(), Error> {
        ensure(
            self.registers_per_thread_max <= 255,
            "resources.registers_per_thread_max exceeds 255",
        )?;
        ensure(
            self.static_shared_bytes_max <= 262_144,
            "resources.static_shared_bytes_max exceeds ceiling",
        )?;
        ensure(
            self.dynamic_shared_bytes_max <= 262_144,
            "resources.dynamic_shared_bytes_max exceeds ceiling",
        )?;
        ensure(
            self.local_memory_bytes_max <= 1_073_741_824,
            "resources.local_memory_bytes_max exceeds ceiling",
        )?;
        ensure(
            (1..=1024).contains(&self.threads_per_block_max),
            "resources.threads_per_block_max outside bounds",
        )
    }
}

impl NumericalClass {
    fn validate_schema(&self) -> Result<(), Error> {
        match self {
            Self::BitExact {
                reference,
                contract_version,
            } => {
                namespaced_id(reference, "numerical_class.reference")?;
                ensure(
                    *contract_version >= 1,
                    "numerical_class.contract_version below 1",
                )
            }
            Self::GateBounded {
                gate_suite,
                contract_version,
            } => {
                namespaced_id(gate_suite, "numerical_class.gate_suite")?;
                ensure(
                    *contract_version >= 1,
                    "numerical_class.contract_version below 1",
                )
            }
            Self::DiagnosticOnly => Ok(()),
        }
    }
}

fn validate_extension_list(values: &[Extension], label: &str) -> Result<(), Error> {
    bounded_len(values, 0, 64, label)?;
    ensure_unique(values.iter(), label)?;
    for value in values {
        value.validate_schema(label)?;
    }
    Ok(())
}

fn validate_ref_list(values: &[TypedReference], label: &str) -> Result<(), Error> {
    bounded_len(values, 0, 64, label)?;
    ensure_unique(values.iter(), label)?;
    for value in values {
        let (_, id) = value.key();
        match value {
            TypedReference::Variant { .. } => opaque_id(id, &format!("{label}.id"))?,
            _ => namespaced_id(id, &format!("{label}.id"))?,
        }
    }
    Ok(())
}

impl Manifest {
    fn validate_semantics(&self) -> Result<(), Error> {
        for (label, range) in [
            ("engine_api", &self.engine_api),
            ("backend_abi", &self.backend_abi),
        ] {
            ensure(
                range.min < range.max_exclusive,
                format!("{label} is empty or reversed"),
            )?;
        }

        ensure_sorted(
            &self.required_entitlement_features,
            "pack entitlement features are not sorted",
        )?;
        if self.distribution == Distribution::Community {
            ensure(
                self.required_entitlement_features.is_empty(),
                "community pack requires a commercial entitlement",
            )?;
        }

        let required_extensions = extension_map(&self.required_extensions, "required")?;
        let optional_extensions = extension_map(&self.optional_extensions, "optional")?;
        ensure(
            required_extensions
                .keys()
                .all(|id| !optional_extensions.contains_key(id)),
            "required and optional extensions overlap",
        )?;
        let declared_extensions: HashMap<&str, u32> = required_extensions
            .into_iter()
            .chain(optional_extensions)
            .collect();

        ensure_unique(self.modules.iter().map(|module| &module.id), "module id")?;
        ensure_unique(
            self.modules.iter().map(|module| &module.file),
            "module path",
        )?;
        ensure_unique(
            self.modules.iter().map(|module| &module.sha256),
            "module digest",
        )?;
        let module_bytes = self.modules.iter().try_fold(0_u64, |total, module| {
            total
                .checked_add(module.bytes)
                .ok_or_else(|| Error::new("declared module bytes overflow"))
        })?;
        ensure(
            module_bytes <= 536_870_912,
            "declared module bytes exceed pack ceiling",
        )?;
        let module_ids: HashSet<&str> = self
            .modules
            .iter()
            .map(|module| module.id.as_str())
            .collect();

        ensure_unique(
            self.choice_groups
                .iter()
                .map(|group| &group.choice_group_id),
            "choice-group id",
        )?;
        let groups: HashMap<&str, &ChoiceGroup> = self
            .choice_groups
            .iter()
            .map(|group| (group.choice_group_id.as_str(), group))
            .collect();
        for group in &self.choice_groups {
            group.workload.validate_semantics("workload")?;
            if let Some(cross_check) = &group.cross_check {
                cross_check.validate_semantics("cross-check")?;
            }
            ensure(
                !group.joint_with.contains(&group.choice_group_id),
                "choice group is joint with itself",
            )?;
        }
        for group in &self.choice_groups {
            for joint in &group.joint_with {
                if let Some(other) = groups.get(joint.as_str()) {
                    ensure(
                        other.joint_with.contains(&group.choice_group_id),
                        "choice-group joint relation is not symmetric",
                    )?;
                }
            }
        }

        ensure_unique(
            self.variants.iter().map(|variant| &variant.variant_id),
            "variant id",
        )?;
        ensure_unique(
            self.variants.iter().map(|variant| &variant.config_id),
            "config id",
        )?;
        let variants: HashMap<&str, &Variant> = self
            .variants
            .iter()
            .map(|variant| (variant.variant_id.as_str(), variant))
            .collect();
        let variant_ids: HashSet<&str> = variants.keys().copied().collect();
        let mut required_entitlements = HashSet::new();

        for variant in &self.variants {
            ensure_sorted(
                &variant.required_entitlement_features,
                "variant entitlement features are not sorted",
            )?;
            let group = groups
                .get(variant.choice_group_id.as_str())
                .ok_or_else(|| Error::new("variant references missing choice group"))?;
            ensure(
                module_ids.contains(variant.module_id.as_str()),
                "variant references missing module",
            )?;
            ensure(
                variant.contract == group.contract,
                "variant and choice-group contracts differ",
            )?;
            ensure(
                variant.bit_affecting == group.bit_affecting,
                "variant and choice-group bit-affecting flags differ",
            )?;
            ensure(
                set_of(&variant.joint_with) == set_of(&group.joint_with),
                "variant and choice-group joint sets differ",
            )?;
            if let Some(extension) = &variant.extension {
                let revision =
                    declared_extensions.get(extension.id.as_str()).ok_or_else(
                        || Error::new("variant references undeclared extension"),
                    )?;
                ensure(
                    *revision == extension.revision,
                    "variant references wrong extension revision",
                )?;
            }
            variant.validate_semantics(&variant_ids, &variants, &groups)?;
            required_entitlements.extend(
                variant
                    .required_entitlement_features
                    .iter()
                    .map(String::as_str),
            );
        }

        ensure(
            required_entitlements == set_of(&self.required_entitlement_features),
            "pack and variant entitlement feature sets differ",
        )?;

        for variant in &self.variants {
            for conflict in &variant.conflicts {
                if let TypedReference::Variant { id } = conflict {
                    let reverse = variants[id.as_str()].conflicts.iter().any(|item| {
                        matches!(
                            item,
                            TypedReference::Variant { id: reverse_id }
                                if reverse_id == &variant.variant_id
                        )
                    });
                    ensure(reverse, "variant conflict relation is not symmetric")?;
                }
            }
        }

        self.validate_dependency_graph(&variants)?;
        let selected_groups: HashSet<&str> = self
            .variants
            .iter()
            .map(|variant| variant.choice_group_id.as_str())
            .collect();
        ensure(
            selected_groups == groups.keys().copied().collect(),
            "choice group has no variant",
        )
    }

    fn validate_dependency_graph(
        &self,
        variants: &HashMap<&str, &Variant>,
    ) -> Result<(), Error> {
        let mut variants_by_group: HashMap<&str, Vec<&str>> = HashMap::new();
        for variant in &self.variants {
            variants_by_group
                .entry(variant.choice_group_id.as_str())
                .or_default()
                .push(variant.variant_id.as_str());
        }
        let mut state: HashMap<&str, u8> = HashMap::new();
        for variant_id in variants.keys().copied() {
            let mut stack = vec![(variant_id, false)];
            while let Some((current, exiting)) = stack.pop() {
                if exiting {
                    state.insert(current, 2);
                    continue;
                }
                match state.get(current).copied().unwrap_or(0) {
                    2 => continue,
                    1 => return Err(Error::new("variant dependency cycle")),
                    _ => {}
                }
                state.insert(current, 1);
                stack.push((current, true));
                for relation in variants[current].requires.iter().rev() {
                    match relation {
                        TypedReference::Variant { id } => stack.push((id, false)),
                        TypedReference::ChoiceGroup { id } => {
                            if let Some(targets) = variants_by_group.get(id.as_str()) {
                                if targets.len() == 1 {
                                    stack.push((targets[0], false));
                                }
                            }
                        }
                        TypedReference::Feature { .. } => {}
                    }
                }
            }
        }
        Ok(())
    }
}

impl Workload {
    fn validate_semantics(&self, label: &str) -> Result<(), Error> {
        match self.workload_id {
            WorkloadId::NarrowMix => ensure(
                self.parameters.narrow_tokens.is_some(),
                format!("{label} narrow_mix requires exactly narrow_tokens"),
            )?,
            _ => ensure(
                self.parameters.narrow_tokens.is_none(),
                format!("{label} parameterless workload contains parameters"),
            )?,
        }
        ensure(
            self.parameters_sha256 == workload_parameters_digest(&self.parameters),
            format!("{label} parameter digest mismatch"),
        )
    }
}

impl Variant {
    fn validate_semantics(
        &self,
        variant_ids: &HashSet<&str>,
        variants: &HashMap<&str, &Variant>,
        groups: &HashMap<&str, &ChoiceGroup>,
    ) -> Result<(), Error> {
        ensure_unique(
            self.constraints
                .shapes
                .iter()
                .map(|item| (&item.slot, item.axis)),
            "shape constraint",
        )?;
        for shape in &self.constraints.shapes {
            ensure(shape.min <= shape.max, "shape range is reversed")?;
            if let Some(one_of) = &shape.one_of {
                for value in one_of {
                    ensure(
                        shape.min <= *value && *value <= shape.max,
                        "shape one_of value is outside range",
                    )?;
                    ensure(
                        *value % shape.multiple_of == 0,
                        "shape one_of value violates multiple_of",
                    )?;
                }
            }
            let adjustment = shape.multiple_of - 1;
            let first_multiple =
                shape.min.checked_add(adjustment).ok_or_else(|| {
                    Error::new("shape multiple computation overflows")
                })? / shape.multiple_of
                    * shape.multiple_of;
            ensure(
                first_multiple <= shape.max,
                "shape constraint has no satisfiable value",
            )?;
        }
        ensure_unique(
            self.constraints.dtypes.iter().map(|item| &item.slot),
            "dtypes slot constraint",
        )?;
        ensure_unique(
            self.constraints.quantizations.iter().map(|item| &item.slot),
            "quantizations slot constraint",
        )?;
        ensure_unique(
            self.constraints.layouts.iter().map(|item| &item.slot),
            "layouts slot constraint",
        )?;
        ensure_unique(
            self.constraints.alignments.iter().map(|item| &item.slot),
            "alignments slot constraint",
        )?;

        ensure_unique(
            self.effects.iter().map(|effect| &effect.slot),
            "effect slot",
        )?;
        let effect_slots: HashSet<&str> = self
            .effects
            .iter()
            .map(|effect| effect.slot.as_str())
            .collect();
        for effect in &self.effects {
            ensure_unique(
                effect.aliasing.iter().map(|alias| &alias.target_slot),
                "alias target",
            )?;
            for alias in &effect.aliasing {
                ensure(alias.target_slot != effect.slot, "effect aliases itself")?;
                ensure(
                    effect_slots.contains(alias.target_slot.as_str()),
                    "effect aliases an unknown slot",
                )?;
            }
        }

        let threads = u64::from(self.launch.block.x)
            * u64::from(self.launch.block.y)
            * u64::from(self.launch.block.z);
        ensure(
            threads <= 1024
                && threads <= u64::from(self.resources.threads_per_block_max),
            "block exceeds thread ceiling",
        )?;
        let unsigned_slots: HashSet<&str> = self
            .launch
            .arguments
            .iter()
            .filter_map(|argument| match argument {
                LaunchArgument::Slot {
                    slot,
                    wire_type: WireType::ScalarU32 | WireType::ScalarU64,
                } => Some(slot.as_str()),
                _ => None,
            })
            .collect();
        for (axis, expression, ceiling) in [
            ("x", &self.launch.grid.x, 2_147_483_647_u64),
            ("y", &self.launch.grid.y, 65_535_u64),
            ("z", &self.launch.grid.z, 65_535_u64),
        ] {
            let (_, constant) = expression.walk(&unsigned_slots, 1)?;
            if let Some(value) = constant {
                ensure(
                    (1..=ceiling).contains(&value),
                    format!("grid {axis} is outside limits"),
                )?;
            }
        }
        let (_, dynamic_shared) =
            self.launch.dynamic_shared_bytes.walk(&unsigned_slots, 1)?;
        if let Some(value) = dynamic_shared {
            ensure(
                value
                    <= u64::from(self.resources.dynamic_shared_bytes_max.min(262_144)),
                "dynamic shared memory exceeds ceiling",
            )?;
        }
        ensure(
            self.graph_capture == GraphCapture::ReplayUpdateSafe
                || self.graph_update_slots.is_empty(),
            "non-replay graph route declares update slots",
        )?;

        let required: HashSet<(&str, &str)> =
            self.requires.iter().map(TypedReference::key).collect();
        let conflicts: HashSet<(&str, &str)> =
            self.conflicts.iter().map(TypedReference::key).collect();
        ensure(
            required.len() == self.requires.len(),
            "duplicate required dependency",
        )?;
        ensure(
            conflicts.len() == self.conflicts.len(),
            "duplicate conflict dependency",
        )?;
        ensure(
            required.is_disjoint(&conflicts),
            "dependency is both required and conflicted",
        )?;
        for relation in self.requires.iter().chain(&self.conflicts) {
            match relation {
                TypedReference::Variant { id } => {
                    ensure(
                        variant_ids.contains(id.as_str()),
                        "dependency references missing variant",
                    )?;
                    ensure(id != &self.variant_id, "variant references itself")?;
                }
                TypedReference::ChoiceGroup { id } => {
                    ensure(
                        id != &self.choice_group_id,
                        "variant references its own choice group",
                    )?;
                    // External catalog choice groups are intentionally valid.
                    let _ = groups.get(id.as_str());
                }
                TypedReference::Feature { .. } => {}
            }
        }
        for relation in &self.requires {
            if let TypedReference::Variant { id } = relation {
                ensure(
                    variants[id.as_str()].choice_group_id != self.choice_group_id,
                    "variant requires another variant in its exclusive choice group",
                )?;
            }
        }
        Ok(())
    }
}

impl IntegerExpression {
    fn walk(
        &self,
        allowed_slots: &HashSet<&str>,
        depth: usize,
    ) -> Result<(usize, Option<u64>), Error> {
        ensure(depth <= 8, "launch expression exceeds depth limit")?;
        match self {
            Self::Const { value } => Ok((1, Some(*value))),
            Self::Slot { slot } => {
                ensure(
                    allowed_slots.contains(slot.as_str()),
                    "launch expression references an unknown unsigned slot",
                )?;
                Ok((1, None))
            }
            Self::Op { op, args } => {
                let (left_nodes, left) = args[0].walk(allowed_slots, depth + 1)?;
                let (right_nodes, right) = args[1].walk(allowed_slots, depth + 1)?;
                let nodes = 1 + left_nodes + right_nodes;
                ensure(nodes <= 64, "launch expression exceeds node limit")?;
                if *op == IntegerOp::CeilDiv && right == Some(0) {
                    return Err(Error::new("launch expression divides by zero"));
                }
                let (Some(left), Some(right)) = (left, right) else {
                    return Ok((nodes, None));
                };
                let result = match op {
                    IntegerOp::Add => left.checked_add(right),
                    IntegerOp::Mul => left.checked_mul(right),
                    IntegerOp::CeilDiv => left
                        .checked_add(right - 1)
                        .map(|numerator| numerator / right),
                    IntegerOp::Min => Some(left.min(right)),
                    IntegerOp::Max => Some(left.max(right)),
                }
                .ok_or_else(|| Error::new("launch expression overflows u64"))?;
                Ok((nodes, Some(result)))
            }
        }
    }
}

fn ensure(condition: bool, message: impl Into<String>) -> Result<(), Error> {
    if condition {
        Ok(())
    } else {
        Err(Error::new(message))
    }
}

fn bounded_len<T>(
    values: &[T],
    min: usize,
    max: usize,
    label: &str,
) -> Result<(), Error> {
    ensure(
        (min..=max).contains(&values.len()),
        format!("{label} length is outside {min}..={max}"),
    )
}

fn ensure_unique<T, I>(values: I, label: &str) -> Result<(), Error>
where
    T: Eq + std::hash::Hash,
    I: IntoIterator<Item = T>,
{
    let mut seen = HashSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(Error::new(format!("duplicate {label}")));
        }
    }
    Ok(())
}

fn ensure_sorted(values: &[String], message: &str) -> Result<(), Error> {
    ensure(values.windows(2).all(|pair| pair[0] <= pair[1]), message)
}

fn set_of(values: &[String]) -> HashSet<&str> {
    values.iter().map(String::as_str).collect()
}

fn validate_string_list(
    values: &[String],
    max: usize,
    label: &str,
    validate: fn(&str, &str) -> Result<(), Error>,
) -> Result<(), Error> {
    bounded_len(values, 0, max, label)?;
    ensure_unique(values.iter(), label)?;
    for value in values {
        validate(value, label)?;
    }
    Ok(())
}

fn extension_map<'a>(
    extensions: &'a [Extension],
    label: &str,
) -> Result<HashMap<&'a str, u32>, Error> {
    let mut out = HashMap::new();
    for extension in extensions {
        if out
            .insert(extension.id.as_str(), extension.revision)
            .is_some()
        {
            return Err(Error::new(format!("duplicate {label} extension id")));
        }
    }
    Ok(out)
}

fn is_lower_hex(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn sha256(value: &str, label: &str) -> Result<(), Error> {
    ensure(
        value.len() == 64 && is_lower_hex(value.as_bytes()),
        format!("{label} must be 64 lowercase hexadecimal characters"),
    )
}

fn opaque_id(value: &str, label: &str) -> Result<(), Error> {
    sha256(value, label)
}

fn git_revision(value: &str, label: &str) -> Result<(), Error> {
    ensure(
        value.len() == 40 && is_lower_hex(value.as_bytes()),
        format!("{label} must be 40 lowercase hexadecimal characters"),
    )
}

fn namespaced_id(value: &str, label: &str) -> Result<(), Error> {
    let bytes = value.as_bytes();
    let valid_len = (3..=128).contains(&bytes.len());
    let mut saw_separator = false;
    let mut component_len = 0_usize;
    let mut valid = valid_len;
    for byte in bytes {
        if byte.is_ascii_lowercase() || byte.is_ascii_digit() {
            component_len += 1;
        } else if matches!(byte, b'.' | b'_' | b'-') {
            valid &= component_len > 0;
            saw_separator = true;
            component_len = 0;
        } else {
            valid = false;
        }
    }
    valid &= saw_separator && component_len > 0;
    ensure(valid, format!("{label} is not a valid namespaced id"))
}

fn slot_id(value: &str, label: &str) -> Result<(), Error> {
    let bytes = value.as_bytes();
    ensure(
        (1..=64).contains(&bytes.len())
            && bytes[0].is_ascii_lowercase()
            && bytes.iter().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'_'
            }),
        format!("{label} is not a valid slot id"),
    )
}

fn stable_semver(value: &str, label: &str) -> Result<(), Error> {
    ensure(
        (5..=64).contains(&value.len()) && numeric_version(value, 3, 3, false),
        format!("{label} is not a stable semantic version core"),
    )
}

fn tool_version(value: &str, label: &str) -> Result<(), Error> {
    ensure(
        (3..=32).contains(&value.len()) && numeric_version(value, 2, 3, true),
        format!("{label} is not a supported tool version"),
    )
}

fn python_version(value: &str, label: &str) -> Result<(), Error> {
    let major = value.split('.').next();
    ensure(
        (3..=32).contains(&value.len())
            && numeric_version(value, 2, 3, true)
            && major
                .and_then(|major| major.parse::<u64>().ok())
                .is_some_and(|major| major >= 3)
            && major.is_some_and(|major| major == "0" || !major.starts_with('0')),
        format!("{label} is not a supported Python version"),
    )
}

fn numeric_version(
    value: &str,
    min_parts: usize,
    max_parts: usize,
    leading_ok: bool,
) -> bool {
    let parts: Vec<_> = value.split('.').collect();
    (min_parts..=max_parts).contains(&parts.len())
        && parts.iter().all(|part| {
            !part.is_empty()
                && part.bytes().all(|byte| byte.is_ascii_digit())
                && (leading_ok || part == &"0" || !part.starts_with('0'))
        })
}

fn alignment(value: u32, label: &str) -> Result<(), Error> {
    ensure(
        matches!(
            value,
            1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096
        ),
        format!("{label} is not an allowed power-of-two alignment"),
    )
}

fn workload_parameters_digest(parameters: &WorkloadParameters) -> String {
    let payload = if let Some(narrow_tokens) = parameters.narrow_tokens {
        format!("{{\"narrow_tokens\":{narrow_tokens}}}").into_bytes()
    } else {
        b"{}".to_vec()
    };
    domain_sha256("imparo-program-workload-parameters-v1", &payload)
}

fn domain_sha256(domain: &str, payload: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(domain.as_bytes());
    hash.update([0]);
    hash.update((payload.len() as u64).to_le_bytes());
    hash.update(payload);
    let digest = hash.finalize();
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const EXAMPLE: &[u8] =
        include_bytes!("../../../schemas/examples/program-pack-v1.minimal.json");

    fn value() -> Value {
        serde_json::from_slice(EXAMPLE).unwrap()
    }

    #[allow(clippy::needless_pass_by_value)]
    fn parse_value(value: Value) -> Result<Manifest, Error> {
        Manifest::parse(&serde_json::to_vec(&value).unwrap())
    }

    fn two_group_manifest() -> Value {
        let mut root = value();
        let mut second_group = root["choice_groups"][0].clone();
        second_group["choice_group_id"] = json!("imparo.cuda.noop.second.v1");
        root["choice_groups"]
            .as_array_mut()
            .unwrap()
            .push(second_group);

        let mut second_variant = root["variants"][0].clone();
        second_variant["variant_id"] = json!("9".repeat(64));
        second_variant["config_id"] = json!("8".repeat(64));
        second_variant["choice_group_id"] = json!("imparo.cuda.noop.second.v1");
        root["variants"]
            .as_array_mut()
            .unwrap()
            .push(second_variant);
        root
    }

    #[test]
    fn parses_canonical_example() {
        let manifest = Manifest::parse(EXAMPLE).unwrap();
        assert_eq!(manifest.schema, 1);
        assert_eq!(manifest.target.sm, 86);
        assert_eq!(manifest.modules.len(), 1);
    }

    #[test]
    fn explicit_null_is_not_treated_as_an_omitted_optional_field() {
        let mut narrow = value();
        narrow["choice_groups"][0]["workload"]["parameters"]["narrow_tokens"] =
            Value::Null;
        assert!(parse_value(narrow).is_err());

        let mut cross_check = value();
        cross_check["choice_groups"][0]["cross_check"] = Value::Null;
        assert!(parse_value(cross_check).is_err());

        let mut one_of = value();
        one_of["variants"][0]["constraints"]["shapes"] = json!([{
            "slot": "input", "axis": 0, "min": 1, "max": 2,
            "multiple_of": 1, "one_of": null
        }]);
        assert!(parse_value(one_of).is_err());

        let mut extension = value();
        extension["variants"][0]["extension"] = Value::Null;
        assert!(parse_value(extension).is_err());
    }

    #[test]
    fn rejects_unknown_duplicate_truncated_and_oversize_inputs() {
        let unknown = String::from_utf8(EXAMPLE.to_vec()).unwrap().replacen(
            '{',
            "{\"unknown\":true,",
            1,
        );
        assert!(
            Manifest::parse(unknown.as_bytes())
                .unwrap_err()
                .to_string()
                .contains("unknown field")
        );

        let duplicate = String::from_utf8(EXAMPLE.to_vec()).unwrap().replacen(
            "\"schema\": 1,",
            "\"schema\": 1,\n  \"schema\": 1,",
            1,
        );
        assert!(
            Manifest::parse(duplicate.as_bytes())
                .unwrap_err()
                .to_string()
                .contains("duplicate field")
        );

        let nested_duplicate = String::from_utf8(EXAMPLE.to_vec()).unwrap().replacen(
            "\"min\": 1,",
            "\"min\": 1, \"min\": 1,",
            1,
        );
        assert!(
            Manifest::parse(nested_duplicate.as_bytes())
                .unwrap_err()
                .to_string()
                .contains("duplicate field")
        );

        let closing_brace = EXAMPLE
            .iter()
            .rposition(|byte| *byte == b'}')
            .expect("fixture has a closing brace");
        assert!(Manifest::parse(&EXAMPLE[..closing_brace]).is_err());
        assert!(
            Manifest::parse(&vec![b' '; MAX_MANIFEST_BYTES + 1])
                .unwrap_err()
                .to_string()
                .contains("exceeds")
        );
        assert!(
            Manifest::parse(&[0xff])
                .unwrap_err()
                .to_string()
                .contains("UTF-8")
        );
    }

    #[test]
    fn rejects_schema_patterns_bounds_and_constants() {
        let mut root = value();
        root["pack_id"] = json!("Bad.Id");
        assert!(parse_value(root).is_err());

        let mut root = value();
        root["modules"][0]["bytes"] = json!(268_435_457_u64);
        assert!(parse_value(root).is_err());

        let mut root = value();
        root["target"]["warp_size"] = json!(64);
        assert!(parse_value(root).is_err());

        let mut root = value();
        root["variants"][0]["symbol"] = json!("ip_not-content-addressed");
        assert!(parse_value(root).is_err());
    }

    #[test]
    fn rejects_ranges_extensions_and_workload_digest() {
        let mut root = value();
        root["engine_api"]["min"] = json!(2);
        root["engine_api"]["max_exclusive"] = json!(2);
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("empty or reversed")
        );

        let mut root = value();
        let extension = json!({"id":"imparo.cuda.ext","revision":1});
        root["required_extensions"] = json!([extension.clone()]);
        root["optional_extensions"] = json!([extension]);
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("overlap")
        );

        let mut root = value();
        root["choice_groups"][0]["workload"]["parameters_sha256"] =
            json!("0".repeat(64));
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("parameter digest")
        );
    }

    #[test]
    fn rejects_unsatisfiable_constraints_and_bad_effects() {
        let mut root = value();
        root["variants"][0]["constraints"]["shapes"] =
            json!([{"slot":"input","axis":0,"min":3,"max":3,"multiple_of":2}]);
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("no satisfiable")
        );

        let mut root = value();
        root["variants"][0]["effects"][0]["aliasing"] =
            json!([{"mode":"may_alias","target_slot":"missing"}]);
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("unknown slot")
        );
    }

    #[test]
    fn rejects_expression_division_overflow_depth_and_unknown_slots() {
        let mut root = value();
        root["variants"][0]["launch"]["grid"]["x"] = json!({
            "kind":"op","op":"ceil_div","args":[
                {"kind":"const","value":1},
                {"kind":"const","value":0}
            ]
        });
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("divides by zero")
        );

        let mut root = value();
        root["variants"][0]["launch"]["grid"]["x"] = json!({
            "kind":"op","op":"add","args":[
                {"kind":"const","value":9_007_199_254_740_991_u64},
                {"kind":"const","value":9_007_199_254_740_991_u64}
            ]
        });
        // This sum remains below u64; force multiplication to exceed it.
        root["variants"][0]["launch"]["grid"]["x"]["op"] = json!("mul");
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("overflows")
        );

        let mut expression = json!({"kind":"const","value":1});
        for _ in 0..8 {
            expression = json!({
                "kind":"op","op":"add","args":[expression,{"kind":"const","value":1}]
            });
        }
        let mut root = value();
        root["variants"][0]["launch"]["grid"]["x"] = expression;
        assert!(parse_value(root).unwrap_err().to_string().contains("depth"));

        let mut root = value();
        root["variants"][0]["launch"]["grid"]["x"] =
            json!({"kind":"slot","slot":"input"});
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("unknown unsigned slot")
        );
    }

    #[test]
    fn rejects_joint_asymmetry_dependency_cycles_and_entitlement_mismatch() {
        let mut root = two_group_manifest();
        root["choice_groups"][0]["joint_with"] = json!(["imparo.cuda.noop.second.v1"]);
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("not symmetric")
        );

        let mut root = two_group_manifest();
        let first = root["variants"][0]["variant_id"].clone();
        let second = root["variants"][1]["variant_id"].clone();
        root["variants"][0]["requires"] = json!([{"kind":"variant","id":second}]);
        root["variants"][1]["requires"] = json!([{"kind":"variant","id":first}]);
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("dependency cycle")
        );

        let mut root = value();
        root["required_entitlement_features"] = json!(["imparo.pro.pack"]);
        root["variants"][0]["required_entitlement_features"] =
            json!(["imparo.pro.pack"]);
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("community")
        );

        let mut root = value();
        root["distribution"] = json!("commercial");
        root["required_entitlement_features"] = json!(["imparo.pro.pack"]);
        assert!(
            parse_value(root)
                .unwrap_err()
                .to_string()
                .contains("feature sets differ")
        );
    }
}
