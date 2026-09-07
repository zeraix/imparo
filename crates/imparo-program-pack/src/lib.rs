//! Trust boundary for signed, data-only CUDA Program Packs.
//!
//! This crate deliberately stops before CUDA module loading. It validates immutable
//! bytes, trust policy, and content-addressed installation so later backend code can
//! consume only admitted objects. A pack signature is not a correctness receipt or an
//! entitlement, and no type in this crate grants either authority.

pub mod extensions;
pub mod identity;
pub mod install;
pub mod manifest;
pub mod policy;
pub mod trust;

pub use extensions::{ExtensionRegistry, ExtensionResolution};
pub use install::{Admission, AdmittedModule, AdmittedPack, ChannelState, Installer};
pub use manifest::Manifest;
pub use policy::Policy;
pub use trust::{DistributionScope, TrustStore, TrustedKey};

/// Program Pack ABI implemented by the frozen v1 schema.
pub const PROGRAM_PACK_ABI: u32 = 1;
