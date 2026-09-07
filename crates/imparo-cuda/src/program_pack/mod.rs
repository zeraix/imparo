//! CUDA-only Program Pack adapter layer. The shared model workflow remains unaware of
//! CUDA/Triton; only authenticated cubin bytes and engine-owned contract adapters cross
//! this boundary.

mod constraints;
mod launch;
mod loader;
pub mod profile_identity;
mod registry;
#[cfg(all(test, imparo_cuda_program_smoke))]
mod sm86_smoke;

pub use registry::{InstallReport, install};

/// Quiescent shutdown hook. Native synchronizes the sole stream and destroys Graph
/// state before unloading modules; failure keeps modules pinned.
pub fn shutdown() -> Result<(), String> {
    registry::shutdown()
}

impl imparo_backend::BackendPrograms for crate::CudaBackend {
    fn program_catalog_identity(&self) -> imparo_backend::ProgramCatalogIdentity {
        registry::catalog_identity()
    }

    fn program_choices(
        &self,
        facts: &imparo_backend::ModelFacts,
        profile: &imparo_backend::DeviceProfile,
    ) -> Vec<imparo_backend::ProgramChoiceDecl> {
        registry::choices(facts, profile)
    }

    fn bind_program_choice(
        &self,
        group: &str,
        variant: &[u8; 32],
    ) -> Result<(), String> {
        registry::bind(group, variant)
    }

    fn current_program_choice(&self, group: &str) -> Option<[u8; 32]> {
        registry::current(group)
    }

    fn freeze_program_catalog(&self) -> Result<(), String> {
        registry::freeze()
    }
}
