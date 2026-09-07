//! Stable numerical-route identities shared by backends and receipt validators.
//!
//! These types describe *correctness semantics*, not tuning candidates. A route may
//! only be admitted when a separate correctness receipt binds this full identity to
//! the running model, device, driver and backend binary.

/// How an implementation claims to relate numerically to a named reference.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum NumericalClass {
    /// Every compared output byte must equal the named reference contract.
    BitExact {
        /// Stable reference name, for example a committed baseline route.
        reference: String,
        /// Version of the comparison contract, not the implementation version.
        contract_version: u32,
    },
    /// The route must pass every gate in the named, versioned suite.
    GateBounded {
        /// Stable gate-suite name. Thresholds live in that versioned suite.
        gate_suite: String,
        /// Version of the suite contract.
        contract_version: u32,
    },
    /// A development-only route which a correctness receipt must never admit.
    DiagnosticOnly,
}

impl NumericalClass {
    /// Whether a passing correctness receipt is allowed to grant this class authority.
    #[must_use]
    pub fn is_receipt_admissible(&self) -> bool {
        !matches!(self, Self::DiagnosticOnly)
            && match self {
                Self::BitExact {
                    reference,
                    contract_version,
                } => !reference.trim().is_empty() && *contract_version != 0,
                Self::GateBounded {
                    gate_suite,
                    contract_version,
                } => !gate_suite.trim().is_empty() && *contract_version != 0,
                Self::DiagnosticOnly => false,
            }
    }
}

/// Complete stable identity of one numerical implementation route.
///
/// `domain_sha256` binds the supported shape/type domain. `parameters_sha256` binds
/// reduction ownership, tile geometry, Stream-K topology and other values which may
/// change floating-point association without changing the high-level operation name.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RouteKey {
    pub backend: String,
    pub operation: String,
    pub implementation: String,
    pub implementation_version: u32,
    pub selector_version: u32,
    pub domain_sha256: String,
    pub parameters_sha256: String,
    pub numerical_class: NumericalClass,
}

/// Structural reason a route identity cannot be used for receipt admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteKeyError {
    EmptyBackend,
    EmptyOperation,
    EmptyImplementation,
    ZeroImplementationVersion,
    ZeroSelectorVersion,
    InvalidDomainSha256,
    InvalidParametersSha256,
    InvalidNumericalClass,
}

impl RouteKey {
    /// Validate the route's stable identity without consulting a backend or filesystem.
    pub fn validate(&self) -> Result<(), RouteKeyError> {
        if self.backend.trim().is_empty() {
            return Err(RouteKeyError::EmptyBackend);
        }
        if self.operation.trim().is_empty() {
            return Err(RouteKeyError::EmptyOperation);
        }
        if self.implementation.trim().is_empty() {
            return Err(RouteKeyError::EmptyImplementation);
        }
        if self.implementation_version == 0 {
            return Err(RouteKeyError::ZeroImplementationVersion);
        }
        if self.selector_version == 0 {
            return Err(RouteKeyError::ZeroSelectorVersion);
        }
        if !is_sha256_hex(&self.domain_sha256) {
            return Err(RouteKeyError::InvalidDomainSha256);
        }
        if !is_sha256_hex(&self.parameters_sha256) {
            return Err(RouteKeyError::InvalidParametersSha256);
        }
        if !self.numerical_class.is_receipt_admissible() {
            return Err(RouteKeyError::InvalidNumericalClass);
        }
        Ok(())
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::{NumericalClass, RouteKey, RouteKeyError};

    fn hash(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn route() -> RouteKey {
        RouteKey {
            backend: "cuda".into(),
            operation: "attention.decode".into(),
            implementation: "d256.vec.q4".into(),
            implementation_version: 1,
            selector_version: 1,
            domain_sha256: hash('a'),
            parameters_sha256: hash('b'),
            numerical_class: NumericalClass::GateBounded {
                gate_suite: "cuda-step6".into(),
                contract_version: 1,
            },
        }
    }

    #[test]
    fn valid_route_is_admissible() {
        assert_eq!(route().validate(), Ok(()));
    }

    #[test]
    fn diagnostic_route_is_never_receipt_admissible() {
        let mut route = route();
        route.numerical_class = NumericalClass::DiagnosticOnly;
        assert_eq!(route.validate(), Err(RouteKeyError::InvalidNumericalClass));
    }

    #[test]
    fn class_declaration_requires_name_and_nonzero_version() {
        let mut route = route();
        route.numerical_class = NumericalClass::BitExact {
            reference: String::new(),
            contract_version: 1,
        };
        assert_eq!(route.validate(), Err(RouteKeyError::InvalidNumericalClass));
        route.numerical_class = NumericalClass::GateBounded {
            gate_suite: "suite".into(),
            contract_version: 0,
        };
        assert_eq!(route.validate(), Err(RouteKeyError::InvalidNumericalClass));
    }

    #[test]
    fn hashes_are_lowercase_full_sha256() {
        let mut route = route();
        route.domain_sha256 = hash('A');
        assert_eq!(route.validate(), Err(RouteKeyError::InvalidDomainSha256));
        route.domain_sha256 = hash('a');
        route.parameters_sha256 = "abcd".into();
        assert_eq!(
            route.validate(),
            Err(RouteKeyError::InvalidParametersSha256)
        );
    }

    #[test]
    fn numerical_identity_includes_domain_parameters_and_class() {
        let baseline = route();
        let mut changed = baseline.clone();
        changed.parameters_sha256 = hash('c');
        assert_ne!(baseline, changed);
        changed = baseline.clone();
        changed.domain_sha256 = hash('d');
        assert_ne!(baseline, changed);
        changed = baseline.clone();
        changed.numerical_class = NumericalClass::BitExact {
            reference: "cuda-safe-default".into(),
            contract_version: 1,
        };
        assert_ne!(baseline, changed);
    }
}
