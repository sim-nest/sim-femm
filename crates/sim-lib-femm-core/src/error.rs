use sim_kernel::Symbol;
use thiserror::Error;

/// The error domain shared by every FEMM crate.
///
/// Each variant maps to a stable `femm` error category; [`From`] converts it
/// into a kernel [`sim_kernel::Error`] for protocol-level reporting.
#[derive(Debug, Error)]
pub enum FemmError {
    /// A referenced parameter is not bound in the active parameter set.
    #[error("UnknownFemmParameter: {0}")]
    UnknownFemmParameter(String),
    /// A required material was not found.
    #[error("MissingMaterial: {0}")]
    MissingMaterial(String),
    /// The requested physics kind is not supported.
    #[error("UnsupportedPhysics: {0}")]
    UnsupportedPhysics(String),
    /// A mesh exceeded a FEMM resource ceiling.
    #[error("MeshLimitExceeded: {0}")]
    MeshLimitExceeded(String),
    /// A solver failed to reach convergence.
    #[error("SolveDidNotConverge: {0}")]
    SolveDidNotConverge(String),
    /// Sensitivity (adjoint) data was requested but is unavailable.
    #[error("SensitivityUnavailable: {0}")]
    SensitivityUnavailable(String),
    /// A resource budget was exhausted.
    #[error("BudgetExceeded: {0}")]
    BudgetExceeded(String),
    /// Geometry input was malformed or could not be interpreted.
    #[error("InvalidGeometry: {0}")]
    InvalidGeometry(String),
    /// A field query fell outside the model's valid domain.
    #[error("FieldOutOfDomain: {0}")]
    FieldOutOfDomain(String),
    /// A CSR matrix violated its structural invariants.
    #[error("MalformedMatrix: {0}")]
    MalformedMatrix(String),
}

impl From<FemmError> for sim_kernel::Error {
    fn from(error: FemmError) -> Self {
        let category = match &error {
            FemmError::UnknownFemmParameter(_) => "unknown-parameter",
            FemmError::MissingMaterial(_) => "missing-material",
            FemmError::UnsupportedPhysics(_) => "unsupported-physics",
            FemmError::MeshLimitExceeded(_) => "mesh-limit",
            FemmError::SolveDidNotConverge(_) => "solve-did-not-converge",
            FemmError::SensitivityUnavailable(_) => "sensitivity-unavailable",
            FemmError::BudgetExceeded(_) => "budget-exceeded",
            FemmError::InvalidGeometry(_) => "invalid-geometry",
            FemmError::FieldOutOfDomain(_) => "field-out-of-domain",
            FemmError::MalformedMatrix(_) => "malformed-matrix",
        };
        sim_kernel::Error::domain_error(
            Symbol::new("femm"),
            Symbol::qualified("femm", category),
            error.to_string(),
        )
    }
}

/// Result type for FEMM operations, carrying a [`FemmError`] on failure.
pub type FemmResult<T> = std::result::Result<T, FemmError>;
