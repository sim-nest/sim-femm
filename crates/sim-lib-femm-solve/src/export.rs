//! Export metadata for completed FEMM solves.

use sim_kernel::{Claim, ContentId, Cx};
use sim_lib_femm_core::{FemmResult, StableId};

use crate::{GradientTrust, SteadySolve, certificate::rebuild_certificate_claim};

/// Export metadata for a completed FEMM solve.
///
/// This is an open metadata record over existing kernel claim data; it does not
/// add a kernel type or closed registry.
#[derive(Clone, Debug)]
pub struct SolveExportRecord {
    /// Stable identifier of the solution vector.
    pub solution_id: StableId,
    /// Physics kind formatted for logs and external metadata.
    pub physics: String,
    /// Solver method tag, for example `femm-direct` or `femm-ptc`.
    pub method: String,
    /// Whether the solve converged within its configured limits.
    pub converged: bool,
    /// Final absolute residual norm.
    pub final_residual: f64,
    /// Iteration count reported by the solve certificate.
    pub iterations: u32,
    /// Human-readable gradient trust tag.
    pub gradient_trust: String,
    /// Stable content key of the certificate claim.
    pub certificate_claim_key: String,
}

impl From<&SteadySolve> for SolveExportRecord {
    fn from(solve: &SteadySolve) -> Self {
        let cert = &solve.certificate;
        Self {
            solution_id: solve.solution.id,
            physics: format!("{:?}", solve.solution.physics),
            method: cert.method.clone(),
            converged: cert.converged,
            final_residual: cert.final_residual,
            iterations: cert.iterations,
            gradient_trust: gradient_trust_label(cert.gradient_trust.as_ref()),
            certificate_claim_key: claim_key(&cert.claim),
        }
    }
}

/// Rebuilds the solve certificate claim from the solution fingerprint.
///
/// The returned claim is content-equivalent to the certificate claim carried on
/// the solve and can be re-keyed through [`Claim::content_id`].
pub fn certificate_claim(cx: &mut Cx, solve: &SteadySolve) -> FemmResult<Claim> {
    rebuild_certificate_claim(cx, &solve.solution, &solve.certificate)
}

fn gradient_trust_label(trust: Option<&GradientTrust>) -> String {
    match trust {
        None => "none".to_owned(),
        Some(GradientTrust::AdjointVerified { tol }) => format!("adjoint-verified-{tol:.0e}"),
        Some(GradientTrust::FiniteDifferenceOnly) => "fd-only".to_owned(),
        Some(GradientTrust::AdjointUnverified) => "adjoint-unverified".to_owned(),
    }
}

fn claim_key(claim: &Claim) -> String {
    claim
        .canonical_datum()
        .content_id()
        .map(|id| content_id_hex(&id))
        .unwrap_or_else(|err| format!("invalid-claim-{}", sanitize_error(&err)))
}

fn content_id_hex(id: &ContentId) -> String {
    id.bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sanitize_error(err: &impl std::fmt::Display) -> String {
    err.to_string()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect()
}
