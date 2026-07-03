//! Residual and convergence certificates for completed FEMM solves.

use sim_kernel::{Claim, Cx, Datum, NumberLiteral, Ref, Symbol};
use sim_lib_femm_core::{FemmError, FemmResult};
use sim_lib_femm_flow::{FemmSolveEvent, SolveDiagnostics};
use sim_lib_femm_post::FemmSolution;

/// Residual and convergence certificate attached to a completed FEMM solve.
///
/// The certificate carries a kernel [`Claim`] so callers can store or
/// re-verify solve quality independently of the solution vector.
#[derive(Clone, Debug)]
pub struct SolveCertificate {
    /// Solver method tag, for example `femm-direct` or `femm-ptc`.
    pub method: String,
    /// Whether the solver converged within its limits.
    pub converged: bool,
    /// Final absolute residual norm.
    pub final_residual: f64,
    /// Iteration count, one for direct linear solves.
    pub iterations: u32,
    /// Kernel claim carrying method, residual, and solution fingerprint.
    pub claim: Claim,
    /// Gradient trust level, absent when the solve emits no gradient certificate.
    pub gradient_trust: Option<GradientTrust>,
}

/// Trust level for an adjoint gradient associated with a solve certificate.
#[derive(Clone, Debug, PartialEq)]
pub enum GradientTrust {
    /// Exact adjoint verified against finite difference within `tol`.
    AdjointVerified {
        /// Verification tolerance.
        tol: f64,
    },
    /// Finite difference only; no exact adjoint path is available.
    FiniteDifferenceOnly,
    /// Adjoint path exists but has not been verified for this solve.
    AdjointUnverified,
}

impl SolveCertificate {
    /// Records the trust level for gradients derived from this solve.
    pub fn set_gradient_trust(&mut self, trust: GradientTrust) {
        self.gradient_trust = Some(trust);
    }
}

pub(crate) fn make_linear_certificate(
    cx: &mut Cx,
    solution: &FemmSolution,
) -> FemmResult<SolveCertificate> {
    let fingerprint = solution_fingerprint(solution);
    let claim = Claim::content_object(
        cx.datum_store_mut(),
        Ref::Symbol(Symbol::qualified("femm-solve", solution.id.0.to_string())),
        Symbol::qualified("femm", "solve-certificate"),
        certificate_datum(
            "femm-direct",
            true,
            solution.diagnostics.final_residual,
            1,
            solution.id.0,
            fingerprint,
        ),
    )
    .map_err(|err| FemmError::SolveDidNotConverge(err.to_string()))?;

    Ok(SolveCertificate {
        method: "femm-direct".to_owned(),
        converged: true,
        final_residual: solution.diagnostics.final_residual,
        iterations: 1,
        claim,
        gradient_trust: None,
    })
}

pub(crate) fn make_ptc_certificate(
    cx: &mut Cx,
    diagnostics: &SolveDiagnostics,
    solution_id: u64,
    u: &[f64],
) -> FemmResult<SolveCertificate> {
    let iterations = ptc_step_count(diagnostics);
    let fingerprint = state_fingerprint(solution_id, u);
    let claim = Claim::content_object(
        cx.datum_store_mut(),
        Ref::Symbol(Symbol::qualified("femm-solve", solution_id.to_string())),
        Symbol::qualified("femm", "solve-certificate"),
        certificate_datum(
            "femm-ptc",
            diagnostics.converged,
            diagnostics.final_residual,
            iterations,
            solution_id,
            fingerprint,
        ),
    )
    .map_err(|err| FemmError::SolveDidNotConverge(err.to_string()))?;

    Ok(SolveCertificate {
        method: "femm-ptc".to_owned(),
        converged: diagnostics.converged,
        final_residual: diagnostics.final_residual,
        iterations,
        claim,
        gradient_trust: None,
    })
}

pub(crate) fn rebuild_certificate_claim(
    cx: &mut Cx,
    solution: &FemmSolution,
    certificate: &SolveCertificate,
) -> FemmResult<Claim> {
    Claim::content_object(
        cx.datum_store_mut(),
        Ref::Symbol(Symbol::qualified("femm-solve", solution.id.0.to_string())),
        Symbol::qualified("femm", "solve-certificate"),
        certificate_datum(
            &certificate.method,
            certificate.converged,
            certificate.final_residual,
            certificate.iterations,
            solution.id.0,
            solution_fingerprint(solution),
        ),
    )
    .map_err(|err| FemmError::SolveDidNotConverge(err.to_string()))
}

fn certificate_datum(
    method: &str,
    converged: bool,
    final_residual: f64,
    iterations: u32,
    solution_id: u64,
    fingerprint: u64,
) -> Datum {
    Datum::Node {
        tag: Symbol::qualified("femm", "solve-certificate-v1"),
        fields: vec![
            field("method", Datum::String(method.to_owned())),
            field("converged", Datum::Bool(converged)),
            field("residual", f64_datum(final_residual)),
            field("iterations", u64_datum(u64::from(iterations))),
            field("solution-id", u64_datum(solution_id)),
            field("solution-fingerprint", u64_datum(fingerprint)),
        ],
    }
}

fn solution_fingerprint(solution: &FemmSolution) -> u64 {
    state_fingerprint(solution.id.0, &solution.u)
}

fn state_fingerprint(seed: u64, u: &[f64]) -> u64 {
    u.iter()
        .fold(seed, |acc, value| acc.wrapping_add(value.to_bits()))
}

fn ptc_step_count(diagnostics: &SolveDiagnostics) -> u32 {
    diagnostics
        .events
        .iter()
        .filter(|event| matches!(event, FemmSolveEvent::Step { .. }))
        .count() as u32
}

fn field(name: &str, value: Datum) -> (Symbol, Datum) {
    (Symbol::new(name), value)
}

fn f64_datum(value: f64) -> Datum {
    Datum::Number(NumberLiteral {
        domain: Symbol::qualified("numbers", "f64"),
        canonical: value.to_string(),
    })
}

fn u64_datum(value: u64) -> Datum {
    Datum::Number(NumberLiteral {
        domain: Symbol::qualified("core", "u64"),
        canonical: value.to_string(),
    })
}
