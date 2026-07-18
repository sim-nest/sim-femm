#![forbid(unsafe_code)]
//! Quality evidence for FEMM function calls.

use sim_kernel::{Cx, Symbol};
use sim_lib_femm_core::FemmResult;
use sim_lib_femm_post::{QuantitySpec, quantity};
use sim_lib_femm_query::{ModelCallable, resolve_excitation};
use sim_lib_femm_sensitiv::total_gradient;
use sim_lib_femm_solve::{GradientTrust, SolveCertificate, SteadySolve};

/// Quantity value, certificate, and optional total gradient for a completed solve.
#[derive(Clone, Debug)]
pub struct QualityAnswer {
    /// Scalar value of the requested quantity.
    pub value: f64,
    /// Certificate describing residual, convergence, and gradient trust.
    pub certificate: SolveCertificate,
    /// Gradient values and trust tag when a parameter list is supplied.
    pub gradient: Option<(Vec<f64>, GradientTrust)>,
}

/// Returns the requested quantity and the certificate for a completed solve.
///
/// Passing `Some(params)` for `wrt` also computes a trust-labelled total
/// gradient and annotates the returned certificate with its trust level.
/// Passing `None` skips gradient work.
pub fn quality(
    cx: &mut Cx,
    solve: &SteadySolve,
    quantity_spec: &QuantitySpec,
    wrt: Option<&[Symbol]>,
) -> FemmResult<QualityAnswer> {
    let excitation = resolve_excitation(cx, &solve.model, &solve.solution.params, quantity_spec)?;
    let value = quantity(&solve.solution, quantity_spec, &excitation)?;
    let mut certificate = solve.certificate.clone();
    let gradient = match wrt {
        None => None,
        Some(params) => {
            let callable = ModelCallable {
                model: solve.model.clone(),
            };
            let mut solve_for_gradient = SteadySolve {
                model: solve.model.clone(),
                solution: solve.solution.clone(),
                factor: solve.factor.clone(),
                certificate: solve.certificate.clone(),
            };
            let result = total_gradient(
                cx,
                &callable,
                &mut solve_for_gradient,
                std::slice::from_ref(quantity_spec),
                params,
            )?;
            let values = result.gradient.into_iter().next().unwrap_or_default();
            let trust = result
                .trust
                .into_iter()
                .next()
                .unwrap_or(GradientTrust::FiniteDifferenceOnly);
            certificate.set_gradient_trust(trust.clone());
            Some((values, trust))
        }
    };
    Ok(QualityAnswer {
        value,
        certificate,
        gradient,
    })
}
