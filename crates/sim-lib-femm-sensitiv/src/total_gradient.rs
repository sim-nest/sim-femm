//! Total-gradient pipeline over solved FEMM quantities.

use sim_kernel::{Cx, Symbol};
use sim_lib_femm_core::{FemmError, FemmResult};
use sim_lib_femm_post::QuantitySpec;
use sim_lib_femm_query::{ModelCallable, OutputQuery, resolve_model_params};
use sim_lib_femm_solve::{GradientTrust, SteadySolve};

use crate::{gradient_answer, nonlinear_adjoint::nonlinear_adjoint_gradient};

/// Result of a total-gradient evaluation over a set of quantities and params.
#[derive(Clone, Debug)]
pub struct TotalGradientResult {
    /// `gradient[i][j] = d(quantities[i]) / d(wrt[j])`.
    pub gradient: Vec<Vec<f64>>,
    /// Per-quantity trust level, aligned with the quantities slice.
    pub trust: Vec<GradientTrust>,
}

impl TotalGradientResult {
    /// Returns the least-trusted level that should be attached to a solve certificate.
    pub fn certificate_trust(&self) -> Option<GradientTrust> {
        if self
            .trust
            .iter()
            .any(|trust| matches!(trust, GradientTrust::FiniteDifferenceOnly))
        {
            return Some(GradientTrust::FiniteDifferenceOnly);
        }
        self.trust.iter().find_map(|trust| match trust {
            GradientTrust::AdjointVerified { tol } => {
                Some(GradientTrust::AdjointVerified { tol: *tol })
            }
            GradientTrust::AdjointUnverified => Some(GradientTrust::AdjointUnverified),
            GradientTrust::FiniteDifferenceOnly => None,
        })
    }
}

/// Computes a finite total gradient for every requested quantity and parameter.
///
/// The exact linear adjoint path is tried first. Quantities that do not have an
/// exact adjoint result fall back to parameter-level finite differences, and the
/// supplied solve certificate is annotated with the aggregate trust level.
pub fn total_gradient(
    cx: &mut Cx,
    callable: &ModelCallable,
    solve: &mut SteadySolve,
    quantities: &[QuantitySpec],
    wrt: &[Symbol],
) -> FemmResult<TotalGradientResult> {
    let params = resolve_model_params(&callable.model, solve.solution.params.clone())?;
    let mut gradient = Vec::with_capacity(quantities.len());
    let mut trust = Vec::with_capacity(quantities.len());
    let nonlinear = solve.solution.diagnostics.method == Symbol::new("femm-ptc");

    for quantity_spec in quantities {
        if nonlinear {
            let (row, row_trust) =
                nonlinear_adjoint_gradient(cx, callable, solve, &params, quantity_spec, wrt)?;
            gradient.push(row);
            trust.push(row_trust);
            continue;
        }
        let answer = gradient_answer(
            cx,
            callable,
            OutputQuery::Quantity(quantity_spec.clone()),
            params.clone(),
            wrt,
        )?;
        gradient.push(ordered_gradient(&answer.values, wrt)?);
        trust.push(answer.trust);
    }

    let result = TotalGradientResult { gradient, trust };
    if let Some(trust) = result.certificate_trust() {
        solve.certificate.set_gradient_trust(trust);
    }
    Ok(result)
}

fn ordered_gradient(gradient: &[(Symbol, f64)], wrt: &[Symbol]) -> FemmResult<Vec<f64>> {
    wrt.iter()
        .map(|symbol| {
            gradient
                .iter()
                .find(|(name, _)| name == symbol)
                .map(|(_, value)| *value)
                .ok_or_else(|| {
                    FemmError::SensitivityUnavailable(format!(
                        "adjoint gradient omitted parameter {symbol}"
                    ))
                })
        })
        .collect()
}
