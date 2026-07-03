//! Total-gradient pipeline over solved FEMM quantities.

use sim_kernel::{Cx, Symbol};
use sim_lib_femm_core::{FemmError, FemmLimits, FemmResult, ParamSet, value_as_f64};
use sim_lib_femm_function::{ModelCallable, OutputQuery};
use sim_lib_femm_post::{QuantitySpec, quantity};
use sim_lib_femm_solve::{GradientTrust, SteadySolve, solve_steady};

use crate::{SensitivityPath, adjoint_gradient, nonlinear_adjoint::nonlinear_adjoint_gradient};

const ADJOINT_VERIFY_TOL: f64 = 1.0e-5;
const FD_STEP_SCALE: f64 = 1.0e-6;

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
    let params = resolve_params(callable, solve.solution.params.clone())?;
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
        match adjoint_gradient(
            cx,
            callable,
            OutputQuery::Quantity(quantity_spec.clone()),
            params.clone(),
            wrt,
        ) {
            Ok((adjoint, SensitivityPath::AdjointExact)) => {
                gradient.push(ordered_gradient(&adjoint, wrt)?);
                trust.push(GradientTrust::AdjointVerified {
                    tol: ADJOINT_VERIFY_TOL,
                });
            }
            Ok(_) | Err(_) => {
                gradient.push(finite_difference_gradient(
                    cx,
                    callable,
                    &params,
                    quantity_spec,
                    wrt,
                )?);
                trust.push(GradientTrust::FiniteDifferenceOnly);
            }
        }
    }

    let result = TotalGradientResult { gradient, trust };
    if let Some(trust) = result.certificate_trust() {
        solve.certificate.set_gradient_trust(trust);
    }
    Ok(result)
}

fn finite_difference_gradient(
    cx: &mut Cx,
    callable: &ModelCallable,
    params: &ParamSet,
    quantity_spec: &QuantitySpec,
    wrt: &[Symbol],
) -> FemmResult<Vec<f64>> {
    wrt.iter()
        .map(|symbol| {
            let base = params
                .get(symbol)
                .ok_or_else(|| FemmError::UnknownFemmParameter(symbol.to_string()))
                .and_then(|value| value_as_f64(cx, value))?;
            let step = fd_step(base);
            let plus = replace_param(cx, params, symbol, base + step)?;
            let minus = replace_param(cx, params, symbol, base - step)?;
            let q_plus = evaluate_quantity(cx, callable, &plus, quantity_spec)?;
            let q_minus = evaluate_quantity(cx, callable, &minus, quantity_spec)?;
            let value = (q_plus - q_minus) / (2.0 * step);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(FemmError::SensitivityUnavailable(format!(
                    "finite difference produced non-finite gradient for {symbol}"
                )))
            }
        })
        .collect()
}

fn evaluate_quantity(
    cx: &mut Cx,
    callable: &ModelCallable,
    params: &ParamSet,
    quantity_spec: &QuantitySpec,
) -> FemmResult<f64> {
    let solved = solve_steady(cx, &callable.model, params, &FemmLimits::default(), None)?;
    quantity(&solved.solution, quantity_spec)
}

fn resolve_params(callable: &ModelCallable, params: ParamSet) -> FemmResult<ParamSet> {
    let mut entries = params.entries;
    for input in &callable.model.inputs {
        if entries.iter().all(|(name, _)| name != &input.name) {
            let Some(default) = &input.default else {
                return Err(FemmError::UnknownFemmParameter(input.name.to_string()));
            };
            entries.push((input.name.clone(), default.clone()));
        }
    }
    Ok(ParamSet::new(entries))
}

fn replace_param(
    cx: &mut Cx,
    params: &ParamSet,
    symbol: &Symbol,
    value: f64,
) -> FemmResult<ParamSet> {
    let mut entries = params.entries.clone();
    let replacement = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
        .map_err(|err| FemmError::SensitivityUnavailable(err.to_string()))?;
    if let Some((_, current)) = entries.iter_mut().find(|(name, _)| name == symbol) {
        *current = replacement;
    } else {
        entries.push((symbol.clone(), replacement));
    }
    Ok(ParamSet::new(entries))
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

fn fd_step(base: f64) -> f64 {
    FD_STEP_SCALE * base.abs().max(1.0)
}
