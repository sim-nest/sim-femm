#![forbid(unsafe_code)]
//! Trust-labelled scalar gradient facade for FEMM model queries.

use sim_kernel::{Cx, Symbol};
use sim_lib_femm_core::{FemmError, FemmLimits, FemmResult, ParamSet, value_as_f64};
use sim_lib_femm_query::{
    FemmCall, FemmCallable, ModelCallable, OutputQuery, resolve_model_params,
};
use sim_lib_femm_solve::GradientTrust;

use crate::{SensitivityPath, adjoint_gradient};

/// Trust-labelled gradient values for a scalar FEMM model query.
#[derive(Clone, Debug)]
pub struct GradientAnswer {
    /// One derivative per requested parameter.
    pub values: Vec<(Symbol, f64)>,
    /// Aggregate trust for `values`.
    pub trust: GradientTrust,
}

/// Stable public label for a [`GradientTrust`] value.
pub fn gradient_trust_label(trust: &GradientTrust) -> &'static str {
    match trust {
        GradientTrust::AdjointVerified { .. } => "adjoint-verified",
        GradientTrust::AdjointUnverified => "adjoint-unverified",
        GradientTrust::FiniteDifferenceOnly => "finite-difference-only",
    }
}

/// Computes a scalar query gradient, trying exact adjoint before finite difference.
///
/// This is the common gradient facade for public function gradients,
/// certificate-backed quality gradients, total-gradient rows, and the
/// `femm-adjoint` numeric differentiator plugin.
pub fn gradient_answer(
    cx: &mut Cx,
    callable: &ModelCallable,
    query: OutputQuery,
    params: ParamSet,
    wrt: &[Symbol],
) -> FemmResult<GradientAnswer> {
    if matches!(query, OutputQuery::Field(_) | OutputQuery::Solution) {
        return Err(FemmError::SensitivityUnavailable(
            "FEMM gradients require a scalar quantity query".to_owned(),
        ));
    }
    let params = resolve_model_params(&callable.model, params)?;
    match adjoint_gradient(cx, callable, query.clone(), params.clone(), wrt) {
        Ok((values, SensitivityPath::AdjointExact)) => Ok(GradientAnswer {
            values,
            trust: GradientTrust::AdjointVerified { tol: 1.0e-5 },
        }),
        Ok((_values, path)) => {
            cx.push_info(format!(
                "femm gradient fallback trust=finite-difference-only path={path:?}"
            ));
            finite_difference_answer(cx, callable, query, &params, wrt)
        }
        Err(err) => {
            cx.push_info(format!(
                "femm gradient fallback trust=finite-difference-only reason={err}"
            ));
            finite_difference_answer(cx, callable, query, &params, wrt)
        }
    }
}

fn finite_difference_answer(
    cx: &mut Cx,
    callable: &ModelCallable,
    query: OutputQuery,
    params: &ParamSet,
    wrt: &[Symbol],
) -> FemmResult<GradientAnswer> {
    let mut values = Vec::with_capacity(wrt.len());
    for symbol in wrt {
        let base = params
            .get(symbol)
            .ok_or_else(|| FemmError::UnknownFemmParameter(symbol.to_string()))
            .and_then(|value| value_as_f64(cx, value))?;
        if !base.is_finite() {
            return Err(FemmError::SensitivityUnavailable(format!(
                "non-finite FEMM parameter {symbol}"
            )));
        }
        let step = fd_step(base);
        let plus = replace_param(cx, params, symbol, base + step)?;
        let minus = replace_param(cx, params, symbol, base - step)?;
        let q_plus = evaluate_scalar_query(cx, callable, query.clone(), plus)?;
        let q_minus = evaluate_scalar_query(cx, callable, query.clone(), minus)?;
        let value = (q_plus - q_minus) / (2.0 * step);
        if !value.is_finite() {
            return Err(FemmError::SensitivityUnavailable(format!(
                "finite difference produced non-finite gradient for {symbol}"
            )));
        }
        values.push((symbol.clone(), value));
    }
    Ok(GradientAnswer {
        values,
        trust: GradientTrust::FiniteDifferenceOnly,
    })
}

fn evaluate_scalar_query(
    cx: &mut Cx,
    callable: &ModelCallable,
    query: OutputQuery,
    params: ParamSet,
) -> FemmResult<f64> {
    let eval = callable.eval(
        cx,
        FemmCall {
            params,
            query,
            want_grad: None,
            limits: FemmLimits::default(),
        },
    )?;
    value_as_f64(cx, &eval.value)
}

fn replace_param(
    cx: &mut Cx,
    params: &ParamSet,
    symbol: &Symbol,
    value: f64,
) -> FemmResult<ParamSet> {
    let replacement = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
        .map_err(|err| FemmError::SensitivityUnavailable(err.to_string()))?;
    let mut entries = params.entries.clone();
    let Some((_, current)) = entries.iter_mut().find(|(name, _)| name == symbol) else {
        return Err(FemmError::UnknownFemmParameter(symbol.to_string()));
    };
    *current = replacement;
    Ok(ParamSet::new(entries))
}

fn fd_step(base: f64) -> f64 {
    1.0e-6 * base.abs().max(1.0)
}
