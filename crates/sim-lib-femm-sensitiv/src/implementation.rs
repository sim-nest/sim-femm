#![forbid(unsafe_code)]
//! Gradient computation and differentiator registration for FEMM quantities.
//!
//! Defines the sensitivity-path selection and the entry point that computes
//! parameter gradients of a model quantity, registering FEMM as a runtime
//! differentiator.

use std::sync::{Arc, OnceLock};

use sim_kernel::{Cx, Error, Expr, Result as KernelResult, Symbol, Value};
use sim_lib_femm_core::{FemmResult, ParamSet, normalize_femm_expr};
use sim_lib_femm_material::{Boundary, BoundaryKind, Material, MeshPolicy, Source};
use sim_lib_femm_mesh::FemmModel;
use sim_lib_femm_post::QuantitySpec;
use sim_lib_femm_query::{FemmFuncPayload, ModelCallable, OutputQuery, resolve_model_params};
use sim_lib_numbers_func::Func;
use sim_lib_numbers_numeric::{
    DiffOpts, Differentiator, NumericKind, NumericPlugin, register_differentiator,
};

use crate::expr_eval::{direct_expr_derivative, reverse_expr_gradient};
use crate::sensitivity_solve::built_in_quantity_gradient;
use crate::{gradient_answer, gradient_trust_label};

/// Which differentiation route produced a sensitivity result.
///
/// Reported alongside every gradient so callers can tell an exact derivative
/// from a fallback; an `Unavailable` path means no gradient could be computed.
/// See the [crate README](index.html).
///
/// # Examples
///
/// ```
/// use sim_lib_femm_sensitiv::SensitivityPath;
///
/// // Exact and approximate paths are distinguishable by the caller.
/// assert_ne!(SensitivityPath::AdjointExact, SensitivityPath::FiniteDifference);
/// assert_eq!(SensitivityPath::Unavailable, SensitivityPath::Unavailable);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SensitivityPath {
    /// Exact reverse-mode (adjoint) derivative.
    AdjointExact,
    /// Exact forward-mode (direct/dual) derivative.
    DirectExact,
    /// Numerical finite-difference approximation.
    FiniteDifference,
    /// No gradient is available for this query.
    Unavailable,
}

/// Computes parameter sensitivities of a model quantity by a forward path.
///
/// Returns one `(parameter, derivative)` pair per entry in `wrt`, together with
/// the [`SensitivityPath`] taken. Custom expression quantities differentiate
/// exactly via dual numbers; built-in quantities use the built-in quantity
/// gradient; field and solution queries are reported as
/// [`SensitivityPath::Unavailable`].
pub fn gradient(
    cx: &mut Cx,
    callable: &ModelCallable,
    query: OutputQuery,
    params: ParamSet,
    wrt: &[Symbol],
) -> FemmResult<(Vec<(Symbol, f64)>, SensitivityPath)> {
    let params = resolve_model_params(&callable.model, params)?;
    if matches!(query, OutputQuery::Field(_) | OutputQuery::Solution) {
        return Ok((Vec::new(), SensitivityPath::Unavailable));
    }
    if let OutputQuery::Quantity(sim_lib_femm_post::QuantitySpec::Custom { expr, .. }) = &query {
        let gradient = wrt
            .iter()
            .map(|symbol| {
                Ok((
                    symbol.clone(),
                    direct_expr_derivative(cx, expr, &params, symbol)?,
                ))
            })
            .collect::<FemmResult<Vec<_>>>()?;
        return Ok((gradient, SensitivityPath::DirectExact));
    }
    let (zero, path) =
        exact_if_parameter_independent(&callable.model, wrt, SensitivityPath::DirectExact)?;
    if path == SensitivityPath::DirectExact {
        return Ok((zero, path));
    }
    let OutputQuery::Quantity(spec) = query else {
        return Ok((Vec::new(), SensitivityPath::Unavailable));
    };
    built_in_quantity_gradient(cx, callable, &spec, params, wrt)
        .map(|gradient| (gradient, SensitivityPath::DirectExact))
}

/// Computes parameter sensitivities of a model quantity by an adjoint path.
///
/// The reverse-mode counterpart to [`gradient`]: custom expression quantities
/// differentiate exactly via a reverse AD tape, while built-in quantities reuse
/// the built-in quantity gradient. Returns the gradient and the
/// [`SensitivityPath`] taken.
pub fn adjoint_gradient(
    cx: &mut Cx,
    callable: &ModelCallable,
    query: OutputQuery,
    params: ParamSet,
    wrt: &[Symbol],
) -> FemmResult<(Vec<(Symbol, f64)>, SensitivityPath)> {
    let params = resolve_model_params(&callable.model, params)?;
    if matches!(query, OutputQuery::Field(_) | OutputQuery::Solution) {
        return Ok((Vec::new(), SensitivityPath::Unavailable));
    }
    if let OutputQuery::Quantity(sim_lib_femm_post::QuantitySpec::Custom { expr, .. }) = &query {
        return reverse_expr_gradient(cx, expr, &params, wrt)
            .map(|gradient| (gradient, SensitivityPath::AdjointExact));
    }
    let (zero, path) =
        exact_if_parameter_independent(&callable.model, wrt, SensitivityPath::AdjointExact)?;
    if path == SensitivityPath::AdjointExact {
        return Ok((zero, path));
    }
    let OutputQuery::Quantity(spec) = query else {
        return Ok((Vec::new(), SensitivityPath::Unavailable));
    };
    built_in_quantity_gradient(cx, callable, &spec, params, wrt)
        .map(|gradient| (gradient, SensitivityPath::AdjointExact))
}

fn exact_if_parameter_independent(
    model: &FemmModel,
    wrt: &[Symbol],
    path: SensitivityPath,
) -> FemmResult<(Vec<(Symbol, f64)>, SensitivityPath)> {
    for symbol in wrt {
        if model_uses_symbol(model, symbol) {
            return Ok((Vec::new(), SensitivityPath::Unavailable));
        }
    }
    Ok((
        wrt.iter().cloned().map(|symbol| (symbol, 0.0)).collect(),
        path,
    ))
}

struct FemmAdjointPlugin;

impl NumericPlugin for FemmAdjointPlugin {
    fn name(&self) -> Symbol {
        Symbol::new("femm-adjoint")
    }

    fn kind(&self) -> NumericKind {
        NumericKind::Differentiator
    }
}

impl Differentiator for FemmAdjointPlugin {
    fn diff_at(
        &self,
        cx: &mut Cx,
        f: &Func,
        var: &Symbol,
        point: &Value,
        _opt: DiffOpts,
    ) -> KernelResult<Value> {
        let payload = f
            .metadata
            .payload
            .as_ref()
            .and_then(|value| value.object().downcast_ref::<FemmFuncPayload>())
            .ok_or_else(|| {
                sim_kernel::Error::Eval("femm-adjoint requires a FEMM function payload".to_owned())
            })?;
        let callable = ModelCallable {
            model: payload.model.clone(),
        };
        let params = resolve_model_params(
            &payload.model,
            ParamSet::new(vec![(var.clone(), point.clone())]),
        )
        .map_err(sim_kernel::Error::from)?;
        let answer = gradient_answer(
            cx,
            &callable,
            payload.query.clone(),
            params,
            std::slice::from_ref(var),
        )
        .map_err(sim_kernel::Error::from)?;
        cx.push_info(format!(
            "femm-adjoint trust={}",
            gradient_trust_label(&answer.trust)
        ));
        let derivative = answer
            .values
            .first()
            .map(|(_, value)| *value)
            .ok_or_else(|| sim_kernel::Error::Eval("missing FEMM derivative".to_owned()))?;
        cx.factory()
            .number_literal(Symbol::qualified("numbers", "f64"), derivative.to_string())
    }
}

/// Registers FEMM's adjoint plugin as a runtime differentiator.
///
/// After registration the runtime can differentiate a model-derived [`Func`]
/// (one carrying a [`FemmFuncPayload`]) through the `femm-adjoint` path, routing
/// `grad`/`diff` requests into [`adjoint_gradient`].
pub fn register_femm_adjoint() -> KernelResult<()> {
    static REGISTERED: OnceLock<std::result::Result<(), String>> = OnceLock::new();

    REGISTERED
        .get_or_init(|| {
            register_differentiator(Arc::new(FemmAdjointPlugin)).map_err(|err| err.to_string())
        })
        .clone()
        .map_err(Error::Eval)
}

/// Whether the excitation a derived quantity is measured against depends on
/// `symbol`.
///
/// Inductance and flux linkage are referenced to a coil current and capacitance
/// to an applied potential. The exact analytic derivative assumes that drive is
/// independent of the design parameter (so `dW/dp` alone fixes the sensitivity);
/// when the drive expression itself uses `symbol`, that assumption fails and the
/// caller must fall back to finite differences instead of silently dropping the
/// `dI/dp` (or `dV/dp`) term.
pub(crate) fn excitation_uses_symbol(
    model: &FemmModel,
    spec: &QuantitySpec,
    symbol: &Symbol,
) -> bool {
    match spec {
        QuantitySpec::Inductance { circuit } | QuantitySpec::FluxLinkage { circuit } => {
            model.sources.iter().any(|source| {
                matches!(
                    source,
                    Source::CircuitCoil { name, current, .. }
                        if name == circuit && expr_uses_symbol(current, symbol)
                )
            })
        }
        QuantitySpec::Capacitance { conductor } => model.boundaries.iter().any(|boundary| {
            &boundary.name == conductor
                && matches!(boundary.kind, BoundaryKind::Dirichlet)
                && expr_uses_symbol(&boundary.value, symbol)
        }),
        _ => false,
    }
}

fn model_uses_symbol(model: &FemmModel, symbol: &Symbol) -> bool {
    geometry_uses_symbol(&model.geometry, symbol)
        || materials_use_symbol(&model.materials, symbol)
        || boundaries_use_symbol(&model.boundaries, symbol)
        || sources_use_symbol(&model.sources, symbol)
        || model
            .frequency_hz
            .as_ref()
            .is_some_and(|expr| expr_uses_symbol(expr, symbol))
        || model
            .depth
            .as_ref()
            .is_some_and(|expr| expr_uses_symbol(expr, symbol))
        || model
            .solve_policy
            .as_ref()
            .is_some_and(|expr| expr_uses_symbol(expr, symbol))
        || mesh_policy_uses_symbol(&model.mesh_policy, symbol)
        || model
            .outputs
            .iter()
            .any(|output| expr_uses_symbol(&output.query, symbol))
}

fn geometry_uses_symbol(geometry: &sim_lib_femm_geometry::Geometry2, symbol: &Symbol) -> bool {
    geometry
        .nodes
        .iter()
        .any(|node| node.xy.iter().any(|expr| expr_uses_symbol(expr, symbol)))
        || geometry
            .arcs
            .iter()
            .any(|arc| expr_uses_symbol(&arc.angle_deg, symbol))
        || geometry
            .labels
            .iter()
            .any(|label| label.at.iter().any(|expr| expr_uses_symbol(expr, symbol)))
        || geometry.analytic.iter().any(|region| match region {
            sim_lib_femm_geometry::AnalyticRegion2::Rect { xy, wh, .. } => xy
                .iter()
                .chain(wh.iter())
                .any(|expr| expr_uses_symbol(expr, symbol)),
            sim_lib_femm_geometry::AnalyticRegion2::Circle { center, radius, .. } => {
                center.iter().any(|expr| expr_uses_symbol(expr, symbol))
                    || expr_uses_symbol(radius, symbol)
            }
            sim_lib_femm_geometry::AnalyticRegion2::Polygon { points, .. } => points
                .iter()
                .flat_map(|point| point.iter())
                .any(|expr| expr_uses_symbol(expr, symbol)),
            sim_lib_femm_geometry::AnalyticRegion2::OuterBox { margin, .. } => {
                expr_uses_symbol(margin, symbol)
            }
        })
}

fn materials_use_symbol(materials: &[Material], symbol: &Symbol) -> bool {
    materials.iter().any(|material| {
        [
            material.mu_r.as_ref(),
            material.nu_of_b2.as_ref(),
            material.epsilon_r.as_ref(),
            material.sigma.as_ref(),
            material.thermal_k.as_ref(),
            material.heat_source.as_ref(),
        ]
        .into_iter()
        .flatten()
        .any(|expr| expr_uses_symbol(expr, symbol))
            || material.remanence.as_ref().is_some_and(|remanence| {
                remanence.iter().any(|expr| expr_uses_symbol(expr, symbol))
            })
    })
}

fn boundaries_use_symbol(boundaries: &[Boundary], symbol: &Symbol) -> bool {
    boundaries
        .iter()
        .any(|boundary| expr_uses_symbol(&boundary.value, symbol))
}

fn sources_use_symbol(sources: &[Source], symbol: &Symbol) -> bool {
    sources.iter().any(|source| match source {
        Source::CurrentDensity { value, .. }
        | Source::ChargeDensity { value, .. }
        | Source::HeatSource { value, .. } => expr_uses_symbol(value, symbol),
        Source::CircuitCoil { turns, current, .. } => {
            expr_uses_symbol(turns, symbol) || expr_uses_symbol(current, symbol)
        }
    })
}

fn mesh_policy_uses_symbol(policy: &MeshPolicy, symbol: &Symbol) -> bool {
    policy
        .max_area
        .as_ref()
        .is_some_and(|expr| expr_uses_symbol(expr, symbol))
        || policy
            .min_angle_deg
            .as_ref()
            .is_some_and(|expr| expr_uses_symbol(expr, symbol))
}

fn expr_uses_symbol(expr: &Expr, symbol: &Symbol) -> bool {
    let expr = normalize_femm_expr(expr).unwrap_or_else(|_| expr.clone());
    expr_uses_symbol_canonical(&expr, symbol)
}

fn expr_uses_symbol_canonical(expr: &Expr, symbol: &Symbol) -> bool {
    match expr {
        Expr::Symbol(candidate) | Expr::Local(candidate) => candidate == symbol,
        Expr::List(items) | Expr::Vector(items) | Expr::Set(items) | Expr::Block(items) => {
            items.iter().any(|expr| expr_uses_symbol(expr, symbol))
        }
        Expr::Map(entries) => entries
            .iter()
            .any(|(key, value)| expr_uses_symbol(key, symbol) || expr_uses_symbol(value, symbol)),
        Expr::Call { args, .. } => args
            .iter()
            .any(|arg| expr_uses_symbol_canonical(arg, symbol)),
        Expr::Infix { left, right, .. } => {
            expr_uses_symbol(left, symbol) || expr_uses_symbol(right, symbol)
        }
        Expr::Prefix { arg, .. }
        | Expr::Postfix { arg, .. }
        | Expr::Extension { payload: arg, .. } => expr_uses_symbol(arg, symbol),
        Expr::Annotated { expr, annotations } => {
            expr_uses_symbol(expr, symbol)
                || annotations
                    .iter()
                    .any(|(_, annotation)| expr_uses_symbol(annotation, symbol))
        }
        Expr::Quote { .. }
        | Expr::Nil
        | Expr::Bool(_)
        | Expr::Number(_)
        | Expr::String(_)
        | Expr::Bytes(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use sim_kernel::{Expr, NumberLiteral, Symbol};
    use sim_value::build::sym;

    use super::expr_uses_symbol;

    fn num(text: &str) -> Expr {
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "f64"),
            canonical: text.to_owned(),
        })
    }

    #[test]
    fn symbol_scanning_normalizes_femm_operator_forms() {
        let gap = Symbol::new("gap");
        let expressions = [
            Expr::Infix {
                operator: Symbol::new("+"),
                left: Box::new(num("1.0")),
                right: Box::new(sym("gap")),
            },
            Expr::Prefix {
                operator: Symbol::new("-"),
                arg: Box::new(sym("gap")),
            },
            Expr::Postfix {
                operator: Symbol::new("sqrt"),
                arg: Box::new(sym("gap")),
            },
            Expr::Call {
                operator: Box::new(Expr::Symbol(Symbol::new("*"))),
                args: vec![
                    Expr::Infix {
                        operator: Symbol::new("+"),
                        left: Box::new(sym("gap")),
                        right: Box::new(num("1.0")),
                    },
                    num("2.0"),
                ],
            },
        ];
        for expr in expressions {
            assert!(expr_uses_symbol(&expr, &gap), "{expr:?}");
        }
    }
}
