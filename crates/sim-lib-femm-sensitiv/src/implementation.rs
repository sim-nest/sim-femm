#![forbid(unsafe_code)]
//! Gradient computation and differentiator registration for FEMM quantities.
//!
//! Defines the sensitivity-path selection and the entry point that computes
//! parameter gradients of a model quantity, registering FEMM as a runtime
//! differentiator.

use std::sync::Arc;

use sim_kernel::{Cx, Expr, Result as KernelResult, Symbol, Value};
use sim_lib_femm_core::{FemmError, FemmResult, ParamSet};
use sim_lib_femm_function::{FemmFuncPayload, ModelCallable, OutputQuery};
use sim_lib_femm_material::{Boundary, Material, MeshPolicy, Source};
use sim_lib_femm_mesh::FemmModel;
use sim_lib_numbers_ad::{Dual, Scalarish, Tape, Var};
use sim_lib_numbers_func::Func;
use sim_lib_numbers_numeric::{
    DiffOpts, Differentiator, NumericKind, NumericPlugin, register_differentiator,
};

use crate::sensitivity_solve::built_in_quantity_gradient;

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

fn direct_expr_derivative(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    wrt: &Symbol,
) -> FemmResult<f64> {
    Ok(eval_expr_dual(cx, expr, params, Some(wrt), &[])?.d[0])
}

pub(crate) fn eval_expr_dual(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    wrt: Option<&Symbol>,
    coords: &[(&str, f64)],
) -> FemmResult<Dual<1>> {
    match expr {
        Expr::Number(number) => parse_number(&number.canonical).map(Dual::<1>::cst),
        Expr::Symbol(symbol) | Expr::Local(symbol) => {
            if let Some((_, value)) = coords
                .iter()
                .find(|(name, _)| symbol.name.as_ref() == *name)
            {
                return Ok(Dual::<1>::cst(*value));
            }
            let value = params
                .get(symbol)
                .ok_or_else(|| FemmError::UnknownFemmParameter(symbol.to_string()))?;
            let scalar = sim_lib_femm_core::value_as_f64(cx, value)?;
            if wrt == Some(symbol) {
                Ok(Dual::<1>::var(scalar, 0))
            } else {
                Ok(Dual::<1>::cst(scalar))
            }
        }
        Expr::Call { operator, args } => {
            let Expr::Symbol(symbol) = operator.as_ref() else {
                return Err(FemmError::SensitivityUnavailable(
                    "unsupported non-symbol operator".to_owned(),
                ));
            };
            let values = args
                .iter()
                .map(|arg| eval_expr_dual(cx, arg, params, wrt, coords))
                .collect::<FemmResult<Vec<_>>>()?;
            apply_dual_op(symbol, &values)
        }
        _ => Err(FemmError::SensitivityUnavailable(
            "unsupported expression in exact direct gradient".to_owned(),
        )),
    }
}

fn apply_dual_op(symbol: &Symbol, values: &[Dual<1>]) -> FemmResult<Dual<1>> {
    match (symbol.name.as_ref(), values) {
        ("+", []) => Ok(Dual::cst(0.0)),
        ("+", values) => Ok(values
            .iter()
            .copied()
            .fold(Dual::cst(0.0), |acc, value| acc + value)),
        ("*", []) => Ok(Dual::cst(1.0)),
        ("*", values) => Ok(values
            .iter()
            .copied()
            .fold(Dual::cst(1.0), |acc, value| acc * value)),
        ("-", [value]) => Ok(-*value),
        ("-", [left, right]) => Ok(*left - *right),
        ("/", [left, right]) => Ok(*left / *right),
        ("pow", [base, exp]) => Ok((base.ln() * *exp).exp()),
        ("sin", [arg]) => Ok(arg.sin()),
        ("cos", [arg]) => Ok(arg.cos()),
        ("exp", [arg]) => Ok(arg.exp()),
        ("ln", [arg]) => Ok(arg.ln()),
        ("sqrt", [arg]) => Ok(arg.sqrt()),
        _ => Err(FemmError::SensitivityUnavailable(format!(
            "unsupported operator {symbol} in exact direct gradient"
        ))),
    }
}

fn reverse_expr_gradient(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    wrt: &[Symbol],
) -> FemmResult<Vec<(Symbol, f64)>> {
    let mut tape = Tape::new();
    let inputs = wrt
        .iter()
        .enumerate()
        .map(|(slot, symbol)| {
            let value = params
                .get(symbol)
                .ok_or_else(|| FemmError::UnknownFemmParameter(symbol.to_string()))
                .and_then(|value| sim_lib_femm_core::value_as_f64(cx, value))?;
            Ok((symbol.clone(), slot, tape.input(slot, value)))
        })
        .collect::<FemmResult<Vec<_>>>()?;
    let output = eval_expr_tape(cx, expr, params, &inputs, &mut tape, &[])?;
    let gradient = tape.grad(output, wrt.len());
    Ok(inputs
        .into_iter()
        .map(|(symbol, slot, _)| (symbol, gradient[slot]))
        .collect())
}

fn eval_expr_tape(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    wrt: &[(Symbol, usize, Var)],
    tape: &mut Tape,
    coords: &[(&str, f64)],
) -> FemmResult<Var> {
    match expr {
        Expr::Number(number) => parse_number(&number.canonical).map(|value| tape.constant(value)),
        Expr::Symbol(symbol) | Expr::Local(symbol) => {
            if let Some((_, value)) = coords
                .iter()
                .find(|(name, _)| symbol.name.as_ref() == *name)
            {
                return Ok(tape.constant(*value));
            }
            if let Some((_, _, var)) = wrt.iter().find(|(name, _, _)| name == symbol) {
                return Ok(*var);
            }
            let value = params
                .get(symbol)
                .ok_or_else(|| FemmError::UnknownFemmParameter(symbol.to_string()))?;
            sim_lib_femm_core::value_as_f64(cx, value).map(|value| tape.constant(value))
        }
        Expr::Call { operator, args } => {
            let Expr::Symbol(symbol) = operator.as_ref() else {
                return Err(FemmError::SensitivityUnavailable(
                    "unsupported non-symbol operator".to_owned(),
                ));
            };
            let values = args
                .iter()
                .map(|arg| eval_expr_tape(cx, arg, params, wrt, tape, coords))
                .collect::<FemmResult<Vec<_>>>()?;
            apply_tape_op(symbol, &values, tape)
        }
        _ => Err(FemmError::SensitivityUnavailable(
            "unsupported expression in exact adjoint gradient".to_owned(),
        )),
    }
}

fn apply_tape_op(symbol: &Symbol, values: &[Var], tape: &mut Tape) -> FemmResult<Var> {
    match (symbol.name.as_ref(), values) {
        ("+", []) => Ok(tape.constant(0.0)),
        ("+", values) => {
            let mut acc = tape.constant(0.0);
            for value in values {
                acc = tape.add(acc, *value);
            }
            Ok(acc)
        }
        ("*", []) => Ok(tape.constant(1.0)),
        ("*", values) => {
            let mut acc = tape.constant(1.0);
            for value in values {
                acc = tape.mul(acc, *value);
            }
            Ok(acc)
        }
        ("-", [value]) => {
            let zero = tape.constant(0.0);
            Ok(tape.sub(zero, *value))
        }
        ("-", [left, right]) => Ok(tape.sub(*left, *right)),
        ("/", [left, right]) => Ok(tape.div(*left, *right)),
        ("pow", [base, exp]) => {
            let ln_base = tape.ln(*base);
            let scaled = tape.mul(ln_base, *exp);
            Ok(tape.exp(scaled))
        }
        ("sin", [arg]) => Ok(tape.sin(*arg)),
        ("cos", [arg]) => Ok(tape.cos(*arg)),
        ("exp", [arg]) => Ok(tape.exp(*arg)),
        ("ln", [arg]) => Ok(tape.ln(*arg)),
        ("sqrt", [arg]) => Ok(tape.sqrt(*arg)),
        _ => Err(FemmError::SensitivityUnavailable(format!(
            "unsupported operator {symbol} in exact adjoint gradient"
        ))),
    }
}

fn parse_number(text: &str) -> FemmResult<f64> {
    sim_lib_femm_core::parse_displayed_number(text)
        .ok_or_else(|| FemmError::SensitivityUnavailable(format!("bad number literal {text}")))
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
        let params = ParamSet::new(vec![(var.clone(), point.clone())]);
        let (gradient, path) = adjoint_gradient(
            cx,
            &callable,
            payload.query.clone(),
            params,
            std::slice::from_ref(var),
        )
        .map_err(sim_kernel::Error::from)?;
        if path != SensitivityPath::AdjointExact {
            return Err(sim_kernel::Error::Eval(format!(
                "femm-adjoint could not produce an exact adjoint path: {path:?}"
            )));
        }
        let derivative = gradient
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
    register_differentiator(Arc::new(FemmAdjointPlugin))
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
    match expr {
        Expr::Symbol(candidate) | Expr::Local(candidate) => candidate == symbol,
        Expr::List(items) | Expr::Vector(items) | Expr::Set(items) | Expr::Block(items) => {
            items.iter().any(|expr| expr_uses_symbol(expr, symbol))
        }
        Expr::Map(entries) => entries
            .iter()
            .any(|(key, value)| expr_uses_symbol(key, symbol) || expr_uses_symbol(value, symbol)),
        Expr::Call { args, .. } => args.iter().any(|arg| expr_uses_symbol(arg, symbol)),
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
