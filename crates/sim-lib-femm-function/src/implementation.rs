#![forbid(unsafe_code)]
//! Callable wrapper that evaluates a model to a quantity, field, or solution.
//!
//! Defines the call request, output query, evaluation result, and the callable
//! payload that turns a FEMM model into a runtime function of its parameters.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    ClassId, ClassRef, Cx, DefaultFactory, Expr, Factory, Object, ObjectEncode, ObjectEncoding,
    Result as KernelResult, Symbol, Value,
};
use sim_lib_femm_core::{FemmError, FemmLimits, FemmResult, ParamSet, StableId, value_as_f64};
use sim_lib_femm_field::{Field, Projection, field_as_func};
use sim_lib_femm_material::{BoundaryKind, Source};
use sim_lib_femm_mesh::FemmModel;
use sim_lib_femm_post::{Excitation, FemmSolution, QuantitySpec, quantity};
use sim_lib_femm_solve::{GradientTrust, SolveCertificate, SteadySolve, solve_steady};
use sim_lib_numbers_func::{Func, FuncMetadata};

/// One evaluation request against a model: which parameters, output, and limits.
///
/// Couples a parameter binding with the [`OutputQuery`] to compute and the
/// solver [`FemmLimits`]; `want_grad` names parameters whose sensitivities the
/// caller also wants. See the [crate README](index.html).
#[derive(Clone, Debug)]
pub struct FemmCall {
    /// The parameter binding the model is evaluated at.
    pub params: ParamSet,
    /// The output to compute from the solved model.
    pub query: OutputQuery,
    /// Parameters to also report sensitivities for, if any.
    pub want_grad: Option<Vec<Symbol>>,
    /// Solver budget and tolerances for this evaluation.
    pub limits: FemmLimits,
}

/// The kind of output an evaluation produces from a solved model.
#[derive(Clone, Debug)]
pub enum OutputQuery {
    /// A scalar quantity reduced from the solution (energy, flux, capacitance).
    Quantity(QuantitySpec),
    /// A projected field (potential or a derived component) over the mesh.
    Field(Projection),
    /// The full solved model solution as an opaque value.
    Solution,
}

/// The result of evaluating a model: the output value plus optional gradient.
#[derive(Clone, Debug)]
pub struct FemmEval {
    /// The computed output value (scalar, field, or solution).
    pub value: Value,
    /// Per-parameter sensitivities, when a gradient was requested.
    pub gradient: Option<Vec<(Symbol, f64)>>,
    /// Diagnostics emitted while solving and reducing the output.
    pub diagnostics: Vec<sim_kernel::Diagnostic>,
}

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

/// The opaque payload carried by a model-derived runtime function.
///
/// Recorded in a [`Func`]'s metadata so a differentiator can recover the model,
/// its free variables, and the queried output to build an adjoint pass.
#[derive(Clone)]
pub struct FemmFuncPayload {
    /// The model the function evaluates.
    pub model: FemmModel,
    /// The model inputs treated as the function's free variables.
    pub vars: Vec<Symbol>,
    /// The output the function returns.
    pub query: OutputQuery,
}

impl Object for FemmFuncPayload {
    fn display(&self, _cx: &mut Cx) -> KernelResult<String> {
        Ok(format!(
            "#<femm-payload model={} query={}>",
            self.model.id.0,
            describe_query(&self.query)
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for FemmFuncPayload {
    fn class(&self, cx: &mut Cx) -> KernelResult<ClassRef> {
        if let Some(class) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("femm", "FuncPayload"))
        {
            return Ok(class.clone());
        }
        DefaultFactory.class_stub(ClassId(33), Symbol::qualified("femm", "FuncPayload"))
    }
    fn as_expr(&self, cx: &mut Cx) -> KernelResult<Expr> {
        sim_citizen::constructor_expr(cx, self)
    }
    fn as_object_encoder(&self) -> Option<&dyn ObjectEncode> {
        Some(self)
    }
}

impl ObjectEncode for FemmFuncPayload {
    fn object_encoding(&self, _cx: &mut Cx) -> KernelResult<ObjectEncoding> {
        Ok(ObjectEncoding::Constructor {
            class: func_payload_class_symbol(),
            args: payload_constructor_args(self),
        })
    }
}

impl sim_citizen::Citizen for FemmFuncPayload {
    fn citizen_symbol() -> Symbol {
        func_payload_class_symbol()
    }

    fn citizen_version() -> u32 {
        1
    }

    fn citizen_arity() -> usize {
        3
    }

    fn citizen_fields() -> &'static [&'static str] {
        &["model_id", "query", "vars"]
    }
}

fn func_payload_class_symbol() -> Symbol {
    Symbol::qualified("femm", "FuncPayload")
}

fn payload_constructor_args(payload: &FemmFuncPayload) -> Vec<Expr> {
    vec![
        Expr::Symbol(Symbol::new("v1")),
        int_expr(payload.model.id.0),
        Expr::String(describe_query(&payload.query)),
        Expr::List(
            payload
                .vars
                .iter()
                .map(|name| Expr::String(name.to_string()))
                .collect(),
        ),
    ]
}

fn int_expr(value: impl ToString) -> Expr {
    Expr::Number(sim_kernel::NumberLiteral {
        domain: Symbol::qualified("citizen", "int"),
        canonical: value.to_string(),
    })
}

/// Something that can be evaluated as a FEMM function of its parameters.
pub trait FemmCallable {
    /// Evaluates the callable for one [`FemmCall`], returning its output.
    fn eval(&self, cx: &mut Cx, call: FemmCall) -> FemmResult<FemmEval>;
}

/// A [`FemmCallable`] that solves a concrete model on each evaluation.
///
/// Resolves defaults for any unbound inputs, runs the steady solve, and reduces
/// the solution to the requested [`OutputQuery`].
#[derive(Clone)]
pub struct ModelCallable {
    /// The model solved on each call.
    pub model: FemmModel,
}

impl ModelCallable {
    fn resolve_params(&self, params: &ParamSet) -> FemmResult<ParamSet> {
        let mut entries = params.entries.clone();
        for input in &self.model.inputs {
            if entries.iter().all(|(name, _)| name != &input.name) {
                if let Some(default) = &input.default {
                    entries.push((input.name.clone(), default.clone()));
                } else {
                    return Err(FemmError::UnknownFemmParameter(input.name.to_string()));
                }
            }
        }
        Ok(ParamSet::new(entries))
    }

    fn solve_solution(
        &self,
        cx: &mut Cx,
        params: &ParamSet,
        limits: &FemmLimits,
    ) -> FemmResult<Arc<FemmSolution>> {
        let resolved = self.resolve_params(params)?;
        solve_steady(cx, &self.model, &resolved, limits, None).map(|out| out.solution)
    }
}

impl FemmCallable for ModelCallable {
    fn eval(&self, cx: &mut Cx, call: FemmCall) -> FemmResult<FemmEval> {
        let resolved = self.resolve_params(&call.params)?;
        match call.query {
            OutputQuery::Quantity(QuantitySpec::Custom { expr, .. }) => {
                let value = sim_lib_femm_geometry::eval_expr_f64(cx, &expr, &resolved, &[])?;
                Ok(FemmEval {
                    value: cx
                        .factory()
                        .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
                        .map_err(|err| FemmError::SensitivityUnavailable(err.to_string()))?,
                    gradient: None,
                    diagnostics: Vec::new(),
                })
            }
            OutputQuery::Quantity(spec) => {
                let solution = self.solve_solution(cx, &resolved, &call.limits)?;
                let excitation = resolve_excitation(cx, &self.model, &resolved, &spec)?;
                let scalar = quantity(&solution, &spec, &excitation)?;
                Ok(FemmEval {
                    value: cx
                        .factory()
                        .number_literal(Symbol::qualified("numbers", "f64"), scalar.to_string())
                        .map_err(|err| FemmError::SensitivityUnavailable(err.to_string()))?,
                    gradient: None,
                    diagnostics: Vec::new(),
                })
            }
            OutputQuery::Field(projection) => {
                let solution = self.solve_solution(cx, &resolved, &call.limits)?;
                let field = Field::new(solution, projection);
                Ok(FemmEval {
                    value: cx
                        .factory()
                        .opaque(Arc::new(field))
                        .map_err(|err| FemmError::SensitivityUnavailable(err.to_string()))?,
                    gradient: None,
                    diagnostics: Vec::new(),
                })
            }
            OutputQuery::Solution => {
                let solution = self.solve_solution(cx, &resolved, &call.limits)?;
                Ok(FemmEval {
                    value: cx
                        .factory()
                        .opaque(solution)
                        .map_err(|err| FemmError::SensitivityUnavailable(err.to_string()))?,
                    gradient: None,
                    diagnostics: Vec::new(),
                })
            }
        }
    }
}

/// Resolves the [`Excitation`] a derived quantity is evaluated against.
///
/// Inductance and flux linkage read the driving current of the named circuit
/// coil source; capacitance reads the applied potential of the named conductor
/// (its Dirichlet boundary). Quantities that do not depend on an excitation
/// resolve to [`Excitation::none`]. A source the model does not define leaves
/// the excitation unset, so [`quantity`] reports the precise missing-drive
/// error rather than a silent wrong value.
pub fn resolve_excitation(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    spec: &QuantitySpec,
) -> FemmResult<Excitation> {
    match spec {
        QuantitySpec::Inductance { circuit } | QuantitySpec::FluxLinkage { circuit } => {
            Ok(coil_current(cx, model, params, circuit)?
                .map(Excitation::with_current)
                .unwrap_or_else(Excitation::none))
        }
        QuantitySpec::Capacitance { conductor } => {
            Ok(conductor_potential(cx, model, params, conductor)?
                .map(Excitation::with_potential)
                .unwrap_or_else(Excitation::none))
        }
        _ => Ok(Excitation::none()),
    }
}

fn coil_current(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    circuit: &Symbol,
) -> FemmResult<Option<f64>> {
    for source in &model.sources {
        if let Source::CircuitCoil { name, current, .. } = source
            && name == circuit
        {
            return sim_lib_femm_geometry::eval_expr_f64(cx, current, params, &[]).map(Some);
        }
    }
    Ok(None)
}

fn conductor_potential(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    conductor: &Symbol,
) -> FemmResult<Option<f64>> {
    for boundary in &model.boundaries {
        if &boundary.name == conductor && matches!(boundary.kind, BoundaryKind::Dirichlet) {
            return sim_lib_femm_geometry::eval_expr_f64(cx, &boundary.value, params, &[])
                .map(Some);
        }
    }
    Ok(None)
}

/// Returns the requested quantity and the certificate for a completed solve.
///
/// Passing `Some(params)` for `wrt` also computes a total finite-difference
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
            let (values, trust) =
                finite_difference_quality_gradient(cx, solve, quantity_spec, params)?;
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

fn finite_difference_quality_gradient(
    cx: &mut Cx,
    solve: &SteadySolve,
    quantity_spec: &QuantitySpec,
    wrt: &[Symbol],
) -> FemmResult<(Vec<f64>, GradientTrust)> {
    let callable = ModelCallable {
        model: solve.model.clone(),
    };
    let base_params = callable.resolve_params(&solve.solution.params)?;
    let mut out = Vec::with_capacity(wrt.len());
    for symbol in wrt {
        let base_value = base_params
            .get(symbol)
            .ok_or_else(|| FemmError::UnknownFemmParameter(symbol.to_string()))?;
        let x = value_as_f64(cx, base_value)?;
        if !x.is_finite() {
            return Err(FemmError::SensitivityUnavailable(format!(
                "non-finite FEMM parameter {symbol}"
            )));
        }
        let h = fd_step(x);
        let plus = replace_param_value(cx, &base_params, symbol, x + h)?;
        let minus = replace_param_value(cx, &base_params, symbol, x - h)?;
        let plus_value = quality_at_params(cx, &solve.model, plus, quantity_spec)?;
        let minus_value = quality_at_params(cx, &solve.model, minus, quantity_spec)?;
        out.push((plus_value - minus_value) / (2.0 * h));
    }
    Ok((out, GradientTrust::FiniteDifferenceOnly))
}

fn quality_at_params(
    cx: &mut Cx,
    model: &FemmModel,
    params: ParamSet,
    quantity_spec: &QuantitySpec,
) -> FemmResult<f64> {
    let solved = solve_steady(cx, model, &params, &FemmLimits::default(), None)?;
    let excitation = resolve_excitation(cx, model, &params, quantity_spec)?;
    quantity(&solved.solution, quantity_spec, &excitation)
}

fn replace_param_value(
    cx: &mut Cx,
    params: &ParamSet,
    name: &Symbol,
    value: f64,
) -> FemmResult<ParamSet> {
    let mut found = false;
    let mut entries = params.entries.clone();
    for (symbol, slot) in &mut entries {
        if symbol == name {
            *slot = cx
                .factory()
                .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
                .map_err(|err| FemmError::SensitivityUnavailable(err.to_string()))?;
            found = true;
        }
    }
    if found {
        Ok(ParamSet::new(entries))
    } else {
        Err(FemmError::UnknownFemmParameter(name.to_string()))
    }
}

fn fd_step(value: f64) -> f64 {
    1.0e-6 * value.abs().max(1.0)
}

/// Wraps a model as a sim-numbers [`Func`] of the named variables.
///
/// The returned function solves the model on call and reduces it to `query`;
/// its metadata carries a [`FemmFuncPayload`] and an adjoint differentiator
/// hint so sensitivity analysis can recover the model.
///
/// # Examples
///
/// ```
/// use sim_kernel::Symbol;
/// use sim_lib_femm_fixtures::parallel_plate_capacitor;
/// use sim_lib_femm_function::{femm_as_func, OutputQuery};
/// use sim_lib_femm_post::QuantitySpec;
///
/// let vars = vec![Symbol::new("gap-mm")];
/// let func = femm_as_func(
///     parallel_plate_capacitor(),
///     vars.clone(),
///     OutputQuery::Quantity(QuantitySpec::Energy { region: None }),
/// );
/// assert_eq!(func.vars, vars);
/// ```
pub fn femm_as_func(model: FemmModel, vars: Vec<Symbol>, query: OutputQuery) -> Func {
    let callable = ModelCallable {
        model: model.clone(),
    };
    let closure_vars = vars.clone();
    let payload_vars = closure_vars.clone();
    let closure_query = query.clone();
    let mut func = Func::native(
        vars,
        Arc::new(move |cx, args| {
            let params = ParamSet::new(
                closure_vars
                    .iter()
                    .cloned()
                    .zip(args.iter().cloned())
                    .collect::<Vec<_>>(),
            );
            callable
                .eval(
                    cx,
                    FemmCall {
                        params,
                        query: closure_query.clone(),
                        want_grad: None,
                        limits: FemmLimits::default(),
                    },
                )
                .map(|out| out.value)
                .map_err(sim_kernel::Error::from)
        }),
    );
    func.metadata = FuncMetadata {
        source: Some(Symbol::qualified("femm", "model")),
        differentiator_hint: Some(Symbol::new("femm-adjoint")),
        payload: DefaultFactory
            .opaque(Arc::new(FemmFuncPayload {
                model: model.clone(),
                vars: payload_vars,
                query: query.clone(),
            }))
            .ok(),
    };
    func
}

/// Wraps a model's potential field as a sim-numbers [`Func`] over position.
///
/// Builds a trivial single-element solution for `model` and exposes its
/// potential projection as a spatial function, used where a model is consumed
/// as a field-valued function rather than a parameter-to-scalar map.
pub fn femm_field_func(model: FemmModel) -> Func {
    let field = Arc::new(FemmSolution {
        id: StableId(model.id.0 + 1),
        model_id: model.id,
        physics: model.physics.clone(),
        formulation: model.formulation.clone(),
        params: ParamSet::default(),
        mesh: sim_lib_femm_mesh::FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        },
        u: vec![0.0, 1.0, 1.0],
        diagnostics: sim_lib_femm_flow::SolveDiagnostics {
            method: Symbol::new("femm-ptc"),
            converged: true,
            iterations: 1,
            final_residual: 0.0,
            events: Vec::new(),
            diagnostics: Vec::new(),
        },
    });
    field_as_func(Field::new(field, Projection::Potential))
}

pub(crate) fn describe_query(query: &OutputQuery) -> String {
    match query {
        OutputQuery::Quantity(QuantitySpec::Custom { name, .. }) => format!("quantity:{name}"),
        OutputQuery::Quantity(_) => "quantity".to_owned(),
        OutputQuery::Field(projection) => format!("field:{projection:?}"),
        OutputQuery::Solution => "solution".to_owned(),
    }
}
