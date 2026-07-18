#![forbid(unsafe_code)]
//! Model query callables shared by FEMM function and sensitivity crates.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    ClassId, ClassRef, Cx, DefaultFactory, Expr, Factory, Object, ObjectEncode, ObjectEncoding,
    Result as KernelResult, Symbol, Value,
};
use sim_lib_femm_core::{FemmError, FemmLimits, FemmResult, ParamSet};
use sim_lib_femm_field::{Field, Projection};
use sim_lib_femm_material::{BoundaryKind, Source};
use sim_lib_femm_mesh::FemmModel;
use sim_lib_femm_post::{Excitation, FemmSolution, QuantitySpec, quantity};
use sim_lib_femm_solve::solve_steady;
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
    fn solve_solution(
        &self,
        cx: &mut Cx,
        params: &ParamSet,
        limits: &FemmLimits,
    ) -> FemmResult<Arc<FemmSolution>> {
        let resolved = resolve_model_params(&self.model, params.clone())?;
        solve_steady(cx, &self.model, &resolved, limits, None).map(|out| out.solution)
    }
}

impl FemmCallable for ModelCallable {
    fn eval(&self, cx: &mut Cx, call: FemmCall) -> FemmResult<FemmEval> {
        let resolved = resolve_model_params(&self.model, call.params)?;
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

/// Resolves a model parameter set by inserting defaults for missing model inputs.
///
/// A missing input without a default is an error. This is the single defaulting
/// rule used by model calls and sensitivity plugin evaluation.
pub fn resolve_model_params(model: &FemmModel, params: ParamSet) -> FemmResult<ParamSet> {
    let mut entries = params.entries;
    for input in &model.inputs {
        if entries.iter().all(|(name, _)| name != &input.name) {
            let Some(default) = &input.default else {
                return Err(FemmError::UnknownFemmParameter(input.name.to_string()));
            };
            entries.push((input.name.clone(), default.clone()));
        }
    }
    Ok(ParamSet::new(entries))
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
/// use sim_lib_femm_post::QuantitySpec;
/// use sim_lib_femm_query::{OutputQuery, femm_as_func};
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
/// The returned function solves `model` with its default parameters and samples
/// the solved potential field at `(x, y)`. Mesh or solve failures propagate
/// through the callable boundary; this path never fabricates a replacement
/// solution.
pub fn femm_field_func(model: FemmModel) -> Func {
    Func::native(
        vec![Symbol::new("x"), Symbol::new("y")],
        Arc::new(move |cx, args| {
            let x =
                sim_lib_femm_core::value_as_f64(cx, &args[0]).map_err(sim_kernel::Error::from)?;
            let y =
                sim_lib_femm_core::value_as_f64(cx, &args[1]).map_err(sim_kernel::Error::from)?;
            let solution = solve_steady(
                cx,
                &model,
                &ParamSet::default(),
                &FemmLimits::default(),
                None,
            )
            .map_err(sim_kernel::Error::from)?
            .solution;
            let field = Field::new(solution, Projection::Potential);
            cx.factory().number_literal(
                Symbol::qualified("numbers", "f64"),
                field.at(x, y).map_err(sim_kernel::Error::from)?.to_string(),
            )
        }),
    )
}

/// Human-readable label for an output query.
pub fn describe_query(query: &OutputQuery) -> String {
    match query {
        OutputQuery::Quantity(QuantitySpec::Custom { name, .. }) => format!("quantity:{name}"),
        OutputQuery::Quantity(_) => "quantity".to_owned(),
        OutputQuery::Field(projection) => format!("field:{projection:?}"),
        OutputQuery::Solution => "solution".to_owned(),
    }
}
