#![forbid(unsafe_code)]
//! FEMM models cast as ODE/DAE right-hand sides for time integration.
//!
//! Defines the model-backed right-hand side and the DAE residual interface
//! that let a solved model drive a time-dependent system through the
//! sim-numbers ODE solvers.

use std::sync::{Arc, Mutex};

use std::any::Any;

use sim_kernel::{
    AbiVersion, Args, Callable, ClassRef, Cx, DefaultFactory, Dependency, Error, Export, Expr,
    Factory, Lib, LibManifest, LibTarget, Linker, Object, RawArgs, Result as KernelResult, Symbol,
    Value, Version,
};
use sim_lib_femm_core::{CsrMatrix, FemmError, FemmResult};
use sim_lib_femm_function::ModelValue;
use sim_lib_femm_mesh::FemmModel;
use sim_lib_femm_post::{FemmSolution, QuantitySpec};
use sim_lib_femm_tape::SolveTape;
use sim_lib_numbers_func::{Func, FuncMetadata};

/// A FEMM model cast as the right-hand side of a first-order ODE/DAE system.
///
/// Each evaluation maps the current state vector onto FEMM parameters, solves
/// the model (reusing the [`SolveTape`] cache), reads any required post-processed
/// quantities, and evaluates the state-derivative expressions. See the
/// [crate README](https://github.com/sim/sim-femm).
///
/// # Examples
///
/// ```
/// use std::sync::{Arc, Mutex};
/// use sim_kernel::{Args, Cx, DefaultFactory, EagerPolicy, Expr, NumberLiteral, Symbol};
/// use sim_lib_femm_fixtures::parallel_plate_capacitor;
/// use sim_lib_femm_tape::SolveTape;
/// use sim_lib_femm_ode::FemmOdeRhs;
///
/// let f64_domain = Symbol::qualified("numbers", "f64");
/// let num = |text: &str| Expr::Number(NumberLiteral {
///     domain: f64_domain.clone(),
///     canonical: text.to_owned(),
/// });
/// // dx/dt = v, dv/dt = -4 x: a harmonic state equation, ignoring the model field.
/// let rhs = FemmOdeRhs {
///     model: parallel_plate_capacitor(),
///     state_vars: vec![Symbol::new("x"), Symbol::new("v")],
///     param_map: Vec::new(),
///     need: Vec::new(),
///     rhs: vec![
///         Expr::Symbol(Symbol::new("v")),
///         Expr::Call {
///             operator: Box::new(Expr::Symbol(Symbol::new("*"))),
///             args: vec![num("-4.0"), Expr::Symbol(Symbol::new("x"))],
///         },
///     ],
///     tape: Arc::new(Mutex::new(SolveTape::default())),
/// };
/// let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
/// let func = cx.factory().opaque(Arc::new(rhs.as_func())).unwrap();
/// let out = cx
///     .call_value(
///         func,
///         Args::new(vec![
///             cx.factory().number_literal(f64_domain.clone(), "0.5".to_owned()).unwrap(),
///             cx.factory().number_literal(f64_domain.clone(), "0.25".to_owned()).unwrap(),
///         ]),
///     )
///     .unwrap();
/// let Expr::List(items) = out.object().as_expr(&mut cx).unwrap() else {
///     panic!("expected derivative list");
/// };
/// assert_eq!(items.len(), 2);
/// ```
#[derive(Clone)]
pub struct FemmOdeRhs {
    /// The FEMM model solved at each time step.
    pub model: FemmModel,
    /// State variables, in order, that form the ODE state vector.
    pub state_vars: Vec<Symbol>,
    /// State-to-parameter bindings: each state variable feeds the named model parameter.
    pub param_map: Vec<(Symbol, Symbol)>,
    /// Post-processed quantities to read from the solution and bind before evaluating `rhs`.
    pub need: Vec<QuantitySpec>,
    /// State-derivative expressions, one per state variable, evaluated each step.
    pub rhs: Vec<sim_kernel::Expr>,
    /// Shared solve cache reused across steps and derivative sweeps.
    pub tape: Arc<Mutex<SolveTape>>,
}

/// Differential-algebraic residual interface for implicit FEMM time stepping.
///
/// Implementors return the residual `F(t, z, zdot)` whose root advances the
/// coupled state, with an optional analytic Jacobian for Newton solves.
pub trait DaeResidual {
    /// Returns the DAE residual `F(t, z, zdot)` at the given time and state.
    fn residual(&self, cx: &mut Cx, t: &Value, z: &Value, zdot: &Value) -> KernelResult<Value>;

    /// Returns the residual Jacobian, or `None` when the solver must approximate it.
    fn jacobian(
        &self,
        _cx: &mut Cx,
        _t: &Value,
        _z: &Value,
        _zdot: &Value,
    ) -> KernelResult<Option<CsrMatrix>> {
        Ok(None)
    }
}

impl FemmOdeRhs {
    /// Compiles this model-backed right-hand side into a callable sim-numbers [`Func`].
    ///
    /// The resulting function takes the state vector and returns the list of
    /// state derivatives, solving the model and reading needed quantities per call.
    pub fn as_func(&self) -> Func {
        let model = self.model.clone();
        let state_vars = self.state_vars.clone();
        let body_vars = state_vars.clone();
        let param_map = self.param_map.clone();
        let need = self.need.clone();
        let rhs = self.rhs.clone();
        let tape = self.tape.clone();
        Func {
            vars: state_vars,
            body_cas: None,
            body_native: Some(Arc::new(move |cx, args| {
                let params = sim_lib_femm_core::ParamSet::new(
                    body_vars
                        .iter()
                        .cloned()
                        .zip(args.iter().cloned())
                        .collect(),
                );
                let mut eval_params = params.clone();
                for (param, state) in &param_map {
                    let value = params
                        .get(state)
                        .ok_or_else(|| {
                            sim_kernel::Error::Eval(format!("missing state variable {state}"))
                        })?
                        .clone();
                    eval_params.entries.push((param.clone(), value));
                }
                let solution = cached_solution(cx, &model, &eval_params, &tape)
                    .map_err(sim_kernel::Error::from)?;
                let mut rhs_params = eval_params.clone();
                for quantity in &need {
                    let excitation = sim_lib_femm_function::resolve_excitation(
                        cx,
                        &model,
                        &eval_params,
                        quantity,
                    )
                    .map_err(sim_kernel::Error::from)?;
                    let value = sim_lib_femm_post::quantity(&solution, quantity, &excitation)
                        .map_err(sim_kernel::Error::from)?;
                    rhs_params.entries.push((
                        quantity_name(quantity),
                        cx.factory().number_literal(
                            Symbol::qualified("numbers", "f64"),
                            value.to_string(),
                        )?,
                    ));
                }
                let values =
                    rhs.iter()
                        .map(|expr| {
                            sim_lib_femm_geometry::eval_expr_f64(cx, expr, &rhs_params, &[])
                                .and_then(|value| {
                                    cx.factory()
                                        .number_literal(
                                            Symbol::qualified("numbers", "f64"),
                                            value.to_string(),
                                        )
                                        .map_err(|err| {
                                            FemmError::SensitivityUnavailable(err.to_string())
                                        })
                                })
                        })
                        .collect::<FemmResult<Vec<_>>>()
                        .map_err(sim_kernel::Error::from)?;
                cx.factory().list(values)
            })),
            metadata: FuncMetadata::default(),
        }
    }
}

fn cached_solution(
    cx: &mut Cx,
    model: &FemmModel,
    params: &sim_lib_femm_core::ParamSet,
    tape: &Arc<Mutex<SolveTape>>,
) -> FemmResult<Arc<FemmSolution>> {
    let mut guard = tape.lock().unwrap();
    guard.solve(cx, model, params, &sim_lib_femm_core::FemmLimits::default())
}

fn quantity_name(quantity: &QuantitySpec) -> Symbol {
    match quantity {
        QuantitySpec::Energy { .. } => Symbol::new("energy"),
        QuantitySpec::Coenergy { .. } => Symbol::new("coenergy"),
        QuantitySpec::ForceY { .. } => Symbol::new("force-y"),
        QuantitySpec::Torque { .. } => Symbol::new("torque"),
        QuantitySpec::FluxLinkage { .. } => Symbol::new("flux-linkage"),
        QuantitySpec::Inductance { .. } => Symbol::new("inductance"),
        QuantitySpec::Capacitance { .. } => Symbol::new("capacitance"),
        QuantitySpec::JouleLoss { .. } => Symbol::new("joule-loss"),
        QuantitySpec::FieldAt { field, .. } => field.clone(),
        QuantitySpec::Custom { name, .. } => name.clone(),
    }
}

/// Library that registers the `femm/as-ode-rhs` form for time integration.
///
/// Loading it exposes the form that casts a solved model plus state and
/// derivative expressions into a callable ODE right-hand side.
pub struct FemmOdeLib;

impl FemmOdeLib {
    /// Creates the FEMM ODE library handle.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FemmOdeLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for FemmOdeLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("femm", "ode"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![Dependency {
                id: Symbol::qualified("femm", "function"),
                minimum_version: None,
            }],
            capabilities: Vec::new(),
            exports: vec![Export::Function {
                symbol: Symbol::qualified("femm", "as-ode-rhs"),
                function_id: None,
            }],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.function_value(
            Symbol::qualified("femm", "as-ode-rhs"),
            DefaultFactory.opaque(Arc::new(FemmAsOdeRhsFunction))?,
        )?;
        Ok(())
    }
}

#[derive(Clone)]
struct FemmAsOdeRhsFunction;

impl Object for FemmAsOdeRhsFunction {
    fn display(&self, _cx: &mut Cx) -> KernelResult<String> {
        Ok("#<function femm/as-ode-rhs>".to_owned())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for FemmAsOdeRhsFunction {
    fn class(&self, cx: &mut Cx) -> KernelResult<ClassRef> {
        if let Some(class) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "Function"))
        {
            return Ok(class.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("core", "Function"),
        )
    }
    fn as_expr(&self, _cx: &mut Cx) -> KernelResult<Expr> {
        Ok(Expr::Symbol(Symbol::qualified("femm", "as-ode-rhs")))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for FemmAsOdeRhsFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> KernelResult<Value> {
        let [model, state, rhs] = args.values() else {
            return Err(Error::Eval(
                "femm/as-ode-rhs expects model, state list, rhs expr list".to_owned(),
            ));
        };
        let model = model
            .object()
            .downcast_ref::<ModelValue>()
            .map(|value| value.model.clone())
            .ok_or_else(|| Error::Eval("expected FEMM model value".to_owned()))?;
        let state_vars = parse_symbol_list(cx, state)?;
        let rhs = parse_expr_list(cx, rhs)?;
        let func = FemmOdeRhs {
            model,
            state_vars: state_vars.clone(),
            param_map: state_vars
                .iter()
                .cloned()
                .map(|symbol| (symbol.clone(), symbol))
                .collect(),
            need: Vec::new(),
            rhs,
            tape: Arc::new(Mutex::new(SolveTape::default())),
        }
        .as_func();
        cx.factory().opaque(Arc::new(func))
    }

    fn call_exprs(&self, cx: &mut Cx, args: RawArgs) -> KernelResult<Value> {
        let values = args
            .into_exprs()
            .into_iter()
            .map(|expr| cx.eval_expr(expr))
            .collect::<KernelResult<Vec<_>>>()?;
        self.call(cx, Args::new(values))
    }
}

fn parse_symbol_list(cx: &mut Cx, value: &Value) -> KernelResult<Vec<Symbol>> {
    match value.object().as_expr(cx)? {
        Expr::List(items) | Expr::Vector(items) => items
            .into_iter()
            .map(|expr| match expr {
                Expr::Symbol(symbol) => Ok(symbol),
                Expr::Quote { expr, .. } => match *expr {
                    Expr::Symbol(symbol) => Ok(symbol),
                    _ => Err(Error::Eval("expected quoted symbol".to_owned())),
                },
                _ => Err(Error::Eval("expected symbol".to_owned())),
            })
            .collect(),
        _ => Err(Error::Eval("expected symbol list".to_owned())),
    }
}

fn parse_expr_list(cx: &mut Cx, value: &Value) -> KernelResult<Vec<Expr>> {
    match value.object().as_expr(cx)? {
        Expr::List(items) | Expr::Vector(items) => Ok(items),
        _ => Err(Error::Eval("expected expression list".to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sim_kernel::{Args, DefaultFactory, EagerPolicy, Expr};
    use sim_lib_femm_fixtures::parallel_plate_capacitor;

    use super::*;

    fn num(text: &str) -> Expr {
        sim_value::build::num_q(Some("numbers"), "f64", text)
    }

    #[test]
    fn mock_force_rhs_reproduces_linear_state_equations() {
        let rhs = FemmOdeRhs {
            model: parallel_plate_capacitor(),
            state_vars: vec![Symbol::new("x"), Symbol::new("v")],
            param_map: Vec::new(),
            need: Vec::new(),
            rhs: vec![
                Expr::Symbol(Symbol::new("v")),
                Expr::Call {
                    operator: Box::new(Expr::Symbol(Symbol::new("*"))),
                    args: vec![num("-4.0"), Expr::Symbol(Symbol::new("x"))],
                },
            ],
            tape: Arc::new(Mutex::new(SolveTape::default())),
        };
        let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
        let value = cx
            .call_value(
                cx.factory().opaque(Arc::new(rhs.as_func())).unwrap(),
                Args::new(vec![
                    cx.factory()
                        .number_literal(Symbol::qualified("numbers", "f64"), "0.5".to_owned())
                        .unwrap(),
                    cx.factory()
                        .number_literal(Symbol::qualified("numbers", "f64"), "0.25".to_owned())
                        .unwrap(),
                ]),
            )
            .unwrap();
        let expr = value.object().as_expr(&mut cx).unwrap();
        let Expr::List(items) = expr else {
            panic!("expected list output");
        };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], num("0.25"));
        assert_eq!(items[1], num("-2"));
    }
}
