//! Library registration that exposes FEMM callables to the runtime.
//!
//! Defines the `Lib` that installs the FEMM function exports and the built-in
//! fixture models so the runtime can call them by name.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    AbiVersion, Args, Callable, ClassRef, Cx, DefaultFactory, Dependency, Error, Export, Expr,
    Factory, Lib, LibManifest, LibTarget, Linker, Object, RawArgs, Result as KernelResult, Symbol,
    Value, Version,
};
use sim_lib_femm_core::{FemmLimits, ParamSet};
use sim_lib_femm_field::Projection;
use sim_lib_femm_fixtures::{
    air_core_solenoid, field_as_number_line_integration, gapped_ei_core_inductor,
    parallel_plate_capacitor, plunger_actuator_ode, slab_heat_conductor,
    uniform_conductor_resistance,
};
use sim_lib_femm_mesh::FemmModel;
use sim_lib_femm_post::QuantitySpec;

use crate::model_value::{ModelValue, model_value};
use crate::{FemmCall, FemmCallable, ModelCallable, OutputQuery, femm_as_func};

/// The runtime library that installs the FEMM function exports.
///
/// Registers `femm/model`, `femm/eval`, `femm/as-func`, `femm/field`, and
/// `femm/grad` as callables so the runtime can build, evaluate, and
/// differentiate models by name. See the [crate README](index.html).
pub struct FemmFunctionLib;

impl FemmFunctionLib {
    /// Creates the library installer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FemmFunctionLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for FemmFunctionLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("femm", "function"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![Dependency {
                id: Symbol::qualified("femm", "field"),
                minimum_version: None,
            }],
            capabilities: Vec::new(),
            exports: function_symbols()
                .into_iter()
                .map(|symbol| Export::Function {
                    symbol,
                    function_id: None,
                })
                .collect(),
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        for symbol in function_symbols() {
            linker.function_value(
                symbol.clone(),
                DefaultFactory.opaque(Arc::new(FemmFunctionValue { symbol }))?,
            )?;
        }
        Ok(())
    }
}

fn function_symbols() -> Vec<Symbol> {
    vec![
        Symbol::qualified("femm", "model"),
        Symbol::qualified("femm", "eval"),
        Symbol::qualified("femm", "as-func"),
        Symbol::qualified("femm", "field"),
        Symbol::qualified("femm", "grad"),
    ]
}

#[derive(Clone)]
struct FemmFunctionValue {
    symbol: Symbol,
}

impl Object for FemmFunctionValue {
    fn display(&self, _cx: &mut Cx) -> KernelResult<String> {
        Ok(format!("#<function {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for FemmFunctionValue {
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
        Ok(Expr::Symbol(self.symbol.clone()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for FemmFunctionValue {
    fn call(&self, cx: &mut Cx, args: Args) -> KernelResult<Value> {
        match self.symbol.to_string().as_str() {
            "femm/model" => call_model(cx, args.into_vec()),
            "femm/eval" => call_eval(cx, args.into_vec()),
            "femm/as-func" => call_as_func(cx, args.into_vec()),
            "femm/field" => call_field(cx, args.into_vec()),
            "femm/grad" => call_grad(cx, args.into_vec()),
            _ => Err(Error::Eval(format!(
                "Unknown FEMM function {}",
                self.symbol
            ))),
        }
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

fn call_model(cx: &mut Cx, args: Vec<Value>) -> KernelResult<Value> {
    let model = match args.as_slice() {
        [] => parallel_plate_capacitor(),
        [name] => example_model(symbolish_or_string(cx, name)?.as_str())
            .ok_or_else(|| Error::Eval("unknown FEMM example model".to_owned()))?,
        _ => {
            return Err(Error::Eval(
                "femm/model expects zero or one example name".to_owned(),
            ));
        }
    };
    cx.factory().opaque(Arc::new(model_value(model)))
}

fn call_eval(cx: &mut Cx, args: Vec<Value>) -> KernelResult<Value> {
    let [model, query, params] = args.as_slice() else {
        return Err(Error::Eval(
            "femm/eval expects model, query, params".to_owned(),
        ));
    };
    let model = model_from_value(model)?;
    let query = scalar_query_from_value(cx, query)?;
    let params = params_from_value(cx, params)?;
    ModelCallable { model }
        .eval(
            cx,
            FemmCall {
                params,
                query,
                want_grad: None,
                limits: FemmLimits::default(),
            },
        )
        .map(|out| out.value)
        .map_err(Error::from)
}

fn call_as_func(cx: &mut Cx, args: Vec<Value>) -> KernelResult<Value> {
    let [model, vars, query] = args.as_slice() else {
        return Err(Error::Eval(
            "femm/as-func expects model, vars, query".to_owned(),
        ));
    };
    let model = model_from_value(model)?;
    let vars = symbol_list_from_value(cx, vars)?;
    let query = scalar_query_from_value(cx, query)?;
    cx.factory()
        .opaque(Arc::new(femm_as_func(model, vars, query)))
}

fn call_field(cx: &mut Cx, args: Vec<Value>) -> KernelResult<Value> {
    let [model, projection, params] = args.as_slice() else {
        return Err(Error::Eval(
            "femm/field expects model, projection, params".to_owned(),
        ));
    };
    let model = model_from_value(model)?;
    let projection = projection_from_value(cx, projection)?;
    let params = params_from_value(cx, params)?;
    ModelCallable { model }
        .eval(
            cx,
            FemmCall {
                params,
                query: OutputQuery::Field(projection),
                want_grad: None,
                limits: FemmLimits::default(),
            },
        )
        .map(|out| out.value)
        .map_err(Error::from)
}

fn call_grad(cx: &mut Cx, args: Vec<Value>) -> KernelResult<Value> {
    let [model, query, wrt, params] = args.as_slice() else {
        return Err(Error::Eval(
            "femm/grad expects model, query, wrt, params".to_owned(),
        ));
    };
    let model = model_from_value(model)?;
    let query = scalar_query_from_value(cx, query)?;
    let wrt = symbol_list_from_value(cx, wrt)?;
    let params = params_from_value(cx, params)?;
    let gradient = gradient_pairs(cx, &ModelCallable { model }, query, params, &wrt)?;
    cx.factory().list(
        gradient
            .into_iter()
            .map(|(symbol, value)| {
                cx.factory().list(vec![
                    cx.factory().symbol(symbol)?,
                    cx.factory()
                        .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())?,
                ])
            })
            .collect::<KernelResult<Vec<_>>>()?,
    )
}

fn gradient_pairs(
    cx: &mut Cx,
    callable: &ModelCallable,
    query: OutputQuery,
    params: ParamSet,
    wrt: &[Symbol],
) -> KernelResult<Vec<(Symbol, f64)>> {
    let mut out = Vec::new();
    for symbol in wrt {
        let base_value = params
            .get(symbol)
            .ok_or_else(|| Error::Eval(format!("unknown FEMM parameter {symbol}")))?;
        let x = sim_lib_femm_core::value_as_f64(cx, base_value).map_err(Error::from)?;
        let h = 1.0e-6;
        let plus = replace_param(cx, &params, symbol, x + h)?;
        let minus = replace_param(cx, &params, symbol, x - h)?;
        let plus_value = eval_scalar(cx, callable, query.clone(), plus)?;
        let minus_value = eval_scalar(cx, callable, query.clone(), minus)?;
        out.push((symbol.clone(), (plus_value - minus_value) / (2.0 * h)));
    }
    Ok(out)
}

fn model_from_value(value: &Value) -> KernelResult<FemmModel> {
    value
        .object()
        .downcast_ref::<ModelValue>()
        .map(|model| model.model.clone())
        .ok_or_else(|| Error::Eval("expected FEMM model value".to_owned()))
}

fn example_model(name: &str) -> Option<FemmModel> {
    Some(match name {
        "parallel-plate-capacitor" => parallel_plate_capacitor(),
        "slab-heat-conductor" => slab_heat_conductor(),
        "uniform-conductor-resistance" => uniform_conductor_resistance(),
        "air-core-solenoid" => air_core_solenoid(),
        "gapped-ei-core-inductor" => gapped_ei_core_inductor(),
        "plunger-actuator-ode" => plunger_actuator_ode(),
        "field-as-number-line-integration" => field_as_number_line_integration(),
        _ => return None,
    })
}

fn symbolish_or_string(cx: &mut Cx, value: &Value) -> KernelResult<String> {
    match value.object().as_expr(cx)? {
        Expr::Symbol(symbol) => Ok(symbol.to_string()),
        Expr::String(text) => Ok(text),
        Expr::Quote { expr, .. } => match *expr {
            Expr::Symbol(symbol) => Ok(symbol.to_string()),
            _ => Err(Error::Eval("expected symbol or string".to_owned())),
        },
        _ => Err(Error::Eval("expected symbol or string".to_owned())),
    }
}

fn symbol_list_from_value(cx: &mut Cx, value: &Value) -> KernelResult<Vec<Symbol>> {
    match value.object().as_expr(cx)? {
        Expr::List(items) | Expr::Vector(items) => items
            .into_iter()
            .map(expr_to_symbol)
            .collect::<KernelResult<Vec<_>>>(),
        _ => Err(Error::Eval("expected symbol list".to_owned())),
    }
}

fn expr_to_symbol(expr: Expr) -> KernelResult<Symbol> {
    match expr {
        Expr::Symbol(symbol) => Ok(symbol),
        Expr::Quote { expr, .. } => match *expr {
            Expr::Symbol(symbol) => Ok(symbol),
            _ => Err(Error::Eval("expected quoted symbol".to_owned())),
        },
        _ => Err(Error::Eval("expected symbol".to_owned())),
    }
}

fn params_from_value(cx: &mut Cx, value: &Value) -> KernelResult<ParamSet> {
    match value.object().as_expr(cx)? {
        Expr::Map(entries) => Ok(ParamSet::new(
            entries
                .into_iter()
                .map(|(key, value_expr)| Ok((expr_to_symbol(key)?, cx.eval_expr(value_expr)?)))
                .collect::<KernelResult<Vec<_>>>()?,
        )),
        Expr::List(items) | Expr::Vector(items) => Ok(ParamSet::new(
            items
                .into_iter()
                .map(|item| match item {
                    Expr::List(pair) | Expr::Vector(pair) if pair.len() == 2 => Ok((
                        expr_to_symbol(pair[0].clone())?,
                        cx.eval_expr(pair[1].clone())?,
                    )),
                    _ => Err(Error::Eval(
                        "expected [symbol value] param entry".to_owned(),
                    )),
                })
                .collect::<KernelResult<Vec<_>>>()?,
        )),
        Expr::Nil => Ok(ParamSet::default()),
        _ => Err(Error::Eval(
            "expected parameter table or pair list".to_owned(),
        )),
    }
}

fn scalar_query_from_value(cx: &mut Cx, value: &Value) -> KernelResult<OutputQuery> {
    Ok(OutputQuery::Quantity(QuantitySpec::Custom {
        name: Symbol::new("q"),
        expr: value.object().as_expr(cx)?,
    }))
}

fn projection_from_value(cx: &mut Cx, value: &Value) -> KernelResult<Projection> {
    match symbolish_or_string(cx, value)?.as_str() {
        "potential" => Ok(Projection::Potential),
        "bx" => Ok(Projection::Bx),
        "by" => Ok(Projection::By),
        "bmag" => Ok(Projection::Bmag),
        "ex" => Ok(Projection::Ex),
        "ey" => Ok(Projection::Ey),
        "emag" => Ok(Projection::Emag),
        "heat-flux-mag" => Ok(Projection::HeatFluxMag),
        other => Err(Error::Eval(format!("unknown FEMM projection {other}"))),
    }
}

fn replace_param(
    cx: &mut Cx,
    params: &ParamSet,
    name: &Symbol,
    value: f64,
) -> KernelResult<ParamSet> {
    let mut entries = params.entries.clone();
    for (symbol, slot) in &mut entries {
        if symbol == name {
            *slot = cx
                .factory()
                .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())?;
        }
    }
    Ok(ParamSet::new(entries))
}

fn eval_scalar(
    cx: &mut Cx,
    callable: &ModelCallable,
    query: OutputQuery,
    params: ParamSet,
) -> KernelResult<f64> {
    let eval = callable
        .eval(
            cx,
            FemmCall {
                params,
                query,
                want_grad: None,
                limits: FemmLimits::default(),
            },
        )
        .map_err(Error::from)?;
    sim_lib_femm_core::value_as_f64(cx, &eval.value).map_err(Error::from)
}
