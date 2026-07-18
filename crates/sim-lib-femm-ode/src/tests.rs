use std::sync::{Arc, Mutex};

use sim_kernel::{Args, Cx, DefaultFactory, EagerPolicy, Expr, Symbol, Value};
use sim_lib_femm_fixtures::parallel_plate_capacitor;
use sim_lib_femm_function::{OutputQuery, femm_as_func, model_value};
use sim_lib_femm_post::QuantitySpec;
use sim_lib_femm_sensitiv::register_femm_adjoint;
use sim_lib_femm_tape::SolveTape;
use sim_lib_numbers_numeric::{NumericNumbersLib, global_numeric_registry, numeric_diff_symbol};
use sim_lib_numbers_quad::QuadNumbersLib;
use sim_lib_numbers_rk::RkNumbersLib;

use crate::{FemmOdeLib, FemmOdeRhs};

fn num(text: &str) -> Expr {
    sim_value::build::num_q(Some("numbers"), "f64", text)
}

fn numeric_cx() -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    cx.load_lib(&NumericNumbersLib::new()).unwrap();
    cx.load_lib(&RkNumbersLib::new()).unwrap();
    cx.load_lib(&QuadNumbersLib::new()).unwrap();
    cx
}

fn f64_value(cx: &mut Cx, value: f64) -> Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
        .unwrap()
}

fn value_to_f64(cx: &mut Cx, value: &Value) -> f64 {
    sim_lib_femm_core::value_as_f64(cx, value).unwrap()
}

fn expr_to_f64(cx: &mut Cx, expr: Expr) -> f64 {
    let value = cx.eval_expr(expr).unwrap();
    value_to_f64(cx, &value)
}

fn last_ode_y(cx: &mut Cx, value: &Value) -> f64 {
    let Expr::List(points) = value.object().as_expr(cx).unwrap() else {
        panic!("expected ODE point list");
    };
    let Expr::List(pair) = points.last().cloned().unwrap() else {
        panic!("expected ODE point pair");
    };
    expr_to_f64(cx, pair[1].clone())
}

fn twice_gap_query(gap: &Symbol) -> OutputQuery {
    OutputQuery::Quantity(QuantitySpec::Custom {
        name: Symbol::new("q"),
        expr: Expr::Call {
            operator: Box::new(Expr::Symbol(Symbol::new("*"))),
            args: vec![num("2.0"), Expr::Symbol(gap.clone())],
        },
    })
}

fn expect_ode_shape_err(
    state_vars: Vec<Symbol>,
    param_map: Vec<(Symbol, Symbol)>,
    rhs: Vec<Expr>,
    expected: &str,
) {
    let err = match FemmOdeRhs::new(
        parallel_plate_capacitor(),
        state_vars,
        param_map,
        Vec::new(),
        rhs,
        Arc::new(Mutex::new(SolveTape::default())),
    ) {
        Ok(_) => panic!("expected malformed ODE RHS shape"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains(expected),
        "expected {expected:?}, got {err}"
    );
}

#[test]
fn mock_force_rhs_reproduces_linear_state_equations() {
    let rhs = FemmOdeRhs::new(
        parallel_plate_capacitor(),
        vec![Symbol::new("x"), Symbol::new("v")],
        Vec::new(),
        Vec::new(),
        vec![
            Expr::Symbol(Symbol::new("v")),
            Expr::Infix {
                operator: Symbol::new("*"),
                left: Box::new(num("-4.0")),
                right: Box::new(Expr::Symbol(Symbol::new("x"))),
            },
        ],
        Arc::new(Mutex::new(SolveTape::default())),
    )
    .unwrap();
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let func = cx.factory().opaque(Arc::new(rhs.as_func())).unwrap();
    let t = f64_value(&mut cx, 0.5);
    let y = f64_value(&mut cx, 0.25);
    let value = cx.call_value(func, Args::new(vec![t, y])).unwrap();
    let expr = value.object().as_expr(&mut cx).unwrap();
    let Expr::List(items) = expr else {
        panic!("expected list output");
    };
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], num("0.25"));
    assert_eq!(items[1], num("-2"));
}

#[test]
fn femm_ode_rhs_integrates_through_numbers_ode_solve() {
    let mut cx = numeric_cx();
    let time = Symbol::new("t");
    let state = Symbol::new("y");
    let rhs = FemmOdeRhs::new(
        parallel_plate_capacitor(),
        vec![time.clone(), state.clone()],
        Vec::new(),
        Vec::new(),
        vec![Expr::Symbol(state.clone())],
        Arc::new(Mutex::new(SolveTape::default())),
    )
    .unwrap();
    let method = cx.factory().symbol(Symbol::new("rk4")).unwrap();
    let step = f64_value(&mut cx, 0.1);
    let options = cx
        .factory()
        .table(vec![
            (Symbol::new(":method"), method),
            (Symbol::new(":h"), step),
        ])
        .unwrap();
    let rhs = cx.factory().opaque(Arc::new(rhs.as_func())).unwrap();
    let time = cx.factory().symbol(time).unwrap();
    let state = cx.factory().symbol(state).unwrap();
    let t0 = f64_value(&mut cx, 0.0);
    let y0 = f64_value(&mut cx, 1.0);
    let t1 = f64_value(&mut cx, 1.0);
    let ode_output = cx
        .call_function(
            &Symbol::new("ode-solve"),
            Args::new(vec![rhs, time, state, t0, y0, t1, options]),
        )
        .unwrap();

    let last_y = last_ode_y(&mut cx, &ode_output);
    assert!((last_y - std::f64::consts::E).abs() < 5.0e-6);
}

#[test]
fn femm_ode_rhs_constructor_rejects_malformed_shapes() {
    expect_ode_shape_err(
        Vec::new(),
        Vec::new(),
        vec![num("1.0")],
        "must not be empty",
    );
    expect_ode_shape_err(
        vec![Symbol::new("x"), Symbol::new("x")],
        Vec::new(),
        vec![num("1.0")],
        "duplicate ODE state variable",
    );
    expect_ode_shape_err(
        vec![Symbol::new("x"), Symbol::new("y")],
        Vec::new(),
        vec![num("1.0"), num("2.0"), num("3.0")],
        "RHS arity",
    );
    expect_ode_shape_err(
        vec![Symbol::new("x"), Symbol::new("y")],
        vec![
            (Symbol::new("gap"), Symbol::new("x")),
            (Symbol::new("gap"), Symbol::new("y")),
        ],
        vec![num("1.0")],
        "duplicate ODE parameter-map entry",
    );
    expect_ode_shape_err(
        vec![Symbol::new("x")],
        vec![(Symbol::new("gap"), Symbol::new("missing"))],
        vec![num("1.0")],
        "unknown state",
    );
}

#[test]
fn femm_as_ode_rhs_rejects_malformed_adapter_inputs() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&FemmOdeLib::new()).unwrap();
    let model = cx
        .factory()
        .opaque(Arc::new(model_value(parallel_plate_capacitor())))
        .unwrap();
    let state = cx
        .factory()
        .list(vec![
            cx.factory().symbol(Symbol::new("x")).unwrap(),
            cx.factory().symbol(Symbol::new("x")).unwrap(),
        ])
        .unwrap();
    let rhs = cx
        .factory()
        .expr(Expr::List(vec![Expr::Symbol(Symbol::new("x"))]))
        .unwrap();
    let err = cx
        .call_function(
            &Symbol::qualified("femm", "as-ode-rhs"),
            Args::new(vec![model, state, rhs]),
        )
        .unwrap_err();
    assert!(err.to_string().contains("duplicate ODE state variable"));
}

#[test]
fn femm_func_still_first_class() {
    let mut cx = numeric_cx();
    register_femm_adjoint().unwrap();
    let gap = Symbol::new("gap-mm");
    let func = femm_as_func(
        parallel_plate_capacitor(),
        vec![gap.clone()],
        twice_gap_query(&gap),
    );

    assert_eq!(
        func.metadata.differentiator_hint,
        Some(Symbol::new("femm-adjoint"))
    );
    assert!(
        global_numeric_registry()
            .read()
            .unwrap()
            .differentiator(&Symbol::new("femm-adjoint"))
            .is_some()
    );

    let func_value = cx.factory().opaque(Arc::new(func)).unwrap();
    assert!(
        func_value
            .object()
            .downcast_ref::<sim_lib_numbers_func::Func>()
            .is_some()
    );

    let point = f64_value(&mut cx, 0.5);
    let direct = cx
        .call_value(func_value.clone(), Args::new(vec![point]))
        .unwrap();
    assert!((value_to_f64(&mut cx, &direct) - 1.0).abs() < f64::EPSILON);

    let var = cx.factory().symbol(gap.clone()).unwrap();
    let point = f64_value(&mut cx, 0.5);
    let auto_diff = cx
        .call_function(
            &numeric_diff_symbol(),
            Args::new(vec![func_value.clone(), var, point]),
        )
        .unwrap();
    assert!((value_to_f64(&mut cx, &auto_diff) - 2.0).abs() < 1.0e-8);
    cx.take_diagnostics();

    let explicit_options = cx
        .factory()
        .table(vec![(
            Symbol::new(":method"),
            cx.factory().symbol(Symbol::new("femm-adjoint")).unwrap(),
        )])
        .unwrap();
    let var = cx.factory().symbol(gap.clone()).unwrap();
    let point = f64_value(&mut cx, 0.5);
    let explicit_diff = cx
        .call_function(
            &numeric_diff_symbol(),
            Args::new(vec![func_value.clone(), var, point, explicit_options]),
        )
        .unwrap();
    assert!((value_to_f64(&mut cx, &explicit_diff) - 2.0).abs() < 1.0e-12);
    let diagnostics = cx.take_diagnostics();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("method=femm-adjoint")),
        "{diagnostics:?}"
    );

    let simpson = cx.factory().symbol(Symbol::new("simpson")).unwrap();
    let n = f64_value(&mut cx, 64.0);
    let integrate_options = cx
        .factory()
        .table(vec![
            (Symbol::new(":method"), simpson),
            (Symbol::new(":n"), n),
        ])
        .unwrap();
    let var = cx.factory().symbol(gap).unwrap();
    let lo = f64_value(&mut cx, 0.0);
    let hi = f64_value(&mut cx, 1.0);
    let integral = cx
        .call_function(
            &Symbol::new("integrate"),
            Args::new(vec![func_value, var, lo, hi, integrate_options]),
        )
        .unwrap();
    assert!((value_to_f64(&mut cx, &integral) - 1.0).abs() < 1.0e-10);
}
