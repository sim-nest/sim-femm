use std::sync::Arc;

use sim_kernel::{Cx, DefaultFactory, EagerPolicy, Expr, Symbol};
use sim_lib_femm_core::{
    FEMM_EXPR_OPERATORS, FemmResult, Formulation, LengthUnit, ParamRole, ParamSet, ParamSpec,
    PhysicsKind, StableId, eval_expr_f64, value_as_f64,
};
use sim_lib_femm_function::{ModelCallable, OutputQuery};
use sim_lib_femm_geometry::{Geometry2, dummy_origin};
use sim_lib_femm_material::MeshPolicy;
use sim_lib_femm_post::QuantitySpec;
use sim_lib_femm_sensitiv::{SensitivityPath, adjoint_gradient, gradient};

fn num(text: &str) -> Expr {
    sim_value::build::num_q(Some("numbers"), "f64", text)
}

fn call(operator: &str, args: Vec<Expr>) -> Expr {
    Expr::Call {
        operator: Box::new(Expr::Symbol(Symbol::new(operator))),
        args,
    }
}

fn model() -> ModelCallable {
    ModelCallable {
        model: sim_lib_femm_mesh::FemmModel {
            id: StableId(15),
            name: Symbol::new("review15"),
            physics: PhysicsKind::Electrostatic,
            formulation: Formulation::Planar,
            length_unit: LengthUnit::Meter,
            depth: None,
            frequency_hz: None,
            inputs: vec![ParamSpec {
                name: Symbol::new("gap"),
                default: None,
                unit: None,
                role: ParamRole::Design,
            }],
            geometry: Geometry2::default(),
            materials: Vec::new(),
            boundaries: Vec::new(),
            sources: Vec::new(),
            outputs: Vec::new(),
            mesh_policy: MeshPolicy {
                kind: Symbol::new("det"),
                max_area: None,
                min_angle_deg: None,
            },
            solve_policy: None,
            origin: dummy_origin(),
        },
    }
}

fn param(cx: &mut Cx, symbol: &Symbol, value: f64) -> ParamSet {
    ParamSet::new(vec![(
        symbol.clone(),
        cx.factory()
            .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
            .unwrap(),
    )])
}

fn replace_param(cx: &mut Cx, params: &ParamSet, symbol: &Symbol, value: f64) -> ParamSet {
    let mut entries = params.entries.clone();
    let replacement = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
        .unwrap();
    entries
        .iter_mut()
        .find(|(name, _)| name == symbol)
        .unwrap()
        .1 = replacement;
    ParamSet::new(entries)
}

fn query(expr: Expr) -> OutputQuery {
    OutputQuery::Quantity(QuantitySpec::Custom {
        name: Symbol::new("custom"),
        expr,
    })
}

fn central_difference(
    cx: &mut Cx,
    expr: &Expr,
    params: &ParamSet,
    symbol: &Symbol,
) -> FemmResult<f64> {
    let base = value_as_f64(cx, params.get(symbol).unwrap())?;
    let step = 1.490_116_119_384_765_6e-8 * base.abs().max(1.0);
    let plus = replace_param(cx, params, symbol, base + step);
    let minus = replace_param(cx, params, symbol, base - step);
    let q_plus = eval_expr_f64(cx, expr, &plus, &[])?;
    let q_minus = eval_expr_f64(cx, expr, &minus, &[])?;
    Ok((q_plus - q_minus) / (2.0 * step))
}

#[test]
fn pow_integer_exponent_derivative_is_finite_at_negative_base() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let gap = Symbol::new("gap");
    let params = param(&mut cx, &gap, -2.0);
    let expr = call("pow", vec![Expr::Symbol(gap.clone()), num("2.0")]);
    let fd = central_difference(&mut cx, &expr, &params, &gap).unwrap();

    let (direct, direct_path) = gradient(
        &mut cx,
        &model(),
        query(expr.clone()),
        params.clone(),
        std::slice::from_ref(&gap),
    )
    .unwrap();
    let (adjoint, adjoint_path) = adjoint_gradient(
        &mut cx,
        &model(),
        query(expr),
        params,
        std::slice::from_ref(&gap),
    )
    .unwrap();

    assert_eq!(direct_path, SensitivityPath::DirectExact);
    assert_eq!(adjoint_path, SensitivityPath::AdjointExact);
    assert!(direct[0].1.is_finite());
    assert!(adjoint[0].1.is_finite());
    assert!((direct[0].1 + 4.0).abs() < 1.0e-12);
    assert!((adjoint[0].1 + 4.0).abs() < 1.0e-12);
    assert!((direct[0].1 - fd).abs() < 1.0e-6);
}

#[test]
fn primal_and_ad_agree_on_operator_set() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let gap = Symbol::new("gap");
    let params = param(&mut cx, &gap, 0.5);
    for operator in FEMM_EXPR_OPERATORS {
        let expr = match *operator {
            "+" => call("+", vec![Expr::Symbol(gap.clone()), num("1.0")]),
            "*" => call("*", vec![Expr::Symbol(gap.clone()), num("2.0")]),
            "-" => call("-", vec![Expr::Symbol(gap.clone()), num("0.25")]),
            "/" => call("/", vec![Expr::Symbol(gap.clone()), num("2.0")]),
            "pow" => call("pow", vec![Expr::Symbol(gap.clone()), num("2.0")]),
            "sin" | "cos" | "exp" | "ln" | "sqrt" => {
                call(operator, vec![Expr::Symbol(gap.clone())])
            }
            other => panic!("unhandled FEMM expression operator {other}"),
        };
        assert!(
            eval_expr_f64(&mut cx, &expr, &params, &[]).is_ok(),
            "{operator}"
        );
        let (direct, _) = gradient(
            &mut cx,
            &model(),
            query(expr.clone()),
            params.clone(),
            std::slice::from_ref(&gap),
        )
        .unwrap();
        let (adjoint, _) = adjoint_gradient(
            &mut cx,
            &model(),
            query(expr),
            params.clone(),
            std::slice::from_ref(&gap),
        )
        .unwrap();
        assert!(direct[0].1.is_finite(), "{operator}");
        assert!(adjoint[0].1.is_finite(), "{operator}");
    }
}
