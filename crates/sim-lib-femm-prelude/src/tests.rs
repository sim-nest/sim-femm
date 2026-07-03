use std::sync::Arc;

use sim_kernel::{Args, Cx, DefaultFactory, EagerPolicy, NumberValue, Symbol};
use sim_lib_numbers_numeric::global_numeric_registry;

use crate::FemmPreludeLib;

#[test]
fn prelude_exposes_stable_stack() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    FemmPreludeLib::new().install_all(&mut cx).unwrap();
    FemmPreludeLib::new().install_all(&mut cx).unwrap();
    assert!(
        cx.registry()
            .lib(&Symbol::qualified("femm", "core"))
            .is_some()
    );
    assert!(
        cx.registry()
            .number_domain_by_symbol(&Symbol::qualified("numbers", "field"))
            .is_some()
    );
    for symbol in [
        Symbol::qualified("femm", "model"),
        Symbol::qualified("femm", "eval"),
        Symbol::qualified("femm", "as-func"),
        Symbol::qualified("femm", "field"),
        Symbol::qualified("femm", "grad"),
        Symbol::qualified("femm", "as-ode-rhs"),
    ] {
        assert!(
            cx.registry().function_by_symbol(&symbol).is_some(),
            "missing {symbol}"
        );
    }
    let guard = global_numeric_registry().read().unwrap();
    assert!(guard.ode_fixed(&Symbol::new("femm-ptc")).is_some());
    assert!(guard.differentiator(&Symbol::new("femm-adjoint")).is_some());
}

#[test]
fn documented_femm_forms_are_callable() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    FemmPreludeLib::new().install_all(&mut cx).unwrap();

    let model = cx
        .call_function(&Symbol::qualified("femm", "model"), Args::default())
        .unwrap();
    let eval = cx
        .call_function(
            &Symbol::qualified("femm", "eval"),
            Args::new(vec![
                model.clone(),
                cx.factory()
                    .expr(sim_kernel::Expr::Number(sim_kernel::NumberLiteral {
                        domain: Symbol::qualified("numbers", "f64"),
                        canonical: "3.0".to_owned(),
                    }))
                    .unwrap(),
                cx.factory().nil().unwrap(),
            ]),
        )
        .unwrap();
    assert_eq!(eval.object().display(&mut cx).unwrap(), "3");

    let grad = cx
        .call_function(
            &Symbol::qualified("femm", "grad"),
            Args::new(vec![
                model.clone(),
                cx.factory()
                    .expr(sim_kernel::Expr::Symbol(Symbol::new("gap-mm")))
                    .unwrap(),
                cx.factory()
                    .list(vec![cx.factory().symbol(Symbol::new("gap-mm")).unwrap()])
                    .unwrap(),
                cx.factory()
                    .list(vec![
                        cx.factory()
                            .list(vec![
                                cx.factory().symbol(Symbol::new("gap-mm")).unwrap(),
                                cx.factory()
                                    .number_literal(
                                        Symbol::qualified("numbers", "f64"),
                                        "0.5".to_owned(),
                                    )
                                    .unwrap(),
                            ])
                            .unwrap(),
                    ])
                    .unwrap(),
            ]),
        )
        .unwrap();
    assert!(matches!(
        grad.object().as_expr(&mut cx).unwrap(),
        sim_kernel::Expr::List(_)
    ));

    let field = cx
        .call_function(
            &Symbol::qualified("femm", "field"),
            Args::new(vec![
                model.clone(),
                cx.factory().string("potential".to_owned()).unwrap(),
                cx.factory().nil().unwrap(),
            ]),
        )
        .unwrap();
    assert_eq!(
        field
            .object()
            .downcast_ref::<sim_lib_femm_field::Field>()
            .unwrap()
            .number_domain(&mut cx)
            .unwrap(),
        Symbol::qualified("numbers", "field")
    );

    let ode = cx
        .call_function(
            &Symbol::qualified("femm", "as-ode-rhs"),
            Args::new(vec![
                model,
                cx.factory()
                    .list(vec![cx.factory().symbol(Symbol::new("x")).unwrap()])
                    .unwrap(),
                cx.factory()
                    .list(vec![
                        cx.factory()
                            .expr(sim_kernel::Expr::Symbol(Symbol::new("x")))
                            .unwrap(),
                    ])
                    .unwrap(),
            ]),
        )
        .unwrap();
    assert!(ode.object().as_callable().is_some());
}
