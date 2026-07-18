use std::sync::Arc;

use sim_kernel::{Args, Cx, DefaultFactory, EagerPolicy, Expr};
use sim_lib_femm_core::FemmLimits;
use sim_lib_femm_fixtures::gapped_ei_core_inductor;
use sim_lib_femm_material::Source;
use sim_lib_femm_post::QuantitySpec;
use sim_lib_femm_query::{ModelCallable, OutputQuery, femm_as_func};
use sim_lib_femm_solve::{GradientTrust, solve_steady};
use sim_lib_numbers_numeric::{NumericNumbersLib, global_numeric_registry, numeric_diff_symbol};

#[path = "tests_support.rs"]
mod support;

use crate::{
    SensitivityPath, adjoint_gradient, gradient, gradient_answer, register_femm_adjoint,
    total_gradient,
};
use support::{
    boundary_model, call, central_fd_quantity_gradient, custom_query,
    custom_query_with_default_offset, gap_mm_params, gap_params, model, model_with_default_offset,
    num, parametric_box_model, params, scalar_fd_quantity_gradient, width_height_params,
};

#[test]
fn direct_exact_gradient_matches_adjoint_and_fd_fallback_paths() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let base_params = params(&mut cx);
    let (direct, direct_path) = gradient(
        &mut cx,
        &model(),
        custom_query(),
        base_params.clone(),
        &[sim_kernel::Symbol::new("gap")],
    )
    .unwrap();
    let (adjoint, adjoint_path) = adjoint_gradient(
        &mut cx,
        &model(),
        custom_query(),
        base_params.clone(),
        &[sim_kernel::Symbol::new("gap")],
    )
    .unwrap();
    let (builtin, builtin_path) = adjoint_gradient(
        &mut cx,
        &model(),
        OutputQuery::Quantity(QuantitySpec::JouleLoss { region: None }),
        base_params,
        &[sim_kernel::Symbol::new("gap")],
    )
    .unwrap();
    assert_eq!(direct_path, SensitivityPath::DirectExact);
    assert_eq!(adjoint_path, SensitivityPath::AdjointExact);
    assert_eq!(builtin_path, SensitivityPath::AdjointExact);
    assert!((direct[0].1 - 2.0).abs() < 1.0e-12);
    assert!((adjoint[0].1 - 2.0).abs() < 1.0e-12);
    assert_eq!(builtin[0].1, 0.0);
}

#[test]
fn dependent_builtin_quantity_uses_exact_adjoint_path() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let base_params = params(&mut cx);
    let (gradient, path) = adjoint_gradient(
        &mut cx,
        &boundary_model(),
        OutputQuery::Quantity(QuantitySpec::Energy { region: None }),
        base_params,
        &[sim_kernel::Symbol::new("gap")],
    )
    .unwrap();
    assert_eq!(path, SensitivityPath::AdjointExact);
    assert!((gradient[0].1 - 0.5).abs() < 1.0e-9);
}

#[test]
fn boundary_derivative_errors_are_not_zeroed() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let mut callable = boundary_model();
    callable.model.boundaries[0].value =
        call("sqrt", vec![Expr::Symbol(sim_kernel::Symbol::new("gap"))]);
    let gap = sim_kernel::Symbol::new("gap");
    let base_params = gap_params(&mut cx, "0.0");
    let err = adjoint_gradient(
        &mut cx,
        &callable,
        OutputQuery::Quantity(QuantitySpec::Energy { region: None }),
        base_params,
        std::slice::from_ref(&gap),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        sim_lib_femm_core::FemmError::SensitivityUnavailable(_)
    ));
}

#[test]
fn linear_total_gradient_covers_energy_and_flux() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let callable = parametric_box_model();
    let params = width_height_params(&mut cx);
    let mut solve = solve_steady(
        &mut cx,
        &callable.model,
        &params,
        &FemmLimits::default(),
        None,
    )
    .unwrap();
    let wrt = vec![
        sim_kernel::Symbol::new("width"),
        sim_kernel::Symbol::new("height"),
    ];
    let result = total_gradient(
        &mut cx,
        &callable,
        &mut solve,
        &[
            QuantitySpec::Energy { region: None },
            QuantitySpec::FluxLinkage {
                circuit: sim_kernel::Symbol::new("plate"),
            },
        ],
        &wrt,
    )
    .unwrap();
    assert_eq!(result.gradient.len(), 2);
    assert_eq!(result.gradient[0].len(), 2);
    assert_eq!(result.gradient[1].len(), 2);
    for row in &result.gradient {
        assert!(row.iter().all(|value| value.is_finite()));
    }
    for trust in &result.trust {
        assert!(!matches!(trust, GradientTrust::AdjointUnverified));
    }
    assert!(!matches!(
        solve.certificate.gradient_trust,
        Some(GradientTrust::AdjointUnverified)
    ));
}

#[test]
fn nonlinear_total_gradient_covers_gapped_ei_core_inductor() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let callable = ModelCallable {
        model: gapped_ei_core_inductor(),
    };
    let params = gap_mm_params(&mut cx, "1.0");
    let mut solve = solve_steady(
        &mut cx,
        &callable.model,
        &params,
        &FemmLimits::default(),
        None,
    )
    .unwrap();
    assert_eq!(
        solve.solution.diagnostics.method,
        sim_kernel::Symbol::new("femm-ptc")
    );
    let wrt = vec![sim_kernel::Symbol::new("gap-mm")];
    let spec = QuantitySpec::Energy { region: None };
    let result = total_gradient(
        &mut cx,
        &callable,
        &mut solve,
        std::slice::from_ref(&spec),
        &wrt,
    )
    .unwrap();
    let fd = scalar_fd_quantity_gradient(&mut cx, &callable, &params, &spec).unwrap();
    assert_eq!(result.gradient.len(), 1);
    assert_eq!(result.gradient[0].len(), 1);
    assert!((result.gradient[0][0] - fd).abs() < 1.0e-4);
    assert!(matches!(
        result.trust[0],
        GradientTrust::AdjointVerified { .. } | GradientTrust::FiniteDifferenceOnly
    ));
    assert!(!matches!(result.trust[0], GradientTrust::AdjointUnverified));
    assert!(matches!(
        solve.certificate.gradient_trust,
        Some(GradientTrust::AdjointVerified { .. }) | Some(GradientTrust::FiniteDifferenceOnly)
    ));
    assert!(!matches!(
        solve.certificate.gradient_trust,
        Some(GradientTrust::AdjointUnverified)
    ));
}

#[test]
fn linear_builtin_derivatives_match_finite_difference() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let callable = parametric_box_model();
    let params = width_height_params(&mut cx);
    let width = sim_kernel::Symbol::new("width");
    let specs = [
        QuantitySpec::Inductance {
            circuit: sim_kernel::Symbol::new("plate"),
        },
        QuantitySpec::FluxLinkage {
            circuit: sim_kernel::Symbol::new("plate"),
        },
        QuantitySpec::Capacitance {
            conductor: sim_kernel::Symbol::new("wall"),
        },
    ];
    for spec in specs {
        let (gradient, path) = adjoint_gradient(
            &mut cx,
            &callable,
            OutputQuery::Quantity(spec.clone()),
            params.clone(),
            std::slice::from_ref(&width),
        )
        .unwrap();
        assert_eq!(
            path,
            SensitivityPath::AdjointExact,
            "expected the exact analytic path for {spec:?}"
        );
        let analytic = gradient[0].1;
        let fd = central_fd_quantity_gradient(&mut cx, &callable, &params, &spec, &width).unwrap();
        assert!(
            (analytic - fd).abs() < 1.0e-4,
            "{spec:?}: analytic {analytic} vs fd {fd}"
        );
    }
}

#[test]
fn excitation_dependent_inductance_falls_back_to_finite_difference() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let mut callable = parametric_box_model();
    // A drive that itself depends on the design parameter: dI/dp != 0.
    callable.model.sources = vec![Source::CircuitCoil {
        name: sim_kernel::Symbol::new("plate"),
        region: sim_kernel::Symbol::new("air"),
        turns: num("1.0"),
        current: Expr::Symbol(sim_kernel::Symbol::new("width")),
    }];
    let params = width_height_params(&mut cx);
    let width = sim_kernel::Symbol::new("width");
    let spec = QuantitySpec::Inductance {
        circuit: sim_kernel::Symbol::new("plate"),
    };
    // The exact analytic path must refuse rather than drop the dI/dp term.
    assert!(
        adjoint_gradient(
            &mut cx,
            &callable,
            OutputQuery::Quantity(spec.clone()),
            params.clone(),
            std::slice::from_ref(&width),
        )
        .is_err(),
        "excitation-dependent inductance must not use the exact analytic path"
    );
    // total_gradient turns that refusal into a finite-difference fallback.
    let mut solve = solve_steady(
        &mut cx,
        &callable.model,
        &params,
        &FemmLimits::default(),
        None,
    )
    .unwrap();
    let result = total_gradient(
        &mut cx,
        &callable,
        &mut solve,
        std::slice::from_ref(&spec),
        std::slice::from_ref(&width),
    )
    .unwrap();
    assert!(matches!(
        result.trust[0],
        GradientTrust::FiniteDifferenceOnly
    ));
    assert!(result.gradient[0][0].is_finite());
}

#[test]
fn gradient_answer_reports_finite_difference_trust() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let mut callable = parametric_box_model();
    callable.model.sources = vec![Source::CircuitCoil {
        name: sim_kernel::Symbol::new("plate"),
        region: sim_kernel::Symbol::new("air"),
        turns: num("1.0"),
        current: Expr::Symbol(sim_kernel::Symbol::new("width")),
    }];
    let width = sim_kernel::Symbol::new("width");
    let spec = QuantitySpec::Inductance {
        circuit: sim_kernel::Symbol::new("plate"),
    };
    let params = width_height_params(&mut cx);
    let answer = gradient_answer(
        &mut cx,
        &callable,
        OutputQuery::Quantity(spec),
        params,
        std::slice::from_ref(&width),
    )
    .unwrap();
    assert!(matches!(answer.trust, GradientTrust::FiniteDifferenceOnly));
    assert!(answer.values[0].1.is_finite());
    assert!(
        cx.take_diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.message.contains("trust=finite-difference-only"))
    );
}

#[test]
fn nonlinear_state_derivative_is_energy_only() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let callable = ModelCallable {
        model: gapped_ei_core_inductor(),
    };
    let params = gap_mm_params(&mut cx, "1.0");
    let solve = solve_steady(
        &mut cx,
        &callable.model,
        &params,
        &FemmLimits::default(),
        None,
    )
    .unwrap();
    // Energy has a closed-form state derivative; inductance/flux/capacitance do
    // not, so the nonlinear adjoint stays on its correct finite-difference path.
    assert!(
        crate::nonlinear_adjoint::quantity_state_derivative(
            &solve.solution,
            &QuantitySpec::Energy { region: None },
        )
        .unwrap()
        .is_some()
    );
    for spec in [
        QuantitySpec::Inductance {
            circuit: sim_kernel::Symbol::new("coil"),
        },
        QuantitySpec::FluxLinkage {
            circuit: sim_kernel::Symbol::new("coil"),
        },
        QuantitySpec::Capacitance {
            conductor: sim_kernel::Symbol::new("plate"),
        },
    ] {
        assert!(
            crate::nonlinear_adjoint::quantity_state_derivative(&solve.solution, &spec)
                .unwrap()
                .is_none(),
            "{spec:?} must have no closed-form state derivative"
        );
    }
}

#[test]
fn numeric_diff_with_femm_adjoint_uses_plugin_payload() {
    register_femm_adjoint().unwrap();
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&NumericNumbersLib::new()).unwrap();
    let func = femm_as_func(
        model().model.clone(),
        vec![sim_kernel::Symbol::new("gap")],
        custom_query(),
    );
    let out = cx
        .call_function(
            &numeric_diff_symbol(),
            Args::new(vec![
                cx.factory().opaque(Arc::new(func)).unwrap(),
                cx.factory().symbol(sim_kernel::Symbol::new("gap")).unwrap(),
                cx.factory()
                    .number_literal(
                        sim_kernel::Symbol::qualified("numbers", "f64"),
                        "0.5".to_owned(),
                    )
                    .unwrap(),
                cx.factory()
                    .table(vec![(
                        sim_kernel::Symbol::new(":method"),
                        cx.factory()
                            .symbol(sim_kernel::Symbol::new("femm-adjoint"))
                            .unwrap(),
                    )])
                    .unwrap(),
            ]),
        )
        .unwrap();
    assert!((sim_lib_femm_core::value_as_f64(&mut cx, &out).unwrap() - 2.0).abs() < 1.0e-12);
}

#[test]
fn numeric_diff_with_femm_adjoint_resolves_model_defaults() {
    register_femm_adjoint().unwrap();
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&NumericNumbersLib::new()).unwrap();
    let callable = model_with_default_offset(&mut cx);
    let func = femm_as_func(
        callable.model,
        vec![sim_kernel::Symbol::new("gap")],
        custom_query_with_default_offset(),
    );
    let out = cx
        .call_function(
            &numeric_diff_symbol(),
            Args::new(vec![
                cx.factory().opaque(Arc::new(func)).unwrap(),
                cx.factory().symbol(sim_kernel::Symbol::new("gap")).unwrap(),
                cx.factory()
                    .number_literal(
                        sim_kernel::Symbol::qualified("numbers", "f64"),
                        "0.5".to_owned(),
                    )
                    .unwrap(),
                cx.factory()
                    .table(vec![(
                        sim_kernel::Symbol::new(":method"),
                        cx.factory()
                            .symbol(sim_kernel::Symbol::new("femm-adjoint"))
                            .unwrap(),
                    )])
                    .unwrap(),
            ]),
        )
        .unwrap();
    assert!((sim_lib_femm_core::value_as_f64(&mut cx, &out).unwrap() - 2.0).abs() < 1.0e-12);
    assert!(cx.take_diagnostics().iter().any(|diagnostic| {
        diagnostic
            .message
            .contains("femm-adjoint trust=adjoint-verified")
    }));
}

#[test]
fn adjoint_plugin_registers() {
    register_femm_adjoint().unwrap();
    let guard = global_numeric_registry().read().unwrap();
    assert!(
        guard
            .differentiator(&sim_kernel::Symbol::new("femm-adjoint"))
            .is_some()
    );
}
