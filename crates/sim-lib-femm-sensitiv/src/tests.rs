use std::sync::Arc;

use sim_kernel::{Args, Cx, DefaultFactory, EagerPolicy, Expr};
use sim_lib_femm_core::{
    FemmLimits, Formulation, LengthUnit, ParamRole, ParamSet, ParamSpec, PhysicsKind, StableId,
};
use sim_lib_femm_fixtures::gapped_ei_core_inductor;
use sim_lib_femm_function::{ModelCallable, OutputQuery, resolve_excitation};
use sim_lib_femm_geometry::{BlockLabel2, Geometry2, Node2, Segment2, dummy_origin};
use sim_lib_femm_material::{Boundary, BoundaryKind, Material, MeshPolicy};
use sim_lib_femm_post::{QuantitySpec, quantity};
use sim_lib_femm_solve::{GradientTrust, solve_steady};
use sim_lib_numbers_numeric::{NumericNumbersLib, global_numeric_registry, numeric_diff_symbol};

use crate::{SensitivityPath, adjoint_gradient, gradient, register_femm_adjoint, total_gradient};

fn num(text: &str) -> Expr {
    sim_value::build::num_q(Some("numbers"), "f64", text)
}

fn model() -> ModelCallable {
    ModelCallable {
        model: sim_lib_femm_mesh::FemmModel {
            id: StableId(8),
            name: sim_kernel::Symbol::new("grad"),
            physics: PhysicsKind::Electrostatic,
            formulation: Formulation::Planar,
            length_unit: LengthUnit::Meter,
            depth: None,
            frequency_hz: None,
            inputs: vec![ParamSpec {
                name: sim_kernel::Symbol::new("gap"),
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
                kind: sim_kernel::Symbol::new("det"),
                max_area: None,
                min_angle_deg: None,
            },
            solve_policy: None,
            origin: dummy_origin(),
        },
    }
}

fn boundary_model() -> ModelCallable {
    let mut callable = model();
    callable.model.geometry = Geometry2 {
        nodes: vec![
            Node2 {
                xy: [num("0.0"), num("0.0")],
            },
            Node2 {
                xy: [num("1.0"), num("0.0")],
            },
            Node2 {
                xy: [num("1.0"), num("1.0")],
            },
            Node2 {
                xy: [num("0.0"), num("1.0")],
            },
        ],
        segments: vec![
            Segment2 {
                a: 0,
                b: 1,
                boundary: Some(sim_kernel::Symbol::new("wall")),
            },
            Segment2 {
                a: 1,
                b: 2,
                boundary: Some(sim_kernel::Symbol::new("wall")),
            },
            Segment2 {
                a: 2,
                b: 3,
                boundary: Some(sim_kernel::Symbol::new("wall")),
            },
            Segment2 {
                a: 3,
                b: 0,
                boundary: Some(sim_kernel::Symbol::new("wall")),
            },
        ],
        labels: vec![BlockLabel2 {
            name: sim_kernel::Symbol::new("air"),
            at: [num("0.5"), num("0.5")],
            material: sim_kernel::Symbol::new("air"),
        }],
        analytic: Vec::new(),
        arcs: Vec::new(),
    };
    callable.model.materials = vec![Material {
        name: sim_kernel::Symbol::new("air"),
        mu_r: Some(num("1.0")),
        nu_of_b2: None,
        epsilon_r: Some(num("1.0")),
        sigma: Some(num("1.0")),
        thermal_k: Some(num("1.0")),
        heat_source: None,
        remanence: None,
    }];
    callable.model.boundaries = vec![Boundary {
        name: sim_kernel::Symbol::new("wall"),
        kind: BoundaryKind::Dirichlet,
        value: Expr::Symbol(sim_kernel::Symbol::new("gap")),
    }];
    callable
}

fn params(cx: &mut Cx) -> sim_lib_femm_core::ParamSet {
    sim_lib_femm_core::ParamSet::new(vec![(
        sim_kernel::Symbol::new("gap"),
        cx.factory()
            .number_literal(
                sim_kernel::Symbol::qualified("numbers", "f64"),
                "0.5".to_owned(),
            )
            .unwrap(),
    )])
}

fn width_height_params(cx: &mut Cx) -> sim_lib_femm_core::ParamSet {
    sim_lib_femm_core::ParamSet::new(vec![
        (
            sim_kernel::Symbol::new("width"),
            cx.factory()
                .number_literal(
                    sim_kernel::Symbol::qualified("numbers", "f64"),
                    "1.0".to_owned(),
                )
                .unwrap(),
        ),
        (
            sim_kernel::Symbol::new("height"),
            cx.factory()
                .number_literal(
                    sim_kernel::Symbol::qualified("numbers", "f64"),
                    "1.0".to_owned(),
                )
                .unwrap(),
        ),
    ])
}

fn gap_mm_params(cx: &mut Cx, value: &str) -> ParamSet {
    ParamSet::new(vec![(
        sim_kernel::Symbol::new("gap-mm"),
        cx.factory()
            .number_literal(
                sim_kernel::Symbol::qualified("numbers", "f64"),
                value.to_owned(),
            )
            .unwrap(),
    )])
}

fn custom_query() -> OutputQuery {
    OutputQuery::Quantity(QuantitySpec::Custom {
        name: sim_kernel::Symbol::new("q"),
        expr: Expr::Call {
            operator: Box::new(Expr::Symbol(sim_kernel::Symbol::new("*"))),
            args: vec![num("2.0"), Expr::Symbol(sim_kernel::Symbol::new("gap"))],
        },
    })
}

fn parametric_box_model() -> ModelCallable {
    let mut callable = model();
    callable.model.inputs = vec![
        ParamSpec {
            name: sim_kernel::Symbol::new("width"),
            default: None,
            unit: None,
            role: ParamRole::Geometry,
        },
        ParamSpec {
            name: sim_kernel::Symbol::new("height"),
            default: None,
            unit: None,
            role: ParamRole::Geometry,
        },
    ];
    callable.model.geometry = Geometry2 {
        nodes: vec![
            Node2 {
                xy: [num("0.0"), num("0.0")],
            },
            Node2 {
                xy: [Expr::Symbol(sim_kernel::Symbol::new("width")), num("0.0")],
            },
            Node2 {
                xy: [
                    Expr::Symbol(sim_kernel::Symbol::new("width")),
                    Expr::Symbol(sim_kernel::Symbol::new("height")),
                ],
            },
            Node2 {
                xy: [num("0.0"), Expr::Symbol(sim_kernel::Symbol::new("height"))],
            },
        ],
        segments: vec![
            Segment2 {
                a: 0,
                b: 1,
                boundary: Some(sim_kernel::Symbol::new("wall")),
            },
            Segment2 {
                a: 1,
                b: 2,
                boundary: Some(sim_kernel::Symbol::new("wall")),
            },
            Segment2 {
                a: 2,
                b: 3,
                boundary: Some(sim_kernel::Symbol::new("wall")),
            },
            Segment2 {
                a: 3,
                b: 0,
                boundary: Some(sim_kernel::Symbol::new("wall")),
            },
        ],
        labels: vec![BlockLabel2 {
            name: sim_kernel::Symbol::new("air"),
            at: [num("0.5"), num("0.5")],
            material: sim_kernel::Symbol::new("air"),
        }],
        analytic: Vec::new(),
        arcs: Vec::new(),
    };
    callable.model.materials = vec![Material {
        name: sim_kernel::Symbol::new("air"),
        mu_r: Some(num("1.0")),
        nu_of_b2: None,
        epsilon_r: Some(num("1.0")),
        sigma: Some(num("1.0")),
        thermal_k: Some(num("1.0")),
        heat_source: None,
        remanence: None,
    }];
    callable.model.boundaries = vec![Boundary {
        name: sim_kernel::Symbol::new("wall"),
        kind: BoundaryKind::Dirichlet,
        value: num("1.0"),
    }];
    callable
}

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
    let fd = scalar_fd_quantity_gradient(&mut cx, &callable, &params, &spec, &wrt[0]).unwrap();
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

fn scalar_fd_quantity_gradient(
    cx: &mut Cx,
    callable: &ModelCallable,
    params: &ParamSet,
    quantity_spec: &QuantitySpec,
    symbol: &sim_kernel::Symbol,
) -> sim_lib_femm_core::FemmResult<f64> {
    let base = sim_lib_femm_core::value_as_f64(cx, params.get(symbol).unwrap())?;
    let step = 1.490_116_119_384_765_6e-8 * base.abs().max(1.0);
    let plus = gap_mm_params(cx, &(base + step).to_string());
    let minus = gap_mm_params(cx, &(base - step).to_string());
    let solved_plus = solve_steady(cx, &callable.model, &plus, &FemmLimits::default(), None)?;
    let exc_plus = resolve_excitation(cx, &callable.model, &plus, quantity_spec)?;
    let q_plus = quantity(&solved_plus.solution, quantity_spec, &exc_plus)?;
    let solved_minus = solve_steady(cx, &callable.model, &minus, &FemmLimits::default(), None)?;
    let exc_minus = resolve_excitation(cx, &callable.model, &minus, quantity_spec)?;
    let q_minus = quantity(&solved_minus.solution, quantity_spec, &exc_minus)?;
    Ok((q_plus - q_minus) / (2.0 * step))
}

#[test]
fn numeric_diff_with_femm_adjoint_uses_plugin_payload() {
    register_femm_adjoint().unwrap();
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&NumericNumbersLib::new()).unwrap();
    let func = sim_lib_femm_function::femm_as_func(
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
fn adjoint_plugin_registers() {
    register_femm_adjoint().unwrap();
    let guard = global_numeric_registry().read().unwrap();
    assert!(
        guard
            .differentiator(&sim_kernel::Symbol::new("femm-adjoint"))
            .is_some()
    );
}
