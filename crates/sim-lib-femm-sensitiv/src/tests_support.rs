use sim_kernel::{Cx, Expr};
use sim_lib_femm_core::{
    FemmLimits, Formulation, LengthUnit, ParamRole, ParamSet, ParamSpec, PhysicsKind, StableId,
};
use sim_lib_femm_geometry::{BlockLabel2, Geometry2, Node2, Segment2, dummy_origin};
use sim_lib_femm_material::{Boundary, BoundaryKind, Material, MeshPolicy, Source};
use sim_lib_femm_post::{QuantitySpec, quantity};
use sim_lib_femm_query::{ModelCallable, OutputQuery, resolve_excitation};
use sim_lib_femm_solve::solve_steady;

pub(super) fn num(text: &str) -> Expr {
    sim_value::build::num_q(Some("numbers"), "f64", text)
}

pub(super) fn call(operator: &str, args: Vec<Expr>) -> Expr {
    Expr::Call {
        operator: Box::new(Expr::Symbol(sim_kernel::Symbol::new(operator))),
        args,
    }
}

pub(super) fn model() -> ModelCallable {
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

pub(super) fn boundary_model() -> ModelCallable {
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

pub(super) fn params(cx: &mut Cx) -> ParamSet {
    gap_params(cx, "0.5")
}

pub(super) fn gap_params(cx: &mut Cx, value: &str) -> ParamSet {
    ParamSet::new(vec![(
        sim_kernel::Symbol::new("gap"),
        cx.factory()
            .number_literal(
                sim_kernel::Symbol::qualified("numbers", "f64"),
                value.to_owned(),
            )
            .unwrap(),
    )])
}

pub(super) fn width_height_params(cx: &mut Cx) -> ParamSet {
    ParamSet::new(vec![
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

pub(super) fn gap_mm_params(cx: &mut Cx, value: &str) -> ParamSet {
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

pub(super) fn custom_query() -> OutputQuery {
    OutputQuery::Quantity(QuantitySpec::Custom {
        name: sim_kernel::Symbol::new("q"),
        expr: Expr::Call {
            operator: Box::new(Expr::Symbol(sim_kernel::Symbol::new("*"))),
            args: vec![num("2.0"), Expr::Symbol(sim_kernel::Symbol::new("gap"))],
        },
    })
}

pub(super) fn custom_query_with_default_offset() -> OutputQuery {
    OutputQuery::Quantity(QuantitySpec::Custom {
        name: sim_kernel::Symbol::new("q"),
        expr: call(
            "+",
            vec![
                call(
                    "*",
                    vec![num("2.0"), Expr::Symbol(sim_kernel::Symbol::new("gap"))],
                ),
                Expr::Symbol(sim_kernel::Symbol::new("offset")),
            ],
        ),
    })
}

pub(super) fn model_with_default_offset(cx: &mut Cx) -> ModelCallable {
    let mut callable = model();
    callable.model.inputs.push(ParamSpec {
        name: sim_kernel::Symbol::new("offset"),
        default: Some(
            cx.factory()
                .number_literal(
                    sim_kernel::Symbol::qualified("numbers", "f64"),
                    "1.5".to_owned(),
                )
                .unwrap(),
        ),
        unit: None,
        role: ParamRole::Design,
    });
    callable
}

pub(super) fn parametric_box_model() -> ModelCallable {
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
    // Applied potential 2.0 V, distinct from 1.0 so the capacitance derivative
    // (2/V^2) dW/dp is numerically distinguishable from a missing normalization.
    callable.model.boundaries = vec![Boundary {
        name: sim_kernel::Symbol::new("wall"),
        kind: BoundaryKind::Dirichlet,
        value: num("2.0"),
    }];
    // A parameter-independent coil so inductance/flux linkage are well-defined
    // (current fixed at 2.0 A); geometry alone carries the width/height design
    // parameters, keeping the drive independent of them.
    callable.model.sources = vec![Source::CircuitCoil {
        name: sim_kernel::Symbol::new("plate"),
        region: sim_kernel::Symbol::new("air"),
        turns: num("1.0"),
        current: num("2.0"),
    }];
    callable
}

pub(super) fn scalar_fd_quantity_gradient(
    cx: &mut Cx,
    callable: &ModelCallable,
    params: &ParamSet,
    quantity_spec: &QuantitySpec,
) -> sim_lib_femm_core::FemmResult<f64> {
    let symbol = sim_kernel::Symbol::new("gap-mm");
    let base = sim_lib_femm_core::value_as_f64(cx, params.get(&symbol).unwrap())?;
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

fn replace_param(
    cx: &mut Cx,
    params: &ParamSet,
    symbol: &sim_kernel::Symbol,
    value: f64,
) -> ParamSet {
    let mut entries = params.entries.clone();
    let replacement = cx
        .factory()
        .number_literal(
            sim_kernel::Symbol::qualified("numbers", "f64"),
            value.to_string(),
        )
        .unwrap();
    if let Some((_, current)) = entries.iter_mut().find(|(name, _)| name == symbol) {
        *current = replacement;
    } else {
        entries.push((symbol.clone(), replacement));
    }
    ParamSet::new(entries)
}

/// Central finite difference of the corrected forward [`quantity`] for `symbol`.
///
/// Re-resolves the excitation on each perturbed solve, so it is the independent
/// oracle the analytic derivative is checked against.
pub(super) fn central_fd_quantity_gradient(
    cx: &mut Cx,
    callable: &ModelCallable,
    params: &ParamSet,
    spec: &QuantitySpec,
    symbol: &sim_kernel::Symbol,
) -> sim_lib_femm_core::FemmResult<f64> {
    let base = sim_lib_femm_core::value_as_f64(cx, params.get(symbol).unwrap())?;
    let step = 1.490_116_119_384_765_6e-8 * base.abs().max(1.0);
    let plus = replace_param(cx, params, symbol, base + step);
    let minus = replace_param(cx, params, symbol, base - step);
    let solved_plus = solve_steady(cx, &callable.model, &plus, &FemmLimits::default(), None)?;
    let exc_plus = resolve_excitation(cx, &callable.model, &plus, spec)?;
    let q_plus = quantity(&solved_plus.solution, spec, &exc_plus)?;
    let solved_minus = solve_steady(cx, &callable.model, &minus, &FemmLimits::default(), None)?;
    let exc_minus = resolve_excitation(cx, &callable.model, &minus, spec)?;
    let q_minus = quantity(&solved_minus.solution, spec, &exc_minus)?;
    Ok((q_plus - q_minus) / (2.0 * step))
}
