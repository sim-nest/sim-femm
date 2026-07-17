use std::sync::Arc;

use sim_kernel::{DefaultFactory, EagerPolicy};
use sim_kernel::{Expr, Symbol};
use sim_lib_femm_core::{Formulation, LengthUnit, ParamRole, ParamSpec, PhysicsKind, StableId};
use sim_lib_femm_geometry::{BlockLabel2, Geometry2, dummy_origin};
use sim_lib_femm_material::{Boundary, BoundaryKind, Material, MeshPolicy};
use sim_lib_femm_mesh::{FemMesh2, FemmModel, MeshedModel};

use super::*;

fn test_cx() -> Cx {
    Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory))
}

struct PoissonFront;

impl PhysicsFront for PoissonFront {
    fn kind(&self) -> sim_lib_femm_core::PhysicsKind {
        PhysicsKind::Electrostatic
    }

    fn element_residual<S: Scalarish>(
        &self,
        elem: &ElementGeom,
        u_e: [S; 3],
        _coeff: &CoeffEval,
    ) -> [S; 3] {
        let grad_u = [
            elem.grad
                .iter()
                .zip(u_e)
                .map(|(grad, u)| S::from_f64(grad[0]) * u)
                .fold(S::from_f64(0.0), |acc, x| acc + x),
            elem.grad
                .iter()
                .zip(u_e)
                .map(|(grad, u)| S::from_f64(grad[1]) * u)
                .fold(S::from_f64(0.0), |acc, x| acc + x),
        ];
        std::array::from_fn(|i| {
            let dot =
                grad_u[0] * S::from_f64(elem.grad[i][0]) + grad_u[1] * S::from_f64(elem.grad[i][1]);
            dot * S::from_f64(elem.area)
        })
    }
}

fn num(text: &str) -> Expr {
    sim_value::build::num_q(Some("numbers"), "f64", text)
}

fn call(operator: &str, args: Vec<Expr>) -> Expr {
    Expr::Call {
        operator: Box::new(Expr::Symbol(Symbol::new(operator))),
        args,
    }
}

fn param(cx: &mut Cx, symbol: &str, canonical: &str) -> ParamSet {
    ParamSet::new(vec![(
        Symbol::new(symbol),
        cx.factory()
            .number_literal(Symbol::qualified("numbers", "f64"), canonical.to_owned())
            .unwrap(),
    )])
}

fn model() -> FemmModel {
    FemmModel {
        id: StableId(1),
        name: Symbol::new("poisson"),
        physics: PhysicsKind::Electrostatic,
        formulation: Formulation::Planar,
        length_unit: LengthUnit::Meter,
        depth: None,
        frequency_hz: None,
        inputs: vec![ParamSpec {
            name: Symbol::new("x"),
            default: None,
            unit: None,
            role: ParamRole::Design,
        }],
        geometry: Geometry2 {
            labels: vec![BlockLabel2 {
                name: Symbol::new("air"),
                at: [num("0.1"), num("0.1")],
                material: Symbol::new("air"),
            }],
            ..Geometry2::default()
        },
        materials: vec![Material {
            name: Symbol::new("air"),
            mu_r: Some(num("1.0")),
            nu_of_b2: None,
            epsilon_r: Some(num("1.0")),
            sigma: None,
            thermal_k: Some(num("1.0")),
            heat_source: None,
            remanence: None,
        }],
        boundaries: vec![Boundary {
            name: Symbol::new("wall"),
            kind: BoundaryKind::Dirichlet,
            value: num("0.0"),
        }],
        sources: Vec::new(),
        outputs: Vec::new(),
        mesh_policy: MeshPolicy {
            kind: Symbol::new("det"),
            max_area: None,
            min_angle_deg: None,
        },
        solve_policy: None,
        origin: dummy_origin(),
    }
}

#[test]
fn one_triangle_matrix_is_symmetric() {
    let mut cx = test_cx();
    let meshed = MeshedModel {
        model_id: StableId(1),
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        },
        diagnostics: Vec::new(),
    };
    let assembled = assemble_system(
        &mut cx,
        &PoissonFront,
        &model(),
        &meshed,
        &FemmLimits::default(),
    )
    .unwrap();
    let dense = assembled.k.to_dense().unwrap();
    assert!((dense[0][1] - dense[1][0]).abs() < 1.0e-12);
}

#[test]
fn dirichlet_elimination_pins_dof() {
    let mut cx = test_cx();
    let meshed = MeshedModel {
        model_id: StableId(1),
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: vec![(0, 1, Symbol::new("wall"))],
        },
        diagnostics: Vec::new(),
    };
    let assembled = assemble_system(
        &mut cx,
        &PoissonFront,
        &model(),
        &meshed,
        &FemmLimits::default(),
    )
    .unwrap();
    let dense = assembled.k.to_dense().unwrap();
    assert_eq!(dense[0][0], 1.0);
    assert_eq!(dense[0][1], 0.0);
}

struct LinearOnlyFront;

impl PhysicsFront for LinearOnlyFront {
    fn kind(&self) -> sim_lib_femm_core::PhysicsKind {
        PhysicsKind::Magnetostatic
    }

    fn element_residual<S: Scalarish>(
        &self,
        _elem: &ElementGeom,
        _u_e: [S; 3],
        _coeff: &CoeffEval,
    ) -> [S; 3] {
        std::array::from_fn(|_| S::from_f64(0.0))
    }

    fn validate_coeff(&self, coeff: &CoeffEval) -> FemmResult<()> {
        if coeff.nonlinear_bh {
            return Err(FemmError::UnsupportedPhysics(
                "nonlinear B-H not supported".to_owned(),
            ));
        }
        Ok(())
    }
}

#[test]
fn coeff_eval_flags_nonlinear_bh_and_front_fails_closed() {
    // A material carrying `nu_of_b2` must surface as `nonlinear_bh` so a
    // linear-only front can reject it instead of silently solving linearly.
    let mut cx = test_cx();
    let mut model = model();
    model.materials[0].nu_of_b2 = Some(num("0.02"));
    let meshed = MeshedModel {
        model_id: StableId(1),
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        },
        diagnostics: Vec::new(),
    };
    let result = assemble_system(
        &mut cx,
        &LinearOnlyFront,
        &model,
        &meshed,
        &FemmLimits::default(),
    );
    assert!(matches!(result, Err(FemmError::UnsupportedPhysics(_))));
}

#[test]
fn linear_material_still_assembles() {
    let mut cx = test_cx();
    let meshed = MeshedModel {
        model_id: StableId(1),
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        },
        diagnostics: Vec::new(),
    };
    assert!(
        assemble_system(
            &mut cx,
            &LinearOnlyFront,
            &model(),
            &meshed,
            &FemmLimits::default(),
        )
        .is_ok()
    );
}

#[test]
fn out_of_range_boundary_node_errors_without_panic() {
    // A boundary edge naming a node index past the mesh must fail closed
    // rather than panic on the dense-matrix scatter.
    let mut cx = test_cx();
    let meshed = MeshedModel {
        model_id: StableId(1),
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: vec![(0, 99, Symbol::new("wall"))],
        },
        diagnostics: Vec::new(),
    };
    let result = assemble_system(
        &mut cx,
        &PoissonFront,
        &model(),
        &meshed,
        &FemmLimits::default(),
    );
    assert!(matches!(result, Err(FemmError::InvalidGeometry(_))));
}

#[test]
fn invalid_boundary_number_errors_instead_of_grounding_zero() {
    let mut cx = test_cx();
    let mut bad_model = model();
    bad_model.boundaries[0].value = num("1/0");
    let meshed = MeshedModel {
        model_id: StableId(1),
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: vec![(0, 1, Symbol::new("wall"))],
        },
        diagnostics: Vec::new(),
    };
    let result = assemble_system(
        &mut cx,
        &PoissonFront,
        &bad_model,
        &meshed,
        &FemmLimits::default(),
    );
    assert!(matches!(result, Err(FemmError::InvalidGeometry(_))));
}

#[test]
fn boundary_division_by_zero_errors_instead_of_grounding_zero() {
    let mut cx = test_cx();
    let mut bad_model = model();
    bad_model.boundaries[0].value = call("/", vec![num("1.0"), num("0.0")]);
    let meshed = MeshedModel {
        model_id: StableId(1),
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: vec![(0, 1, Symbol::new("wall"))],
        },
        diagnostics: Vec::new(),
    };
    let result = assemble_system(
        &mut cx,
        &PoissonFront,
        &bad_model,
        &meshed,
        &FemmLimits::default(),
    );
    assert!(matches!(result, Err(FemmError::InvalidGeometry(_))));
}

#[test]
fn unknown_boundary_parameter_errors_instead_of_grounding_zero() {
    let mut cx = test_cx();
    let mut bad_model = model();
    bad_model.boundaries[0].value = Expr::Symbol(Symbol::new("missing"));
    let meshed = MeshedModel {
        model_id: StableId(1),
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: vec![(0, 1, Symbol::new("wall"))],
        },
        diagnostics: Vec::new(),
    };
    let result = assemble_system(
        &mut cx,
        &PoissonFront,
        &bad_model,
        &meshed,
        &FemmLimits::default(),
    );
    assert!(matches!(result, Err(FemmError::UnknownFemmParameter(_))));
}

#[test]
fn non_finite_boundary_parameter_errors_instead_of_grounding_zero() {
    let mut cx = test_cx();
    let mut bad_model = model();
    bad_model.boundaries[0].value = Expr::Symbol(Symbol::new("x"));
    let meshed = MeshedModel {
        model_id: StableId(1),
        params: param(&mut cx, "x", "inf"),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: vec![(0, 1, Symbol::new("wall"))],
        },
        diagnostics: Vec::new(),
    };
    let result = assemble_system(
        &mut cx,
        &PoissonFront,
        &bad_model,
        &meshed,
        &FemmLimits::default(),
    );
    assert!(matches!(result, Err(FemmError::InvalidGeometry(_))));
}

#[test]
fn axisymmetric_assembly_uses_radial_weight() {
    let mut cx = test_cx();
    let mut axisym = model();
    axisym.formulation = Formulation::Axisymmetric;
    let meshed = MeshedModel {
        model_id: StableId(1),
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        },
        diagnostics: Vec::new(),
    };
    let planar = assemble_system(
        &mut cx,
        &PoissonFront,
        &model(),
        &meshed,
        &FemmLimits::default(),
    )
    .unwrap()
    .k
    .to_dense()
    .unwrap();
    let weighted = assemble_system(
        &mut cx,
        &PoissonFront,
        &axisym,
        &meshed,
        &FemmLimits::default(),
    )
    .unwrap()
    .k
    .to_dense()
    .unwrap();
    let ratio = weighted[0][0] / planar[0][0];
    let expected = 2.0 * std::f64::consts::PI * (4.0 / 3.0);
    assert!((ratio - expected).abs() < 1.0e-12);
}
