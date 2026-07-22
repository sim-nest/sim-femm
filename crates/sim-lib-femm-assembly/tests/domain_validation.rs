use std::sync::Arc;

use sim_kernel::{Cx, DefaultFactory, EagerPolicy, Expr, Symbol};
use sim_lib_femm_assembly::{CoeffEval, PhysicsFront, assemble_system};
use sim_lib_femm_core::{
    FemmError, FemmLimits, Formulation, LengthUnit, ParamSet, PhysicsKind, StableId,
};
use sim_lib_femm_geometry::{Geometry2, dummy_origin};
use sim_lib_femm_material::{Boundary, BoundaryKind, Material, MeshPolicy};
use sim_lib_femm_mesh::{FemMesh2, FemmModel, MeshedModel};
use sim_lib_femm_space::ElementGeom;
use sim_lib_numbers_ad::Scalarish;

struct PoissonFront;

impl PhysicsFront for PoissonFront {
    fn kind(&self) -> PhysicsKind {
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

fn test_cx() -> Cx {
    Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory))
}

fn num(text: &str) -> Expr {
    sim_value::build::num_q(Some("numbers"), "f64", text)
}

fn material(name: &str, epsilon: &str) -> Material {
    Material {
        name: Symbol::new(name),
        mu_r: Some(num("1.0")),
        nu_of_b2: None,
        epsilon_r: Some(num(epsilon)),
        sigma: None,
        thermal_k: Some(num("1.0")),
        heat_source: None,
        remanence: None,
    }
}

fn model() -> FemmModel {
    FemmModel {
        id: StableId(1),
        name: Symbol::new("domain-validation"),
        physics: PhysicsKind::Electrostatic,
        formulation: Formulation::Planar,
        length_unit: LengthUnit::Meter,
        depth: None,
        frequency_hz: None,
        inputs: Vec::new(),
        geometry: Geometry2::default(),
        materials: vec![material("air", "1.0"), material("steel", "2.0")],
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

fn meshed(regions: Vec<Symbol>) -> MeshedModel {
    MeshedModel {
        model_id: StableId(1),
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2], [0, 2, 3]],
            elem_region: regions,
            edge_boundary: vec![(0, 1, Symbol::new("wall"))],
        },
        diagnostics: Vec::new(),
    }
}

#[test]
fn assembly_rejects_unlabeled_elements() {
    let mut cx = test_cx();
    let err = assemble_system(
        &mut cx,
        &PoissonFront,
        &model(),
        &meshed(vec![Symbol::new("air")]),
        &FemmLimits::default(),
    )
    .unwrap_err();
    assert!(matches!(err, FemmError::InvalidGeometry(_)));
}

#[test]
fn assembly_rejects_unknown_region_labels() {
    let mut cx = test_cx();
    let err = assemble_system(
        &mut cx,
        &PoissonFront,
        &model(),
        &meshed(vec![Symbol::new("air"), Symbol::new("missing")]),
        &FemmLimits::default(),
    )
    .unwrap_err();
    assert!(matches!(err, FemmError::MissingMaterial(_)));
}

#[test]
fn assembly_rejects_unsupported_boundary_kinds() {
    let mut cx = test_cx();
    let mut model = model();
    model.boundaries[0].kind = BoundaryKind::Neumann;
    let err = assemble_system(
        &mut cx,
        &PoissonFront,
        &model,
        &meshed(vec![Symbol::new("air"), Symbol::new("steel")]),
        &FemmLimits::default(),
    )
    .unwrap_err();
    assert!(matches!(err, FemmError::InvalidGeometry(_)));
}

#[test]
fn assembly_accepts_valid_multi_material_regions() {
    let mut cx = test_cx();
    let assembled = assemble_system(
        &mut cx,
        &PoissonFront,
        &model(),
        &meshed(vec![Symbol::new("air"), Symbol::new("steel")]),
        &FemmLimits::default(),
    )
    .unwrap();
    assert_eq!(assembled.k.rows(), 4);
}
