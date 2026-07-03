use sim_kernel::Symbol;
use sim_lib_femm_core::{FemmError, FemmLimits, Formulation, ParamSet, PhysicsKind, StableId};
use sim_lib_femm_flow::SolveDiagnostics;
use sim_lib_femm_mesh::FemMesh2;

use crate::{FemmSolution, QuantitySpec, energy, quantity, sample_grid, sample_potential};

fn solution() -> FemmSolution {
    FemmSolution {
        id: StableId(10),
        model_id: StableId(1),
        physics: PhysicsKind::Electrostatic,
        formulation: Formulation::Planar,
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        },
        u: vec![1.0, 3.0, 4.0],
        diagnostics: SolveDiagnostics {
            method: Symbol::new("femm-ptc"),
            converged: true,
            iterations: 1,
            final_residual: 0.0,
            events: Vec::new(),
            diagnostics: Vec::new(),
        },
    }
}

#[test]
fn linear_interpolation_is_exact() {
    let out = sample_potential(&solution(), 0.25, 0.25).unwrap();
    assert!((out - 2.25).abs() < 1.0e-12);
}

#[test]
fn energy_is_nonnegative() {
    assert!(energy(&solution()).unwrap() >= 0.0);
}

#[test]
fn grid_sampling_honors_cap() {
    let err = sample_grid(
        &solution(),
        &[0.0, 0.5],
        &[0.0, 0.5],
        &FemmLimits {
            max_output_samples: 3,
            ..FemmLimits::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, FemmError::BudgetExceeded(_)));
}

#[test]
fn missing_region_is_named_in_errors() {
    let err = quantity(
        &solution(),
        &QuantitySpec::ForceY {
            region: Symbol::new("missing"),
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("missing"));
}

#[test]
fn scalar_quantities_cover_regions_circuits_and_losses() {
    let solution = solution();
    assert!(quantity(&solution, &QuantitySpec::Energy { region: None }).unwrap() > 0.0);
    assert!(
        quantity(
            &solution,
            &QuantitySpec::Energy {
                region: Some(Symbol::new("air"))
            }
        )
        .unwrap()
            > 0.0
    );
    assert!(
        quantity(
            &solution,
            &QuantitySpec::ForceY {
                region: Symbol::new("air")
            }
        )
        .unwrap()
            < 0.0
    );
    assert!(
        quantity(
            &solution,
            &QuantitySpec::Torque {
                region: Symbol::new("air"),
                center: [0.0, 0.0],
            }
        )
        .unwrap()
            < 0.0
    );
    assert!(
        quantity(
            &solution,
            &QuantitySpec::FluxLinkage {
                circuit: Symbol::new("coil")
            }
        )
        .unwrap()
            > 0.0
    );
    assert!(
        quantity(
            &solution,
            &QuantitySpec::Inductance {
                circuit: Symbol::new("coil")
            }
        )
        .unwrap()
            > 0.0
    );
    assert!(
        quantity(
            &solution,
            &QuantitySpec::Capacitance {
                conductor: Symbol::new("plate")
            }
        )
        .unwrap()
            > 0.0
    );
    assert!(quantity(&solution, &QuantitySpec::JouleLoss { region: None }).unwrap() > 0.0);
}

#[test]
fn axisymmetric_energy_uses_radial_measure() {
    let mut axisym = solution();
    axisym.formulation = Formulation::Axisymmetric;
    for point in &mut axisym.mesh.xy {
        point[0] += 1.0;
    }
    let planar = energy(&solution()).unwrap();
    let weighted = energy(&axisym).unwrap();
    assert!(weighted > planar);
}
