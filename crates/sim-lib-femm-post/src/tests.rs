use std::sync::Arc;

use sim_kernel::{Cx, DefaultFactory, EagerPolicy, ObjectEncode, Symbol};
use sim_lib_femm_core::{FemmError, FemmLimits, Formulation, ParamSet, PhysicsKind, StableId};
use sim_lib_femm_flow::SolveDiagnostics;
use sim_lib_femm_mesh::FemMesh2;

use crate::{
    Excitation, FemmSolution, QuantitySpec, energy, quantity, sample_gradient, sample_grid,
    sample_potential,
};

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

fn malformed_solution() -> FemmSolution {
    let mut solution = solution();
    solution.mesh.tri = vec![[0, 1, 3]];
    solution.mesh.elem_region = vec![Symbol::new("air")];
    solution
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
fn malformed_solution_consumer_paths_error_without_panic() {
    let malformed = malformed_solution();
    assert!(matches!(
        sample_potential(&malformed, 0.25, 0.25).unwrap_err(),
        FemmError::InvalidGeometry(_)
    ));
    assert!(matches!(
        sample_gradient(&malformed, [0, 1, 3]).unwrap_err(),
        FemmError::InvalidGeometry(_)
    ));
    assert!(matches!(
        energy(&malformed).unwrap_err(),
        FemmError::InvalidGeometry(_)
    ));
    assert!(matches!(
        sample_grid(&malformed, &[0.0], &[0.0], &FemmLimits::default()).unwrap_err(),
        FemmError::InvalidGeometry(_)
    ));
    assert!(matches!(
        quantity(
            &malformed,
            &QuantitySpec::Energy { region: None },
            &Excitation::none(),
        )
        .unwrap_err(),
        FemmError::InvalidGeometry(_)
    ));
}

#[test]
fn solution_validate_rejects_value_count_mismatch() {
    let mut malformed = solution();
    malformed.u.pop();
    assert!(matches!(
        malformed.validate().unwrap_err(),
        FemmError::InvalidGeometry(_)
    ));
}

#[test]
fn solution_read_construct_encoding_validates_public_value() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let malformed = malformed_solution();
    assert!(malformed.object_encoding(&mut cx).is_err());
}

#[test]
fn missing_region_is_named_in_errors() {
    let err = quantity(
        &solution(),
        &QuantitySpec::ForceY {
            region: Symbol::new("missing"),
        },
        &Excitation::none(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("missing"));
}

#[test]
fn scalar_quantities_cover_regions_circuits_and_losses() {
    let solution = solution();
    let none = Excitation::none();
    assert!(quantity(&solution, &QuantitySpec::Energy { region: None }, &none).unwrap() > 0.0);
    assert!(
        quantity(
            &solution,
            &QuantitySpec::Energy {
                region: Some(Symbol::new("air"))
            },
            &none,
        )
        .unwrap()
            > 0.0
    );
    assert!(
        quantity(
            &solution,
            &QuantitySpec::ForceY {
                region: Symbol::new("air")
            },
            &none,
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
            },
            &none,
        )
        .unwrap()
            < 0.0
    );
    assert!(
        quantity(
            &solution,
            &QuantitySpec::FluxLinkage {
                circuit: Symbol::new("coil")
            },
            &Excitation::with_current(2.0),
        )
        .unwrap()
            > 0.0
    );
    assert!(
        quantity(
            &solution,
            &QuantitySpec::Inductance {
                circuit: Symbol::new("coil")
            },
            &Excitation::with_current(2.0),
        )
        .unwrap()
            > 0.0
    );
    assert!(
        quantity(
            &solution,
            &QuantitySpec::Capacitance {
                conductor: Symbol::new("plate")
            },
            &Excitation::with_potential(2.0),
        )
        .unwrap()
            > 0.0
    );
    assert!(quantity(&solution, &QuantitySpec::JouleLoss { region: None }, &none).unwrap() > 0.0);
}

#[test]
fn inductance_scales_with_current_squared() {
    let solution = solution();
    let w = energy(&solution).unwrap();
    let l = quantity(
        &solution,
        &QuantitySpec::Inductance {
            circuit: Symbol::new("coil"),
        },
        &Excitation::with_current(2.0),
    )
    .unwrap();
    // L = 2W / I^2 with I = 2, i.e. 2W/4 -- NOT the old 2W.
    assert!((l - 2.0 * w / 4.0).abs() < 1.0e-12);
    assert!((l - 2.0 * w).abs() > 1.0e-9);
}

#[test]
fn flux_linkage_is_two_energy_over_current() {
    let solution = solution();
    let w = energy(&solution).unwrap();
    let lambda = quantity(
        &solution,
        &QuantitySpec::FluxLinkage {
            circuit: Symbol::new("coil"),
        },
        &Excitation::with_current(3.0),
    )
    .unwrap();
    assert!((lambda - 2.0 * w / 3.0).abs() < 1.0e-12);
}

#[test]
fn missing_or_zero_excitation_is_rejected() {
    let solution = solution();
    let missing = quantity(
        &solution,
        &QuantitySpec::Inductance {
            circuit: Symbol::new("coil"),
        },
        &Excitation::none(),
    )
    .unwrap_err();
    assert!(missing.to_string().contains("current"));

    let zero = quantity(
        &solution,
        &QuantitySpec::Capacitance {
            conductor: Symbol::new("plate"),
        },
        &Excitation::with_potential(0.0),
    )
    .unwrap_err();
    assert!(zero.to_string().contains("potential"));
}

#[test]
fn custom_quantity_errors_rather_than_returning_zero() {
    let err = quantity(
        &solution(),
        &QuantitySpec::Custom {
            name: Symbol::new("mine"),
            expr: sim_kernel::Expr::Symbol(Symbol::new("mine")),
        },
        &Excitation::none(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("custom quantity"));
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
