//! Deterministic FEMM model fixtures for codec, solve, ODE, and regression tests.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: a fixed catalog of canonical 2D
//! finite-element problems (one per physics surface: electrostatic, thermal,
//! conductive, and magnetostatic), each with an analytically known reference so
//! that other FEMM crates can solve a deterministic model and regression-check
//! the result. It is a runtime fixture catalog, not an illustrative examples
//! package. See the [crate README](index.html).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use sim_kernel::{Factory, Symbol};
use sim_lib_femm_core::{Formulation, LengthUnit, ParamRole, ParamSpec, PhysicsKind, StableId};
use sim_lib_femm_geometry::{AnalyticRegion2, BlockLabel2, Geometry2, dummy_origin};
use sim_lib_femm_material::{Material, MeshPolicy};
use sim_lib_femm_mesh::FemmModel;

mod support;

use support::{air, num};

fn base_model(id: u64, name: &str, physics: PhysicsKind, material: Material) -> FemmModel {
    FemmModel {
        id: StableId(id),
        name: Symbol::new(name),
        physics,
        formulation: Formulation::Planar,
        length_unit: LengthUnit::Millimeter,
        depth: None,
        frequency_hz: None,
        inputs: vec![ParamSpec {
            name: Symbol::new("gap-mm"),
            default: Some(
                sim_kernel::DefaultFactory
                    .number_literal(Symbol::qualified("numbers", "f64"), "1.0".to_owned())
                    .unwrap(),
            ),
            unit: None,
            role: ParamRole::Geometry,
        }],
        geometry: Geometry2 {
            labels: vec![BlockLabel2 {
                name: material.name.clone(),
                at: [num("0.5"), num("0.5")],
                material: material.name.clone(),
            }],
            analytic: vec![AnalyticRegion2::Rect {
                name: material.name.clone(),
                xy: [num("0.0"), num("0.0")],
                wh: [num("1.0"), num("1.0")],
            }],
            ..Geometry2::default()
        },
        materials: vec![material],
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
    }
}

/// Canonical electrostatic fixture: a parallel-plate capacitor.
///
/// A unit air-filled square whose plate gap is exposed as the `gap-mm` input,
/// giving a closed-form capacitance to check field and post-processing output
/// against.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_fixtures::parallel_plate_capacitor;
///
/// let model = parallel_plate_capacitor();
/// assert_eq!(model.name.as_qualified_str(), "parallel-plate-capacitor");
/// assert_eq!(model.inputs.len(), 1);
/// ```
pub fn parallel_plate_capacitor() -> FemmModel {
    base_model(
        100,
        "parallel-plate-capacitor",
        PhysicsKind::Electrostatic,
        air(),
    )
}

/// Canonical thermal fixture: steady heat conduction through a slab.
///
/// A unit square solved as a `HeatSteady` problem, whose linear temperature
/// profile across the slab is the analytic reference.
pub fn slab_heat_conductor() -> FemmModel {
    base_model(101, "slab-heat-conductor", PhysicsKind::HeatSteady, air())
}

/// Canonical conductive fixture: DC resistance of a uniform conductor.
///
/// A unit square of finite conductivity solved as a `CurrentSteady` problem;
/// the closed-form resistance checks conductive assembly and post-processing.
pub fn uniform_conductor_resistance() -> FemmModel {
    base_model(
        102,
        "uniform-conductor-resistance",
        PhysicsKind::CurrentSteady,
        Material {
            sigma: Some(num("5.8e7")),
            ..air()
        },
    )
}

/// Canonical magnetostatic fixture: an air-core solenoid.
///
/// A linear `Magnetostatic` problem with no magnetic material, giving the
/// textbook air-core inductance as the reference.
pub fn air_core_solenoid() -> FemmModel {
    base_model(103, "air-core-solenoid", PhysicsKind::Magnetostatic, air())
}

/// Nonlinear magnetostatic fixture: a gapped E-I core inductor.
///
/// A high-permeability steel region with field-dependent reluctivity and an
/// air gap, exercising the nonlinear magnetostatic solve path.
pub fn gapped_ei_core_inductor() -> FemmModel {
    base_model(
        104,
        "gapped-ei-core-inductor",
        PhysicsKind::Magnetostatic,
        Material {
            name: Symbol::new("steel"),
            mu_r: Some(num("4000.0")),
            nu_of_b2: Some(num("0.02")),
            epsilon_r: None,
            sigma: Some(num("2.0e6")),
            thermal_k: Some(num("40.0")),
            heat_source: None,
            remanence: None,
        },
    )
}

/// Transient fixture: a plunger actuator driven through the ODE integrator.
///
/// A `Magnetostatic` model whose moving-gap input feeds the ODE time-stepping
/// path, used to regression-check coupled field/ODE evolution.
pub fn plunger_actuator_ode() -> FemmModel {
    base_model(
        105,
        "plunger-actuator-ode",
        PhysicsKind::Magnetostatic,
        air(),
    )
}

/// Post-processing fixture: a field reduced to a number by line integration.
///
/// A `Magnetostatic` model used to check that a solved field can be integrated
/// along a path into a single scalar output.
pub fn field_as_number_line_integration() -> FemmModel {
    base_model(
        106,
        "field-as-number-line-integration",
        PhysicsKind::Magnetostatic,
        air(),
    )
}
/// Returns the full catalog of fixture models, one per canonical problem.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_fixtures::fixture_models;
///
/// let models = fixture_models();
/// assert_eq!(models.len(), 7);
/// ```
pub fn fixture_models() -> Vec<FemmModel> {
    vec![
        parallel_plate_capacitor(),
        slab_heat_conductor(),
        uniform_conductor_resistance(),
        air_core_solenoid(),
        gapped_ei_core_inductor(),
        plunger_actuator_ode(),
        field_as_number_line_integration(),
    ]
}

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
