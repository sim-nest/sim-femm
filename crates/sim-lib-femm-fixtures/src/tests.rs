use sim_kernel::{ContentId, Symbol};
use std::sync::Arc;

use sim_kernel::{Cx, DefaultFactory, EagerPolicy};
use sim_lib_femm_core::{FemmError, FemmLimits, Formulation, ParamSet, PhysicsKind};
use sim_lib_femm_function::quality;
use sim_lib_femm_physics::{
    conductor_resistance, harmonic_joule_loss, long_solenoid_b, parallel_plate_capacitance,
    slab_heat_resistance,
};
use sim_lib_femm_post::{QuantitySpec, energy};
use sim_lib_femm_solve::{GradientTrust, SolveExportRecord, certificate_claim, solve_steady};

use crate::{fixture_models, parallel_plate_capacitor, slab_heat_conductor};

#[test]
fn fixtures_have_stable_names() {
    assert_eq!(
        parallel_plate_capacitor().name,
        Symbol::new("parallel-plate-capacitor")
    );
    assert_eq!(slab_heat_conductor().physics, PhysicsKind::HeatSteady);
    assert_eq!(
        fixture_models()
            .into_iter()
            .map(|model| model.name.to_string())
            .collect::<Vec<_>>(),
        vec![
            "parallel-plate-capacitor",
            "slab-heat-conductor",
            "uniform-conductor-resistance",
            "air-core-solenoid",
            "gapped-ei-core-inductor",
            "plunger-actuator-ode",
            "field-as-number-line-integration",
        ]
    );
}

#[test]
fn fixture_reference_outputs_match_expected_analytics() {
    assert!((parallel_plate_capacitance(8.854e-12, 0.02, 0.001) - 1.7708e-10).abs() < 1.0e-14);
    assert!((slab_heat_resistance(0.1, 200.0, 0.01) - 0.05).abs() < 1.0e-12);
    assert!((conductor_resistance(2.0, 5.8e7, 1.0e-6) - 0.0344827586).abs() < 1.0e-9);
    assert!(
        (long_solenoid_b(4.0e-7 * std::f64::consts::PI, 5000.0, 2.0) - 0.0125663706).abs()
            < 1.0e-10
    );
    assert!((harmonic_joule_loss(5.8e7, 60.0, 0.001) - 1740.0).abs() < 1.0e-9);
}

#[test]
fn every_fixture_solves_in_planar_and_axisymmetric_modes() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    for mut model in fixture_models() {
        // A material carrying a nonlinear B-H curve (`nu_of_b2`) is rejected by
        // the linear magnetic fronts: the solve fails closed rather than
        // silently returning the linear answer. Such fixtures are checked
        // separately by `nonlinear_bh_fixture_fails_closed`.
        let nonlinear = model
            .materials
            .iter()
            .any(|material| material.nu_of_b2.is_some());
        if nonlinear {
            continue;
        }
        for formulation in [Formulation::Planar, Formulation::Axisymmetric] {
            model.formulation = formulation;
            let solved = solve_steady(
                &mut cx,
                &model,
                &ParamSet::default(),
                &FemmLimits::default(),
                None,
            )
            .unwrap();
            assert_eq!(solved.solution.mesh.tri.len(), 2);
            assert_eq!(solved.solution.u.len(), solved.solution.mesh.xy.len());
            assert!(energy(&solved.solution).unwrap() >= 0.0);
        }
    }
}

#[test]
fn nonlinear_bh_ptc_solve_emits_certificate() {
    use crate::gapped_ei_core_inductor;
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let model = gapped_ei_core_inductor();
    assert!(
        model
            .materials
            .iter()
            .any(|material| material.nu_of_b2.is_some())
    );
    let result = solve_steady(
        &mut cx,
        &model,
        &ParamSet::default(),
        &FemmLimits::default(),
        None,
    )
    .unwrap();
    let cert = &result.certificate;
    assert!(cert.converged);
    assert!(cert.final_residual.is_finite());
    assert!(cert.final_residual < 1.0e-8);
    assert_eq!(cert.method, "femm-ptc");
    assert!(cert.iterations > 0);
    let claim_id = cert.claim.content_id(cx.datum_store_mut()).unwrap();
    assert!(claim_id.bytes.iter().any(|byte| *byte != 0));
}

#[test]
fn non_convergent_ptc_error_message_carries_method_tag() {
    use crate::gapped_ei_core_inductor;
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let tight = FemmLimits {
        max_solve_iters: 1,
        ..FemmLimits::default()
    };
    let err = match solve_steady(
        &mut cx,
        &gapped_ei_core_inductor(),
        &ParamSet::default(),
        &tight,
        None,
    ) {
        Ok(_) => panic!("expected SolveDidNotConverge"),
        Err(err) => err,
    };
    let FemmError::SolveDidNotConverge(message) = err else {
        panic!("expected SolveDidNotConverge");
    };
    assert!(
        message.contains("femm-ptc"),
        "error must name the method: {message}"
    );
    assert!(
        message.contains("partial certificate"),
        "error must name the partial certificate: {message}"
    );
}

#[test]
fn linear_solve_emits_valid_certificate() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let result = solve_steady(
        &mut cx,
        &parallel_plate_capacitor(),
        &ParamSet::default(),
        &FemmLimits::default(),
        None,
    )
    .unwrap();
    let cert = &result.certificate;
    assert!(cert.converged);
    assert!(cert.final_residual.is_finite());
    assert!(cert.final_residual < 1.0e-10);
    assert_eq!(cert.method, "femm-direct");
    assert_eq!(cert.iterations, 1);
    let claim_id = cert.claim.content_id(cx.datum_store_mut()).unwrap();
    assert!(claim_id.bytes.iter().any(|byte| *byte != 0));
    assert!(cert.gradient_trust.is_none());
}

#[test]
fn quality_query_solve_export_record_fills_all_fields() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let solve = solve_steady(
        &mut cx,
        &parallel_plate_capacitor(),
        &ParamSet::default(),
        &FemmLimits::default(),
        None,
    )
    .unwrap();
    let record = SolveExportRecord::from(&solve);
    assert_eq!(record.solution_id, solve.solution.id);
    assert!(!record.physics.is_empty());
    assert!(!record.method.is_empty());
    assert!(record.converged);
    assert!(record.final_residual.is_finite());
    assert_eq!(record.iterations, 1);
    assert!(!record.gradient_trust.is_empty());
    assert!(!record.certificate_claim_key.is_empty());

    let rebuilt = certificate_claim(&mut cx, &solve).unwrap();
    let rebuilt_id = rebuilt.content_id(cx.datum_store_mut()).unwrap();
    assert_eq!(record.certificate_claim_key, content_id_hex(&rebuilt_id));
}

#[test]
fn quality_query_returns_certificate_and_value() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let solve = solve_steady(
        &mut cx,
        &parallel_plate_capacitor(),
        &ParamSet::default(),
        &FemmLimits::default(),
        None,
    )
    .unwrap();
    let quantity_spec = QuantitySpec::Energy { region: None };
    let wrt = [Symbol::new("gap-mm")];
    let answer = quality(&mut cx, &solve, &quantity_spec, Some(&wrt)).unwrap();
    assert!(answer.value.is_finite());
    assert!(answer.certificate.converged);
    let claim_id = answer
        .certificate
        .claim
        .content_id(cx.datum_store_mut())
        .unwrap();
    assert!(claim_id.bytes.iter().any(|byte| *byte != 0));
    let Some((gradient, trust)) = answer.gradient else {
        panic!("expected quality gradient");
    };
    assert_eq!(gradient.len(), 1);
    assert!(gradient.iter().all(|value| value.is_finite()));
    assert_ne!(trust, GradientTrust::AdjointUnverified);
}

#[test]
fn all_physics_kinds_validate_solve_and_postprocess() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    for physics in [
        PhysicsKind::Magnetostatic,
        PhysicsKind::MagneticsHarmonic,
        PhysicsKind::Electrostatic,
        PhysicsKind::HeatSteady,
        PhysicsKind::CurrentSteady,
    ] {
        for formulation in [Formulation::Planar, Formulation::Axisymmetric] {
            let mut model = parallel_plate_capacitor();
            model.physics = physics.clone();
            model.formulation = formulation;
            let solved = solve_steady(
                &mut cx,
                &model,
                &ParamSet::default(),
                &FemmLimits::default(),
                None,
            )
            .unwrap();
            assert!(energy(&solved.solution).unwrap() >= 0.0);
        }
    }
}

fn content_id_hex(id: &ContentId) -> String {
    id.bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
