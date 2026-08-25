use sim_lib_femm_physics::{FemmAuditInput, adapt_femm};
use sim_lib_physics_adapter::{AdapterRefusal, DataOrigin, ENERGY, POWER};
fn input() -> FemmAuditInput {
    FemmAuditInput {
        model_id: "femm/model/analytic".into(),
        solution_id: "femm/solution/1".into(),
        boundary_id: "femm/boundary/domain".into(),
        boundary_complete: true,
        port: Some((
            "port/electrical".into(),
            "si:voltage".into(),
            "si:current".into(),
        )),
        stored_energy_j: 2.5,
        loss_w: 0.25,
        excitations: vec!["coil=1A".into()],
        influenced_states: vec!["state/temperature".into()],
        model_evidence: vec!["analytic-uniform-field".into()],
        solve_certificate: "converged:residual=0".into(),
        sensitivity_certificate: Some("adjoint-verified".into()),
        origin: DataOrigin::Modeled,
    }
}
#[test]
fn maps_analytic_quantities_certificates_and_influence() {
    let out = adapt_femm(input()).unwrap();
    assert_eq!(out.observations[0].dimension, ENERGY);
    assert_eq!(out.observations[1].dimension, POWER);
    assert!(out.solver_evidence.iter().any(|v| v == "adjoint-verified"));
    assert_eq!(out.influences[0].as_str(), "state/temperature");
    out.validate_lumped_audit().unwrap();
}
#[test]
fn refuses_incomplete_boundary() {
    let mut value = input();
    value.boundary_complete = false;
    assert_eq!(adapt_femm(value), Err(AdapterRefusal::IncompleteBoundary));
}
