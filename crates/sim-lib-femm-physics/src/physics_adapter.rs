//! Lowering of FEMM-owned result summaries into the shared audit contract.
use sim_lib_physics_adapter::{
    AdaptedModel, AdapterRefusal, DataOrigin, ENERGY, Observation, POWER, PortDeclaration,
    StableIdentity,
};

/// Immutable FEMM result data required by the adapter; populated by FEMM-owned solution layers.
#[derive(Clone, Debug, PartialEq)]
pub struct FemmAuditInput {
    /// Stable model identity.
    pub model_id: String,
    /// Stable solution/state identity.
    pub solution_id: String,
    /// Explicit audited boundary.
    pub boundary_id: String,
    /// Whether every boundary exchange is enumerated.
    pub boundary_complete: bool,
    /// Optional conjugate port `(id, effort kind, flow kind)`.
    pub port: Option<(String, String, String)>,
    /// Field-derived stored energy in joules.
    pub stored_energy_j: f64,
    /// Field-derived loss in watts.
    pub loss_w: f64,
    /// Stable excitation descriptions.
    pub excitations: Vec<String>,
    /// ODE/DAE state identities influenced by this solution.
    pub influenced_states: Vec<String>,
    /// Formulation and constitutive-model evidence.
    pub model_evidence: Vec<String>,
    /// Preserved solve certificate summary.
    pub solve_certificate: String,
    /// Optional sensitivity certificate summary.
    pub sensitivity_certificate: Option<String>,
    /// Modeled or observed record origin.
    pub origin: DataOrigin,
}

/// Lowers FEMM results without solving, probing, or moving domain behavior.
pub fn adapt_femm(input: FemmAuditInput) -> Result<AdaptedModel, AdapterRefusal> {
    let ports = input
        .port
        .map(|(id, effort_kind, flow_kind)| {
            Ok(PortDeclaration {
                id: StableIdentity::new(id)?,
                effort_kind,
                flow_kind,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, AdapterRefusal>>()?;
    let mut solver_evidence = vec![input.solve_certificate];
    if let Some(value) = input.sensitivity_certificate {
        solver_evidence.push(value);
    }
    solver_evidence.extend(input.excitations.iter().map(|v| format!("excitation:{v}")));
    let result = AdaptedModel {
        model_id: StableIdentity::new(input.model_id)?,
        state_id: StableIdentity::new(input.solution_id)?,
        boundary_id: StableIdentity::new(input.boundary_id)?,
        boundary_complete: input.boundary_complete,
        ports,
        stores: vec![StableIdentity::new("store/field-energy")?],
        events: vec![StableIdentity::new("event/femm-solve")?],
        observations: vec![
            Observation {
                id: StableIdentity::new("observation/field-energy")?,
                kind: "si:energy".into(),
                dimension: ENERGY,
                value: input.stored_energy_j,
                unit: "J".into(),
                origin: input.origin,
            },
            Observation {
                id: StableIdentity::new("observation/field-loss")?,
                kind: "si:power".into(),
                dimension: POWER,
                value: input.loss_w,
                unit: "W".into(),
                origin: input.origin,
            },
        ],
        influences: input
            .influenced_states
            .into_iter()
            .map(StableIdentity::new)
            .collect::<Result<_, _>>()?,
        model_evidence: input.model_evidence,
        solver_evidence,
    };
    result.validate()?;
    Ok(result)
}
