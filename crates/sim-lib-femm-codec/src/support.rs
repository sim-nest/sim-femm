use sim_lib_femm_core::{FemmError, FemmResult, Formulation, PhysicsKind};

pub(crate) fn physics_name(physics: &PhysicsKind) -> &'static str {
    match physics {
        PhysicsKind::Magnetostatic => "magnetostatic",
        PhysicsKind::MagneticsHarmonic => "magnetics-harmonic",
        PhysicsKind::Electrostatic => "electrostatic",
        PhysicsKind::HeatSteady => "heat-steady",
        PhysicsKind::CurrentSteady => "current-steady",
    }
}

pub(crate) fn formulation_name(formulation: &Formulation) -> &'static str {
    match formulation {
        Formulation::Planar => "planar",
        Formulation::Axisymmetric => "axisymmetric",
    }
}

pub(crate) fn parse_u64_field(text: &str, needle: &str) -> FemmResult<u64> {
    text.split(needle)
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| FemmError::InvalidGeometry(format!("missing field {needle}")))
}

pub(crate) fn parse_atom_field(text: &str, needle: &str) -> FemmResult<String> {
    text.split(needle)
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .map(|value| value.trim_end_matches(')').to_owned())
        .ok_or_else(|| FemmError::InvalidGeometry(format!("missing field {needle}")))
}

pub(crate) fn parse_lisp_params(text: &str) -> FemmResult<Vec<String>> {
    let params = text
        .split(":params (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .ok_or_else(|| FemmError::InvalidGeometry("missing lisp params".to_owned()))?;
    Ok(params
        .split_whitespace()
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect())
}

pub(crate) fn parse_json_u64_field(text: &str, needle: &str) -> FemmResult<u64> {
    text.split(needle)
        .nth(1)
        .and_then(|rest| rest.split(',').next())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| FemmError::InvalidGeometry(format!("missing json field {needle}")))
}

pub(crate) fn parse_json_string_field(text: &str, needle: &str) -> FemmResult<String> {
    text.split(needle)
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .map(str::to_owned)
        .ok_or_else(|| FemmError::InvalidGeometry(format!("missing json field {needle}")))
}

pub(crate) fn parse_json_params(text: &str) -> FemmResult<Vec<String>> {
    let params = text
        .split("\"params\":[")
        .nth(1)
        .and_then(|rest| rest.split(']').next())
        .ok_or_else(|| FemmError::InvalidGeometry("missing json params".to_owned()))?;
    Ok(params
        .split(',')
        .map(|item| item.trim().trim_matches('"'))
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect())
}
