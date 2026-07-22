//! Lisp/JSON summary forms for FEMM models, solutions, and fields.
//!
//! Encodes models, solutions, and fields to their textual summary forms and
//! parses those forms back into the corresponding summary records.

use sim_lib_femm_core::{FemmError, FemmResult};
use sim_lib_femm_field::{Field, Projection};
use sim_lib_femm_mesh::FemmModel;
use sim_lib_femm_post::FemmSolution;

use crate::support::{
    formulation_name, parse_atom_field, parse_json_params, parse_json_string_field,
    parse_json_u64_field, parse_lisp_params, parse_u64_field, physics_name,
};

/// Parsed summary form of a [`FemmModel`].
///
/// The record a model's Lisp or JSON summary string decodes into, and the
/// source the encoders read from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSummary {
    /// Model id.
    pub id: u64,
    /// Model name.
    pub name: String,
    /// Physics kind name.
    pub physics: String,
    /// Formulation name.
    pub formulation: String,
    /// Input parameter names.
    pub params: Vec<String>,
}

/// Parsed summary form of a [`FemmSolution`].
///
/// The record a solution's Lisp or JSON summary string decodes into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolutionSummary {
    /// Solution id.
    pub id: u64,
    /// Id of the model the solution was produced from.
    pub model_id: u64,
    /// Physics kind name.
    pub physics: String,
    /// Formulation name.
    pub formulation: String,
    /// Solve parameter names.
    pub params: Vec<String>,
}

/// Parsed summary form of a solved [`Field`].
///
/// The record a field's read-construct string decodes into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldSummary {
    /// Id of the solution the field was projected from.
    pub solution_id: u64,
    /// Name of the projected quantity.
    pub projection: String,
}

/// Encodes a [`FemmModel`] to its parenthesized Lisp summary form.
pub fn model_to_lisp(model: &FemmModel) -> String {
    format!(
        "(femm/model :id {} :name {} :physics {} :formulation {} :params ({}))",
        model.id.0,
        model.name,
        physics_name(&model.physics),
        formulation_name(&model.formulation),
        model
            .inputs
            .iter()
            .map(|param| param.name.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    )
}

/// Decodes a model's Lisp summary form into a [`ModelSummary`].
///
/// # Examples
///
/// ```
/// use sim_lib_femm_codec::model_from_lisp;
///
/// let text = "(femm/model :id 7 :name plate :physics electrostatic \
///              :formulation planar :params (gap-mm))";
/// let summary = model_from_lisp(text).unwrap();
/// assert_eq!(summary.id, 7);
/// assert_eq!(summary.name, "plate");
/// assert_eq!(summary.params, vec!["gap-mm".to_owned()]);
/// ```
pub fn model_from_lisp(text: &str) -> FemmResult<ModelSummary> {
    Ok(ModelSummary {
        id: parse_u64_field(text, ":id ")?,
        name: parse_atom_field(text, ":name ")?,
        physics: parse_atom_field(text, ":physics ")?,
        formulation: parse_atom_field(text, ":formulation ")?,
        params: parse_lisp_params(text)?,
    })
}

/// Encodes a [`FemmModel`] to its JSON summary form.
pub fn model_to_json(model: &FemmModel) -> String {
    let params = model
        .inputs
        .iter()
        .map(|param| format!("\"{}\"", param.name))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"id\":{},\"name\":\"{}\",\"physics\":\"{}\",\"formulation\":\"{}\",\"params\":[{}]}}",
        model.id.0,
        model.name,
        physics_name(&model.physics),
        formulation_name(&model.formulation),
        params
    )
}

/// Decodes a model's JSON summary form into a [`ModelSummary`].
///
/// # Examples
///
/// ```
/// use sim_lib_femm_codec::model_from_json;
///
/// let text = "{\"id\":7,\"name\":\"plate\",\"physics\":\"electrostatic\",\
///              \"formulation\":\"planar\",\"params\":[\"gap-mm\"]}";
/// let summary = model_from_json(text).unwrap();
/// assert_eq!(summary.id, 7);
/// assert_eq!(summary.formulation, "planar");
/// assert_eq!(summary.params, vec!["gap-mm".to_owned()]);
/// ```
pub fn model_from_json(text: &str) -> FemmResult<ModelSummary> {
    Ok(ModelSummary {
        id: parse_json_u64_field(text, "\"id\":")?,
        name: parse_json_string_field(text, "\"name\":\"")?,
        physics: parse_json_string_field(text, "\"physics\":\"")?,
        formulation: parse_json_string_field(text, "\"formulation\":\"")?,
        params: parse_json_params(text)?,
    })
}

/// Accepts a binary frame tag in the reserved FEMM range, rejecting any other.
///
/// FEMM binary frames use tags `0xF0..=0xF7`; any tag outside that range is an
/// [`FemmError::InvalidGeometry`].
///
/// # Examples
///
/// ```
/// use sim_lib_femm_codec::reject_unknown_binary_tag;
///
/// assert!(reject_unknown_binary_tag(0xF0).is_ok());
/// assert!(reject_unknown_binary_tag(0x00).is_err());
/// ```
pub fn reject_unknown_binary_tag(tag: u8) -> FemmResult<()> {
    match tag {
        0xF0..=0xF7 => Ok(()),
        _ => Err(FemmError::InvalidGeometry(format!(
            "unknown frame tag {tag:#x}"
        ))),
    }
}

/// Encodes a [`FemmSolution`] to its parenthesized Lisp summary form.
pub fn solution_to_lisp(solution: &FemmSolution) -> String {
    format!(
        "(femm/solution :id {} :model {} :physics {} :formulation {} :params ({}))",
        solution.id.0,
        solution.model_id.0,
        physics_name(&solution.physics),
        formulation_name(&solution.formulation),
        solution
            .params
            .entries
            .iter()
            .map(|(param, _)| param.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    )
}

/// Decodes a solution's Lisp summary form into a [`SolutionSummary`].
pub fn solution_from_lisp(text: &str) -> FemmResult<SolutionSummary> {
    Ok(SolutionSummary {
        id: parse_u64_field(text, ":id ")?,
        model_id: parse_u64_field(text, ":model ")?,
        physics: parse_atom_field(text, ":physics ")?,
        formulation: parse_atom_field(text, ":formulation ")?,
        params: parse_lisp_params(text)?,
    })
}

/// Encodes a [`FemmSolution`] to its JSON summary form.
pub fn solution_to_json(solution: &FemmSolution) -> String {
    let params = solution
        .params
        .entries
        .iter()
        .map(|(param, _)| format!("\"{param}\""))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"id\":{},\"model\":{},\"physics\":\"{}\",\"formulation\":\"{}\",\"params\":[{}]}}",
        solution.id.0,
        solution.model_id.0,
        physics_name(&solution.physics),
        formulation_name(&solution.formulation),
        params
    )
}

/// Decodes a solution's JSON summary form into a [`SolutionSummary`].
pub fn solution_from_json(text: &str) -> FemmResult<SolutionSummary> {
    Ok(SolutionSummary {
        id: parse_json_u64_field(text, "\"id\":")?,
        model_id: parse_json_u64_field(text, "\"model\":")?,
        physics: parse_json_string_field(text, "\"physics\":\"")?,
        formulation: parse_json_string_field(text, "\"formulation\":\"")?,
        params: parse_json_params(text)?,
    })
}

/// Encodes a solved [`Field`] to its versioned `#(femm/Field ...)` read-construct.
pub fn field_read_construct(field: &Field) -> String {
    format!(
        "#(femm/Field v1 {} \"{}\")",
        field.solution_id().0,
        projection_name(&field.projection())
    )
}

/// Decodes a field read-construct into a [`FieldSummary`].
///
/// Accepts both the versioned `#(femm/Field v1 ...)` form and the lowercase
/// compatibility `#(femm/field ...)` form.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_codec::field_from_read_construct;
///
/// let summary = field_from_read_construct("#(femm/Field v1 3 \"bmag\")").unwrap();
/// assert_eq!(summary.solution_id, 3);
/// assert_eq!(summary.projection, "bmag");
/// ```
pub fn field_from_read_construct(text: &str) -> FemmResult<FieldSummary> {
    let (rest, versioned) = if let Some(rest) = text
        .strip_prefix("#(femm/Field v1 ")
        .and_then(|rest| rest.strip_suffix(')'))
    {
        (rest, true)
    } else {
        let rest = text
            .strip_prefix("#(femm/field ")
            .and_then(|rest| rest.strip_suffix(')'))
            .ok_or_else(|| FemmError::InvalidGeometry("bad field read-construct".to_owned()))?;
        (rest, false)
    };
    let mut parts = rest.split_whitespace();
    let solution_id = parts
        .next()
        .and_then(|part| part.parse::<u64>().ok())
        .ok_or_else(|| FemmError::InvalidGeometry("bad field solution id".to_owned()))?;
    let projection = parts
        .next()
        .ok_or_else(|| FemmError::InvalidGeometry("missing field projection".to_owned()))?;
    Ok(FieldSummary {
        solution_id,
        projection: if versioned {
            projection.trim_matches('"').to_owned()
        } else {
            projection.to_owned()
        },
    })
}

fn projection_name(projection: &Projection) -> String {
    match projection {
        Projection::Potential => "potential".to_owned(),
        Projection::Bx => "bx".to_owned(),
        Projection::By => "by".to_owned(),
        Projection::Bmag => "bmag".to_owned(),
        Projection::Ex => "ex".to_owned(),
        Projection::Ey => "ey".to_owned(),
        Projection::Emag => "emag".to_owned(),
        Projection::HeatFluxMag => "heat-flux-mag".to_owned(),
        Projection::Custom(symbol) => symbol.to_string(),
    }
}
