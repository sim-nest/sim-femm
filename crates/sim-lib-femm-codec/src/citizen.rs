//! Citizen descriptors for FEMM domain objects.
//!
//! Compact, codec-addressable records (field, geometry, material, mesh, space)
//! that let FEMM objects round-trip across the runtime's codec surfaces.

use sim_citizen_derive::Citizen;
use sim_kernel::Symbol;
use sim_lib_femm_field::{Field, Projection};
use sim_lib_femm_function::{FemmFuncPayload, OutputQuery};
use sim_lib_femm_mesh::{FemMesh2, FemmModel};
use sim_lib_femm_post::{FemmSolution, QuantitySpec};

use crate::support::{formulation_name, physics_name};

/// Codec-addressable descriptor for a solved [`Field`] projection.
///
/// Carries the source solution id and the projected quantity name so a field
/// round-trips across codec surfaces under the `femm/Field` citizen symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Field", version = 1)]
pub struct FemmFieldDescriptor {
    /// Id of the solution the field was projected from.
    pub solution_id: u64,
    /// Name of the projected quantity (for example `"bmag"`).
    #[citizen(with = "descriptor_text")]
    pub projection: String,
}

/// Codec-addressable descriptor for a FEMM model geometry.
///
/// Summarizes region and boundary counts plus an artifact reference under the
/// `femm/Geometry` citizen symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Geometry", version = 1)]
pub struct FemmGeometryDescriptor {
    /// Number of labeled regions in the geometry.
    pub region_count: usize,
    /// Number of boundary segments in the geometry.
    pub boundary_count: usize,
    /// Reference to the backing geometry artifact.
    #[citizen(with = "descriptor_text")]
    pub artifact_ref: String,
}

/// Codec-addressable descriptor for a FEMM material.
///
/// Names the material and lists its property keys under the `femm/Material`
/// citizen symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Material", version = 1)]
pub struct FemmMaterialDescriptor {
    /// Material name (for example `"air"`).
    #[citizen(with = "descriptor_text")]
    pub name: String,
    /// Property keys carried by the material (for example `"mu-r"`).
    pub properties: Vec<String>,
}

/// Codec-addressable descriptor for a FEMM mesh.
///
/// Summarizes node and element counts plus an artifact reference under the
/// `femm/Mesh` citizen symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Mesh", version = 1)]
pub struct FemmMeshDescriptor {
    /// Number of mesh nodes.
    pub nodes: usize,
    /// Number of mesh elements.
    pub elements: usize,
    /// Reference to the backing mesh artifact.
    #[citizen(with = "descriptor_text")]
    pub artifact_ref: String,
}

/// Codec-addressable descriptor for a function space.
///
/// Summarizes element count and formulation under the `femm/Space` citizen
/// symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Space", version = 1)]
pub struct FemmSpaceDescriptor {
    /// Number of elements the space spans.
    pub element_count: usize,
    /// Formulation name (for example `"planar"`).
    #[citizen(with = "descriptor_text")]
    pub formulation: String,
}

/// Codec-addressable descriptor for a physics problem.
///
/// Names the physics kind and formulation under the `femm/Physics` citizen
/// symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Physics", version = 1)]
pub struct FemmPhysicsDescriptor {
    /// Physics kind name (for example `"electrostatic"`).
    #[citizen(with = "descriptor_text")]
    pub physics: String,
    /// Formulation name (for example `"planar"`).
    #[citizen(with = "descriptor_text")]
    pub formulation: String,
}

/// Codec-addressable descriptor for a solve step.
///
/// Names the linear-solve method and matrix reference under the `femm/Solve`
/// citizen symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Solve", version = 1)]
pub struct FemmSolveDescriptor {
    /// Solve method name (for example `"sparse-lu"`).
    #[citizen(with = "descriptor_text")]
    pub method: String,
    /// Reference to the backing system matrix.
    #[citizen(with = "descriptor_text")]
    pub matrix_ref: String,
}

/// Codec-addressable descriptor for a [`FemmSolution`].
///
/// Summarizes the solution's identity, physics, parameters, and mesh size under
/// the `femm/Solution` citizen symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Solution", version = 1)]
pub struct FemmSolutionDescriptor {
    /// Solution id.
    pub id: u64,
    /// Id of the model the solution was produced from.
    pub model_id: u64,
    /// Physics kind name (for example `"electrostatic"`).
    #[citizen(with = "descriptor_text")]
    pub physics: String,
    /// Formulation name (for example `"planar"`).
    #[citizen(with = "descriptor_text")]
    pub formulation: String,
    /// Solve parameter names bound for the solution.
    pub params: Vec<String>,
    /// Number of mesh nodes in the solution.
    pub nodes: usize,
    /// Number of mesh elements in the solution.
    pub elements: usize,
}

/// Codec-addressable descriptor for a post-processing query.
///
/// Names the requested quantity and its target under the `femm/Post` citizen
/// symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Post", version = 1)]
pub struct FemmPostDescriptor {
    /// Requested quantity name (for example `"energy"`).
    #[citizen(with = "descriptor_text")]
    pub quantity: String,
    /// Target the quantity is evaluated over (for example `"region:air"`).
    #[citizen(with = "descriptor_text")]
    pub target: String,
}

/// Codec-addressable descriptor for a parameterized output function.
///
/// Names the model, output query, and free variables under the `femm/Function`
/// citizen symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Function", version = 1)]
pub struct FemmFunctionDescriptor {
    /// Id of the model the function evaluates.
    pub model_id: u64,
    /// Output query name (for example `"quantity:energy"`).
    #[citizen(with = "descriptor_text")]
    pub query: String,
    /// Free variable names the function is parameterized over.
    pub vars: Vec<String>,
}

/// Codec-addressable descriptor for a function-evaluation payload.
///
/// Mirrors [`FemmFunctionDescriptor`] under the `femm/FuncPayload` citizen
/// symbol, naming the payload's model, query, and variables.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/FuncPayload", version = 1)]
pub struct FemmFuncPayloadDescriptor {
    /// Id of the model the payload evaluates.
    pub model_id: u64,
    /// Output query name (for example `"quantity:energy"`).
    #[citizen(with = "descriptor_text")]
    pub query: String,
    /// Free variable names the payload is parameterized over.
    pub vars: Vec<String>,
}

/// Codec-addressable descriptor for a [`FemmModel`].
///
/// Summarizes the model's identity, physics, formulation, and input parameters
/// under the `femm/Model` citizen symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Model", version = 1)]
pub struct FemmModelDescriptor {
    /// Model id.
    pub id: u64,
    /// Model name (for example `"parallel-plate-capacitor"`).
    #[citizen(with = "descriptor_text")]
    pub name: String,
    /// Physics kind name (for example `"electrostatic"`).
    #[citizen(with = "descriptor_text")]
    pub physics: String,
    /// Formulation name (for example `"planar"`).
    #[citizen(with = "descriptor_text")]
    pub formulation: String,
    /// Input parameter names declared by the model.
    pub params: Vec<String>,
}

/// Codec-addressable descriptor for a sensitivity computation.
///
/// Names the differentiation path and the parameters differentiated with
/// respect to, under the `femm/Sensitivity` citizen symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Sensitivity", version = 1)]
pub struct FemmSensitivityDescriptor {
    /// Sensitivity path name (for example `"direct-exact"`).
    #[citizen(with = "descriptor_text")]
    pub path: String,
    /// Parameter names the sensitivity is taken with respect to.
    pub wrt: Vec<String>,
}

/// Codec-addressable descriptor for a sensitivity tape.
///
/// Summarizes recorded factor and solution counts plus an artifact reference
/// under the `femm/Tape` citizen symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Tape", version = 1)]
pub struct FemmTapeDescriptor {
    /// Number of factors recorded on the tape.
    pub factors: usize,
    /// Number of solutions recorded on the tape.
    pub solutions: usize,
    /// Reference to the backing tape artifact.
    #[citizen(with = "descriptor_text")]
    pub artifact_ref: String,
}

/// Codec-addressable descriptor for an ODE integration coupling.
///
/// Names the integrated state variables and the field quantities they require,
/// under the `femm/Ode` citizen symbol.
#[derive(Clone, Debug, PartialEq, Citizen)]
#[citizen(symbol = "femm/Ode", version = 1)]
pub struct FemmOdeDescriptor {
    /// Integrated state variable names.
    pub state_vars: Vec<String>,
    /// Field quantity names the integration consumes.
    pub quantity_needs: Vec<String>,
}

impl FemmFieldDescriptor {
    /// Builds the descriptor from a solved [`Field`].
    pub fn from_field(field: &Field) -> Self {
        Self {
            solution_id: field.solution_id().0,
            projection: projection_name(&field.projection()),
        }
    }
}

impl FemmMeshDescriptor {
    /// Builds the descriptor from a [`FemMesh2`] and its artifact reference.
    pub fn from_mesh(mesh: &FemMesh2, artifact_ref: impl Into<String>) -> Self {
        Self {
            nodes: mesh.xy.len(),
            elements: mesh.tri.len(),
            artifact_ref: artifact_ref.into(),
        }
    }
}

impl FemmSolutionDescriptor {
    /// Builds the descriptor from a [`FemmSolution`].
    pub fn from_solution(solution: &FemmSolution) -> Self {
        Self {
            id: solution.id.0,
            model_id: solution.model_id.0,
            physics: physics_name(&solution.physics).to_owned(),
            formulation: formulation_name(&solution.formulation).to_owned(),
            params: solution
                .params
                .entries
                .iter()
                .map(|(name, _)| name.to_string())
                .collect(),
            nodes: solution.mesh.xy.len(),
            elements: solution.mesh.tri.len(),
        }
    }
}

impl FemmFunctionDescriptor {
    /// Builds the descriptor from a [`FemmFuncPayload`].
    pub fn from_payload(payload: &FemmFuncPayload) -> Self {
        Self {
            model_id: payload.model.id.0,
            query: query_name(&payload.query),
            vars: payload.vars.iter().map(ToString::to_string).collect(),
        }
    }
}

impl FemmFuncPayloadDescriptor {
    /// Builds the descriptor from a [`FemmFuncPayload`].
    pub fn from_payload(payload: &FemmFuncPayload) -> Self {
        let descriptor = FemmFunctionDescriptor::from_payload(payload);
        Self {
            model_id: descriptor.model_id,
            query: descriptor.query,
            vars: descriptor.vars,
        }
    }
}

impl FemmModelDescriptor {
    /// Builds the descriptor from a [`FemmModel`].
    pub fn from_model(model: &FemmModel) -> Self {
        Self {
            id: model.id.0,
            name: model.name.to_string(),
            physics: physics_name(&model.physics).to_owned(),
            formulation: formulation_name(&model.formulation).to_owned(),
            params: model
                .inputs
                .iter()
                .map(|param| param.name.to_string())
                .collect(),
        }
    }
}

impl Default for FemmFieldDescriptor {
    fn default() -> Self {
        Self {
            solution_id: 1,
            projection: "potential".to_owned(),
        }
    }
}

impl Default for FemmGeometryDescriptor {
    fn default() -> Self {
        Self {
            region_count: 1,
            boundary_count: 0,
            artifact_ref: "table:femm/geometry/citizen".to_owned(),
        }
    }
}

impl Default for FemmMaterialDescriptor {
    fn default() -> Self {
        Self {
            name: "air".to_owned(),
            properties: vec!["epsilon-r".to_owned(), "mu-r".to_owned()],
        }
    }
}

impl Default for FemmMeshDescriptor {
    fn default() -> Self {
        Self {
            nodes: 3,
            elements: 1,
            artifact_ref: "table:femm/mesh/citizen".to_owned(),
        }
    }
}

impl Default for FemmSpaceDescriptor {
    fn default() -> Self {
        Self {
            element_count: 1,
            formulation: "planar".to_owned(),
        }
    }
}

impl Default for FemmPhysicsDescriptor {
    fn default() -> Self {
        Self {
            physics: "electrostatic".to_owned(),
            formulation: "planar".to_owned(),
        }
    }
}

impl Default for FemmSolveDescriptor {
    fn default() -> Self {
        Self {
            method: "sparse-lu".to_owned(),
            matrix_ref: "stable:femm/matrix/citizen".to_owned(),
        }
    }
}

impl Default for FemmSolutionDescriptor {
    fn default() -> Self {
        Self {
            id: 1,
            model_id: 1,
            physics: "electrostatic".to_owned(),
            formulation: "planar".to_owned(),
            params: Vec::new(),
            nodes: 3,
            elements: 1,
        }
    }
}

impl Default for FemmPostDescriptor {
    fn default() -> Self {
        Self {
            quantity: "energy".to_owned(),
            target: "region:air".to_owned(),
        }
    }
}

impl Default for FemmFunctionDescriptor {
    fn default() -> Self {
        Self {
            model_id: 1,
            query: "quantity:energy".to_owned(),
            vars: vec!["gap".to_owned()],
        }
    }
}

impl Default for FemmFuncPayloadDescriptor {
    fn default() -> Self {
        Self {
            model_id: 1,
            query: "quantity:energy".to_owned(),
            vars: vec!["gap".to_owned()],
        }
    }
}

impl Default for FemmModelDescriptor {
    fn default() -> Self {
        Self {
            id: 1,
            name: "parallel-plate-capacitor".to_owned(),
            physics: "electrostatic".to_owned(),
            formulation: "planar".to_owned(),
            params: vec!["gap-mm".to_owned()],
        }
    }
}

impl Default for FemmSensitivityDescriptor {
    fn default() -> Self {
        Self {
            path: "direct-exact".to_owned(),
            wrt: vec!["gap".to_owned()],
        }
    }
}

impl Default for FemmTapeDescriptor {
    fn default() -> Self {
        Self {
            factors: 1,
            solutions: 1,
            artifact_ref: "stable:femm/tape/citizen".to_owned(),
        }
    }
}

impl Default for FemmOdeDescriptor {
    fn default() -> Self {
        Self {
            state_vars: vec!["x".to_owned(), "v".to_owned()],
            quantity_needs: vec!["energy".to_owned()],
        }
    }
}

macro_rules! class_symbol_fn {
    ($name:ident, $class:literal) => {
        #[doc = concat!("Citizen class [`Symbol`] `femm/", $class, "` for the matching descriptor.")]
        pub fn $name() -> Symbol {
            Symbol::qualified("femm", $class)
        }
    };
}

class_symbol_fn!(femm_field_class_symbol, "Field");
class_symbol_fn!(femm_geometry_class_symbol, "Geometry");
class_symbol_fn!(femm_material_class_symbol, "Material");
class_symbol_fn!(femm_mesh_class_symbol, "Mesh");
class_symbol_fn!(femm_space_class_symbol, "Space");
class_symbol_fn!(femm_physics_class_symbol, "Physics");
class_symbol_fn!(femm_solve_class_symbol, "Solve");
class_symbol_fn!(femm_solution_class_symbol, "Solution");
class_symbol_fn!(femm_post_class_symbol, "Post");
class_symbol_fn!(femm_function_class_symbol, "Function");
class_symbol_fn!(femm_func_payload_class_symbol, "FuncPayload");
class_symbol_fn!(femm_model_class_symbol, "Model");
class_symbol_fn!(femm_sensitivity_class_symbol, "Sensitivity");
class_symbol_fn!(femm_tape_class_symbol, "Tape");
class_symbol_fn!(femm_ode_class_symbol, "Ode");

pub(crate) mod descriptor_text {
    use sim_kernel::{Error, Expr, Result};

    pub fn encode(text: &str) -> Expr {
        Expr::String(text.to_owned())
    }

    pub fn decode(expr: &Expr) -> Result<String> {
        let Expr::String(text) = expr else {
            return Err(Error::Eval(
                "FEMM descriptor text must be a string".to_owned(),
            ));
        };
        validate_descriptor_text(text)?;
        Ok(text.clone())
    }

    fn validate_descriptor_text(text: &str) -> Result<()> {
        if text.trim().is_empty() {
            return Err(Error::Eval(
                "FEMM descriptor text cannot be empty".to_owned(),
            ));
        }
        if !text.is_ascii() {
            return Err(Error::Eval("FEMM descriptor text must be ASCII".to_owned()));
        }
        Ok(())
    }
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

fn query_name(query: &OutputQuery) -> String {
    match query {
        OutputQuery::Quantity(spec) => format!("quantity:{}", quantity_name(spec)),
        OutputQuery::Field(projection) => format!("field:{}", projection_name(projection)),
        OutputQuery::Solution => "solution".to_owned(),
    }
}

fn quantity_name(spec: &QuantitySpec) -> String {
    match spec {
        QuantitySpec::Energy { region } => optional_region("energy", region.as_ref()),
        QuantitySpec::Coenergy { region } => optional_region("coenergy", region.as_ref()),
        QuantitySpec::ForceY { region } => format!("force-y:{region}"),
        QuantitySpec::Torque { region, .. } => format!("torque:{region}"),
        QuantitySpec::FluxLinkage { circuit } => format!("flux-linkage:{circuit}"),
        QuantitySpec::Inductance { circuit } => format!("inductance:{circuit}"),
        QuantitySpec::Capacitance { conductor } => format!("capacitance:{conductor}"),
        QuantitySpec::JouleLoss { region } => optional_region("joule-loss", region.as_ref()),
        QuantitySpec::FieldAt { field, .. } => format!("field-at:{field}"),
        QuantitySpec::Custom { name, .. } => format!("custom:{name}"),
    }
}

fn optional_region(prefix: &str, region: Option<&Symbol>) -> String {
    region
        .map(|region| format!("{prefix}:{region}"))
        .unwrap_or_else(|| prefix.to_owned())
}
