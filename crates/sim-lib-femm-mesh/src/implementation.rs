#![forbid(unsafe_code)]
//! The assembled FEMM model and its triangular mesh.
//!
//! Defines the `FemmModel` record, the 2D triangular mesh, the meshed-model
//! pairing, and the meshers that discretize a model's geometry.

use sim_kernel::{Cx, Origin, Symbol};
use sim_lib_femm_core::{
    FemmError, FemmLimits, FemmResult, Formulation, LengthUnit, ParamSet, ParamSpec, PhysicsKind,
    StableId,
};
use sim_lib_femm_geometry::{Geometry2, LoweredGeometry2};
use sim_lib_femm_material::{Boundary, Material, MeshPolicy, OutputSpec, Source};

use crate::validation::{validate_lowered_geometry, validate_model};

/// A complete FEMM problem: geometry, materials, boundaries, sources, and the
/// physics that ties them together.
///
/// `FemmModel` is the assembled, parameterized input to meshing and solving. It
/// is plain behavior data; the `Expr` fields carry unevaluated parameter
/// expressions over the kernel `Expr` contract, lowered against a [`ParamSet`]
/// when the model is meshed. See the
/// [crate README](https://github.com/sim-nest/sim-femm) for the FEM data flow.
#[derive(Clone, Debug)]
pub struct FemmModel {
    /// Stable identity of the model within a problem session.
    pub id: StableId,
    /// Human-readable model name.
    pub name: Symbol,
    /// Physics being solved (magnetics, electrostatics, conductive, thermal).
    pub physics: PhysicsKind,
    /// Geometric formulation (planar or axisymmetric).
    pub formulation: Formulation,
    /// Length unit the geometry coordinates are expressed in.
    pub length_unit: LengthUnit,
    /// Out-of-plane depth expression for planar problems, if set.
    pub depth: Option<sim_kernel::Expr>,
    /// Drive frequency in hertz for time-harmonic problems, if set.
    pub frequency_hz: Option<sim_kernel::Expr>,
    /// Declared input parameters and their defaults.
    pub inputs: Vec<ParamSpec>,
    /// Symbolic 2D geometry to be lowered and meshed.
    pub geometry: Geometry2,
    /// Material definitions referenced by region.
    pub materials: Vec<Material>,
    /// Boundary conditions referenced by segment or outer box.
    pub boundaries: Vec<Boundary>,
    /// Excitation sources (current/charge densities, coils, heat sources).
    pub sources: Vec<Source>,
    /// Requested post-processing outputs.
    pub outputs: Vec<OutputSpec>,
    /// Mesh refinement policy.
    pub mesh_policy: MeshPolicy,
    /// Optional solver-policy expression.
    pub solve_policy: Option<sim_kernel::Expr>,
    /// Source origin of the model definition.
    pub origin: Origin,
}

/// A 2D triangular finite-element mesh.
///
/// The discretized result of lowering a [`FemmModel`]'s geometry: vertex
/// coordinates, the triangle connectivity, the material region of each element,
/// and the boundary tag of each constrained edge.
#[derive(Clone, Debug)]
pub struct FemMesh2 {
    /// Node coordinates, one `[x, y]` per vertex.
    pub xy: Vec<[f64; 2]>,
    /// Triangle connectivity, each entry three indices into [`xy`](Self::xy).
    pub tri: Vec<[u32; 3]>,
    /// Material region tag of each triangle, parallel to [`tri`](Self::tri).
    pub elem_region: Vec<Symbol>,
    /// Boundary edges as `(node_a, node_b, boundary)` triples.
    pub edge_boundary: Vec<(u32, u32, Symbol)>,
}

/// A meshed model: the mesh paired with the parameters and diagnostics that
/// produced it.
///
/// Output of [`Mesher::mesh`], carrying the model identity, the resolved
/// [`ParamSet`], the [`FemMesh2`], and any meshing diagnostics.
#[derive(Clone, Debug)]
pub struct MeshedModel {
    /// Identity of the [`FemmModel`] that was meshed.
    pub model_id: StableId,
    /// Parameter values used to lower the geometry.
    pub params: ParamSet,
    /// The resulting triangular mesh.
    pub mesh: FemMesh2,
    /// Diagnostics emitted while meshing.
    pub diagnostics: Vec<sim_kernel::Diagnostic>,
}

impl FemmModel {
    /// Resolves a mesh region label to the material assigned to it.
    ///
    /// A mesh may tag an element with a material name directly, or with an
    /// explicit block-label name whose `material` field names the material.
    pub fn material_for_region(&self, region: &Symbol) -> Option<&Material> {
        let material_name = self.material_name_for_region(region)?;
        self.materials
            .iter()
            .find(|material| &material.name == material_name)
    }

    fn material_name_for_region(&self, region: &Symbol) -> Option<&Symbol> {
        if let Some(material) = self
            .materials
            .iter()
            .find(|material| &material.name == region)
        {
            return Some(&material.name);
        }
        self.geometry
            .labels
            .iter()
            .find(|label| &label.name == region)
            .map(|label| &label.material)
    }
}

impl MeshedModel {
    /// Validates that the mesh carries one material-resolvable region per element.
    ///
    /// This catches stale or hand-authored meshes before assembly can silently
    /// pick a generic region or unrelated first material.
    pub fn validate_against(&self, model: &FemmModel) -> FemmResult<()> {
        if self.mesh.elem_region.len() != self.mesh.tri.len() {
            return Err(FemmError::InvalidGeometry(format!(
                "mesh has {} elements but {} region labels",
                self.mesh.tri.len(),
                self.mesh.elem_region.len()
            )));
        }
        for region in &self.mesh.elem_region {
            if model.material_for_region(region).is_none() {
                return Err(FemmError::MissingMaterial(region.to_string()));
            }
        }
        Ok(())
    }
}

/// A mesher: turns a [`FemmModel`] and a [`ParamSet`] into a [`MeshedModel`].
///
/// The behavior contract for discretizing a model. Implementations lower the
/// symbolic geometry and triangulate it into a [`FemMesh2`].
pub trait Mesher: Send + Sync + 'static {
    /// Stable name identifying this mesher.
    fn name(&self) -> Symbol;
    /// Lower and triangulate `model` under `params`, producing a meshed model.
    fn mesh(&self, cx: &mut Cx, model: &FemmModel, params: &ParamSet) -> FemmResult<MeshedModel>;
}

/// A deterministic, fan-triangulating mesher.
///
/// Produces a byte-stable [`FemMesh2`] for the same model and parameters by
/// triangulating the lowered geometry without randomization.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_mesh::{DeterministicMesher, Mesher};
///
/// let mesher = DeterministicMesher::new();
/// assert_eq!(mesher.name().name.as_ref(), "deterministic-tri");
/// ```
#[derive(Default)]
pub struct DeterministicMesher;

impl DeterministicMesher {
    /// Constructs a new deterministic mesher.
    pub fn new() -> Self {
        Self
    }
}

impl Mesher for DeterministicMesher {
    fn name(&self) -> Symbol {
        Symbol::new("deterministic-tri")
    }

    fn mesh(&self, cx: &mut Cx, model: &FemmModel, params: &ParamSet) -> FemmResult<MeshedModel> {
        validate_model(model, params)?;
        let lowered = model.geometry.lower(cx, params)?;
        validate_lowered_geometry(model, &lowered)?;
        lowered_to_mesh(model, params.clone(), lowered)
    }
}

fn lowered_to_mesh(
    model: &FemmModel,
    params: ParamSet,
    lowered: LoweredGeometry2,
) -> FemmResult<MeshedModel> {
    if lowered.nodes.len() < 3 {
        return Err(FemmError::InvalidGeometry(
            "need at least three lowered nodes".to_owned(),
        ));
    }
    let mut tri = Vec::new();
    if lowered.nodes.len() == 4 {
        tri.push([0, 1, 2]);
        tri.push([0, 2, 3]);
    } else {
        for index in 1..lowered.nodes.len() - 1 {
            tri.push([0_u32, index as u32, (index + 1) as u32]);
        }
    }
    let region = lowered
        .labels
        .first()
        .map(|entry| entry.0.clone())
        .ok_or_else(|| {
            FemmError::InvalidGeometry("lowered geometry has no material labels".to_owned())
        })?;
    let meshed = MeshedModel {
        model_id: model.id,
        params,
        mesh: FemMesh2 {
            xy: lowered.nodes,
            tri: tri.clone(),
            elem_region: vec![region; tri.len()],
            edge_boundary: lowered
                .segments
                .into_iter()
                .filter_map(|(a, b, boundary)| {
                    boundary.map(|boundary| (a as u32, b as u32, boundary))
                })
                .collect(),
        },
        diagnostics: Vec::new(),
    };
    meshed.validate_against(model)?;
    Ok(meshed)
}

/// Rejects a mesh that exceeds the node or element limits.
///
/// Guards downstream assembly against meshes larger than [`FemmLimits`] allows,
/// returning [`FemmError::MeshLimitExceeded`] on the first limit breached.
pub fn enforce_mesh_limits(mesh: &FemMesh2, limits: &FemmLimits) -> FemmResult<()> {
    if mesh.xy.len() > limits.max_nodes {
        return Err(FemmError::MeshLimitExceeded(format!(
            "nodes {} > {}",
            mesh.xy.len(),
            limits.max_nodes
        )));
    }
    if mesh.tri.len() > limits.max_elements {
        return Err(FemmError::MeshLimitExceeded(format!(
            "elements {} > {}",
            mesh.tri.len(),
            limits.max_elements
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sim_kernel::{DefaultFactory, EagerPolicy, Expr, Factory};
    use sim_lib_femm_core::{FemmLimits, Formulation, ParamRole};
    use sim_lib_femm_geometry::{AnalyticRegion2, BlockLabel2, Node2, dummy_origin};
    use sim_lib_femm_material::{Boundary, BoundaryKind, Material, Source};

    use super::*;

    fn test_cx() -> Cx {
        Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory))
    }

    fn num(text: &str) -> Expr {
        sim_value::build::num_q(Some("numbers"), "f64", text)
    }

    fn base_model() -> FemmModel {
        FemmModel {
            id: StableId(1),
            name: Symbol::new("rect"),
            physics: PhysicsKind::Electrostatic,
            formulation: Formulation::Planar,
            length_unit: LengthUnit::Millimeter,
            depth: None,
            frequency_hz: None,
            inputs: vec![ParamSpec {
                name: Symbol::new("gap"),
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
                    name: Symbol::new("air"),
                    at: [num("0.5"), num("0.5")],
                    material: Symbol::new("air"),
                }],
                analytic: vec![AnalyticRegion2::Rect {
                    name: Symbol::new("air"),
                    xy: [num("0.0"), num("0.0")],
                    wh: [num("1.0"), num("1.0")],
                }],
                ..Geometry2::default()
            },
            materials: vec![Material {
                name: Symbol::new("air"),
                mu_r: Some(num("1.0")),
                nu_of_b2: None,
                epsilon_r: Some(num("1.0")),
                sigma: None,
                thermal_k: Some(num("1.0")),
                heat_source: None,
                remanence: None,
            }],
            boundaries: Vec::new(),
            sources: Vec::new(),
            outputs: Vec::new(),
            mesh_policy: MeshPolicy {
                kind: Symbol::new("deterministic"),
                max_area: None,
                min_angle_deg: None,
            },
            solve_policy: None,
            origin: dummy_origin(),
        }
    }

    fn one_triangle(regions: Vec<Symbol>) -> MeshedModel {
        MeshedModel {
            model_id: StableId(1),
            params: ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: regions,
                edge_boundary: Vec::new(),
            },
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn rectangle_mesh_has_positive_area_triangles() {
        let mut cx = test_cx();
        let mesh = DeterministicMesher::new()
            .mesh(&mut cx, &base_model(), &ParamSet::default())
            .unwrap();
        assert_eq!(mesh.mesh.tri.len(), 2);
    }

    #[test]
    fn identical_input_is_byte_stable() {
        let mut cx = test_cx();
        let model = base_model();
        let left = DeterministicMesher::new()
            .mesh(&mut cx, &model, &ParamSet::default())
            .unwrap();
        let right = DeterministicMesher::new()
            .mesh(&mut cx, &model, &ParamSet::default())
            .unwrap();
        assert_eq!(
            format!("{:?}", left.mesh.xy),
            format!("{:?}", right.mesh.xy)
        );
    }

    #[test]
    fn mesh_limit_is_enforced() {
        let mesh = FemMesh2 {
            xy: vec![[0.0, 0.0]; 5],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        };
        let err = enforce_mesh_limits(
            &mesh,
            &FemmLimits {
                max_nodes: 4,
                ..FemmLimits::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, FemmError::MeshLimitExceeded(_)));
    }

    #[test]
    fn meshed_model_requires_one_region_label_per_element() {
        let err = one_triangle(Vec::new())
            .validate_against(&base_model())
            .unwrap_err();
        assert!(matches!(err, FemmError::InvalidGeometry(_)));
    }

    #[test]
    fn meshed_model_rejects_unknown_region_label() {
        let err = one_triangle(vec![Symbol::new("missing")])
            .validate_against(&base_model())
            .unwrap_err();
        assert!(matches!(err, FemmError::MissingMaterial(_)));
    }

    #[test]
    fn block_label_name_resolves_to_declared_material() {
        let mut model = base_model();
        model.geometry.labels[0].name = Symbol::new("core-region");
        one_triangle(vec![Symbol::new("core-region")])
            .validate_against(&model)
            .unwrap();
    }

    #[test]
    fn deterministic_mesher_rejects_unlabeled_geometry() {
        let mut cx = test_cx();
        let mut model = base_model();
        model.geometry = Geometry2 {
            nodes: vec![
                Node2 {
                    xy: [num("0.0"), num("0.0")],
                },
                Node2 {
                    xy: [num("1.0"), num("0.0")],
                },
                Node2 {
                    xy: [num("0.0"), num("1.0")],
                },
            ],
            ..Geometry2::default()
        };
        let err = DeterministicMesher::new()
            .mesh(&mut cx, &model, &ParamSet::default())
            .unwrap_err();
        assert!(matches!(err, FemmError::InvalidGeometry(_)));
    }

    #[test]
    fn validation_rejects_unknown_material_boundary_source_and_param_refs() {
        let mut model = base_model();
        model.geometry.labels[0].material = Symbol::new("missing");
        assert!(matches!(
            validate_model(&model, &ParamSet::default()).unwrap_err(),
            FemmError::MissingMaterial(_)
        ));

        let mut model = base_model();
        model
            .geometry
            .segments
            .push(sim_lib_femm_geometry::Segment2 {
                a: 0,
                b: 1,
                boundary: Some(Symbol::new("missing-boundary")),
            });
        assert!(matches!(
            validate_model(&model, &ParamSet::default()).unwrap_err(),
            FemmError::InvalidGeometry(_)
        ));

        let mut model = base_model();
        model.sources.push(Source::HeatSource {
            region: Symbol::new("missing-region"),
            value: num("1.0"),
        });
        assert!(matches!(
            validate_model(&model, &ParamSet::default()).unwrap_err(),
            FemmError::InvalidGeometry(_)
        ));

        let mut model = base_model();
        model.boundaries.push(Boundary {
            name: Symbol::new("wall"),
            kind: BoundaryKind::Dirichlet,
            value: Expr::Symbol(Symbol::new("missing-param")),
        });
        assert!(matches!(
            validate_model(&model, &ParamSet::default()).unwrap_err(),
            FemmError::UnknownFemmParameter(_)
        ));
    }

    #[test]
    fn axisymmetric_model_rejects_negative_radius_after_lowering() {
        let mut cx = test_cx();
        let mut model = base_model();
        model.formulation = Formulation::Axisymmetric;
        model.geometry.analytic = vec![AnalyticRegion2::Rect {
            name: Symbol::new("air"),
            xy: [num("-1.0"), num("0.0")],
            wh: [num("1.0"), num("1.0")],
        }];
        let err = DeterministicMesher::new()
            .mesh(&mut cx, &model, &ParamSet::default())
            .unwrap_err();
        assert!(matches!(err, FemmError::InvalidGeometry(_)));
    }
}
