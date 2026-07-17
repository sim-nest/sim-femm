#![forbid(unsafe_code)]
//! Element-level residuals and global system assembly.
//!
//! Defines the physics-front trait and the assembly routine that walks the
//! mesh, evaluates coefficients, and builds the global stiffness matrix and
//! load vector for a FEMM model.

use std::time::Instant;

use sim_kernel::{Cx, Symbol};
use sim_lib_femm_core::{CsrMatrix, FemmError, FemmLimits, FemmResult, ParamSet};
use sim_lib_femm_geometry::eval_expr_f64;
use sim_lib_femm_material::{Boundary, BoundaryKind, Material};
use sim_lib_femm_mesh::{FemmModel, MeshedModel};
use sim_lib_femm_space::ElementGeom;
use sim_lib_numbers_ad::{Dual, Scalarish};

/// Material coefficients resolved for one mesh region.
///
/// Assembly evaluates each region's [`Material`] expressions against the model
/// parameters once, then hands the resulting constants to a [`PhysicsFront`] so
/// element residuals never re-enter the expression evaluator.
#[derive(Clone, Debug)]
pub struct CoeffEval {
    /// Region (block-label) the element belongs to.
    pub region: Symbol,
    /// Source material whose expressions produced these coefficients.
    pub material: Material,
    /// Model parameters the expressions were evaluated against.
    pub params: ParamSet,
    /// Relative permittivity used by the electrostatic model.
    pub epsilon_r: f64,
    /// Electrical conductivity used by the conductive model.
    pub sigma: f64,
    /// Thermal conductivity used by the heat model.
    pub thermal_k: f64,
    /// Relative permeability used by the magnetic model.
    pub mu_r: f64,
    /// Volumetric source density (current, charge, or heat) for the region.
    pub source_density: f64,
    /// Excitation frequency in hertz; zero for static problems.
    pub frequency_hz: f64,
    /// Whether the source material carries a nonlinear B-H reluctivity curve
    /// (`nu_of_b2`). A front that can only solve the linear case must reject
    /// such a material via [`PhysicsFront::validate_coeff`] rather than silently
    /// solving it linearly.
    pub nonlinear_bh: bool,
}

/// A governing physics model in the form assembly consumes.
///
/// Each front contributes one element's residual over the three linear-triangle
/// nodes; assembly differentiates it (via the dual-number scalar) and scatters
/// the result into the global stiffness matrix and load vector. The concrete
/// models live in `sim-lib-femm-physics`; this trait is the assembly-side
/// contract they implement. See the
/// [crate README](https://github.com/sim-nest/sim-femm) for the FEM role.
///
/// # Examples
///
/// A minimal Laplacian front evaluated over one real triangle:
///
/// ```
/// use sim_kernel::Symbol;
/// use sim_lib_femm_assembly::{CoeffEval, PhysicsFront};
/// use sim_lib_femm_core::{ParamSet, PhysicsKind};
/// use sim_lib_femm_material::Material;
/// use sim_lib_femm_mesh::FemMesh2;
/// use sim_lib_femm_space::ElementGeom;
/// use sim_lib_numbers_ad::Scalarish;
///
/// struct Laplacian;
/// impl PhysicsFront for Laplacian {
///     fn kind(&self) -> PhysicsKind {
///         PhysicsKind::Electrostatic
///     }
///     fn element_residual<S: Scalarish>(
///         &self,
///         elem: &ElementGeom,
///         u_e: [S; 3],
///         _coeff: &CoeffEval,
///     ) -> [S; 3] {
///         let gx = (0..3).fold(S::from_f64(0.0), |a, i| a + S::from_f64(elem.grad[i][0]) * u_e[i]);
///         let gy = (0..3).fold(S::from_f64(0.0), |a, i| a + S::from_f64(elem.grad[i][1]) * u_e[i]);
///         std::array::from_fn(|i| {
///             (gx * S::from_f64(elem.grad[i][0]) + gy * S::from_f64(elem.grad[i][1]))
///                 * S::from_f64(elem.area)
///         })
///     }
/// }
///
/// let mesh = FemMesh2 {
///     xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
///     tri: vec![[0, 1, 2]],
///     elem_region: vec![Symbol::new("air")],
///     edge_boundary: Vec::new(),
/// };
/// let elem = ElementGeom::from_mesh(&mesh, [0, 1, 2]).unwrap();
/// let material = Material {
///     name: Symbol::new("air"),
///     mu_r: None,
///     nu_of_b2: None,
///     epsilon_r: None,
///     sigma: None,
///     thermal_k: None,
///     heat_source: None,
///     remanence: None,
/// };
/// let coeff = CoeffEval {
///     region: Symbol::new("air"),
///     material,
///     params: ParamSet::default(),
///     epsilon_r: 1.0,
///     sigma: 0.0,
///     thermal_k: 1.0,
///     mu_r: 1.0,
///     source_density: 0.0,
///     frequency_hz: 0.0,
///     nonlinear_bh: false,
/// };
/// // A constant field has zero residual.
/// let r = Laplacian.element_residual(&elem, [2.0_f64, 2.0, 2.0], &coeff);
/// assert!(r.iter().all(|v| v.abs() < 1.0e-12));
/// ```
pub trait PhysicsFront: Send + Sync + 'static {
    /// Physics kind this front models.
    fn kind(&self) -> sim_lib_femm_core::PhysicsKind;

    /// Element residual at nodal values `u_e` for one linear triangle.
    ///
    /// Returned as a generic [`Scalarish`] so assembly can pass dual numbers and
    /// recover the local stiffness block by automatic differentiation.
    fn element_residual<S: Scalarish>(
        &self,
        elem: &ElementGeom,
        u_e: [S; 3],
        coeff: &CoeffEval,
    ) -> [S; 3];

    /// Per-node source contribution subtracted from the residual.
    ///
    /// Defaults to zero; physics with a body source override it.
    fn source_term(&self, _elem: &ElementGeom, _coeff: &CoeffEval) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }

    /// Reject coefficients this front cannot model before assembly proceeds.
    ///
    /// [`element_residual`](Self::element_residual) returns a bare scalar array
    /// and so cannot signal an unsupported material; this hook is assembly's
    /// fail-closed gate. A front that solves only the linear regime overrides it
    /// to return an [`FemmError`] when, for example, the region material carries
    /// a nonlinear B-H curve ([`CoeffEval::nonlinear_bh`]) or a nonzero
    /// excitation frequency it cannot represent. Defaults to accepting every
    /// coefficient.
    fn validate_coeff(&self, _coeff: &CoeffEval) -> FemmResult<()> {
        Ok(())
    }
}

/// The global linear system produced by [`assemble_system`].
///
/// Holds the assembled stiffness matrix, the load (residual) vector, and the
/// node-to-degree-of-freedom map, ready for a solver in `sim-lib-femm-solve`.
#[derive(Clone, Debug)]
pub struct AssembledSystem {
    /// Global stiffness matrix in compressed sparse row form.
    pub k: CsrMatrix,
    /// Right-hand side / residual vector, one entry per node.
    pub r: Vec<f64>,
    /// Degree-of-freedom index for each mesh node, or `None` if eliminated.
    pub dof_of_node: Vec<Option<usize>>,
}

/// Assemble the global stiffness matrix and load vector for a meshed model.
///
/// Walks every mesh triangle, resolves its region coefficients, evaluates the
/// `front` element residual, scatters it into the global system, applies
/// Dirichlet conditions, and enforces the [`FemmLimits`] element/nnz/wall-clock
/// budgets. The kernel supplies the evaluation context [`Cx`]; the linear
/// algebra and number domains come from sim-numbers.
pub fn assemble_system<F: PhysicsFront>(
    cx: &mut Cx,
    front: &F,
    model: &FemmModel,
    meshed: &MeshedModel,
    limits: &FemmLimits,
) -> FemmResult<AssembledSystem> {
    let started = Instant::now();
    if meshed.mesh.tri.len() > limits.max_elements {
        return Err(FemmError::MeshLimitExceeded(format!(
            "elements {} > {}",
            meshed.mesh.tri.len(),
            limits.max_elements
        )));
    }
    meshed.validate_against(model)?;
    require_supported_boundaries(model)?;
    let n = meshed.mesh.xy.len();
    let mut dense = vec![vec![0.0; n]; n];
    let mut residual = vec![0.0; n];
    for (elem_index, tri) in meshed.mesh.tri.iter().copied().enumerate() {
        if started.elapsed().as_millis() as u64 > limits.max_wall_ms {
            return Err(FemmError::BudgetExceeded(
                "assembly wall clock exceeded".to_owned(),
            ));
        }
        let elem = ElementGeom::from_mesh(&meshed.mesh, tri)?;
        let region = meshed
            .mesh
            .elem_region
            .get(elem_index)
            .cloned()
            .ok_or_else(|| {
                FemmError::InvalidGeometry(format!("element {elem_index} has no region label"))
            })?;
        let material = model
            .material_for_region(&region)
            .cloned()
            .ok_or_else(|| FemmError::MissingMaterial(region.to_string()))?;
        let coeff = coeff_eval(cx, model, &meshed.params, region, material)?;
        front.validate_coeff(&coeff)?;
        let weight = elem.axisymmetric_weight(&model.formulation)?;
        let u_e = [
            Dual::<3>::var(0.0, 0),
            Dual::<3>::var(0.0, 1),
            Dual::<3>::var(0.0, 2),
        ];
        let local = front.element_residual(&elem, u_e, &coeff);
        let source = front.source_term(&elem, &coeff);
        let ids = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        for local_row in 0..3 {
            residual[ids[local_row]] += weight * (local[local_row].v - source[local_row]);
            for local_col in 0..3 {
                dense[ids[local_row]][ids[local_col]] += weight * local[local_row].d[local_col];
            }
        }
    }
    apply_dirichlet_conditions(model, meshed, &mut dense, &mut residual)?;
    let nnz = dense
        .iter()
        .map(|row| row.iter().filter(|value| value.abs() > 0.0).count())
        .sum::<usize>();
    if nnz > limits.max_nnz {
        return Err(FemmError::MeshLimitExceeded(format!(
            "nnz {nnz} > {}",
            limits.max_nnz
        )));
    }
    Ok(AssembledSystem {
        k: dense_to_csr(&dense)?,
        r: residual,
        dof_of_node: (0..n).map(Some).collect(),
    })
}

fn coeff_eval(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    region: Symbol,
    material: Material,
) -> FemmResult<CoeffEval> {
    let nonlinear_bh = material.nu_of_b2.is_some();
    Ok(CoeffEval {
        epsilon_r: eval_optional(cx, material.epsilon_r.as_ref(), params, 1.0)?,
        sigma: eval_optional(cx, material.sigma.as_ref(), params, 0.0)?,
        thermal_k: eval_optional(cx, material.thermal_k.as_ref(), params, 1.0)?,
        mu_r: eval_optional(cx, material.mu_r.as_ref(), params, 1.0)?,
        source_density: source_density(cx, model, params, &region)?,
        frequency_hz: eval_optional(cx, model.frequency_hz.as_ref(), params, 0.0)?,
        nonlinear_bh,
        region,
        material,
        params: params.clone(),
    })
}

fn eval_optional(
    cx: &mut Cx,
    expr: Option<&sim_kernel::Expr>,
    params: &ParamSet,
    default: f64,
) -> FemmResult<f64> {
    expr.map(|expr| eval_expr_f64(cx, expr, params, &[]))
        .transpose()?
        .map_or(Ok(default), Ok)
}

fn source_density(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    region: &Symbol,
) -> FemmResult<f64> {
    model.sources.iter().try_fold(0.0, |acc, source| {
        let contribution = match source {
            sim_lib_femm_material::Source::CurrentDensity { region: src, value }
            | sim_lib_femm_material::Source::ChargeDensity { region: src, value }
            | sim_lib_femm_material::Source::HeatSource { region: src, value }
                if src == region =>
            {
                eval_expr_f64(cx, value, params, &[])?
            }
            sim_lib_femm_material::Source::CircuitCoil {
                region: src,
                turns,
                current,
                ..
            } if src == region => {
                let value = eval_expr_f64(cx, turns, params, &[])?
                    * eval_expr_f64(cx, current, params, &[])?;
                if !value.is_finite() {
                    return Err(FemmError::InvalidGeometry(
                        "non-finite circuit coil source".to_owned(),
                    ));
                }
                value
            }
            _ => 0.0,
        };
        Ok(acc + contribution)
    })
}

fn dense_to_csr(dense: &[Vec<f64>]) -> FemmResult<CsrMatrix> {
    let mut rowptr = Vec::with_capacity(dense.len() + 1);
    let mut colind = Vec::new();
    let mut vals = Vec::new();
    rowptr.push(0);
    for row in dense {
        for (col, value) in row.iter().copied().enumerate() {
            if value.abs() > 1.0e-14 {
                colind.push(col);
                vals.push(value);
            }
        }
        rowptr.push(colind.len());
    }
    CsrMatrix::new(rowptr, colind, vals)
}

fn require_supported_boundaries(model: &FemmModel) -> FemmResult<()> {
    for boundary in &model.boundaries {
        require_supported_boundary(boundary)?;
    }
    Ok(())
}

fn require_supported_boundary(boundary: &Boundary) -> FemmResult<()> {
    if boundary.kind == BoundaryKind::Dirichlet {
        Ok(())
    } else {
        Err(FemmError::InvalidGeometry(format!(
            "unsupported boundary kind {}",
            boundary.kind
        )))
    }
}

fn apply_dirichlet_conditions(
    model: &FemmModel,
    meshed: &MeshedModel,
    dense: &mut [Vec<f64>],
    residual: &mut [f64],
) -> FemmResult<()> {
    for boundary in &model.boundaries {
        require_supported_boundary(boundary)?;
        for (a, b, name) in &meshed.mesh.edge_boundary {
            if name != &boundary.name {
                continue;
            }
            for node in [*a as usize, *b as usize] {
                if node >= dense.len() {
                    return Err(FemmError::InvalidGeometry(format!(
                        "boundary edge node index {node} out of range for {} mesh nodes",
                        dense.len()
                    )));
                }
                let value = boundary_value(boundary, &meshed.params)?;
                for row in 0..dense.len() {
                    if row != node {
                        residual[row] -= dense[row][node] * value;
                    }
                }
                dense[node].fill(0.0);
                for row in dense.iter_mut() {
                    row[node] = 0.0;
                }
                dense[node][node] = 1.0;
                residual[node] = value;
            }
        }
    }
    Ok(())
}

fn boundary_value(
    boundary: &sim_lib_femm_material::Boundary,
    params: &ParamSet,
) -> FemmResult<f64> {
    let mut cx = Cx::new(
        std::sync::Arc::new(sim_kernel::EagerPolicy),
        std::sync::Arc::new(sim_kernel::DefaultFactory),
    );
    eval_expr_f64(&mut cx, &boundary.value, params, &[])
}

#[cfg(test)]
#[path = "implementation_tests.rs"]
mod tests;
