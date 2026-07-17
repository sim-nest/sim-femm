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
use sim_lib_femm_material::{BoundaryKind, Material};
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
            .unwrap_or_else(|| Symbol::new("region"));
        let material = model
            .materials
            .iter()
            .find(|material| material.name == region)
            .cloned()
            .or_else(|| model.materials.first().cloned())
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

fn apply_dirichlet_conditions(
    model: &FemmModel,
    meshed: &MeshedModel,
    dense: &mut [Vec<f64>],
    residual: &mut [f64],
) -> FemmResult<()> {
    for boundary in &model.boundaries {
        if boundary.kind != BoundaryKind::Dirichlet {
            continue;
        }
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
mod tests {
    use std::sync::Arc;

    use sim_kernel::{DefaultFactory, EagerPolicy};
    use sim_kernel::{Expr, Symbol};
    use sim_lib_femm_core::{Formulation, LengthUnit, ParamRole, ParamSpec, PhysicsKind, StableId};
    use sim_lib_femm_geometry::{BlockLabel2, Geometry2, dummy_origin};
    use sim_lib_femm_material::{Boundary, BoundaryKind, Material, MeshPolicy};
    use sim_lib_femm_mesh::{FemMesh2, FemmModel, MeshedModel};

    use super::*;

    fn test_cx() -> Cx {
        Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory))
    }

    struct PoissonFront;

    impl PhysicsFront for PoissonFront {
        fn kind(&self) -> sim_lib_femm_core::PhysicsKind {
            PhysicsKind::Electrostatic
        }

        fn element_residual<S: Scalarish>(
            &self,
            elem: &ElementGeom,
            u_e: [S; 3],
            _coeff: &CoeffEval,
        ) -> [S; 3] {
            let grad_u = [
                elem.grad
                    .iter()
                    .zip(u_e)
                    .map(|(grad, u)| S::from_f64(grad[0]) * u)
                    .fold(S::from_f64(0.0), |acc, x| acc + x),
                elem.grad
                    .iter()
                    .zip(u_e)
                    .map(|(grad, u)| S::from_f64(grad[1]) * u)
                    .fold(S::from_f64(0.0), |acc, x| acc + x),
            ];
            std::array::from_fn(|i| {
                let dot = grad_u[0] * S::from_f64(elem.grad[i][0])
                    + grad_u[1] * S::from_f64(elem.grad[i][1]);
                dot * S::from_f64(elem.area)
            })
        }
    }

    fn num(text: &str) -> Expr {
        sim_value::build::num_q(Some("numbers"), "f64", text)
    }

    fn call(operator: &str, args: Vec<Expr>) -> Expr {
        Expr::Call {
            operator: Box::new(Expr::Symbol(Symbol::new(operator))),
            args,
        }
    }

    fn param(cx: &mut Cx, symbol: &str, canonical: &str) -> ParamSet {
        ParamSet::new(vec![(
            Symbol::new(symbol),
            cx.factory()
                .number_literal(Symbol::qualified("numbers", "f64"), canonical.to_owned())
                .unwrap(),
        )])
    }

    fn model() -> FemmModel {
        FemmModel {
            id: StableId(1),
            name: Symbol::new("poisson"),
            physics: PhysicsKind::Electrostatic,
            formulation: Formulation::Planar,
            length_unit: LengthUnit::Meter,
            depth: None,
            frequency_hz: None,
            inputs: vec![ParamSpec {
                name: Symbol::new("x"),
                default: None,
                unit: None,
                role: ParamRole::Design,
            }],
            geometry: Geometry2 {
                labels: vec![BlockLabel2 {
                    name: Symbol::new("air"),
                    at: [num("0.1"), num("0.1")],
                    material: Symbol::new("air"),
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
            boundaries: vec![Boundary {
                name: Symbol::new("wall"),
                kind: BoundaryKind::Dirichlet,
                value: num("0.0"),
            }],
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

    #[test]
    fn one_triangle_matrix_is_symmetric() {
        let mut cx = test_cx();
        let meshed = MeshedModel {
            model_id: StableId(1),
            params: ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: Vec::new(),
            },
            diagnostics: Vec::new(),
        };
        let assembled = assemble_system(
            &mut cx,
            &PoissonFront,
            &model(),
            &meshed,
            &FemmLimits::default(),
        )
        .unwrap();
        let dense = assembled.k.to_dense().unwrap();
        assert!((dense[0][1] - dense[1][0]).abs() < 1.0e-12);
    }

    #[test]
    fn dirichlet_elimination_pins_dof() {
        let mut cx = test_cx();
        let meshed = MeshedModel {
            model_id: StableId(1),
            params: ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: vec![(0, 1, Symbol::new("wall"))],
            },
            diagnostics: Vec::new(),
        };
        let assembled = assemble_system(
            &mut cx,
            &PoissonFront,
            &model(),
            &meshed,
            &FemmLimits::default(),
        )
        .unwrap();
        let dense = assembled.k.to_dense().unwrap();
        assert_eq!(dense[0][0], 1.0);
        assert_eq!(dense[0][1], 0.0);
    }

    struct LinearOnlyFront;

    impl PhysicsFront for LinearOnlyFront {
        fn kind(&self) -> sim_lib_femm_core::PhysicsKind {
            PhysicsKind::Magnetostatic
        }

        fn element_residual<S: Scalarish>(
            &self,
            _elem: &ElementGeom,
            _u_e: [S; 3],
            _coeff: &CoeffEval,
        ) -> [S; 3] {
            std::array::from_fn(|_| S::from_f64(0.0))
        }

        fn validate_coeff(&self, coeff: &CoeffEval) -> FemmResult<()> {
            if coeff.nonlinear_bh {
                return Err(FemmError::UnsupportedPhysics(
                    "nonlinear B-H not supported".to_owned(),
                ));
            }
            Ok(())
        }
    }

    #[test]
    fn coeff_eval_flags_nonlinear_bh_and_front_fails_closed() {
        // A material carrying `nu_of_b2` must surface as `nonlinear_bh` so a
        // linear-only front can reject it instead of silently solving linearly.
        let mut cx = test_cx();
        let mut model = model();
        model.materials[0].nu_of_b2 = Some(num("0.02"));
        let meshed = MeshedModel {
            model_id: StableId(1),
            params: ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: Vec::new(),
            },
            diagnostics: Vec::new(),
        };
        let result = assemble_system(
            &mut cx,
            &LinearOnlyFront,
            &model,
            &meshed,
            &FemmLimits::default(),
        );
        assert!(matches!(result, Err(FemmError::UnsupportedPhysics(_))));
    }

    #[test]
    fn linear_material_still_assembles() {
        let mut cx = test_cx();
        let meshed = MeshedModel {
            model_id: StableId(1),
            params: ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: Vec::new(),
            },
            diagnostics: Vec::new(),
        };
        assert!(
            assemble_system(
                &mut cx,
                &LinearOnlyFront,
                &model(),
                &meshed,
                &FemmLimits::default(),
            )
            .is_ok()
        );
    }

    #[test]
    fn out_of_range_boundary_node_errors_without_panic() {
        // A boundary edge naming a node index past the mesh must fail closed
        // rather than panic on the dense-matrix scatter.
        let mut cx = test_cx();
        let meshed = MeshedModel {
            model_id: StableId(1),
            params: ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: vec![(0, 99, Symbol::new("wall"))],
            },
            diagnostics: Vec::new(),
        };
        let result = assemble_system(
            &mut cx,
            &PoissonFront,
            &model(),
            &meshed,
            &FemmLimits::default(),
        );
        assert!(matches!(result, Err(FemmError::InvalidGeometry(_))));
    }

    #[test]
    fn invalid_boundary_number_errors_instead_of_grounding_zero() {
        let mut cx = test_cx();
        let mut bad_model = model();
        bad_model.boundaries[0].value = num("1/0");
        let meshed = MeshedModel {
            model_id: StableId(1),
            params: ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: vec![(0, 1, Symbol::new("wall"))],
            },
            diagnostics: Vec::new(),
        };
        let result = assemble_system(
            &mut cx,
            &PoissonFront,
            &bad_model,
            &meshed,
            &FemmLimits::default(),
        );
        assert!(matches!(result, Err(FemmError::InvalidGeometry(_))));
    }

    #[test]
    fn boundary_division_by_zero_errors_instead_of_grounding_zero() {
        let mut cx = test_cx();
        let mut bad_model = model();
        bad_model.boundaries[0].value = call("/", vec![num("1.0"), num("0.0")]);
        let meshed = MeshedModel {
            model_id: StableId(1),
            params: ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: vec![(0, 1, Symbol::new("wall"))],
            },
            diagnostics: Vec::new(),
        };
        let result = assemble_system(
            &mut cx,
            &PoissonFront,
            &bad_model,
            &meshed,
            &FemmLimits::default(),
        );
        assert!(matches!(result, Err(FemmError::InvalidGeometry(_))));
    }

    #[test]
    fn unknown_boundary_parameter_errors_instead_of_grounding_zero() {
        let mut cx = test_cx();
        let mut bad_model = model();
        bad_model.boundaries[0].value = Expr::Symbol(Symbol::new("missing"));
        let meshed = MeshedModel {
            model_id: StableId(1),
            params: ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: vec![(0, 1, Symbol::new("wall"))],
            },
            diagnostics: Vec::new(),
        };
        let result = assemble_system(
            &mut cx,
            &PoissonFront,
            &bad_model,
            &meshed,
            &FemmLimits::default(),
        );
        assert!(matches!(result, Err(FemmError::UnknownFemmParameter(_))));
    }

    #[test]
    fn non_finite_boundary_parameter_errors_instead_of_grounding_zero() {
        let mut cx = test_cx();
        let mut bad_model = model();
        bad_model.boundaries[0].value = Expr::Symbol(Symbol::new("x"));
        let meshed = MeshedModel {
            model_id: StableId(1),
            params: param(&mut cx, "x", "inf"),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: vec![(0, 1, Symbol::new("wall"))],
            },
            diagnostics: Vec::new(),
        };
        let result = assemble_system(
            &mut cx,
            &PoissonFront,
            &bad_model,
            &meshed,
            &FemmLimits::default(),
        );
        assert!(matches!(result, Err(FemmError::InvalidGeometry(_))));
    }

    #[test]
    fn axisymmetric_assembly_uses_radial_weight() {
        let mut cx = test_cx();
        let mut axisym = model();
        axisym.formulation = Formulation::Axisymmetric;
        let meshed = MeshedModel {
            model_id: StableId(1),
            params: ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: Vec::new(),
            },
            diagnostics: Vec::new(),
        };
        let planar = assemble_system(
            &mut cx,
            &PoissonFront,
            &model(),
            &meshed,
            &FemmLimits::default(),
        )
        .unwrap()
        .k
        .to_dense()
        .unwrap();
        let weighted = assemble_system(
            &mut cx,
            &PoissonFront,
            &axisym,
            &meshed,
            &FemmLimits::default(),
        )
        .unwrap()
        .k
        .to_dense()
        .unwrap();
        let ratio = weighted[0][0] / planar[0][0];
        let expected = 2.0 * std::f64::consts::PI * (4.0 / 3.0);
        assert!((ratio - expected).abs() < 1.0e-12);
    }
}
