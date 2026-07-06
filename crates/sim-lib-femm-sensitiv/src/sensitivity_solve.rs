use sim_kernel::{Cx, Symbol};
use sim_lib_femm_core::{FemmError, FemmLimits, FemmResult, Formulation, ParamSet, PhysicsKind};
use sim_lib_femm_function::{ModelCallable, resolve_excitation};
use sim_lib_femm_material::{Boundary, BoundaryKind, Material, Source};
use sim_lib_femm_mesh::{FemMesh2, FemmModel};
use sim_lib_femm_post::QuantitySpec;
use sim_lib_femm_solve::{DenseFallbackSolver, solve_steady};
use sim_lib_numbers_ad::Dual;

use crate::implementation::{eval_expr_dual, excitation_uses_symbol};
use crate::sensitivity_mesh::differentiated_mesh;
use crate::sensitivity_quantity::quantity_derivative;
use crate::sensitivity_types::{DiffMesh, DiffSolution, DualGeom};

struct CoeffDual {
    epsilon_r: Dual<1>,
    sigma: Dual<1>,
    thermal_k: Dual<1>,
    mu_r: Dual<1>,
    source_density: Dual<1>,
    frequency_hz: Dual<1>,
}

struct AssemblyDerivative {
    dense: Vec<Vec<f64>>,
    ddense: Vec<Vec<f64>>,
    residual: Vec<f64>,
    dresidual: Vec<f64>,
}

impl AssemblyDerivative {
    fn new(size: usize) -> Self {
        Self {
            dense: vec![vec![0.0; size]; size],
            ddense: vec![vec![0.0; size]; size],
            residual: vec![0.0; size],
            dresidual: vec![0.0; size],
        }
    }
}

pub(crate) fn built_in_quantity_gradient(
    cx: &mut Cx,
    callable: &ModelCallable,
    spec: &QuantitySpec,
    params: ParamSet,
    wrt: &[Symbol],
) -> FemmResult<Vec<(Symbol, f64)>> {
    let params = resolve_params(&callable.model, params);
    let excitation = resolve_excitation(cx, &callable.model, &params, spec)?;
    wrt.iter()
        .map(|symbol| {
            if excitation_uses_symbol(&callable.model, spec, symbol) {
                // The exact analytic derivative assumes a parameter-independent
                // drive; when the excitation itself depends on `symbol`, fail
                // closed so the caller falls back to the (correct) finite-
                // difference path rather than silently dropping the dI/dp term.
                return Err(FemmError::SensitivityUnavailable(format!(
                    "excitation depends on design parameter {symbol}"
                )));
            }
            let diff = differentiate_solution(cx, &callable.model, &params, symbol)?;
            quantity_derivative(cx, &diff, spec, &excitation).map(|value| (symbol.clone(), value))
        })
        .collect()
}

fn resolve_params(model: &FemmModel, params: ParamSet) -> ParamSet {
    let mut entries = params.entries;
    for input in &model.inputs {
        if entries.iter().all(|(name, _)| name != &input.name)
            && let Some(default) = &input.default
        {
            entries.push((input.name.clone(), default.clone()));
        }
    }
    ParamSet::new(entries)
}

fn differentiate_solution(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    wrt: &Symbol,
) -> FemmResult<DiffSolution> {
    let limits = FemmLimits::default();
    let solved = solve_steady(cx, model, params, &limits, None)?;
    let diff_mesh = differentiated_mesh(cx, model, params, wrt)?;
    let assembly = assemble_derivative(cx, model, params, wrt, &diff_mesh)?;
    let u = &solved.solution.u;
    let rhs = (0..u.len())
        .map(|row| {
            let dk_u = assembly.ddense[row]
                .iter()
                .zip(u)
                .map(|(k, u)| k * u)
                .sum::<f64>();
            -(assembly.dresidual[row] + dk_u)
        })
        .collect::<Vec<_>>();
    let du = solve_dense_regularized(&assembly.dense, &rhs)?;
    Ok(DiffSolution {
        solution: solved.solution,
        du,
        dxy: diff_mesh.dxy,
    })
}

fn assemble_derivative(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    wrt: &Symbol,
    diff_mesh: &DiffMesh,
) -> FemmResult<AssemblyDerivative> {
    let n = diff_mesh.mesh.xy.len();
    let mut assembly = AssemblyDerivative::new(n);
    for (elem_index, tri) in diff_mesh.mesh.tri.iter().copied().enumerate() {
        let elem = dual_geom(diff_mesh, tri)?;
        let region = diff_mesh
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
        let coeff = coeff_eval_dual(cx, model, params, wrt, &region, &material)?;
        let measure = elem.area * axisymmetric_weight(&elem, &model.formulation)?;
        let stiffness = physics_coeff(model.physics.clone(), &coeff);
        let ids = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        for local_row in 0..3 {
            let source = measure * coeff.source_density / Dual::cst(3.0);
            assembly.residual[ids[local_row]] -= source.v;
            assembly.dresidual[ids[local_row]] -= source.d[0];
            for local_col in 0..3 {
                let dot = elem.grad[local_row][0] * elem.grad[local_col][0]
                    + elem.grad[local_row][1] * elem.grad[local_col][1];
                let value = measure * stiffness * dot;
                assembly.dense[ids[local_row]][ids[local_col]] += value.v;
                assembly.ddense[ids[local_row]][ids[local_col]] += value.d[0];
            }
        }
    }
    apply_dirichlet(cx, model, params, wrt, &diff_mesh.mesh, &mut assembly)?;
    Ok(assembly)
}

fn coeff_eval_dual(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    wrt: &Symbol,
    region: &Symbol,
    material: &Material,
) -> FemmResult<CoeffDual> {
    Ok(CoeffDual {
        epsilon_r: eval_optional(cx, material.epsilon_r.as_ref(), params, wrt, 1.0)?,
        sigma: eval_optional(cx, material.sigma.as_ref(), params, wrt, 0.0)?,
        thermal_k: eval_optional(cx, material.thermal_k.as_ref(), params, wrt, 1.0)?,
        mu_r: eval_optional(cx, material.mu_r.as_ref(), params, wrt, 1.0)?,
        source_density: source_density(cx, model, params, wrt, region)?,
        frequency_hz: eval_optional(cx, model.frequency_hz.as_ref(), params, wrt, 0.0)?,
    })
}

fn eval_optional(
    cx: &mut Cx,
    expr: Option<&sim_kernel::Expr>,
    params: &ParamSet,
    wrt: &Symbol,
    default: f64,
) -> FemmResult<Dual<1>> {
    expr.map(|expr| eval_expr_dual(cx, expr, params, Some(wrt), &[]))
        .transpose()
        .map(|value| value.unwrap_or_else(|| Dual::cst(default)))
}

fn source_density(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    wrt: &Symbol,
    region: &Symbol,
) -> FemmResult<Dual<1>> {
    model
        .sources
        .iter()
        .try_fold(Dual::cst(0.0), |acc, source| {
            let contribution = match source {
                Source::CurrentDensity { region: src, value }
                | Source::ChargeDensity { region: src, value }
                | Source::HeatSource { region: src, value }
                    if src == region =>
                {
                    eval_expr_dual(cx, value, params, Some(wrt), &[])?
                }
                Source::CircuitCoil {
                    region: src,
                    turns,
                    current,
                    ..
                } if src == region => {
                    eval_expr_dual(cx, turns, params, Some(wrt), &[])?
                        * eval_expr_dual(cx, current, params, Some(wrt), &[])?
                }
                _ => Dual::cst(0.0),
            };
            Ok(acc + contribution)
        })
}

fn physics_coeff(physics: PhysicsKind, coeff: &CoeffDual) -> Dual<1> {
    match physics {
        PhysicsKind::Electrostatic => coeff.epsilon_r,
        PhysicsKind::HeatSteady => coeff.thermal_k,
        PhysicsKind::CurrentSteady => floor_dual(coeff.sigma, 1.0e-12),
        PhysicsKind::Magnetostatic => Dual::cst(1.0) / floor_dual(coeff.mu_r, 1.0e-12),
        PhysicsKind::MagneticsHarmonic => {
            Dual::cst(1.0) / floor_dual(coeff.mu_r, 1.0e-12)
                + abs_dual(coeff.sigma) * abs_dual(coeff.frequency_hz)
        }
    }
}

fn apply_dirichlet(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    wrt: &Symbol,
    mesh: &FemMesh2,
    assembly: &mut AssemblyDerivative,
) -> FemmResult<()> {
    for boundary in &model.boundaries {
        if boundary.kind != BoundaryKind::Dirichlet {
            continue;
        }
        let value = boundary_value(cx, boundary, params, wrt)?;
        for (a, b, name) in &mesh.edge_boundary {
            if name != &boundary.name {
                continue;
            }
            for node in [*a as usize, *b as usize] {
                for row in 0..assembly.dense.len() {
                    if row != node {
                        assembly.dresidual[row] -= assembly.ddense[row][node] * value.v
                            + assembly.dense[row][node] * value.d[0];
                        assembly.residual[row] -= assembly.dense[row][node] * value.v;
                    }
                }
                assembly.dense[node].fill(0.0);
                assembly.ddense[node].fill(0.0);
                for row in 0..assembly.dense.len() {
                    assembly.dense[row][node] = 0.0;
                    assembly.ddense[row][node] = 0.0;
                }
                assembly.dense[node][node] = 1.0;
                assembly.residual[node] = value.v;
                assembly.dresidual[node] = value.d[0];
            }
        }
    }
    Ok(())
}

fn boundary_value(
    cx: &mut Cx,
    boundary: &Boundary,
    params: &ParamSet,
    wrt: &Symbol,
) -> FemmResult<Dual<1>> {
    eval_expr_dual(cx, &boundary.value, params, Some(wrt), &[])
}

pub(crate) fn dual_geom(diff_mesh: &DiffMesh, tri: [u32; 3]) -> FemmResult<DualGeom> {
    let xy = std::array::from_fn(|local| {
        let index = tri[local] as usize;
        [
            Dual {
                v: diff_mesh.mesh.xy[index][0],
                d: [diff_mesh.dxy[index][0]],
            },
            Dual {
                v: diff_mesh.mesh.xy[index][1],
                d: [diff_mesh.dxy[index][1]],
            },
        ]
    });
    let area2 = (xy[1][0] - xy[0][0]) * (xy[2][1] - xy[0][1])
        - (xy[2][0] - xy[0][0]) * (xy[1][1] - xy[0][1]);
    let area = abs_dual(area2) * Dual::cst(0.5);
    if area.v <= f64::EPSILON {
        return Err(FemmError::InvalidGeometry("degenerate triangle".to_owned()));
    }
    let denom = area * Dual::cst(2.0);
    let grad = [
        [(xy[1][1] - xy[2][1]) / denom, (xy[2][0] - xy[1][0]) / denom],
        [(xy[2][1] - xy[0][1]) / denom, (xy[0][0] - xy[2][0]) / denom],
        [(xy[0][1] - xy[1][1]) / denom, (xy[1][0] - xy[0][0]) / denom],
    ];
    Ok(DualGeom { xy, area, grad })
}

pub(crate) fn axisymmetric_weight(
    elem: &DualGeom,
    formulation: &Formulation,
) -> FemmResult<Dual<1>> {
    match formulation {
        Formulation::Planar => Ok(Dual::cst(1.0)),
        Formulation::Axisymmetric => {
            let r = (elem.xy[0][0] + elem.xy[1][0] + elem.xy[2][0]) / Dual::cst(3.0);
            if r.v < 0.0 {
                return Err(FemmError::InvalidGeometry("axisymmetric r < 0".to_owned()));
            }
            Ok(r * Dual::cst(2.0 * std::f64::consts::PI))
        }
    }
}

fn solve_dense_regularized(matrix: &[Vec<f64>], rhs: &[f64]) -> FemmResult<Vec<f64>> {
    match DenseFallbackSolver::dense_solve(matrix, rhs) {
        Ok(out) => Ok(out),
        Err(FemmError::SolveDidNotConverge(_)) => {
            let mut shifted = matrix.to_vec();
            for (index, row) in shifted.iter_mut().enumerate() {
                row[index] += 1.0e-9;
            }
            DenseFallbackSolver::dense_solve(&shifted, rhs)
        }
        Err(err) => Err(err),
    }
}

fn abs_dual(value: Dual<1>) -> Dual<1> {
    if value.v > 0.0 {
        value
    } else if value.v < 0.0 {
        -value
    } else {
        Dual::cst(0.0)
    }
}

fn floor_dual(value: Dual<1>, floor: f64) -> Dual<1> {
    if value.v > floor {
        value
    } else {
        Dual::cst(floor)
    }
}
