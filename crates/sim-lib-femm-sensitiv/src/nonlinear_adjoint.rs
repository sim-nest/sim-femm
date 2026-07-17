use std::sync::Arc;

use sim_kernel::{Cx, Symbol};
use sim_lib_femm_core::{FemmError, FemmLimits, FemmResult, ParamSet, value_as_f64};
use sim_lib_femm_function::{ModelCallable, resolve_excitation};
use sim_lib_femm_geometry::eval_expr_f64;
use sim_lib_femm_material::BoundaryKind;
use sim_lib_femm_mesh::{DeterministicMesher, FemmModel, Mesher};
use sim_lib_femm_post::{Excitation, FemmSolution, QuantitySpec, quantity};
use sim_lib_femm_solve::{DenseFallbackSolver, GradientTrust, SteadySolve, solve_steady};
use sim_lib_femm_space::ElementGeom;

const ADJOINT_VERIFY_TOL: f64 = 1.0e-4;
const FD_STEP_SCALE: f64 = 1.490_116_119_384_765_6e-8;

pub(crate) struct ParamJacobian {
    columns: Vec<ParamJacobianColumn>,
}

struct ParamJacobianColumn {
    step: f64,
    plus: Arc<FemmSolution>,
    minus: Arc<FemmSolution>,
    residual_column: Option<Vec<f64>>,
}

impl ParamJacobian {
    fn quantity_gradient(
        &self,
        spec: &QuantitySpec,
        excitation: &Excitation,
    ) -> FemmResult<Vec<f64>> {
        self.columns
            .iter()
            .map(|column| {
                let q_plus = quantity(&column.plus, spec, excitation)?;
                let q_minus = quantity(&column.minus, spec, excitation)?;
                let value = (q_plus - q_minus) / (2.0 * column.step);
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(FemmError::SensitivityUnavailable(
                        "nonlinear finite difference produced non-finite gradient".to_owned(),
                    ))
                }
            })
            .collect()
    }
}

pub(crate) fn nonlinear_adjoint_gradient(
    cx: &mut Cx,
    callable: &ModelCallable,
    solve: &SteadySolve,
    params: &ParamSet,
    quantity_spec: &QuantitySpec,
    wrt: &[Symbol],
) -> FemmResult<(Vec<f64>, GradientTrust)> {
    if solve.solution.diagnostics.method != Symbol::new("femm-ptc") {
        return Err(FemmError::SensitivityUnavailable(
            "nonlinear adjoint requested for a non-PTC solve".to_owned(),
        ));
    }
    let jacobian = assemble_param_jacobian(cx, callable, solve, params, wrt)?;
    let excitation = resolve_excitation(cx, &callable.model, params, quantity_spec)?;
    if let Some(gradient) = exact_nonlinear_adjoint(solve, quantity_spec, &jacobian, &excitation)? {
        return Ok((
            gradient,
            GradientTrust::AdjointVerified {
                tol: ADJOINT_VERIFY_TOL,
            },
        ));
    }
    Ok((
        jacobian.quantity_gradient(quantity_spec, &excitation)?,
        GradientTrust::FiniteDifferenceOnly,
    ))
}

pub(crate) fn assemble_param_jacobian(
    cx: &mut Cx,
    callable: &ModelCallable,
    solve: &SteadySolve,
    params: &ParamSet,
    wrt: &[Symbol],
) -> FemmResult<ParamJacobian> {
    let mut columns = Vec::with_capacity(wrt.len());
    for symbol in wrt {
        let base = params
            .get(symbol)
            .ok_or_else(|| FemmError::UnknownFemmParameter(symbol.to_string()))
            .and_then(|value| value_as_f64(cx, value))?;
        let step = fd_step(base);
        let plus_params = replace_param(cx, params, symbol, base + step)?;
        let minus_params = replace_param(cx, params, symbol, base - step)?;
        let residual_column = residual_column(
            cx,
            &callable.model,
            solve,
            &plus_params,
            &minus_params,
            step,
        );
        let plus = solve_steady(
            cx,
            &callable.model,
            &plus_params,
            &FemmLimits::default(),
            None,
        )?;
        let minus = solve_steady(
            cx,
            &callable.model,
            &minus_params,
            &FemmLimits::default(),
            None,
        )?;
        columns.push(ParamJacobianColumn {
            step,
            plus: plus.solution,
            minus: minus.solution,
            residual_column,
        });
    }
    Ok(ParamJacobian { columns })
}

fn exact_nonlinear_adjoint(
    solve: &SteadySolve,
    quantity_spec: &QuantitySpec,
    jacobian: &ParamJacobian,
    excitation: &Excitation,
) -> FemmResult<Option<Vec<f64>>> {
    let Some(dq_du) = quantity_state_derivative(&solve.solution, quantity_spec)? else {
        return Ok(None);
    };
    let transpose = transpose(&solve.factor.dense);
    let Ok(lambda) = solve_dense_regularized(&transpose, &dq_du) else {
        return Ok(None);
    };
    let mut gradient = Vec::with_capacity(jacobian.columns.len());
    for column in &jacobian.columns {
        let Some(residual_column) = &column.residual_column else {
            return Ok(None);
        };
        gradient.push(
            -lambda
                .iter()
                .zip(residual_column)
                .map(|(left, right)| left * right)
                .sum::<f64>(),
        );
    }
    let finite_difference = jacobian.quantity_gradient(quantity_spec, excitation)?;
    if verified_against_fd(&gradient, &finite_difference) {
        Ok(Some(gradient))
    } else {
        Ok(None)
    }
}

pub(crate) fn quantity_state_derivative(
    solution: &FemmSolution,
    quantity_spec: &QuantitySpec,
) -> FemmResult<Option<Vec<f64>>> {
    match quantity_spec {
        QuantitySpec::Energy { region } | QuantitySpec::Coenergy { region } => {
            energy_state_derivative(solution, region.as_ref()).map(Some)
        }
        _ => Ok(None),
    }
}

fn energy_state_derivative(
    solution: &FemmSolution,
    region: Option<&Symbol>,
) -> FemmResult<Vec<f64>> {
    let mut derivative = vec![0.0; solution.u.len()];
    for (elem_index, tri) in solution.mesh.tri.iter().enumerate() {
        if !region_matches(solution, elem_index, region) {
            continue;
        }
        let geom = sim_lib_femm_space::ElementGeom::from_mesh(&solution.mesh, *tri)?;
        let measure = geom.area * geom.axisymmetric_weight(&solution.formulation)?;
        let mean = (solution.u[tri[0] as usize]
            + solution.u[tri[1] as usize]
            + solution.u[tri[2] as usize])
            / 3.0;
        for node in tri {
            derivative[*node as usize] += measure * mean / 3.0;
        }
    }
    Ok(derivative)
}

fn region_matches(solution: &FemmSolution, index: usize, region: Option<&Symbol>) -> bool {
    region.is_none_or(|region| solution.mesh.elem_region.get(index) == Some(region))
}

fn residual_column(
    cx: &mut Cx,
    model: &FemmModel,
    solve: &SteadySolve,
    plus_params: &ParamSet,
    minus_params: &ParamSet,
    step: f64,
) -> Option<Vec<f64>> {
    let plus = nonlinear_residual_at(cx, model, plus_params, &solve.solution.u).ok()?;
    let minus = nonlinear_residual_at(cx, model, minus_params, &solve.solution.u).ok()?;
    if plus.len() != minus.len() {
        return None;
    }
    Some(
        plus.iter()
            .zip(&minus)
            .map(|(plus, minus)| (plus - minus) / (2.0 * step))
            .collect(),
    )
}

fn nonlinear_residual_at(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    u: &[f64],
) -> FemmResult<Vec<f64>> {
    let meshed = DeterministicMesher::new().mesh(cx, model, params)?;
    if meshed.mesh.xy.len() != u.len() {
        return Err(FemmError::SensitivityUnavailable(
            "nonlinear residual perturbation changed the solution dimension".to_owned(),
        ));
    }
    let mut residual = vec![0.0; u.len()];
    for (elem_index, tri) in meshed.mesh.tri.iter().copied().enumerate() {
        let geom = ElementGeom::from_mesh(&meshed.mesh, tri)?;
        let region = meshed
            .mesh
            .elem_region
            .get(elem_index)
            .cloned()
            .ok_or_else(|| {
                FemmError::InvalidGeometry(format!("element {elem_index} has no region label"))
            })?;
        let mu_r = region_mu_r(cx, model, params, &region)?;
        let ids = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let local = nonlinear_element_residual(&geom, [u[ids[0]], u[ids[1]], u[ids[2]]], mu_r);
        let weight = geom.axisymmetric_weight(&model.formulation)?;
        for local_row in 0..3 {
            residual[ids[local_row]] += weight * local[local_row];
        }
    }
    apply_dirichlet_residual(
        cx,
        model,
        params,
        &meshed.mesh.edge_boundary,
        u,
        &mut residual,
    )?;
    Ok(residual)
}

fn region_mu_r(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    region: &Symbol,
) -> FemmResult<f64> {
    let material = model
        .material_for_region(region)
        .ok_or_else(|| FemmError::MissingMaterial(region.to_string()))?;
    material
        .mu_r
        .as_ref()
        .map(|expr| eval_expr_f64(cx, expr, params, &[]))
        .transpose()
        .map(|value| value.unwrap_or(1.0))
}

fn nonlinear_element_residual(elem: &ElementGeom, u_e: [f64; 3], mu_r: f64) -> [f64; 3] {
    let grad_u = [
        elem.grad
            .iter()
            .zip(u_e)
            .map(|(grad, u)| grad[0] * u)
            .sum::<f64>(),
        elem.grad
            .iter()
            .zip(u_e)
            .map(|(grad, u)| grad[1] * u)
            .sum::<f64>(),
    ];
    let b2 = grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1];
    let reluctivity = 1.0 / mu_r.max(1.0e-12) + b2 * 0.02;
    std::array::from_fn(|index| {
        let dot = grad_u[0] * elem.grad[index][0] + grad_u[1] * elem.grad[index][1];
        dot * reluctivity * elem.area
    })
}

fn apply_dirichlet_residual(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    edges: &[(u32, u32, Symbol)],
    u: &[f64],
    residual: &mut [f64],
) -> FemmResult<()> {
    for boundary in &model.boundaries {
        if boundary.kind != BoundaryKind::Dirichlet {
            continue;
        }
        let value = eval_expr_f64(cx, &boundary.value, params, &[])?;
        for (a, b, name) in edges {
            if name != &boundary.name {
                continue;
            }
            for node in [*a as usize, *b as usize] {
                if node >= residual.len() {
                    return Err(FemmError::InvalidGeometry(format!(
                        "boundary edge node index {node} out of range for {} mesh nodes",
                        residual.len()
                    )));
                }
                residual[node] = u[node] + value;
            }
        }
    }
    Ok(())
}

fn verified_against_fd(adjoint: &[f64], finite_difference: &[f64]) -> bool {
    adjoint.len() == finite_difference.len()
        && adjoint
            .iter()
            .zip(finite_difference)
            .all(|(adjoint, fd)| (adjoint - fd).abs() <= ADJOINT_VERIFY_TOL * fd.abs().max(1.0))
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

fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let mut out = vec![vec![0.0; matrix.len()]; matrix[0].len()];
    for (row, values) in matrix.iter().enumerate() {
        for (col, value) in values.iter().enumerate() {
            out[col][row] = *value;
        }
    }
    out
}

fn replace_param(
    cx: &mut Cx,
    params: &ParamSet,
    symbol: &Symbol,
    value: f64,
) -> FemmResult<ParamSet> {
    let mut entries = params.entries.clone();
    let replacement = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
        .map_err(|err| FemmError::SensitivityUnavailable(err.to_string()))?;
    if let Some((_, current)) = entries.iter_mut().find(|(name, _)| name == symbol) {
        *current = replacement;
    } else {
        entries.push((symbol.clone(), replacement));
    }
    Ok(ParamSet::new(entries))
}

fn fd_step(base: f64) -> f64 {
    FD_STEP_SCALE * base.abs().max(1.0)
}
