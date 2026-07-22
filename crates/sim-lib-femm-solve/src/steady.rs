#![forbid(unsafe_code)]
//! The steady-state solve pipeline.
//!
//! Meshes a model, assembles the physics-specific system, and runs the linear
//! solve to produce a `FemmSolution` and its reusable factorization.

use std::sync::Arc;

use sim_kernel::{Cx, DefaultFactory, Factory, Symbol};
use sim_lib_femm_assembly::{AssembledSystem, CoeffEval, PhysicsFront, assemble_system};
use sim_lib_femm_core::{FemmError, FemmLimits, FemmResult, ParamSet, PhysicsKind, StableId};
use sim_lib_femm_flow::{FemmSolveEvent, PtcOptions, SolveDiagnostics, ptc_solve_report};
use sim_lib_femm_material::BoundaryKind;
use sim_lib_femm_mesh::{DeterministicMesher, FemmModel, Mesher, enforce_mesh_limits};
use sim_lib_femm_physics::{
    CurrentSteadyFront, ElectrostaticFront, HeatSteadyFront, MagneticsHarmonicFront,
    MagnetostaticFront,
};
use sim_lib_femm_post::FemmSolution;
use sim_lib_femm_space::ElementGeom;
use sim_lib_numbers_ad::Scalarish;

use crate::{
    DenseFallbackSolver, FactorHandle, LinearMethod, SolveCertificate,
    certificate::{make_linear_certificate, make_ptc_certificate},
    implementation::dense_residual_norm,
};

const DIRECT_RESIDUAL_TOL: f64 = 1.0e-8;

/// The result of a steady-state solve: the solution and its factorization.
///
/// The [`FactorHandle`] can be threaded back into [`solve_steady`] to reuse the
/// factorization when the assembled matrix is unchanged.
pub struct SteadySolve {
    /// The source model for the solution.
    pub model: FemmModel,
    /// The computed field solution and its diagnostics.
    pub solution: Arc<FemmSolution>,
    /// Reusable factorization of the assembled stiffness matrix.
    pub factor: FactorHandle,
    /// Residual and convergence certificate for the completed solve.
    pub certificate: SolveCertificate,
}

/// Run the steady-state FEM pipeline: mesh, assemble, and solve a model.
///
/// Meshes the model deterministically, selects the physics front for
/// `model.physics`, assembles the global system under `limits`, and solves it
/// with a dense fallback (reusing `cached_factor` when its matrix fingerprint
/// matches). The kernel supplies the evaluation context [`Cx`].
pub fn solve_steady(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    limits: &FemmLimits,
    cached_factor: Option<&FactorHandle>,
) -> FemmResult<SteadySolve> {
    let meshed = DeterministicMesher::new().mesh(cx, model, params)?;
    enforce_mesh_limits(&meshed.mesh, limits)?;
    require_active_dirichlet_boundary(model, &meshed)?;
    if has_nonlinear_bh(model) && model.physics == PhysicsKind::Magnetostatic {
        return solve_nonlinear_ptc(cx, model, params, limits, cached_factor, meshed);
    }
    let assembled = match model.physics {
        sim_lib_femm_core::PhysicsKind::Electrostatic => {
            assemble_system(cx, &ElectrostaticFront, model, &meshed, limits)?
        }
        sim_lib_femm_core::PhysicsKind::HeatSteady => {
            assemble_system(cx, &HeatSteadyFront, model, &meshed, limits)?
        }
        sim_lib_femm_core::PhysicsKind::CurrentSteady => {
            assemble_system(cx, &CurrentSteadyFront, model, &meshed, limits)?
        }
        sim_lib_femm_core::PhysicsKind::Magnetostatic => {
            assemble_system(cx, &MagnetostaticFront, model, &meshed, limits)?
        }
        sim_lib_femm_core::PhysicsKind::MagneticsHarmonic => {
            assemble_system(cx, &MagneticsHarmonicFront, model, &meshed, limits)?
        }
    };
    let factor = reuse_or_factor(&assembled, cached_factor)?;
    let rhs = assembled.r.iter().map(|value| -value).collect::<Vec<_>>();
    let DenseSolveReport {
        x: u,
        final_residual,
        method,
    } = solve_dense_checked(&factor.dense, &rhs)?;
    let solve_id = StableId(model.id.0 ^ params.fingerprint(cx).0 ^ factor.matrix_fingerprint.0);
    let solution = FemmSolution {
        id: solve_id,
        model_id: model.id,
        physics: model.physics.clone(),
        formulation: model.formulation.clone(),
        params: params.clone(),
        mesh: meshed.mesh.clone(),
        u,
        diagnostics: SolveDiagnostics {
            method,
            converged: true,
            iterations: 1,
            final_residual,
            events: vec![
                FemmSolveEvent::Validated,
                FemmSolveEvent::Meshed {
                    elements: meshed.mesh.tri.len(),
                },
                FemmSolveEvent::Finished {
                    solution_id: solve_id,
                },
            ],
            diagnostics: meshed.diagnostics,
        },
    };
    solution.validate()?;
    let solution = Arc::new(solution);
    let certificate = make_linear_certificate(cx, &solution)?;
    Ok(SteadySolve {
        model: model.clone(),
        solution,
        factor,
        certificate,
    })
}

fn solve_nonlinear_ptc(
    cx: &mut Cx,
    model: &FemmModel,
    params: &ParamSet,
    limits: &FemmLimits,
    cached_factor: Option<&FactorHandle>,
    meshed: sim_lib_femm_mesh::MeshedModel,
) -> FemmResult<SteadySolve> {
    let assembled = assemble_system(cx, &PtcMagnetostaticFront, model, &meshed, limits)?;
    let factor = reuse_or_factor(&assembled, cached_factor)?;
    let dense = factor.dense.clone();
    let residual_offset = assembled.r.clone();
    let initial = ptc_initial_state(assembled.k.rows());
    let options = PtcOptions {
        dtau0: 1.0,
        tol: 1.0e-10,
        max_steps: limits.max_solve_iters,
        freeze_jacobian: true,
    };
    let solve_id = StableId(model.id.0 ^ params.fingerprint(cx).0 ^ factor.matrix_fingerprint.0);
    match ptc_solve_report(initial, &options, |u| {
        Ok((linear_residual(&dense, &residual_offset, u), dense.clone()))
    }) {
        Ok((u, mut diagnostics)) => {
            diagnostics.events.insert(
                1,
                FemmSolveEvent::Meshed {
                    elements: meshed.mesh.tri.len(),
                },
            );
            diagnostics.events.push(FemmSolveEvent::Finished {
                solution_id: solve_id,
            });
            diagnostics.iterations = ptc_iterations(&diagnostics);
            diagnostics.diagnostics.extend(meshed.diagnostics);
            let solution = FemmSolution {
                id: solve_id,
                model_id: model.id,
                physics: model.physics.clone(),
                formulation: model.formulation.clone(),
                params: params.clone(),
                mesh: meshed.mesh,
                u,
                diagnostics,
            };
            solution.validate()?;
            let solution = Arc::new(solution);
            let certificate =
                make_ptc_certificate(cx, &solution.diagnostics, solve_id.0, &solution.u)?;
            Ok(SteadySolve {
                model: model.clone(),
                solution,
                factor,
                certificate,
            })
        }
        Err(mut failure) => {
            failure.diagnostics.events.insert(
                1,
                FemmSolveEvent::Meshed {
                    elements: meshed.mesh.tri.len(),
                },
            );
            failure.diagnostics.iterations = ptc_iterations(&failure.diagnostics);
            let partial =
                make_ptc_certificate(cx, &failure.diagnostics, solve_id.0, &failure.state)?;
            let partial_id = partial
                .claim
                .content_id(cx.datum_store_mut())
                .map_err(|err| FemmError::SolveDidNotConverge(err.to_string()))?;
            Err(FemmError::SolveDidNotConverge(format!(
                "femm-ptc did not converge; partial certificate {}",
                partial_id
                    .bytes
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            )))
        }
    }
}

fn has_nonlinear_bh(model: &FemmModel) -> bool {
    model
        .materials
        .iter()
        .any(|material| material.nu_of_b2.is_some())
}

fn require_active_dirichlet_boundary(
    model: &FemmModel,
    meshed: &sim_lib_femm_mesh::MeshedModel,
) -> FemmResult<()> {
    let active = model
        .boundaries
        .iter()
        .filter(|boundary| boundary.kind == BoundaryKind::Dirichlet)
        .any(|boundary| {
            meshed
                .mesh
                .edge_boundary
                .iter()
                .any(|(_, _, name)| name == &boundary.name)
        });
    if active {
        Ok(())
    } else {
        Err(FemmError::SolveDidNotConverge(
            "underconstrained system: no active Dirichlet boundary".to_owned(),
        ))
    }
}

fn ptc_initial_state(len: usize) -> Vec<f64> {
    (0..len).map(|index| 1.0 + index as f64).collect()
}

fn linear_residual(matrix: &[Vec<f64>], offset: &[f64], u: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .zip(offset)
        .map(|(row, offset)| {
            row.iter()
                .zip(u)
                .map(|(entry, value)| entry * value)
                .sum::<f64>()
                + offset
        })
        .collect()
}

fn ptc_iterations(diagnostics: &SolveDiagnostics) -> usize {
    diagnostics
        .events
        .iter()
        .filter(|event| matches!(event, FemmSolveEvent::Step { .. }))
        .count()
}

struct PtcMagnetostaticFront;

impl PhysicsFront for PtcMagnetostaticFront {
    fn kind(&self) -> PhysicsKind {
        PhysicsKind::Magnetostatic
    }

    fn element_residual<S: Scalarish>(
        &self,
        elem: &ElementGeom,
        u_e: [S; 3],
        coeff: &CoeffEval,
    ) -> [S; 3] {
        let grad_u = [
            elem.grad
                .iter()
                .zip(u_e)
                .map(|(grad, u)| S::from_f64(grad[0]) * u)
                .fold(S::from_f64(0.0), |acc, value| acc + value),
            elem.grad
                .iter()
                .zip(u_e)
                .map(|(grad, u)| S::from_f64(grad[1]) * u)
                .fold(S::from_f64(0.0), |acc, value| acc + value),
        ];
        let b2 = grad_u[0] * grad_u[0] + grad_u[1] * grad_u[1];
        let reluctivity = S::from_f64(1.0 / coeff.mu_r.max(1.0e-12)) + b2 * S::from_f64(0.02);
        std::array::from_fn(|index| {
            let dot = grad_u[0] * S::from_f64(elem.grad[index][0])
                + grad_u[1] * S::from_f64(elem.grad[index][1]);
            dot * reluctivity * S::from_f64(elem.area)
        })
    }
}

fn reuse_or_factor(
    assembled: &AssembledSystem,
    cached_factor: Option<&FactorHandle>,
) -> FemmResult<FactorHandle> {
    let matrix_fingerprint = assembled.k.fingerprint();
    if let Some(handle) = cached_factor
        && handle.matrix_fingerprint == matrix_fingerprint
    {
        return Ok(handle.clone());
    }
    Ok(FactorHandle {
        method: LinearMethod::SparseLu,
        matrix_fingerprint,
        payload: DefaultFactory
            .string("dense-factor".to_owned())
            .map_err(|err| sim_lib_femm_core::FemmError::SolveDidNotConverge(err.to_string()))?,
        dense: assembled.k.to_dense()?,
    })
}

struct DenseSolveReport {
    x: Vec<f64>,
    final_residual: f64,
    method: Symbol,
}

fn solve_dense_checked(matrix: &[Vec<f64>], rhs: &[f64]) -> FemmResult<DenseSolveReport> {
    let x = DenseFallbackSolver::dense_solve(matrix, rhs)?;
    let final_residual = dense_residual_norm(matrix, &x, rhs)?;
    if final_residual > DIRECT_RESIDUAL_TOL {
        return Err(FemmError::SolveDidNotConverge(format!(
            "direct residual {final_residual:e} exceeds tolerance"
        )));
    }
    Ok(DenseSolveReport {
        x,
        final_residual,
        method: Symbol::new("femm-direct"),
    })
}

#[cfg(test)]
mod tests {
    use sim_kernel::Expr;
    use sim_lib_femm_core::{Formulation, LengthUnit, ParamRole, ParamSpec, PhysicsKind};
    use sim_lib_femm_geometry::{BlockLabel2, Geometry2, Node2, Segment2, dummy_origin};
    use sim_lib_femm_material::{Boundary, BoundaryKind, Material, MeshPolicy};

    use super::*;

    fn num(text: &str) -> Expr {
        sim_value::build::num_q(Some("numbers"), "f64", text)
    }

    fn one_box_model() -> FemmModel {
        FemmModel {
            id: StableId(77),
            name: Symbol::new("box"),
            physics: PhysicsKind::Electrostatic,
            formulation: Formulation::Planar,
            length_unit: LengthUnit::Meter,
            depth: None,
            frequency_hz: None,
            inputs: vec![ParamSpec {
                name: Symbol::new("vtop"),
                default: Some(
                    DefaultFactory
                        .number_literal(Symbol::qualified("numbers", "f64"), "1.0".to_owned())
                        .unwrap(),
                ),
                unit: None,
                role: ParamRole::Excitation,
            }],
            geometry: Geometry2 {
                nodes: vec![
                    Node2 {
                        xy: [num("0.0"), num("0.0")],
                    },
                    Node2 {
                        xy: [num("1.0"), num("0.0")],
                    },
                    Node2 {
                        xy: [num("1.0"), num("1.0")],
                    },
                    Node2 {
                        xy: [num("0.0"), num("1.0")],
                    },
                ],
                segments: vec![
                    Segment2 {
                        a: 0,
                        b: 1,
                        boundary: Some(Symbol::new("bottom")),
                    },
                    Segment2 {
                        a: 1,
                        b: 2,
                        boundary: None,
                    },
                    Segment2 {
                        a: 2,
                        b: 3,
                        boundary: Some(Symbol::new("top")),
                    },
                    Segment2 {
                        a: 3,
                        b: 0,
                        boundary: None,
                    },
                ],
                labels: vec![BlockLabel2 {
                    name: Symbol::new("air"),
                    at: [num("0.5"), num("0.5")],
                    material: Symbol::new("air"),
                }],
                ..Geometry2::default()
            },
            materials: vec![Material {
                name: Symbol::new("air"),
                mu_r: Some(num("1.0")),
                nu_of_b2: None,
                epsilon_r: Some(num("2.0")),
                sigma: Some(num("3.0")),
                thermal_k: Some(num("4.0")),
                heat_source: None,
                remanence: None,
            }],
            boundaries: vec![
                Boundary {
                    name: Symbol::new("top"),
                    kind: BoundaryKind::Dirichlet,
                    value: num("1.0"),
                },
                Boundary {
                    name: Symbol::new("bottom"),
                    kind: BoundaryKind::Dirichlet,
                    value: num("0.0"),
                },
            ],
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
    fn steady_solve_produces_mesh_and_solution() {
        let mut cx = Cx::new(
            std::sync::Arc::new(sim_kernel::EagerPolicy),
            std::sync::Arc::new(DefaultFactory),
        );
        let out = solve_steady(
            &mut cx,
            &one_box_model(),
            &ParamSet::default(),
            &FemmLimits::default(),
            None,
        )
        .unwrap();
        assert_eq!(out.solution.mesh.tri.len(), 2);
        assert_eq!(out.solution.u.len(), out.solution.mesh.xy.len());
        assert_eq!(out.certificate.method, "femm-direct");
        assert!(out.certificate.final_residual < DIRECT_RESIDUAL_TOL);
        assert_eq!(out.certificate.gradient_trust, None::<crate::GradientTrust>);
    }

    #[test]
    fn underconstrained_linear_solve_errors_without_regularization() {
        let mut cx = Cx::new(
            std::sync::Arc::new(sim_kernel::EagerPolicy),
            std::sync::Arc::new(DefaultFactory),
        );
        let mut model = one_box_model();
        model.boundaries.clear();
        for segment in &mut model.geometry.segments {
            segment.boundary = None;
        }
        let err = match solve_steady(
            &mut cx,
            &model,
            &ParamSet::default(),
            &FemmLimits::default(),
            None,
        ) {
            Ok(_) => panic!("expected SolveDidNotConverge"),
            Err(err) => err,
        };
        let FemmError::SolveDidNotConverge(message) = err else {
            panic!("expected SolveDidNotConverge");
        };
        assert!(message.contains("underconstrained system"));
    }
}
