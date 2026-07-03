#![forbid(unsafe_code)]
//! Pseudo-transient continuation solve and its diagnostics.
//!
//! Defines the PTC options, the nonlinear iteration that drives a residual to
//! convergence, and the event and diagnostic records describing the solve.

use std::sync::Arc;

use sim_kernel::{Cx, Diagnostic, Result as KernelResult, Severity, Symbol, Value};
use sim_lib_femm_core::{FemmError, FemmResult, StableId};
use sim_lib_numbers_numeric::{NumericKind, OdeOpts, OdeProblem, OdeSolver, register_ode_solver};

/// Tuning knobs for the [`ptc_solve`] pseudo-transient continuation iteration.
///
/// Pseudo-transient continuation drives a nonlinear residual to zero by adding
/// a shrinking `1/dtau` shift to the Jacobian diagonal, so early steps behave
/// like a damped explicit march and later steps approach a full Newton step as
/// `dtau` grows. See the [crate README](../sim_lib_femm_flow/index.html).
#[derive(Clone, Debug)]
pub struct PtcOptions {
    /// Initial pseudo-time step `dtau`; larger values approach Newton sooner.
    pub dtau0: f64,
    /// Convergence tolerance on the residual 2-norm.
    pub tol: f64,
    /// Maximum number of pseudo-time steps before the solve is abandoned.
    pub max_steps: usize,
    /// Reuse the Jacobian across steps instead of reassembling each step.
    pub freeze_jacobian: bool,
}

/// A single observable event in a [`ptc_solve`] run, recorded for diagnostics.
///
/// The event stream forms a chronological trace of the solve, from validation
/// through meshing and per-step residual reduction to a finished or failed
/// terminal state.
#[derive(Clone, Debug)]
pub enum FemmSolveEvent {
    /// The problem passed pre-solve validation.
    Validated,
    /// A mesh was produced with the given number of elements.
    Meshed {
        /// Number of mesh elements.
        elements: usize,
    },
    /// One pseudo-time step `k` reached `residual` at pseudo-step size `dtau`.
    Step {
        /// Zero-based step index.
        k: usize,
        /// Residual 2-norm after this step.
        residual: f64,
        /// Pseudo-time step size used for this step.
        dtau: f64,
    },
    /// The solve converged and stored its solution under `solution_id`.
    Finished {
        /// Stable identifier of the stored solution.
        solution_id: StableId,
    },
    /// The solve failed, carrying the kernel [`Diagnostic`] that explains why.
    Failed {
        /// Diagnostic describing the failure.
        diagnostic: Diagnostic,
    },
}

/// Outcome summary and event trace returned by [`ptc_solve`].
///
/// Bundles the kernel-facing method [`Symbol`] and [`Diagnostic`] records with
/// the convergence flag, iteration count, final residual, and the full
/// [`FemmSolveEvent`] stream.
#[derive(Clone, Debug)]
pub struct SolveDiagnostics {
    /// Name of the solve method that produced these diagnostics.
    pub method: Symbol,
    /// Whether the residual fell below the tolerance before the step budget.
    pub converged: bool,
    /// Number of pseudo-time steps taken.
    pub iterations: usize,
    /// Residual 2-norm at the final recorded step.
    pub final_residual: f64,
    /// Chronological trace of solve events.
    pub events: Vec<FemmSolveEvent>,
    /// Kernel diagnostics collected during the solve.
    pub diagnostics: Vec<Diagnostic>,
}

/// Non-converged PTC run with the last state and diagnostics.
#[derive(Debug)]
pub struct PtcSolveFailure {
    /// Last state vector reached before the solver stopped.
    pub state: Vec<f64>,
    /// Diagnostics and event trace recorded for the stopped run.
    pub diagnostics: SolveDiagnostics,
    /// Error describing why the PTC run stopped.
    pub error: FemmError,
}

/// Drives a nonlinear system to convergence by pseudo-transient continuation.
///
/// `u` is the initial state vector and `system` returns the residual vector and
/// dense Jacobian at a candidate state. Each step solves the diagonally shifted
/// system `(J + I/dtau) du = -r`, applies the update, and adapts `dtau` from the
/// ratio of successive residuals. Returns the converged state and its
/// [`SolveDiagnostics`], or [`FemmError::SolveDidNotConverge`] if the step
/// budget in [`PtcOptions::max_steps`] is exhausted.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_flow::{ptc_solve, PtcOptions};
///
/// // Solve the scalar root u - 3 = 0; a large initial dtau makes the first
/// // shifted step essentially a full Newton step.
/// let (u, diag) = ptc_solve(
///     vec![0.0],
///     &PtcOptions { dtau0: 1.0e12, tol: 1.0e-10, max_steps: 8, freeze_jacobian: false },
///     |u| Ok((vec![u[0] - 3.0], vec![vec![1.0]])),
/// )
/// .unwrap();
/// assert!(diag.converged);
/// assert!((u[0] - 3.0).abs() < 1.0e-9);
/// ```
pub fn ptc_solve(
    u: Vec<f64>,
    options: &PtcOptions,
    system: impl Fn(&[f64]) -> FemmResult<(Vec<f64>, Vec<Vec<f64>>)>,
) -> FemmResult<(Vec<f64>, SolveDiagnostics)> {
    ptc_solve_report(u, options, system).map_err(|failure| failure.error)
}

/// Drives a nonlinear system and returns diagnostics for non-converged runs.
pub fn ptc_solve_report(
    mut u: Vec<f64>,
    options: &PtcOptions,
    system: impl Fn(&[f64]) -> FemmResult<(Vec<f64>, Vec<Vec<f64>>)>,
) -> std::result::Result<(Vec<f64>, SolveDiagnostics), Box<PtcSolveFailure>> {
    let mut dtau = options.dtau0;
    let mut events = vec![FemmSolveEvent::Validated];
    let mut final_residual = f64::INFINITY;
    for step in 0..options.max_steps {
        let (r, jac) = system(&u).map_err(|error| {
            Box::new(PtcSolveFailure {
                state: u.clone(),
                diagnostics: failed_diagnostics(
                    events.clone(),
                    step,
                    final_residual,
                    error.to_string(),
                ),
                error,
            })
        })?;
        let residual = r.iter().map(|value| value * value).sum::<f64>().sqrt();
        final_residual = residual;
        events.push(FemmSolveEvent::Step {
            k: step,
            residual,
            dtau,
        });
        if residual < options.tol {
            let diagnostics = SolveDiagnostics {
                method: Symbol::new("femm-ptc"),
                converged: true,
                iterations: step,
                final_residual: residual,
                events,
                diagnostics: Vec::new(),
            };
            return Ok((u, diagnostics));
        }
        let n = jac.len();
        let mut shifted = jac.clone();
        for (index, row) in shifted.iter_mut().enumerate().take(n) {
            row[index] += 1.0 / dtau;
        }
        let delta = dense_solve(&shifted, &r.iter().map(|value| -value).collect::<Vec<_>>())
            .map_err(|error| {
                Box::new(PtcSolveFailure {
                    state: u.clone(),
                    diagnostics: failed_diagnostics(
                        events.clone(),
                        step + 1,
                        final_residual,
                        error.to_string(),
                    ),
                    error,
                })
            })?;
        for (state, update) in u.iter_mut().zip(delta) {
            *state += update;
        }
        let next_residual = system(&u)
            .map_err(|error| {
                Box::new(PtcSolveFailure {
                    state: u.clone(),
                    diagnostics: failed_diagnostics(
                        events.clone(),
                        step + 1,
                        final_residual,
                        error.to_string(),
                    ),
                    error,
                })
            })?
            .0
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if next_residual > 0.0 {
            dtau *= residual / next_residual;
        }
    }
    let message = "femm-ptc max steps exceeded".to_owned();
    Err(Box::new(PtcSolveFailure {
        state: u,
        diagnostics: failed_diagnostics(events, options.max_steps, final_residual, message.clone()),
        error: FemmError::SolveDidNotConverge(message),
    }))
}

fn failed_diagnostics(
    mut events: Vec<FemmSolveEvent>,
    iterations: usize,
    final_residual: f64,
    message: String,
) -> SolveDiagnostics {
    let diagnostic = failure_diagnostic(&message);
    events.push(FemmSolveEvent::Failed {
        diagnostic: diagnostic.clone(),
    });
    SolveDiagnostics {
        method: Symbol::new("femm-ptc"),
        converged: false,
        iterations,
        final_residual,
        events,
        diagnostics: vec![diagnostic],
    }
}

fn dense_solve(matrix: &[Vec<f64>], rhs: &[f64]) -> FemmResult<Vec<f64>> {
    let n = rhs.len();
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    for pivot in 0..n {
        let diag = a[pivot][pivot];
        if diag.abs() < 1.0e-12 {
            return Err(FemmError::SolveDidNotConverge(
                "singular dense solve".to_owned(),
            ));
        }
        for value in a[pivot].iter_mut().skip(pivot) {
            *value /= diag;
        }
        b[pivot] /= diag;
        let pivot_tail = a[pivot][pivot..n].to_vec();
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            for (value, pivot_value) in a[row].iter_mut().skip(pivot).zip(&pivot_tail) {
                *value -= factor * pivot_value;
            }
            b[row] -= factor * b[pivot];
        }
    }
    Ok(b)
}

struct FemmPtcPlugin;

impl sim_lib_numbers_numeric::NumericPlugin for FemmPtcPlugin {
    fn name(&self) -> Symbol {
        Symbol::new("femm-ptc")
    }

    fn kind(&self) -> NumericKind {
        NumericKind::OdeFixed
    }
}

impl OdeSolver for FemmPtcPlugin {
    fn solve(
        &self,
        _cx: &mut Cx,
        _problem: OdeProblem<'_>,
        _opt: OdeOpts,
    ) -> KernelResult<Vec<(Value, Value)>> {
        Err(sim_kernel::Error::Eval(
            "femm-ptc is a steady FEMM solver registration hook".to_owned(),
        ))
    }
}

/// Registers the `femm-ptc` solver with the sim-numbers ODE registry.
///
/// The registration is a steady-FEMM hook: it names the method so the kernel
/// can refer to it, while the actual nonlinear march is performed by
/// [`ptc_solve`] rather than the ODE-stepping path.
pub fn register_femm_ptc() -> KernelResult<()> {
    register_ode_solver(Arc::new(FemmPtcPlugin))
}

/// Builds an error-severity kernel [`Diagnostic`] carrying `message`.
///
/// A convenience for populating [`FemmSolveEvent::Failed`] and
/// [`SolveDiagnostics::diagnostics`] when a solve aborts.
pub fn failure_diagnostic(message: &str) -> Diagnostic {
    Diagnostic {
        severity: Severity::Error,
        message: message.to_owned(),
        source: None,
        span: None,
        code: None,
        related: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use sim_lib_numbers_numeric::global_numeric_registry;

    use super::*;

    #[test]
    fn linear_problem_converges_in_one_ptc_step() {
        let (u, diagnostics) = ptc_solve(
            vec![0.0],
            &PtcOptions {
                dtau0: 1.0e12,
                tol: 1.0e-10,
                max_steps: 8,
                freeze_jacobian: false,
            },
            |u| Ok((vec![u[0] - 3.0], vec![vec![1.0]])),
        )
        .unwrap();
        assert!((u[0] - 3.0).abs() < 1.0e-9);
        assert!(diagnostics.converged);
    }

    #[test]
    fn ptc_method_is_registered() {
        register_femm_ptc().unwrap();
        let guard = global_numeric_registry().read().unwrap();
        assert!(guard.ode_fixed(&Symbol::new("femm-ptc")).is_some());
    }
}
