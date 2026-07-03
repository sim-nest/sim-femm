#![forbid(unsafe_code)]
//! Linear-solver interface and factorization handles.
//!
//! Defines the linear methods, the factor handle and solver trait, and the
//! dense fallback solver used when no sparse backend is available.

use sim_kernel::Value;
use sim_lib_femm_core::{CsrMatrix, FemmError, FemmResult, StableId};

/// Linear-solver method used to factor and solve the assembled system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinearMethod {
    /// Conjugate gradient, for symmetric positive-definite systems.
    Cg,
    /// Stabilized biconjugate gradient, for nonsymmetric systems.
    Bicgstab,
    /// Sparse LU direct factorization.
    SparseLu,
}

/// A reusable factorization of one assembled stiffness matrix.
///
/// Carries the method, a fingerprint of the matrix it was built from (so a
/// solve can reuse it when the matrix is unchanged), an opaque kernel [`Value`]
/// payload, and the dense factor used by the fallback solver.
#[derive(Clone, Debug)]
pub struct FactorHandle {
    /// Method that produced this factorization.
    pub method: LinearMethod,
    /// Fingerprint of the factored matrix, used to validate reuse.
    pub matrix_fingerprint: StableId,
    /// Backend-specific factor payload as a kernel [`Value`].
    pub payload: Value,
    /// Dense form of the matrix used by the fallback solver.
    pub dense: Vec<Vec<f64>>,
}

/// Contract for a pluggable linear-solver backend.
///
/// Backends factor a [`CsrMatrix`] once and reuse the [`FactorHandle`] across
/// forward and transpose solves; the latter backs adjoint/sensitivity passes.
pub trait LinearSolver {
    /// Factor the matrix `k`, returning a reusable handle.
    fn factor(&mut self, k: &CsrMatrix) -> FemmResult<FactorHandle>;
    /// Solve `K x = b` using a prior factorization `f`.
    fn solve(&self, f: &FactorHandle, b: &[f64]) -> FemmResult<Vec<f64>>;
    /// Solve the transpose system `K^T x = b` using factorization `f`.
    fn solve_transpose(&self, f: &FactorHandle, b: &[f64]) -> FemmResult<Vec<f64>>;
}

/// Dense Gaussian-elimination solver used when no sparse backend is available.
pub struct DenseFallbackSolver;

impl DenseFallbackSolver {
    /// Solve the dense system `matrix * x = rhs` by Gauss-Jordan elimination.
    ///
    /// Returns [`FemmError::SolveDidNotConverge`] on a (near-)singular pivot.
    pub fn dense_solve(matrix: &[Vec<f64>], rhs: &[f64]) -> FemmResult<Vec<f64>> {
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
}

/// Solve a symmetric positive-definite system by conjugate gradient.
///
/// Iterates until the residual norm drops below `tol` or `max_iters` is reached,
/// returning [`FemmError::SolveDidNotConverge`] if it does not converge. See the
/// [crate README](https://github.com/sim-nest/sim-femm) for the FEM role.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_core::CsrMatrix;
/// use sim_lib_femm_solve::cg_solve;
///
/// // Tridiagonal SPD system with known solution x = [5, 10, 10].
/// let k = CsrMatrix::new(
///     vec![0, 2, 5, 7],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0],
/// )
/// .unwrap();
/// let x = cg_solve(&k, &[15.0, 10.0, 10.0], 1.0e-10, 64).unwrap();
/// assert!((x[0] - 5.0).abs() < 1.0e-8);
/// ```
pub fn cg_solve(k: &CsrMatrix, b: &[f64], tol: f64, max_iters: usize) -> FemmResult<Vec<f64>> {
    let mut x = vec![0.0; b.len()];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);
    for iter in 0..max_iters {
        let ap = k.matvec(&p);
        let alpha = rs_old / dot(&p, &ap);
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        if rs_new.sqrt() < tol {
            return Ok(x);
        }
        let beta = rs_new / rs_old;
        for i in 0..p.len() {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
        if iter + 1 == max_iters {
            return Err(FemmError::SolveDidNotConverge(format!(
                "cg residual={} iterations={}",
                rs_new.sqrt(),
                max_iters
            )));
        }
    }
    Err(FemmError::SolveDidNotConverge("cg failed".to_owned()))
}

/// Solve a general (possibly nonsymmetric) system by stabilized BiCG.
///
/// Iterates until the residual norm drops below `tol` or `max_iters` is reached,
/// returning [`FemmError::SolveDidNotConverge`] if it does not converge.
pub fn bicgstab_solve(
    k: &CsrMatrix,
    b: &[f64],
    tol: f64,
    max_iters: usize,
) -> FemmResult<Vec<f64>> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let r_hat = r.clone();
    let mut rho_prev = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;
    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];
    for iter in 0..max_iters {
        let rho = dot(&r_hat, &r);
        if rho.abs() < 1.0e-14 {
            break;
        }
        let beta = (rho / rho_prev) * (alpha / omega);
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        v = k.matvec(&p);
        alpha = rho / dot(&r_hat, &v);
        let s = (0..n).map(|i| r[i] - alpha * v[i]).collect::<Vec<_>>();
        if norm(&s) < tol {
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            return Ok(x);
        }
        let t = k.matvec(&s);
        omega = dot(&t, &s) / dot(&t, &t);
        for i in 0..n {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }
        if norm(&r) < tol {
            return Ok(x);
        }
        rho_prev = rho;
        if iter + 1 == max_iters {
            return Err(FemmError::SolveDidNotConverge(format!(
                "bicgstab residual={} iterations={}",
                norm(&r),
                max_iters
            )));
        }
    }
    Err(FemmError::SolveDidNotConverge("bicgstab failed".to_owned()))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(l, r)| l * r).sum()
}

fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

#[cfg(test)]
mod tests {
    use sim_kernel::{DefaultFactory, Factory, Symbol};

    use super::*;

    fn payload() -> Value {
        DefaultFactory
            .symbol(Symbol::new("factor"))
            .expect("symbol payload")
    }

    fn spd_matrix() -> CsrMatrix {
        CsrMatrix::new(
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0],
        )
        .unwrap()
    }

    #[test]
    fn cg_solves_small_spd_system() {
        let x = cg_solve(&spd_matrix(), &[15.0, 10.0, 10.0], 1.0e-10, 64).unwrap();
        assert!((x[0] - 5.0).abs() < 1.0e-8);
    }

    #[test]
    fn bicgstab_solves_nonsymmetric_system() {
        let matrix =
            CsrMatrix::new(vec![0, 2, 4], vec![0, 1, 0, 1], vec![4.0, 1.0, 2.0, 3.0]).unwrap();
        let x = bicgstab_solve(&matrix, &[1.0, 1.0], 1.0e-10, 64).unwrap();
        assert!((4.0 * x[0] + x[1] - 1.0).abs() < 1.0e-8);
    }

    #[test]
    fn dense_fallback_solves_three_by_three() {
        let out = DenseFallbackSolver::dense_solve(&spd_matrix().to_dense(), &[15.0, 10.0, 10.0])
            .unwrap();
        assert!((out[0] - 5.0).abs() < 1.0e-8);
    }

    #[test]
    fn factor_reuse_metadata_matches() {
        let first = FactorHandle {
            method: LinearMethod::SparseLu,
            matrix_fingerprint: spd_matrix().fingerprint(),
            payload: payload(),
            dense: spd_matrix().to_dense(),
        };
        let second = first.clone();
        assert_eq!(first.matrix_fingerprint, second.matrix_fingerprint);
    }
}
