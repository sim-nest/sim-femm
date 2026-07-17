#![forbid(unsafe_code)]
//! Linear-solver interface and factorization handles.
//!
//! Defines the linear methods, the factor handle and solver trait, and the
//! dense fallback solver used when no sparse backend is available.

use sim_kernel::Value;
use sim_lib_femm_core::{CsrMatrix, FemmError, FemmResult, StableId};

const PIVOT_TOL: f64 = 1.0e-12;
const BREAKDOWN_TOL: f64 = 1.0e-14;

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
        validate_dense_system(matrix, rhs)?;
        let n = rhs.len();
        let mut a = matrix.to_vec();
        let mut b = rhs.to_vec();
        for pivot in 0..n {
            let pivot_row = (pivot..n)
                .max_by(|&left, &right| a[left][pivot].abs().total_cmp(&a[right][pivot].abs()))
                .expect("pivot range is non-empty");
            if pivot_row != pivot {
                a.swap(pivot, pivot_row);
                b.swap(pivot, pivot_row);
            }
            let diag = a[pivot][pivot];
            if diag.abs() < PIVOT_TOL {
                return Err(FemmError::SolveDidNotConverge(
                    "singular dense solve".to_owned(),
                ));
            }
            for value in a[pivot].iter_mut().skip(pivot) {
                *value = finite(*value / diag, "non-finite dense solve pivot row")?;
            }
            b[pivot] = finite(b[pivot] / diag, "non-finite dense solve pivot rhs")?;
            let pivot_tail = a[pivot][pivot..n].to_vec();
            let pivot_rhs = b[pivot];
            for row in 0..n {
                if row == pivot {
                    continue;
                }
                let factor = a[row][pivot];
                for (value, pivot_value) in a[row].iter_mut().skip(pivot).zip(&pivot_tail) {
                    *value = finite(
                        *value - factor * pivot_value,
                        "non-finite dense solve elimination",
                    )?;
                }
                b[row] = finite(b[row] - factor * pivot_rhs, "non-finite dense solve rhs")?;
            }
        }
        validate_solution_vector(b, "dense solve")
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
    validate_iterative_system(k, b, tol, max_iters, "cg")?;
    if b.iter().all(|value| *value == 0.0) {
        return Ok(vec![0.0; b.len()]);
    }
    let mut x = vec![0.0; b.len()];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r, "cg residual norm")?;
    for iter in 0..max_iters {
        let ap = k.matvec(&p)?;
        let denom = dot(&p, &ap, "cg denominator")?;
        if denom.abs() < BREAKDOWN_TOL {
            return Err(FemmError::SolveDidNotConverge(
                "cg breakdown: zero denominator".to_owned(),
            ));
        }
        let alpha = finite(rs_old / denom, "non-finite cg alpha")?;
        for i in 0..x.len() {
            x[i] = finite(x[i] + alpha * p[i], "non-finite cg solution")?;
            r[i] = finite(r[i] - alpha * ap[i], "non-finite cg residual")?;
        }
        let rs_new = dot(&r, &r, "cg residual norm")?;
        let residual = finite(rs_new.sqrt(), "non-finite cg residual norm")?;
        if residual < tol {
            return validate_solution_vector(x, "cg");
        }
        if rs_old.abs() < BREAKDOWN_TOL {
            return Err(FemmError::SolveDidNotConverge(
                "cg breakdown: residual denominator vanished".to_owned(),
            ));
        }
        let beta = finite(rs_new / rs_old, "non-finite cg beta")?;
        for i in 0..p.len() {
            p[i] = finite(r[i] + beta * p[i], "non-finite cg direction")?;
        }
        rs_old = rs_new;
        if iter + 1 == max_iters {
            return Err(FemmError::SolveDidNotConverge(format!(
                "cg residual={} iterations={}",
                residual, max_iters
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
    validate_iterative_system(k, b, tol, max_iters, "bicgstab")?;
    if b.iter().all(|value| *value == 0.0) {
        return Ok(vec![0.0; b.len()]);
    }
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let r_hat = r.clone();
    let mut rho_prev = 1.0_f64;
    let mut alpha = 1.0_f64;
    let mut omega = 1.0_f64;
    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];
    for iter in 0..max_iters {
        let rho = dot(&r_hat, &r, "bicgstab rho")?;
        if rho.abs() < 1.0e-14 {
            return Err(FemmError::SolveDidNotConverge(
                "bicgstab breakdown: rho denominator vanished".to_owned(),
            ));
        }
        if rho_prev.abs() < BREAKDOWN_TOL || omega.abs() < BREAKDOWN_TOL {
            return Err(FemmError::SolveDidNotConverge(
                "bicgstab breakdown: update denominator vanished".to_owned(),
            ));
        }
        let beta = finite(
            (rho / rho_prev) * (alpha / omega),
            "non-finite bicgstab beta",
        )?;
        for i in 0..n {
            p[i] = finite(
                r[i] + beta * (p[i] - omega * v[i]),
                "non-finite bicgstab direction",
            )?;
        }
        v = k.matvec(&p)?;
        let alpha_denom = dot(&r_hat, &v, "bicgstab alpha denominator")?;
        if alpha_denom.abs() < BREAKDOWN_TOL {
            return Err(FemmError::SolveDidNotConverge(
                "bicgstab breakdown: alpha denominator vanished".to_owned(),
            ));
        }
        alpha = finite(rho / alpha_denom, "non-finite bicgstab alpha")?;
        let s = (0..n)
            .map(|i| finite(r[i] - alpha * v[i], "non-finite bicgstab residual stage"))
            .collect::<FemmResult<Vec<_>>>()?;
        if norm(&s, "bicgstab residual stage")? < tol {
            for i in 0..n {
                x[i] = finite(x[i] + alpha * p[i], "non-finite bicgstab solution")?;
            }
            return validate_solution_vector(x, "bicgstab");
        }
        let t = k.matvec(&s)?;
        let omega_den = dot(&t, &t, "bicgstab omega denominator")?;
        if omega_den.abs() < BREAKDOWN_TOL {
            return Err(FemmError::SolveDidNotConverge(
                "bicgstab breakdown: omega denominator vanished".to_owned(),
            ));
        }
        omega = finite(
            dot(&t, &s, "bicgstab omega numerator")? / omega_den,
            "non-finite bicgstab omega",
        )?;
        for i in 0..n {
            x[i] = finite(
                x[i] + alpha * p[i] + omega * s[i],
                "non-finite bicgstab solution",
            )?;
            r[i] = finite(s[i] - omega * t[i], "non-finite bicgstab residual")?;
        }
        let residual = norm(&r, "bicgstab residual")?;
        if residual < tol {
            return validate_solution_vector(x, "bicgstab");
        }
        rho_prev = rho;
        if iter + 1 == max_iters {
            return Err(FemmError::SolveDidNotConverge(format!(
                "bicgstab residual={} iterations={}",
                residual, max_iters
            )));
        }
    }
    Err(FemmError::SolveDidNotConverge("bicgstab failed".to_owned()))
}

pub(crate) fn validate_dense_system(matrix: &[Vec<f64>], rhs: &[f64]) -> FemmResult<()> {
    if matrix.len() != rhs.len() {
        return Err(FemmError::MalformedMatrix(format!(
            "dense matrix has {} rows but rhs has {} entries",
            matrix.len(),
            rhs.len()
        )));
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(FemmError::MalformedMatrix(
            "rhs values must be finite".to_owned(),
        ));
    }
    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != rhs.len() {
            return Err(FemmError::MalformedMatrix(format!(
                "dense matrix row {row_index} has {} columns but expected {}",
                row.len(),
                rhs.len()
            )));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(FemmError::MalformedMatrix(format!(
                "dense matrix row {row_index} contains a non-finite value"
            )));
        }
    }
    Ok(())
}

pub(crate) fn dense_residual_norm(matrix: &[Vec<f64>], x: &[f64], rhs: &[f64]) -> FemmResult<f64> {
    validate_dense_system(matrix, rhs)?;
    if x.len() != rhs.len() {
        return Err(FemmError::MalformedMatrix(format!(
            "solution length {} does not match rhs length {}",
            x.len(),
            rhs.len()
        )));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(FemmError::MalformedMatrix(
            "solution values must be finite".to_owned(),
        ));
    }
    let mut norm_sq = 0.0;
    for (row, rhs_value) in matrix.iter().zip(rhs) {
        let mut residual = -*rhs_value;
        for (entry, value) in row.iter().zip(x) {
            residual = finite(residual + entry * value, "non-finite dense residual")?;
        }
        norm_sq = finite(
            norm_sq + residual * residual,
            "non-finite dense residual norm",
        )?;
    }
    finite(norm_sq.sqrt(), "non-finite dense residual norm")
}

fn validate_iterative_system(
    k: &CsrMatrix,
    b: &[f64],
    tol: f64,
    max_iters: usize,
    method: &str,
) -> FemmResult<()> {
    k.validate()?;
    if k.rows() != b.len() {
        return Err(FemmError::MalformedMatrix(format!(
            "{method} rhs length {} does not match matrix rows {}",
            b.len(),
            k.rows()
        )));
    }
    if b.iter().any(|value| !value.is_finite()) {
        return Err(FemmError::MalformedMatrix(format!(
            "{method} rhs values must be finite"
        )));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(FemmError::SolveDidNotConverge(format!(
            "{method} tolerance must be positive and finite"
        )));
    }
    if max_iters == 0 {
        return Err(FemmError::SolveDidNotConverge(format!(
            "{method} max_iters must be greater than zero"
        )));
    }
    Ok(())
}

fn validate_solution_vector(solution: Vec<f64>, method: &str) -> FemmResult<Vec<f64>> {
    if solution.iter().all(|value| value.is_finite()) {
        Ok(solution)
    } else {
        Err(FemmError::SolveDidNotConverge(format!(
            "{method} produced a non-finite solution"
        )))
    }
}

fn dot(left: &[f64], right: &[f64], context: &str) -> FemmResult<f64> {
    if left.len() != right.len() {
        return Err(FemmError::MalformedMatrix(format!(
            "{context} vector length mismatch: {} != {}",
            left.len(),
            right.len()
        )));
    }
    let mut sum = 0.0;
    for (l, r) in left.iter().zip(right) {
        sum = finite(sum + l * r, context)?;
    }
    Ok(sum)
}

fn norm(values: &[f64], context: &str) -> FemmResult<f64> {
    finite(dot(values, values, context)?.sqrt(), context)
}

fn finite(value: f64, context: &str) -> FemmResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FemmError::SolveDidNotConverge(context.to_owned()))
    }
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
        let out = DenseFallbackSolver::dense_solve(
            &spd_matrix().to_dense().unwrap(),
            &[15.0, 10.0, 10.0],
        )
        .unwrap();
        assert!((out[0] - 5.0).abs() < 1.0e-8);
    }

    #[test]
    fn dense_fallback_rejects_malformed_and_singular_systems() {
        let err = DenseFallbackSolver::dense_solve(&[vec![1.0]], &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, FemmError::MalformedMatrix(_)));

        let err = DenseFallbackSolver::dense_solve(&[vec![1.0, 2.0]], &[1.0]).unwrap_err();
        assert!(matches!(err, FemmError::MalformedMatrix(_)));

        let err = DenseFallbackSolver::dense_solve(&[vec![f64::NAN]], &[1.0]).unwrap_err();
        assert!(matches!(err, FemmError::MalformedMatrix(_)));

        let err = DenseFallbackSolver::dense_solve(&[vec![0.0]], &[1.0]).unwrap_err();
        let FemmError::SolveDidNotConverge(message) = err else {
            panic!("expected SolveDidNotConverge");
        };
        assert!(message.contains("singular dense solve"));
    }

    #[test]
    fn iterative_solvers_handle_zero_rhs_and_reject_bad_inputs() {
        assert_eq!(
            cg_solve(&spd_matrix(), &[0.0, 0.0, 0.0], 1.0e-10, 8).unwrap(),
            vec![0.0, 0.0, 0.0]
        );
        assert_eq!(
            bicgstab_solve(&spd_matrix(), &[0.0, 0.0, 0.0], 1.0e-10, 8).unwrap(),
            vec![0.0, 0.0, 0.0]
        );

        let err = cg_solve(&spd_matrix(), &[1.0], 1.0e-10, 8).unwrap_err();
        assert!(matches!(err, FemmError::MalformedMatrix(_)));

        let raw = CsrMatrix {
            rowptr: vec![0, 1],
            colind: vec![0],
            vals: vec![f64::INFINITY],
        };
        let err = bicgstab_solve(&raw, &[1.0], 1.0e-10, 8).unwrap_err();
        assert!(matches!(err, FemmError::MalformedMatrix(_)));
    }

    #[test]
    fn iterative_solvers_reject_breakdown_denominators() {
        let zero = CsrMatrix::new(vec![0, 0], vec![], vec![]).unwrap();

        let err = cg_solve(&zero, &[1.0], 1.0e-10, 8).unwrap_err();
        let FemmError::SolveDidNotConverge(message) = err else {
            panic!("expected SolveDidNotConverge");
        };
        assert!(message.contains("breakdown"));

        let err = bicgstab_solve(&zero, &[1.0], 1.0e-10, 8).unwrap_err();
        let FemmError::SolveDidNotConverge(message) = err else {
            panic!("expected SolveDidNotConverge");
        };
        assert!(message.contains("breakdown"));
    }

    #[test]
    fn factor_reuse_metadata_matches() {
        let first = FactorHandle {
            method: LinearMethod::SparseLu,
            matrix_fingerprint: spd_matrix().fingerprint(),
            payload: payload(),
            dense: spd_matrix().to_dense().unwrap(),
        };
        let second = first.clone();
        assert_eq!(first.matrix_fingerprint, second.matrix_fingerprint);
    }
}
