use sim_kernel::{DefaultFactory, Factory, Symbol, Value};
use sim_lib_femm_core::{CsrMatrix, FemmError};

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
    let matrix = CsrMatrix::new(vec![0, 2, 4], vec![0, 1, 0, 1], vec![4.0, 1.0, 2.0, 3.0]).unwrap();
    let x = bicgstab_solve(&matrix, &[1.0, 1.0], 1.0e-10, 64).unwrap();
    assert!((4.0 * x[0] + x[1] - 1.0).abs() < 1.0e-8);
}

#[test]
fn dense_fallback_solves_three_by_three() {
    let out =
        DenseFallbackSolver::dense_solve(&spd_matrix().to_dense().unwrap(), &[15.0, 10.0, 10.0])
            .unwrap();
    assert!((out[0] - 5.0).abs() < 1.0e-8);
}

// conformance: FEMM linear solvers expose residual evidence for certificates.
#[test]
fn linear_solver_residual_norm_tracks_certificate_basis() {
    let rhs = [15.0, 10.0, 10.0];
    let matrix = spd_matrix();
    let x = cg_solve(&matrix, &rhs, 1.0e-10, 64).unwrap();
    let residual = dense_residual_norm(&matrix.to_dense().unwrap(), &x, &rhs).unwrap();

    assert!(residual < 1.0e-8);
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
    let first = FactorHandle::new(LinearFactor {
        method: LinearMethod::SparseLu,
        matrix_fingerprint: spd_matrix().fingerprint(),
        transpose: TransposeSupport::Supported,
        lifecycle: FactorLifecycle::Fresh,
        payload: payload(),
    });
    let second = first.clone();
    assert_eq!(first.matrix_fingerprint(), second.matrix_fingerprint());
    assert_eq!(first.lifecycle(), FactorLifecycle::Fresh);
    assert_eq!(first.transpose(), TransposeSupport::Supported);
}

#[test]
fn dense_fallback_factor_solves_transpose_without_public_dense_field() {
    let solver = DenseFallbackSolver;
    let matrix = CsrMatrix::new(vec![0, 2, 4], vec![0, 1, 0, 1], vec![4.0, 1.0, 2.0, 3.0]).unwrap();
    let factor = solver.factor(&matrix).unwrap();

    assert!(solver.can_reuse(&factor));
    assert_eq!(factor.method(), &LinearMethod::SparseLu);
    assert_eq!(factor.matrix_fingerprint(), matrix.fingerprint());
    assert_eq!(factor.transpose(), TransposeSupport::Supported);

    let x = solver.solve(&factor, &[1.0, 1.0]).unwrap();
    assert!((4.0 * x[0] + x[1] - 1.0).abs() < 1.0e-8);

    let xt = solver.solve_transpose(&factor, &[1.0, 1.0]).unwrap();
    assert!((4.0 * xt[0] + 2.0 * xt[1] - 1.0).abs() < 1.0e-8);
}
