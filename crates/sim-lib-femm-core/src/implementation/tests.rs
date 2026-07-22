use super::*;
use crate::{CsrMatrix, FemmError};

// conformance: FEMM domain stack builds finite-element core records.

#[test]
fn capabilities_list_all_physics_kinds() {
    for physics in [
        "Magnetostatic",
        "MagneticsHarmonic",
        "Electrostatic",
        "HeatSteady",
        "CurrentSteady",
    ] {
        assert!(
            femm_capabilities(false, false, false)
                .iter()
                .any(|entry| entry == physics),
            "missing {physics}"
        );
    }
}

#[test]
fn capabilities_describe_missing_optional_backends_as_unavailable() {
    let capabilities = femm_capabilities(false, false, false);
    assert!(capabilities.contains(&"femm-ptc:unavailable".to_owned()));
    assert!(capabilities.contains(&"femm-adjoint:unavailable".to_owned()));
    assert!(capabilities.contains(&"numbers/field:unavailable".to_owned()));
}

#[test]
fn stable_summary_is_deterministic() {
    assert_eq!(
        stable_summary("Thing", &[("a", "1".to_owned()), ("b", "2".to_owned())]),
        "Thing(a=1, b=2)"
    );
}

#[test]
fn malformed_csr_is_rejected() {
    let err = CsrMatrix::new(vec![1, 1], vec![], vec![]).unwrap_err();
    assert!(matches!(err, FemmError::MalformedMatrix(_)));
    let err = CsrMatrix::new(vec![0, 1], vec![3], vec![1.0]).unwrap_err();
    assert!(matches!(err, FemmError::MalformedMatrix(_)));
    let err = CsrMatrix::new(vec![0, 1], vec![0], vec![f64::NAN]).unwrap_err();
    assert!(matches!(err, FemmError::MalformedMatrix(_)));
}

#[test]
fn csr_checked_ops_reject_bad_vectors_and_raw_mutations() {
    let matrix = CsrMatrix::identity(2);
    assert_eq!(matrix.matvec(&[1.0, 2.0]).unwrap(), vec![1.0, 2.0]);
    let err = matrix.matvec(&[1.0]).unwrap_err();
    assert!(matches!(err, FemmError::MalformedMatrix(_)));
    let err = matrix.matvec(&[f64::INFINITY, 1.0]).unwrap_err();
    assert!(matches!(err, FemmError::MalformedMatrix(_)));

    let raw = CsrMatrix {
        rowptr: vec![0, 1],
        colind: vec![0],
        vals: vec![f64::INFINITY],
    };
    let err = raw.to_dense().unwrap_err();
    assert!(matches!(err, FemmError::MalformedMatrix(_)));
}
