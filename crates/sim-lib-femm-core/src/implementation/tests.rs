use super::*;

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
}
