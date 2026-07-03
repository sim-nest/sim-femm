use std::sync::Arc;

use sim_kernel::{ContentId, Cx, DefaultFactory, EagerPolicy, Expr, NumberValue, Symbol};
use sim_lib_femm_core::{
    FemmLimits, Formulation, LengthUnit, ParamRole, ParamSet, ParamSpec, PhysicsKind, StableId,
};
use sim_lib_femm_field::{Field, Projection};
use sim_lib_femm_fixtures::parallel_plate_capacitor;
use sim_lib_femm_geometry::{Geometry2, dummy_origin};
use sim_lib_femm_material::MeshPolicy;
use sim_lib_femm_mesh::FemmModel;
use sim_lib_femm_post::QuantitySpec;
use sim_lib_femm_solve::{GradientTrust, SolveExportRecord, certificate_claim, solve_steady};

use crate::{FemmCall, FemmCallable, ModelCallable, OutputQuery, femm_as_func, quality};

fn num(text: &str) -> Expr {
    sim_value::build::num_q(Some("numbers"), "f64", text)
}

fn model() -> FemmModel {
    FemmModel {
        id: StableId(7),
        name: sim_kernel::Symbol::new("callable"),
        physics: PhysicsKind::Electrostatic,
        formulation: Formulation::Planar,
        length_unit: LengthUnit::Meter,
        depth: None,
        frequency_hz: None,
        inputs: vec![ParamSpec {
            name: sim_kernel::Symbol::new("gap-mm"),
            default: None,
            unit: None,
            role: ParamRole::Design,
        }],
        geometry: Geometry2::default(),
        materials: Vec::new(),
        boundaries: Vec::new(),
        sources: Vec::new(),
        outputs: Vec::new(),
        mesh_policy: MeshPolicy {
            kind: sim_kernel::Symbol::new("det"),
            max_area: None,
            min_angle_deg: None,
        },
        solve_policy: None,
        origin: dummy_origin(),
    }
}

#[test]
fn projected_scalar_model_returns_expected_value() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let func = femm_as_func(
        model(),
        vec![sim_kernel::Symbol::new("gap-mm")],
        OutputQuery::Quantity(QuantitySpec::Custom {
            name: sim_kernel::Symbol::new("force"),
            expr: num("3.5"),
        }),
    );
    let value = cx
        .call_value(
            cx.factory().opaque(Arc::new(func)).unwrap(),
            sim_kernel::Args::new(vec![
                cx.factory()
                    .number_literal(
                        sim_kernel::Symbol::qualified("numbers", "f64"),
                        "0.4".to_owned(),
                    )
                    .unwrap(),
            ]),
        )
        .unwrap();
    assert_eq!(value.object().display(&mut cx).unwrap(), "3.5");
}

#[test]
fn projected_field_returns_field_domain() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let gap = cx
        .factory()
        .number_literal(
            sim_kernel::Symbol::qualified("numbers", "f64"),
            "0.4".to_owned(),
        )
        .unwrap();
    let eval = ModelCallable {
        model: parallel_plate_capacitor(),
    }
    .eval(
        &mut cx,
        FemmCall {
            params: ParamSet::new(vec![(sim_kernel::Symbol::new("gap-mm"), gap)]),
            query: OutputQuery::Field(Projection::Potential),
            want_grad: None,
            limits: FemmLimits::default(),
        },
    )
    .unwrap();
    let field = eval.value.object().downcast_ref::<Field>().unwrap();
    assert_eq!(
        field.number_domain(&mut cx).unwrap(),
        sim_kernel::Symbol::qualified("numbers", "field")
    );
}

#[test]
fn quality_query_returns_certificate_and_value() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let solve = solve_steady(
        &mut cx,
        &parallel_plate_capacitor(),
        &ParamSet::default(),
        &FemmLimits::default(),
        None,
    )
    .unwrap();
    let quantity_spec = QuantitySpec::Energy { region: None };

    let no_gradient = quality(&mut cx, &solve, &quantity_spec, None).unwrap();
    assert!(no_gradient.value.is_finite());
    assert!(no_gradient.certificate.converged);
    let no_gradient_claim_id = no_gradient
        .certificate
        .claim
        .content_id(cx.datum_store_mut())
        .unwrap();
    assert!(no_gradient_claim_id.bytes.iter().any(|byte| *byte != 0));
    assert!(no_gradient.gradient.is_none());

    let wrt = [Symbol::new("gap-mm")];
    let answer = quality(&mut cx, &solve, &quantity_spec, Some(&wrt)).unwrap();
    assert!(answer.value.is_finite());
    assert!(answer.certificate.converged);
    let Some((gradient, trust)) = &answer.gradient else {
        panic!("expected quality gradient");
    };
    assert_eq!(gradient.len(), 1);
    assert!(gradient.iter().all(|value| value.is_finite()));
    assert_ne!(*trust, GradientTrust::AdjointUnverified);
    assert_eq!(answer.certificate.gradient_trust.as_ref(), Some(trust));

    let rebuilt = certificate_claim(&mut cx, &solve).unwrap();
    let rebuilt_id = rebuilt.content_id(cx.datum_store_mut()).unwrap();
    let carried_id = solve
        .certificate
        .claim
        .content_id(cx.datum_store_mut())
        .unwrap();
    assert_eq!(rebuilt_id, carried_id);
}

#[test]
fn solve_export_record_fills_all_fields() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let solve = solve_steady(
        &mut cx,
        &parallel_plate_capacitor(),
        &ParamSet::default(),
        &FemmLimits::default(),
        None,
    )
    .unwrap();
    let record = SolveExportRecord::from(&solve);
    assert_eq!(record.solution_id, solve.solution.id);
    assert!(!record.physics.is_empty());
    assert!(!record.method.is_empty());
    assert!(record.converged);
    assert!(record.final_residual.is_finite());
    assert_eq!(record.iterations, 1);
    assert!(!record.gradient_trust.is_empty());
    assert!(!record.certificate_claim_key.is_empty());

    let rebuilt = certificate_claim(&mut cx, &solve).unwrap();
    let rebuilt_id = rebuilt.content_id(cx.datum_store_mut()).unwrap();
    assert_eq!(record.certificate_claim_key, content_id_hex(&rebuilt_id));
}

fn content_id_hex(id: &ContentId) -> String {
    id.bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
