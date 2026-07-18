use std::sync::Arc;

use sim_kernel::{ContentId, Cx, DefaultFactory, EagerPolicy, Expr, NumberValue, Symbol};
use sim_lib_femm_core::{
    FemmLimits, Formulation, LengthUnit, ParamRole, ParamSet, ParamSpec, PhysicsKind, StableId,
};
use sim_lib_femm_field::{Field, Projection};
use sim_lib_femm_fixtures::parallel_plate_capacitor;
use sim_lib_femm_geometry::{AnalyticRegion2, Geometry2, dummy_origin};
use sim_lib_femm_material::MeshPolicy;
use sim_lib_femm_mesh::FemmModel;
use sim_lib_femm_post::QuantitySpec;
use sim_lib_femm_sensitiv::gradient_trust_label;
use sim_lib_femm_solve::{SolveExportRecord, certificate_claim, solve_steady};
use sim_lib_numbers_numeric::global_numeric_registry;

use crate::{
    FemmCall, FemmCallable, FemmFunctionLib, ModelCallable, OutputQuery, femm_as_func,
    femm_field_func, quality,
};

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

fn unmeshable_model() -> FemmModel {
    let mut model = parallel_plate_capacitor();
    model.formulation = Formulation::Axisymmetric;
    model.geometry.analytic = vec![AnalyticRegion2::Rect {
        name: Symbol::new("air"),
        xy: [num("-1.0"), num("0.0")],
        wh: [num("1.0"), num("1.0")],
    }];
    model
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
fn femm_as_func_still_callable_and_diffable() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let func = femm_as_func(
        model(),
        vec![sim_kernel::Symbol::new("gap-mm")],
        OutputQuery::Quantity(QuantitySpec::Custom {
            name: sim_kernel::Symbol::new("force"),
            expr: num("3.5"),
        }),
    );

    assert_eq!(
        func.metadata.differentiator_hint,
        Some(sim_kernel::Symbol::new("femm-adjoint"))
    );
    assert!(
        func.metadata
            .payload
            .as_ref()
            .and_then(|value| value.object().downcast_ref::<crate::FemmFuncPayload>())
            .is_some()
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
fn direct_function_load_registers_adjoint_hint() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&FemmFunctionLib::new()).unwrap();
    let guard = global_numeric_registry().read().unwrap();
    assert!(guard.differentiator(&Symbol::new("femm-adjoint")).is_some());
}

#[test]
fn femm_grad_returns_values_and_trust() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&FemmFunctionLib::new()).unwrap();
    let model = cx
        .call_function(
            &Symbol::qualified("femm", "model"),
            sim_kernel::Args::default(),
        )
        .unwrap();
    let grad = cx
        .call_function(
            &Symbol::qualified("femm", "grad"),
            sim_kernel::Args::new(vec![
                model,
                cx.factory()
                    .expr(Expr::Symbol(Symbol::new("gap-mm")))
                    .unwrap(),
                cx.factory()
                    .list(vec![cx.factory().symbol(Symbol::new("gap-mm")).unwrap()])
                    .unwrap(),
                cx.factory()
                    .list(vec![
                        cx.factory()
                            .list(vec![
                                cx.factory().symbol(Symbol::new("gap-mm")).unwrap(),
                                cx.factory()
                                    .number_literal(
                                        Symbol::qualified("numbers", "f64"),
                                        "0.5".to_owned(),
                                    )
                                    .unwrap(),
                            ])
                            .unwrap(),
                    ])
                    .unwrap(),
            ]),
        )
        .unwrap();
    let Expr::Map(entries) = grad.object().as_expr(&mut cx).unwrap() else {
        panic!("expected gradient answer table");
    };
    assert!(map_entry(&entries, "gradient").is_some());
    assert_eq!(
        map_entry(&entries, "trust"),
        Some(&Expr::String("adjoint-verified".to_owned()))
    );
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
fn field_func_requires_real_solution() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let func = femm_field_func(unmeshable_model());
    let x = cx
        .factory()
        .number_literal(
            sim_kernel::Symbol::qualified("numbers", "f64"),
            "0.25".to_owned(),
        )
        .unwrap();
    let y = cx
        .factory()
        .number_literal(
            sim_kernel::Symbol::qualified("numbers", "f64"),
            "0.25".to_owned(),
        )
        .unwrap();
    let err = cx
        .call_value(
            cx.factory().opaque(Arc::new(func)).unwrap(),
            sim_kernel::Args::new(vec![x, y]),
        )
        .unwrap_err();

    assert!(err.to_string().contains("invalid-geometry"));
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
    assert!(matches!(
        gradient_trust_label(trust),
        "adjoint-verified" | "finite-difference-only"
    ));
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

fn map_entry<'a>(entries: &'a [(Expr, Expr)], key: &str) -> Option<&'a Expr> {
    entries
        .iter()
        .find_map(|(entry_key, value)| match entry_key {
            Expr::Symbol(symbol) if symbol == &Symbol::new(key) => Some(value),
            _ => None,
        })
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
