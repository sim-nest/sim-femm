use std::sync::Arc;

use sim_kernel::{Cx, DefaultFactory, EagerPolicy, Expr, NumberValue, Symbol};
use sim_lib_femm_core::{
    FemmLimits, Formulation, LengthUnit, ParamRole, ParamSet, ParamSpec, PhysicsKind, StableId,
};
use sim_lib_femm_field::{Field, Projection};
use sim_lib_femm_fixtures::parallel_plate_capacitor;
use sim_lib_femm_geometry::{AnalyticRegion2, Geometry2, dummy_origin};
use sim_lib_femm_material::MeshPolicy;
use sim_lib_femm_mesh::FemmModel;
use sim_lib_femm_post::QuantitySpec;

use crate::{FemmCall, FemmCallable, FemmFuncPayload, ModelCallable, OutputQuery, femm_as_func};

fn num(text: &str) -> Expr {
    sim_value::build::num_q(Some("numbers"), "f64", text)
}

fn model() -> FemmModel {
    FemmModel {
        id: StableId(7),
        name: Symbol::new("callable"),
        physics: PhysicsKind::Electrostatic,
        formulation: Formulation::Planar,
        length_unit: LengthUnit::Meter,
        depth: None,
        frequency_hz: None,
        inputs: vec![ParamSpec {
            name: Symbol::new("gap-mm"),
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
            kind: Symbol::new("det"),
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
        vec![Symbol::new("gap-mm")],
        OutputQuery::Quantity(QuantitySpec::Custom {
            name: Symbol::new("force"),
            expr: num("3.5"),
        }),
    );
    let value = cx
        .call_value(
            cx.factory().opaque(Arc::new(func)).unwrap(),
            sim_kernel::Args::new(vec![
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "f64"), "0.4".to_owned())
                    .unwrap(),
            ]),
        )
        .unwrap();
    assert_eq!(value.object().display(&mut cx).unwrap(), "3.5");
}

#[test]
fn femm_as_func_carries_adjoint_payload() {
    let func = femm_as_func(
        model(),
        vec![Symbol::new("gap-mm")],
        OutputQuery::Quantity(QuantitySpec::Custom {
            name: Symbol::new("force"),
            expr: num("3.5"),
        }),
    );

    assert_eq!(
        func.metadata.differentiator_hint,
        Some(Symbol::new("femm-adjoint"))
    );
    assert!(
        func.metadata
            .payload
            .as_ref()
            .and_then(|value| value.object().downcast_ref::<FemmFuncPayload>())
            .is_some()
    );
}

#[test]
fn projected_field_returns_field_domain() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let gap = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "f64"), "0.4".to_owned())
        .unwrap();
    let eval = ModelCallable {
        model: parallel_plate_capacitor(),
    }
    .eval(
        &mut cx,
        FemmCall {
            params: ParamSet::new(vec![(Symbol::new("gap-mm"), gap)]),
            query: OutputQuery::Field(Projection::Potential),
            want_grad: None,
            limits: FemmLimits::default(),
        },
    )
    .unwrap();
    let field = eval.value.object().downcast_ref::<Field>().unwrap();
    assert_eq!(
        field.number_domain(&mut cx).unwrap(),
        Symbol::qualified("numbers", "field")
    );
}

#[test]
fn field_func_requires_real_solution() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let func = crate::femm_field_func(unmeshable_model());
    let x = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "f64"), "0.25".to_owned())
        .unwrap();
    let y = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "f64"), "0.25".to_owned())
        .unwrap();
    let err = cx
        .call_value(
            cx.factory().opaque(Arc::new(func)).unwrap(),
            sim_kernel::Args::new(vec![x, y]),
        )
        .unwrap_err();

    assert!(err.to_string().contains("invalid-geometry"));
}
