use std::sync::Arc;

use sim_codec::{Input, Output, decode_with_codec, encode_with_codec};
use sim_kernel::{
    CapabilitySet, Cx, DefaultFactory, EagerPolicy, EncodeOptions, Expr, Factory, NumberLiteral,
    ObjectCompat, ObjectEncode, ObjectEncoding, ReadPolicy, Symbol, TrustLevel,
    read_construct_capability,
};
use sim_lib_femm_core::{Formulation, ParamSet, PhysicsKind, StableId};
use sim_lib_femm_field::{Field, Projection};
use sim_lib_femm_flow::SolveDiagnostics;
use sim_lib_femm_function::{FemmFuncPayload, OutputQuery, model_value};
use sim_lib_femm_material::Material;
use sim_lib_femm_mesh::FemMesh2;
use sim_lib_femm_post::{FemmSolution, QuantitySpec};
use sim_lib_femm_solve::LinearMethod;

use crate::support::{formulation_name, physics_name};
use crate::{
    FemmFieldDescriptor, FemmMaterialDescriptor, FemmMeshDescriptor, FemmPostDescriptor,
    FemmSolveDescriptor, FieldSummary, ModelSummary, SolutionSummary, femm_field_class_symbol,
    femm_material_class_symbol, femm_mesh_class_symbol, femm_post_class_symbol,
    femm_solve_class_symbol, field_from_read_construct, field_read_construct, model_from_json,
    model_from_lisp, model_to_json, model_to_lisp, reject_unknown_binary_tag, solution_from_json,
    solution_from_lisp, solution_to_json, solution_to_lisp,
};

#[test]
fn binary_rejects_unknown_tags() {
    assert!(reject_unknown_binary_tag(0x00).is_err());
}

#[test]
fn field_read_construct_round_trips_textually() {
    let field = Field::new(
        Arc::new(FemmSolution {
            id: StableId(2),
            model_id: StableId(1),
            physics: PhysicsKind::Electrostatic,
            formulation: Formulation::Planar,
            params: ParamSet::default(),
            mesh: FemMesh2 {
                xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                tri: vec![[0, 1, 2]],
                elem_region: vec![Symbol::new("air")],
                edge_boundary: Vec::new(),
            },
            u: vec![0.0, 1.0, 1.0],
            diagnostics: SolveDiagnostics {
                method: Symbol::new("femm-ptc"),
                converged: true,
                iterations: 1,
                final_residual: 0.0,
                events: Vec::new(),
                diagnostics: Vec::new(),
            },
        }),
        Projection::Potential,
    );
    let text = field_read_construct(&field);
    assert_eq!(text, "#(femm/Field v1 2 \"potential\")");
    assert_eq!(
        field_from_read_construct(&text).unwrap(),
        FieldSummary {
            solution_id: 2,
            projection: "potential".to_owned(),
        }
    );
    assert_eq!(
        field_from_read_construct("#(femm/field 2 Potential)").unwrap(),
        FieldSummary {
            solution_id: 2,
            projection: "Potential".to_owned(),
        }
    );
}

#[test]
fn lisp_and_json_model_round_trip_preserve_example_summaries() {
    for model in sim_lib_femm_fixtures::fixture_models() {
        let expected = ModelSummary {
            id: model.id.0,
            name: model.name.to_string(),
            physics: physics_name(&model.physics).to_owned(),
            formulation: formulation_name(&model.formulation).to_owned(),
            params: model
                .inputs
                .iter()
                .map(|param| param.name.to_string())
                .collect(),
        };
        assert_eq!(model_from_lisp(&model_to_lisp(&model)).unwrap(), expected);
        assert_eq!(model_from_json(&model_to_json(&model)).unwrap(), expected);
    }
}

#[test]
fn solution_lisp_and_json_round_trip_public_summary() {
    let solution = FemmSolution {
        id: StableId(20),
        model_id: StableId(10),
        physics: PhysicsKind::HeatSteady,
        formulation: Formulation::Axisymmetric,
        params: ParamSet::new(vec![(Symbol::new("current-a"), value_symbol("two"))]),
        mesh: FemMesh2 {
            xy: vec![[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        },
        u: vec![0.0, 1.0, 1.0],
        diagnostics: SolveDiagnostics {
            method: Symbol::new("femm-ptc"),
            converged: true,
            iterations: 1,
            final_residual: 0.0,
            events: Vec::new(),
            diagnostics: Vec::new(),
        },
    };
    let expected = SolutionSummary {
        id: 20,
        model_id: 10,
        physics: "heat-steady".to_owned(),
        formulation: "axisymmetric".to_owned(),
        params: vec!["current-a".to_owned()],
    };
    assert_eq!(
        solution_from_lisp(&solution_to_lisp(&solution)).unwrap(),
        expected
    );
    assert_eq!(
        solution_from_json(&solution_to_json(&solution)).unwrap(),
        expected
    );
}

#[test]
fn public_femm_object_descriptors_round_trip_through_installed_codecs() {
    let mut cx = codec_cx();
    let model = sim_lib_femm_fixtures::parallel_plate_capacitor();
    let solution = Arc::new(FemmSolution {
        id: StableId(30),
        model_id: model.id,
        physics: model.physics.clone(),
        formulation: model.formulation.clone(),
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        },
        u: vec![0.0, 1.0, 1.0],
        diagnostics: SolveDiagnostics {
            method: Symbol::new("femm-ptc"),
            converged: true,
            iterations: 1,
            final_residual: 0.0,
            events: Vec::new(),
            diagnostics: Vec::new(),
        },
    });
    let descriptors = vec![
        model_value(model.clone()).as_expr(&mut cx).unwrap(),
        solution.as_expr(&mut cx).unwrap(),
        Field::new(solution, Projection::Potential)
            .as_expr(&mut cx)
            .unwrap(),
        FemmFuncPayload {
            model,
            vars: vec![Symbol::new("gap-mm")],
            query: OutputQuery::Quantity(QuantitySpec::Energy { region: None }),
        }
        .as_expr(&mut cx)
        .unwrap(),
    ];
    for descriptor in descriptors {
        assert_installed_codec_round_trip(&mut cx, &descriptor);
    }
}

#[test]
fn femm_citizens_construct_required_fixtures() {
    let mut cx = codec_cx();
    let solution = solution_fixture(30, 10);
    let field = Field::new(solution.clone(), Projection::Potential);
    let field_descriptor = FemmFieldDescriptor::from_field(&field);
    let mesh_descriptor = FemmMeshDescriptor::from_mesh(&solution.mesh, "table:femm/mesh/tiny");
    let material = Material {
        name: Symbol::new("air"),
        mu_r: Some(num_expr("1.0")),
        nu_of_b2: None,
        epsilon_r: Some(num_expr("1.0")),
        sigma: None,
        thermal_k: None,
        heat_source: None,
        remanence: None,
    };
    let material_descriptor = FemmMaterialDescriptor {
        name: material.name.to_string(),
        properties: vec!["epsilon-r".to_owned(), "mu-r".to_owned()],
    };
    let solve_descriptor = FemmSolveDescriptor {
        method: linear_method_name(&LinearMethod::SparseLu).to_owned(),
        matrix_ref: "stable:femm/matrix/tiny".to_owned(),
    };
    let post_descriptor = FemmPostDescriptor {
        quantity: "energy:air".to_owned(),
        target: "solution:30".to_owned(),
    };

    let field_args = vec![
        version(&mut cx),
        int_value(&mut cx, field_descriptor.solution_id),
        string_value(&mut cx, &field_descriptor.projection),
    ];
    assert_read_constructs_to(
        &mut cx,
        femm_field_class_symbol(),
        field_args,
        &field_descriptor,
    );

    let mesh_args = vec![
        version(&mut cx),
        int_value(&mut cx, mesh_descriptor.nodes),
        int_value(&mut cx, mesh_descriptor.elements),
        string_value(&mut cx, &mesh_descriptor.artifact_ref),
    ];
    assert_read_constructs_to(
        &mut cx,
        femm_mesh_class_symbol(),
        mesh_args,
        &mesh_descriptor,
    );

    let material_args = vec![
        version(&mut cx),
        string_value(&mut cx, &material_descriptor.name),
        string_list_value(&mut cx, &material_descriptor.properties),
    ];
    assert_read_constructs_to(
        &mut cx,
        femm_material_class_symbol(),
        material_args,
        &material_descriptor,
    );

    let solve_args = vec![
        version(&mut cx),
        string_value(&mut cx, &solve_descriptor.method),
        string_value(&mut cx, &solve_descriptor.matrix_ref),
    ];
    assert_read_constructs_to(
        &mut cx,
        femm_solve_class_symbol(),
        solve_args,
        &solve_descriptor,
    );

    let post_args = vec![
        version(&mut cx),
        string_value(&mut cx, &post_descriptor.quantity),
        string_value(&mut cx, &post_descriptor.target),
    ];
    assert_read_constructs_to(
        &mut cx,
        femm_post_class_symbol(),
        post_args,
        &post_descriptor,
    );
}

#[test]
fn public_femm_objects_encode_as_constructor_descriptors() {
    let mut cx = codec_cx();
    let model = sim_lib_femm_fixtures::parallel_plate_capacitor();
    let solution = solution_fixture(40, model.id.0);
    for encoding in [
        model_value(model.clone()).object_encoding(&mut cx).unwrap(),
        solution.object_encoding(&mut cx).unwrap(),
        Field::new(solution.clone(), Projection::Potential)
            .object_encoding(&mut cx)
            .unwrap(),
        FemmFuncPayload {
            model,
            vars: vec![Symbol::new("gap-mm")],
            query: OutputQuery::Quantity(QuantitySpec::Energy { region: None }),
        }
        .object_encoding(&mut cx)
        .unwrap(),
    ] {
        assert!(
            matches!(encoding, ObjectEncoding::Constructor { .. }),
            "FEMM public object encoded as non-constructor"
        );
    }
}

fn solution_fixture(id: u64, model_id: u64) -> Arc<FemmSolution> {
    Arc::new(FemmSolution {
        id: StableId(id),
        model_id: StableId(model_id),
        physics: PhysicsKind::Electrostatic,
        formulation: Formulation::Planar,
        params: ParamSet::default(),
        mesh: FemMesh2 {
            xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            tri: vec![[0, 1, 2]],
            elem_region: vec![Symbol::new("air")],
            edge_boundary: Vec::new(),
        },
        u: vec![0.0, 1.0, 1.0],
        diagnostics: SolveDiagnostics {
            method: Symbol::new("femm-ptc"),
            converged: true,
            iterations: 1,
            final_residual: 0.0,
            events: Vec::new(),
            diagnostics: Vec::new(),
        },
    })
}

fn value_symbol(name: &str) -> sim_kernel::Value {
    sim_kernel::DefaultFactory
        .symbol(Symbol::new(name))
        .expect("symbol value")
}

fn codec_cx() -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.grant(read_construct_capability());
    cx.load_lib(&sim_citizen::CitizenLib::namespace("femm"))
        .unwrap();
    let lisp = sim_codec_lisp::LispCodecLib::new(cx.registry_mut().fresh_codec_id()).unwrap();
    cx.load_lib(&lisp).unwrap();
    let json = sim_codec_json::JsonCodecLib::new(cx.registry_mut().fresh_codec_id());
    cx.load_lib(&json).unwrap();
    let binary = sim_codec_binary::BinaryCodecLib::new(cx.registry_mut().fresh_codec_id());
    cx.load_lib(&binary).unwrap();
    let binary_base64 =
        sim_codec_binary_base64::BinaryBase64CodecLib::new(cx.registry_mut().fresh_codec_id());
    cx.load_lib(&binary_base64).unwrap();
    let algol = sim_codec_algol::AlgolCodecLib::new(cx.registry_mut().fresh_codec_id());
    cx.load_lib(&algol).unwrap();
    cx
}

fn assert_installed_codec_round_trip(cx: &mut Cx, expr: &sim_kernel::Expr) {
    for codec in [
        Symbol::qualified("codec", "lisp"),
        Symbol::qualified("codec", "json"),
        Symbol::qualified("codec", "binary"),
        Symbol::qualified("codec", "binary-base64"),
        Symbol::qualified("codec", "algol"),
    ] {
        let encoded = encode_with_codec(cx, &codec, expr, EncodeOptions::default()).unwrap();
        let input = match encoded {
            Output::Text(text) => Input::Text(text),
            Output::Bytes(bytes) => Input::Bytes(bytes),
        };
        let decoded = decode_with_codec(cx, &codec, input, read_policy_with_construct()).unwrap();
        assert_eq!(decoded, *expr, "codec {codec}");
    }
}

fn assert_read_constructs_to<T>(
    cx: &mut Cx,
    class: Symbol,
    args: Vec<sim_kernel::Value>,
    expected: &T,
) where
    T: Clone + PartialEq + std::fmt::Debug + 'static,
{
    let decoded = cx.read_construct(&class, args).unwrap();
    assert_eq!(
        decoded.object().downcast_ref::<T>(),
        Some(expected),
        "read construct {class}"
    );
}

fn version(cx: &mut Cx) -> sim_kernel::Value {
    cx.factory().symbol(Symbol::new("v1")).unwrap()
}

fn int_value(cx: &mut Cx, value: impl ToString) -> sim_kernel::Value {
    cx.factory()
        .number_literal(Symbol::qualified("citizen", "int"), value.to_string())
        .unwrap()
}

fn string_value(cx: &mut Cx, value: &str) -> sim_kernel::Value {
    cx.factory().string(value.to_owned()).unwrap()
}

fn string_list_value(cx: &mut Cx, values: &[String]) -> sim_kernel::Value {
    let values = values
        .iter()
        .map(|value| string_value(cx, value))
        .collect::<Vec<_>>();
    cx.factory().list(values).unwrap()
}

fn num_expr(text: &str) -> Expr {
    Expr::Number(NumberLiteral {
        domain: Symbol::qualified("numbers", "f64"),
        canonical: text.to_owned(),
    })
}

fn linear_method_name(method: &LinearMethod) -> &'static str {
    match method {
        LinearMethod::Cg => "cg",
        LinearMethod::Bicgstab => "bicgstab",
        LinearMethod::SparseLu => "sparse-lu",
    }
}

fn read_policy_with_construct() -> ReadPolicy {
    ReadPolicy {
        trust: TrustLevel::TrustedSource,
        capabilities: CapabilitySet::new().grant(read_construct_capability()),
    }
}
