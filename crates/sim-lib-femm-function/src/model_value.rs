//! The runtime object wrapping a FEMM model as a first-class value.
//!
//! Defines the model value and its kernel object integration so a model can be
//! held, displayed, and dispatched on inside the runtime.

use std::any::Any;

use sim_kernel::{
    ClassId, ClassRef, Cx, DefaultFactory, Expr, Factory, Object, ObjectEncode, ObjectEncoding,
    Result as KernelResult, Symbol, Value,
};
use sim_lib_femm_core::{Formulation, PhysicsKind};
use sim_lib_femm_mesh::FemmModel;

/// A FEMM model held as a first-class runtime object.
///
/// Wraps a [`FemmModel`] so it can be stored in a [`Value`], displayed,
/// encoded, and dispatched on inside the runtime; the model itself stays
/// behavior defined by this constellation, while the kernel supplies the
/// object/encoding contracts. See the [crate README](index.html).
///
/// # Examples
///
/// ```
/// use sim_lib_femm_fixtures::parallel_plate_capacitor;
/// use sim_lib_femm_function::model_value;
///
/// let value = model_value(parallel_plate_capacitor());
/// assert_eq!(value.model.name.as_qualified_str(), "parallel-plate-capacitor");
/// ```
#[derive(Clone)]
pub struct ModelValue {
    /// The wrapped finite-element model.
    pub model: FemmModel,
}

impl Object for ModelValue {
    fn display(&self, _cx: &mut Cx) -> KernelResult<String> {
        Ok(format!(
            "#<femm-model {}:{}>",
            self.model.id.0, self.model.name
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for ModelValue {
    fn class(&self, cx: &mut Cx) -> KernelResult<ClassRef> {
        if let Some(class) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("femm", "Model"))
        {
            return Ok(class.clone());
        }
        DefaultFactory.class_stub(ClassId(34), Symbol::qualified("femm", "Model"))
    }
    fn as_expr(&self, cx: &mut Cx) -> KernelResult<Expr> {
        sim_citizen::constructor_expr(cx, self)
    }
    fn as_table(&self, cx: &mut Cx) -> KernelResult<Value> {
        cx.factory().table(vec![
            (
                Symbol::new("kind"),
                cx.factory().string("femm-model".to_owned())?,
            ),
            (
                Symbol::new("id"),
                cx.factory().string(self.model.id.0.to_string())?,
            ),
            (
                Symbol::new("name"),
                cx.factory().string(self.model.name.to_string())?,
            ),
            (
                Symbol::new("query-kind"),
                cx.factory().string("solution".to_owned())?,
            ),
        ])
    }
    fn as_object_encoder(&self) -> Option<&dyn ObjectEncode> {
        Some(self)
    }
}

impl ObjectEncode for ModelValue {
    fn object_encoding(&self, _cx: &mut Cx) -> KernelResult<ObjectEncoding> {
        Ok(ObjectEncoding::Constructor {
            class: model_class_symbol(),
            args: model_constructor_args(self),
        })
    }
}

impl sim_citizen::Citizen for ModelValue {
    fn citizen_symbol() -> Symbol {
        model_class_symbol()
    }

    fn citizen_version() -> u32 {
        1
    }

    fn citizen_arity() -> usize {
        5
    }

    fn citizen_fields() -> &'static [&'static str] {
        &["id", "name", "physics", "formulation", "params"]
    }
}

/// Wraps a model as a [`ModelValue`] runtime object.
pub fn model_value(model: FemmModel) -> ModelValue {
    ModelValue { model }
}

fn model_class_symbol() -> Symbol {
    Symbol::qualified("femm", "Model")
}

fn model_constructor_args(value: &ModelValue) -> Vec<Expr> {
    vec![
        Expr::Symbol(Symbol::new("v1")),
        int_expr(value.model.id.0),
        Expr::String(value.model.name.to_string()),
        Expr::String(physics_name(&value.model.physics).to_owned()),
        Expr::String(formulation_name(&value.model.formulation).to_owned()),
        Expr::List(
            value
                .model
                .inputs
                .iter()
                .map(|param| Expr::String(param.name.to_string()))
                .collect(),
        ),
    ]
}

fn int_expr(value: impl ToString) -> Expr {
    Expr::Number(sim_kernel::NumberLiteral {
        domain: Symbol::qualified("citizen", "int"),
        canonical: value.to_string(),
    })
}

fn physics_name(physics: &PhysicsKind) -> &'static str {
    match physics {
        PhysicsKind::Magnetostatic => "magnetostatic",
        PhysicsKind::MagneticsHarmonic => "magnetics-harmonic",
        PhysicsKind::Electrostatic => "electrostatic",
        PhysicsKind::HeatSteady => "heat-steady",
        PhysicsKind::CurrentSteady => "current-steady",
    }
}

fn formulation_name(formulation: &Formulation) -> &'static str {
    match formulation {
        Formulation::Planar => "planar",
        Formulation::Axisymmetric => "axisymmetric",
    }
}
