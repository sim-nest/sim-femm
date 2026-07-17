#![forbid(unsafe_code)]
//! Field projections and field expressions over a solved model.
//!
//! Defines the derived projections (potential, flux density, field strength,
//! heat flux) and the field-expression algebra sampled from a `FemmSolution`,
//! exposed to the runtime as callable functions.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    AbiVersion, ClassId, ClassRef, Cx, DefaultFactory, Dependency, Expr, Factory, Lib, LibManifest,
    LibTarget, Linker, NumberDomain, NumberLiteral, NumberValue, Object, ObjectEncode,
    ObjectEncoding, Result as KernelResult, Symbol, Value, ValuePromotionRule, Version,
};
use sim_lib_femm_core::{FemmResult, StableId, stable_summary};
use sim_lib_femm_post::{FemmSolution, locate_triangle, sample_gradient, sample_potential};
use sim_lib_numbers_core::DomainNumberValueShape;
use sim_lib_numbers_func::Func;
use sim_shape::shape_value;

/// A scalar projection of a vector or potential solution field.
///
/// Selects which derived quantity a [`Field`] samples: the potential itself, a
/// flux-density or field-strength component, a magnitude, or a custom name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Projection {
    /// The scalar potential (magnetic A, electric V, or temperature).
    Potential,
    /// x component of magnetic flux density.
    Bx,
    /// y component of magnetic flux density.
    By,
    /// Magnitude of magnetic flux density.
    Bmag,
    /// x component of electric field strength.
    Ex,
    /// y component of electric field strength.
    Ey,
    /// Magnitude of electric field strength.
    Emag,
    /// Magnitude of heat flux.
    HeatFluxMag,
    /// A custom projection named by symbol.
    Custom(Symbol),
}

#[derive(Clone)]
enum FieldExpr {
    Base {
        solution: Arc<FemmSolution>,
        projection: Projection,
    },
    AddScalar {
        field: Arc<FieldExpr>,
        scalar: f64,
    },
    AddField {
        left: Arc<FieldExpr>,
        right: Arc<FieldExpr>,
    },
}

/// A sampleable scalar field derived from a solved model.
///
/// A `Field` is a lazily-evaluated projection of a [`FemmSolution`], optionally
/// combined by a small algebra (scalar offset, field sum). It samples at a point
/// via [`at`](Self::at) and is exposed to the runtime as a [`NumberValue`] and a
/// callable [`Func`]. See the
/// [crate README](https://github.com/sim-nest/sim-femm) for the FEM role.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::Symbol;
/// use sim_lib_femm_core::{Formulation, ParamSet, PhysicsKind, StableId};
/// use sim_lib_femm_flow::SolveDiagnostics;
/// use sim_lib_femm_mesh::FemMesh2;
/// use sim_lib_femm_post::FemmSolution;
/// use sim_lib_femm_field::{Field, Projection};
///
/// let solution = Arc::new(FemmSolution {
///     id: StableId(1),
///     model_id: StableId(1),
///     physics: PhysicsKind::Electrostatic,
///     formulation: Formulation::Planar,
///     params: ParamSet::default(),
///     mesh: FemMesh2 {
///         xy: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
///         tri: vec![[0, 1, 2]],
///         elem_region: vec![Symbol::new("air")],
///         edge_boundary: Vec::new(),
///     },
///     u: vec![0.0, 1.0, 1.0],
///     diagnostics: SolveDiagnostics {
///         method: Symbol::new("femm-ptc"),
///         converged: true,
///         iterations: 1,
///         final_residual: 0.0,
///         events: Vec::new(),
///         diagnostics: Vec::new(),
///     },
/// });
/// let field = Field::new(solution, Projection::Potential);
/// assert!((field.at(0.25, 0.25).unwrap() - 0.5).abs() < 1.0e-12);
/// ```
#[derive(Clone)]
pub struct Field {
    expr: Arc<FieldExpr>,
}

impl Field {
    /// Builds a base field that projects `solution` through `projection`.
    pub fn new(solution: Arc<FemmSolution>, projection: Projection) -> Self {
        Self {
            expr: Arc::new(FieldExpr::Base {
                solution,
                projection,
            }),
        }
    }

    /// Returns a new field offset by a constant `scalar` everywhere.
    pub fn add_scalar(&self, scalar: f64) -> Self {
        Self {
            expr: Arc::new(FieldExpr::AddScalar {
                field: self.expr.clone(),
                scalar,
            }),
        }
    }

    /// Returns a new field that is the pointwise sum of this and `right`.
    pub fn add_field(&self, right: &Field) -> Self {
        Self {
            expr: Arc::new(FieldExpr::AddField {
                left: self.expr.clone(),
                right: right.expr.clone(),
            }),
        }
    }

    /// Samples the field at `(x, y)`, evaluating the projection algebra.
    ///
    /// Returns [`FemmError::FieldOutOfDomain`](sim_lib_femm_core::FemmError) when
    /// the point lies outside the mesh.
    pub fn at(&self, x: f64, y: f64) -> FemmResult<f64> {
        match self.expr.as_ref() {
            FieldExpr::Base {
                solution,
                projection,
            } => match projection {
                Projection::Potential => sample_potential(solution, x, y),
                Projection::Bx | Projection::Ex => {
                    let grad = sample_gradient(solution, locate_triangle(solution, [x, y])?.0)?;
                    Ok(grad[0])
                }
                Projection::By | Projection::Ey => {
                    let grad = sample_gradient(solution, locate_triangle(solution, [x, y])?.0)?;
                    Ok(grad[1])
                }
                Projection::Bmag | Projection::Emag | Projection::HeatFluxMag => {
                    let grad = sample_gradient(solution, locate_triangle(solution, [x, y])?.0)?;
                    Ok((grad[0] * grad[0] + grad[1] * grad[1]).sqrt())
                }
                Projection::Custom(_) => sample_potential(solution, x, y),
            },
            FieldExpr::AddScalar { field, scalar } => Field {
                expr: field.clone(),
            }
            .at(x, y)
            .map(|value| value + scalar),
            FieldExpr::AddField { left, right } => Ok(Field { expr: left.clone() }.at(x, y)?
                + Field {
                    expr: right.clone(),
                }
                .at(x, y)?),
        }
    }

    /// Returns the projection of a base field, or a `derived` marker for a
    /// field built by the projection algebra.
    pub fn projection(&self) -> Projection {
        match self.expr.as_ref() {
            FieldExpr::Base { projection, .. } => projection.clone(),
            _ => Projection::Custom(Symbol::new("derived")),
        }
    }

    /// Returns the identity of the underlying [`FemmSolution`].
    pub fn solution_id(&self) -> StableId {
        match self.expr.as_ref() {
            FieldExpr::Base { solution, .. } => solution.id,
            FieldExpr::AddScalar { field, .. } | FieldExpr::AddField { left: field, .. } => Field {
                expr: field.clone(),
            }
            .solution_id(),
        }
    }
}

/// Wraps a [`Field`] as a two-argument callable `(x, y)` runtime function.
///
/// Bridges the field into the `numbers/func` domain so it can be evaluated and
/// promoted through the kernel callable contract.
pub fn field_as_func(field: Field) -> Func {
    Func::native(
        vec![Symbol::new("x"), Symbol::new("y")],
        Arc::new(move |cx, args| {
            let x =
                sim_lib_femm_core::value_as_f64(cx, &args[0]).map_err(sim_kernel::Error::from)?;
            let y =
                sim_lib_femm_core::value_as_f64(cx, &args[1]).map_err(sim_kernel::Error::from)?;
            cx.factory().number_literal(
                Symbol::qualified("numbers", "f64"),
                field.at(x, y).map_err(sim_kernel::Error::from)?.to_string(),
            )
        }),
    )
}

fn field_domain_symbol() -> Symbol {
    Symbol::qualified("numbers", "field")
}

fn field_class_symbol() -> Symbol {
    Symbol::qualified("femm", "Field")
}

fn field_shape_symbol() -> Symbol {
    sim_lib_numbers_core::value_shape_symbol(&field_domain_symbol())
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/field number-domain marker; reconstruct by loading the FEMM field lib",
    kind = "marker",
    descriptor = "numbers/field"
)]
struct FieldDomain;

impl NumberDomain for FieldDomain {
    fn symbol(&self) -> Symbol {
        field_domain_symbol()
    }

    fn parse_priority(&self) -> i32 {
        -100
    }

    fn parse_literal(&self, _cx: &mut Cx, _text: &str) -> KernelResult<Option<Value>> {
        Ok(None)
    }

    fn encode_literal(&self, _cx: &mut Cx, _value: Value) -> KernelResult<Option<NumberLiteral>> {
        Ok(None)
    }
}

impl Object for FieldDomain {
    fn display(&self, _cx: &mut Cx) -> KernelResult<String> {
        Ok("#<number-domain numbers/field>".to_owned())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for FieldDomain {
    fn class(&self, cx: &mut Cx) -> KernelResult<ClassRef> {
        if let Some(class) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "NumberDomain"))
        {
            return Ok(class.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_NUMBER_DOMAIN_CLASS_ID,
            Symbol::qualified("core", "NumberDomain"),
        )
    }
    fn as_expr(&self, _cx: &mut Cx) -> KernelResult<Expr> {
        Ok(Expr::Symbol(field_domain_symbol()))
    }
    fn as_number_domain(&self) -> Option<&dyn NumberDomain> {
        Some(self)
    }
}

impl Object for Field {
    fn display(&self, _cx: &mut Cx) -> KernelResult<String> {
        Ok(format!(
            "#(femm/Field v1 {} \"{}\")",
            self.solution_id().0,
            projection_name(&self.projection())
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for Field {
    fn class(&self, cx: &mut Cx) -> KernelResult<ClassRef> {
        if let Some(class) = cx.registry().class_by_symbol(&field_class_symbol()) {
            return Ok(class.clone());
        }
        DefaultFactory.class_stub(ClassId(31), field_class_symbol())
    }
    fn as_expr(&self, cx: &mut Cx) -> KernelResult<Expr> {
        sim_citizen::constructor_expr(cx, self)
    }
    fn as_table(&self, cx: &mut Cx) -> KernelResult<Value> {
        cx.factory().table(vec![
            (
                Symbol::new("kind"),
                cx.factory().string("field".to_owned())?,
            ),
            (
                Symbol::new("summary"),
                cx.factory().string(stable_summary(
                    "Field",
                    &[
                        ("solution", self.solution_id().0.to_string()),
                        ("projection", format!("{:?}", self.projection())),
                    ],
                ))?,
            ),
        ])
    }
    fn as_number_value(&self) -> Option<&dyn NumberValue> {
        Some(self)
    }
    fn as_object_encoder(&self) -> Option<&dyn ObjectEncode> {
        Some(self)
    }
}

impl ObjectEncode for Field {
    fn object_encoding(&self, _cx: &mut Cx) -> KernelResult<ObjectEncoding> {
        Ok(ObjectEncoding::Constructor {
            class: field_class_symbol(),
            args: field_constructor_args(self),
        })
    }
}

impl sim_citizen::Citizen for Field {
    fn citizen_symbol() -> Symbol {
        field_class_symbol()
    }

    fn citizen_version() -> u32 {
        1
    }

    fn citizen_arity() -> usize {
        2
    }

    fn citizen_fields() -> &'static [&'static str] {
        &["solution_id", "projection"]
    }
}

fn field_constructor_args(field: &Field) -> Vec<Expr> {
    vec![
        Expr::Symbol(Symbol::new("v1")),
        Expr::Number(sim_kernel::NumberLiteral {
            domain: Symbol::qualified("citizen", "int"),
            canonical: field.solution_id().0.to_string(),
        }),
        Expr::String(projection_name(&field.projection())),
    ]
}

fn projection_name(projection: &Projection) -> String {
    match projection {
        Projection::Potential => "potential".to_owned(),
        Projection::Bx => "bx".to_owned(),
        Projection::By => "by".to_owned(),
        Projection::Bmag => "bmag".to_owned(),
        Projection::Ex => "ex".to_owned(),
        Projection::Ey => "ey".to_owned(),
        Projection::Emag => "emag".to_owned(),
        Projection::HeatFluxMag => "heat-flux-mag".to_owned(),
        Projection::Custom(symbol) => symbol.to_string(),
    }
}

impl NumberValue for Field {
    fn number_domain(&self, _cx: &mut Cx) -> KernelResult<Symbol> {
        Ok(field_domain_symbol())
    }
}

/// The loadable library that registers the FEMM field number domain.
///
/// Implements the kernel [`Lib`] contract: it exports the `numbers/field`
/// number domain, the `femm/Field` class, the field value shape, and a
/// promotion rule from fields to `numbers/func`. The behavior lives here; the
/// kernel only supplies the loading contract.
pub struct FemmFieldLib;

impl FemmFieldLib {
    /// Constructs the field library.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FemmFieldLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for FemmFieldLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("femm", "field"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![Dependency {
                id: Symbol::qualified("numbers", "func"),
                minimum_version: None,
            }],
            capabilities: Vec::new(),
            exports: vec![
                sim_kernel::Export::NumberDomain {
                    symbol: field_domain_symbol(),
                    number_domain_id: None,
                },
                sim_kernel::Export::Class {
                    symbol: field_class_symbol(),
                    class_id: None,
                },
                sim_kernel::Export::Shape {
                    symbol: field_shape_symbol(),
                    shape_id: None,
                },
            ],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.number_domain_value(
            field_domain_symbol(),
            DefaultFactory.opaque(Arc::new(FieldDomain))?,
        )?;
        linker.shape_value(
            field_shape_symbol(),
            shape_value(
                field_shape_symbol(),
                Arc::new(DomainNumberValueShape::new(
                    field_domain_symbol(),
                    "FieldValue",
                    ["field-valued number in the numbers/field domain"],
                )),
            ),
        )?;
        linker.value_promotion_rule(ValuePromotionRule {
            from_domain: field_domain_symbol(),
            to_domain: Symbol::qualified("numbers", "func"),
            cost: 2,
            convert: |cx, value| {
                let field = value
                    .object()
                    .downcast_ref::<Field>()
                    .ok_or_else(|| sim_kernel::Error::Eval("expected field".to_owned()))?
                    .clone();
                cx.factory().opaque(Arc::new(field_as_func(field)))
            },
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use sim_kernel::Symbol;
    use sim_lib_femm_core::{Formulation, ParamSet, PhysicsKind, StableId};
    use sim_lib_femm_flow::SolveDiagnostics;
    use sim_lib_femm_mesh::FemMesh2;
    use sim_lib_femm_post::FemmSolution;

    use super::*;

    fn solution() -> Arc<FemmSolution> {
        Arc::new(FemmSolution {
            id: StableId(1),
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
        })
    }

    #[test]
    fn linear_field_samples_exactly() {
        let field = Field::new(solution(), Projection::Potential);
        assert!((field.at(0.25, 0.25).unwrap() - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn adding_scalar_offsets_samples() {
        let field = Field::new(solution(), Projection::Potential).add_scalar(1.0);
        assert!((field.at(0.25, 0.25).unwrap() - 1.5).abs() < 1.0e-12);
    }
}
