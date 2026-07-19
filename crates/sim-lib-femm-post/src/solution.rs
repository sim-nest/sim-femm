use std::{any::Any, sync::Arc};

use sim_kernel::{
    ClassId, ClassRef, Cx, DefaultFactory, Expr, Factory, Object, ObjectEncode, ObjectEncoding,
    Symbol, Value,
};
use sim_lib_femm_core::{
    FemmError, FemmResult, Formulation, ParamSet, PhysicsKind, StableId, stable_summary,
};
use sim_lib_femm_flow::SolveDiagnostics;
use sim_lib_femm_mesh::FemMesh2;

/// A solved FEMM problem: the mesh paired with its nodal solution.
///
/// The output of the solver and the input to post-processing. It carries the
/// mesh, the per-node degree-of-freedom values (`u`), and enough provenance to
/// interpret them. `FemmSolution` is a runtime [`Object`]: it round-trips as a
/// constructor over the kernel [`Expr`] contract via [`Citizen`]. See the
/// [crate README](https://github.com/sim-nest/sim-femm) for the FEM role.
///
/// [`Citizen`]: sim_citizen::Citizen
#[derive(Clone, Debug)]
pub struct FemmSolution {
    /// Stable identity of this solution.
    pub id: StableId,
    /// Identity of the model that was solved.
    pub model_id: StableId,
    /// Physics that was solved.
    pub physics: PhysicsKind,
    /// Geometric formulation (planar or axisymmetric).
    pub formulation: Formulation,
    /// Parameter values used for the solve.
    pub params: ParamSet,
    /// The triangular mesh the solution is defined on.
    pub mesh: FemMesh2,
    /// Per-node degree-of-freedom values, parallel to the mesh nodes.
    pub u: Vec<f64>,
    /// Diagnostics emitted by the solver.
    pub diagnostics: SolveDiagnostics,
}

impl FemmSolution {
    /// Validates mesh/solution cardinality and finite scalar values.
    pub fn validate(&self) -> FemmResult<()> {
        self.mesh.validate()?;
        if self.u.len() != self.mesh.xy.len() {
            return Err(FemmError::InvalidGeometry(format!(
                "solution has {} values but {} mesh nodes",
                self.u.len(),
                self.mesh.xy.len()
            )));
        }
        for (index, value) in self.u.iter().enumerate() {
            if !value.is_finite() {
                return Err(FemmError::InvalidGeometry(format!(
                    "solution value {index} is non-finite"
                )));
            }
        }
        if !self.diagnostics.final_residual.is_finite() {
            return Err(FemmError::InvalidGeometry(
                "solution final residual is non-finite".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Object for FemmSolution {
    fn display(&self, _cx: &mut Cx) -> sim_kernel::Result<String> {
        Ok(stable_summary(
            "FemmSolution",
            &[
                ("id", self.id.0.to_string()),
                ("model", self.model_id.0.to_string()),
            ],
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for FemmSolution {
    fn class(&self, cx: &mut Cx) -> sim_kernel::Result<ClassRef> {
        if let Some(class) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("femm", "Solution"))
        {
            return Ok(class.clone());
        }
        DefaultFactory.class_stub(ClassId(32), Symbol::qualified("femm", "Solution"))
    }

    fn as_expr(&self, cx: &mut Cx) -> sim_kernel::Result<Expr> {
        sim_citizen::constructor_expr(cx, self)
    }

    fn as_table(&self, cx: &mut Cx) -> sim_kernel::Result<Value> {
        cx.factory().table(vec![
            (
                Symbol::new("kind"),
                cx.factory().string("femm-solution".to_owned())?,
            ),
            (
                Symbol::new("id"),
                cx.factory().string(self.id.0.to_string())?,
            ),
            (
                Symbol::new("summary"),
                cx.factory().string(stable_summary(
                    "FemmSolution",
                    &[("id", self.id.0.to_string())],
                ))?,
            ),
        ])
    }

    fn as_object_encoder(&self) -> Option<&dyn ObjectEncode> {
        Some(self)
    }
}

impl ObjectEncode for FemmSolution {
    fn object_encoding(&self, _cx: &mut Cx) -> sim_kernel::Result<ObjectEncoding> {
        self.validate().map_err(sim_kernel::Error::from)?;
        Ok(ObjectEncoding::Constructor {
            class: solution_class_symbol(),
            args: solution_constructor_args(self),
        })
    }
}

impl sim_citizen::Citizen for FemmSolution {
    fn citizen_symbol() -> Symbol {
        solution_class_symbol()
    }

    fn citizen_version() -> u32 {
        1
    }

    fn citizen_arity() -> usize {
        7
    }

    fn citizen_fields() -> &'static [&'static str] {
        &[
            "id",
            "model_id",
            "physics",
            "formulation",
            "params",
            "nodes",
            "elements",
        ]
    }
}

fn solution_class_symbol() -> Symbol {
    Symbol::qualified("femm", "Solution")
}

fn solution_constructor_args(solution: &FemmSolution) -> Vec<Expr> {
    vec![
        Expr::Symbol(Symbol::new("v1")),
        int_expr(solution.id.0),
        int_expr(solution.model_id.0),
        Expr::String(physics_name(&solution.physics).to_owned()),
        Expr::String(formulation_name(&solution.formulation).to_owned()),
        Expr::List(
            solution
                .params
                .entries
                .iter()
                .map(|(name, _)| Expr::String(name.to_string()))
                .collect(),
        ),
        int_expr(solution.mesh.xy.len()),
        int_expr(solution.mesh.tri.len()),
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

/// A reference-counted, shareable [`FemmSolution`].
pub type SharedSolution = Arc<FemmSolution>;
