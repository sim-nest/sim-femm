#![forbid(unsafe_code)]
//! Solved-model record and derived-quantity evaluation.
//!
//! Defines the `FemmSolution`, the quantity specifications, and the routines
//! that read energy, force, flux, and sampled-field results from a solution.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    ClassId, ClassRef, Cx, DefaultFactory, Expr, Factory, Object, ObjectEncode, ObjectEncoding,
    Symbol, Value,
};
use sim_lib_femm_core::{
    FemmError, FemmLimits, FemmResult, Formulation, ParamSet, PhysicsKind, StableId, stable_summary,
};
use sim_lib_femm_flow::SolveDiagnostics;
use sim_lib_femm_mesh::FemMesh2;
use sim_lib_femm_space::ElementGeom;

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

/// Specification of a derived quantity to evaluate from a [`FemmSolution`].
///
/// Names what to compute and over which region or entity; passed to
/// [`quantity`] to produce a scalar result.
#[derive(Clone, Debug)]
pub enum QuantitySpec {
    /// Stored field energy, optionally restricted to a region.
    Energy {
        /// Region to integrate over, or `None` for the whole domain.
        region: Option<Symbol>,
    },
    /// Coenergy, optionally restricted to a region.
    Coenergy {
        /// Region to integrate over, or `None` for the whole domain.
        region: Option<Symbol>,
    },
    /// Net force in the y direction on a region.
    ForceY {
        /// Region the force acts on.
        region: Symbol,
    },
    /// Torque on a region about a center point.
    Torque {
        /// Region the torque acts on.
        region: Symbol,
        /// `[x, y]` center the torque is taken about.
        center: [f64; 2],
    },
    /// Flux linkage of a circuit.
    FluxLinkage {
        /// Circuit to evaluate.
        circuit: Symbol,
    },
    /// Inductance of a circuit.
    Inductance {
        /// Circuit to evaluate.
        circuit: Symbol,
    },
    /// Capacitance of a conductor.
    Capacitance {
        /// Conductor to evaluate.
        conductor: Symbol,
    },
    /// Joule (ohmic) loss, optionally restricted to a region.
    JouleLoss {
        /// Region to integrate over, or `None` for the whole domain.
        region: Option<Symbol>,
    },
    /// A named field sampled at given points.
    FieldAt {
        /// Field name (for example `bx`, `bmag`, `potential`).
        field: Symbol,
        /// Sample point(s) as a kernel [`Value`].
        points: Value,
    },
    /// A custom quantity named by `name` and defined by `expr`.
    Custom {
        /// Name of the custom quantity.
        name: Symbol,
        /// Defining expression over the kernel [`Expr`] contract.
        expr: Expr,
    },
}

/// The circuit excitation a derived quantity is evaluated against.
///
/// Inductance and flux linkage are defined relative to the driving coil
/// current, and capacitance relative to the applied conductor potential; the
/// stored field energy alone does not determine them. This carries the resolved
/// excitation scalar so [`quantity`] can compute `L = 2W/I^2`, `lambda = 2W/I`,
/// and `C = 2W/V^2` instead of silently assuming a unit drive.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Excitation {
    current: Option<f64>,
    potential: Option<f64>,
}

impl Excitation {
    /// An excitation with neither a current nor a potential specified.
    ///
    /// Quantities that need one (inductance, flux linkage, capacitance) error
    /// rather than return a physically meaningless value.
    pub fn none() -> Self {
        Self::default()
    }

    /// An excitation carrying a coil `current` in amperes.
    pub fn with_current(current: f64) -> Self {
        Self {
            current: Some(current),
            potential: None,
        }
    }

    /// An excitation carrying an applied `potential` in volts.
    pub fn with_potential(potential: f64) -> Self {
        Self {
            current: None,
            potential: Some(potential),
        }
    }

    /// The driving coil current, if one was resolved.
    pub fn current(&self) -> Option<f64> {
        self.current
    }

    /// The applied conductor potential, if one was resolved.
    pub fn potential(&self) -> Option<f64> {
        self.potential
    }
}

/// Samples the scalar potential of a solution at `(x, y)`.
///
/// Locates the containing triangle and interpolates the nodal values with the
/// barycentric basis. Returns [`FemmError::FieldOutOfDomain`] outside the mesh.
pub fn sample_potential(solution: &FemmSolution, x: f64, y: f64) -> FemmResult<f64> {
    solution.validate()?;
    let (tri, bary) = locate_triangle(solution, [x, y])?;
    let values = triangle_values(solution, tri)?;
    Ok((0..3).map(|index| bary[index] * values[index]).sum())
}

/// Returns the constant solution gradient `[d/dx, d/dy]` on triangle `tri`.
///
/// For a P1 element the field gradient is constant; this is the building block
/// for flux-density and field-strength quantities.
pub fn sample_gradient(solution: &FemmSolution, tri: [u32; 3]) -> FemmResult<[f64; 2]> {
    solution.validate()?;
    let geom = ElementGeom::from_mesh(&solution.mesh, tri)?;
    let values = triangle_values(solution, tri)?;
    Ok([
        geom.grad
            .iter()
            .zip(values)
            .map(|(grad, value)| grad[0] * value)
            .sum(),
        geom.grad
            .iter()
            .zip(values)
            .map(|(grad, value)| grad[1] * value)
            .sum(),
    ])
}

/// Returns the total stored field energy over the whole solution domain.
///
/// Integrates the half-square energy density across every element, applying the
/// axisymmetric radial measure when the formulation requires it.
///
/// # Examples
///
/// ```
/// use sim_kernel::Symbol;
/// use sim_lib_femm_core::{Formulation, ParamSet, PhysicsKind, StableId};
/// use sim_lib_femm_flow::SolveDiagnostics;
/// use sim_lib_femm_mesh::FemMesh2;
/// use sim_lib_femm_post::{energy, FemmSolution};
///
/// let solution = FemmSolution {
///     id: StableId(10),
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
///     u: vec![1.0, 3.0, 4.0],
///     diagnostics: SolveDiagnostics {
///         method: Symbol::new("femm-ptc"),
///         converged: true,
///         iterations: 1,
///         final_residual: 0.0,
///         events: Vec::new(),
///         diagnostics: Vec::new(),
///     },
/// };
/// assert!(energy(&solution).unwrap() >= 0.0);
/// ```
pub fn energy(solution: &FemmSolution) -> FemmResult<f64> {
    solution.validate()?;
    let mut total = 0.0;
    for tri in &solution.mesh.tri {
        let geom = ElementGeom::from_mesh(&solution.mesh, *tri)?;
        let measure = element_measure(solution, &geom)?;
        let mean = triangle_mean(solution, *tri)?;
        total += 0.5 * measure * mean * mean;
    }
    Ok(total)
}

/// Samples the potential on the tensor grid `xs` by `ys`, row-major.
///
/// Rejects requests whose sample count exceeds
/// [`FemmLimits::max_output_samples`] with [`FemmError::BudgetExceeded`].
pub fn sample_grid(
    solution: &FemmSolution,
    xs: &[f64],
    ys: &[f64],
    limits: &FemmLimits,
) -> FemmResult<Vec<f64>> {
    solution.validate()?;
    let sample_count = xs
        .len()
        .checked_mul(ys.len())
        .ok_or_else(|| FemmError::BudgetExceeded("sample grid size overflows usize".to_owned()))?;
    if sample_count > limits.max_output_samples {
        return Err(FemmError::BudgetExceeded(format!(
            "samples {sample_count} > {}",
            limits.max_output_samples
        )));
    }
    let mut out = Vec::with_capacity(sample_count);
    for x in xs {
        for y in ys {
            out.push(sample_potential(solution, *x, *y)?);
        }
    }
    Ok(out)
}

/// Evaluates a derived scalar quantity from a solution and its excitation.
///
/// Dispatches on the [`QuantitySpec`] to compute energy, force, torque, flux,
/// inductance, capacitance, loss, or a sampled field. Inductance and flux
/// linkage are taken relative to the coil current in `excitation`, and
/// capacitance relative to its applied potential, via the linear
/// magnetostatic/electrostatic identity `W = 0.5*L*I^2` (`C = 2W/V^2`); a
/// missing or zero excitation is rejected with a [`FemmError`] rather than
/// returning a physically wrong scalar. Unknown regions, out-of-domain samples,
/// and unimplemented custom quantities also surface as [`FemmError`].
pub fn quantity(
    solution: &FemmSolution,
    spec: &QuantitySpec,
    excitation: &Excitation,
) -> FemmResult<f64> {
    solution.validate()?;
    match spec {
        QuantitySpec::Energy { region } | QuantitySpec::Coenergy { region } => {
            region_energy(solution, region.as_ref())
        }
        QuantitySpec::ForceY { region } | QuantitySpec::Torque { region, .. } => {
            let force = region_force_y(solution, region)?;
            if let QuantitySpec::Torque { center, .. } = spec {
                Ok(force * (region_centroid_x(solution, region)? - center[0]))
            } else {
                Ok(force)
            }
        }
        QuantitySpec::FluxLinkage { .. } => {
            let current = require_current(excitation, "flux linkage")?;
            // lambda = 2W / I for a linear magnetostatic solve.
            Ok(2.0 * energy(solution)? / current)
        }
        QuantitySpec::Inductance { .. } => {
            let current = require_current(excitation, "inductance")?;
            // W = 0.5*L*I^2  =>  L = 2W / I^2.
            Ok(2.0 * energy(solution)? / (current * current))
        }
        QuantitySpec::Capacitance { .. } => {
            let potential = require_potential(excitation, "capacitance")?;
            // W = 0.5*C*V^2  =>  C = 2W / V^2.
            Ok(2.0 * energy(solution)? / (potential * potential))
        }
        QuantitySpec::JouleLoss { region } => joule_loss(solution, region.as_ref()),
        QuantitySpec::FieldAt { field, points } => {
            let point = decode_point(points)?;
            sample_named_field(solution, field, point)
        }
        QuantitySpec::Custom { name, .. } => Err(FemmError::FieldOutOfDomain(format!(
            "custom quantity {name} is not implemented"
        ))),
    }
}

fn require_current(excitation: &Excitation, quantity: &str) -> FemmResult<f64> {
    let current = excitation
        .current()
        .ok_or_else(|| FemmError::FieldOutOfDomain(format!("{quantity} needs a coil current")))?;
    if current == 0.0 {
        return Err(FemmError::InvalidGeometry(format!(
            "{quantity} undefined at zero current"
        )));
    }
    Ok(current)
}

fn require_potential(excitation: &Excitation, quantity: &str) -> FemmResult<f64> {
    let potential = excitation.potential().ok_or_else(|| {
        FemmError::FieldOutOfDomain(format!("{quantity} needs an applied potential"))
    })?;
    if potential == 0.0 {
        return Err(FemmError::InvalidGeometry(format!(
            "{quantity} undefined at zero potential"
        )));
    }
    Ok(potential)
}

fn element_measure(solution: &FemmSolution, geom: &ElementGeom) -> FemmResult<f64> {
    Ok(geom.area * geom.axisymmetric_weight(&solution.formulation)?)
}

fn region_energy(solution: &FemmSolution, region: Option<&Symbol>) -> FemmResult<f64> {
    let mut total = 0.0;
    for (index, tri) in solution.mesh.tri.iter().enumerate() {
        if !region_matches(solution, index, region) {
            continue;
        }
        let geom = ElementGeom::from_mesh(&solution.mesh, *tri)?;
        let measure = element_measure(solution, &geom)?;
        let mean = triangle_mean(solution, *tri)?;
        total += 0.5 * measure * mean * mean;
    }
    if let Some(region) = region
        && total == 0.0
    {
        ensure_region(solution, region)?;
    }
    Ok(total)
}

fn region_force_y(solution: &FemmSolution, region: &Symbol) -> FemmResult<f64> {
    ensure_region(solution, region)?;
    let mut total = 0.0;
    for (index, tri) in solution.mesh.tri.iter().enumerate() {
        if !region_matches(solution, index, Some(region)) {
            continue;
        }
        let geom = ElementGeom::from_mesh(&solution.mesh, *tri)?;
        let grad = sample_gradient(solution, *tri)?;
        total -= element_measure(solution, &geom)? * grad[1] * grad[1] * 0.5;
    }
    Ok(total)
}

fn region_centroid_x(solution: &FemmSolution, region: &Symbol) -> FemmResult<f64> {
    ensure_region(solution, region)?;
    let mut weighted = 0.0;
    let mut measure_total = 0.0;
    for (index, tri) in solution.mesh.tri.iter().enumerate() {
        if !region_matches(solution, index, Some(region)) {
            continue;
        }
        let geom = ElementGeom::from_mesh(&solution.mesh, *tri)?;
        let measure = element_measure(solution, &geom)?;
        let centroid_x = geom.xy.iter().map(|point| point[0]).sum::<f64>() / 3.0;
        weighted += centroid_x * measure;
        measure_total += measure;
    }
    Ok(weighted / measure_total)
}

fn joule_loss(solution: &FemmSolution, region: Option<&Symbol>) -> FemmResult<f64> {
    let mut total = 0.0;
    for (index, tri) in solution.mesh.tri.iter().enumerate() {
        if !region_matches(solution, index, region) {
            continue;
        }
        let geom = ElementGeom::from_mesh(&solution.mesh, *tri)?;
        let grad = sample_gradient(solution, *tri)?;
        total += element_measure(solution, &geom)? * (grad[0] * grad[0] + grad[1] * grad[1]);
    }
    if let Some(region) = region
        && total == 0.0
    {
        ensure_region(solution, region)?;
    }
    Ok(total)
}

fn region_matches(solution: &FemmSolution, index: usize, region: Option<&Symbol>) -> bool {
    region.is_none_or(|region| {
        solution
            .mesh
            .elem_region
            .get(index)
            .is_some_and(|name| name == region)
    })
}

fn ensure_region(solution: &FemmSolution, region: &Symbol) -> FemmResult<()> {
    if solution.mesh.elem_region.iter().any(|name| name == region) {
        Ok(())
    } else {
        Err(FemmError::FieldOutOfDomain(format!(
            "missing region {region}"
        )))
    }
}

/// Decode a two-element `[x y]` point literal into `[f64; 2]`.
///
/// Re-exposes the shared [`sim_lib_femm_core::decode_point2`] under this
/// crate's `decode_point` name.
pub fn decode_point(points: &Value) -> FemmResult<[f64; 2]> {
    sim_lib_femm_core::decode_point2(points)
}

/// Samples a named field of a solution at `point`.
///
/// Resolves field aliases (`potential`/`a`/`v`, `bx`/`ex`, `bmag`/`emag`, ...)
/// to the appropriate potential or gradient quantity. Returns
/// [`FemmError::FieldOutOfDomain`] for an unknown field or out-of-mesh point.
pub fn sample_named_field(
    solution: &FemmSolution,
    field: &Symbol,
    point: [f64; 2],
) -> FemmResult<f64> {
    solution.validate()?;
    match field.name.as_ref() {
        "potential" | "a" | "v" => sample_potential(solution, point[0], point[1]),
        "bx" | "ex" => Ok(sample_gradient(solution, locate_triangle(solution, point)?.0)?[0]),
        "by" | "ey" => Ok(sample_gradient(solution, locate_triangle(solution, point)?.0)?[1]),
        "bmag" | "emag" | "heat-flux-mag" => {
            let grad = sample_gradient(solution, locate_triangle(solution, point)?.0)?;
            Ok((grad[0] * grad[0] + grad[1] * grad[1]).sqrt())
        }
        _ => Err(FemmError::FieldOutOfDomain(format!(
            "unknown field {field}"
        ))),
    }
}

/// Finds the mesh triangle containing `point` and its barycentric coordinates.
///
/// Returns the connectivity and barycentric weights of the first triangle whose
/// coordinates are all non-negative, or [`FemmError::FieldOutOfDomain`] if the
/// point lies outside the mesh.
pub fn locate_triangle(
    solution: &FemmSolution,
    point: [f64; 2],
) -> FemmResult<([u32; 3], [f64; 3])> {
    solution.validate()?;
    for tri in &solution.mesh.tri {
        let geom = ElementGeom::from_mesh(&solution.mesh, *tri)?;
        let bary = geom.barycentric(point);
        if bary.iter().all(|value| *value >= -1.0e-9) {
            return Ok((*tri, bary));
        }
    }
    Err(FemmError::FieldOutOfDomain(format!(
        "point ({}, {})",
        point[0], point[1]
    )))
}

fn triangle_values(solution: &FemmSolution, tri: [u32; 3]) -> FemmResult<[f64; 3]> {
    Ok([
        *solution.u.get(tri[0] as usize).ok_or_else(|| {
            FemmError::InvalidGeometry(format!("triangle node {} has no solution value", tri[0]))
        })?,
        *solution.u.get(tri[1] as usize).ok_or_else(|| {
            FemmError::InvalidGeometry(format!("triangle node {} has no solution value", tri[1]))
        })?,
        *solution.u.get(tri[2] as usize).ok_or_else(|| {
            FemmError::InvalidGeometry(format!("triangle node {} has no solution value", tri[2]))
        })?,
    ])
}

fn triangle_mean(solution: &FemmSolution, tri: [u32; 3]) -> FemmResult<f64> {
    let values = triangle_values(solution, tri)?;
    Ok((values[0] + values[1] + values[2]) / 3.0)
}

/// A reference-counted, shareable [`FemmSolution`].
pub type SharedSolution = Arc<FemmSolution>;
