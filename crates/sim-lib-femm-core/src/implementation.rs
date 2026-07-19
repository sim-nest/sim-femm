#![forbid(unsafe_code)]
//! Core FEMM substrate: ids, vocabulary, parameters, numeric helpers, and library wiring.
//!
//! Defines the stable ids, physics/formulation/unit vocabulary, parameter
//! specs and sets, limits, scalar decoding helpers, and runtime registration
//! shared by every other FEMM crate.

use std::{
    any::Any,
    collections::BTreeSet,
    hash::{Hash, Hasher},
    sync::Arc,
    time::Duration,
};

use sim_kernel::{
    AbiVersion, Args, Callable, ClassRef, Cx, DefaultFactory, Dependency, Expr, Factory, Lib,
    LibManifest, LibTarget, Linker, Object, RawArgs, Result as KernelResult, Symbol, Value,
    Version,
};

use crate::{FemmError, FemmResult};

/// Stable 64-bit identity derived by hashing a value's content.
///
/// FEMM uses these as content fingerprints for caches and change detection
/// across parameter sets and matrices; they are deterministic for a given
/// input but not portable across hasher implementations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableId(pub u64);

impl StableId {
    /// Compute a [`StableId`] from any [`Hash`]able value.
    pub fn from_hashable<T: Hash>(value: &T) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hasher);
        Self(hasher.finish())
    }
}

/// The physics problem a FEMM model solves.
///
/// The supported finite-element formulations the downstream physics crates
/// dispatch on; see [`femm_capabilities`] for the advertised set.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PhysicsKind {
    /// Time-invariant magnetic field problem.
    Magnetostatic,
    /// Time-harmonic (frequency-domain) magnetics problem.
    MagneticsHarmonic,
    /// Time-invariant electric field problem.
    Electrostatic,
    /// Steady-state heat conduction problem.
    HeatSteady,
    /// Steady-state electric current flow problem.
    CurrentSteady,
}

/// Geometric formulation under which a 2D model is interpreted.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Formulation {
    /// Planar (extruded) geometry with unit depth.
    Planar,
    /// Axisymmetric geometry revolved about an axis.
    Axisymmetric,
}

/// Length unit a model's coordinates are expressed in.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LengthUnit {
    /// SI meter.
    Meter,
    /// Millimeter.
    Millimeter,
    /// Inch.
    Inch,
    /// A caller-named unit identified by [`Symbol`].
    Custom(Symbol),
}

/// The role a model parameter plays in a FEMM study.
///
/// Classifies entries of a [`ParamSet`] so downstream crates can route design,
/// excitation, ODE state, and other parameters appropriately.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParamRole {
    /// A design variable that may be swept or optimized.
    Design,
    /// A source or boundary excitation magnitude.
    Excitation,
    /// A state variable advanced by an ODE integrator.
    OdeState,
    /// A time coordinate.
    Time,
    /// A geometric dimension.
    Geometry,
    /// A material property value.
    Material,
}

/// Declaration of a single model parameter: its name, default, unit, and role.
///
/// Describes the shape of an input a FEMM model accepts, independent of any
/// concrete binding in a [`ParamSet`].
#[derive(Clone, Debug)]
pub struct ParamSpec {
    /// Parameter name.
    pub name: Symbol,
    /// Default kernel [`Value`] used when the parameter is unbound, if any.
    pub default: Option<Value>,
    /// Unit symbol the value is expressed in, if any.
    pub unit: Option<Symbol>,
    /// The [`ParamRole`] this parameter plays in a study.
    pub role: ParamRole,
}

/// An ordered set of name-to-[`Value`] parameter bindings for a model run.
///
/// The concrete inputs supplied to a FEMM evaluation; lookups are by [`Symbol`]
/// and the whole set can be fingerprinted into a [`StableId`]. See the
/// [crate README](https://github.com/sim-nest/sim-femm) for the FEM role of
/// parameter sets.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_core::ParamSet;
/// use sim_kernel::{DefaultFactory, Factory, Symbol};
///
/// let radius = Symbol::new("radius");
/// let value = DefaultFactory.string("0.5".to_owned()).unwrap();
/// let params = ParamSet::new(vec![(radius.clone(), value)]);
/// assert!(params.get(&radius).is_some());
/// assert!(params.symbols().contains(&radius));
/// ```
#[derive(Clone, Debug, Default)]
pub struct ParamSet {
    /// The name/value bindings, in insertion order.
    pub entries: Vec<(Symbol, Value)>,
}

impl ParamSet {
    /// Build a [`ParamSet`] from name/value bindings.
    pub fn new(entries: Vec<(Symbol, Value)>) -> Self {
        Self { entries }
    }

    /// Look up the [`Value`] bound to `name`, if present.
    pub fn get(&self, name: &Symbol) -> Option<&Value> {
        self.entries
            .iter()
            .find(|(symbol, _)| symbol == name)
            .map(|(_, value)| value)
    }

    /// Return the bound parameter names as a sorted set.
    pub fn symbols(&self) -> BTreeSet<Symbol> {
        self.entries
            .iter()
            .map(|(symbol, _)| symbol.clone())
            .collect()
    }

    /// Compute a content [`StableId`] over the displayed bindings.
    ///
    /// Renders each value through the kernel's display path under `cx`, so the
    /// fingerprint reflects value content rather than object identity.
    pub fn fingerprint(&self, cx: &mut Cx) -> StableId {
        let mut text = String::new();
        for (symbol, value) in &self.entries {
            let display = value
                .object()
                .display(cx)
                .unwrap_or_else(|_| "#<display-error>".to_owned());
            text.push_str(&symbol.to_string());
            text.push('=');
            text.push_str(&display);
            text.push(';');
        }
        StableId::from_hashable(&text)
    }
}

/// Resource ceilings enforced across a FEMM study to bound work.
///
/// Caps mesh size, solver effort, output volume, and wall time so a single
/// evaluation cannot exhaust host resources; [`Default`] supplies safe limits.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FemmLimits {
    /// Maximum mesh node count.
    pub max_nodes: usize,
    /// Maximum mesh element count.
    pub max_elements: usize,
    /// Maximum number of stored nonzeros in an assembled matrix.
    pub max_nnz: usize,
    /// Maximum iterations for a single linear/nonlinear solve.
    pub max_solve_iters: usize,
    /// Maximum number of output samples produced by post-processing.
    pub max_output_samples: usize,
    /// Maximum number of FEMM solves in one study.
    pub max_femm_solves: usize,
    /// Maximum wall-clock budget in milliseconds.
    pub max_wall_ms: u64,
}

impl Default for FemmLimits {
    fn default() -> Self {
        Self {
            max_nodes: 10_000,
            max_elements: 20_000,
            max_nnz: 200_000,
            max_solve_iters: 4_000,
            max_output_samples: 20_000,
            max_femm_solves: 1_000,
            max_wall_ms: Duration::from_secs(30).as_millis() as u64,
        }
    }
}

/// List the capability tokens this FEMM build advertises.
///
/// Combines the always-present [`PhysicsKind`] names with availability flags
/// for optional sim-numbers backends: field domain, fixed-step ODE, and adjoint
/// differentiator.
pub fn femm_capabilities(
    installed_field: bool,
    installed_ptc: bool,
    installed_adjoint: bool,
) -> Vec<String> {
    let mut values = vec![
        "Magnetostatic".to_owned(),
        "MagneticsHarmonic".to_owned(),
        "Electrostatic".to_owned(),
        "HeatSteady".to_owned(),
        "CurrentSteady".to_owned(),
    ];
    values.push(
        if installed_ptc {
            "femm-ptc:installed"
        } else {
            "femm-ptc:unavailable"
        }
        .to_owned(),
    );
    values.push(
        if installed_adjoint {
            "femm-adjoint:installed"
        } else {
            "femm-adjoint:unavailable"
        }
        .to_owned(),
    );
    values.push(
        if installed_field {
            "numbers/field:installed"
        } else {
            "numbers/field:unavailable"
        }
        .to_owned(),
    );
    values
}

/// Parse a finite scalar, accepting a plain decimal or a `num/den` rational.
///
/// The shared text-to-`f64` rule used by FEMM expression and value decoders.
/// Malformed text, zero rational denominators, and non-finite values are
/// rejected.
///
/// # Examples
///
/// ```
/// use sim_lib_femm_core::parse_finite_number;
///
/// assert_eq!(parse_finite_number("0.5"), Some(0.5));
/// assert_eq!(parse_finite_number("3/4"), Some(0.75));
/// assert_eq!(parse_finite_number("1/0"), None);
/// assert_eq!(parse_finite_number("inf"), None);
/// ```
pub fn parse_finite_number(text: &str) -> Option<f64> {
    let value = if let Some((num, den)) = text.split_once('/') {
        let num = num.parse::<f64>().ok()?;
        let den = den.parse::<f64>().ok()?;
        if den == 0.0 {
            return None;
        }
        num / den
    } else {
        text.parse::<f64>().ok()?
    };
    value.is_finite().then_some(value)
}

/// Parse a displayed scalar, accepting a plain decimal or a `num/den` rational.
///
/// This compatibility wrapper uses [`parse_finite_number`], so displayed
/// scalars are finite-only.
pub fn parse_displayed_number(text: &str) -> Option<f64> {
    parse_finite_number(text)
}

/// Decode a kernel [`Value`] to `f64` via its display and [`parse_displayed_number`].
///
/// Fails with [`FemmError::InvalidGeometry`] when the value does not display as
/// a scalar number.
pub fn value_as_f64(cx: &mut Cx, value: &Value) -> FemmResult<f64> {
    let display = value
        .object()
        .display(cx)
        .map_err(|err| FemmError::InvalidGeometry(err.to_string()))?;
    parse_displayed_number(&display)
        .ok_or_else(|| FemmError::InvalidGeometry(format!("expected scalar number, got {display}")))
}

/// Render a deterministic `name(field=value, ...)` summary string.
///
/// Used to build stable, human-readable descriptors that also feed content
/// fingerprints.
pub fn stable_summary(name: &str, fields: &[(&str, String)]) -> String {
    let mut out = format!("{name}(");
    for (index, (field, value)) in fields.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(field);
        out.push('=');
        out.push_str(value);
    }
    out.push(')');
    out
}

fn version_symbol() -> Symbol {
    Symbol::qualified("femm", "version")
}

fn capabilities_symbol() -> Symbol {
    Symbol::qualified("femm", "capabilities")
}

#[derive(Clone)]
struct FemmCoreFunction {
    symbol: Symbol,
}

impl Object for FemmCoreFunction {
    fn display(&self, _cx: &mut Cx) -> KernelResult<String> {
        Ok(format!("#<function {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for FemmCoreFunction {
    fn class(&self, cx: &mut Cx) -> KernelResult<ClassRef> {
        if let Some(class) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "Function"))
        {
            return Ok(class.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("core", "Function"),
        )
    }
    fn as_expr(&self, _cx: &mut Cx) -> KernelResult<Expr> {
        Ok(Expr::Symbol(self.symbol.clone()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for FemmCoreFunction {
    fn call(&self, cx: &mut Cx, _args: Args) -> KernelResult<Value> {
        if self.symbol == version_symbol() {
            return cx.factory().string("0.1.0".to_owned());
        }
        let installed_field = cx
            .registry()
            .number_domain_by_symbol(&Symbol::qualified("numbers", "field"))
            .is_some();
        let installed_ptc = sim_lib_numbers_numeric::global_numeric_registry()
            .read()
            .map(|registry| registry.ode_fixed(&Symbol::new("femm-ptc")).is_some())
            .unwrap_or(false);
        let installed_adjoint = sim_lib_numbers_numeric::global_numeric_registry()
            .read()
            .map(|registry| {
                registry
                    .differentiator(&Symbol::new("femm-adjoint"))
                    .is_some()
            })
            .unwrap_or(false);
        let values = femm_capabilities(installed_field, installed_ptc, installed_adjoint)
            .into_iter()
            .map(|item| cx.factory().string(item))
            .collect::<KernelResult<Vec<_>>>()?;
        cx.factory().list(values)
    }

    fn call_exprs(&self, cx: &mut Cx, _args: RawArgs) -> KernelResult<Value> {
        self.call(cx, Args::default())
    }
}

/// The loadable [`Lib`] that registers the FEMM core functions with a runtime.
///
/// Realizes the kernel [`Lib`] contract: its manifest declares the `femm.core`
/// library and exports the `femm.version` and `femm.capabilities` functions,
/// and [`Lib::load`] links them into the host registry.
pub struct FemmCoreLib;

impl FemmCoreLib {
    /// Construct the FEMM core library handle.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FemmCoreLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for FemmCoreLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("femm", "core"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![Dependency {
                id: Symbol::qualified("numbers", "numeric"),
                minimum_version: None,
            }],
            capabilities: Vec::new(),
            exports: vec![
                sim_kernel::Export::Function {
                    symbol: version_symbol(),
                    function_id: None,
                },
                sim_kernel::Export::Function {
                    symbol: capabilities_symbol(),
                    function_id: None,
                },
            ],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        for symbol in [version_symbol(), capabilities_symbol()] {
            linker.function_value(
                symbol.clone(),
                DefaultFactory.opaque(Arc::new(FemmCoreFunction { symbol }))?,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
