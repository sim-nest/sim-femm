#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Material, boundary, and source descriptions for FEMM models.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: the symbolic material properties, boundary
//! conditions, sources, and mesh/output policies attached to a model.

use sim_kernel::{Expr, Symbol};

/// A material model: symbolic multiphysics properties keyed by a region name.
///
/// Every property is optional so one record can describe a magnetic,
/// electrostatic, conductive, or thermal material (or a coupled mix). Values
/// stay symbolic [`Expr`]s, including the field-dependent reluctivity, so a
/// model can carry nonlinear and parameter-swept behavior.
/// See the [crate README](index.html).
///
/// # Examples
///
/// ```
/// use sim_kernel::{Expr, NumberLiteral, Symbol};
/// use sim_lib_femm_material::Material;
///
/// let num = |t: &str| Expr::Number(NumberLiteral {
///     domain: Symbol::qualified("numbers", "f64"),
///     canonical: t.to_owned(),
/// });
/// let steel = Material {
///     name: Symbol::new("steel"),
///     mu_r: Some(num("4000.0")),
///     nu_of_b2: None,
///     epsilon_r: None,
///     sigma: Some(num("1.0")),
///     thermal_k: None,
///     heat_source: None,
///     remanence: None,
/// };
/// assert_eq!(steel.name, Symbol::new("steel"));
/// assert!(steel.mu_r.is_some());
/// ```
#[derive(Clone, Debug)]
pub struct Material {
    /// Material name, referenced by block labels.
    pub name: Symbol,
    /// Relative magnetic permeability (linear case).
    pub mu_r: Option<Expr>,
    /// Reluctivity as a function of `B^2`, for nonlinear magnetics.
    pub nu_of_b2: Option<Expr>,
    /// Relative electric permittivity.
    pub epsilon_r: Option<Expr>,
    /// Electrical conductivity.
    pub sigma: Option<Expr>,
    /// Thermal conductivity.
    pub thermal_k: Option<Expr>,
    /// Volumetric heat source within the material.
    pub heat_source: Option<Expr>,
    /// Permanent-magnet remanence as a `[Bx, By]` vector.
    pub remanence: Option<[Expr; 2]>,
}

/// The kind of boundary condition imposed on a [`Boundary`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundaryKind {
    /// Prescribed value (essential) condition.
    Dirichlet,
    /// Prescribed flux (natural) condition.
    Neumann,
    /// Mixed value/flux (Robin) condition.
    Robin,
    /// Periodic condition linking matched edges.
    Periodic,
}

impl std::fmt::Display for BoundaryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Dirichlet => "dirichlet",
            Self::Neumann => "neumann",
            Self::Robin => "robin",
            Self::Periodic => "periodic",
        })
    }
}

/// A named boundary condition: a [`BoundaryKind`] with a symbolic value.
#[derive(Clone, Debug)]
pub struct Boundary {
    /// Boundary name, referenced by geometry edges.
    pub name: Symbol,
    /// Kind of condition imposed.
    pub kind: BoundaryKind,
    /// Symbolic condition value (its meaning depends on `kind`).
    pub value: Expr,
}

/// An applied excitation attached to a region of the model.
///
/// Covers the field-source forms across the supported physics: magnetic
/// current density and circuit coils, electrostatic charge density, and a
/// thermal heat source.
#[derive(Clone, Debug)]
pub enum Source {
    /// Imposed current density over a region.
    CurrentDensity {
        /// Target region name.
        region: Symbol,
        /// Symbolic current density.
        value: Expr,
    },
    /// A circuit coil carrying `current` through `turns` over a region.
    CircuitCoil {
        /// Coil name.
        name: Symbol,
        /// Target region name.
        region: Symbol,
        /// Number of turns.
        turns: Expr,
        /// Coil current.
        current: Expr,
    },
    /// Imposed charge density over a region.
    ChargeDensity {
        /// Target region name.
        region: Symbol,
        /// Symbolic charge density.
        value: Expr,
    },
    /// Imposed volumetric heat source over a region.
    HeatSource {
        /// Target region name.
        region: Symbol,
        /// Symbolic heat-source density.
        value: Expr,
    },
}

/// Meshing controls attached to a model.
#[derive(Clone, Debug)]
pub struct MeshPolicy {
    /// Mesher selector.
    pub kind: Symbol,
    /// Maximum element area, if bounded.
    pub max_area: Option<Expr>,
    /// Minimum element angle in degrees, if constrained.
    pub min_angle_deg: Option<Expr>,
}

/// A named post-processing output: a symbolic query evaluated after a solve.
#[derive(Clone, Debug)]
pub struct OutputSpec {
    /// Output name.
    pub name: Symbol,
    /// Symbolic query expression evaluated against the solution.
    pub query: Expr,
}

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests {
    use sim_kernel::Expr;

    use super::*;

    fn num(text: &str) -> Expr {
        sim_value::build::num_q(Some("numbers"), "f64", text)
    }

    #[test]
    fn material_holds_constant_properties() {
        let material = Material {
            name: Symbol::new("steel"),
            mu_r: Some(num("4000.0")),
            nu_of_b2: None,
            epsilon_r: None,
            sigma: Some(num("1.0")),
            thermal_k: None,
            heat_source: None,
            remanence: None,
        };
        assert_eq!(material.name, Symbol::new("steel"));
        assert!(material.mu_r.is_some());
    }
}
