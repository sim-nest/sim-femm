#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]
//! Umbrella prelude that installs the full FEMM library stack.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: a single entry point that installs the
//! sim-numbers prelude and every FEMM library into a runtime context.

use sim_kernel::{Cx, Lib, Result, Symbol};
use sim_lib_femm_core::FemmCoreLib;
use sim_lib_femm_field::FemmFieldLib;
use sim_lib_femm_flow::register_femm_ptc;
use sim_lib_femm_function::FemmFunctionLib;
use sim_lib_femm_ode::FemmOdeLib;
use sim_lib_femm_sensitiv::register_femm_adjoint;
use sim_lib_numbers_prelude::NumbersPreludeLib;

/// Umbrella library that installs the full FEMM stack into a runtime context.
///
/// Installing it loads the sim-numbers prelude and every FEMM library (core,
/// field, function, ODE) plus the FEMM time-stepper and adjoint registrations,
/// skipping any already present. See the
/// [crate README](https://github.com/sim/sim-femm).
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::{Cx, DefaultFactory, EagerPolicy, Symbol};
/// use sim_lib_femm_prelude::FemmPreludeLib;
///
/// let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
/// FemmPreludeLib::new().install_all(&mut cx).unwrap();
/// assert!(cx.registry().lib(&Symbol::qualified("femm", "core")).is_some());
/// assert!(
///     cx.registry()
///         .function_by_symbol(&Symbol::qualified("femm", "as-ode-rhs"))
///         .is_some()
/// );
/// ```
pub struct FemmPreludeLib;

impl FemmPreludeLib {
    /// Creates the FEMM prelude library handle.
    pub fn new() -> Self {
        Self
    }

    /// Installs the sim-numbers prelude and every FEMM library into `cx`.
    ///
    /// Each library is loaded only if not already present, so repeated installs
    /// are idempotent.
    pub fn install_all(&self, cx: &mut Cx) -> Result<()> {
        NumbersPreludeLib::new().install_all(cx)?;
        install_if_missing(cx, Symbol::qualified("femm", "core"), &FemmCoreLib::new())?;
        install_if_missing(cx, Symbol::qualified("femm", "field"), &FemmFieldLib::new())?;
        install_if_missing(
            cx,
            Symbol::qualified("femm", "function"),
            &FemmFunctionLib::new(),
        )?;
        install_if_missing(cx, Symbol::qualified("femm", "ode"), &FemmOdeLib::new())?;
        register_femm_ptc()?;
        register_femm_adjoint()?;
        Ok(())
    }
}

impl Default for FemmPreludeLib {
    fn default() -> Self {
        Self::new()
    }
}

fn install_if_missing(cx: &mut Cx, symbol: Symbol, lib: &dyn Lib) -> Result<()> {
    if cx.registry().lib(&symbol).is_none() {
        cx.load_lib(lib)?;
    }
    Ok(())
}

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
