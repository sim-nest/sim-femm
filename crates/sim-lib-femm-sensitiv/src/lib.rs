#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Sensitivity and adjoint analysis of FEMM quantities.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: gradients of model quantities with respect
//! to parameters via adjoint, direct, or finite-difference paths.
//!
//! [`total_gradient`] evaluates supported output quantities against registered
//! input parameters and returns finite rows with explicit trust labels. Exact
//! adjoint paths are reported as `GradientTrust::AdjointVerified`; quantities
//! without an exact adjoint use finite-difference fallback and
//! `GradientTrust::FiniteDifferenceOnly`. Implemented paths do not emit
//! `GradientTrust::AdjointUnverified`.

mod expr_eval;
mod implementation;
mod nonlinear_adjoint;
mod sensitivity_mesh;
mod sensitivity_quantity;
mod sensitivity_solve;
mod sensitivity_types;
mod total_gradient;

pub use implementation::*;
pub use total_gradient::*;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
