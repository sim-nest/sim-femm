#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]
//! Time integration of FEMM models as ODE right-hand sides and DAE residuals.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: it casts a model coupled to external state
//! as an explicit ODE right-hand side for sim-numbers solvers and defines the
//! residual contract hosts use for implicit DAE solves.

mod implementation;

pub use implementation::*;

#[cfg(test)]
mod tests;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
