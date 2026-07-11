#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]
//! Time integration of FEMM models as ODE/DAE systems.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: it casts a model coupled to external state
//! as an ODE/DAE right-hand side and integrates it over time.

mod implementation;

pub use implementation::*;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
