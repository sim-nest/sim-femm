#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! FEMM model definition, meshing, and model validation.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: the assembled `FemmModel`, the triangular
//! mesh and meshers that discretize it, and the checks that validate a model
//! before it is meshed.

mod implementation;
mod validation;

pub use implementation::*;
pub use validation::validate_model;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
