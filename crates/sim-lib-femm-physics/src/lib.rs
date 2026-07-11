#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Governing physics fronts for the supported FEMM formulations.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: the per-element residuals and source terms
//! for the magnetostatic, harmonic, electrostatic, heat, and current physics.

mod implementation;

pub use implementation::*;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
