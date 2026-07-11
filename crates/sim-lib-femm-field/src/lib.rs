#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Field representations and projections over FEMM solutions.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: derived field projections (potential, flux
//! density, field strength, fluxes) sampled from a solved model.

mod implementation;

pub use implementation::*;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
