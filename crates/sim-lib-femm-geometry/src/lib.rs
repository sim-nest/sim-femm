#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Two-dimensional FEMM geometry: nodes, segments, arcs, regions, and labels.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: the symbolic 2D geometry description and
//! its lowering to concrete coordinates for meshing.

mod implementation;

pub use implementation::*;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
