#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Shared FEMM substrate: core types, stable ids, errors, and domain vocabulary.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM substrate the other FEMM crates build on, including
//! the physics/formulation vocabulary, parameter sets, limits, and the sparse
//! matrix and value-to-scalar decoding helpers.

mod error;
mod implementation;
mod matrix;
mod points;

pub use error::{FemmError, FemmResult};
pub use implementation::*;
pub use matrix::CsrMatrix;
pub use points::{
    FEMM_EXPR_OPERATORS, decode_point2, eval_expr_f64, integer_exponent, normalize_femm_expr,
};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
