#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Shared FEMM substrate: core types, stable ids, errors, and domain vocabulary.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM substrate the other FEMM crates build on, including
//! the physics/formulation vocabulary, parameter sets, limits, and the sparse
//! matrix and value-to-scalar decoding helpers.

mod implementation;
mod points;

pub use implementation::*;
pub use points::{decode_point2, eval_expr_f64};
