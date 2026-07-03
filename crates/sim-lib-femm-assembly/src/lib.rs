#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Finite-element system assembly for the FEMM domain.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM assembly behavior, turning meshed models and physics
//! fronts into element residuals and the global stiffness/load system.

mod implementation;

pub use implementation::*;
