#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Nonlinear solve flow: pseudo-transient continuation and solve diagnostics.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: the pseudo-transient continuation iteration
//! that drives a nonlinear system to convergence and the event/diagnostic
//! records describing that solve.

mod implementation;

pub use implementation::*;
