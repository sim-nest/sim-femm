#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Solve tape: cached factorizations and solutions keyed by fingerprint.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: a bounded cache of linear factors and
//! solutions keyed by model, mesh, and parameter fingerprints so repeated
//! solves and derivative sweeps reuse prior work.

mod implementation;

pub use implementation::*;
