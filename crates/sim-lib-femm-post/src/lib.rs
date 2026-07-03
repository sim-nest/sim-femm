#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]
//! Post-processing of FEMM solutions into derived quantities.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: the solved-model record and the quantity
//! evaluations (energy, force, flux, inductance, sampled fields) read from it.

mod implementation;
#[cfg(test)]
mod tests;

pub use implementation::*;
