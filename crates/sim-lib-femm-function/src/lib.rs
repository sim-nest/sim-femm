#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]
//! FEMM models as first-class callable runtime functions.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: it wraps a model as a callable that maps
//! parameters to quantities, fields, or solutions, and registers those
//! callables and the model value with the runtime.
//!
//! The [`quality`] surface returns a quantity value together with the
//! [`sim_lib_femm_solve::SolveCertificate`] that proves solve fidelity. Passing
//! `Some(params)` for `wrt` adds a total gradient and annotates the returned
//! certificate with the corresponding `GradientTrust`; passing `None` returns
//! the value and certificate without gradient work.

mod exports;
mod implementation;
mod model_value;

pub use exports::*;
pub use implementation::*;
pub use model_value::*;

#[cfg(test)]
mod tests;
