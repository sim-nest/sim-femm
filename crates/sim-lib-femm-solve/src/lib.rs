#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Linear solvers and the steady-state FEMM solve pipeline.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: the linear-solver interface and fallbacks
//! plus the steady solve that meshes, assembles, and solves a model.
//!
//! Every completed steady solve emits a [`SolveCertificate`]. The certificate
//! carries a kernel `Claim` with the method tag, convergence flag, final
//! residual, iteration count, solution fingerprint, and gradient trust. Linear
//! solves use the `femm-direct` method tag; nonlinear magnetostatic B-H solves
//! use `femm-ptc` and carry pseudo-transient continuation evidence.
//!
//! [`SolveExportRecord`] exposes the certificate fields as open metadata, and
//! [`certificate_claim`] re-derives the checkable claim for a completed solve.

mod certificate;
mod export;
mod implementation;
mod steady;

pub use certificate::{GradientTrust, SolveCertificate};
pub use export::{SolveExportRecord, certificate_claim};
pub use implementation::*;
pub use steady::*;
