#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]
//! Codec surface for FEMM model, solution, and field descriptors.
//!
//! The kernel defines the `Value`/`Expr`/`Shape`/codec protocol contracts and
//! sim-numbers supplies the number domains, tensors, and linear algebra; this
//! crate supplies the FEM behavior: the citizen descriptors and Lisp/JSON
//! summary forms that round-trip FEMM domain objects across codec surfaces.

mod citizen;
mod support;
mod values;

pub use citizen::*;
pub use values::*;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
