#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]
//! Shared FEMM model-query callables and payloads.
//!
//! This crate owns the common substrate that turns a FEMM model plus an output
//! query into a callable value. Function exports and sensitivity analysis both
//! use these types, so model defaults, excitation resolution, and payload
//! metadata stay consistent without a dependency cycle.

mod implementation;

pub use implementation::*;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
