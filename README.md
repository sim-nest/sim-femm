# sim-femm

Draw a 2D field problem -- magnetics, electrostatics, heat, or current -- attach
its materials and boundary conditions, and get back a solved answer that carries
its own checkable convergence certificate.

sim-femm is a set of loadable **libraries** for the SIM runtime, not a
standalone binary. You add the crates you need, install them into a runtime
context, and build models from geometry through solve to derived quantities. SIM
itself ships as the `sim` CLI (`cargo install sim-run`); the full walkthrough
lives in sim-say.

## Examples

Install the whole FEMM stack into a fresh runtime context and confirm the
libraries and their functions registered:

```bash
cargo add sim-lib-femm-prelude sim-kernel
```

```rust
use std::sync::Arc;
use sim_kernel::{Cx, DefaultFactory, EagerPolicy, Symbol};
use sim_lib_femm_prelude::FemmPreludeLib;

let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
FemmPreludeLib::new().install_all(&mut cx).unwrap();

// The FEMM libraries are now loaded and their functions callable.
assert!(cx.registry().lib(&Symbol::qualified("femm", "core")).is_some());
assert!(
    cx.registry()
        .function_by_symbol(&Symbol::qualified("femm", "as-ode-rhs"))
        .is_some()
);
```

(from `sim-lib-femm-prelude` `src/lib.rs:29`, a passing doctest)

Or start from a canonical, ready-made model -- one of the built-in fixtures with
an analytically known reference:

```bash
cargo add sim-lib-femm-fixtures
```

```rust
use sim_lib_femm_fixtures::parallel_plate_capacitor;

let model = parallel_plate_capacitor();
assert_eq!(model.name.as_qualified_str(), "parallel-plate-capacitor");
assert_eq!(model.inputs.len(), 1);
```

(from `sim-lib-femm-fixtures` `src/lib.rs:79`, a passing doctest)

## How it works

sim-femm is the finite-element domain (FEMM) of the SIM constellation. It loads
a stack of libraries that describe, mesh, assemble, solve, and post-process 2D
finite-element field problems across the magnetostatic, harmonic, electrostatic,
heat, and current physics. The kernel (`sim-kernel`) defines the
`Value`/`Expr`/`Shape`/codec protocol contracts and sim-numbers supplies the
number domains, tensors, and linear algebra; these crates supply the FEM
behavior on top of that substrate.

The pipeline runs from a symbolic 2D geometry with attached materials, boundary
conditions, and sources, through triangular meshing and function spaces, into
per-element residuals and a global system, to linear and nonlinear solvers, and
out as derived fields and quantities. Models are first-class callable runtime
values, so they can be evaluated as functions of their parameters, integrated as
explicit ODE right-hand sides through sim-numbers, and differentiated for
sensitivity analysis.

Every completed steady solve carries a `SolveCertificate`: a kernel `Claim` with
the solver method, convergence flag, final residual, iteration count, solution
fingerprint, and gradient trust. Linear solves use the `femm-direct` method tag;
nonlinear magnetostatic B-H solves use `femm-ptc` and carry pseudo-transient
continuation evidence. Callers use `quality(cx, solve, quantity, wrt)` to get a
quantity value and its checkable certificate in one call; passing parameters in
`wrt` adds a total gradient and annotates the certificate with a
`GradientTrust` value. Supported quantities receive finite, trust-labeled
gradients, with exact adjoint paths distinguished from finite-difference
fallbacks.

## Crates

### Foundations

- `sim-lib-femm-core` -- shared FEMM substrate: core types, stable ids, errors,
  physics/formulation vocabulary, parameter sets, limits, and sparse-matrix and
  value-to-scalar decoding helpers.
- `sim-lib-femm-codec` -- citizen descriptors and Lisp/JSON summary forms that
  round-trip FEMM model, solution, and field objects across codec surfaces.
- `sim-lib-femm-prelude` -- umbrella entry point that installs the sim-numbers
  prelude and every FEMM library into a runtime context.
- `sim-lib-femm-fixtures` -- deterministic catalog of canonical 2D problems (one
  per physics surface) with analytically known references for regression tests.

### Model definition

- `sim-lib-femm-geometry` -- symbolic 2D geometry (nodes, segments, arcs,
  regions, labels) and its lowering to concrete coordinates for meshing.
- `sim-lib-femm-material` -- material properties, boundary conditions, sources,
  and mesh/output policies attached to a model.
- `sim-lib-femm-mesh` -- the assembled `FemmModel`, the triangular mesh and
  meshers that discretize it, and the checks that validate a model before mesh.
- `sim-lib-femm-space` -- function and DOF spaces over the mesh: per-element
  geometry, basis gradients, and barycentric mapping behind the element space.

### Assembly and solve

- `sim-lib-femm-physics` -- governing physics fronts: per-element residuals and
  source terms for the magnetostatic, harmonic, electrostatic, heat, and current
  formulations.
- `sim-lib-femm-assembly` -- turns meshed models and physics fronts into element
  residuals and the global stiffness/load system.
- `sim-lib-femm-solve` -- the linear-solver interface, steady solve pipeline,
  solve certificates, and export records for checkable residual and convergence
  evidence.
- `sim-lib-femm-flow` -- pseudo-transient continuation that drives a nonlinear
  system to convergence, with its event and diagnostic records.
- `sim-lib-femm-tape` -- a bounded cache of linear factors and solutions keyed by
  model, mesh, and parameter fingerprints so repeated solves reuse prior work.

### Results

- `sim-lib-femm-field` -- derived field projections (potential, flux density,
  field strength, fluxes) sampled from a solved model.
- `sim-lib-femm-post` -- the solved-model record and the quantity evaluations
  (energy, force, flux, inductance, sampled fields) read from it.

### Analysis and integration

- `sim-lib-femm-function` -- wraps a model as a callable mapping parameters to
  quantities, fields, or solutions, registers it with the runtime, and exposes
  `quality()` for value-plus-certificate queries.
- `sim-lib-femm-ode` -- casts a model coupled to external state as an explicit
  ODE right-hand side for sim-numbers solvers and defines DAE residual contracts
  for host implicit solvers.
- `sim-lib-femm-sensitiv` -- total gradients of supported model quantities with
  respect to registered parameters via exact adjoint, direct, or
  finite-difference paths with explicit trust labels.

### Rustdoc conventions

Public API documentation in `src/` follows one house style:

- Every public item opens with a one-line summary sentence, then context.
- The kernel defines the `Value`/`Expr`/`Shape`/codec contracts and sim-numbers
  supplies the number domains, tensors, and linear algebra; these crates supply
  the finite-element behavior (geometry, mesh, materials, function spaces,
  assembly, solvers, fields, physics, post-processing, ODE integration, and
  sensitivity). Each item is framed by its FEM role.
- The first-reach types carry a `# Examples` doctest that compiles and passes.
- Cross-reference with intra-doc links, and link back to this README rather than
  restating it.

The public API is documentation-gated: each crate's `lib.rs` denies
`missing_docs`, so every public item, field, and variant must be documented for
the crate to build.

Each crate's runnable examples are its embedded `recipes/` tree plus the rustdoc
`# Examples` doctests; there are no stub recipe directories.

## Validation

These commands run in the generated constellation workspace so local
constellation dependencies resolve from sibling checkouts.

```bash
cargo fmt --check && cargo test --workspace && cargo clippy --workspace -- -D warnings && cargo doc --workspace --no-deps
cargo run -p xtask -- simdoc --check
```

## Documentation Lanes

`cargo run -p xtask -- simdoc` builds the public documentation lanes:

- API docs: `target/doc/`
- Agent cards: `docs/agents/cards.jsonl` and `docs/agents/card-index.json`
- Human docs: `docs/humans/`
- Diagrams: `docs/diagrams/src/` and `docs/diagrams/generated/`

The same command writes split contract files under `docs/generated/`.
