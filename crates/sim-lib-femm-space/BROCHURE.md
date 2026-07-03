# sim-lib-femm-space

In one line: It works out the fine geometric details inside each triangle so the solver knows how values vary across it.

## What it gives you

A mesh gives you triangles, but a solver needs more than their corners; it needs to know how a quantity changes as you move across each one and how the pieces knit together into a smooth whole. This supplies that inner machinery. For every element it works out the local shape, how a value is interpreted between the corners, and how steeply it changes from place to place. It also keeps track of which unknowns belong where across the whole model. This is the quiet groundwork that lets the physics be expressed cleanly on top of a plain grid of triangles.

## Why you will be glad

- The delicate per-triangle geometry is handled correctly so the physics can stay simple.
- Values are interpreted smoothly across each element, not just known at the corners.
- The bookkeeping of which unknown sits where is done for you across the whole model.

## Where it fits

This is the connective tissue between the mesh and the physics in the SIM finite-element stack. Meshing hands over bare triangles; physics and assembly need to know how quantities behave inside and between them, and this is what tells them. You seldom address it directly, but every accurate solve rests on the careful element groundwork it provides.
