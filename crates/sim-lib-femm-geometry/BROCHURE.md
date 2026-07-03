# sim-lib-femm-geometry

In one line: It is where you draw the two-dimensional shape you want to study.

## What it gives you

Every simulation starts with a shape, and this is where that shape is described. You lay down points, connect them with straight lines and curved arcs, close off regions, and label the areas so each one can be told what it is made of. It lets you build the cross-section of the object you care about in clear, named terms. When the drawing is complete, it works out the exact coordinates and hands off a clean, concrete outline ready to be divided into a mesh. In short, it is the drawing board that turns your idea of a shape into something the rest of the tools can act on.

## Why you will be glad

- You describe a shape in plain terms of points, lines, arcs, and labeled regions.
- Named regions let you attach materials later without hunting for the right area.
- The tidy outline it produces hands cleanly to meshing with no guesswork in between.

## Where it fits

This is the starting point of the SIM finite-element stack. Before anything can be meshed, solved, or reported, there has to be a shape, and this is where that shape is defined. It feeds the meshing step directly, giving it a precise geometric description to divide up. Think of it as the blueprint stage that everything downstream depends on.
