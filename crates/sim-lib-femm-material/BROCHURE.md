# sim-lib-femm-material

In one line: It is where you say what each part is made of and what is pushing on it.

## What it gives you

A shape is only half a model; the other half is what fills it and what acts on it. This is where you spell that out. You assign materials to the regions of your drawing, set the conditions along its edges, place the sources that drive the physics, and choose how finely and how thoroughly the model should be handled. It gives you a clear, named way to attach all of that to a shape. With these descriptions in place, an outline stops being an empty picture and becomes a real physical situation the solver can reason about.

## Why you will be glad

- You give each region its material and each edge its condition in plain, named terms.
- Sources and drives are attached where they belong instead of buried in settings.
- The same shape can be studied under different materials just by swapping these descriptions.

## Where it fits

This is the properties layer of the SIM finite-element stack. Geometry supplies the shape; this dresses that shape with materials, boundaries, and sources so it means something physically. Meshing and solving then read those descriptions to know how each part should behave. It is the step that turns a drawing into a stated problem, ready to be worked out.
