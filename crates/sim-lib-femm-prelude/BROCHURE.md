# sim-lib-femm-prelude

In one line: It switches on the whole finite-element toolkit in a single step.

## What it gives you

The finite-element domain is made of many separate libraries, each doing one job. Wiring them all into a working session by hand would be tedious and easy to get wrong. This spares you that. It is one entry point that installs the entire stack at once, along with the number tools underneath it, so that geometry, materials, meshing, physics, solving, and reporting are all present and ready together. You bring in this one thing, and everything you need to describe, solve, and inspect a model is set up for you. It is the light switch that brings the whole room on.

## Why you will be glad

- One step readies the complete toolkit instead of a dozen separate setup calls.
- Nothing gets forgotten, because the whole stack comes on together as a set.
- Getting started is simple: pull in this one piece and begin building models.

## Where it fits

This is the convenient front entrance to the SIM finite-element domain. Rather than assembling the libraries yourself and hoping the pieces line up, you reach for this and the constellation's finite-element behavior is installed and connected. It sits above all the individual libraries, gathering them into a single ready-to-use whole. For most people, this is the natural place to begin.
