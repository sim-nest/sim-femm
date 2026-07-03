# sim-lib-femm-post

In one line: It turns a finished solve into the real-world numbers you actually wanted to know.

## What it gives you

Solving a model produces a mass of internal values that is not yet an answer to your question. This is the step that reads that finished solution and works out the quantities engineers actually ask for: the energy stored, the force on a part, the flow through a region, the inductance, and field readings at the spots you choose. You point it at a solved model and ask for what you need, and it computes those meaningful figures for you. It bridges the gap between a technically complete solve and the practical numbers that let you make a decision about your design.

## Why you will be glad

- You get the meaningful figures like force, energy, and flux without deriving them yourself.
- One solved model answers many different questions, each on demand.
- The reported numbers come straight from the solution, so they stay consistent with it.

## Where it fits

This is the results desk of the SIM finite-element stack. Solving does the heavy lifting and hands over a completed solution; this reads that solution and reports the quantities that matter to a person. It works alongside the field-reading library, together turning a raw answer into a set of practical, decision-ready numbers. It is usually the last stop before you look at what your model told you.
