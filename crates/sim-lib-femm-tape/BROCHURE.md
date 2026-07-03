# sim-lib-femm-tape

In one line: It remembers work already done so repeated solves do not start from scratch every time.

## What it gives you

Solving a model is real effort, and much of that effort is wasted when you solve the same or a nearly identical model again and again, as happens during a design sweep or when computing how results change. This keeps a memory of the heavy work. It stores the expensive intermediate results and finished solutions, tagged by a fingerprint of the model, its mesh, and its settings, so that when the same situation comes around again it can reuse what it already has instead of redoing it. It keeps that memory to a sensible size, holding on to what is worth keeping and letting go of the rest.

## Why you will be glad

- Repeated and nearly identical solves finish faster by reusing earlier work.
- Design sweeps and sensitivity runs stop paying full price for every single step.
- The stored memory is kept to a sensible size, so it helps without growing out of hand.

## Where it fits

This is the memory of the SIM finite-element stack. Solving and sensitivity work lean on it so that a burst of related solves shares effort rather than repeating it. It sits quietly beside the solver, matching each request against what has been seen before and handing back a ready result when it can. You feel it as speed, especially when you are exploring many close variations of one model.
