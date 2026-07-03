# sim-lib-femm-flow

In one line: It patiently drives a stubborn, nonlinear model to a settled answer and tells you how it went.

## What it gives you

Some models do not give up their answer in one clean step, because the material behaves differently as the field grows and the problem keeps changing under you. This library handles that. It eases the model toward its answer gradually, taking measured steps and letting the solution settle rather than forcing it, until the whole thing stops moving and a real solution is in hand. Along the way it keeps a running record of what happened: how each step went, whether things were calming down, and when convergence was reached. So you get both the settled result and an honest account of the journey to it.

## Why you will be glad

- Hard nonlinear models reach a stable answer instead of thrashing or giving up.
- The gradual approach avoids the wild swings that a blunt one-shot attempt can cause.
- You get a clear diagnostic trail showing how the solve settled, not just a final number.

## Where it fits

This is the convergence driver for tough finite-element problems in the SIM stack. Where a straightforward solve suffices, the plain steady path handles it; when the material response makes things nonlinear, this takes over and coaxes the system home. It works closely with the solving and assembly layers, wrapping them in the steadying loop that gets difficult models to a trustworthy result.
