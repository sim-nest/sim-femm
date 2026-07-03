# sim-lib-femm-field

In one line: It reads a solved model and tells you the field values anywhere you point.

## What it gives you

Once a model is solved, the answer lives spread across the whole shape as a raw solution. This turns that raw result into the field quantities people actually care about: the potential, the flux density, the field strength, and the flow through a region. You can sample those quantities at the spots that matter to you and read them as clear values rather than as internal numbers. It knows how the solution relates to each of these derived pictures, so you ask for the quantity you want and get a faithful reading, wherever in the model you choose to look.

## Why you will be glad

- You get the field quantities you care about directly, without deriving them by hand.
- Sampling a value at a point of interest is a simple, direct request.
- The different views of a solution stay consistent because they come from one source.

## Where it fits

This is the reading layer over a solved finite-element result in the SIM stack. Solving produces the underlying answer; this presents it as the meaningful fields an engineer inspects. It works hand in hand with the reporting and post-processing parts, giving them and you a clean way to pull sensible field readings out of a completed solve.
