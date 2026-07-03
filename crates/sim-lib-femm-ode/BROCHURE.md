# sim-lib-femm-ode

In one line: It follows your model through time as the world around it keeps changing.

## What it gives you

Not every question is a single frozen snapshot. Sometimes a model is tied to something that moves or shifts, and you want to watch how the whole thing evolves as the clock runs. This lets you do that. It treats the model together with the outside state it is coupled to as a system that changes moment to moment, and it steps that system forward through time, working out where everything stands at each instant. You supply the model and how it connects to the changing world, and you get a trace of how the physics unfolds rather than one still image.

## Why you will be glad

- You can watch a model develop over time instead of only seeing one frozen result.
- The model and the moving world it is tied to are advanced together, staying consistent.
- Time-dependent studies become a natural extension of the same models you already build.

## Where it fits

This is the time dimension of the SIM finite-element stack. Where the steady solve answers how things settle, this answers how they get there and how they respond as conditions change. It builds on the same models and solving machinery, wrapping them in a stepping loop over time. When a problem couples a field to something that evolves, this is the part that carries it forward.
