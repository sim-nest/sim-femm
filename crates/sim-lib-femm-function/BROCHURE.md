# sim-lib-femm-function

In one line: It turns a whole model into a simple knob-in, answer-out function you can call.

## What it gives you

Instead of treating a model as a big thing you set up and run by hand each time, this wraps it as a plain function: you hand it the settings you care about, and it hands back the quantity, field, or full solution you asked for. Ask for a result and, if you want, it also reports how much to trust that result and how the answer would shift as you nudge the settings. That makes a model behave like any other callable value in the system, so you can plug it into searches, sweeps, and tuning loops as easily as calling a formula.

## Why you will be glad

- A model becomes a callable you can drop into optimizers and parameter sweeps directly.
- Each answer can come with a trust reading, so you know how much weight it deserves.
- Asking how the result moves with a setting is a natural part of the same call.

## Where it fits

This is the front door that presents a finite-element model as an ordinary function in the SIM constellation. It leans on solving to produce results and on sensitivity work to report how answers change, then packages both as a registered callable. Whenever another part of the system wants to treat a physical simulation like a tunable formula, this is what makes that possible.
