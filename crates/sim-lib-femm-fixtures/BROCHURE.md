# sim-lib-femm-fixtures

In one line: It is a set of ready-made test problems whose right answers are already known.

## What it gives you

Trusting a solver means checking it against problems where the correct answer is not in doubt. This provides a fixed collection of small, carefully chosen finite-element problems, one for each kind of physics the system handles, each paired with a reference result worked out by hand. You can run any of them through the machinery and compare what comes back against the known truth. Because the problems never drift and their answers are pinned, they make a dependable yardstick: if a solve on one of them starts disagreeing with its reference, you know something changed and needs a look.

## Why you will be glad

- You can confirm the solver still gives correct answers on problems that cannot lie.
- Every part of the stack gets tested against the same steady, trustworthy baselines.
- When a result drifts, these catch it early instead of letting the error hide.

## Where it fits

This is the built-in yardstick for the SIM finite-element domain. The reading, solving, and time-stepping libraries all lean on it to check themselves against problems with known outcomes. It is a working catalog of reference cases meant for testing and regression checking, not a gallery of examples to learn from. Its job is to keep the rest of the stack honest.
