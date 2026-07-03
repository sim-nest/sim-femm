# sim-lib-femm-sensitiv

In one line: It tells you how much your result would shift if you nudged each design setting.

## What it gives you

Knowing the answer to a model is useful; knowing which settings most affect that answer is what lets you improve a design. This works out exactly that. For a chosen result, it reports how sensitive that result is to each of the settings you can control, so you can see which knobs matter and in which direction to turn them. It reaches those sensitivities the most trustworthy way each case allows, and it labels every reading with how it was found, so you always know whether a slope is an exact figure or a careful estimate. You get direction and honesty together.

## Why you will be glad

- You learn which settings drive your result, so tuning stops being guesswork.
- Every sensitivity comes labeled with how much to trust it, exact or estimated.
- Optimizers and design searches get the guidance they need to move the right way.

## Where it fits

This is the design-guidance layer of the SIM finite-element stack. Solving gives the result; this gives the slopes that say how to make it better. It works with the callable-model surface so that a simulation can report both its answer and how that answer responds to change. Whenever a study turns from evaluating a design into improving one, this is the part that points the way.
