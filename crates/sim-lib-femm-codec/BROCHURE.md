# sim-lib-femm-codec

In one line: It writes your model and its results out as readable text and reads them back without losing anything.

## What it gives you

A model, its solution, and the fields it produces all need a way to be saved, shared, and inspected. This turns those things into tidy text summaries you can read, store in a file, or send somewhere else, and it reads them straight back into working objects again. The round trip is faithful: what you write out is what you get back, so a saved model rebuilds exactly and a reported result means the same thing tomorrow. It speaks the shared text forms the rest of the system understands, so your finite-element work travels alongside everything else through the same channels.

## Why you will be glad

- You can save a model and a result to plain text and trust they reload unchanged.
- Sharing work with a teammate or another tool becomes copy, paste, and read back.
- Inspecting what a model actually contains is as easy as looking at its written form.

## Where it fits

This is the reading-and-writing surface for the SIM finite-element domain. It plugs into the constellation's shared way of turning objects into text and back, so models and solutions are first-class citizens that move through the same pipes as every other kind of value. Whenever a model needs to leave memory and come back intact, this handles the translation.
