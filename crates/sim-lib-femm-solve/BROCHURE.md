# sim-lib-femm-solve

In one line: It takes a described model all the way to a solved answer and hands you proof it was done right.

## What it gives you

This is the engine that actually produces the answer. Give it a model and it carries the whole job through: it meshes the shape, gathers the pieces into one system, and works that system out to a settled solution. It knows more than one way to crack the system and falls back sensibly when the first path does not suit. Every completed solve comes with a certificate, a short honest record of how it was done, whether it converged, how close it got, and how much to trust it. So you finish not just with a result but with evidence you can check that the result is sound.

## Why you will be glad

- One call carries a model from description all the way to a finished answer.
- Each solve arrives with a certificate you can inspect to confirm it converged.
- The solver picks a suitable method and steps back to another when needed, on its own.

## Where it fits

This is the heart of the SIM finite-element stack. Geometry, materials, meshing, and physics all lead here, and reporting and sensitivity all read what it produces. It ties the earlier steps together and drives them to a result, then vouches for that result with a checkable record. When you ask a model for its answer, this is what does the work.
