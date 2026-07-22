# sim-femm

In one line: a finite-element modeling stack that carries a drawn shape all the way to trustworthy answers.

## What it gives you

Where you draw the shape and say what each part is made of, meshing that checks the model makes sense first, the physical laws per kind of problem, assembly into one solvable system, linear and nonlinear and time-stepping solves that hand you proof they were done right, field values read anywhere you point, sensitivity to each design setting, and remembered work so repeated solves stay cheap.

## Why you will be glad

- The path from model description to checked engineering quantity stays in one stack.
- Repeated solves can reuse work instead of starting from scratch every time.
- Gradients and certificates travel with the values they explain.

## Where it fits

This is the finite-element modeling layer for the SIM runtime. It sits above the kernel and number libraries, turning geometry, materials, sources, and boundary conditions into solved fields, quantities, ODE right-hand sides, and sensitivity results.
