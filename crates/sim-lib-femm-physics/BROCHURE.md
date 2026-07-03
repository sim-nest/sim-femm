# sim-lib-femm-physics

In one line: It holds the actual laws of nature the simulation obeys, one set per kind of problem.

## What it gives you

A simulation is only as trustworthy as the physics behind it, and this is where that physics lives. It carries the governing rules for each kind of problem the system handles: magnets and their steady and alternating fields, static electric fields, heat spreading through a part, and steady electric current. For every small piece of the mesh, it says what that piece must satisfy and what is driving it. By keeping each kind of physics stated clearly and separately, it lets the same machinery study very different situations simply by choosing which laws apply. This is the part that makes a result mean something real.

## Why you will be glad

- The same tools handle magnets, electric fields, heat, and current by swapping the physics.
- Each kind of problem has its governing rules stated cleanly, not tangled with the others.
- The answers rest on stated physical laws, so a result reflects real behavior.

## Where it fits

This is the law book of the SIM finite-element stack. Assembly asks it what each element contributes, and solving works out the consequences, but the meaning comes from here. It defines what magnetostatic, harmonic, electrostatic, heat, and current problems actually require. Whenever you pick which physics to study, this supplies the rules that make the rest of the pipeline honest.
