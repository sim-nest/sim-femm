# sim-lib-femm-mesh

In one line: It divides your shape into a fine web of triangles and checks the model makes sense first.

## What it gives you

A solver cannot work on a smooth shape directly; it needs the shape broken into many small triangles it can handle one at a time. This does that dividing. It takes the assembled model and covers it with a triangular mesh, fine where detail matters and coarser where it does not, so the physics has somewhere to live. Before it meshes, it looks the model over and flags problems: a region with nothing assigned, an edge that does not close, a setting that cannot hold. Catching those early saves you from a solve that would only fail later, or worse, quietly give a wrong answer.

## Why you will be glad

- Your smooth shape becomes a mesh the solver can actually work through.
- Model mistakes are caught up front, before you waste time on a doomed solve.
- The mesh puts detail where it counts, balancing accuracy against effort for you.

## Where it fits

This is the bridge between a described model and a solvable one in the SIM stack. Geometry and materials define the model; this validates it and cuts it into the triangular pieces that assembly and solving need. It holds the assembled model together and produces the mesh everything downstream reads. Nearly every solve begins by passing through here.
