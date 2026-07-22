# sim-lib-femm-core

In one line: It is the shared vocabulary and foundation every other finite-element piece builds on.

## What it gives you

Before you can describe a magnet or a heated part, everyone involved has to agree on the basic words: what a physics kind is, how a parameter is named, what an error looks like, and how limits are expressed. This provides that common ground. It holds the shared names, the stable identifiers, the error reporting, and the small building blocks that all the other finite-element libraries reach for. Because these definitions live in one place, every part of the stack means the same thing by the same term, and a model described in one library reads correctly in the next.

## Why you will be glad

- The whole finite-element stack speaks one consistent language instead of many dialects.
- Errors and limits are reported in a uniform way you only learn once.
- Identifiers stay stable, so a thing named in one place is the same thing everywhere.

## Where it fits

This is the substrate under the SIM finite-element domain. It sits below meshing, physics, solving, and reporting, giving all of them their shared types and their common vocabulary. You rarely reach for it directly; instead you feel its steadiness every time two libraries use the same terms. It is the quiet base layer that lets the rest of the stack cooperate.
