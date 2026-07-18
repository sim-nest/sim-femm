# sim-lib-femm-query

In one line: It keeps every FEMM answer path speaking the same model-query language.

## What it gives you

FEMM models can be asked for many kinds of answers: a scalar result, a field, or a full solution. This library holds the shared request shape for those questions and the callable wrapper that resolves model inputs the same way every time. That means the function layer and the sensitivity layer both work from the same payload, the same defaults, and the same query meaning instead of carrying separate copies of the rules.

## Why you will be glad

- A model-derived function carries enough context for sensitivity work to inspect it safely.
- Missing inputs are filled from model defaults consistently across ordinary calls and gradient calls.
- Shared query handling keeps callable output and derivative output aligned.

## Where it fits

This is the small common layer underneath FEMM functions and FEMM sensitivities. It does not choose a derivative method or own a user-facing command; it gives those higher layers one dependable way to describe a model question, evaluate it, and carry the payload across runtime boundaries.
