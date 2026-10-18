# ADR 0001: v2 is offline evidence plus compiled policies; the sidecar is observe-only

Date: 2026-10-17. Status: accepted.

## Context

The 2026 LLM-routing market converged on live per-request classifiers that pick a model from
the prompt. Those are commoditized, hard to trust (they decide on every request with no
evidence attached), and in agent traffic they can switch models mid-session and destroy the
prefix cache, raising the bill. Our analysis (`OmniRouter-competitive-analysis.md`, part 4)
and the design doc (`OmniRouter-v2-swap-safety-design.md`, sections 0, 2.1, 2.7 and open
question 4) identify the open position as the decision *layer*: per-workload evidence that a
swap is safe, compiled into the gateway the customer already runs.

There was a real temptation to fold online step routing into the agent sidecar from day one,
because the sidecar sits on the request path anyway.

## Decision

1. v2 produces **offline evidence** (shadow replays of the customer's own traffic, scored and
   estimated with confidence intervals) and **compiled policies** (static routing tables and
   gateway plugins derived from accepted recommendations).
2. There is **no live per-request classifier and no online model-choice path** in v2.
3. The agent sidecar is **observe-only through day 90**: it records steps, tool calls, cache
   read/write tokens, compaction events and outcomes, and forwards every request unchanged to
   the model the harness asked for. Its upstream request body must be byte-identical to what
   it received.
4. The decision API, when it exists, **evaluates compiled policy tables only**. It takes a
   workload fingerprint and a policy version and returns the model that policy names; it does
   not look at prompt content to choose.

## Consequences

- Every routing change is traceable to a recommendation with attached evidence, a sample, a
  replay run and scorer versions. "Why did this request go to model X?" always has an answer.
- Session-level holds (never switch mid-tool-loop) are expressible in a compiled policy; a
  per-request classifier could not promise them.
- We give up some upside: a live router could react to a single hard prompt. We accept that
  until the evidence loop is trusted; revisiting is an explicit decision after day 90.
- Tests for the sidecar assert pass-through fidelity, not routing behavior.
