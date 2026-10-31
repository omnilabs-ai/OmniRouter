# ADR 0002: Be the decision layer, not the gateway

Date: 2026-10-17. Status: accepted.

## Context

The gateway war is over. LiteLLM, Vercel AI Gateway, OpenRouter, claude-code-router, vLLM
semantic-router and the Kubernetes Gateway API Inference Extension already unify providers,
hold keys, retry, and log. Customers have picked one and will not swap it to get a better
routing decision. v1 of OmniRouter tried to be a gateway with a classifier inside; that put it
in direct competition with every one of those projects and made the differentiating part (the
decision) the smallest part of the code.

Each of those gateways exposes a hook where an external decision can be applied: LiteLLM's
`CustomRoutingStrategyBase` and model-group config, claude-code-router's router config / JS
hook, vLLM semantic-router's `ext_proc` rules emitting `x-selected-model`, and client-side
model ordering for Vercel and OpenRouter.

## Decision

- We **do not proxy production traffic** as a requirement of the product. The core ships
  policies and a small decision API into the gateway the customer already runs.
- Provider adapters are **not rebuilt**. Replays use the LiteLLM SDK (not the proxy) or the
  *(superseded by ADR 0008: the official `openai` and `anthropic` SDKs, no LiteLLM dependency)*
  provider SDKs directly behind one thin interface; the only provider-specific code we own is
  the explicit parameter mapping (effort, thinking, response format) that must never be
  silently dropped.
- The **policy compiler is per target** and deliberately thin, so a gateway changing its hook
  API costs us one compiler, not the product.
- The decision API is the fallback for customers with no hook: a small in-VPC service with the
  policy embedded, sub-5 ms, evaluating compiled tables only (ADR 0001).
- The core runs **in the customer's VPC** on the customer's keys. A hosted control plane, if
  built, holds aggregates only unless the customer opts in.

## Consequences

- No gateway to operate, scale or secure on the request path; the product's attack surface
  is the ingest API, the replay workers and the signed receipt ledger.
- We depend on gateway hook APIs remaining available; the decision API is the hedge.
- Receipts are emitted by the gateway plugin or the sidecar, not by a proxy we own, so the
  receipt format must be simple enough to implement in a plugin.
- Marketing and naming follow: the product is an engine that proves swaps, not a router.
