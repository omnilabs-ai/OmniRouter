# OmniRouter v2 — the Swap-Safety Engine

> **Status: pre-alpha.** v2 is being built in this repository as a sequence of small,
> reviewed PRs; see [docs/ROADMAP.md](docs/ROADMAP.md) for where it is going and
> [docs/adr](docs/adr) for why. The working name of the v2 distribution is `swapsafe`; the
> Python package and CLI are `omni`. The v1 router under `serverRouter/` is legacy: its
> README is at [docs/v1/README.md](docs/v1/README.md) and it is removed or ported by later PRs.

Teams change their primary model every month and run dozens of models, but nobody can answer
"is model X safe to swap in for my workload Y, and what does it save?" with evidence. The
Swap-Safety Engine turns the customer's own production traffic into the benchmark: it segments
traffic into workloads, replays stratified samples against candidate models in shadow, scores
results with calibrated judges and task checkers, quantifies the quality/cost/latency delta
with confidence intervals, and compiles the result into a routing policy for whatever gateway
the customer already runs (LiteLLM, Vercel AI Gateway, OpenRouter, claude-code-router, vLLM
semantic-router, or our own small decision API). Every production response then carries a
signed route receipt that links back to the evidence. It re-runs automatically when a new model
ships or the incumbent drifts. We never need to be the gateway, and the customer's data never
has to leave their VPC.

**What this is not:** a provider-unification gateway, a prompt-category classifier, a free-tier
aggregator, a hosted inference marketplace, or a live per-request router. v2 is offline
evidence plus compiled policies; the agent sidecar is observe-only ([ADR 0001](docs/adr/0001-offline-evidence-and-compiled-policies.md)).

## Development

Python 3.12 managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run pre-commit install
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run omni --version
uv run omni db upgrade          # creates .omni/omni.db, or migrates $DATABASE_URL
```

Conventions for PRs, commits, tests, ADRs and secrets are in [CONTRIBUTING.md](CONTRIBUTING.md).
Pull requests target `dev`.

## License

Apache-2.0. See [LICENSE](LICENSE).
