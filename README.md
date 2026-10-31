# OmniRouter

**OmniRouter proves model swaps before you make them.** It turns your own production LLM
traffic into the benchmark: it groups requests into workloads, replays samples against
candidate models in shadow, scores the results with task checkers and calibrated judges, and
tells you, with confidence intervals, what switching would do to quality, cost and latency.
Accepted recommendations compile into routing policies for the gateway you already run.
Every routed response carries a signed receipt that points back at the evidence.

> **Status: pre-alpha, under active construction.** v2 is being built in this repository as a
> sequence of small, reviewed pull requests. What runs today is listed under
> [What works now](#what-works-now). The plan is [docs/ROADMAP.md](docs/ROADMAP.md); the
> reasons behind the big choices are in [docs/adr](docs/adr).

## Why

Teams change their primary model about once a month and run dozens of models across their
products. Every router on the market picks a model per request from a classifier and asks you
to trust it. None of them shows measured, per-workload evidence that a cheaper model would
hold up on *your* traffic, and none re-checks when a new model ships or the one you use
quietly changes. OmniRouter answers that question continuously:

> "Move 38% of `support-triage` from Sonnet to Model X on Provider P: −$4.1k/month at −0.4
> quality points, 95% CI, net of lost prompt cache. Hold the rest."

## How it works

```
your gateway / agent harness ──(1) collect──▶ requests + outcomes
                                                   │
                                        (2) segment into workloads
                                                   │
                                        (3) sample + redact
                                                   │
                                        (4) shadow-replay vs candidate pool
                                                   │
                                        (5) score: checkers + calibrated judges
                                                   │
                                        (6) estimate Δquality (CI), Δcost, Δlatency
                                                   │
                                        (7) recommendation ──▶ (8) policy for your gateway
                                                   │                      │
                                        (10) re-run on new model,   (9) canary + signed receipts
                                             drift, mix change  ◀──── production outcomes
```

Design principles, in one line each:

- **Your traffic is the benchmark.** Public benchmarks are saturated and off-distribution.
- **Decide, don't proxy.** Policies and a small decision API plug into LiteLLM, Vercel AI
  Gateway, OpenRouter, claude-code-router, vLLM semantic-router or Kubernetes GIE
  ([ADR 0002](docs/adr/0002-decision-layer-not-gateway.md)). OmniRouter does not need to sit on
  your request path.
- **Evidence or nothing.** Every recommendation carries its sample, replay runs, scorer
  versions and cost model. Every routed response carries a receipt.
- **Offline evidence plus compiled policies.** There is no live per-request classifier in v2;
  the agent sidecar only observes ([ADR 0001](docs/adr/0001-offline-evidence-and-compiled-policies.md)).
- **In your VPC.** Replays run on your keys inside your network. Content is redacted before it
  is stored and kept by reference, not in the database.

**What this is not:** a provider-unification gateway, a prompt-category router, a free-tier
aggregator, or a hosted inference marketplace.

## What works now

| Area | Status | Command |
|---|---|---|
| Project, CI, conventions, ADRs | done (PR 1) | `omni --version` |
| Core data model and migrations: `ModelCatalog`, `Workload`, `Request`, `Outcome` | done (PR 2) | `omni db upgrade` |
| Model catalog, pricing table, cost calculator | done (PR 3) | `omni catalog sync`, `omni catalog show` |
| Ingest: JSONL import, redaction, content by reference | done (PR 4) | `omni ingest --from FILE.jsonl` |
| Workload segmentation, sampling | next (PRs 5–6) | `omni workloads`, `omni sample` |
| Shadow replay, scoring, recommendation | PRs 7–11 | `omni replay`, `omni score`, `omni recommend` |
| Collectors, observe-only sidecar, policy compiler, receipts | PRs 12–15 | |

The full sequence, with what each PR contains and the review conventions, is in
[docs/ROADMAP.md](docs/ROADMAP.md).

## Quick start (development)

Python 3.12, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups          # install the project and dev tools
uv run pre-commit install     # ruff, gitleaks and hygiene hooks on every commit
uv run omni --version
uv run omni db upgrade        # creates .omni/omni.db and applies migrations
uv run pytest                 # offline tests, no API keys needed
uv run ruff check . && uv run ruff format --check .
```

### Configuration

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | SQLAlchemy URL for the metadata store; Postgres via `postgresql+psycopg://...` | `sqlite:///.omni/omni.db` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, … | Provider keys, read by the provider SDKs during replays | unset |

Keys are never stored by this code and never committed; see `.env.example`.

## Repository layout

```
src/omni/            v2: the `omni` package and CLI (models, db, migrations; more per PR)
tests/               v2 tests (offline)
docs/ROADMAP.md      the PR-by-PR build plan and conventions
docs/adr/            architecture decision records
serverRouter/        v1 router (FastAPI gateway with a benchmark-lookup smart router, 2025)
testLib/             v1 integration tests (need live provider keys)
docs/v1/README.md    how to run v1
```

v1 is the original OmniRouter: a unified chat/image/reasoning API over OpenAI, Anthropic,
Gemini and Together with a six-category "smart router" that ranked models by public benchmark
scores. It is kept as reference while v2 is built; its reusable pieces (the cross-provider
function registry, the decision-trace idea) are ported as the matching v2 features land, and
the rest is retired, one PR at a time. See [docs/v1/README.md](docs/v1/README.md) to run it.

## Contributing

PRs target `dev`. Each PR is one concern, under ~400 lines of non-test diff, with tests and
any ADR in the same PR and the pull-request template filled in. Details in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
