# OmniRouter v2 (Swap-Safety Engine): PR Roadmap and Conventions

Date: 2026-10-17. Companion to `OmniRouter-v2-swap-safety-design.md`. Scope locked: v2 is offline evidence plus compiled policies; the agent sidecar is observe-only.

The build is broken into small PRs that a human can review in under 30 minutes each. One concern per PR, tests in the same PR, no mixing refactors with behavior changes. Each PR lands on `main` before the next starts, so later PRs are always against reviewed code.

---

## PR conventions

**Size.** Target under 400 lines of non-test diff. If a PR is growing past that, split it; the roadmap below is already split with that limit in mind.

**Title.** `area: imperative summary` in under 70 characters. Areas: `repo`, `db`, `catalog`, `ingest`, `workloads`, `sampling`, `replay`, `scoring`, `estimate`, `cli`, `collect`, `sidecar`, `policy`, `receipts`, `docs`.

**Description template** (checked in as `.github/PULL_REQUEST_TEMPLATE.md`):

```
## Why
One paragraph: the problem this PR solves and the design-doc section it implements.

## What changed
Bullets, grouped by module. Name the new public interfaces.

## How to review
Suggested file order and the one or two places that deserve the most attention.

## How I tested
Commands run and what they showed. Fixtures used. Anything not covered.

## Out of scope / follow-ups
What was deliberately left out and which later PR picks it up.

## Risk
Data migrations, config changes, anything irreversible. "None" is a valid answer.
```

**Commits.** Conventional-commit style (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), one logical change per commit, so the PR can be read commit by commit.

**Decisions.** Anything a reviewer might ask "why this way?" about gets a short ADR in `docs/adr/NNNN-title.md` (context, decision, consequences) in the same PR.

**Tests.** Unit tests run with no network; any provider call in tests goes through recorded fixtures. CI must be green before review is requested.

**Secrets.** Never in the repo. `.gitignore` blocks `*credentials*.json`, `*.pem`, `.env*`; a pre-commit secret scan (gitleaks) runs on every commit.

---

## Phase A: foundation (design §3, §4, §5; days 1–10)

**PR 1 — `repo: bootstrap project, CI, conventions`**
Python 3.12 project managed with `uv`; `pyproject.toml`; `ruff` (lint + format); `pytest`; `pre-commit` with ruff and gitleaks; GitHub Actions running lint and tests on push and PR; Apache-2.0 `LICENSE`; `.gitignore` with secret patterns; `README.md` with the one-paragraph thesis and a "status: pre-alpha" banner; `CONTRIBUTING.md` with the conventions above; the PR template; `docs/adr/0001-offline-evidence-and-compiled-policies.md` recording the scope decision; `docs/adr/0002-decision-layer-not-gateway.md`. A placeholder `omni --version` CLI so CI has something to run.
*Review focus:* conventions and ADRs, since everything after follows them.

**PR 2 — `db: core domain models and migrations`**
SQLAlchemy 2.x models for `ModelCatalog`, `Workload`, `Request`, `Outcome` with the fields in design §4; Alembic initial migration; SQLite for development and tests, Postgres via `DATABASE_URL`; a `db` module with session factory; `omni db upgrade` command. Tests: migration runs on a fresh SQLite, round-trip insert/read for each model.
*ADR:* 0003 SQLAlchemy + Alembic, SQLite dev default.

**PR 3 — `catalog: model catalog, pricing table, cost calculator`**
YAML catalog format (`catalog/models.yaml`) with provider, dated revision, context window, capabilities, pricing for input / output / cached-input / reasoning tokens, regions, residency class; loader that upserts into `ModelCatalog`; `cost_of(usage, model)` calculator; `omni catalog sync` and `omni catalog show <model>`. Tests: golden-number cost cases including cached and reasoning tokens, unknown-model error.
*Review focus:* the pricing schema, because every savings number downstream depends on it.

**PR 4 — `ingest: JSONL importer and request normalization`**
A canonical `NormalizedRequest` schema (messages, tools hash, params, usage, cost, latency, model requested/resolved, provider, session/step fields); importer for a documented JSONL format; `omni ingest --from file.jsonl`; a regex-based redaction pass (emails, phone numbers, API-key shapes, customer-supplied patterns) applied before content is stored; content stored to a local object directory with the DB holding only a reference and hash. Tests: fixture file of 20 synthetic requests, redaction cases, idempotent re-import.
*ADR:* 0004 content stored by reference, metadata in DB.

## Phase B: workloads and samples (design §2.2, §2.3; days 10–15)

**PR 5 — `workloads: fingerprinting and segmentation`**
Fingerprint = hash of (system prompt with variable spans masked, tool-schema hash, response format, step kind, app tag); masking heuristics for templated prompts; `Workload` rows created on ingest; `omni workloads list --sort spend` with volume, spend, incumbent model, token mix, cache-read ratio; `omni workloads rename|merge`. Tests: same template with different variables lands in one workload; different tool sets split.
*Review focus:* the masking heuristics and their failure modes.

**PR 6 — `sampling: stratified samples and golden sets`**
`Sample` entity; stratification by prompt-length bucket, tool use, outcome-present, time bucket; size chosen from a target CI width with a floor and a cap; `omni sample create --workload X --size N [--golden]`; samples are versioned and immutable. Tests: strata proportions preserved, determinism under a seed.

## Phase C: replay (design §2.4; days 15–22)

**PR 7 — `replay: provider adapter interface and single-turn replay`**
`ProviderAdapter` protocol with one implementation over the LiteLLM SDK (not the proxy); explicit per-provider parameter mapping, including effort / thinking settings, that fails loudly rather than dropping a parameter; `ReplayRun` and `ReplayResult` entities; `CandidatePool` entity with manual pool definition; per-run USD budget cap enforced before each call using the PR 3 calculator; retries with backoff; `omni replay --workload X --candidates a,b,c --budget 20`. Tests: recorded-fixture adapter so the runner is tested end to end offline; budget stop; parameter-mapping error cases.
*ADR:* 0005 LiteLLM SDK as adapter layer, not a rebuilt provider set.

**PR 8 — `replay: teacher-forced single-step replay for agent traces`**
Reconstruct the prefix (messages plus tool results up to step k) from session/step fields; replay step k only; compare target is the incumbent's step-k output; tool-schema normalization for cross-provider replay (port of the v1 function registry). Tests: fixture trace of a 6-step coding session; prefix reconstruction; tool schema round-trip for OpenAI and Anthropic shapes.

## Phase D: scoring and recommendation (design §2.5, §2.6; days 22–30)

**PR 9 — `scoring: scorer interface and deterministic checkers`**
`Scorer` protocol and `Score` entity; checkers: JSON-schema validity, tool-call validity against the tool schema, tool-call agreement with the incumbent (same tool, compatible arguments), refusal detection; `omni score --run <id> --scorers ...`. Tests: table-driven cases per checker.

**PR 10 — `scoring: pairwise judge with position swap`**
Pairwise judge over two judge models from different families, position-swapped, rubric per workload with a generated default; `ScorerCalibration` entity and a first computation against any human or implicit labels present. Tests: recorded judge fixtures; position-swap aggregation; tie handling.
*Review focus:* how disagreement between the two orders is handled.

**PR 11 — `estimate: paired comparison, cost delta, recommendation`**
Paired bootstrap CI on quality delta per stratum; cost delta projected onto the workload's token mix and volume with the switch-cost term from cache-read ratio; latency delta; risk flags; decision rule (switch / partial / hold) with a customer-set bar; `Recommendation` entity with evidence links; `omni recommend --workload X` prints the per-candidate table. Tests: synthetic distributions with known deltas; hold when CI crosses the bar; switch cost flips a marginal case to hold.
*Review focus:* the statistics and the cost formula; this PR is the product's credibility.

Phase D exit = the day-30 criterion in the design doc: a true, publishable sentence about our own traffic.

## Phase E: collection and policies (design §2.1, §2.7, §2.8; days 31–60, first PRs)

**PR 12 — `collect: LiteLLM CustomLogger and outcome SDK`**
`CustomLogger` subclass posting normalized requests to `POST /v1/ingest`; minimal FastAPI ingest service; Python SDK `report_outcome(request_id, signal, value)`. Tests: logger payload shape; ingest endpoint with the PR 4 normalizer.

**PR 13 — `sidecar: observe-only recorder for Anthropic Messages and OpenAI chat`**
A pass-through proxy that records request, response, tool calls, cache read/write tokens, compaction markers, and forwards unchanged to the requested model. No routing decision, by ADR 0001. Tests: recorded round trips; verify the upstream request body is byte-identical.

**PR 14 — `policy: compile recommendations to LiteLLM`**
`Policy` entity; compiler emitting a LiteLLM config fragment plus a `CustomRoutingStrategyBase` plugin that maps workload fingerprint to deployment with canary percentages and session-hold rules; `omni policy compile --rec <id> --target litellm`. Tests: compiled artifact snapshot; canary percentage math.

**PR 15 — `receipts: signed route receipts`**
`Receipt` entity; Ed25519 per-tenant signing; `omni receipts query|verify`; receipts emitted by the PR 14 plugin and the sidecar. Tests: sign/verify, tamper detection.

Later PRs (days 40–90, to be broken down when reached): claude-code-router target, catalog watcher, drift detection on golden sets, decision API in Rust or Go, Vercel/OpenRouter client helpers, console v0, second workload type.

---

## Working agreement for the build sessions

- One implementation session works on one PR at a time, on a branch named `pr-NN-area-summary`, and opens the PR with the template filled in.
- The PR is reviewed here (by the user, with Claude's review on request) before the next PR starts; feedback goes back to the same session.
- The roadmap is checked into the repo as `docs/ROADMAP.md` in PR 1 and updated when scope changes, so the repo tells its own story.
