# Contributing

The build is a sequence of small PRs that a human can review in under 30 minutes each. One
concern per PR, tests in the same PR, no mixing refactors with behavior changes. Each PR lands
on `dev` (the default branch) before the next starts, so later PRs are always against reviewed code. The sequence
is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Pull requests

- **Size.** Target under 400 lines of non-test diff. If a PR is growing past that, split it.
- **Title.** `area: imperative summary` in under 70 characters. Areas: `repo`, `db`, `catalog`,
  `ingest`, `workloads`, `sampling`, `replay`, `scoring`, `estimate`, `cli`, `collect`,
  `sidecar`, `policy`, `receipts`, `docs`.
- **Branch.** `pr-NN-area-summary`, e.g. `pr-02-db-core-models`.
- **Description.** Fill in every section of `.github/PULL_REQUEST_TEMPLATE.md` (Why, What
  changed, How to review, How I tested, Out of scope / follow-ups, Risk).
- **CI must be green before review is requested.**

## Commits

Conventional-commit style, one logical change per commit, so the PR can be read commit by
commit:

```
feat: ...   a new capability
fix: ...    a bug fix
test: ...   tests only
docs: ...   documentation only
chore: ...  tooling, dependencies, CI
```

## Decisions (ADRs)

Anything a reviewer might ask "why this way?" about gets a short architecture decision record
in `docs/adr/NNNN-title.md`, in the same PR, with three sections: **Context**, **Decision**,
**Consequences**. Numbering is sequential; ADRs are never edited after merge, only superseded
by a new one that says so.

## Tests

- Unit tests run with no network. Any provider call in tests goes through recorded fixtures.
- `uv run pytest` must pass locally before pushing.

## Tooling

- Python 3.12, managed with `uv` (`uv sync --all-groups`).
- `ruff` for lint and format (`uv run ruff check .`, `uv run ruff format .`).
- `pre-commit` runs ruff, gitleaks and basic hygiene hooks on every commit
  (`uv run pre-commit install` once per clone).

## Secrets

Never in the repo. `.gitignore` blocks `*credentials*.json`, `*.pem`, `*.key` and `.env*`;
gitleaks runs as a pre-commit hook and in CI. Provider API keys are read from the environment
by the provider SDKs only and are never stored by this code.
