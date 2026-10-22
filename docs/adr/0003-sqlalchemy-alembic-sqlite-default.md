# ADR 0003: SQLAlchemy + Alembic, SQLite by default, Postgres via DATABASE_URL, JSON for nested structures

Date: 2026-10-21. Status: accepted.

## Context

v2 stores workloads, requests, outcomes, samples, replay runs, scores, recommendations,
policies and receipts (design doc section 4). The core runs in the customer's VPC (ADR 0002),
so the storage layer must install with nothing but Python for a first run, and must run on
Postgres when a team deploys it for real. Most entities carry nested structures (pricing,
capabilities, fingerprint, constraints, usage, latency, params) whose shape is still settling
during the 90-day build.

## Decision

- **SQLAlchemy 2.x declarative models** with typed `Mapped[...]` columns are the schema. No
  raw SQL in application code.
- **Alembic owns schema changes.** Every PR that touches `omni/models.py` ships a migration
  under `omni/migrations/versions/`, generated with `omni db revision --autogenerate -m "..."`
  and reviewed by hand. CI asserts that autogenerate finds no drift between the models and a
  freshly migrated database, so a model change without a migration fails the build.
- **SQLite is the default** (`.omni/omni.db`, created on demand) for development and tests.
  **Postgres** is selected purely by setting `DATABASE_URL` (`postgresql+psycopg://...`);
  there is one code path. Migrations run in Alembic batch mode on SQLite so `ALTER` works
  there too.
- **Nested structures are JSON columns**, not child tables, when they are read and written as
  a unit and never queried field-by-field in the hot path. Fields that are filtered or joined
  on (ids, provider, timestamps, status, fingerprint key, session id) are real columns with
  indexes.
- **Every table has `tenant_id` and `created_at`.** Multi-tenancy is a column from day one so
  a hosted control plane later needs no schema change.

## Consequences

- One mental model (ORM objects) from the CLI to the replay workers; tests run against a
  throwaway SQLite file with no services.
- JSON columns trade queryability for flexibility: reporting that needs per-field filtering on
  `usage` or `pricing` will promote those fields to columns (with a migration) when it arrives.
  On Postgres the columns are `JSON`, not `JSONB`; switch when an index on them is needed.
- SQLite and Postgres differ in details (timezone handling of `DateTime`, JSON operators);
  tests run on SQLite, so Postgres-specific behavior is covered only when a Postgres CI job is
  added.
- Migrations are part of the review surface: reviewers read the generated migration, not just
  the model diff.
