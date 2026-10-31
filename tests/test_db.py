"""Schema tests: the migration builds the models' schema on a fresh SQLite file, autogenerate
sees no drift afterwards, and each model round-trips including its relationships."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, select
from sqlalchemy.engine import make_url
from typer.testing import CliRunner

from omni.cli import app
from omni.db import alembic_config, get_engine, session_scope
from omni.models import Base, ModelCatalog, Outcome, Request, Workload


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{(tmp_path / 'test.db').as_posix()}"


@pytest.fixture
def migrated_engine(db_url):
    command.upgrade(alembic_config(db_url), "head")
    engine = get_engine(db_url)
    yield engine
    engine.dispose()


def test_migration_creates_every_table(migrated_engine):
    tables = set(inspect(migrated_engine).get_table_names()) - {"alembic_version"}
    assert (
        tables
        == set(Base.metadata.tables)
        == {
            "model_catalog",
            "workloads",
            "requests",
            "outcomes",
            "samples",
            "candidate_pools",
            "replay_runs",
            "replay_results",
        }
    )


def test_no_drift_between_models_and_migration(migrated_engine):
    with migrated_engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        diff = compare_metadata(ctx, Base.metadata)
    assert diff == [], f"models and migration disagree: {diff}"


def test_downgrade_removes_every_table(db_url, migrated_engine):
    command.downgrade(alembic_config(db_url), "base")
    assert set(inspect(get_engine(db_url)).get_table_names()) - {"alembic_version"} == set()


def test_default_sqlite_lives_under_dot_omni(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    engine = get_engine()
    with engine.connect():
        pass
    assert (tmp_path / ".omni" / "omni.db").exists()


def test_cli_db_upgrade_on_fresh_checkout(tmp_path, monkeypatch):
    """`omni db upgrade` in an empty directory must create .omni/ itself.

    Regression test: the Alembic environment opens its own engine, so the directory
    creation in get_engine() did not apply and a fresh checkout failed with
    "unable to open database file".
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    result = CliRunner().invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".omni" / "omni.db").exists()
    tables = set(inspect(get_engine()).get_table_names())
    assert {"workloads", "requests", "outcomes", "model_catalog", "alembic_version"} <= tables


def test_database_url_env_wins(monkeypatch, db_url):
    monkeypatch.setenv("DATABASE_URL", db_url)
    assert get_engine().url == make_url(db_url)


def test_round_trip_every_model(migrated_engine):
    ts = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
    with session_scope(migrated_engine) as s:
        s.add(
            ModelCatalog(
                model_id="claude-haiku-5-5",
                provider="anthropic",
                display_name="Claude Haiku 5.5",
                context_window=1_000_000,
                max_output=128_000,
                capabilities={"tools": True, "reasoning_effort_levels": ["low", "high"]},
                pricing={"input": 0.10, "output": 0.50, "cached_input": None},
                regions=["us"],
                tenant_id="t1",
            )
        )
        wl = Workload(
            name="support-triage",
            fingerprint={"system_prompt_hash": "abc", "tool_schema_hash": None},
            fingerprint_key="abc-key",
            incumbent_model_id="claude-sonnet-5-5",
            quality_bar={"min_score": 0.9},
            constraints={"allowed_providers": ["anthropic"]},
            token_mix={"input": 1000, "output": 100, "cached": 400, "reasoning": 0},
            cache_read_ratio=0.4,
            tenant_id="t1",
        )
        req = Request(
            request_id="req_1",
            workload=wl,
            ts=ts,
            source="litellm",
            model_requested="claude-sonnet-5-5",
            model_resolved="claude-sonnet-5-5",
            provider="anthropic",
            params={"temperature": 0, "effort": "high"},
            usage={"input": 1000, "output": 100, "cached_input": 400, "reasoning": 0},
            cost=0.0025,
            latency={"ttft_ms": 300, "total_ms": 1200},
            fallback_chain=["claude-sonnet-5-5"],
            session_id="sess-1",
            step_index=2,
            tenant_id="t1",
        )
        req.outcomes.append(Outcome(signal="thumb", value=1, source="sdk", tenant_id="t1"))
        req.outcomes.append(Outcome(signal="human_label", value="billing", source="console"))
        s.add(req)

    with session_scope(migrated_engine) as s:
        model = s.get(ModelCatalog, "claude-haiku-5-5")
        assert model.pricing["cached_input"] is None
        assert model.capabilities["reasoning_effort_levels"] == ["low", "high"]
        assert model.status == "active" and model.created_at is not None

        wl = s.scalar(select(Workload).where(Workload.name == "support-triage"))
        assert wl.workload_id.startswith("wl_")
        assert wl.cache_read_ratio == pytest.approx(0.4)
        assert [r.request_id for r in wl.requests] == ["req_1"]

        req = s.get(Request, "req_1")
        assert req.workload is wl
        assert req.ts.replace(tzinfo=UTC) == ts
        assert req.usage["cached_input"] == 400 and req.fallback_chain == ["claude-sonnet-5-5"]
        assert req.compaction_event is False
        assert sorted((o.signal, o.value) for o in req.outcomes) == [
            ("human_label", "billing"),
            ("thumb", 1),
        ]
        assert all(o.request is req and o.outcome_id.startswith("oc_") for o in req.outcomes)


def test_workload_fingerprint_is_unique_per_tenant(migrated_engine):
    from sqlalchemy.exc import IntegrityError

    with session_scope(migrated_engine) as s:
        s.add(Workload(name="a", fingerprint={}, fingerprint_key="k", tenant_id="t1"))
        s.add(Workload(name="b", fingerprint={}, fingerprint_key="k", tenant_id="t2"))
    with pytest.raises(IntegrityError), session_scope(migrated_engine) as s:
        s.add(Workload(name="c", fingerprint={}, fingerprint_key="k", tenant_id="t1"))
