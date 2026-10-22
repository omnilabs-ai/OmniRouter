"""Catalog tests: schema validation, golden cost numbers, upsert semantics, CLI."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import select
from typer.testing import CliRunner

from omni.catalog import DEFAULT_CATALOG_PATH, CatalogError, Pricing, cost_of, load, sync
from omni.cli import app
from omni.db import alembic_config, get_engine, session_scope
from omni.models import ModelCatalog

ROOT = Path(__file__).resolve().parents[1]

GOOD = """
version: 1
models:
  - model_id: m-a
    provider: acme
    display_name: Model A
    context_window: 100000
    max_output: 8000
    capabilities: {tools: true, reasoning_effort_levels: [low, high]}
    pricing: {input: 1.0, output: 4.0, cached_input: 0.1, reasoning: 2.0, per_request: 0.001}
    pricing_verified_at: 2026-10-22
    source_url: https://acme.example/pricing
  - model_id: m-b
    provider: acme
    display_name: Model B
    pricing: {input: 0.5, output: 1.5}
    pricing_verified_at: 2026-10-22
    source_url: https://acme.example/pricing
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def engine(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'cat.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(alembic_config(url), "head")
    eng = get_engine(url)
    yield eng
    eng.dispose()


# ----------------------------------------------------------------------------- schema


def test_good_file_loads(tmp_path):
    entries = load(_write(tmp_path, GOOD))
    assert [e.model_id for e in entries] == ["m-a", "m-b"]
    assert entries[0].pricing.reasoning == Decimal("2.0")
    assert entries[1].pricing.cached_input is None and entries[1].status == "active"
    assert entries[0].capabilities.reasoning_effort_levels == ["low", "high"]


def test_unknown_key_is_an_error(tmp_path):
    bad = GOOD.replace(
        "    source_url: https://acme.example/pricing\n  - model_id: m-b",
        "    colour: red\n    source_url: https://acme.example/pricing\n  - model_id: m-b",
    )
    with pytest.raises(CatalogError, match="colour"):
        load(_write(tmp_path, bad))


def test_missing_rate_and_missing_file_are_errors(tmp_path):
    bad = GOOD.replace("pricing: {input: 0.5, output: 1.5}", "pricing: {input: 0.5}")
    with pytest.raises(CatalogError, match="output"):
        load(_write(tmp_path, bad))
    with pytest.raises(CatalogError, match="not found"):
        load(tmp_path / "nope.yaml")


def test_duplicate_model_id_is_an_error(tmp_path):
    with pytest.raises(CatalogError, match="duplicate"):
        load(_write(tmp_path, GOOD.replace("model_id: m-b", "model_id: m-a")))


def test_bundled_catalog_is_valid_and_dated():
    entries = load(ROOT / DEFAULT_CATALOG_PATH)
    assert 6 <= len(entries) <= 10
    assert {e.provider for e in entries} >= {"anthropic", "openai", "google"}
    assert all(e.source_url.startswith("https://") for e in entries)
    assert all(e.status == "active" for e in entries)


# ----------------------------------------------------------------------------- cost


P = Pricing(input=2, output=10, cached_input=Decimal("0.2"), reasoning=None, per_request=0)


def test_cost_golden_uncached_only():
    assert cost_of({"input": 1_000_000, "output": 100_000}, P) == Decimal("3.00000000")


def test_cost_cached_tokens_billed_once_at_cached_rate():
    # 600k uncached at $2 + 400k cached at $0.2 + 0 output = 1.2 + 0.08
    cost = cost_of({"input": 600_000, "cached_input": 400_000}, P)
    assert cost == Decimal("1.28000000")


def test_cost_reasoning_defaults_to_output_rate_or_uses_own_rate():
    assert cost_of({"reasoning": 100_000}, P) == Decimal("1.00000000")
    own = Pricing(input=2, output=10, reasoning=Decimal("4"))
    assert cost_of({"reasoning": 100_000}, own) == Decimal("0.40000000")


def test_cost_per_request_added_once_and_rounding():
    p = Pricing(input=Decimal("0.1"), output=Decimal("0.5"), per_request=Decimal("0.002"))
    assert cost_of({"input": 10_000, "output": 1_000}, p) == Decimal("0.00350000")
    assert cost_of({"input": 1, "output": 1}, p) == Decimal("0.00200060")


def test_cost_accepts_pricing_json_from_the_db_row():
    row_pricing = {
        "input": 2,
        "output": 10,
        "cached_input": 0.2,
        "reasoning": None,
        "per_request": 0,
        "verified_at": "2026-10-22",
    }
    assert cost_of({"input": 1_000_000}, row_pricing) == Decimal("2.00000000")


def test_cost_errors():
    with pytest.raises(CatalogError, match="unknown usage keys"):
        cost_of({"prompt_tokens": 5}, P)
    with pytest.raises(CatalogError, match="no cached_input rate"):
        cost_of({"cached_input": 5}, Pricing(input=1, output=1))
    with pytest.raises(CatalogError, match="invalid pricing"):
        cost_of({"input": 5}, {"input": 1})
    with pytest.raises(CatalogError, match="non-negative"):
        cost_of({"input": -1}, P)


# ----------------------------------------------------------------------------- sync


def test_sync_insert_update_deprecate(tmp_path, engine):
    entries = load(_write(tmp_path, GOOD))
    with session_scope(engine) as s:
        r = sync(s, entries)
        assert r.inserted == ["m-a", "m-b"] and not r.updated and not r.deprecated
    with session_scope(engine) as s:
        assert sync(s, entries).unchanged == ["m-a", "m-b"]

    changed = GOOD.replace("output: 4.0", "output: 5.0")
    only_a = load(_write(tmp_path, changed))[:1]
    with session_scope(engine) as s:
        r = sync(s, only_a)
        assert r.updated == ["m-a"] and r.deprecated == ["m-b"]
    with session_scope(engine) as s:
        a = s.get(ModelCatalog, "m-a")
        b = s.get(ModelCatalog, "m-b")
        assert a.pricing["output"] == 5.0 and a.pricing["verified_at"] == "2026-10-22"
        assert b.status == "deprecated" and b.pricing["output"] == 1.5  # kept, not deleted
        assert (
            s.scalar(select(ModelCatalog.status).where(ModelCatalog.model_id == "m-a")) == "active"
        )


# ----------------------------------------------------------------------------- cli


def test_cli_sync_and_show(tmp_path, engine):
    runner = CliRunner()
    path = _write(tmp_path, GOOD)
    r = runner.invoke(app, ["catalog", "sync", "--file", str(path)])
    assert r.exit_code == 0, r.output
    assert "inserted 2" in r.output
    r = runner.invoke(app, ["catalog", "show", "m-a"])
    assert r.exit_code == 0, r.output
    assert "Model A" in r.output and "verified_at=2026-10-22" in r.output
    assert "$0.01500000" in r.output  # 10k * 1.0 + 1k * 4.0 per 1M + 0.001 per request
    r = runner.invoke(app, ["catalog", "show", "nope"])
    assert r.exit_code == 1 and "not in model_catalog" in r.output
    r = runner.invoke(app, ["catalog", "sync", "--file", str(tmp_path / "missing.yaml")])
    assert r.exit_code == 1 and "not found" in r.output
