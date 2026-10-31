"""Ingest: schema, object store, importer semantics, CLI."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from pydantic import ValidationError
from sqlalchemy import func, select
from typer.testing import CliRunner

from omni.catalog import load, sync
from omni.cli import app
from omni.db import alembic_config, get_engine, session_scope
from omni.ingest import ImportResult, NormalizedRequest, ObjectStore, import_file
from omni.models import Request, Workload

FIXTURE = Path(__file__).parent / "fixtures" / "requests_20.jsonl"

MINIMAL = {"request_id": "r1", "ts": "2026-10-30T14:00:00Z"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'ing.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("OMNI_OBJECT_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("OMNI_REDACTION_FILE", str(tmp_path / "none.yaml"))
    command.upgrade(alembic_config(url), "head")
    engine = get_engine(url)
    with session_scope(engine) as s:
        sync(s, load())
    yield engine, tmp_path / "objects"
    engine.dispose()


# ----------------------------------------------------------------------------- schema


def test_schema_minimal_and_defaults():
    r = NormalizedRequest.model_validate(MINIMAL)
    assert r.source == "import" and r.status == "success" and r.usage.input == 0
    assert r.ts.tzinfo is not None and r.tools_hash is None and r.cost is None


def test_schema_rejects_unknown_key_bad_source_negative_usage():
    with pytest.raises(ValidationError):
        NormalizedRequest.model_validate({**MINIMAL, "colour": "red"})
    with pytest.raises(ValidationError):
        NormalizedRequest.model_validate({**MINIMAL, "source": "csv"})
    with pytest.raises(ValidationError):
        NormalizedRequest.model_validate({**MINIMAL, "usage": {"input": -1}})
    with pytest.raises(ValidationError):
        NormalizedRequest.model_validate({**MINIMAL, "messages": [{"role": "bot", "content": "x"}]})


def test_tools_hash_stable_across_key_order_and_raw_hash_changes_with_content():
    t1 = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
            },
        }
    ]
    t2 = [
        {
            "function": {
                "parameters": {"properties": {"a": {"type": "string"}}, "type": "object"},
                "name": "f",
            },
            "type": "function",
        }
    ]
    a = NormalizedRequest.model_validate({**MINIMAL, "tools": t1})
    b = NormalizedRequest.model_validate({**MINIMAL, "tools": t2})
    assert a.tools_hash == b.tools_hash and len(a.tools_hash) == 16
    c = NormalizedRequest.model_validate(
        {**MINIMAL, "messages": [{"role": "user", "content": "hi"}]}
    )
    assert a.raw_hash == b.raw_hash and c.raw_hash != a.raw_hash


# ----------------------------------------------------------------------------- store


def test_object_store_round_trip_and_layout(tmp_path):
    store = ObjectStore(tmp_path / "obj")
    key = store.put({"b": 1, "a": [1, 2]})
    assert len(key) == 64 and store.exists(key)
    assert store.path_for(key) == tmp_path / "obj" / key[:2] / f"{key}.json"
    assert store.get(key) == {"a": [1, 2], "b": 1}
    assert store.put({"a": [1, 2], "b": 1}) == key  # same content, same key, no second file
    assert len(list((tmp_path / "obj").rglob("*.json"))) == 1


# ----------------------------------------------------------------------------- importer


def test_import_fixture_then_reimport_is_noop(env):
    engine, objects = env
    with session_scope(engine) as s:
        r = import_file(s, FIXTURE)
    assert r.inserted == 20 and r.skipped == 0 and r.conflicts == 0 and r.invalid == []
    assert r.costed == 18  # 20 minus one with cost given minus one unknown model
    assert r.redactions["EMAIL"] >= 4 and r.redactions["CARD"] == 1 and r.redactions["API_KEY"] == 1
    assert r.redactions["PHONE"] == 2 and r.redactions["IPV4"] == 1

    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(Request)) == 20
        workloads = s.scalars(select(Workload)).all()
        assert "unassigned" not in {w.name for w in workloads}  # every line has messages
        assert sum(len(w.requests) for w in workloads) == 20
        assert len(r.workloads_created) == len(workloads)
        given = s.get(Request, "req_0004")
        assert given.cost == pytest.approx(0.0042)  # kept as given
        computed = s.get(Request, "req_0001")  # haiku: 317 in / 41 out
        assert computed.cost == pytest.approx(float(Decimal("0.0000522")), abs=1e-9)
        unknown = s.get(Request, "req_0008")
        assert unknown.cost is None
        cached = s.get(Request, "req_0013")
        assert cached.usage["cached_input"] == 5400 and cached.cost is not None
        step = s.get(Request, "req_0012")
        assert step.session_id == "sess-001" and step.step_index == 3 and step.compaction_event
        assert step.tools_hash and s.get(Request, "req_0009").tools_hash == step.tools_hash
        pii = s.get(Request, "req_0017")
        doc = ObjectStore(objects).get(pii.content_ref)
        text = json.dumps(doc)
        assert "jane.doe@example.com" not in text and "4111" not in text and "555-0133" not in text
        assert "[EMAIL_1]" in text and "[CARD_1]" in text
        assert pii.raw_hash and len(pii.raw_hash) == 64
        key = s.get(Request, "req_0019")
        assert "sk-ant-" not in json.dumps(ObjectStore(objects).get(key.content_ref))

    with session_scope(engine) as s:
        again = import_file(s, FIXTURE)
    assert again.inserted == 0 and again.skipped == 20 and again.conflicts == 0


def test_invalid_lines_reported_and_conflict_detected(env, tmp_path):
    engine, _ = env
    good = json.dumps({**MINIMAL, "messages": [{"role": "user", "content": "a"}]})
    bad_json = "{not json"
    bad_schema = json.dumps({**MINIMAL, "request_id": "r2", "usage": {"input": -5}})
    p = tmp_path / "mixed.jsonl"
    p.write_text("\n".join([good, bad_json, bad_schema, "", "# comment"]) + "\n", encoding="utf-8")
    with session_scope(engine) as s:
        r = import_file(s, p, source="litellm")
    assert r.inserted == 1 and [ln for ln, _ in r.invalid] == [2, 3]
    assert "Expecting" in r.invalid[0][1] and "input" in r.invalid[1][1]
    with session_scope(engine) as s:
        assert s.get(Request, "r1").source == "litellm"

    p.write_text(
        json.dumps({**MINIMAL, "messages": [{"role": "user", "content": "b"}]}) + "\n",
        encoding="utf-8",
    )
    with session_scope(engine) as s:
        r = import_file(s, p)
    assert r.conflicts == 1 and r.inserted == 0
    with session_scope(engine) as s:
        assert s.get(Request, "r1").source == "litellm"  # first write wins


def test_dry_run_writes_nothing(env):
    engine, objects = env
    with session_scope(engine) as s:
        r: ImportResult = import_file(s, FIXTURE, dry_run=True)
    assert r.dry_run and r.inserted == 20 and r.redactions["EMAIL"] >= 4
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(Request)) == 0
        assert s.scalar(select(func.count()).select_from(Workload)) == 0
    assert not objects.exists()


# ----------------------------------------------------------------------------- cli


def test_cli_ingest(env, tmp_path):
    engine, objects = env
    runner = CliRunner()
    r = runner.invoke(app, ["ingest", "--from", str(FIXTURE), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "would insert 20" in r.output and not objects.exists()
    r = runner.invoke(app, ["ingest", "--from", str(FIXTURE), "--source", "sidecar"])
    assert r.exit_code == 0, r.output
    assert "inserted 20" in r.output and "EMAIL=" in r.output
    with session_scope(engine) as s:
        assert s.get(Request, "req_0001").source == "sidecar"
    r = runner.invoke(app, ["ingest", "--from", str(tmp_path / "missing.jsonl")])
    assert r.exit_code == 1 and "does not exist" in r.output
