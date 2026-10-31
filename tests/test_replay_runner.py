"""Runner: budget boundary, retries, fixture end-to-end, dry run, warnings, CLI."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import func, select
from typer.testing import CliRunner

from omni.catalog import cost_of, load, sync
from omni.cli import app
from omni.db import alembic_config, get_engine, session_scope
from omni.ingest import ObjectStore, import_file
from omni.models import ModelCatalog, ReplayResult, ReplayRun, Request, Workload
from omni.replay import Candidate, FixtureAdapter, ReplayResponse, TransientError, run_replay
from omni.sampling import create_sample

FIXTURES = Path(__file__).parent / "fixtures"
REQUESTS = FIXTURES / "requests_20.jsonl"
RESPONSES = FIXTURES / "replay_responses.json"
UNIFORM = {"input": 1000, "output": 100, "cached_input": 0, "reasoning": 0}


@pytest.fixture
def env(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'r.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("OMNI_OBJECT_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("OMNI_REDACTION_FILE", str(tmp_path / "none.yaml"))
    monkeypatch.delenv("OMNI_REPLAY_ADAPTER", raising=False)
    command.upgrade(alembic_config(url), "head")
    engine = get_engine(url)
    with session_scope(engine) as s:
        sync(s, load())
    yield engine, ObjectStore(tmp_path / "objects")
    engine.dispose()


def _uniform_workload(session, store, n=5):
    """n successful requests with identical usage so estimates equal real spend."""
    wl = Workload(tenant_id="local", name="uniform", fingerprint={}, fingerprint_key="u")
    session.add(wl)
    session.flush()
    for i in range(n):
        ref = store.put(
            {
                "request_id": f"u{i}",
                "messages": [{"role": "user", "content": f"q{i}"}],
                "tools": [],
                "params": {"max_tokens": 100},
            }
        )
        session.add(
            Request(
                request_id=f"u{i}",
                tenant_id="local",
                workload_id=wl.workload_id,
                source="import",
                ts=datetime(2026, 10, 27, tzinfo=UTC) + timedelta(hours=i),
                usage=dict(UNIFORM),
                content_ref=ref,
            )
        )
    session.flush()
    return wl, create_sample(session, wl, size=n, seed=1)


class ScriptedAdapter:
    """Fails `failures` times with TransientError, then answers with UNIFORM usage."""

    name = "scripted"

    def __init__(self, failures=0, per_request=False):
        self.failures, self.per_request, self.calls, self.seen = failures, per_request, 0, {}

    def complete(self, req):
        self.calls += 1
        n = self.seen.get(req.request_id, 0)
        self.seen[req.request_id] = n + 1
        if n < self.failures:
            raise TransientError(f"flaky {n}")
        return ReplayResponse(
            content=f"ans {req.request_id}",
            tool_calls=[],
            finish_reason="stop",
            usage=dict(UNIFORM),
            latency_ms=5.0,
            provider="scripted",
            mapped_params={"k": 1},
        )


def _price(session, model_id):
    return session.get(ModelCatalog, model_id).pricing


def test_budget_boundary_is_exact(env):
    engine, store = env
    with session_scope(engine) as s:
        wl, smp = _uniform_workload(s, store)
        e = cost_of(UNIFORM, _price(s, "claude-haiku-5-5"))  # per-call estimate == real cost
        r = run_replay(
            s,
            store,
            smp,
            Candidate("claude-haiku-5-5"),
            budget_usd=float(2 * e),
            adapter=ScriptedAdapter(),
            sleep=lambda _: None,
        )
        assert (
            r.status == "budget_exhausted"
            and r.sent == 2
            and r.spent_usd == pytest.approx(float(2 * e))
        )
        r = run_replay(
            s,
            store,
            smp,
            Candidate("claude-haiku-5-5"),
            budget_usd=float(2 * e - Decimal("0.00000001")),
            adapter=ScriptedAdapter(),
            sleep=lambda _: None,
        )
        assert r.status == "budget_exhausted" and r.sent == 1
        r = run_replay(
            s,
            store,
            smp,
            Candidate("claude-haiku-5-5"),
            budget_usd=1.0,
            adapter=ScriptedAdapter(),
            sleep=lambda _: None,
        )
        assert r.status == "done" and r.sent == 5 and r.errors == 0
        assert r.spent_usd == pytest.approx(float(5 * e)) and r.run.spent_usd == r.spent_usd
        assert (
            r.run.finished_at is not None
            and r.run.mode == "single_turn"
            and r.run.adapter_name == "scripted"
        )


def test_retries_then_success_and_exhausted_retries_recorded(env):
    engine, store = env
    with session_scope(engine) as s:
        wl, smp = _uniform_workload(s, store, n=1)
        slept = []
        flaky = ScriptedAdapter(failures=2)
        r = run_replay(
            s, store, smp, Candidate("claude-haiku-5-5"), 1.0, adapter=flaky, sleep=slept.append
        )
        assert r.status == "done" and r.errors == 0 and flaky.calls == 3 and slept == [0.5, 1.0]
        dead = ScriptedAdapter(failures=4)
        r = run_replay(
            s, store, smp, Candidate("claude-haiku-5-5"), 1.0, adapter=dead, sleep=lambda _: None
        )
        assert dead.calls == 4 and r.errors == 1 and r.sent == 1 and r.status == "failed"
        res = s.scalar(select(ReplayResult).where(ReplayResult.run_id == r.run.run_id))
        assert (
            res.error.startswith("TransientError") and res.output_ref is None and res.cost is None
        )
        assert r.spent_usd == 0.0


def test_unsupported_effort_recorded_without_sending(env):
    engine, store = env
    with session_scope(engine) as s:
        wl, smp = _uniform_workload(s, store, n=2)
        a = ScriptedAdapter()
        r = run_replay(
            s,
            store,
            smp,
            Candidate("Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "high"),
            1.0,
            adapter=a,
        )
        assert a.calls == 0 and r.errors == 2 and r.status == "failed"
        assert all("UnsupportedParameter" in x.error for x in r.run.results)


def test_missing_key_fails_before_a_run_is_created(env, monkeypatch):
    """Regression: a missing provider key used to surface as one error per sampled request
    and a `failed` run; it must raise MissingApiKey before any ReplayRun row exists."""
    from omni.replay import MissingApiKey

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    engine, store = env
    with session_scope(engine) as s:
        wl, smp = _uniform_workload(s, store, n=2)
        with pytest.raises(MissingApiKey, match="ANTHROPIC_API_KEY"):
            run_replay(s, store, smp, Candidate("claude-haiku-5-5"), 1.0)
        assert s.scalar(select(func.count()).select_from(ReplayRun)) == 0


def test_dry_run_estimates_and_writes_nothing(env):
    engine, store = env
    with session_scope(engine) as s:
        wl, smp = _uniform_workload(s, store)
        e = cost_of(UNIFORM, _price(s, "gpt-6-luna"))
        a = ScriptedAdapter()
        r = run_replay(s, store, smp, Candidate("gpt-6-luna", "low"), 0.0, adapter=a, dry_run=True)
        assert r.status == "dry_run" and r.run is None and a.calls == 0
        assert r.sent == 5 and r.estimated_usd == pytest.approx(float(5 * e)) and r.spent_usd == 0.0
        assert s.scalar(select(func.count()).select_from(ReplayRun)) == 0


def test_redaction_version_mismatch_warns(env):
    engine, store = env
    with session_scope(engine) as s:
        wl, smp = _uniform_workload(s, store, n=1)
        smp.redaction_version = "regex-v0"
        r = run_replay(s, store, smp, Candidate("claude-haiku-5-5"), 1.0, adapter=ScriptedAdapter())
        assert any("redaction_version" in w for w in r.warnings) and r.run.warnings == r.warnings
        assert r.status == "done"


def test_fixture_adapter_end_to_end_over_imported_sample(env):
    engine, store = env
    with session_scope(engine) as s:
        import_file(s, REQUESTS)
        wl = s.get(Request, "req_0001").workload  # 8 concise chats
        smp = create_sample(s, wl, size=8, seed=1)
        fx = FixtureAdapter(RESPONSES)
        r = run_replay(s, store, smp, Candidate("gpt-6.1-sol"), budget_usd=5.0, adapter=fx)
        assert r.status == "done" and r.sent == 8 and r.errors == 0 and len(fx.calls) == 8
        pricing = _price(s, "gpt-6.1-sol")
        expected = sum(
            cost_of(fx.responses[f"gpt-6.1-sol|{rid}"]["usage"], pricing) for rid in smp.request_ids
        )
        assert r.spent_usd == pytest.approx(float(expected)) and r.spent_usd > 0
        results = s.scalars(select(ReplayResult).where(ReplayResult.run_id == r.run.run_id)).all()
        assert {x.request_id for x in results} == set(smp.request_ids)
        one = results[0]
        assert store.get(one.output_ref)["content"] and one.mapped_params["fixture"] is True
        assert one.latency_ms and one.usage["cached_input"] == 0
        # the first call sent exactly the stored prompt with the candidate model and no effort
        sent = fx.calls[0]
        assert sent.model_id == "gpt-6.1-sol" and sent.effort is None
        assert sent.messages == store.get(s.get(Request, sent.request_id).content_ref)["messages"]
        # haiku has one recorded hard error in the fixture (req_0015 is not in this workload)
        r2 = run_replay(s, store, smp, Candidate("claude-haiku-5-5", "low"), 5.0, adapter=fx)
        assert r2.status == "done" and r2.errors == 0 and fx.calls[-1].effort == "low"


def test_cli_run_list_show(env, tmp_path):
    engine, store = env
    runner = CliRunner()
    assert runner.invoke(app, ["ingest", "--from", str(REQUESTS)]).exit_code == 0
    with session_scope(engine) as s:
        wl_id = s.get(Request, "req_0001").workload.workload_id
    assert (
        runner.invoke(
            app, ["sample", "create", "--workload", wl_id, "--size", "8", "--seed", "1"]
        ).exit_code
        == 0
    )
    r = runner.invoke(
        app, ["replay", "run", "--workload", wl_id, "--candidates", "gpt-6-luna@low", "--dry-run"]
    )
    assert r.exit_code == 0, r.output
    assert "[dry run]" in r.output and "dry_run" in r.output
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(ReplayRun)) == 0
    r = runner.invoke(
        app,
        [
            "replay",
            "run",
            "--workload",
            wl_id,
            "--candidates",
            "claude-sonnet-5-5@low,gpt-6-luna,nope-model",
            "--budget",
            "2",
        ],
        env={"OMNI_REPLAY_ADAPTER": f"fixture:{RESPONSES}"},
    )
    assert r.exit_code == 0, r.output
    lines = [
        ln
        for ln in r.output.splitlines()
        if ln.startswith(("claude-sonnet-5-5@low", "gpt-6-luna", "nope-model"))
    ]
    assert (
        len(lines) == 3 and "done" in lines[0] and "done" in lines[1] and "not started" in lines[2]
    )
    run_id = lines[0].split()[1]
    r = runner.invoke(app, ["replay", "list", "--workload", wl_id])
    assert r.exit_code == 0 and run_id in r.output and "claude-sonnet-5-5@low" in r.output
    r = runner.invoke(app, ["replay", "show", run_id])
    assert r.exit_code == 0 and "status=done" in r.output and "req_0001" in r.output
    r = runner.invoke(app, ["replay", "show", "run_nope"])
    assert r.exit_code == 1
    r = runner.invoke(
        app,
        ["replay", "run", "--workload", wl_id, "--candidates", "gpt-6-luna", "--sample", "golden"],
    )
    assert r.exit_code == 1 and "no sample" in r.output
