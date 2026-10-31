"""Workloads: masking, fingerprints, step kinds, segmentation on import, refresh, merge, CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import func, select
from typer.testing import CliRunner

from omni.catalog import load, sync
from omni.cli import app
from omni.db import alembic_config, get_engine, session_scope
from omni.ingest import NormalizedRequest, import_file
from omni.models import Request, Workload
from omni.workloads import fingerprint, mask_system_prompt, merge, refresh, segment
from omni.workloads.fingerprint import derive_step_kind

FIXTURE = Path(__file__).parent / "fixtures" / "requests_20.jsonl"
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


def _rec(system: str | None = None, tools=None, rf=None, step_index=None, messages=None, **kw):
    msgs = [{"role": "system", "content": system}] if system else []
    msgs += messages or [{"role": "user", "content": "hi"}]
    return NormalizedRequest.model_validate(
        {
            "request_id": kw.pop("request_id", "r"),
            "ts": "2026-10-30T00:00:00Z",
            "messages": msgs,
            "tools": tools or [],
            "params": {"response_format": rf} if rf else {},
            "step_index": step_index,
            **kw,
        }
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'wl.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("OMNI_OBJECT_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("OMNI_REDACTION_FILE", str(tmp_path / "none.yaml"))
    command.upgrade(alembic_config(url), "head")
    engine = get_engine(url)
    with session_scope(engine) as s:
        sync(s, load())
    yield engine, tmp_path
    engine.dispose()


# ----------------------------------------------------------------------------- masking


def test_masking_collapses_variables_to_one_template():
    a = (
        "You help customer 48213 (id 3f9a8c7e-1b2d-4c5e-8f90-123456789abc) on 2026-10-30 at 14:02. "
        'Order ref a1b2c3d4e5f6; see https://x.io/o/1 and note "a very long quoted piece of '
        'customer supplied text here" from [EMAIL_1]; template {{name}} / {var} / <slot>'
    )
    b = (
        "You help customer 7 (id 00000000-0000-4000-8000-000000000000) on 2025-01-02 at 09:15. "
        'Order ref deadbeef99; see https://y.org/z and note "another very long quoted piece of '
        'completely different text goes here" from [EMAIL_2]; template {{other}} / {x} / <y>'
    )
    assert mask_system_prompt(a) == mask_system_prompt(b)
    m = mask_system_prompt(a)
    for tok in (
        "@NUM@",
        "@UUID@",
        "@DATE@",
        "@TIME@",
        "@HEX@",
        "@URL@",
        "@STR@",
        "@REDACTED@",
        "@VAR@",
    ):
        assert tok in m, tok
    assert "48213" not in m and "x.io" not in m and "{{" not in m
    assert mask_system_prompt("  a   b\n\n c ") == "a b c"


def test_same_template_one_key_different_tools_or_format_split():
    k1 = fingerprint(_rec("Agent for repo 42, session 9a8b7c6d5e")).key
    k2 = fingerprint(_rec("Agent for repo 7, session 1122334455")).key
    assert k1 == k2 and len(k1) == 16
    assert fingerprint(_rec("Agent for repo 42", tools=TOOLS)).key != k1
    other_tools = [
        {"type": "function", "function": {"name": "g", "parameters": {"type": "object"}}}
    ]
    assert (
        fingerprint(_rec("Agent", tools=TOOLS)).key
        != fingerprint(_rec("Agent", tools=other_tools)).key
    )
    assert fingerprint(_rec("Agent", rf={"type": "json_object"})).key != k1
    assert fingerprint(_rec("Agent", rf={"type": "json_object"})).response_format == "json_object"
    assert (
        fingerprint(_rec("Agent for repo 1"), app_tag="a").key
        != fingerprint(_rec("Agent for repo 1"), app_tag="b").key
    )
    assert fingerprint(_rec("Agent for repo 1", app_tag="a")).app_tag == "a"


def test_step_kind_rules():
    u = [{"role": "user", "content": "x"}]
    a = [{"role": "assistant", "content": "y"}]
    t = [{"role": "tool", "tool_call_id": "c", "content": "r"}]
    calls = {"tool_calls": [{"name": "f"}]}
    assert derive_step_kind(u, [], None, None, False) == "chat"
    assert derive_step_kind(u + a + t, [], None, 3, False) == "chat"  # no tools offered
    assert derive_step_kind(u, TOOLS, calls, 0, False) == "first_turn"  # step 0 wins over tool_step
    assert derive_step_kind(u, TOOLS, None, None, False) == "first_turn"  # no prior assistant
    assert derive_step_kind(u + a + t, TOOLS, None, 2, False) == "tool_step"  # last is tool result
    assert (
        derive_step_kind(u + a + u, TOOLS, calls, 2, False) == "tool_step"
    )  # response calls a tool
    assert derive_step_kind(u + a + u, TOOLS, None, 2, False) == "tool_step"  # fallback
    assert derive_step_kind(u + a + t, TOOLS, calls, 5, True) == "compaction"
    assert derive_step_kind(u, [], None, None, True) == "compaction"


# ----------------------------------------------------------------------------- segmentation


def test_import_segments_fixture_into_nine_workloads(env):
    engine, _ = env
    with session_scope(engine) as s:
        r = import_file(s, FIXTURE)
        assert len(r.workloads_created) == 9
        by_name = {w.name: w for w in s.scalars(select(Workload))}
        assert len(by_name) == 9 and all(n.startswith("wl-") for n in by_name)
        sizes = sorted(len(w.requests) for w in by_name.values())
        # 8 concise-assistant chats; 4 chats with no system prompt; the tool session splits by
        # step kind into first_turn (1), tool_step (2), compaction (1); four one-offs.
        assert sizes == [1, 1, 1, 1, 1, 1, 2, 4, 8]
        kinds = {w.fingerprint["step_kind"] for w in by_name.values()}
        assert kinds == {"chat", "first_turn", "tool_step", "compaction"}
        assert s.get(Request, "req_0010").step_kind == "tool_step"
        assert s.get(Request, "req_0012").step_kind == "compaction"
        assert s.get(Request, "req_0020").workload.fingerprint["response_format"] == "json_schema"
        assert s.get(Request, "req_0001").workload is s.get(Request, "req_0008").workload


def test_segment_moves_rows_out_of_unassigned(env):
    engine, tmp = env
    with session_scope(engine) as s:
        import_file(s, FIXTURE)
        # simulate rows imported before segmentation existed: park them in `unassigned`
        from omni.workloads.service import WorkloadResolver

        unassigned = WorkloadResolver(s).unassigned()
        for req in s.scalars(select(Request)):
            req.workload_id = unassigned.workload_id
            req.step_kind = None
        for w in s.scalars(select(Workload).where(Workload.name != "unassigned")):
            s.delete(w)
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(Workload)) == 1
        r = segment(s)
        assert r.moved == 20 and r.unchanged == 0 and r.no_content == 0 and len(r.created) == 9
        assert s.scalar(select(func.count()).select_from(Workload)) == 10
        assert s.get(Request, "req_0012").step_kind == "compaction"
        again = segment(s, all_requests=True)
        assert again.moved == 0 and again.unchanged == 20 and again.created == []
    # a request whose object is missing is counted, not moved
    with session_scope(engine) as s:
        req = s.get(Request, "req_0001")
        req.content_ref = "0" * 64
        req.workload_id = WorkloadResolver(s).unassigned().workload_id
    with session_scope(engine) as s:
        assert segment(s).no_content == 1


# ----------------------------------------------------------------------------- refresh / merge


def _mk(session, wl, rid, ts, model, cost, usage):
    session.add(
        Request(
            request_id=rid,
            tenant_id="local",
            workload_id=wl.workload_id,
            ts=ts,
            source="import",
            model_resolved=model,
            cost=cost,
            usage=usage,
        )
    )


def test_refresh_computes_stats_over_trailing_window(env):
    engine, _ = env
    now = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    with session_scope(engine) as s:
        wl = Workload(tenant_id="local", name="w", fingerprint={}, fingerprint_key="k")
        s.add(wl)
        s.flush()
        u = {"input": 100, "output": 10, "cached_input": 300, "reasoning": 5}
        _mk(s, wl, "a", now - timedelta(days=1), "m1", 0.5, u)
        _mk(s, wl, "b", now - timedelta(days=2), "m1", 0.25, u)
        _mk(s, wl, "c", now - timedelta(days=3), "m2", None, {"input": 100, "output": 0})
        _mk(s, wl, "old", now - timedelta(days=45), "m2", 99.0, {"input": 10**6})  # outside window
        refresh(s, now=now)
        s.refresh(wl)
        assert wl.volume_30d == 3 and wl.spend_30d == pytest.approx(0.75)
        assert wl.token_mix == {"input": 300, "output": 20, "cached": 600, "reasoning": 10}
        assert wl.cache_read_ratio == pytest.approx(600 / 900)
        assert wl.incumbent_model_id == "m1"
        empty = Workload(tenant_id="local", name="e", fingerprint={}, fingerprint_key="e")
        s.add(empty)
        refresh(s, now=now)
        assert (
            empty.volume_30d == 0
            and empty.cache_read_ratio is None
            and empty.incumbent_model_id is None
        )


def test_merge_moves_requests_and_routes_future_imports(env, tmp_path):
    engine, _ = env
    with session_scope(engine) as s:
        import_file(s, FIXTURE)
        src = s.get(Request, "req_0017").workload  # triage one-off
        dst = s.get(Request, "req_0001").workload  # concise-assistant family
        src_key, dst_id = src.fingerprint_key, dst.workload_id
        assert merge(s, src, dst) == 1
        assert src.name.startswith("merged:") and src.merge_rules == {"merged_into": dst_id}
        assert dst.merge_rules["absorbed"] == [src_key]
        assert len(dst.requests) == 9 and src.requests == []
        with pytest.raises(ValueError):
            merge(s, src, dst)  # already merged
        with pytest.raises(ValueError):
            merge(s, dst, dst)
    # a new request with src's fingerprint lands in dst
    line = json.dumps(
        {
            "request_id": "req_new",
            "ts": "2026-10-31T00:00:00Z",
            "messages": [
                {"role": "system", "content": "You triage support tickets."},
                {"role": "user", "content": "Ticket 999: refund"},
            ],
            "response": {"content": "billing"},
        }
    )
    p = tmp_path / "new.jsonl"
    p.write_text(line + "\n", encoding="utf-8")
    with session_scope(engine) as s:
        r = import_file(s, p)
        assert r.inserted == 1 and r.workloads_created == []
        assert s.get(Request, "req_new").workload_id == dst_id
        assert s.scalar(select(func.count()).select_from(Workload)) == 9  # no new workload


# ----------------------------------------------------------------------------- cli


def test_cli_list_rename_merge_segment(env):
    engine, _ = env
    runner = CliRunner()
    r = runner.invoke(app, ["ingest", "--from", str(FIXTURE)])
    assert r.exit_code == 0 and "new workloads:" in r.output
    r = runner.invoke(app, ["workloads", "refresh"])
    assert r.exit_code == 0 and "refreshed 9 workloads" in r.output
    r = runner.invoke(app, ["workloads", "list", "--sort", "volume"])
    assert r.exit_code == 0, r.output
    lines = r.output.strip().splitlines()
    assert lines[0].startswith("workload_id") and len(lines) == 10
    assert "chat=8" in lines[1] and "claude-haiku-5-5" in lines[1]
    with session_scope(engine) as s:
        big = s.get(Request, "req_0001").workload.workload_id
        small = s.get(Request, "req_0017").workload.workload_id
    r = runner.invoke(app, ["workloads", "rename", big, "concise-chat"])
    assert r.exit_code == 0 and "-> concise-chat" in r.output
    r = runner.invoke(app, ["workloads", "merge", small, "concise-chat"])
    assert r.exit_code == 0 and "moved 1 requests" in r.output
    r = runner.invoke(app, ["workloads", "merge", small, "concise-chat"])
    assert r.exit_code == 1 and "already merged" in r.output
    r = runner.invoke(app, ["workloads", "segment"])
    assert r.exit_code == 0 and "moved 0" in r.output
    r = runner.invoke(app, ["workloads", "list", "--sort", "bogus"])
    assert r.exit_code == 1
    r = runner.invoke(app, ["workloads", "rename", "nope", "x"])
    assert r.exit_code == 1 and "not found" in r.output
