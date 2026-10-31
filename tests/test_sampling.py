"""Sampling: size formula, strata, allocation, determinism, versioning, golden, CLI."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import select
from typer.testing import CliRunner

from omni.catalog import load, sync
from omni.cli import app
from omni.db import alembic_config, get_engine, session_scope
from omni.ingest import import_file
from omni.ingest.redact import REDACTION_VERSION
from omni.models import Outcome, Request, Sample, Workload
from omni.sampling import SamplingError, create_sample, golden, list_samples, target_size
from omni.sampling.service import allocate
from omni.sampling.strata import stratum_of

FIXTURE = Path(__file__).parent / "fixtures" / "requests_20.jsonl"
MON = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)  # a Monday
SAT = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)  # a Saturday


@pytest.fixture
def engine(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 's.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("OMNI_OBJECT_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("OMNI_REDACTION_FILE", str(tmp_path / "none.yaml"))
    command.upgrade(alembic_config(url), "head")
    eng = get_engine(url)
    with session_scope(eng) as s:
        sync(s, load())
    yield eng
    eng.dispose()


def _req(rid, ts=MON, prompt=500, tools=False, status="success", content="k"):
    return Request(
        request_id=rid,
        tenant_id="local",
        ts=ts,
        source="import",
        status=status,
        content_ref=content,
        tools_hash="t" if tools else None,
        usage={"input": prompt // 2, "cached_input": prompt - prompt // 2, "output": 10},
    )


# ----------------------------------------------------------------------------- size


def test_target_size_formula_and_clamps():
    assert target_size(10_000) == 385
    assert target_size(10_000, ci_half_width=0.10) == 97
    assert target_size(10_000, ci_half_width=0.03) == 1000  # 1068 capped
    assert target_size(10_000, ci_half_width=0.20) == 50  # 25 floored
    assert target_size(120) == 120  # clamped to available
    assert target_size(0) == 0
    with pytest.raises(ValueError):
        target_size(10, ci_half_width=0.6)
    with pytest.raises(ValueError):
        target_size(10, floor=10, cap=5)


# ----------------------------------------------------------------------------- strata


def test_stratum_boundaries():
    assert stratum_of(_req("a", prompt=1000), False) == "lt1k|notools|nooutcome|weekday"
    assert stratum_of(_req("b", prompt=1001), False) == "1k-8k|notools|nooutcome|weekday"
    assert stratum_of(_req("c", prompt=8000, tools=True), True) == "1k-8k|tools|outcome|weekday"
    assert stratum_of(_req("d", prompt=8001), False).startswith("8k-32k|")
    assert stratum_of(_req("e", prompt=32_001, ts=SAT), False) == "gt32k|notools|nooutcome|weekend"
    r = _req("f")
    r.usage = {}
    assert stratum_of(r, False).startswith("lt1k|")


def test_allocation_is_proportional_with_floor_of_one():
    counts = {"a": 400, "b": 90, "c": 9, "d": 1}
    alloc = allocate(counts, 100)
    assert sum(alloc.values()) == 100
    assert alloc["d"] == 1 and alloc["c"] >= 1
    assert abs(alloc["a"] - 80) <= 1 and abs(alloc["b"] - 18) <= 1
    assert all(alloc[s] <= counts[s] for s in counts)
    assert allocate(counts, 500) == counts  # size >= total: everything
    tiny = allocate(counts, 2)  # fewer slots than strata: no floor, biggest first
    assert sum(tiny.values()) == 2 and tiny["a"] >= 1


# ----------------------------------------------------------------------------- service


def _synthetic_workload(session, n=500):
    wl = Workload(tenant_id="local", name="synthetic", fingerprint={}, fingerprint_key="syn")
    session.add(wl)
    session.flush()
    for i in range(n):
        prompt = [500, 4000, 20_000, 40_000][i % 4] if i % 10 else 100
        r = _req(f"r{i:04d}", ts=MON + timedelta(hours=i), prompt=prompt, tools=(i % 3 == 0))
        r.workload_id = wl.workload_id
        session.add(r)
        if i % 7 == 0:
            session.add(
                Outcome(tenant_id="local", request_id=r.request_id, signal="thumb", value=1)
            )
    # ineligible rows: error status, and no content
    e = _req("err", status="error")
    e.workload_id = wl.workload_id
    nc = _req("noc", content=None)
    nc.workload_id = wl.workload_id
    session.add_all([e, nc])
    session.flush()
    return wl


def test_create_sample_proportions_determinism_versions(engine):
    with session_scope(engine) as s:
        wl = _synthetic_workload(s)
        smp = create_sample(s, wl, seed=7)
        assert smp.size == 385 and smp.version == 1 and smp.seed == 7
        assert smp.created_from_range["eligible"] == 500  # err + noc excluded
        assert smp.redaction_version == REDACTION_VERSION == "regex-v1"
        assert smp.strata_def["version"] == "strata-v1"
        counts = smp.strata_def["counts"]
        assert sum(c["sampled"] for c in counts.values()) == 385
        for c in counts.values():
            assert c["sampled"] >= 1
            expected = 385 * c["population"] / 500
            assert abs(c["sampled"] - expected) <= 1.0
        assert "err" not in smp.request_ids and "noc" not in smp.request_ids
        assert len(set(smp.request_ids)) == 385
        ids_a = list(smp.request_ids)

        again = create_sample(s, wl, seed=7)
        assert (
            again.version == 2 and again.request_ids == ids_a and again.sample_id != smp.sample_id
        )
        other = create_sample(s, wl, seed=8)
        assert other.version == 3 and other.request_ids != ids_a
        assert s.get(Sample, smp.sample_id).request_ids == ids_a  # v1 untouched

        small = create_sample(s, wl, size=40, ci_half_width=0.1)
        assert small.size == 40 and small.strata_def["ci_half_width"] == 0.1
        bounded = create_sample(
            s, wl, since=MON + timedelta(hours=100), until=MON + timedelta(hours=200), size=50
        )
        assert bounded.created_from_range["eligible"] == 100
        assert all(100 <= int(r[1:]) < 200 for r in bounded.request_ids)
        with pytest.raises(SamplingError, match="between 1 and 500"):
            create_sample(s, wl, size=501)


def test_golden_lookup_and_listing(engine):
    with session_scope(engine) as s:
        wl = _synthetic_workload(s, n=60)
        assert golden(s, wl) is None
        g1 = create_sample(s, wl, golden=True, seed=1)
        create_sample(s, wl, seed=2)
        g3 = create_sample(s, wl, golden=True, seed=3)
        assert golden(s, wl).sample_id == g3.sample_id and g1.is_golden
        assert [x.version for x in list_samples(s, wl)] == [1, 2, 3]
        assert len(list_samples(s)) == 3


def test_fixture_workload_needs_explicit_size(engine):
    with session_scope(engine) as s:
        import_file(s, FIXTURE)
        wl = s.get(Request, "req_0001").workload  # the 8-request chat family
        with pytest.raises(SamplingError, match="fewer than the floor"):
            create_sample(s, wl)
        smp = create_sample(s, wl, size=5, seed=1)
        assert smp.size == 5 and set(smp.request_ids) <= {f"req_{i:04d}" for i in range(1, 9)}
        empty = Workload(tenant_id="local", name="empty", fingerprint={}, fingerprint_key="e")
        s.add(empty)
        with pytest.raises(SamplingError, match="no eligible"):
            create_sample(s, empty, size=1)


# ----------------------------------------------------------------------------- cli


def test_cli_create_list_show(engine):
    runner = CliRunner()
    assert runner.invoke(app, ["ingest", "--from", str(FIXTURE)]).exit_code == 0
    with session_scope(engine) as s:
        wl_id = s.get(Request, "req_0001").workload.workload_id
    r = runner.invoke(app, ["sample", "create", "--workload", wl_id])
    assert r.exit_code == 1 and "fewer than the floor" in r.output
    r = runner.invoke(
        app, ["sample", "create", "--workload", wl_id, "--size", "5", "--seed", "3", "--golden"]
    )
    assert r.exit_code == 0, r.output
    assert (
        "v1" in r.output
        and "5 requests" in r.output
        and "golden" in r.output
        and "stratum" in r.output
    )
    sample_id = r.output.split()[0]
    r = runner.invoke(app, ["sample", "list"])
    assert r.exit_code == 0 and sample_id in r.output and "yes" in r.output
    r = runner.invoke(app, ["sample", "show", sample_id])
    assert r.exit_code == 0, r.output
    assert (
        "strata=strata-v1" in r.output
        and "redaction=regex-v1" in r.output
        and "request_ids: req_" in r.output
    )
    r = runner.invoke(app, ["sample", "show", "smp_nope"])
    assert r.exit_code == 1
    r = runner.invoke(
        app, ["sample", "create", "--workload", wl_id, "--size", "2", "--since", "not-a-date"]
    )
    assert r.exit_code == 1 and "ISO-8601" in r.output
    with session_scope(engine) as s:
        assert s.scalar(select(Sample).where(Sample.sample_id == sample_id)).is_golden
