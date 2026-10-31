"""Replay runner v0: single-turn replay of a sample against one candidate, under a USD budget.

Before every call the worst-case cost is estimated from the original request's usage at the
candidate's rates (cached tokens counted as uncached, since the replay has a cold prefix);
the run stops with ``budget_exhausted`` when ``spent + estimate`` would exceed the cap, so the
budget is never overrun. Spend is accumulated from real returned usage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from omni.catalog import CatalogError, cost_of
from omni.ingest.redact import REDACTION_VERSION
from omni.ingest.store import ObjectStore
from omni.models import ModelCatalog, ReplayResult, ReplayRun, Request, Sample
from omni.replay.adapters import (
    ProviderAdapter,
    ReplayRequest,
    TransientError,
    UnsupportedParameter,
    adapter_for,
    check_capabilities,
)

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 4  # 1 try + 3 retries
BACKOFF_S = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Candidate:
    model_id: str
    effort: str | None = None

    @classmethod
    def parse(cls, spec: str) -> Candidate:
        spec = spec.strip()
        if "@" in spec:
            model_id, effort = spec.rsplit("@", 1)
            return cls(model_id, effort or None)
        return cls(spec, None)

    @property
    def label(self) -> str:
        return f"{self.model_id}@{self.effort}" if self.effort else self.model_id

    def as_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "effort": self.effort}


@dataclass
class RunSummary:
    run: ReplayRun | None
    candidate: Candidate
    sent: int = 0
    errors: int = 0
    spent_usd: float = 0.0
    estimated_usd: float = 0.0
    status: str = "done"
    warnings: list[str] = field(default_factory=list)


def _estimate(usage: dict[str, Any], pricing: dict[str, Any]) -> Decimal:
    proxy = {
        "input": int(usage.get("input") or 0) + int(usage.get("cached_input") or 0),
        "output": int(usage.get("output") or 0),
        "reasoning": int(usage.get("reasoning") or 0),
        "cached_input": 0,
    }
    return cost_of(proxy, pricing)


def call_with_retry(adapter: ProviderAdapter, req: ReplayRequest, sleep=time.sleep):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return adapter.complete(req)
        except TransientError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            log.warning("transient error on %s (attempt %d): %s", req.request_id, attempt + 1, e)
            sleep(BACKOFF_S[attempt])
    raise AssertionError("unreachable")  # pragma: no cover


def run_replay(
    session: Session,
    store: ObjectStore,
    sample: Sample,
    candidate: Candidate,
    budget_usd: float,
    adapter: ProviderAdapter | None = None,
    dry_run: bool = False,
    tenant_id: str = "local",
    sleep=time.sleep,
) -> RunSummary:
    model = session.get(ModelCatalog, candidate.model_id)
    if model is None:
        raise CatalogError(
            f"{candidate.model_id!r} is not in model_catalog (run `omni catalog sync`)"
        )
    pricing, capabilities = model.pricing, model.capabilities or {}
    summary = RunSummary(run=None, candidate=candidate)
    if sample.redaction_version != REDACTION_VERSION:
        summary.warnings.append(
            f"sample redaction_version {sample.redaction_version!r} "
            f"!= current {REDACTION_VERSION!r}"
        )
    requests = {
        r.request_id: r
        for r in session.scalars(select(Request).where(Request.request_id.in_(sample.request_ids)))
    }
    ordered = [requests[rid] for rid in sample.request_ids if rid in requests]
    missing = len(sample.request_ids) - len(ordered)
    if missing:
        summary.warnings.append(f"{missing} sampled request ids no longer exist")

    # A dry run sends nothing, so it must work without provider credentials; a real run
    # resolves the adapter (and its key) here, before any row is written.
    if adapter is None and not dry_run:
        adapter = adapter_for(model.provider)
    run = None
    if not dry_run:
        run = ReplayRun(
            tenant_id=tenant_id,
            workload_id=sample.workload_id,
            sample_id=sample.sample_id,
            candidate=candidate.as_dict(),
            mode="single_turn",
            status="running",
            budget_usd=float(budget_usd),
            adapter_name=adapter.name,
            warnings=list(summary.warnings),
        )
        session.add(run)
        session.flush()
        summary.run = run

    spent = Decimal(0)
    for req in ordered:
        if not req.content_ref or not store.exists(req.content_ref):
            summary.warnings.append(f"{req.request_id}: content missing")
            continue
        doc = store.get(req.content_ref)
        params = {k: v for k, v in (doc.get("params") or {}).items() if k != "effort"}
        rreq = ReplayRequest(
            model_id=candidate.model_id,
            messages=doc.get("messages") or [],
            tools=doc.get("tools") or [],
            params=params,
            effort=candidate.effort,
            request_id=req.request_id,
        )
        estimate = _estimate(req.usage or {}, pricing)
        summary.estimated_usd += float(estimate)
        if dry_run:
            summary.sent += 1
            continue
        if spent + estimate > Decimal(str(budget_usd)):
            run.status = "budget_exhausted"
            break
        result = ReplayResult(tenant_id=tenant_id, run_id=run.run_id, request_id=req.request_id)
        try:
            check_capabilities(rreq, capabilities, candidate.model_id)
            resp = call_with_retry(adapter, rreq, sleep=sleep)
        except UnsupportedParameter as e:
            result.error = f"UnsupportedParameter: {e}"
            run.error_count += 1
        except Exception as e:  # noqa: BLE001 - every outcome is recorded
            result.error = f"{type(e).__name__}: {e}"
            run.error_count += 1
        else:
            cost = cost_of(resp.usage, pricing)
            spent += cost
            result.output_ref = store.put(
                {"request_id": req.request_id, **resp.as_output(), "extras": resp.extras}
            )
            result.usage = resp.usage
            result.cost = float(cost)
            result.latency_ms = resp.latency_ms
            result.mapped_params = resp.mapped_params
            result.provider_region = resp.provider_region
        session.add(result)
        summary.sent += 1
    if run is not None:
        run.spent_usd = float(spent)
        if run.status == "running":
            run.status = "failed" if summary.sent and run.error_count == summary.sent else "done"
        run.finished_at = datetime.now(UTC)
        run.warnings = list(summary.warnings)
        session.flush()
        summary.errors, summary.spent_usd, summary.status = (
            run.error_count,
            run.spent_usd,
            run.status,
        )
    else:
        summary.status = "dry_run"
    return summary
