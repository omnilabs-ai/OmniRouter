"""Workload operations: resolve a fingerprint to a workload, re-segment, refresh stats, merge."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from omni.models import Request, Workload
from omni.workloads.fingerprint import Fingerprint, auto_name, fingerprint_parts

if TYPE_CHECKING:  # omni.ingest imports this module at runtime; keep the edge type-only
    from omni.ingest.store import ObjectStore

UNASSIGNED = "unassigned"


class WorkloadResolver:
    """Per-tenant cache: fingerprint key (or an absorbed key after a merge) -> Workload."""

    def __init__(self, session: Session, tenant_id: str = "local"):
        self.session = session
        self.tenant_id = tenant_id
        self.by_key: dict[str, Workload] = {}
        workloads = session.scalars(select(Workload).where(Workload.tenant_id == tenant_id)).all()
        # A merged-away workload keeps its row but must not claim its key; the absorbing
        # workload does, via merge_rules.absorbed (registered last so it always wins).
        for wl in workloads:
            if not (wl.merge_rules or {}).get("merged_into"):
                self.by_key[wl.fingerprint_key] = wl
        for wl in workloads:
            for absorbed in (wl.merge_rules or {}).get("absorbed", []):
                self.by_key[absorbed] = wl

    def unassigned(self) -> Workload:
        wl = self.by_key.get(UNASSIGNED)
        if wl is None:
            wl = Workload(
                tenant_id=self.tenant_id,
                name=UNASSIGNED,
                fingerprint={"kind": UNASSIGNED},
                fingerprint_key=UNASSIGNED,
            )
            self.session.add(wl)
            self.session.flush()
            self.by_key[UNASSIGNED] = wl
        return wl

    def resolve(self, fp: Fingerprint) -> Workload:
        wl = self.by_key.get(fp.key)
        if wl is None:
            wl = Workload(
                tenant_id=self.tenant_id,
                name=auto_name(fp),
                fingerprint=fp.as_dict(),
                fingerprint_key=fp.key,
            )
            self.session.add(wl)
            self.session.flush()
            self.by_key[fp.key] = wl
        return wl


@dataclass
class SegmentResult:
    moved: int = 0
    unchanged: int = 0
    no_content: int = 0
    created: list[str] = field(default_factory=list)


def segment(
    session: Session,
    store: ObjectStore | None = None,
    tenant_id: str = "local",
    all_requests: bool = False,
) -> SegmentResult:
    """Re-fingerprint requests in ``unassigned`` (or every request with ``all_requests``) from
    their stored content and move them to the right workload."""
    if store is None:
        from omni.ingest.store import ObjectStore

        store = ObjectStore()
    resolver = WorkloadResolver(session, tenant_id)
    result = SegmentResult()
    before = set(resolver.by_key)
    q = select(Request).where(Request.tenant_id == tenant_id)
    if not all_requests:
        q = q.where(Request.workload_id == resolver.unassigned().workload_id)
    for req in session.scalars(q):
        if not req.content_ref or not store.exists(req.content_ref):
            result.no_content += 1
            continue
        doc = store.get(req.content_ref)
        if not doc.get("messages"):
            result.no_content += 1
            continue
        fp = fingerprint_parts(
            messages=doc["messages"],
            tools=doc.get("tools") or [],
            tools_hash=req.tools_hash,
            response_format=(doc.get("params") or {}).get("response_format"),
            response=doc.get("response"),
            step_index=req.step_index,
            compaction_event=req.compaction_event,
            app_tag=doc.get("app_tag"),
        )
        wl = resolver.resolve(fp)
        req.step_kind = fp.step_kind
        if req.workload_id != wl.workload_id:
            req.workload_id = wl.workload_id
            result.moved += 1
        else:
            result.unchanged += 1
    result.created = sorted(set(resolver.by_key) - before)
    session.flush()
    return result


def refresh(
    session: Session, tenant_id: str = "local", now: datetime | None = None, days: int = 30
) -> list[Workload]:
    """Recompute per-workload stats over the trailing window from Request rows."""
    now = now or datetime.now(UTC)
    since = now - timedelta(days=days)
    workloads = session.scalars(select(Workload).where(Workload.tenant_id == tenant_id)).all()
    for wl in workloads:
        rows = session.scalars(
            select(Request).where(Request.workload_id == wl.workload_id, Request.ts >= since)
        ).all()
        mix = {"input": 0, "output": 0, "cached": 0, "reasoning": 0}
        spend = 0.0
        models: Counter[str] = Counter()
        for r in rows:
            u = r.usage or {}
            mix["input"] += int(u.get("input") or 0)
            mix["output"] += int(u.get("output") or 0)
            mix["cached"] += int(u.get("cached_input") or 0)
            mix["reasoning"] += int(u.get("reasoning") or 0)
            spend += float(r.cost or 0.0)
            if r.model_resolved:
                models[r.model_resolved] += 1
        wl.volume_30d = len(rows)
        wl.spend_30d = spend
        wl.token_mix = mix
        denom = mix["cached"] + mix["input"]
        wl.cache_read_ratio = (mix["cached"] / denom) if denom else None
        wl.incumbent_model_id = models.most_common(1)[0][0] if models else None
    session.flush()
    return workloads


def step_kind_counts(session: Session, workload_id: str) -> dict[str, int]:
    rows = session.execute(
        select(Request.step_kind, func.count())
        .where(Request.workload_id == workload_id)
        .group_by(Request.step_kind)
    ).all()
    return {k or "?": n for k, n in rows}


def merge(session: Session, src: Workload, dst: Workload) -> int:
    """Move ``src``'s requests into ``dst`` and route ``src``'s fingerprint key (and anything it
    had absorbed) to ``dst`` from now on. ``src`` is kept, renamed ``merged:<name>``, with
    ``merge_rules = {"merged_into": dst_id}`` and zeroed stats."""
    if src.workload_id == dst.workload_id:
        raise ValueError("cannot merge a workload into itself")
    if (src.merge_rules or {}).get("merged_into"):
        raise ValueError(f"{src.name} was already merged into {src.merge_rules['merged_into']}")
    moved = 0
    for req in session.scalars(select(Request).where(Request.workload_id == src.workload_id)):
        req.workload_id = dst.workload_id
        moved += 1
    absorbed = list((dst.merge_rules or {}).get("absorbed", []))
    absorbed.append(src.fingerprint_key)
    absorbed.extend((src.merge_rules or {}).get("absorbed", []))
    dst.merge_rules = {**(dst.merge_rules or {}), "absorbed": sorted(set(absorbed))}
    if not src.name.startswith("merged:"):
        src.name = f"merged:{src.name}"
    src.merge_rules = {"merged_into": dst.workload_id}
    src.volume_30d, src.spend_30d, src.token_mix, src.cache_read_ratio = 0, 0.0, {}, None
    session.flush()
    return moved
