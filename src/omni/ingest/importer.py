"""JSONL importer: validate -> redact -> store by reference -> upsert ``Request`` rows.

Idempotent on ``request_id``. Invalid lines are collected with their line number and never
abort the run. Until segmentation lands (PR 5) every request is attached to a placeholder
workload named ``unassigned``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from omni.catalog import CatalogError, cost_of, load
from omni.ingest.redact import load_custom_patterns, redact_request
from omni.ingest.schema import NormalizedRequest
from omni.ingest.store import ObjectStore
from omni.models import ModelCatalog, Request, Workload

UNASSIGNED = "unassigned"


@dataclass
class ImportResult:
    inserted: int = 0
    skipped: int = 0  # same request_id, same content
    conflicts: int = 0  # same request_id, different content; not applied
    invalid: list[tuple[int, str]] = field(default_factory=list)
    redactions: Counter[str] = field(default_factory=Counter)
    costed: int = 0  # cost filled in from the catalog
    dry_run: bool = False

    @property
    def total(self) -> int:
        return self.inserted + self.skipped + self.conflicts + len(self.invalid)


def iter_lines(path: Path | str) -> Iterator[tuple[int, str]]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip() and not line.lstrip().startswith("#"):
                yield lineno, line


def parse_line(line: str, source: str | None = None) -> NormalizedRequest:
    obj = json.loads(line)
    if source is not None and isinstance(obj, dict):
        obj["source"] = source
    return NormalizedRequest.model_validate(obj)


def _pricing_index(session: Session) -> dict[str, dict[str, Any]]:
    """model_id -> pricing JSON, from model_catalog, else from the packaged catalog file."""
    rows = session.scalars(select(ModelCatalog)).all()
    if rows:
        return {r.model_id: r.pricing for r in rows}
    return {e.model_id: e.pricing.model_dump() for e in load()}


def _unassigned_workload(session: Session, tenant_id: str) -> Workload:
    wl = session.scalar(
        select(Workload).where(
            Workload.tenant_id == tenant_id, Workload.fingerprint_key == UNASSIGNED
        )
    )
    if wl is None:
        wl = Workload(
            tenant_id=tenant_id,
            name=UNASSIGNED,
            fingerprint={"kind": UNASSIGNED},
            fingerprint_key=UNASSIGNED,
        )
        session.add(wl)
        session.flush()
    return wl


def import_file(
    session: Session,
    path: Path | str,
    store: ObjectStore | None = None,
    source: str | None = None,
    dry_run: bool = False,
    tenant_id: str = "local",
) -> ImportResult:
    store = store or ObjectStore()
    result = ImportResult(dry_run=dry_run)
    custom = load_custom_patterns()
    pricing = _pricing_index(session)
    workload = None if dry_run else _unassigned_workload(session, tenant_id)

    for lineno, line in iter_lines(path):
        try:
            rec = parse_line(line, source)
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
            result.invalid.append((lineno, msg[:200]))
            continue
        except (json.JSONDecodeError, TypeError) as e:
            result.invalid.append((lineno, str(e).splitlines()[0][:200]))
            continue

        existing = session.get(Request, rec.request_id)
        if existing is not None:
            if existing.raw_hash == rec.raw_hash:
                result.skipped += 1
            else:
                result.conflicts += 1
            continue

        redacted, counts = redact_request(rec, custom)
        result.redactions.update(counts)

        cost = rec.cost
        if cost is None and rec.model in pricing:
            try:
                cost = cost_of(rec.usage.model_dump(), pricing[rec.model])
                result.costed += 1
            except CatalogError:
                cost = None

        if dry_run:
            result.inserted += 1
            continue

        content_ref = store.put(redacted.content_payload())
        session.add(
            Request(
                request_id=rec.request_id,
                tenant_id=tenant_id,
                workload_id=workload.workload_id,
                ts=rec.ts,
                source=rec.source,
                model_requested=rec.model_requested,
                model_resolved=rec.model_resolved,
                provider=rec.provider,
                region=rec.region,
                params=rec.params.model_dump(mode="json", exclude_none=True),
                tools_hash=rec.tools_hash,
                content_ref=content_ref,
                raw_hash=rec.raw_hash,
                usage=rec.usage.model_dump(),
                cost=None if cost is None else float(cost),
                latency=rec.latency.model_dump(exclude_none=True),
                status=rec.status,
                fallback_chain=rec.fallback_chain,
                session_id=rec.session_id,
                step_index=rec.step_index,
                compaction_event=rec.compaction_event,
            )
        )
        session.flush()
        result.inserted += 1
    return result
