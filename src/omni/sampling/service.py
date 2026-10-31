"""Create and look up samples (ADR 0007)."""

from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from omni.ingest.redact import REDACTION_VERSION
from omni.models import Outcome, Request, Sample, Workload
from omni.sampling.size import target_size
from omni.sampling.strata import STRATA_DEF, stratum_of

DEFAULT_FLOOR = 50
DEFAULT_CAP = 1000


class SamplingError(ValueError):
    pass


def eligible_requests(
    session: Session,
    workload: Workload,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[Request]:
    """Successful requests with stored content, in the range, oldest first."""
    q = select(Request).where(
        Request.workload_id == workload.workload_id,
        Request.status == "success",
        Request.content_ref.is_not(None),
    )
    if since is not None:
        q = q.where(Request.ts >= since)
    if until is not None:
        q = q.where(Request.ts < until)
    return list(session.scalars(q.order_by(Request.ts, Request.request_id)))


def allocate(counts: dict[str, int], size: int) -> dict[str, int]:
    """Proportional allocation with largest-remainder rounding. When ``size`` covers every
    non-empty stratum, each gets at least one; no stratum gets more than it has."""
    total = sum(counts.values())
    if size >= total:
        return dict(counts)
    strata = sorted(counts)
    exact = {s: size * counts[s] / total for s in strata}
    alloc = {s: int(exact[s]) for s in strata}
    if size >= len(strata):
        for s in strata:
            alloc[s] = max(alloc[s], 1)
    # trim overshoot from the strata with the smallest fractional parts, then hand out the
    # remainder to the largest fractional parts; never exceed a stratum's population
    while sum(alloc.values()) > size:
        s = min(
            (s for s in strata if alloc[s] > (1 if size >= len(strata) else 0)),
            key=lambda s: (exact[s] - int(exact[s]), -counts[s]),
        )
        alloc[s] -= 1
    for s in sorted(strata, key=lambda s: (-(exact[s] - int(exact[s])), -counts[s])):
        if sum(alloc.values()) >= size:
            break
        if alloc[s] < counts[s]:
            alloc[s] += 1
    return alloc


def create_sample(
    session: Session,
    workload: Workload,
    size: int | None = None,
    ci_half_width: float = 0.05,
    golden: bool = False,
    since: datetime | None = None,
    until: datetime | None = None,
    seed: int | None = None,
    floor: int = DEFAULT_FLOOR,
    cap: int = DEFAULT_CAP,
) -> Sample:
    reqs = eligible_requests(session, workload, since, until)
    available = len(reqs)
    if available == 0:
        raise SamplingError(f"workload {workload.name!r} has no eligible requests in the range")
    if size is None:
        if available < floor:
            raise SamplingError(
                f"workload {workload.name!r} has {available} eligible requests, fewer than the "
                f"floor of {floor}; pass an explicit --size to sample anyway"
            )
        size = target_size(available, ci_half_width, floor=floor, cap=cap)
    elif size < 1 or size > available:
        raise SamplingError(f"size must be between 1 and {available} (eligible requests)")

    with_outcome = set(
        session.scalars(
            select(Outcome.request_id).where(Outcome.request_id.in_([r.request_id for r in reqs]))
        )
    )
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for r in reqs:
        by_stratum[stratum_of(r, r.request_id in with_outcome)].append(r.request_id)
    alloc = allocate({s: len(ids) for s, ids in by_stratum.items()}, size)

    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    rng = random.Random(seed)
    chosen: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for s in sorted(by_stratum):
        ids = by_stratum[s]
        k = alloc.get(s, 0)
        picked = sorted(rng.sample(ids, k)) if k < len(ids) else list(ids)
        chosen.extend(picked)
        counts[s] = {"population": len(ids), "sampled": len(picked)}

    version = (
        session.scalar(
            select(Sample.version)
            .where(
                Sample.tenant_id == workload.tenant_id, Sample.workload_id == workload.workload_id
            )
            .order_by(Sample.version.desc())
            .limit(1)
        )
        or 0
    ) + 1
    sample = Sample(
        tenant_id=workload.tenant_id,
        workload_id=workload.workload_id,
        version=version,
        strata_def={**STRATA_DEF, "counts": counts, "ci_half_width": ci_half_width},
        request_ids=chosen,
        size=len(chosen),
        is_golden=golden,
        created_from_range={
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
            "first_ts": reqs[0].ts.isoformat(),
            "last_ts": reqs[-1].ts.isoformat(),
            "eligible": available,
        },
        seed=seed,
        redaction_version=REDACTION_VERSION,
    )
    session.add(sample)
    session.flush()
    return sample


def list_samples(session: Session, workload: Workload | None = None) -> list[Sample]:
    q = select(Sample)
    if workload is not None:
        q = q.where(Sample.workload_id == workload.workload_id)
    return list(session.scalars(q.order_by(Sample.workload_id, Sample.version)))


def golden(session: Session, workload: Workload) -> Sample | None:
    return session.scalar(
        select(Sample)
        .where(Sample.workload_id == workload.workload_id, Sample.is_golden.is_(True))
        .order_by(Sample.version.desc())
        .limit(1)
    )
