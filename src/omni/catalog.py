"""Model catalog: a versioned YAML file of models and prices, mirrored into ``ModelCatalog``.

Conventions (ADR 0004):

* Prices are USD per 1M tokens; ``per_request`` is USD per call.
* ``usage["input"]`` means *uncached* input tokens. Cached input tokens are counted in
  ``usage["cached_input"]`` only and billed at ``cached_input``; they are never also billed
  as input.
* Reasoning tokens are billed at ``reasoning`` when the model prices them separately, else
  at the output rate. They are never also counted in ``output``.
* Models missing from the file are marked ``deprecated`` on sync, never deleted, so old
  requests and recommendations keep resolving.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from omni.models import ModelCatalog

DEFAULT_CATALOG_PATH = Path("catalog") / "models.yaml"
MILLION = Decimal(1_000_000)
USAGE_KEYS = ("input", "output", "cached_input", "reasoning")


class CatalogError(ValueError):
    """Invalid catalog file or pricing data."""


class Pricing(BaseModel):
    """USD per 1M tokens. ``None`` means the provider has no separate rate (see cost_of)."""

    model_config = ConfigDict(extra="forbid")

    input: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)
    cached_input: Decimal | None = Field(default=None, ge=0)
    reasoning: Decimal | None = Field(default=None, ge=0)
    per_request: Decimal = Field(default=Decimal(0), ge=0)


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: bool = False
    vision: bool = False
    json_mode: bool = False
    reasoning_effort_levels: list[str] = Field(default_factory=list)


class CatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    display_name: str
    dated_revision: str | None = None
    context_window: int | None = Field(default=None, gt=0)
    max_output: int | None = Field(default=None, gt=0)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    pricing: Pricing
    pricing_verified_at: date
    regions: list[str] = Field(default_factory=list)
    residency_class: str | None = None
    status: Literal["active", "deprecated"] = "active"
    release_date: date | None = None
    source_url: str
    notes: str | None = None


class CatalogFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    models: list[CatalogEntry]


# ------------------------------------------------------------------------------ load / sync


def load(path: Path | str = DEFAULT_CATALOG_PATH) -> list[CatalogEntry]:
    """Parse and validate the catalog file. Unknown keys and missing rates are errors."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise CatalogError(f"catalog file not found: {path}") from e
    except yaml.YAMLError as e:
        raise CatalogError(f"{path}: not valid YAML: {e}") from e
    try:
        parsed = CatalogFile.model_validate(raw)
    except ValidationError as e:
        raise CatalogError(f"{path}: {e}") from e
    seen: set[str] = set()
    for entry in parsed.models:
        if entry.model_id in seen:
            raise CatalogError(f"{path}: duplicate model_id {entry.model_id!r}")
        seen.add(entry.model_id)
    return parsed.models


def _pricing_json(entry: CatalogEntry) -> dict[str, Any]:
    p = {k: (None if v is None else float(v)) for k, v in entry.pricing.model_dump().items()}
    p["verified_at"] = entry.pricing_verified_at.isoformat()
    return p


def _row_values(entry: CatalogEntry) -> dict[str, Any]:
    return {
        "provider": entry.provider,
        "dated_revision": entry.dated_revision,
        "display_name": entry.display_name,
        "context_window": entry.context_window,
        "max_output": entry.max_output,
        "capabilities": entry.capabilities.model_dump(),
        "pricing": _pricing_json(entry),
        "regions": list(entry.regions),
        "residency_class": entry.residency_class,
        "status": entry.status,
        "release_date": entry.release_date.isoformat() if entry.release_date else None,
        "source_url": entry.source_url,
    }


class SyncResult(BaseModel):
    inserted: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    deprecated: list[str] = Field(default_factory=list)


def sync(session: Session, entries: list[CatalogEntry], tenant_id: str = "local") -> SyncResult:
    """Upsert ``entries`` into ``model_catalog``; rows absent from ``entries`` become
    ``deprecated``. Nothing is deleted."""
    result = SyncResult()
    wanted = {e.model_id for e in entries}
    rows = {
        r.model_id: r
        for r in session.scalars(select(ModelCatalog).where(ModelCatalog.tenant_id == tenant_id))
    }
    for entry in entries:
        values = _row_values(entry)
        row = rows.get(entry.model_id)
        if row is None:
            session.add(ModelCatalog(model_id=entry.model_id, tenant_id=tenant_id, **values))
            result.inserted.append(entry.model_id)
        elif any(getattr(row, k) != v for k, v in values.items()):
            for k, v in values.items():
                setattr(row, k, v)
            result.updated.append(entry.model_id)
        else:
            result.unchanged.append(entry.model_id)
    for model_id, row in rows.items():
        if model_id not in wanted and row.status != "deprecated":
            row.status = "deprecated"
            result.deprecated.append(model_id)
    session.flush()
    return result


# ------------------------------------------------------------------------------ cost


def cost_of(usage: dict[str, int], pricing: Pricing | dict[str, Any]) -> Decimal:
    """USD cost of one call, rounded to 8 decimals.

    ``usage`` keys: ``input`` (uncached input tokens), ``output``, ``cached_input``,
    ``reasoning``; missing keys count as 0, unknown keys raise.
    """
    unknown = set(usage) - set(USAGE_KEYS)
    if unknown:
        raise CatalogError(f"unknown usage keys: {sorted(unknown)}; expected {USAGE_KEYS}")
    if not isinstance(pricing, Pricing):
        try:
            pricing = Pricing.model_validate(
                {k: v for k, v in pricing.items() if k in Pricing.model_fields}
            )
        except ValidationError as e:
            raise CatalogError(f"invalid pricing: {e}") from e
    tokens = {k: int(usage.get(k) or 0) for k in USAGE_KEYS}
    if any(v < 0 for v in tokens.values()):
        raise CatalogError("token counts must be non-negative")
    if tokens["cached_input"] and pricing.cached_input is None:
        raise CatalogError("usage has cached_input tokens but pricing has no cached_input rate")
    reasoning_rate = pricing.output if pricing.reasoning is None else pricing.reasoning
    total = (
        Decimal(tokens["input"]) * pricing.input
        + Decimal(tokens["cached_input"]) * (pricing.cached_input or 0)
        + Decimal(tokens["output"]) * pricing.output
        + Decimal(tokens["reasoning"]) * reasoning_rate
    ) / MILLION + pricing.per_request
    return total.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
