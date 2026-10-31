"""Core domain models (design doc section 4): ModelCatalog, Workload, Request, Outcome.

Every table carries ``tenant_id`` and ``created_at``. Nested structures from the design doc
(pricing, capabilities, fingerprint, constraints, usage, latency, params, ...) are JSON
columns rather than child tables: they are read and written whole, and their shape is still
settling (ADR 0003). Later PRs add Sample, CandidatePool, ReplayRun, ReplayResult, Scorer,
Score, Recommendation, Policy, Receipt and DriftCheck next to these.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class TenantMixin:
    tenant_id: Mapped[str] = mapped_column(String(64), default="local", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelCatalog(TenantMixin, Base):
    """A model as the catalog knows it: identity, limits, capabilities, pricing."""

    __tablename__ = "model_catalog"

    model_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    dated_revision: Mapped[str | None] = mapped_column(String(64))
    display_name: Mapped[str] = mapped_column(String(160))
    context_window: Mapped[int | None] = mapped_column(Integer)
    max_output: Mapped[int | None] = mapped_column(Integer)
    # {tools, vision, json_mode, reasoning_effort_levels}
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # {input, output, cached_input, reasoning, per_request} in USD per 1M tokens
    pricing: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    regions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    residency_class: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | deprecated
    release_date: Mapped[str | None] = mapped_column(String(16))
    source_url: Mapped[str | None] = mapped_column(Text)


class Workload(TenantMixin, Base):
    """The unit of recommendation: one prompt template x tool set x response format."""

    __tablename__ = "workloads"

    workload_id: Mapped[str] = mapped_column(
        String(40), primary_key=True, default=lambda: new_id("wl")
    )
    name: Mapped[str] = mapped_column(String(160), index=True)
    # {system_prompt_hash, tool_schema_hash, response_format, step_kind, app_tag}
    fingerprint: Mapped[dict[str, Any]] = mapped_column(JSON)
    fingerprint_key: Mapped[str] = mapped_column(String(80))
    merge_rules: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    incumbent_model_id: Mapped[str | None] = mapped_column(String(160))
    # {scorer_id, min_score | min_winrate}
    quality_bar: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # {allowed_providers, regions, max_latency_ms, max_cost_per_1k_req, required_capabilities}
    constraints: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    volume_30d: Mapped[int] = mapped_column(Integer, default=0)
    spend_30d: Mapped[float] = mapped_column(Float, default=0.0)
    # {input, output, cached, reasoning} token totals
    token_mix: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    cache_read_ratio: Mapped[float | None] = mapped_column(Float)

    requests: Mapped[list[Request]] = relationship(back_populates="workload")

    __table_args__ = (
        Index("ix_workloads_tenant_fingerprint", "tenant_id", "fingerprint_key", unique=True),
    )


class Request(TenantMixin, Base):
    """One production LLM call. Content lives behind ``content_ref``, not in this row."""

    __tablename__ = "requests"

    request_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    workload_id: Mapped[str] = mapped_column(ForeignKey("workloads.workload_id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String(16))  # litellm | otel | sidecar | import
    model_requested: Mapped[str | None] = mapped_column(String(160))
    model_resolved: Mapped[str | None] = mapped_column(String(160), index=True)
    provider: Mapped[str | None] = mapped_column(String(64))
    region: Mapped[str | None] = mapped_column(String(32))
    # {temperature, effort, max_tokens, response_format}
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tools_hash: Mapped[str | None] = mapped_column(String(80))
    content_ref: Mapped[str | None] = mapped_column(String(160))  # object store key, redacted
    raw_hash: Mapped[str | None] = mapped_column(String(64))  # sha256 of the unredacted record
    # {input, output, cached_input, reasoning}
    usage: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    cost: Mapped[float | None] = mapped_column(Float)
    # {ttft_ms, total_ms}
    latency: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="success")
    fallback_chain: Mapped[list[Any]] = mapped_column(JSON, default=list)
    session_id: Mapped[str | None] = mapped_column(String(120), index=True)
    step_index: Mapped[int | None] = mapped_column(Integer)
    step_kind: Mapped[str | None] = mapped_column(
        String(16)
    )  # chat|first_turn|tool_step|compaction
    compaction_event: Mapped[bool] = mapped_column(Boolean, default=False)
    receipt_id: Mapped[str | None] = mapped_column(String(40))

    workload: Mapped[Workload] = relationship(back_populates="requests")
    outcomes: Mapped[list[Outcome]] = relationship(back_populates="request")


class Outcome(TenantMixin, Base):
    """A quality signal attached to a request: explicit (thumb, label) or passive (retry)."""

    __tablename__ = "outcomes"

    outcome_id: Mapped[str] = mapped_column(
        String(40), primary_key=True, default=lambda: new_id("oc")
    )
    request_id: Mapped[str] = mapped_column(ForeignKey("requests.request_id"), index=True)
    # thumb | retry | regenerate | edit_distance | tool_success | tool_error | test_pass |
    # task_complete | human_label | custom
    signal: Mapped[str] = mapped_column(String(32), index=True)
    value: Mapped[Any] = mapped_column(JSON)  # number, string or bool
    source: Mapped[str] = mapped_column(String(16), default="sdk")  # sdk | passive | console
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    request: Mapped[Request] = relationship(back_populates="outcomes")
