"""``NormalizedRequest``: the one shape every collector produces and the JSONL line format.

Field reference and an example line are in docs/INGEST.md. Token usage follows the catalog
convention (ADR 0004): ``usage.input`` is *uncached* input tokens.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Source = Literal["litellm", "otel", "sidecar", "import"]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    effort: str | None = None
    max_tokens: int | None = Field(default=None, ge=0)
    response_format: dict[str, Any] | None = None


class Usage(BaseModel):
    """Token counts. ``input`` excludes ``cached_input``; ``output`` excludes ``reasoning``."""

    model_config = ConfigDict(extra="forbid")

    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cached_input: int = Field(default=0, ge=0)
    reasoning: int = Field(default=0, ge=0)


class Latency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttft_ms: float | None = Field(default=None, ge=0)
    total_ms: float | None = Field(default=None, ge=0)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str
    arguments: Any = None  # dict, or the provider's JSON string


class Message(BaseModel):
    """OpenAI-style chat message. Provider extras (``tool_call_id``, ``name``) pass through."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Any] | None = None
    tool_calls: list[ToolCall] | None = None


class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None


class NormalizedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=80)
    ts: datetime
    source: Source = "import"
    model_requested: str | None = None
    model_resolved: str | None = None
    provider: str | None = None
    region: str | None = None
    params: Params = Field(default_factory=Params)
    messages: list[Message] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    response: Response | None = None
    usage: Usage = Field(default_factory=Usage)
    cost: Decimal | None = Field(default=None, ge=0)
    latency: Latency = Field(default_factory=Latency)
    status: Literal["success", "error", "timeout", "refused"] = "success"
    fallback_chain: list[str] = Field(default_factory=list)
    session_id: str | None = None
    step_index: int | None = Field(default=None, ge=0)
    compaction_event: bool = False
    app_tag: str | None = Field(default=None, max_length=120)  # app/route label for segmentation

    @field_validator("ts", mode="after")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=UTC)

    @property
    def model(self) -> str | None:
        return self.model_resolved or self.model_requested

    @property
    def tools_hash(self) -> str | None:
        """Stable across key order; None when no tools were offered."""
        if not self.tools:
            return None
        return sha256_hex(canonical_json(self.tools))[:16]

    @property
    def raw_hash(self) -> str:
        """Hash of the full normalised record *before* redaction, for idempotency checks."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))

    def content_payload(self) -> dict[str, Any]:
        """What goes to the object store: everything a replay or a scorer needs to re-create
        the call, and the response to compare against. Never written unredacted."""
        return {
            "request_id": self.request_id,
            "app_tag": self.app_tag,
            "messages": [m.model_dump(mode="json", exclude_none=True) for m in self.messages],
            "tools": self.tools,
            "params": self.params.model_dump(mode="json", exclude_none=True),
            "response": self.response.model_dump(mode="json") if self.response else None,
        }
