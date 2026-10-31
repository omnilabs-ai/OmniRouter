"""Workload fingerprints (design doc section 2.2, ADR 0006).

fingerprint = hash(masked system prompt, tool-schema hash, response-format kind,
                   step kind, app tag)

One prompt *template* must map to one workload even though every request fills the template
with different values, so the system prompt is masked before hashing: variable-looking spans
(numbers, dates, ids, URLs, long quoted strings, redaction placeholders, template braces)
become typed tokens. Different tool sets always split.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # ingest imports workloads at runtime; keep this edge type-only
    from omni.ingest.schema import NormalizedRequest


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Applied in order. Tokens use @NAME@ so they cannot collide with the <...> and {...}
# spans masked below.
_MASKS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\{\{.*?\}\}", re.DOTALL), "@VAR@"),
    (re.compile(r"\{[^{}\n]{1,200}\}"), "@VAR@"),
    (re.compile(r"<[^<>\n]{1,200}>"), "@VAR@"),
    (re.compile(r"https?://\S+"), "@URL@"),
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        "@UUID@",
    ),
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
        ),
        "@DATE@",
    ),
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"), "@TIME@"),
    (re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]"), "@REDACTED@"),
    (re.compile(r"(?<![\w-])[0-9a-fA-F]{8,}(?![\w-])"), "@HEX@"),
    (re.compile(r"\"[^\"\n]{41,}\"|'[^'\n]{41,}'"), "@STR@"),
    (re.compile(r"\d+(?:[.,]\d+)*"), "@NUM@"),
    (re.compile(r"\s+"), " "),
]

CHAT, FIRST_TURN, TOOL_STEP, COMPACTION = "chat", "first_turn", "tool_step", "compaction"


def mask_system_prompt(text: str) -> str:
    for pattern, token in _MASKS:
        text = pattern.sub(token, text)
    return text.strip()


def response_format_kind(response_format: dict[str, Any] | None) -> str:
    if not response_format:
        return "text"
    return str(response_format.get("type") or "text")


def system_prompt_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        if m.get("role") != "system":
            continue
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict))
    return "\n".join(parts)


def derive_step_kind(
    messages: list[dict[str, Any]],
    tools: list[Any],
    response: dict[str, Any] | None,
    step_index: int | None,
    compaction_event: bool,
) -> str:
    """See ADR 0006 for the rules and their failure modes."""
    if compaction_event:
        return COMPACTION
    if not tools:
        return CHAT
    has_prior_assistant = any(m.get("role") == "assistant" for m in messages)
    if step_index == 0 or not has_prior_assistant:
        return FIRST_TURN
    # Tools offered mid-session: a tool result, a tool-calling response, or even a plain user
    # turn are all agent traffic. Only `response` is unused here; it is kept in the signature
    # so a finer rule can use it without changing callers.
    del response
    return TOOL_STEP


@dataclass(frozen=True)
class Fingerprint:
    system_prompt_hash: str
    tool_schema_hash: str | None
    response_format: str
    step_kind: str
    app_tag: str | None

    @property
    def key(self) -> str:
        canon = "|".join(
            [
                self.system_prompt_hash,
                self.tool_schema_hash or "-",
                self.response_format,
                self.step_kind,
                self.app_tag or "-",
            ]
        )
        return sha256_hex(canon)[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "system_prompt_hash": self.system_prompt_hash,
            "tool_schema_hash": self.tool_schema_hash,
            "response_format": self.response_format,
            "step_kind": self.step_kind,
            "app_tag": self.app_tag,
            "key": self.key,
        }


def fingerprint_parts(
    messages: list[dict[str, Any]],
    tools: list[Any],
    tools_hash: str | None,
    response_format: dict[str, Any] | None,
    response: dict[str, Any] | None,
    step_index: int | None,
    compaction_event: bool,
    app_tag: str | None,
) -> Fingerprint:
    """Build a fingerprint from stored pieces (used by re-segmentation of existing rows)."""
    return Fingerprint(
        system_prompt_hash=sha256_hex(mask_system_prompt(system_prompt_text(messages)))[:16],
        tool_schema_hash=tools_hash,
        response_format=response_format_kind(response_format),
        step_kind=derive_step_kind(messages, tools, response, step_index, compaction_event),
        app_tag=app_tag,
    )


def fingerprint(rec: NormalizedRequest, app_tag: str | None = None) -> Fingerprint:
    return fingerprint_parts(
        messages=[m.model_dump(mode="json", exclude_none=True) for m in rec.messages],
        tools=rec.tools,
        tools_hash=rec.tools_hash,
        response_format=rec.params.response_format,
        response=rec.response.model_dump(mode="json") if rec.response else None,
        step_index=rec.step_index,
        compaction_event=rec.compaction_event,
        app_tag=app_tag if app_tag is not None else rec.app_tag,
    )


def auto_name(fp: Fingerprint) -> str:
    return f"wl-{fp.key[:8]}"
