"""Stratum assignment (ADR 0007).

A request's stratum is the tuple (prompt-length bucket, tool use, outcome present, day type).
The scheme is versioned so a later change is visible in every sample's ``strata_def``.
"""

from __future__ import annotations

from typing import Any

from omni.models import Request

STRATA_VERSION = "strata-v1"

# Upper bounds (inclusive) of the prompt-length buckets, in prompt tokens = input + cached.
LENGTH_BUCKETS: list[tuple[str, int]] = [("lt1k", 1_000), ("1k-8k", 8_000), ("8k-32k", 32_000)]
LENGTH_OVERFLOW = "gt32k"

STRATA_DEF: dict[str, Any] = {
    "version": STRATA_VERSION,
    "dims": [
        {
            "name": "prompt_length",
            "from": "usage.input + usage.cached_input",
            "buckets": [f"{name} (<= {bound})" for name, bound in LENGTH_BUCKETS]
            + [f"{LENGTH_OVERFLOW} (> 32000)"],
        },
        {"name": "tool_use", "from": "tools_hash is not null", "buckets": ["tools", "notools"]},
        {"name": "outcome_present", "from": "any Outcome row", "buckets": ["outcome", "nooutcome"]},
        {"name": "day_type", "from": "ts weekday", "buckets": ["weekday", "weekend"]},
    ],
}


def length_bucket(prompt_tokens: int) -> str:
    for name, bound in LENGTH_BUCKETS:
        if prompt_tokens <= bound:
            return name
    return LENGTH_OVERFLOW


def stratum_of(req: Request, has_outcome: bool) -> str:
    u = req.usage or {}
    prompt_tokens = int(u.get("input") or 0) + int(u.get("cached_input") or 0)
    return "|".join(
        [
            length_bucket(prompt_tokens),
            "tools" if req.tools_hash else "notools",
            "outcome" if has_outcome else "nooutcome",
            "weekend" if req.ts.weekday() >= 5 else "weekday",
        ]
    )
