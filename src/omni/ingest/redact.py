"""Redaction before persistence.

Every message, tool argument and response passes through here before it reaches the object
store. Matches are replaced by typed placeholders (``[EMAIL_1]``, ``[CARD_2]``) that are
stable within one request: the same email appearing three times becomes the same
``[EMAIL_1]``, so the redacted text stays coherent for replay and judging.

Built-in detectors, applied in this order (secrets first so their digits are not eaten by
the card or phone rules): JWT, Bearer tokens, API-key shapes, email, card numbers (Luhn
checked), phone numbers, IPv4. Customer patterns come from ``.omni/redaction.yaml``
(``OMNI_REDACTION_FILE`` overrides) as ``patterns: [{name, regex}]`` and run last.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from omni.ingest.schema import NormalizedRequest

DEFAULT_REDACTION_FILE = Path(".omni") / "redaction.yaml"

_BUILTIN: list[tuple[str, re.Pattern[str]]] = [
    # Bearer before JWT: "Bearer <jwt>" is one secret, not a word plus a token.
    ("BEARER", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("JWT", re.compile(r"\bey[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    (
        "API_KEY",
        re.compile(
            r"\b(?:sk-ant-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}"
            r"|ghp_[A-Za-z0-9]{16,}|xox[abprs]-[A-Za-z0-9-]{10,})"
        ),
    ),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    (
        # Three separated digit groups, optionally with a country code. Must not sit inside a
        # longer digit run (a non-Luhn card number is left as is, not turned into a phone).
        "PHONE",
        re.compile(
            r"(?<![\w.-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]\d{3,4}[\s.-]\d{3,4}"
            r"(?![\w.-])(?!\s\d)"
        ),
    ),
    ("IPV4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
]


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def load_custom_patterns(path: Path | str | None = None) -> list[tuple[str, re.Pattern[str]]]:
    p = Path(path or os.environ.get("OMNI_REDACTION_FILE") or DEFAULT_REDACTION_FILE)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out = []
    for item in data.get("patterns", []):
        name = str(item["name"]).upper().replace(" ", "_")
        out.append((name, re.compile(item["regex"])))
    return out


class Redactor:
    """One per request, so placeholder numbering is stable within it."""

    def __init__(self, custom: list[tuple[str, re.Pattern[str]]] | None = None):
        self.patterns = _BUILTIN + (custom or [])
        self.counts: Counter[str] = Counter()
        self._placeholders: dict[tuple[str, str], str] = {}

    def _placeholder(self, kind: str, original: str) -> str:
        key = (kind, original)
        if key not in self._placeholders:
            n = sum(1 for k in self._placeholders if k[0] == kind) + 1
            self._placeholders[key] = f"[{kind}_{n}]"
        self.counts[kind] += 1
        return self._placeholders[key]

    def text(self, s: str) -> str:
        for kind, pattern in self.patterns:

            def repl(m: re.Match[str], kind: str = kind) -> str:
                if kind == "CARD" and not luhn_ok(re.sub(r"\D", "", m.group(0))):
                    return m.group(0)
                return self._placeholder(kind, m.group(0))

            s = pattern.sub(repl, s)
        return s

    def value(self, v: Any) -> Any:
        """Redact strings anywhere inside a JSON-like value."""
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, list):
            return [self.value(x) for x in v]
        if isinstance(v, dict):
            return {k: self.value(x) for k, x in v.items()}
        return v


def redact_request(
    rec: NormalizedRequest, custom: list[tuple[str, re.Pattern[str]]] | None = None
) -> tuple[NormalizedRequest, Counter[str]]:
    """Return a redacted copy of ``rec`` (messages, tool arguments, response) and counts by type."""
    r = Redactor(custom)
    data = rec.model_dump(mode="json")
    for m in data["messages"]:
        m["content"] = r.value(m.get("content"))
        for tc in m.get("tool_calls") or []:
            tc["arguments"] = r.value(tc.get("arguments"))
    if data.get("response"):
        data["response"]["content"] = r.value(data["response"].get("content"))
        for tc in data["response"].get("tool_calls") or []:
            tc["arguments"] = r.value(tc.get("arguments"))
    return NormalizedRequest.model_validate(data), r.counts
