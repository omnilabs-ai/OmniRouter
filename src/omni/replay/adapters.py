"""Provider adapters (ADR 0008): one thin protocol over the official OpenAI and Anthropic SDKs.

The adapter's job is to send exactly what it is given, map parameters explicitly, and return
usage in the catalog convention (ADR 0004): ``input`` = uncached input tokens, ``output``
excludes reasoning. Retries and budgets live in the runner.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
}
BASE_URLS = {
    "openai": None,
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "together_ai": "https://api.together.xyz/v1",
}
# Anthropic requires max_tokens; used when the request had none. 16k is the SDK guidance for
# non-streaming calls; the run budget bounds spend, an unused ceiling costs nothing.
DEFAULT_MAX_TOKENS = 16000


class UnsupportedParameter(ValueError):
    """The candidate cannot express a parameter the request needs; nothing was sent."""


class MissingApiKey(RuntimeError):
    pass


class TransientError(RuntimeError):
    """Raised by adapters for rate limits, timeouts and 5xx; the runner retries these."""


@dataclass
class ReplayRequest:
    model_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens, response_format
    effort: str | None = None
    request_id: str | None = None  # original request; used by FixtureAdapter keys

    @property
    def wants_json(self) -> bool:
        rf = self.params.get("response_format") or {}
        return rf.get("type") in ("json_object", "json_schema")


@dataclass
class ReplayResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    usage: dict[str, int]  # input, output, cached_input, reasoning
    latency_ms: float | None
    provider: str
    raw_model: str | None = None
    provider_region: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)  # e.g. cache_write
    mapped_params: dict[str, Any] = field(default_factory=dict)

    def as_output(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
        }


class ProviderAdapter(Protocol):
    name: str

    def complete(self, req: ReplayRequest) -> ReplayResponse: ...


# ----------------------------------------------------------------------------- capabilities


def check_capabilities(req: ReplayRequest, capabilities: dict[str, Any], model_id: str) -> None:
    """Refuse, before sending, anything the candidate cannot express (never drop silently)."""
    levels = capabilities.get("reasoning_effort_levels") or []
    if req.effort is not None:
        if not levels:
            raise UnsupportedParameter(
                f"{model_id} has no reasoning-effort control; drop '@{req.effort}'"
            )
        if req.effort not in levels:
            raise UnsupportedParameter(f"{model_id} accepts effort {levels}, not {req.effort!r}")
    if req.wants_json and not capabilities.get("json_mode"):
        raise UnsupportedParameter(f"{model_id} does not support JSON output modes")
    if req.tools and not capabilities.get("tools"):
        raise UnsupportedParameter(f"{model_id} does not support tools")


# ----------------------------------------------------------------------------- usage


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_openai_usage(usage: Any) -> tuple[dict[str, int], dict[str, Any]]:
    prompt = int(_get(usage, "prompt_tokens", 0) or 0)
    completion = int(_get(usage, "completion_tokens", 0) or 0)
    ptd = _get(usage, "prompt_tokens_details")
    ctd = _get(usage, "completion_tokens_details")
    cached = int(_get(ptd, "cached_tokens", 0) or 0)
    reasoning = int(_get(ctd, "reasoning_tokens", 0) or 0)
    norm = {
        "input": max(prompt - cached, 0),
        "output": max(completion - reasoning, 0),
        "cached_input": cached,
        "reasoning": reasoning,
    }
    return norm, {"cache_write": int(_get(ptd, "cache_write_tokens", 0) or 0)}


def normalize_anthropic_usage(usage: Any) -> tuple[dict[str, int], dict[str, Any]]:
    cache_write = int(_get(usage, "cache_creation_input_tokens", 0) or 0)
    norm = {
        "input": int(_get(usage, "input_tokens", 0) or 0) + cache_write,
        "output": int(_get(usage, "output_tokens", 0) or 0),
        "cached_input": int(_get(usage, "cache_read_input_tokens", 0) or 0),
        "reasoning": 0,  # thinking tokens are inside output_tokens and billed as output
    }
    return norm, {"cache_write": cache_write}


# ----------------------------------------------------------------------------- param mapping


def _portable(params: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v for k, v in params.items() if k in ("temperature", "max_tokens", "response_format")
    }


def map_openai_params(req: ReplayRequest) -> dict[str, Any]:
    p = _portable(req.params)
    kwargs: dict[str, Any] = {"model": req.model_id, "messages": req.messages}
    if p.get("temperature") is not None:
        kwargs["temperature"] = p["temperature"]
    if p.get("max_tokens") is not None:
        kwargs["max_completion_tokens"] = p["max_tokens"]
    if p.get("response_format") is not None:
        kwargs["response_format"] = p["response_format"]
    if req.tools:
        kwargs["tools"] = req.tools
    if req.effort is not None:
        kwargs["reasoning_effort"] = req.effort
    return kwargs


def _anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else json.dumps(content))
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id"),
                "content": content or "",
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant" and m.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in m["tool_calls"]:
                fn = tc.get("function") or tc
                args = fn.get("arguments")
                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": fn.get("name"),
                        "input": args or {},
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": role, "content": content if content is not None else ""})
    return ("\n".join(system_parts) or None), out


def _anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        fn = t.get("function") if t.get("type") == "function" else t
        out.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters")
                or fn.get("input_schema")
                or {"type": "object"},
            }
        )
    return out


def map_anthropic_params(req: ReplayRequest) -> dict[str, Any]:
    p = _portable(req.params)
    system, messages = _anthropic_messages(req.messages)
    kwargs: dict[str, Any] = {
        "model": req.model_id,
        "messages": messages,
        "max_tokens": int(p.get("max_tokens") or DEFAULT_MAX_TOKENS),
    }
    if system:
        kwargs["system"] = system
    if p.get("temperature") is not None:
        # The Messages API rejects sampling parameters (temperature, top_p, top_k) as of API
        # version 2026-10-21 and the installed SDK no longer accepts them. The request is sent
        # with the model's default sampling; the drop is recorded on the result, not silent.
        kwargs["_dropped"] = {"temperature": p["temperature"]}
    if req.tools:
        kwargs["tools"] = _anthropic_tools(req.tools)
    output_config: dict[str, Any] = {}
    if req.effort is not None:
        output_config["effort"] = req.effort
    rf = p.get("response_format") or {}
    if rf.get("type") == "json_schema":
        schema = (rf.get("json_schema") or {}).get("schema") or rf.get("schema")
        output_config["format"] = {"type": "json_schema", "schema": schema}
    elif rf.get("type") == "json_object":
        raise UnsupportedParameter("Anthropic has no schema-less json_object mode; use json_schema")
    if output_config:
        kwargs["output_config"] = output_config
    return kwargs


def _audit(kwargs: dict[str, Any]) -> dict[str, Any]:
    """What we record on the result: everything except the prompt payload itself."""
    return {k: v for k, v in kwargs.items() if k not in ("messages", "system", "tools")} | {
        "n_messages": len(kwargs.get("messages") or []),
        "n_tools": len(kwargs.get("tools") or []),
    }


# ----------------------------------------------------------------------------- adapters


def _api_key(env: str) -> str:
    key = os.environ.get(env)
    if not key:
        raise MissingApiKey(
            f"set {env} to replay against this provider (keys are read from the environment only)"
        )
    return key


def _classify(exc: Exception) -> Exception:
    """Map SDK exceptions to TransientError when the runner should retry."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if name in (
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
    ) or (isinstance(status, int) and (status == 429 or status >= 500)):
        return TransientError(f"{name}: {exc}")
    return exc


class OpenAICompatAdapter:
    def __init__(
        self, provider_name: str, base_url: str | None, api_key_env: str, timeout_s: float = 120.0
    ):
        self.name = provider_name
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        # Resolve the key at construction so a missing key fails before a run is created,
        # instead of once per sampled request inside the runner's catch-all.
        self._key = _api_key(api_key_env)

    def complete(self, req: ReplayRequest) -> ReplayResponse:
        from openai import OpenAI

        client = OpenAI(
            api_key=self._key,
            base_url=self.base_url,
            timeout=self.timeout_s,
            max_retries=0,
        )
        kwargs = map_openai_params(req)
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - classified for the runner
            raise _classify(e) from e
        latency = (time.perf_counter() - t0) * 1000
        choice = resp.choices[0] if resp.choices else None
        msg = choice.message if choice else None
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (getattr(msg, "tool_calls", None) or [])
        ]
        usage, extras = normalize_openai_usage(resp.usage)
        return ReplayResponse(
            content=getattr(msg, "content", None),
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
            latency_ms=latency,
            provider=self.name,
            raw_model=getattr(resp, "model", None),
            extras=extras,
            mapped_params=_audit(kwargs),
        )


class AnthropicAdapter:
    name = "anthropic"

    def __init__(self, timeout_s: float = 120.0):
        self.timeout_s = timeout_s
        self._key = _api_key(KEY_ENV["anthropic"])  # fail before any run is created

    def complete(self, req: ReplayRequest) -> ReplayResponse:
        from anthropic import Anthropic

        client = Anthropic(api_key=self._key, timeout=self.timeout_s, max_retries=0)
        kwargs = map_anthropic_params(req)
        dropped = kwargs.pop("_dropped", None)
        t0 = time.perf_counter()
        try:
            resp = client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - classified for the runner
            raise _classify(e) from e
        latency = (time.perf_counter() - t0) * 1000
        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {"id": block.id, "name": block.name, "arguments": json.dumps(block.input)}
                )
        usage, extras = normalize_anthropic_usage(resp.usage)
        return ReplayResponse(
            content="".join(text_parts) or None,
            tool_calls=tool_calls,
            finish_reason=resp.stop_reason,
            usage=usage,
            latency_ms=latency,
            provider=self.name,
            raw_model=resp.model,
            provider_region=getattr(resp.usage, "inference_geo", None),
            extras=extras,
            mapped_params=_audit(kwargs) | ({"dropped": dropped} if dropped else {}),
        )


class FixtureAdapter:
    """Replays recorded responses keyed by ``"<model_id>|<request_id>"``. No network."""

    name = "fixture"

    def __init__(self, path: Path | str):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.responses: dict[str, dict[str, Any]] = data.get("responses", data)
        self.calls: list[ReplayRequest] = []

    def complete(self, req: ReplayRequest) -> ReplayResponse:
        self.calls.append(req)
        key = f"{req.model_id}|{req.request_id}"
        rec = self.responses.get(key)
        if rec is None:
            raise LookupError(f"no recorded response for {key!r} in {self.path}")
        if rec.get("error"):
            raise (
                TransientError(rec["error"]) if rec.get("transient") else RuntimeError(rec["error"])
            )
        return ReplayResponse(
            content=rec.get("content"),
            tool_calls=rec.get("tool_calls") or [],
            finish_reason=rec.get("finish_reason"),
            usage={
                k: int(rec.get("usage", {}).get(k, 0))
                for k in ("input", "output", "cached_input", "reasoning")
            },
            latency_ms=rec.get("latency_ms"),
            provider=rec.get("provider", "fixture"),
            raw_model=req.model_id,
            mapped_params={"model": req.model_id, "effort": req.effort, "fixture": True},
        )


def adapter_for(provider: str) -> ProviderAdapter:
    """Registry: catalog provider -> adapter.

    ``OMNI_REPLAY_ADAPTER=fixture:<path>`` overrides every provider (tests, offline runs).
    """
    override = os.environ.get("OMNI_REPLAY_ADAPTER")
    if override and override.startswith("fixture:"):
        return FixtureAdapter(override.split(":", 1)[1])
    if provider == "anthropic":
        return AnthropicAdapter()
    if provider in BASE_URLS:
        return OpenAICompatAdapter(provider, BASE_URLS[provider], KEY_ENV[provider])
    raise UnsupportedParameter(
        f"no adapter for provider {provider!r}; known: anthropic, {', '.join(BASE_URLS)}"
    )
