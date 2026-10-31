"""Adapters: usage normalisation per SDK shape, parameter mapping, capability checks, registry."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from omni.replay import (
    FixtureAdapter,
    MissingApiKey,
    OpenAICompatAdapter,
    ReplayRequest,
    TransientError,
    UnsupportedParameter,
    adapter_for,
)
from omni.replay.adapters import (
    _classify,
    check_capabilities,
    map_anthropic_params,
    map_openai_params,
    normalize_anthropic_usage,
    normalize_openai_usage,
)

FULL = {"tools": True, "json_mode": True, "reasoning_effort_levels": ["low", "high"]}
TOOLS = [
    {
        "type": "function",
        "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
    }
]


# ----------------------------------------------------------------------------- usage


def test_openai_usage_subtracts_cached_and_reasoning():
    usage = NS(
        prompt_tokens=1500,
        completion_tokens=700,
        prompt_tokens_details=NS(cached_tokens=900, cache_write_tokens=0),
        completion_tokens_details=NS(reasoning_tokens=500),
    )
    norm, extras = normalize_openai_usage(usage)
    assert norm == {"input": 600, "output": 200, "cached_input": 900, "reasoning": 500}
    assert extras == {"cache_write": 0}
    # details absent (Together, Gemini-compatible): everything is uncached, no reasoning
    norm, _ = normalize_openai_usage({"prompt_tokens": 100, "completion_tokens": 10})
    assert norm == {"input": 100, "output": 10, "cached_input": 0, "reasoning": 0}


def test_anthropic_usage_adds_cache_writes_to_input_and_keeps_reads_separate():
    usage = NS(
        input_tokens=400,
        output_tokens=120,
        cache_read_input_tokens=2200,
        cache_creation_input_tokens=300,
    )
    norm, extras = normalize_anthropic_usage(usage)
    assert norm == {"input": 700, "output": 120, "cached_input": 2200, "reasoning": 0}
    assert extras == {"cache_write": 300}
    norm, _ = normalize_anthropic_usage(NS(input_tokens=50, output_tokens=5))
    assert norm == {"input": 50, "output": 5, "cached_input": 0, "reasoning": 0}


# ----------------------------------------------------------------------------- capabilities


def test_check_capabilities_fails_loudly():
    base = ReplayRequest(model_id="m", messages=[])
    check_capabilities(base, {}, "m")  # nothing requested, nothing to refuse
    with pytest.raises(UnsupportedParameter, match="no reasoning-effort"):
        check_capabilities(
            ReplayRequest("m", [], effort="low"), {"reasoning_effort_levels": []}, "m"
        )
    with pytest.raises(UnsupportedParameter, match="accepts effort"):
        check_capabilities(ReplayRequest("m", [], effort="max"), FULL, "m")
    with pytest.raises(UnsupportedParameter, match="JSON"):
        check_capabilities(
            ReplayRequest("m", [], params={"response_format": {"type": "json_object"}}),
            {"json_mode": False},
            "m",
        )
    with pytest.raises(UnsupportedParameter, match="tools"):
        check_capabilities(ReplayRequest("m", [], tools=TOOLS), {"tools": False}, "m")
    check_capabilities(
        ReplayRequest(
            "m", [], tools=TOOLS, effort="high", params={"response_format": {"type": "json_object"}}
        ),
        FULL,
        "m",
    )


# ----------------------------------------------------------------------------- mapping


def test_openai_params_map_effort_and_max_tokens():
    req = ReplayRequest(
        "gpt-x",
        [{"role": "user", "content": "hi"}],
        tools=TOOLS,
        params={
            "temperature": 0,
            "max_tokens": 50,
            "response_format": {"type": "json_object"},
            "ignored": 1,
        },
        effort="low",
    )
    kw = map_openai_params(req)
    assert kw["model"] == "gpt-x" and kw["reasoning_effort"] == "low"
    assert kw["max_completion_tokens"] == 50 and "max_tokens" not in kw and "ignored" not in kw
    assert kw["response_format"] == {"type": "json_object"} and kw["tools"] == TOOLS
    assert "reasoning_effort" not in map_openai_params(ReplayRequest("gpt-x", []))


def test_anthropic_params_translate_messages_tools_effort_and_schema():
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "name": "f", "arguments": '{"path": "a"}'}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "user", "content": "thanks"},
    ]
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    req = ReplayRequest(
        "claude-x",
        messages,
        tools=TOOLS,
        params={
            "max_tokens": 300,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "s", "schema": schema},
            },
        },
        effort="high",
    )
    kw = map_anthropic_params(req)
    assert kw["system"] == "be terse" and kw["max_tokens"] == 300
    assert kw["messages"][0] == {"role": "user", "content": "fix it"}
    assert kw["messages"][1]["content"] == [
        {"type": "tool_use", "id": "c1", "name": "f", "input": {"path": "a"}}
    ]
    assert kw["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}],
    }
    assert kw["tools"] == [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]
    assert kw["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": schema},
    }
    assert "thinking" not in kw and "budget_tokens" not in str(kw)
    plain = map_anthropic_params(ReplayRequest("claude-x", [{"role": "user", "content": "hi"}]))
    assert plain["max_tokens"] == 4096 and "output_config" not in plain and "system" not in plain
    with pytest.raises(UnsupportedParameter, match="json_object"):
        map_anthropic_params(
            ReplayRequest("claude-x", [], params={"response_format": {"type": "json_object"}})
        )


def test_classify_marks_rate_limit_and_5xx_transient():
    class RateLimitError(Exception):
        status_code = 429

    class BadRequestError(Exception):
        status_code = 400

    class ServerBoom(Exception):
        status_code = 503

    assert isinstance(_classify(RateLimitError("slow down")), TransientError)
    assert isinstance(_classify(ServerBoom("boom")), TransientError)
    bad = BadRequestError("nope")
    assert _classify(bad) is bad


# ----------------------------------------------------------------------------- registry / keys


def test_registry_and_fixture_override(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNI_REPLAY_ADAPTER", raising=False)
    assert adapter_for("anthropic").name == "anthropic"
    a = adapter_for("google")
    assert (
        isinstance(a, OpenAICompatAdapter)
        and a.api_key_env == "GEMINI_API_KEY"
        and "googleapis" in a.base_url
    )
    assert adapter_for("together_ai").api_key_env == "TOGETHER_API_KEY"
    with pytest.raises(UnsupportedParameter, match="no adapter"):
        adapter_for("mystery")
    p = tmp_path / "fx.json"
    p.write_text(
        '{"responses": {"m|r1": {"content": "hi", "usage": {"input": 3, "output": 1}}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("OMNI_REPLAY_ADAPTER", f"fixture:{p}")
    fx = adapter_for("openai")
    assert isinstance(fx, FixtureAdapter)
    resp = fx.complete(ReplayRequest("m", [], request_id="r1"))
    assert resp.content == "hi" and resp.usage == {
        "input": 3,
        "output": 1,
        "cached_input": 0,
        "reasoning": 0,
    }
    with pytest.raises(LookupError):
        fx.complete(ReplayRequest("m", [], request_id="r2"))


def test_missing_api_key_is_a_clear_error_before_any_call(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    adapter = OpenAICompatAdapter("together_ai", "https://api.together.xyz/v1", "TOGETHER_API_KEY")
    with pytest.raises(MissingApiKey, match="TOGETHER_API_KEY"):
        adapter.complete(ReplayRequest("Qwen/x", [{"role": "user", "content": "hi"}]))
