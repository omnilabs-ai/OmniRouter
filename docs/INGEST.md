# Ingest: the JSONL line format

`omni ingest --from FILE.jsonl` reads one JSON object per line, each a `NormalizedRequest`.
Collectors (LiteLLM logger, OTLP receiver, sidecar) produce the same shape. Unknown keys are
rejected so typos are caught at import time. Lines that fail validation are reported with
their line number and skipped; the import continues.

## Fields

| Field | Type | Notes |
|---|---|---|
| `request_id` | string, required | Unique per request; primary key. Re-importing the same id is a no-op. |
| `ts` | ISO-8601 datetime, required | Naive timestamps are taken as UTC. |
| `source` | `litellm` \| `otel` \| `sidecar` \| `import` | Default `import`; `--source` overrides. |
| `model_requested`, `model_resolved` | string | Provider-native ids as in the catalog. Cost uses `model_resolved`, else `model_requested`. |
| `provider`, `region` | string | Optional. |
| `params` | object | `temperature`, `effort`, `max_tokens`, `response_format`. |
| `messages` | list | OpenAI-style `{role, content, tool_calls?, tool_call_id?}`; `role` in system/user/assistant/tool. The prompt as sent, without the response. |
| `tools` | list | Tool schemas as sent. `tools_hash` is computed from them (stable across key order). |
| `response` | object | `{content, tool_calls, finish_reason}`: the incumbent's answer. |
| `usage` | object | `input` (uncached input tokens), `output` (excluding reasoning), `cached_input`, `reasoning`. |
| `cost` | number | USD. If absent and the model is in the catalog, computed with `cost_of`. |
| `latency` | object | `ttft_ms`, `total_ms`. |
| `status` | `success` \| `error` \| `timeout` \| `refused` | Default `success`. |
| `fallback_chain` | list of strings | Models tried, in order. |
| `session_id`, `step_index` | string, int | Agent sessions: step k of the session. |
| `compaction_event` | bool | True when this step followed a context compaction. |

Everything in `messages`, `tools` arguments and `response` is redacted before it is stored
(ADR 0005); the database row keeps only metadata and a `content_ref` key.

## Example line

```json
{"request_id":"req_0007","ts":"2026-10-30T14:02:11Z","source":"sidecar","model_requested":"claude-sonnet-5-5","model_resolved":"claude-sonnet-5-5","provider":"anthropic","params":{"temperature":0,"max_tokens":2048,"effort":"high"},"messages":[{"role":"system","content":"You are a coding agent in repo acme/webapp. One tool call per step."},{"role":"user","content":"Task: fix the off-by-one in pagination."},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","name":"read_file","arguments":"{\"path\": \"src/api/pagination.py\"}"}]},{"role":"tool","tool_call_id":"call_1","content":"...limit + 1..."}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},{"type":"function","function":{"name":"edit_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"]}}}],"response":{"content":null,"tool_calls":[{"id":"call_2","name":"edit_file","arguments":"{\"path\": \"src/api/pagination.py\", \"old\": \"limit + 1\", \"new\": \"limit\"}"}],"finish_reason":"tool_calls"},"usage":{"input":1340,"output":62,"cached_input":2200,"reasoning":180},"latency":{"ttft_ms":410,"total_ms":2210},"status":"success","session_id":"sess-001","step_index":1,"compaction_event":false}
```

## Running an import

```bash
uv run omni db upgrade
uv run omni catalog sync                       # so costs can be computed
uv run omni ingest --from traffic.jsonl        # add --dry-run to validate and redact without writing
```

Objects land in `.omni/objects/` (override with `OMNI_OBJECT_DIR`); custom redaction patterns
go in `.omni/redaction.yaml` as `patterns: [{name: TICKET, regex: "TKT-\\d{6}"}]`.
