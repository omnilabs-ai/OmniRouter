# ADR 0008: Replay adapters use the official `openai` and `anthropic` SDKs; no LiteLLM dependency

Date: 2026-10-31. Status: accepted. Supersedes the "LiteLLM SDK" sentence in ADR 0002.

## Context

ADR 0002 said replays would use the LiteLLM SDK (not the proxy) so we would not rebuild
provider adapters. Two things changed the calculus. First, install footprint: LiteLLM pulls
in every provider SDK and a long tail of transitive packages, which matters for a core that
customers run inside their VPC and audit. Second, the 2026 LiteLLM supply-chain and CVE
record, which our own competitive analysis names as part of the pitch for a small, auditable
decision layer. Meanwhile every provider in the catalog is reachable through two official
SDKs: OpenAI's (OpenAI itself, plus Together and Google Gemini through their OpenAI-compatible
endpoints) and Anthropic's.

## Decision

- **No LiteLLM at runtime.** Dependencies are `openai` and `anthropic` only.
- **One thin `ProviderAdapter` protocol**: `complete(ReplayRequest) -> ReplayResponse`.
  Implementations: `OpenAICompatAdapter(provider_name, base_url, api_key_env)` for
  `openai`, `google` (Gemini's OpenAI-compatible endpoint) and `together_ai`, and
  `AnthropicAdapter()` for `anthropic`. A registry maps the catalog `provider` string to a
  factory. `FixtureAdapter` replays recorded responses for tests and offline runs
  (`OMNI_REPLAY_ADAPTER=fixture:<path>`).
- **Keys come only from environment variables named per provider** (`ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `GEMINI_API_KEY`, `TOGETHER_API_KEY`). A missing key is a clear error
  naming the variable. Keys are never stored or logged; `mapped_params` on a result records
  the exact request kwargs minus anything secret.
- **Usage normalisation is the correctness core** and is unit-tested per SDK shape against
  the catalog convention (ADR 0004: `input` = uncached input tokens, `output` excludes
  reasoning):
  - OpenAI: `prompt_tokens` includes cached tokens → `input = prompt_tokens -
    prompt_tokens_details.cached_tokens`, `cached_input = cached_tokens`;
    `completion_tokens` includes reasoning → `output = completion_tokens -
    completion_tokens_details.reasoning_tokens`, `reasoning = reasoning_tokens`.
  - Anthropic: `input_tokens` already excludes cache reads → `input = input_tokens +
    cache_creation_input_tokens` (cache writes are billed as input for cost purposes and also
    recorded in `extras.cache_write`; the write premium is not modelled yet),
    `cached_input = cache_read_input_tokens`; thinking tokens are inside `output_tokens` and
    the API does not break them out, so `reasoning = 0` and they are billed at the output
    rate, which is how Anthropic bills them.
- **Parameter mapping fails loudly.** `effort` → OpenAI `reasoning_effort`, Anthropic
  `output_config.effort` (adaptive thinking is the default on current models; no
  `budget_tokens`). Before any call, the request is checked against the candidate's catalog
  capabilities: effort requested but `reasoning_effort_levels` empty or the level not listed,
  JSON output requested without `json_mode`, tools without `tools` → `UnsupportedParameter`.
  Nothing is dropped silently, and the mapped kwargs are stored on the result.
- **Message translation for Anthropic**: system messages become the `system` parameter,
  OpenAI-style `tool_calls` become `tool_use` blocks, `role: tool` messages become
  `tool_result` blocks inside a user turn, tool schemas become `{name, description,
  input_schema}`, `response_format` json_schema becomes `output_config.format`.

## Consequences

- Two SDKs to track instead of one aggregator; the surface we use is small (one chat/messages
  call per SDK) and pinned in `uv.lock`.
- A provider outside OpenAI-compatible or Anthropic shapes (Bedrock, Vertex-native) needs a
  new adapter; the protocol is the extension point.
- Streaming is not used in replays (we need the final answer and the usage), so time to first
  token is not measured in PR 7; `latency_ms` is wall-clock for the whole call.
- Retries and budgets live in the runner, not the adapters, so the fixture adapter and a
  flaky fake exercise the same code path as production.
