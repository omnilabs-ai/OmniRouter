# ADR 0006: Workloads are fingerprints of (masked system prompt, tool set, response format, step kind, app tag)

Date: 2026-10-31. Status: accepted.

## Context

The workload is the unit of recommendation (design doc section 2.2): a swap is safe or not
*for a workload*, and savings are counted per workload. Production traffic does not arrive
labelled. Most applications render one prompt template per feature with per-request
variables (ids, dates, user names, retrieved text), so the raw system prompt is almost never
identical twice, while the thing that matters for model behaviour, the template plus the tool
set plus the output contract, is stable.

## Decision

A request's fingerprint is the 16-hex SHA-256 of the canonical tuple:

1. **Masked system prompt hash.** All `system` messages are concatenated and masked before
   hashing, in this order: `{{...}}`, `{...}` and `<...>` spans; URLs; UUIDs; ISO dates and
   datetimes; clock times; redaction placeholders (`[EMAIL_1]`); hex ids of 8+ chars; quoted
   strings longer than 40 chars; digit runs; then whitespace collapsed. Each becomes a typed
   token (`@VAR@`, `@URL@`, `@NUM@`...), so the masked text keeps its shape.
2. **Tool-schema hash** (`tools_hash` from ingest, stable across key order). `None` when no
   tools were offered.
3. **Response-format kind**: `text`, `json_object`, `json_schema` (the `type` field).
4. **Step kind**, derived: `compaction` when the request followed a context compaction; else
   `chat` when no tools were offered; else `first_turn` when `step_index == 0` or there is no
   prior assistant turn; else `tool_step`. First-turn is checked before tool-step so step 0
   of an agent session is classified as its first turn even when it already calls a tool.
5. **App tag** from the collector (`app_tag`), when the application provides one.

The importer attaches each request to the workload with that key for its tenant, creating
one (auto-named `wl-<key[:8]>`) when missing. Requests with no messages at all stay in
`unassigned`. `omni workloads merge SRC DST` moves requests and records `SRC`'s key (and
any it had absorbed) in `DST.merge_rules.absorbed`, so later requests with `SRC`'s
fingerprint land in `DST`; `SRC` is kept, renamed `merged:<name>`, with
`merge_rules = {"merged_into": DST}` and zeroed stats, so history and ids stay valid.

## Failure modes, accepted for now

- **Prompts that differ only in masked spans merge.** Intended: that is the template.
- **A long, variable-only system prompt collapses to one workload.** A prompt that is mostly
  retrieved text (RAG context in the system message) masks down to a few tokens and every
  such request lands in one workload even when the retrieved content drives behaviour.
  Mitigation later: fingerprint the first N characters only, or let the app tag split.
- **Small template edits split a workload.** Changing one word of the system prompt starts a
  new workload; `omni workloads merge` is the manual fix, and near-duplicate clustering
  (pgvector, design doc section 3) is the automated one, days 31-60.
- **Different tool sets always split**, even when one tool was added. Intended for now,
  because tool availability changes what the model does; merge is the escape hatch.
- **Step kind is a heuristic.** A tools-offered request mid-session whose last message is a
  user turn and whose response has no tool calls is still `tool_step`; `compaction` wins
  over every other kind. These rules are deterministic and versioned with the code; changing
  them requires `omni workloads segment --all`.
- **App tags from different deployments of the same app split workloads** when the tag
  encodes the environment. Tag the app, not the deployment.

## Consequences

- Segmentation runs at import time and is reproducible from stored content, so heuristics
  can change and be re-applied with `omni workloads segment --all`.
- `Request.step_kind` is persisted so stats and strata can break down by step kind without
  re-reading content.
- Workload names are opaque until a human renames them; the list view shows incumbent,
  volume and spend so the top workloads are easy to find and name.
