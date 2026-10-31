# ADR 0007: Samples are stratified, sized from a CI target, immutable and versioned

Date: 2026-10-31. Status: accepted.

## Context

Replaying a workload's whole traffic against every candidate is too expensive, and an
unstratified random sample hides the cases where candidates differ most (long prompts, tool
steps, requests we have outcomes for). The design (section 2.3) asks for stratified samples
of 200–1,000 requests per workload with the CI width deciding, versioned and reusable as a
golden set for drift detection, and taken only after redaction.

## Decision

- **Stratum** = (prompt-length bucket, tool use, outcome present, day type), labelled
  `lt1k|tools|outcome|weekday` and so on. Prompt length is `usage.input + usage.cached_input`
  in buckets ≤1k, ≤8k, ≤32k, >32k, because cost and cache behaviour change across those
  bands. Tool use is `tools_hash is not null`. Outcome present is whether any `Outcome` row
  exists, so labelled requests are never crowded out. **Day type is weekday vs weekend**, not
  ISO week: weekends shift the mix of human and batch traffic in most products, while ISO
  weeks would create one stratum per week and shatter small workloads into singletons. The
  scheme is versioned (`strata-v1`) and copied into every sample's `strata_def` with the per-
  stratum population and sampled counts, so a later change is visible per sample.
- **Size** comes from the win-rate CI the recommendation will report: the worst-case
  (p = 0.5) normal-approximation n = ceil((1.96 / 2h)²) for half-width h, so h = 0.05 gives
  385. It is clamped to [50, 1000] and then to the eligible population. The formula assumes
  independent requests and ignores stratification gains and the finite population, so it is
  conservative. A workload with fewer than 50 eligible requests cannot be sampled without an
  explicit size, which is a deliberate speed bump: a recommendation from 12 requests should
  look like one.
- **Allocation** is proportional to stratum population with largest-remainder rounding and a
  floor of one per non-empty stratum whenever the size allows, so rare strata are represented
  and can be reported separately even if their estimates are wide.
- **Determinism**: selection uses `random.Random(seed)` over request ids sorted by time; the
  seed is stored, drawn from the system RNG when not given, so any sample can be reproduced
  from the same database.
- **Eligibility**: `status == "success"` and `content_ref` present (the request can be
  replayed), within the optional `[since, until)` range.
- **Immutable and versioned**: a sample is never updated; `create_sample` always writes the
  next version for the workload (unique per tenant, workload, version). `is_golden` marks the
  version drift checks compare against; the latest golden wins.
- **`redaction_version`** pins the redaction scheme that produced the stored content the
  sample points at (`REDACTION_VERSION` in `omni.ingest.redact`, now `regex-v1`). Replays and
  judges can refuse or flag a sample whose content predates a redaction change.

## Consequences

- Strata are computed at sampling time from row fields and the outcomes table, not stored on
  requests; the scheme can change without a migration, and old samples still say which
  scheme they used.
- Four dimensions give up to 32 strata; small workloads will have most of them empty and a
  floor of one per non-empty stratum can use a noticeable share of a small sample. The
  per-stratum counts in `strata_def` make that visible.
- The CI formula sizes one paired comparison; comparing k candidates on the same sample does
  not change n but does multiply the number of comparisons, which the estimator (PR 11) must
  account for.
- A sample references requests by id; deleting content objects for retention leaves the
  sample valid but un-replayable, which the replay runner must report rather than skip.
