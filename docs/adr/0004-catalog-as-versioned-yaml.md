# ADR 0004: The model catalog is a versioned YAML file with recorded sources and billing conventions

Date: 2026-10-22. Status: accepted.

## Context

Every savings number the product reports is `cost(candidate) - cost(incumbent)` on the
customer's token mix. If a price is wrong, the recommendation is wrong, and nobody can tell
from the output. Providers also bill differently: cached prompt tokens have their own rate,
reasoning tokens are sometimes priced separately and sometimes as output, some providers
charge per request, and prices change on announced dates. A hosted pricing feed is a day-60+
decision (design doc section 3); until then the catalog has to be reviewable by a human.

## Decision

- **The catalog is a file in the repository**, `src/omni/data/models.yaml`, validated by a
  pydantic schema that rejects unknown keys. Changing a price is a reviewed PR with a diff. It
  lives inside the package so the installed CLI finds it from any working directory;
  `omni catalog sync --file` loads a different file.
- **Every entry records where its numbers came from and when**: `source_url` is the
  provider's public pricing page and `pricing_verified_at` is the date someone checked it.
  A model whose price cannot be verified from a public page is left out rather than guessed.
  Retired models are not seeded; models that disappear from the file are marked
  `deprecated` by `omni catalog sync`, never deleted, so historical rows keep resolving.
- **Units**: USD per 1M tokens for `input`, `output`, `cached_input`, `reasoning`; USD per call
  for `per_request`. Prices are stored and computed as `Decimal`; `cost_of` rounds to 8
  decimals.
- **Billing conventions** encoded in `cost_of`:
  - `input` means *uncached* input tokens. Cached prompt tokens are reported in
    `cached_input` only and billed at the cached rate. A usage with cached tokens against a
    pricing with no cached rate is an error, not a silent fallback.
  - `reasoning` tokens are billed at the `reasoning` rate when the provider prices them
    separately, otherwise at the output rate. They are never also counted in `output`.
  - `per_request` is added once per call.
- **Tiers are not modelled yet.** Several providers charge more above a prompt-size threshold
  or change prices on a date; an entry carries the standard-tier price in force on
  `pricing_verified_at` and says so in `notes`. Tiered pricing is a follow-up with its own
  schema change.
- `pricing_verified_at` is stored inside the `pricing` JSON column as `verified_at`, so no
  migration is needed and the verification date travels with the numbers it vouches for.

## Consequences

- Prices are as fresh as the last review; `omni catalog show` prints `verified_at` so stale
  numbers are visible. A catalog watcher (days 61-90) will turn stale entries into alerts.
- Collectors must normalise provider usage into the four token buckets with the conventions
  above; a collector that reports total input (cached + uncached) in `input` will overstate
  cost. That contract is tested at the collector, not here.
- The JSON mirror in `ModelCatalog.pricing` is a cache of the file, not a second source of
  truth; `sync` is idempotent and reports inserted / updated / unchanged / deprecated.
