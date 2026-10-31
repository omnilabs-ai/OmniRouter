# ADR 0005: Request content is redacted before persistence and stored by reference

Date: 2026-10-31. Status: accepted.

## Context

Production traffic is the benchmark (design principle 1), so the system has to keep enough of
each request to replay it and to judge the replay against the original answer. That content
is also where the customer's PII and secrets live. The design (sections 2.3 and 3) requires
redaction before anything is stored, content retention that can be switched off per tenant,
and a core that runs in the customer's VPC with Postgres plus an object store.

## Decision

- **Redaction happens in the importer, before any write.** Message content, tool-call
  arguments and the response go through `omni.ingest.redact` first. Built-in detectors cover
  secrets (JWTs, bearer tokens, API-key shapes), emails, Luhn-valid card numbers, phone
  numbers and IPv4 addresses; customers add patterns in `.omni/redaction.yaml`. Matches become
  typed placeholders (`[EMAIL_1]`) that are stable within a request so the text stays usable.
  The pre-redaction record is hashed (`raw_hash`) and then discarded; it is never written.
- **Content lives in an object store, the database holds keys.** The redacted request
  (messages, tools, params) and response are one JSON document, keyed by the SHA-256 of its
  canonical form, at `<OMNI_OBJECT_DIR>/<key[:2]>/<key>.json`. `Request.content_ref` is that
  key. The default root is `.omni/objects`; an S3-compatible backend behind the same
  two-method interface (`put`, `get`) is the deployment step, not a schema change.
- **The database row is metadata only**: model, provider, params, usage, cost, latency,
  status, session and step fields, `tools_hash`, `content_ref`, `raw_hash`. Nothing in the
  row can reproduce a prompt.
- **Idempotency**: `request_id` is the primary key. Re-importing a line with the same
  `request_id` is a no-op; a line with the same `request_id` but a different `raw_hash` is
  reported as a conflict and not applied.
- **Retention is a property of the object directory**, not of rows: deleting or expiring
  objects (per tenant directory, per age) removes content while rows, costs and scores
  survive. Content-retention-off is simply an importer that skips the store write and leaves
  `content_ref` null.

## Consequences

- Replays and judges read content through the store, so the store interface is the one
  place a remote backend plugs in.
- Redaction is regex-based and will miss things; it is a floor, not a guarantee. A
  Presidio-style NER pass can be added behind the same `Redactor.text` method. Placeholders
  can change what a replayed model does (a redacted email is not an email); scorers must
  compare candidate and incumbent on the *same* redacted prompt, which they do because both
  read the stored document.
- Content-addressing dedupes identical requests for free and makes objects immutable, which
  is what evidence should be.
- Two requests that differ only in redacted values share an object; `raw_hash` still
  distinguishes them.
