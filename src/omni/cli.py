"""The ``omni`` command line interface.

Commands arrive with their features (see docs/ROADMAP.md): ``db`` landed with the core
models (PR 2), ``catalog`` with the pricing table (PR 3); ``ingest`` (PR 4) and the rest
follow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from omni import __version__

app = typer.Typer(
    help="Swap-Safety Engine: prove a model swap on your own traffic.",
    no_args_is_help=True,
)
db_app = typer.Typer(help="Database schema (Alembic).", no_args_is_help=True)
catalog_app = typer.Typer(help="Model catalog and pricing.", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(catalog_app, name="catalog")
workloads_app = typer.Typer(help="Workloads: segmentation, stats, merges.", no_args_is_help=True)
app.add_typer(workloads_app, name="workloads")
sample_app = typer.Typer(help="Samples: stratified subsets and golden sets.", no_args_is_help=True)
app.add_typer(sample_app, name="sample")
replay_app = typer.Typer(help="Shadow replays of samples against candidates.", no_args_is_help=True)
app.add_typer(replay_app, name="replay")


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"omni {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-V", callback=_show_version, is_eager=True, help="Print the version."
        ),
    ] = False,
) -> None:
    """omni: shadow-replay, score and recommend model swaps per workload."""


def _fail(message: str, code: int = 1) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


# ----------------------------------------------------------------------------- db


@db_app.command("upgrade")
def db_upgrade(
    revision: Annotated[str, typer.Argument(help="target revision")] = "head",
) -> None:
    """Apply migrations up to REVISION (default: head)."""
    from alembic import command

    from omni.db import alembic_config

    cfg = alembic_config()
    command.upgrade(cfg, revision)
    typer.echo(f"{cfg.get_main_option('sqlalchemy.url')} is at {revision}")


@db_app.command("revision")
def db_revision(
    message: Annotated[str, typer.Option("-m", "--message", help="revision message")],
    autogenerate: Annotated[
        bool, typer.Option(help="diff the models against the database to draft the migration")
    ] = True,
) -> None:
    """Create a new migration file (development). Review it before committing."""
    from alembic import command

    from omni.db import alembic_config

    cfg = alembic_config()
    command.revision(cfg, message=message, autogenerate=autogenerate)


# ----------------------------------------------------------------------------- catalog


@catalog_app.command("sync")
def catalog_sync(
    file: Annotated[
        Path | None,
        typer.Option("--file", help="catalog YAML to load (default: the bundled catalog)"),
    ] = None,
) -> None:
    """Validate the catalog file and upsert it into the model_catalog table."""
    from omni.catalog import DEFAULT_CATALOG_PATH, CatalogError, load, sync
    from omni.db import session_scope

    file = file or DEFAULT_CATALOG_PATH
    try:
        entries = load(file)
    except CatalogError as e:
        _fail(str(e))
    with session_scope() as s:
        r = sync(s, entries)
    typer.echo(
        f"{file}: {len(entries)} models; inserted {len(r.inserted)}, updated {len(r.updated)}, "
        f"unchanged {len(r.unchanged)}, deprecated {len(r.deprecated)}"
    )


@catalog_app.command("show")
def catalog_show(
    model_id: Annotated[str, typer.Argument(help="model id as in the catalog")],
) -> None:
    """Print one catalog row and the cost of a 10k input / 1k output call."""
    from omni.catalog import cost_of
    from omni.db import session_scope
    from omni.models import ModelCatalog

    with session_scope() as s:
        row = s.get(ModelCatalog, model_id)
        if row is None:
            _fail(f"{model_id!r} is not in model_catalog (run `omni catalog sync`)")
        p = row.pricing
        typer.echo(f"{row.model_id}  ({row.display_name}, {row.provider}, {row.status})")
        typer.echo(f"  context_window={row.context_window} max_output={row.max_output}")
        typer.echo(f"  capabilities={row.capabilities}")
        typer.echo(
            "  pricing USD/1M: "
            f"input={p.get('input')} cached_input={p.get('cached_input')} "
            f"output={p.get('output')} reasoning={p.get('reasoning')} "
            f"per_request={p.get('per_request')}  verified_at={p.get('verified_at')}"
        )
        if row.source_url:
            typer.echo(f"  source: {row.source_url}")
        example = {"input": 10_000, "output": 1_000, "cached_input": 0, "reasoning": 0}
        typer.echo(f"  example 10k in / 1k out: ${cost_of(example, p)}")


# ----------------------------------------------------------------------------- ingest


@app.command("ingest")
def ingest(
    from_path: Annotated[
        Path, typer.Option("--from", help="JSONL file of NormalizedRequest lines")
    ],
    source: Annotated[
        str | None, typer.Option(help="override the source field: litellm|otel|sidecar|import")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="validate and redact, write nothing")
    ] = False,
) -> None:
    """Import production requests from a JSONL file (docs/INGEST.md)."""
    from omni.db import session_scope
    from omni.ingest import import_file

    if not from_path.exists():
        _fail(f"{from_path} does not exist")
    with session_scope() as s:
        r = import_file(s, from_path, source=source, dry_run=dry_run)
        if dry_run:
            s.rollback()
    verb = "would insert" if dry_run else "inserted"
    typer.echo(
        f"{from_path}: {r.total} lines; {verb} {r.inserted}, skipped {r.skipped} "
        f"(already present), conflicts {r.conflicts}, invalid {len(r.invalid)}; "
        f"cost filled from catalog for {r.costed}"
    )
    if r.redactions:
        typer.echo("redacted: " + ", ".join(f"{k}={v}" for k, v in sorted(r.redactions.items())))
    for lineno, err in r.invalid:
        typer.echo(f"  line {lineno}: {err}", err=True)
    if r.workloads_created:
        typer.echo("new workloads: " + ", ".join(r.workloads_created))


# ----------------------------------------------------------------------------- workloads


def _workload(session, ref: str):
    from sqlalchemy import select

    from omni.models import Workload

    wl = session.get(Workload, ref) or session.scalar(select(Workload).where(Workload.name == ref))
    if wl is None:
        _fail(f"workload {ref!r} not found (omni workloads list)")
    return wl


@workloads_app.command("segment")
def workloads_segment(
    all_requests: Annotated[
        bool, typer.Option("--all", help="re-fingerprint every request, not only unassigned")
    ] = False,
) -> None:
    """Attach unassigned requests to workloads by fingerprint (ADR 0006)."""
    from omni.db import session_scope
    from omni.workloads import segment

    with session_scope() as s:
        r = segment(s, all_requests=all_requests)
    typer.echo(
        f"moved {r.moved}, unchanged {r.unchanged}, no content {r.no_content}; "
        f"new workloads: {', '.join(r.created) or 'none'}"
    )


@workloads_app.command("refresh")
def workloads_refresh(days: Annotated[int, typer.Option(help="trailing window")] = 30) -> None:
    """Recompute volume, spend, token mix, cache ratio and incumbent per workload."""
    from omni.db import session_scope
    from omni.workloads import refresh

    with session_scope() as s:
        n = len(refresh(s, days=days))
    typer.echo(f"refreshed {n} workloads over the last {days} days")


@workloads_app.command("list")
def workloads_list(
    sort: Annotated[str, typer.Option(help="spend | volume | name")] = "spend",
) -> None:
    """List workloads ranked by dollars at stake."""
    from sqlalchemy import select

    from omni.db import session_scope
    from omni.models import Workload
    from omni.workloads.service import step_kind_counts

    keys = {
        "spend": lambda w: -w.spend_30d,
        "volume": lambda w: -w.volume_30d,
        "name": lambda w: w.name,
    }
    if sort not in keys:
        _fail("--sort must be spend, volume or name")
    with session_scope() as s:
        rows = sorted(s.scalars(select(Workload)).all(), key=keys[sort])
        typer.echo(
            f"{'workload_id':<22} {'name':<24} {'incumbent':<22} "
            f"{'vol30':>6} {'spend30':>10} {'cache':>6}  step kinds"
        )
        for w in rows:
            kinds = ", ".join(
                f"{k}={n}" for k, n in sorted(step_kind_counts(s, w.workload_id).items())
            )
            cache = "-" if w.cache_read_ratio is None else f"{w.cache_read_ratio:.0%}"
            typer.echo(
                f"{w.workload_id:<22} {w.name[:24]:<24} {(w.incumbent_model_id or '-')[:22]:<22} "
                f"{w.volume_30d:>6} {w.spend_30d:>10.4f} {cache:>6}  {kinds}"
            )


@workloads_app.command("rename")
def workloads_rename(workload: str, name: str) -> None:
    """Give a workload a human name."""
    from omni.db import session_scope

    with session_scope() as s:
        wl = _workload(s, workload)
        old, wl.name = wl.name, name
    typer.echo(f"renamed {old} -> {name}")


@workloads_app.command("merge")
def workloads_merge(src: str, dst: str) -> None:
    """Move SRC's requests into DST and route SRC's fingerprint to DST from now on."""
    from omni.db import session_scope
    from omni.workloads import merge, refresh

    with session_scope() as s:
        a, b = _workload(s, src), _workload(s, dst)
        try:
            moved = merge(s, a, b)
        except ValueError as e:
            _fail(str(e))
        refresh(s)
        typer.echo(f"moved {moved} requests from {a.name} into {b.name}")


# ----------------------------------------------------------------------------- sample


def _parse_ts(value: str | None, name: str):
    from datetime import UTC, datetime

    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(f"--{name} must be an ISO-8601 timestamp, got {value!r}")
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _strata_table(counts: dict) -> str:
    rows = [f"  {'stratum':<40} {'population':>10} {'sampled':>8}"]
    for s in sorted(counts):
        c = counts[s]
        rows.append(f"  {s:<40} {c['population']:>10} {c['sampled']:>8}")
    return "\n".join(rows)


@sample_app.command("create")
def sample_create(
    workload: Annotated[str, typer.Option(help="workload id or name")],
    size: Annotated[int | None, typer.Option(help="explicit size; default from --ci")] = None,
    ci: Annotated[float, typer.Option(help="target 95% CI half-width for a win rate")] = 0.05,
    golden: Annotated[bool, typer.Option("--golden", help="mark as the golden set")] = False,
    since: Annotated[str | None, typer.Option(help="ISO timestamp, inclusive")] = None,
    until: Annotated[str | None, typer.Option(help="ISO timestamp, exclusive")] = None,
    seed: Annotated[int | None, typer.Option(help="RNG seed; stored on the sample")] = None,
) -> None:
    """Create a new, immutable stratified sample version for a workload (ADR 0007)."""
    from omni.db import session_scope
    from omni.sampling import SamplingError, create_sample

    with session_scope() as s:
        wl = _workload(s, workload)
        try:
            smp = create_sample(
                s,
                wl,
                size=size,
                ci_half_width=ci,
                golden=golden,
                since=_parse_ts(since, "since"),
                until=_parse_ts(until, "until"),
                seed=seed,
            )
        except SamplingError as e:
            _fail(str(e))
        typer.echo(
            f"{smp.sample_id} v{smp.version} for {wl.name}: {smp.size} requests "
            f"(eligible {smp.created_from_range['eligible']}, seed {smp.seed}"
            f"{', golden' if smp.is_golden else ''})"
        )
        typer.echo(_strata_table(smp.strata_def["counts"]))


@sample_app.command("list")
def sample_list(
    workload: Annotated[str | None, typer.Option(help="workload id or name")] = None,
) -> None:
    """List samples, oldest version first."""
    from omni.db import session_scope
    from omni.sampling import list_samples

    with session_scope() as s:
        wl = _workload(s, workload) if workload else None
        rows = list_samples(s, wl)
        typer.echo(f"{'sample_id':<22} {'workload':<24} {'ver':>3} {'size':>5} golden  created")
        for smp in rows:
            typer.echo(
                f"{smp.sample_id:<22} {smp.workload.name[:24]:<24} {smp.version:>3} {smp.size:>5} "
                f"{'yes' if smp.is_golden else 'no':<6}  {smp.created_at:%Y-%m-%d %H:%M}"
            )


@sample_app.command("show")
def sample_show(sample_id: str) -> None:
    """Print a sample's strata table and the first request ids."""
    from omni.db import session_scope
    from omni.models import Sample

    with session_scope() as s:
        smp = s.get(Sample, sample_id)
        if smp is None:
            _fail(f"sample {sample_id!r} not found")
        rng = smp.created_from_range
        typer.echo(
            f"{smp.sample_id} v{smp.version} workload={smp.workload.name} size={smp.size} "
            f"golden={smp.is_golden} seed={smp.seed} strata={smp.strata_def['version']} "
            f"redaction={smp.redaction_version}"
        )
        typer.echo(
            f"  range: {rng.get('first_ts')} .. {rng.get('last_ts')} "
            f"(eligible {rng.get('eligible')})"
        )
        typer.echo(_strata_table(smp.strata_def["counts"]))
        head = smp.request_ids[:10]
        more = (
            f" ... (+{len(smp.request_ids) - len(head)})"
            if len(smp.request_ids) > len(head)
            else ""
        )
        typer.echo("  request_ids: " + ", ".join(head) + more)


# ----------------------------------------------------------------------------- replay


def _resolve_sample(session, wl, ref: str):
    from omni.models import Sample
    from omni.sampling import golden, list_samples

    if ref == "latest":
        rows = list_samples(session, wl)
        smp = rows[-1] if rows else None
    elif ref == "golden":
        smp = golden(session, wl)
    else:
        smp = session.get(Sample, ref)
    if smp is None or smp.workload_id != wl.workload_id:
        _fail(f"no sample {ref!r} for workload {wl.name} (omni sample create)")
    return smp


@replay_app.command("run")
def replay_run(
    workload: Annotated[str, typer.Option(help="workload id or name")],
    candidates: Annotated[str, typer.Option(help="comma list of model_id[@effort]")],
    sample: Annotated[str, typer.Option(help="sample id, 'latest' or 'golden'")] = "latest",
    budget: Annotated[float, typer.Option(help="USD cap per candidate run")] = 5.0,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="estimate cost, send nothing")] = False,
) -> None:
    """Replay a sample against candidate models, one budgeted run per candidate (ADR 0008)."""
    from omni.catalog import CatalogError
    from omni.db import session_scope
    from omni.ingest.store import ObjectStore
    from omni.replay import Candidate, MissingApiKey, UnsupportedParameter, run_replay

    specs = [Candidate.parse(c) for c in candidates.split(",") if c.strip()]
    if not specs:
        _fail("--candidates is empty")
    store = ObjectStore()
    with session_scope() as s:
        wl = _workload(s, workload)
        smp = _resolve_sample(s, wl, sample)
        typer.echo(
            f"replaying {wl.name} sample {smp.sample_id} v{smp.version} (n={smp.size}) "
            f"against {len(specs)} candidate(s), budget ${budget:.2f} each"
            f"{' [dry run]' if dry_run else ''}"
        )
        typer.echo(
            f"{'candidate':<32} {'run':<22} {'status':<16} "
            f"{'sent':>5} {'errors':>6} {'spent':>10} {'estimate':>10}"
        )
        for cand in specs:
            try:
                r = run_replay(s, store, smp, cand, budget, dry_run=dry_run)
            except (CatalogError, UnsupportedParameter, MissingApiKey) as e:
                typer.echo(f"{cand.label:<32} {'-':<22} {'not started':<16} {e}")
                continue
            run_id = r.run.run_id if r.run else "-"
            typer.echo(
                f"{cand.label:<32} {run_id:<22} {r.status:<16} {r.sent:>5} {r.errors:>6} "
                f"{r.spent_usd:>10.4f} {r.estimated_usd:>10.4f}"
            )
            for w in r.warnings:
                typer.echo(f"  warning: {w}")
        if dry_run:
            s.rollback()


@replay_app.command("list")
def replay_list(
    workload: Annotated[str | None, typer.Option(help="workload id or name")] = None,
) -> None:
    """List replay runs, newest first."""
    from sqlalchemy import select

    from omni.db import session_scope
    from omni.models import ReplayRun

    with session_scope() as s:
        q = select(ReplayRun).order_by(ReplayRun.started_at.desc())
        if workload:
            q = q.where(ReplayRun.workload_id == _workload(s, workload).workload_id)
        typer.echo(
            f"{'run_id':<22} {'candidate':<32} {'status':<16} "
            f"{'spent':>10} {'budget':>8} {'errors':>6}  sample"
        )
        for r in s.scalars(q):
            typer.echo(
                f"{r.run_id:<22} {r.candidate_label[:32]:<32} {r.status:<16} {r.spent_usd:>10.4f} "
                f"{r.budget_usd:>8.2f} {r.error_count:>6}  {r.sample_id}"
            )


@replay_app.command("show")
def replay_show(run_id: str) -> None:
    """Print a run and its per-request results."""
    from omni.db import session_scope
    from omni.models import ReplayRun

    with session_scope() as s:
        r = s.get(ReplayRun, run_id)
        if r is None:
            _fail(f"run {run_id!r} not found")
        typer.echo(
            f"{r.run_id} {r.candidate_label} status={r.status} adapter={r.adapter_name} "
            f"spent=${r.spent_usd:.4f} of ${r.budget_usd:.2f} "
            f"errors={r.error_count} sample={r.sample_id}"
        )
        for w in r.warnings or []:
            typer.echo(f"  warning: {w}")
        typer.echo(
            f"  {'request':<14} {'cost':>10} {'ms':>7} {'in':>6} {'out':>5} "
            f"{'cached':>6} {'reason':>6}  error"
        )
        for res in r.results:
            u = res.usage or {}
            typer.echo(
                f"  {res.request_id:<14} {(res.cost or 0):>10.6f} {(res.latency_ms or 0):>7.0f} "
                f"{u.get('input', 0):>6} {u.get('output', 0):>5} {u.get('cached_input', 0):>6} "
                f"{u.get('reasoning', 0):>6}  {res.error or ''}"
            )


if __name__ == "__main__":  # pragma: no cover
    app()
