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


if __name__ == "__main__":  # pragma: no cover
    app()
