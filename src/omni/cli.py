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
    file: Annotated[Path, typer.Option("--file", help="catalog YAML to load")] = Path(
        "catalog/models.yaml"
    ),
) -> None:
    """Validate the catalog file and upsert it into the model_catalog table."""
    from omni.catalog import CatalogError, load, sync
    from omni.db import session_scope

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


if __name__ == "__main__":  # pragma: no cover
    app()
