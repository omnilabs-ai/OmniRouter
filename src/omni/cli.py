"""The omni command line interface."""

from __future__ import annotations

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
            "--version",
            "-V",
            callback=_show_version,
            is_eager=True,
            help="Print the version.",
        ),
    ] = False,
) -> None:
    """omni: shadow-replay, score and recommend model swaps per workload."""


# Database commands


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
        bool, typer.Option(help="diff models against the database")
    ] = True,
) -> None:
    """Create a migration file (development). Review it before committing."""
    from alembic import command

    from omni.db import alembic_config

    cfg = alembic_config()
    command.revision(cfg, message=message, autogenerate=autogenerate)


# Catalog commands


@catalog_app.command("sync")
def catalog_sync() -> None:
    """Load catalog/models.yaml and synchronize it with the database."""
    from omni.catalog import sync_catalog
    from omni.db import session_scope

    with session_scope() as session:
        count = sync_catalog(session)

    typer.echo(f"Synced models: {count}")


@catalog_app.command("show")
def catalog_show(
    model_id: Annotated[str, typer.Argument(help="Model ID to display")],
) -> None:
    """Display a model from the database catalog."""
    from omni.db import session_scope
    from omni.models import ModelCatalog

    with session_scope() as session:
        model = session.get(ModelCatalog, model_id)

        if model is None:
            typer.echo(f"Model not found: {model_id}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Model ID: {model.model_id}")
        typer.echo(f"Provider: {model.provider}")
        typer.echo(f"Name: {model.display_name}")
        typer.echo(f"Context window: {model.context_window}")
        typer.echo(f"Max output: {model.max_output}")
        typer.echo(f"Pricing: {model.pricing}")
        typer.echo(f"Status: {model.status}")


if __name__ == "__main__":  # pragma: no cover
    app()
