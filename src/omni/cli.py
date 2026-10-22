"""The ``omni`` command line interface.

Commands arrive with their features (see docs/ROADMAP.md): ``db`` landed with the core
models (PR 2); ``catalog`` (PR 3), ``ingest`` (PR 4) and the rest follow.
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
app.add_typer(db_app, name="db")


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


# ----------------------------------------------------------------------------- db


def alembic_config(url: str | None = None):
    """Alembic config pointing at the packaged migrations; no alembic.ini needed."""
    from alembic.config import Config

    from omni.db import database_url

    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).resolve().parent / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url or database_url())
    return cfg


@db_app.command("upgrade")
def db_upgrade(
    revision: Annotated[str, typer.Argument(help="target revision")] = "head",
) -> None:
    """Apply migrations up to REVISION (default: head)."""
    from alembic import command

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

    cfg = alembic_config()
    command.revision(cfg, message=message, autogenerate=autogenerate)


if __name__ == "__main__":  # pragma: no cover
    app()
