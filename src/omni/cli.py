"""The ``omni`` command line interface.

PR 1 ships only ``omni --version`` so that CI has something to run. Commands arrive with
their features: ``db`` (PR 2), ``catalog`` (PR 3), ``ingest`` (PR 4), and so on; see
docs/ROADMAP.md.
"""

from __future__ import annotations

from typing import Annotated

import typer

from omni import __version__

app = typer.Typer(
    help="Swap-Safety Engine: prove a model swap on your own traffic.",
    no_args_is_help=True,
)


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


if __name__ == "__main__":  # pragma: no cover
    app()
