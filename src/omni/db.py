"""Engine, session factory and Alembic configuration.

The database URL comes from ``DATABASE_URL``. When it is unset, a local SQLite file under
``.omni/`` is used so development and tests need no setup. Postgres is the same code path
with a ``postgresql+psycopg://`` URL (ADR 0003).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_SQLITE_PATH = Path(".omni") / "omni.db"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    return f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"


def ensure_sqlite_parent(url: str) -> None:
    """Create the directory a file-backed SQLite URL points into.

    SQLite creates the file but not its parent directory, so without this a fresh checkout
    fails with "unable to open database file". Every code path that opens the database
    (``get_engine`` and the Alembic environment) goes through here.
    """
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)


def get_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    ensure_sqlite_parent(url)
    engine = create_engine(url, future=True)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_conn, _record):  # pragma: no cover - trivial
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """A transaction-per-block session: commits on success, rolls back on error."""
    session = session_factory(engine)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def alembic_config(url: str | None = None):
    """Alembic config pointing at the packaged migrations; no alembic.ini needed."""
    from alembic.config import Config

    url = url or database_url()
    ensure_sqlite_parent(url)
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg
