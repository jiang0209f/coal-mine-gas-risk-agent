"""SQLAlchemy Core 存储与共享事务。"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.pool import StaticPool

__all__ = ["Store", "metadata", "observations", "records"]

metadata = MetaData()

records = Table(
    "records",
    metadata,
    Column("kind", String, primary_key=True),
    Column("id", String, primary_key=True),
    Column("payload", JSON, nullable=False),
)

observations = Table(
    "observations",
    metadata,
    Column("id", String, primary_key=True),
    Column("point_id", String, nullable=False, index=True),
    Column("source_id", String, nullable=False),
    Column("event_time", String, nullable=False),
    Column("available_at", String, nullable=False),
    Column("raw_value", Text, nullable=True),
    Column("value", Float, nullable=True),
    Column("quality_flags", JSON, nullable=False),
    Column("source_file_id", String, nullable=False),
    Column("row_number", Integer, nullable=False),
    Column("import_id", String, nullable=False),
    Index("ix_observations_point_event_available", "point_id", "event_time", "available_at"),
)


class Store:
    def __init__(self, database_url: str) -> None:
        url = make_url(database_url)
        options: dict[str, Any] = {"hide_parameters": True}
        if url.get_backend_name() == "sqlite":
            options["connect_args"] = {"check_same_thread": False}
            if url.database in (None, "", ":memory:") or url.query.get("mode") == "memory":
                options["poolclass"] = StaticPool
            else:
                Path(url.database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.engine = create_engine(database_url, **options)
        try:
            metadata.create_all(self.engine)
        except Exception:
            self.engine.dispose()
            raise

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        with self._lock:
            with self.engine.begin() as connection:
                yield connection

    @contextmanager
    def _connection(self, connection: Connection | None) -> Iterator[Connection]:
        if connection is not None:
            yield connection
        else:
            with self.transaction() as active:
                yield active

    def get(
        self, kind: str, id: str, connection: Connection | None = None
    ) -> dict | None:
        with self._connection(connection) as active:
            return active.execute(
                select(records.c.payload).where(records.c.kind == kind, records.c.id == id)
            ).scalar_one_or_none()

    def list(self, kind: str, connection: Connection | None = None) -> list[dict]:
        with self._connection(connection) as active:
            return list(
                active.execute(
                    select(records.c.payload).where(records.c.kind == kind).order_by(records.c.id)
                ).scalars()
            )

    def put(
        self, kind: str, id: str, payload: dict, connection: Connection | None = None
    ) -> dict:
        with self._connection(connection) as active:
            condition = (records.c.kind == kind, records.c.id == id)
            existing = active.execute(select(records.c.id).where(*condition)).first()
            if existing is None:
                active.execute(insert(records).values(kind=kind, id=id, payload=payload))
            else:
                active.execute(update(records).where(*condition).values(payload=payload))
        return payload

    def delete(self, kind: str, id: str, connection: Connection | None = None) -> None:
        with self._connection(connection) as active:
            active.execute(delete(records).where(records.c.kind == kind, records.c.id == id))

    def close(self) -> None:
        with self._lock:
            self.engine.dispose()
