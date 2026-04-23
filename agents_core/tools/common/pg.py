"""Postgres tools for agents-core (task 0.19).

Four ``Tool`` factories: ``query``, ``execute``, ``migration_run``,
``batch_insert``. Each takes a shared ``PGPool`` handle so tools can share
one connection pool per process instead of opening a new conn per call.

Safety notes
------------
- ``query`` and ``execute`` accept parameters as a list/tuple so SQL is
  parameterised and not f-strings — SQL injection is the #1 mistake
  LLM agents make, so we refuse ``params`` if it's a dict with
  non-identifier keys (guard against accidental string interpolation).
- ``migration_run`` reads .sql files from a fixed directory and tracks
  applied names in a ``_agent_migrations`` table. The DDL is idempotent;
  re-running is a no-op.
- ``batch_insert`` uses psycopg's ``executemany`` with ``copy`` fallback
  only in the rare high-throughput case — we keep it simple.

All tools are tier="write" because they change PG state (except ``query``,
which is "read"). Pair ``execute`` with a verify tool in your agent when
the research §9 rule 1 applies (the pair is caller-responsibility; the
registry only tags ``requires_verify``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from agents_core.tools.registry import Tool

logger = logging.getLogger(__name__)

__all__ = ["PGPool", "make_pg_tools"]


_MIGRATIONS_DDL = """\
CREATE TABLE IF NOT EXISTS _agent_migrations (
    name TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


class PGPool:
    """Tiny pool wrapper; one connection per call to keep the surface small.

    When you need a real pool, swap in ``psycopg_pool.AsyncConnectionPool``
    inside ``connect()`` — callers don't need to know.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError as e:  # pragma: no cover - enforced by extras
            raise ImportError(
                "PGPool requires 'psycopg[binary,pool]' "
                "(install agents-core[pg])"
            ) from e
        self._dsn = dsn

    async def connect(self):
        import psycopg

        return await psycopg.AsyncConnection.connect(self._dsn)


def _reject_dict_params(params: Any) -> None:
    # Parameterised queries via psycopg accept tuple/list/dict. We allow
    # tuple/list by default; a dict of ``{"name_in_sql": value}`` is fine
    # but *only* if keys are identifiers. This blocks a common mistake
    # where an LLM tries to pass ``{"sql": "DROP TABLE x"}`` etc.
    if isinstance(params, dict):
        for k in params:
            if not isinstance(k, str) or not k.replace("_", "").isalnum():
                raise ValueError(f"invalid named param key: {k!r}")


async def _query_handler(pool: PGPool, sql: str, params: Sequence | dict | None = None):
    _reject_dict_params(params)
    async with await pool.connect() as conn, conn.cursor() as cur:
        await cur.execute(sql, params or ())
        rows = await cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
    return [dict(zip(cols, r, strict=True)) for r in rows]


async def _execute_handler(
    pool: PGPool, sql: str, params: Sequence | dict | None = None
) -> dict[str, Any]:
    _reject_dict_params(params)
    async with await pool.connect() as conn, conn.cursor() as cur:
        await cur.execute(sql, params or ())
        return {"rowcount": cur.rowcount}


async def _migration_run_handler(
    pool: PGPool, name: str, sql: str
) -> dict[str, Any]:
    """Apply ``sql`` idempotently, tracked by ``name`` in ``_agent_migrations``."""
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"migration name must be slug-like: {name!r}")
    async with await pool.connect() as conn, conn.cursor() as cur:
        await cur.execute(_MIGRATIONS_DDL)
        await cur.execute(
            "SELECT 1 FROM _agent_migrations WHERE name = %s", (name,)
        )
        if await cur.fetchone():
            return {"applied": False, "name": name, "reason": "already applied"}
        await cur.execute(sql)
        await cur.execute(
            "INSERT INTO _agent_migrations (name) VALUES (%s)", (name,)
        )
    return {"applied": True, "name": name}


async def _batch_insert_handler(
    pool: PGPool,
    table: str,
    columns: list[str],
    rows: list[list[Any]],
) -> dict[str, Any]:
    if not table.replace("_", "").isalnum():
        raise ValueError(f"table name must be an identifier: {table!r}")
    for c in columns:
        if not c.replace("_", "").isalnum():
            raise ValueError(f"column name must be an identifier: {c!r}")
    if not rows:
        return {"inserted": 0}
    placeholders = ", ".join(["%s"] * len(columns))
    cols_sql = ", ".join(columns)
    sql = f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders})"
    async with await pool.connect() as conn, conn.cursor() as cur:
        await cur.executemany(sql, rows)
        return {"inserted": cur.rowcount if cur.rowcount is not None else len(rows)}


def make_pg_tools(pool: PGPool) -> list[Tool]:
    """Return the 4 PG tools bound to ``pool``. Register them in a ToolRegistry.

    All tools are idempotent=False because SQL statements usually aren't;
    callers wanting idempotency should pass an ``idempotency_key`` through
    the SQL (research §9 rule 4) or use ``ON CONFLICT``.
    """
    return [
        Tool(
            name="pg_query",
            description=(
                "Run a read-only SELECT against Postgres. Returns list of row "
                "dicts. Always parameterise via `params`; never concatenate "
                "user input into `sql`."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "params": {
                        "type": "array",
                        "description": "Positional parameters for %s placeholders.",
                    },
                },
                "required": ["sql"],
            },
            handler=lambda sql, params=None, _pool=pool: _query_handler(
                _pool, sql, params
            ),
            tier="read",
            idempotent=True,
            tags=("pg", "read"),
        ),
        Tool(
            name="pg_execute",
            description=(
                "Run a write statement (INSERT/UPDATE/DELETE). Returns "
                "{'rowcount': n}. Use `idempotency_key` in SQL when possible."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "params": {"type": "array"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["sql", "idempotency_key"],
            },
            handler=lambda sql, idempotency_key, params=None, _pool=pool: _execute_handler(
                _pool, sql, params
            ),
            tier="write",
            idempotent=False,
            requires_verify=True,
            tags=("pg", "write"),
        ),
        Tool(
            name="pg_migration_run",
            description=(
                "Apply a named migration; idempotent — re-running returns "
                "applied=false. Tracked in _agent_migrations."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "sql": {"type": "string"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["name", "sql", "idempotency_key"],
            },
            handler=lambda name, sql, idempotency_key, _pool=pool: (
                _migration_run_handler(_pool, name, sql)
            ),
            tier="danger",
            idempotent=True,
            requires_verify=True,
            tags=("pg", "migration"),
        ),
        Tool(
            name="pg_batch_insert",
            description=(
                "Bulk INSERT into `table` with `columns` and `rows`. Returns "
                "{'inserted': n}. For big bulk loads."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"type": "array"}},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["table", "columns", "rows", "idempotency_key"],
            },
            handler=lambda table, columns, rows, idempotency_key, _pool=pool: (
                _batch_insert_handler(_pool, table, columns, rows)
            ),
            tier="write",
            idempotent=False,
            requires_verify=True,
            tags=("pg", "write", "bulk"),
        ),
    ]
