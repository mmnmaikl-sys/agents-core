"""Unit tests for agents_core.tools.common.pg (task 0.19)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("psycopg")

from agents_core.tools.common.pg import PGPool, make_pg_tools  # noqa: E402
from agents_core.tools.registry import ToolRegistry  # noqa: E402


def _mock_conn(fetchall=None, rowcount=1, description=None, fetchone=None):
    cur = AsyncMock()
    cur.fetchall = AsyncMock(return_value=fetchall or [])
    cur.fetchone = AsyncMock(return_value=fetchone)
    cur.rowcount = rowcount
    cur.description = description
    cm_cur = MagicMock()
    cm_cur.__aenter__ = AsyncMock(return_value=cur)
    cm_cur.__aexit__ = AsyncMock(return_value=False)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cm_cur)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    return conn, cur


class TestPGPool:
    def test_construction_requires_psycopg(self):
        pool = PGPool("postgresql://u:p@h/d")
        assert pool._dsn.startswith("postgresql://")


class TestFactoryShape:
    def test_returns_four_tools_with_correct_tiers(self):
        pool = PGPool("postgresql://u:p@h/d")
        tools = make_pg_tools(pool)
        assert [t.name for t in tools] == [
            "pg_query",
            "pg_execute",
            "pg_migration_run",
            "pg_batch_insert",
        ]
        tiers = {t.name: t.tier for t in tools}
        assert tiers == {
            "pg_query": "read",
            "pg_execute": "write",
            "pg_migration_run": "danger",
            "pg_batch_insert": "write",
        }

    def test_registers_into_registry(self):
        pool = PGPool("postgresql://u:p@h/d")
        reg = ToolRegistry(make_pg_tools(pool))
        api = reg.for_api()
        assert len(api) == 4
        assert all("input_schema" in t for t in api)

    def test_write_tools_require_idempotency_key(self):
        pool = PGPool("postgresql://u:p@h/d")
        for t in make_pg_tools(pool):
            if t.tier in ("write", "danger"):
                assert "idempotency_key" in t.input_schema["required"], t.name


@pytest.mark.asyncio
class TestQueryHandler:
    async def test_query_returns_list_of_dicts(self):
        pool = PGPool("postgresql://u:p@h/d")
        conn, cur = _mock_conn(
            fetchall=[("Мяклов", 42)],
            description=[("name",), ("age",)],
        )
        with patch("psycopg.AsyncConnection.connect", AsyncMock(return_value=conn)):
            q = next(t for t in make_pg_tools(pool) if t.name == "pg_query")
            result = await q.handler(sql="SELECT name, age FROM u WHERE id = %s", params=[1])
        assert result == [{"name": "Мяклов", "age": 42}]
        assert cur.execute.call_args.args == ("SELECT name, age FROM u WHERE id = %s", [1])

    async def test_query_rejects_dict_params_with_bad_keys(self):
        pool = PGPool("postgresql://u:p@h/d")
        q = next(t for t in make_pg_tools(pool) if t.name == "pg_query")
        with pytest.raises(ValueError, match="invalid named param key"):
            await q.handler(sql="SELECT 1", params={"name; DROP TABLE x": 1})

    async def test_query_accepts_dict_params_with_valid_keys(self):
        pool = PGPool("postgresql://u:p@h/d")
        conn, cur = _mock_conn(fetchall=[], description=[])
        with patch("psycopg.AsyncConnection.connect", AsyncMock(return_value=conn)):
            q = next(t for t in make_pg_tools(pool) if t.name == "pg_query")
            await q.handler(sql="SELECT %(name)s", params={"name": "a"})
        assert cur.execute.await_count == 1


@pytest.mark.asyncio
class TestExecuteHandler:
    async def test_execute_returns_rowcount(self):
        pool = PGPool("postgresql://u:p@h/d")
        conn, cur = _mock_conn(rowcount=3)
        with patch("psycopg.AsyncConnection.connect", AsyncMock(return_value=conn)):
            e = next(t for t in make_pg_tools(pool) if t.name == "pg_execute")
            result = await e.handler(
                sql="UPDATE u SET name = %s WHERE id = %s",
                params=["x", 1],
                idempotency_key="k1",
            )
        assert result == {"rowcount": 3}


@pytest.mark.asyncio
class TestMigrationRun:
    async def test_fresh_migration_applies(self):
        pool = PGPool("postgresql://u:p@h/d")
        conn, cur = _mock_conn(fetchone=None)  # not applied yet
        with patch("psycopg.AsyncConnection.connect", AsyncMock(return_value=conn)):
            m = next(t for t in make_pg_tools(pool) if t.name == "pg_migration_run")
            result = await m.handler(
                name="add_agent_reflections",
                sql="CREATE TABLE x (id INT);",
                idempotency_key="k",
            )
        assert result == {"applied": True, "name": "add_agent_reflections"}
        # 4 queries: DDL, SELECT check, user SQL, INSERT into _agent_migrations
        assert cur.execute.await_count == 4

    async def test_already_applied_migration_is_noop(self):
        pool = PGPool("postgresql://u:p@h/d")
        conn, cur = _mock_conn(fetchone=(1,))  # already applied
        with patch("psycopg.AsyncConnection.connect", AsyncMock(return_value=conn)):
            m = next(t for t in make_pg_tools(pool) if t.name == "pg_migration_run")
            result = await m.handler(
                name="add_x", sql="SELECT 1", idempotency_key="k"
            )
        assert result["applied"] is False
        assert "already applied" in result["reason"]

    async def test_bad_migration_name_rejected(self):
        pool = PGPool("postgresql://u:p@h/d")
        m = next(t for t in make_pg_tools(pool) if t.name == "pg_migration_run")
        with pytest.raises(ValueError, match="slug-like"):
            await m.handler(
                name="bad; DROP TABLE x",
                sql="SELECT 1",
                idempotency_key="k",
            )


@pytest.mark.asyncio
class TestBatchInsert:
    async def test_batch_insert_executemany(self):
        pool = PGPool("postgresql://u:p@h/d")
        conn, cur = _mock_conn(rowcount=2)
        with patch("psycopg.AsyncConnection.connect", AsyncMock(return_value=conn)):
            b = next(t for t in make_pg_tools(pool) if t.name == "pg_batch_insert")
            res = await b.handler(
                table="people",
                columns=["name", "age"],
                rows=[["a", 1], ["b", 2]],
                idempotency_key="k",
            )
        assert res == {"inserted": 2}
        sql = cur.executemany.call_args.args[0]
        assert sql == "INSERT INTO people (name, age) VALUES (%s, %s)"

    async def test_empty_rows_no_insert(self):
        pool = PGPool("postgresql://u:p@h/d")
        b = next(t for t in make_pg_tools(pool) if t.name == "pg_batch_insert")
        res = await b.handler(
            table="people",
            columns=["name"],
            rows=[],
            idempotency_key="k",
        )
        assert res == {"inserted": 0}

    async def test_bad_table_or_column_rejected(self):
        pool = PGPool("postgresql://u:p@h/d")
        b = next(t for t in make_pg_tools(pool) if t.name == "pg_batch_insert")
        with pytest.raises(ValueError, match="table name"):
            await b.handler(
                table="people; DROP",
                columns=["n"],
                rows=[["a"]],
                idempotency_key="k",
            )
        with pytest.raises(ValueError, match="column name"):
            await b.handler(
                table="people",
                columns=["name; DROP"],
                rows=[["a"]],
                idempotency_key="k",
            )
