"""Unit tests for agents_core.observability.audit (task 0.15).

Full integration (real PG INSERT/SELECT) requires a Postgres — kept for
the integration smoke phase. Here we test shape and SQL behaviour with
mocked psycopg.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents_core.observability import AUDIT_LOG_DDL, AuditEvent, AuditLog


class TestAuditEvent:
    def test_dataclass_defaults(self):
        e = AuditEvent(
            agent_name="hr-director",
            session_id="s1",
            event_type="tool_use",
        )
        assert e.turn is None
        assert e.content is None
        assert e.created_at  # auto-filled

    def test_custom_fields(self):
        e = AuditEvent(
            agent_name="a",
            session_id="s",
            event_type="llm_call",
            turn=3,
            content={"model": "haiku"},
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.00012,
            latency_ms=450,
            is_error=False,
        )
        assert e.content == {"model": "haiku"}
        assert e.cost_usd == 0.00012


class TestDDLShape:
    def test_ddl_creates_table_and_index(self):
        sql = AUDIT_LOG_DDL.lower()
        assert "create table if not exists agent_audit_log" in sql
        assert "create index if not exists" in sql
        # Required columns from research §5
        for col in (
            "agent_name",
            "session_id",
            "turn",
            "event_type",
            "content",
            "tokens_in",
            "tokens_out",
            "cost_usd",
            "latency_ms",
            "is_error",
            "created_at",
        ):
            assert col in sql


class TestAuditLogConstructor:
    def test_requires_psycopg_extras(self):
        pytest.importorskip("psycopg")  # skip if extras not installed
        log = AuditLog("postgresql://u:p@h:5432/d")
        assert log._dsn.startswith("postgresql://")


@pytest.mark.asyncio
class TestWriter:
    async def test_ensure_schema_runs_ddl(self):
        pytest.importorskip("psycopg")
        cur = AsyncMock()
        cm_cur = MagicMock()
        cm_cur.__aenter__ = AsyncMock(return_value=cur)
        cm_cur.__aexit__ = AsyncMock(return_value=False)
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=cm_cur)
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "psycopg.AsyncConnection.connect", AsyncMock(return_value=conn)
        ):
            await AuditLog("postgresql://x").ensure_schema()
        cur.execute.assert_awaited_once()
        assert "agent_audit_log" in cur.execute.call_args.args[0].lower()

    async def test_record_inserts_all_fields(self):
        pytest.importorskip("psycopg")
        cur = AsyncMock()
        cm_cur = MagicMock()
        cm_cur.__aenter__ = AsyncMock(return_value=cur)
        cm_cur.__aexit__ = AsyncMock(return_value=False)
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=cm_cur)
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "psycopg.AsyncConnection.connect", AsyncMock(return_value=conn)
        ):
            event = AuditEvent(
                agent_name="hr",
                session_id="sess-1",
                event_type="tool_use",
                turn=2,
                content={"tool": "deal_get", "input": {"id": 1}},
                tokens_in=10,
                tokens_out=5,
                cost_usd=0.00001,
                latency_ms=120,
                is_error=False,
            )
            await AuditLog("postgresql://x").record(event)
        sql, params = cur.execute.call_args.args
        assert "insert into agent_audit_log" in sql.lower()
        assert params[0] == "hr"
        assert params[1] == "sess-1"
        assert params[2] == 2
        assert params[3] == "tool_use"
        # content serialized
        assert '"deal_get"' in params[4]
        assert params[5] == 10 and params[6] == 5
        assert params[7] == 0.00001

    async def test_recent_builds_24h_query(self):
        pytest.importorskip("psycopg")
        cur = AsyncMock()
        cur.description = [
            ("agent_name",), ("session_id",), ("turn",), ("event_type",),
            ("content",), ("tokens_in",), ("tokens_out",), ("cost_usd",),
            ("latency_ms",), ("is_error",), ("created_at",),
        ]
        cur.fetchall = AsyncMock(return_value=[
            ("hr", "s1", 1, "tool_use", {"x": 1}, 10, 5, 0.0, 100, False, "2026-04-24"),
        ])
        cm_cur = MagicMock()
        cm_cur.__aenter__ = AsyncMock(return_value=cur)
        cm_cur.__aexit__ = AsyncMock(return_value=False)
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=cm_cur)
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "psycopg.AsyncConnection.connect", AsyncMock(return_value=conn)
        ):
            rows = await AuditLog("postgresql://x").recent("hr", limit=50)
        sql = cur.execute.call_args.args[0].lower()
        assert "where agent_name = %s" in sql
        assert "order by created_at desc" in sql
        assert "interval" in sql  # since_hours=24 path
        assert rows == [{
            "agent_name": "hr", "session_id": "s1", "turn": 1,
            "event_type": "tool_use", "content": {"x": 1},
            "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.0,
            "latency_ms": 100, "is_error": False, "created_at": "2026-04-24",
        }]
