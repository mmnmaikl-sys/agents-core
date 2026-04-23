"""PG audit log for compliance-grade trace of every agent action.

Every tool_use / tool_result / llm_call / evaluation becomes one INSERT.
For legal businesses this is the "who did what, when" record — auditors
need it, post-hoc debugging needs it. Langfuse is great for UX but its
retention is the cloud's, not ours.

Schema (research/evaluation.md §5):

    CREATE TABLE agent_audit_log (
        id            BIGSERIAL PRIMARY KEY,
        agent_name    TEXT NOT NULL,
        session_id    TEXT NOT NULL,
        turn          INTEGER,
        event_type    TEXT,     -- tool_use | tool_result | llm_call | evaluation
        content       JSONB,
        tokens_in     INTEGER,
        tokens_out    INTEGER,
        cost_usd      NUMERIC(10, 6),
        latency_ms    INTEGER,
        is_error      BOOLEAN,
        created_at    TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX ON agent_audit_log (agent_name, created_at DESC);

``ensure_schema`` applies the DDL idempotently on startup; call it once
per agent process. The writer uses one short-lived connection per call —
under light agent load (tens of events/sec) this is fine; if that
changes, swap ``psycopg.AsyncConnectionPool`` in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

__all__ = ["AuditEvent", "AuditLog", "AUDIT_LOG_DDL"]

EventType = Literal["tool_use", "tool_result", "llm_call", "evaluation"]


AUDIT_LOG_DDL = """\
CREATE TABLE IF NOT EXISTS agent_audit_log (
    id BIGSERIAL PRIMARY KEY,
    agent_name TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn INTEGER,
    event_type TEXT NOT NULL,
    content JSONB,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd NUMERIC(10, 6),
    latency_ms INTEGER,
    is_error BOOLEAN,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_agent_audit_log_agent_created
    ON agent_audit_log (agent_name, created_at DESC);
"""


@dataclass
class AuditEvent:
    agent_name: str
    session_id: str
    event_type: EventType
    turn: int | None = None
    content: dict[str, Any] | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    is_error: bool | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class AuditLog:
    """Thin async writer over Postgres.

    Parameters
    ----------
    dsn:
        Postgres connection string. Same format as other agents
        (``postgresql://user:pass@host:port/db``).
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError as e:  # pragma: no cover - enforced by extras
            raise ImportError(
                "AuditLog requires 'psycopg[binary,pool]' "
                "(install agents-core[pg])"
            ) from e
        self._dsn = dsn

    async def ensure_schema(self) -> None:
        import psycopg

        async with await psycopg.AsyncConnection.connect(self._dsn) as conn, conn.cursor() as cur:
            await cur.execute(AUDIT_LOG_DDL)

    async def record(self, event: AuditEvent) -> None:
        import psycopg

        async with await psycopg.AsyncConnection.connect(self._dsn) as conn, conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO agent_audit_log
                  (agent_name, session_id, turn, event_type, content,
                   tokens_in, tokens_out, cost_usd, latency_ms, is_error,
                   created_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)
                """,
                (
                    event.agent_name,
                    event.session_id,
                    event.turn,
                    event.event_type,
                    json.dumps(event.content) if event.content else None,
                    event.tokens_in,
                    event.tokens_out,
                    event.cost_usd,
                    event.latency_ms,
                    event.is_error,
                    event.created_at,
                ),
            )

    async def recent(
        self,
        agent_name: str,
        limit: int = 100,
        *,
        since_hours: int | None = 24,
    ) -> list[dict[str, Any]]:
        """Return rows for ``agent_name`` sorted newest first.

        DoD: "SELECT за последние 24h работает" — pass ``since_hours=24``.
        """
        import psycopg

        sql = (
            "SELECT agent_name, session_id, turn, event_type, content, "
            "tokens_in, tokens_out, cost_usd, latency_ms, is_error, created_at "
            "FROM agent_audit_log WHERE agent_name = %s "
        )
        params: list = [agent_name]
        if since_hours is not None:
            sql += "AND created_at >= NOW() - (%s || ' hours')::interval "
            params.append(str(int(since_hours)))
        sql += "ORDER BY created_at DESC LIMIT %s"
        params.append(int(limit))

        async with await psycopg.AsyncConnection.connect(self._dsn) as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in rows]
