"""Reflection memory (Reflexion pattern, research §3 layer 4).

A ``Reflection`` is a short actionable lesson extracted from a failed
trajectory by a Self-Reflection step. The store lets future runs retrieve
top-k relevant reflections for a similar task, so the actor can read past
mistakes before replanning.

Schema_version lets us invalidate old lessons when tools change.
Decay (age-based scoring) lives in ``memory.decay`` and plugs into retrieval
via the ``score_fn`` callback.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

__all__ = [
    "Reflection",
    "ReflectionStore",
    "InMemoryReflectionStore",
    "PGReflectionStore",
    "task_hash",
]


def task_hash(task: str) -> str:
    """Stable short hash for a task description (for exact-match retrieval)."""
    return hashlib.sha256(task.strip().encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Reflection:
    task_hash: str
    trial_n: int
    trajectory_summary: str  # 2-3 lines, not the full trajectory
    failure_mode: str  # "wrong_filter" | "hallucination" | "loop" | ...
    lesson: str  # 1-2 sentences, actionable
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    confidence: float = 0.5  # 0..1; used by ReflectionScorer
    schema_version: str = "v1"  # bump when tool registry changes

    def age_seconds(self, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        return max(0.0, (now - self.created_at).total_seconds())


ScoreFn = Callable[[Reflection], float]


class ReflectionStore(Protocol):
    """Minimal interface: store + retrieve top-k for a task_hash."""

    async def save(self, reflection: Reflection) -> None: ...

    async def retrieve(
        self,
        task_hash: str,
        k: int = 3,
        *,
        score_fn: ScoreFn | None = None,
        schema_version: str | None = None,
    ) -> list[Reflection]: ...


class InMemoryReflectionStore:
    """Process-local store. Good for tests, small agents, dev loops."""

    def __init__(self) -> None:
        self._by_hash: dict[str, list[Reflection]] = {}

    async def save(self, reflection: Reflection) -> None:
        self._by_hash.setdefault(reflection.task_hash, []).append(reflection)

    async def retrieve(
        self,
        task_hash: str,
        k: int = 3,
        *,
        score_fn: ScoreFn | None = None,
        schema_version: str | None = None,
    ) -> list[Reflection]:
        if k <= 0:
            return []
        candidates = list(self._by_hash.get(task_hash, ()))
        if schema_version is not None:
            candidates = [r for r in candidates if r.schema_version == schema_version]
        if score_fn is None:
            candidates.sort(key=lambda r: r.created_at, reverse=True)
        else:
            candidates.sort(key=score_fn, reverse=True)
        return candidates[:k]

    def size(self) -> int:
        return sum(len(v) for v in self._by_hash.values())


# --- PG backend ---------------------------------------------------------
# psycopg is an optional dep (extras = ["pg"]). Import lazily so the module
# imports cleanly in environments that only use the in-memory store.

_PG_DDL = """\
CREATE TABLE IF NOT EXISTS agent_reflections (
    id BIGSERIAL PRIMARY KEY,
    task_hash TEXT NOT NULL,
    trial_n INTEGER NOT NULL,
    trajectory_summary TEXT NOT NULL,
    failure_mode TEXT NOT NULL,
    lesson TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    schema_version TEXT NOT NULL DEFAULT 'v1'
);
CREATE INDEX IF NOT EXISTS ix_agent_reflections_hash_created
    ON agent_reflections (task_hash, created_at DESC);
"""


class PGReflectionStore:
    """Postgres-backed store. Requires ``psycopg[binary,pool]``.

    No pgvector dependency: retrieval is by exact task_hash + score_fn.
    Upgrade path to pgvector: add ``embedding vector(1024)`` column and
    ORDER BY embedding <-> query_embedding LIMIT k. Kept out of v1 so the
    core stays thin and installs work without native deps.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError as e:  # pragma: no cover - enforced by extras
            raise ImportError(
                "PGReflectionStore requires 'psycopg[binary,pool]' "
                "(install agents-core[pg])"
            ) from e
        self._dsn = dsn

    async def ensure_schema(self) -> None:
        import psycopg

        async with await psycopg.AsyncConnection.connect(self._dsn) as conn, conn.cursor() as cur:
            await cur.execute(_PG_DDL)

    async def save(self, reflection: Reflection) -> None:
        import psycopg

        async with await psycopg.AsyncConnection.connect(self._dsn) as conn, conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO agent_reflections
                  (task_hash, trial_n, trajectory_summary, failure_mode,
                   lesson, created_at, confidence, schema_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    reflection.task_hash,
                    reflection.trial_n,
                    reflection.trajectory_summary,
                    reflection.failure_mode,
                    reflection.lesson,
                    reflection.created_at,
                    reflection.confidence,
                    reflection.schema_version,
                ),
            )

    async def retrieve(
        self,
        task_hash: str,
        k: int = 3,
        *,
        score_fn: ScoreFn | None = None,
        schema_version: str | None = None,
    ) -> list[Reflection]:
        if k <= 0:
            return []
        import psycopg

        sql = (
            "SELECT task_hash, trial_n, trajectory_summary, failure_mode, "
            "lesson, created_at, confidence, schema_version "
            "FROM agent_reflections WHERE task_hash = %s"
        )
        params: list = [task_hash]
        if schema_version is not None:
            sql += " AND schema_version = %s"
            params.append(schema_version)
        # Over-fetch so score_fn can re-rank; DB can't run arbitrary Python.
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(max(k * 5, k))

        async with await psycopg.AsyncConnection.connect(self._dsn) as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

        rs = [
            Reflection(
                task_hash=r[0],
                trial_n=r[1],
                trajectory_summary=r[2],
                failure_mode=r[3],
                lesson=r[4],
                created_at=r[5],
                confidence=r[6],
                schema_version=r[7],
            )
            for r in rows
        ]
        if score_fn is not None:
            rs.sort(key=score_fn, reverse=True)
        return rs[:k]
