"""Unit tests for agents_core.memory.reflection (task 0.10) and decay (0.11)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agents_core.memory import (
    InMemoryReflectionStore,
    Reflection,
    age_decay_score,
    make_age_decay_score,
    task_hash,
)


def _r(
    hash_: str,
    lesson: str,
    *,
    created: datetime | None = None,
    confidence: float = 1.0,
    trial: int = 1,
    failure_mode: str = "wrong_filter",
    schema: str = "v1",
) -> Reflection:
    return Reflection(
        task_hash=hash_,
        trial_n=trial,
        trajectory_summary="…",
        failure_mode=failure_mode,
        lesson=lesson,
        created_at=created or datetime.now(UTC),
        confidence=confidence,
        schema_version=schema,
    )


class TestTaskHash:
    def test_stable_and_short(self):
        h = task_hash("  Запроси сделку 1234  ")
        assert len(h) == 16
        assert h == task_hash("Запроси сделку 1234")  # whitespace-trimmed

    def test_different_inputs_different_hashes(self):
        assert task_hash("a") != task_hash("b")


class TestReflectionDataclass:
    def test_defaults(self):
        r = _r("h", "use filter X")
        assert r.confidence == 1.0
        assert r.schema_version == "v1"
        assert r.age_seconds() >= 0.0
        assert r.trajectory_summary

    def test_age_seconds_handles_future_created_at(self):
        future = datetime.now(UTC) + timedelta(days=1)
        r = _r("h", "l", created=future)
        assert r.age_seconds() == 0.0  # clamped to 0


class TestInMemoryStore:
    @pytest.mark.asyncio
    async def test_save_and_retrieve_roundtrip(self):
        store = InMemoryReflectionStore()
        r = _r(task_hash("task1"), "lesson 1")
        await store.save(r)
        got = await store.retrieve(task_hash("task1"), k=3)
        assert got == [r]

    @pytest.mark.asyncio
    async def test_retrieve_empty_hash(self):
        store = InMemoryReflectionStore()
        assert await store.retrieve("missing", k=3) == []

    @pytest.mark.asyncio
    async def test_retrieve_k_zero_returns_empty(self):
        store = InMemoryReflectionStore()
        await store.save(_r("h", "l"))
        assert await store.retrieve("h", k=0) == []

    @pytest.mark.asyncio
    async def test_retrieve_limits_to_k(self):
        store = InMemoryReflectionStore()
        now = datetime.now(UTC)
        for i in range(5):
            await store.save(_r("h", f"l{i}", created=now - timedelta(minutes=i)))
        got = await store.retrieve("h", k=2)
        assert len(got) == 2
        # Default ordering = newest first
        assert got[0].lesson == "l0"
        assert got[1].lesson == "l1"

    @pytest.mark.asyncio
    async def test_retrieve_respects_schema_version(self):
        store = InMemoryReflectionStore()
        await store.save(_r("h", "old", schema="v1"))
        await store.save(_r("h", "new", schema="v2"))
        got = await store.retrieve("h", k=5, schema_version="v2")
        assert [r.lesson for r in got] == ["new"]

    @pytest.mark.asyncio
    async def test_custom_score_fn_reranks(self):
        store = InMemoryReflectionStore()
        now = datetime.now(UTC)
        old_high = _r("h", "old_high", created=now - timedelta(days=60), confidence=0.9)
        new_low = _r("h", "new_low", created=now, confidence=0.1)
        await store.save(old_high)
        await store.save(new_low)
        # Default (by created_at) puts new_low first
        default = await store.retrieve("h", k=5)
        assert default[0].lesson == "new_low"
        # Confidence-only scoring flips order
        by_conf = await store.retrieve("h", k=5, score_fn=lambda r: r.confidence)
        assert [r.lesson for r in by_conf] == ["old_high", "new_low"]

    @pytest.mark.asyncio
    async def test_size_counts_all_hashes(self):
        store = InMemoryReflectionStore()
        await store.save(_r("h1", "a"))
        await store.save(_r("h1", "b"))
        await store.save(_r("h2", "c"))
        assert store.size() == 3


class TestAgeDecay:
    def test_newer_scores_higher_than_older_same_confidence(self):
        now = datetime.now(UTC)
        score = make_age_decay_score(half_life_days=30.0, now=now)
        new = _r("h", "l", created=now, confidence=1.0)
        old = _r("h", "l", created=now - timedelta(days=60), confidence=1.0)
        assert score(new) > score(old)

    def test_half_life_cuts_score_in_half(self):
        now = datetime.now(UTC)
        hl = 30.0
        score = make_age_decay_score(half_life_days=hl, now=now)
        young = _r("h", "l", created=now, confidence=1.0)
        at_hl = _r("h", "l", created=now - timedelta(days=hl), confidence=1.0)
        assert score(young) == pytest.approx(1.0, rel=1e-6)
        assert score(at_hl) == pytest.approx(0.5, rel=1e-6)

    def test_confidence_multiplies_score(self):
        now = datetime.now(UTC)
        score = make_age_decay_score(half_life_days=30.0, now=now)
        high = _r("h", "l", created=now, confidence=0.9)
        low = _r("h", "l", created=now, confidence=0.1)
        assert score(high) == pytest.approx(0.9, rel=1e-6)
        assert score(low) == pytest.approx(0.1, rel=1e-6)

    def test_invalid_half_life_raises(self):
        with pytest.raises(ValueError):
            make_age_decay_score(half_life_days=0.0)
        with pytest.raises(ValueError):
            make_age_decay_score(half_life_days=-1.0)

    def test_default_score_fn_is_callable(self):
        r = _r("h", "l", confidence=0.7)
        assert age_decay_score(r) == pytest.approx(0.7, rel=1e-3)

    @pytest.mark.asyncio
    async def test_decay_plugs_into_store_retrieval(self):
        store = InMemoryReflectionStore()
        now = datetime.now(UTC)
        old_high = _r("h", "old_high", created=now - timedelta(days=60), confidence=0.95)
        new_low = _r("h", "new_low", created=now, confidence=0.30)
        await store.save(old_high)
        await store.save(new_low)
        score = make_age_decay_score(half_life_days=30.0, now=now)
        got = await store.retrieve("h", k=5, score_fn=score)
        # new_low (0.30 × 1) > old_high (0.95 × 0.25) → new_low wins
        assert got[0].lesson == "new_low"


class TestPGReflectionStore:
    def test_requires_psycopg_import_ok(self):
        # psycopg is in [pg] extras; if installed, construct is allowed.
        pytest.importorskip("psycopg")
        from agents_core.memory import PGReflectionStore

        store = PGReflectionStore("postgresql://fake:fake@localhost:1/fake")
        assert store._dsn.startswith("postgresql://")
