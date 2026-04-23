"""Evaluators and metrics for agent trajectories (research/evaluation.md §3)."""

from .judge import Judge, JudgeResult
from .metrics import MetricsSummary, bucket_by_trial, learning_curve, summarize

__all__ = [
    "Judge",
    "JudgeResult",
    "MetricsSummary",
    "bucket_by_trial",
    "learning_curve",
    "summarize",
]
