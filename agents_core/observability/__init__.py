"""agents_core.observability — Langfuse wrapper + audit log.

Re-exports:
    langfuse_wrap module (get_langfuse, start_generation, end_generation_ok/err, calc_cost)
"""
from agents_core.observability import langfuse_wrap

__all__ = ["langfuse_wrap"]
