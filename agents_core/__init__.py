"""agents_core — internal Python library for building reliable Claude-based agents.

Based on deep research at /Бизнес/Проекты/smart-agents/specs/research/ (2026-04-23):
- 4-layer architecture: Knowledge&API / Tool Registry / Agent Loop / Reflection&Learning
- Anthropic canonical building blocks (Workflow > Agent > Multi-agent hierarchy)
- ReAct loop with budget tripwires, Reflexion memory, multi-signal evaluator
- Prompt caching, Langfuse observability, SET/VERIFY symmetry, trust-levels

Keep the CORE (llm/ tools/ loop/ memory/ evaluation/ observability/ safety/)
at or below 500 LOC total. tools/common/ grows as new integrations are added;
concrete agents consume via `from agents_core.loop import ReActLoop` etc.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
