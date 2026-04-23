# agents-core

Internal Python library for building reliable Claude-based ReAct agents at 24bankrotstvo.
Based on deep research captured in `/Бизнес/Проекты/smart-agents/specs/research/`.

## Status

**WIP (Волна 0, kill-date 2026-04-30).** Skeleton only — each subpackage is a stub
that will be filled in tasks 0.2 through 0.26 from
`/Бизнес/Проекты/smart-agents/specs/CHANGE_MAP.md`.

## Install (pinned to main)

```bash
pip install "git+https://github.com/mmnmaikl-sys/agents-core.git@main"
```

From an agent service consuming the library:

```python
from agents_core import __version__
print(__version__)
```

## Architecture (target, after Волна 0)

4 layers, per the research document:

```
┌─ 4. Reflection & Learning    → agents_core.memory, agents_core.evaluation
├─ 3. Agent Loop (ReAct)       → agents_core.loop
├─ 2. Tool Registry            → agents_core.tools
└─ 1. Knowledge & API          → agents_core.llm, agents_core.tools.common,
                                  agents_core.observability
```

Plus `agents_core.safety` for tier-based access / HITL confirmation.

## Design principles (non-negotiable)

1. Core ≤ 500 LOC (llm, tools, loop, memory, evaluation, observability, safety).
2. Zero heavy deps — only Anthropic SDK, OpenAI SDK (DeepSeek), Pydantic, httpx, instructor, langfuse (optional), psycopg (optional).
3. Every SET is paired with VERIFY (GET-after-SET).
4. `tier=read|write|danger`; danger requires `user_confirmed=true`.
5. Budget tripwires: 12 steps, $0.50 / task, 40K tokens total.
6. Langfuse trace per agent run, audit log in Postgres for compliance.
7. No LangChain / CrewAI / AutoGPT — vanilla Claude + this library.

## Related

- Source of truth for plan: `/Бизнес/Проекты/smart-agents/specs/CHANGE_MAP.md`
- Research: `/Бизнес/Проекты/smart-agents/specs/research/{architecture,evaluation,migration-plan}.md`
- Predecessor spike: `/Бизнес/Проекты/agent-reliability-upgrade/specs/reliability-playbook.md`
