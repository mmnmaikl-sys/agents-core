"""Anti-infinite-loop detection for ReAct (research/architecture.md §3 and §9).

When the model calls the same tool with the same arguments N times in a row
it is almost certainly stuck (misparsing a result, looping on a transient
error it cannot recover from). Rule from research §9:

> Если зациклился (один tool 3 раза с теми же args) — "Can't proceed,
> escalating to human".

`LoopDetector` watches the stream of tool calls and raises `LoopDetectedError`
once the same `(tool_name, args_fingerprint)` has been observed more than
`threshold` times in the most recent `window` calls. Default: threshold=3,
window=6 (so 3 identical calls among the last 6 is enough).

Use:
    detector = LoopDetector(threshold=3)
    for tool_use in turn.tool_uses:
        detector.observe(tool_use.name, tool_use.input)
    # raises LoopDetectedError when triggered
"""
from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any


def _fingerprint(args: Any) -> str:
    """Canonical stable representation of tool args. None == empty dict."""
    if args is None:
        args = {}
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(args)


@dataclass(frozen=True)
class RepeatedCall:
    tool: str
    args_fingerprint: str
    count: int
    threshold: int


class LoopDetectedError(RuntimeError):
    def __init__(self, repeat: RepeatedCall) -> None:
        super().__init__(
            f"infinite-loop guard: tool {repeat.tool!r} called {repeat.count} times "
            f"with identical args (threshold={repeat.threshold})"
        )
        self.repeat = repeat


class LoopDetector:
    """Sliding-window repeat counter.

    Parameters
    ----------
    threshold:
        Number of identical (tool, args) calls in the window that trips the
        guard (default 3). Must be >= 2.
    window:
        How many most-recent calls to keep in memory (default threshold * 2).
        Older calls fall off and don't contribute to repeat counts.
    """

    def __init__(self, threshold: int = 3, window: int | None = None) -> None:
        if threshold < 2:
            raise ValueError("threshold must be >= 2")
        self._threshold = threshold
        self._window = window if window is not None else threshold * 2
        self._recent: deque[tuple[str, str]] = deque(maxlen=self._window)

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def window(self) -> int:
        return self._window

    def observe(self, tool_name: str, args: Any) -> None:
        """Record one call and raise if threshold hit."""
        key = (tool_name, _fingerprint(args))
        self._recent.append(key)
        counts = Counter(self._recent)
        count = counts[key]
        if count >= self._threshold:
            raise LoopDetectedError(RepeatedCall(
                tool=tool_name,
                args_fingerprint=key[1],
                count=count,
                threshold=self._threshold,
            ))

    def reset(self) -> None:
        self._recent.clear()
