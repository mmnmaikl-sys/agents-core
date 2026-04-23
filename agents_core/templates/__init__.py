"""Prompt templates bundled with agents-core.

Files in this package are shipped via ``package-data`` (see pyproject.toml).
Load them by name with ``load_template("canonical")``.
"""

from __future__ import annotations

from importlib.resources import files

__all__ = ["CANONICAL_SYSTEM_PROMPT", "load_template"]


def load_template(name: str) -> str:
    """Return the text of a template bundled in this package.

    ``name`` is the stem without extension, e.g. ``"canonical"`` →
    ``canonical.txt``. Raises ``FileNotFoundError`` if missing.
    """
    resource = files(__package__).joinpath(f"{name}.txt")
    return resource.read_text(encoding="utf-8")


CANONICAL_SYSTEM_PROMPT: str = load_template("canonical")
