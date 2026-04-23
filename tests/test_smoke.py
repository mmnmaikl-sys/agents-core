"""Skeleton smoke test — just makes sure the package imports cleanly.

The real tests per module land alongside each task in Волна 0.
"""


def test_import_version():
    import agents_core

    assert agents_core.__version__ == "0.0.1"


def test_subpackages_importable():
    # Every subpackage has a docstring-only __init__.py in the skeleton.
    # This test guards against accidental deletion or typos before we
    # start filling them in Волна 0.
    import agents_core.evaluation  # noqa: F401
    import agents_core.llm  # noqa: F401
    import agents_core.loop  # noqa: F401
    import agents_core.memory  # noqa: F401
    import agents_core.observability  # noqa: F401
    import agents_core.safety  # noqa: F401
    import agents_core.tools  # noqa: F401
    import agents_core.tools.common  # noqa: F401
