"""Annotations for task runner map() parameter handling.

Provides the unmapped sentinel to mark parameters as static (not mapped over)
when using runner.map().
"""

from __future__ import annotations

from typing import Any


class unmapped:
    """Mark a value as static (not mapped over) in runner.map().

    Example:
        runner.map(fn, x=[1, 2, 3], config=unmapped(my_config))
        # fn(x=1, config=my_config), fn(x=2, config=my_config), fn(x=3, config=my_config)
    """

    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:
        return f"unmapped({self.value!r})"
