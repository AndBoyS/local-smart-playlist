from collections.abc import Iterable
from typing import Any, Self


def verify_type[T](x: Any, type: type[T]) -> T:
    assert isinstance(x, type)
    return x


def verify_not_none[T](x: T | None) -> T:
    assert x is not None
    return x


class NonEmptyTuple[T](tuple[T, ...]):
    """Tuple that rejects empty iterables at construction."""

    def __new__(cls, values: Iterable[T]) -> Self:
        items = tuple(values)
        if len(items) == 0:
            raise ValueError("NonEmptyTuple cannot be empty")
        return super().__new__(cls, items)
