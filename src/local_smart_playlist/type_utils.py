from typing import Any


def verify_type[T](x: Any, type: type[T]) -> T:
    assert isinstance(x, type)
    return x


def verify_not_none[T](x: T | None) -> T:
    assert x is not None
    return x
