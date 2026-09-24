from typing import Any


def verify_type[T](x: Any, type: type[T]) -> T:
    assert isinstance(x, type)
    return x
