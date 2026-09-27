"""Batching tests for non-empty sequences."""

import pytest

from local_smart_playlist.audio.features import batches
from local_smart_playlist.type_utils import NonEmptyTuple


def test_batches_return_non_empty_tuples() -> None:
    result = list(batches(NonEmptyTuple((1, 2, 3)), batch_size=2))
    assert result == [NonEmptyTuple((1, 2)), NonEmptyTuple((3,))]
    assert all(isinstance(batch, NonEmptyTuple) for batch in result)


def test_batches_reject_non_positive_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        _ = list(batches(NonEmptyTuple((1,)), batch_size=0))
