from unittest.mock import Mock

import pytest
from fastapi import Request

from prodml.api.dependencies import get_predictor


def test_get_predictor_returns_loaded_predictor() -> None:
    """Return the predictor stored in application state."""

    predictor = Mock()

    request = Mock(spec=Request)
    request.app.state.predictor = predictor

    result = get_predictor(request)

    assert result is predictor


def test_get_predictor_raises_when_predictor_is_not_loaded() -> None:
    """Raise RuntimeError when no predictor is loaded."""

    request = Mock(spec=Request)
    request.app.state.predictor = None

    with pytest.raises(
        RuntimeError,
        match="Wildfire predictor is not loaded.",
    ):
        get_predictor(request)
