from fastapi import Request

from prodml.models.predictor import WildfirePredictor


def get_predictor(request: Request) -> WildfirePredictor:
    """Return the loaded wildfire predictor from application state."""

    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        raise RuntimeError("Wildfire predictor is not loaded.")

    return predictor
