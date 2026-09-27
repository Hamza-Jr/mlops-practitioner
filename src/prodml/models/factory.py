from sklearn.ensemble import RandomForestClassifier

from prodml.utils.config import settings


def create_model(model_name: str) -> RandomForestClassifier:
    """Create a configured classification model."""

    if model_name == "random_forest":
        return RandomForestClassifier(
            **settings.random_forest_params,
            random_state=settings.random_state,
        )

    raise ValueError(f"Unsupported model: {model_name}")
