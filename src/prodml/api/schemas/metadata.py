from pydantic import BaseModel, Field


class MetadataResponse(BaseModel):
    """Response schema for model metadata."""

    model_version: str = Field(description="Version of the production model.")
    training_date: str = Field(
        description="Date when the production model was trained."
    )
    feature_names: list[str] = Field(description="Feature names used by the model.")
    framework: str = Field(description="Machine learning framework used by the model.")
    artifact_hash: str = Field(description="Hash identifying the model artifact.")
