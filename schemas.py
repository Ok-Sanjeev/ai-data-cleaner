from typing import Any, Literal

from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


ALLOWED_ACTIONS = Literal[
    "impute_missing",
    "remove_duplicates",
    "convert_to_date",
    "standardize_categories"
]


class CleaningAction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    action: ALLOWED_ACTIONS
    column: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_action_parameters(self):

        # Actions that operate on a specific column
        if self.action in {
            "impute_missing",
            "convert_to_date",
            "standardize_categories"
        }:
            if not self.column:
                raise ValueError(
                    f"Column is required for action '{self.action}'"
                )

        # Validate imputation parameters
        if self.action == "impute_missing":

            method = self.params.get("method")

            if method not in {
                "median",
                "mean",
                "mode"
            }:
                raise ValueError(
                    "impute_missing requires a method of "
                    "'median', 'mean', or 'mode'"
                )

        # Validate category standardization parameters
        if self.action == "standardize_categories":

            mapping = self.params.get("mapping")

            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(
                    "standardize_categories requires a "
                    "non-empty mapping"
                )

        return self


class CleaningPlan(BaseModel):
    summary: str
    actions: list[CleaningAction]