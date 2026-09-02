from typing import Any, Literal

from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


ALLOWED_ACTIONS = Literal[
    "impute_missing",
    "remove_duplicates",
    "convert_to_date",
    "convert_to_numeric",
    "convert_to_boolean",
    "standardize_categories",
    "normalize_text",
    "normalize_whitespace",
    "normalize_column_names",
    "replace_empty_strings",
    "remove_invalid_values",
    "clip_outliers"
]


class CleaningAction(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    action: ALLOWED_ACTIONS

    column: str | None = None

    params: dict[str, Any] = Field(
        default_factory=dict
    )

    reasoning: str

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    @model_validator(mode="after")
    def validate_action_parameters(self):

        column_actions = {
            "impute_missing",
            "convert_to_date",
            "convert_to_numeric",
            "convert_to_boolean",
            "standardize_categories",
            "normalize_text",
            "normalize_whitespace",
            "replace_empty_strings",
            "remove_invalid_values",
            "clip_outliers"
        }

        if self.action in column_actions and not self.column:
            raise ValueError(
                f"Column is required for action '{self.action}'"
            )

        if self.action == "impute_missing":

            method = self.params.get("method")

            if method not in {
                "median",
                "mean",
                "mode",
                "constant"
            }:
                raise ValueError(
                    "impute_missing requires method: "
                    "median, mean, mode, or constant"
                )

            if (
                method == "constant"
                and "value" not in self.params
            ):
                raise ValueError(
                    "Constant imputation requires a value"
                )

        elif self.action == "standardize_categories":

            mapping = self.params.get("mapping")

            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(
                    "standardize_categories requires "
                    "a non-empty mapping"
                )

        elif self.action == "normalize_text":

            case = self.params.get(
                "case",
                "lower"
            )

            if case not in {
                "lower",
                "upper",
                "title",
                "none"
            }:
                raise ValueError(
                    "normalize_text case must be "
                    "lower, upper, title, or none"
                )

        elif self.action == "remove_invalid_values":

            values = self.params.get(
                "invalid_values"
            )

            if not isinstance(values, list) or not values:
                raise ValueError(
                    "remove_invalid_values requires "
                    "a non-empty invalid_values list"
                )

        return self


class CleaningPlan(BaseModel):
    summary: str
    actions: list[CleaningAction]