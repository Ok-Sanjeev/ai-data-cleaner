from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


ALLOWED_ACTIONS = Literal[
    "impute_missing",
    "drop_missing_rows",
    "remove_duplicates",
    "remove_column",
    "rename_column",
    "convert_to_date",
    "convert_to_numeric",
    "convert_to_boolean",
    "standardize_categories",
    "normalize_text",
    "normalize_whitespace",
    "normalize_column_names",
    "replace_empty_strings",
    "remove_invalid_values",
    "clip_outliers",
    "clean_phone_number",
    # Text Cleaning
    "trim_whitespace",
    "remove_special_characters",
    "remove_non_printable_characters",
    "normalize_unicode",
    # Data Type & Format Standardization
    "standardize_date_format",
    "extract_date_components",
    "standardize_currency",
    "extract_numeric_values",
    # Numeric Cleaning
    "round_numeric",
    "validate_numeric_ranges",
    "replace_negative_values",
    "convert_percentage",
    # Data Validation
    "validate_email",
    "extract_email",
    "validate_url",
    "remove_empty_rows",
    "detect_invalid_categories",
    # Structural Cleaning
    "split_column",
    "merge_columns",
    "deduplicate_by_columns",
    "remove_high_missing_columns",
    "remove_constant_columns",
    # Value Mapping
    "map_values",
    # Advanced Missing Value & Group Imputation
    "fill_with_constant",
    "fill_with_forward_value",
    "fill_with_backward_value",
    "fill_with_group_median",
    "fill_with_group_mode",
    "drop_columns_by_missing_threshold",
    # Advanced Text & Formatting
    "remove_html_tags",
    "handle_infinite_values",
    "validate_date_range",
    "reorder_columns",
    "extract_pattern",
    "standardize_phone_number",
    "standardize_country_names",
    # OpenRefine-Inspired Capabilities
    "grel_transform",
    "merge_clusters",
    "reconcile_column",
]


class CleaningAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    action: ALLOWED_ACTIONS

    column: str | None = None

    params: dict[str, Any] = Field(
        default_factory=dict
    )

    reasoning: str

    risk_level: Literal["green", "yellow", "red"] = "yellow"
    risk_label: Optional[str] = None
    warning: Optional[str] = None
    expected_impact: Optional[str] = None

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
            "remove_invalid_values",
            "clip_outliers",
            "remove_column",
            "rename_column",
            "clean_phone_number",
            "trim_whitespace",
            "remove_special_characters",
            "remove_non_printable_characters",
            "normalize_unicode",
            "standardize_date_format",
            "extract_date_components",
            "standardize_currency",
            "extract_numeric_values",
            "round_numeric",
            "validate_numeric_ranges",
            "replace_negative_values",
            "convert_percentage",
            "validate_email",
            "extract_email",
            "validate_url",
            "detect_invalid_categories",
            "split_column",
            "map_values",
            "fill_with_constant",
            "fill_with_forward_value",
            "fill_with_backward_value",
            "fill_with_group_median",
            "fill_with_group_mode",
            "remove_html_tags",
            "handle_infinite_values",
            "validate_date_range",
            "extract_pattern",
            "standardize_phone_number",
            "standardize_country_names",
            "grel_transform",
            "merge_clusters",
            "reconcile_column",
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
                "constant",
            }:
                raise ValueError(
                    "impute_missing requires method: "
                    "median, mean, mode, or constant"
                )

            if method == "constant" and "value" not in self.params:
                raise ValueError(
                    "Constant imputation requires a value"
                )

        elif self.action == "standardize_categories":
            mapping = self.params.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(
                    "standardize_categories requires a non-empty mapping"
                )

        elif self.action == "map_values":
            mapping = self.params.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(
                    "map_values requires a non-empty mapping dictionary"
                )

        elif self.action == "normalize_text":
            case = self.params.get("case", "lower")
            if case not in {"lower", "upper", "title", "none"}:
                raise ValueError(
                    "normalize_text case must be lower, upper, title, or none"
                )

        elif self.action == "remove_invalid_values":
            values = self.params.get("invalid_values")
            if not isinstance(values, list) or not values:
                raise ValueError(
                    "remove_invalid_values requires a non-empty invalid_values list"
                )

        elif self.action == "rename_column":
            new_name = self.params.get("new_name")
            if not isinstance(new_name, str) or not new_name.strip():
                raise ValueError(
                    "rename_column requires a non-empty string in params['new_name']"
                )

        elif self.action == "drop_missing_rows":
            how = self.params.get("how", "any")
            if how not in {"any", "all"}:
                raise ValueError("drop_missing_rows 'how' parameter must be 'any' or 'all'")

        elif self.action == "detect_invalid_categories":
            allowed = self.params.get("allowed_categories")
            if not isinstance(allowed, list) or not allowed:
                raise ValueError("detect_invalid_categories requires a non-empty allowed_categories list")

        elif self.action == "merge_columns":
            cols = self.params.get("columns")
            if not isinstance(cols, list) or len(cols) < 2:
                raise ValueError("merge_columns requires params['columns'] with at least 2 column names")

        elif self.action == "deduplicate_by_columns":
            subset = self.params.get("subset")
            if not isinstance(subset, list) or not subset:
                raise ValueError("deduplicate_by_columns requires params['subset'] with a list of column names")

        elif self.action == "replace_negative_values":
            method = self.params.get("method", "nan")
            if method not in {"nan", "zero", "abs"}:
                raise ValueError("replace_negative_values method must be 'nan', 'zero', or 'abs'")

        elif self.action == "remove_high_missing_columns":
            threshold = float(self.params.get("threshold", 0.8))
            if not (0.0 < threshold <= 1.0):
                raise ValueError("remove_high_missing_columns threshold must be between 0.0 and 1.0")

        elif self.action in {"fill_with_group_median", "fill_with_group_mode"}:
            if not self.params.get("group_column"):
                raise ValueError(f"{self.action} requires params['group_column']")

        elif self.action == "drop_columns_by_missing_threshold":
            th = float(self.params.get("threshold", 0.5))
            if not (0.0 < th <= 1.0):
                raise ValueError("drop_columns_by_missing_threshold threshold must be between 0.0 and 1.0")

        elif self.action == "reorder_columns":
            order = self.params.get("column_order")
            if not isinstance(order, list) or not order:
                raise ValueError("reorder_columns requires a non-empty params['column_order'] list")

        elif self.action == "extract_pattern":
            pat = self.params.get("regex_pattern")
            if not isinstance(pat, str) or not pat:
                raise ValueError("extract_pattern requires a non-empty string in params['regex_pattern']")

        elif self.action == "grel_transform":
            expr = self.params.get("expression")
            if not isinstance(expr, str) or not expr.strip():
                raise ValueError("grel_transform requires a non-empty params['expression'] string")

        elif self.action == "merge_clusters":
            if not self.params.get("clusters") and not self.params.get("mapping"):
                raise ValueError("merge_clusters requires params['clusters'] or params['mapping']")

        elif self.action == "reconcile_column":
            mapping = self.params.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("reconcile_column requires a non-empty params['mapping'] dictionary")

        return self


class CleaningPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    actions: list[CleaningAction]


# ============================================================
# OPENREFINE-INSPIRED API REQUEST MODELS
# ============================================================

class ClusterDetectRequest(BaseModel):
    session_id: str
    column: str
    method: str = "fingerprint"
    params: dict[str, Any] = Field(default_factory=dict)


class ClusterApplyRequest(BaseModel):
    session_id: str
    column: str
    clusters: list[dict[str, Any]] = Field(default_factory=list)
    mapping: Optional[dict[str, str]] = None


class GRELPreviewRequest(BaseModel):
    session_id: str
    column: str
    expression: str
    sample_limit: int = 15


class GRELApplyRequest(BaseModel):
    session_id: str
    column: str
    expression: str
    on_error: str = "keep"


class FacetComputeRequest(BaseModel):
    session_id: str
    column: str
    facet_type: str = "auto"
    top_n: int = 100
    bins: int = 15
    search: Optional[str] = None
    active_filters: Optional[dict[str, Any]] = None


class ExploreRequest(BaseModel):
    session_id: str
    filters: Optional[dict[str, Any]] = None
    global_search: Optional[str] = None
    sort_by: Optional[str] = None
    sort_ascending: bool = True
    offset: int = 0
    limit: int = 50


class ReconcileRequest(BaseModel):
    session_id: str
    column: str
    catalog: str = "countries"
    custom_entities: Optional[list[str]] = None
    threshold: float = 0.75


class ReconcileApplyRequest(BaseModel):
    session_id: str
    column: str
    mapping: dict[str, str]
    catalog: str = "reconciled"


class RecipeApplyRequest(BaseModel):
    session_id: str
    recipe: dict[str, Any]
    stop_on_error: bool = False

