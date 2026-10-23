"""Build a Gemini-safe profile while preserving useful column semantics.

This module never places cell samples or category labels in the AI payload.
Original column names are retained so the model can reason about fields such as
email, phone_number, date_of_birth, and episode_duration_min.
"""
from __future__ import annotations

from typing import Any


_ALLOWED_DETAIL_KEYS = (
    "dtype", "missing_count", "missing_percentage", "unique_count",
    "empty_column", "constant_column", "semantic_hints", "semantic_type",
    "sensitive_field", "statistics", "invalid_format_count",
)


def build_ai_safe_profile(profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Return aggregate metadata only; keep original column names.

    The second return value is retained for backward compatibility and is an
    identity map (column name -> same column name).
    """
    source_columns = profile.get("column_profiles", {})
    safe_columns: dict[str, Any] = {}

    for original_name, raw_details in source_columns.items():
        details = raw_details if isinstance(raw_details, dict) else {}
        safe_details: dict[str, Any] = {}
        for key in _ALLOWED_DETAIL_KEYS:
            if key in details:
                safe_details[key] = details[key]

        # Numeric aggregate statistics only; never allow arbitrary text values.
        stats = safe_details.get("statistics")
        if isinstance(stats, dict):
            safe_details["statistics"] = {
                str(key): value for key, value in stats.items()
                if isinstance(value, (int, float, bool)) or value is None
            }

        # A few frequency aggregates can help diagnose quality without exposing
        # category labels. The profiler may provide this as separate fields.
        for key in ("category_count", "most_frequent_category_count", "inconsistent_category_count"):
            if key in details and isinstance(details[key], (int, float, bool)):
                safe_details[key] = details[key]

        safe_columns[str(original_name)] = safe_details

    safe_profile = {
        "rows": profile.get("rows"),
        "columns": profile.get("columns", len(source_columns)),
        "duplicate_rows": profile.get("duplicate_rows"),
        # Do not send original/suggested name pairs: they are unnecessary for planning.
        "column_name_issue_count": len(profile.get("column_name_issues", [])),
        "column_profiles": safe_columns,
        "privacy_note": (
            "Original column names and locally detected semantic types are included. "
            "No cell values, sample rows, category labels, or value-frequency keys are included."
        ),
    }
    identity_map = {str(name): str(name) for name in source_columns}
    return safe_profile, identity_map


def remap_plan_columns(plan_data: dict[str, Any], token_to_column: dict[str, str]) -> dict[str, Any]:
    """Backward-compatible no-op for plans using original column names."""
    actions = plan_data.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError("Cleaning plan actions must be a list")
    valid_columns = list(token_to_column.values())
    exact_columns = set(valid_columns)
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("Each cleaning action must be an object")
        column = action.get("column")
        if column is None or not valid_columns or column in exact_columns:
            continue
        # Safely repair casing-only mismatches (e.g. join_date -> Join_Date),
        # but never guess when the match is ambiguous or materially different.
        matches = [name for name in valid_columns if str(name).casefold() == str(column).casefold()]
        if len(matches) == 1:
            action["column"] = matches[0]
        else:
            raise ValueError(f"The AI referenced unknown column '{column}' or an ambiguous casing match. No repair was performed.")
    return plan_data
