"""
Build and validate Gemini-safe profiles while preserving useful column semantics.

Strict Privacy Mandate:
- Zero raw data leakage: Never send cell values, sample rows, category label dictionaries,
  clusters, or full CSV data to external AI models.
- Centralized outbound payload validation enforcing strictly allowlisted metadata.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


class PrivacyViolationError(ValueError):
    """Raised when an outbound payload destined for external AI violates privacy constraints."""
    pass


PII_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PII_PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
PII_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _check_no_pii_strings(obj: Any) -> None:
    """Scan strings recursively to ensure no raw email, phone, or SSN values leak to external AI."""
    if isinstance(obj, str):
        if PII_EMAIL_PATTERN.search(obj):
            raise PrivacyViolationError(f"Potential raw email address detected in outbound AI payload: {obj}")
        if PII_SSN_PATTERN.search(obj):
            raise PrivacyViolationError(f"Potential SSN detected in outbound AI payload: {obj}")
        if PII_PHONE_PATTERN.search(obj):
            raise PrivacyViolationError(f"Potential raw phone number detected in outbound AI payload: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_no_pii_strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            _check_no_pii_strings(item)


_ALLOWED_ROOT_KEYS: Set[str] = {
    "rows",
    "columns",
    "duplicate_rows",
    "column_name_issue_count",
    "column_profiles",
    "privacy_note",
}

_ALLOWED_DETAIL_KEYS: Set[str] = {
    "dtype",
    "missing_count",
    "missing_percentage",
    "unique_count",
    "empty_column",
    "constant_column",
    "semantic_hints",
    "semantic_type",
    "sensitive_field",
    "statistics",
    "invalid_format_count",
    "category_count",
    "most_frequent_category_count",
    "inconsistent_category_count",
}

_FORBIDDEN_KEY_PATTERNS: Set[str] = {
    "sample",
    "sample_values",
    "value_counts",
    "raw_data",
    "rows_sample",
    "cells",
    "records",
    "data",
    "cluster_members",
    "candidates",
}


def validate_outbound_payload(payload: Dict[str, Any]) -> None:
    """
    Centralized validation enforcing that NO raw row values, samples, or unauthorized
    keys can be transmitted to external LLM providers.
    """
    if not isinstance(payload, dict):
        raise PrivacyViolationError("Outbound AI payload must be a JSON dictionary.")

    # 1. Enforce allowed root keys
    for key in payload.keys():
        if key not in _ALLOWED_ROOT_KEYS:
            raise PrivacyViolationError(
                f"Unauthorized root key '{key}' in outbound AI payload. "
                f"Allowed keys are: {sorted(_ALLOWED_ROOT_KEYS)}"
            )

    column_profiles = payload.get("column_profiles", {})
    if not isinstance(column_profiles, dict):
        raise PrivacyViolationError("'column_profiles' must be a dictionary.")

    # 2. Inspect each column profile
    for col_name, col_meta in column_profiles.items():
        if not isinstance(col_meta, dict):
            raise PrivacyViolationError(f"Column profile for '{col_name}' must be a dictionary.")

        for key, val in col_meta.items():
            if key not in _ALLOWED_DETAIL_KEYS:
                raise PrivacyViolationError(
                    f"Unauthorized column metadata key '{key}' for column '{col_name}'. "
                    f"Forbidden from AI payload."
                )

            # Check for forbidden patterns
            lower_key = str(key).lower()
            if any(forbidden in lower_key for forbidden in _FORBIDDEN_KEY_PATTERNS):
                raise PrivacyViolationError(
                    f"Forbidden data leakage key '{key}' detected in column '{col_name}'."
                )

            # If statistics, only numeric/null/bool allowed
            if key == "statistics":
                if not isinstance(val, dict):
                    raise PrivacyViolationError(f"Column '{col_name}' statistics must be a dictionary.")
                for stat_k, stat_v in val.items():
                    if stat_v is not None and not isinstance(stat_v, (int, float, bool)):
                        raise PrivacyViolationError(
                            f"Non-numeric statistic '{stat_k}={stat_v}' found in column '{col_name}'. "
                            f"Only aggregate numbers are permitted."
                        )

            # Ensure no lists of raw text values exist
            if isinstance(val, (list, set, tuple)):
                if key not in ("semantic_hints",):
                    raise PrivacyViolationError(
                        f"Non-scalar list key '{key}' found in column '{col_name}'. "
                        f"Raw value arrays are strictly disallowed in AI payloads."
                    )

    # 3. Deep regex scan to prevent raw PII (emails, SSNs, phone numbers) in payload
    _check_no_pii_strings(payload)


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

    # Centralized Outbound Privacy Verification
    validate_outbound_payload(safe_profile)

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
