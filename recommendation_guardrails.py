"""Local recommendation guardrails for transformations that alter text casing."""
from __future__ import annotations

import re
import pandas as pd


_TEXT_FIELDS_TO_PRESERVE = {
    "description", "descriptions", "summary", "review", "reviews",
    "comment", "comments", "feedback", "content", "body", "text",
    "job_description", "job_posting", "skills", "skills_required",
    "company", "company_name", "product", "product_name", "brand",
    "title", "job_title", "role", "position", "salary", "salary_raw",
    "email", "phone", "phone_number", "address", "name", "full_name",
}


def is_free_text_or_case_sensitive_field(column_name: str, semantic_type: str = "") -> bool:
    """Conservatively identify fields where changing letter case is risky."""
    name = re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower()).strip("_")
    tokens = set(name.split("_"))
    if semantic_type == "descriptive_text":
        return True
    if name in _TEXT_FIELDS_TO_PRESERVE:
        return True
    if tokens & {"description", "descriptions", "summary", "review", "reviews", "comment", "comments", "feedback", "content", "skills", "posting", "salary", "company", "brand", "product"}:
        return True
    return False


def filter_unsafe_text_normalization(actions, df: pd.DataFrame, profile: dict) -> tuple[list, list[str]]:
    """Drop normalize_text recommendations unless they are safe categorical cleanup.

    This guard is local and runs before approval. It never examines or sends cell
    contents to Gemini; it uses only local dataframe/profile metadata.
    """
    kept = []
    removed_reasons = []
    column_profiles = profile.get("column_profiles", {})
    for action in actions:
        if getattr(action, "action", None) != "normalize_text":
            kept.append(action)
            continue
        column = getattr(action, "column", None)
        if column not in df.columns:
            # The regular column validator handles invalid references.
            kept.append(action)
            continue
        details = column_profiles.get(column, {})
        semantic_type = details.get("semantic_type", "")
        series = df[column]
        if is_free_text_or_case_sensitive_field(column, semantic_type):
            removed_reasons.append(
                f"Removed normalize_text for '{column}': free text or capitalization-sensitive fields are preserved."
            )
            continue
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            removed_reasons.append(
                f"Removed normalize_text for '{column}': column is not a text/category dtype."
            )
            continue
        hints = set(details.get("semantic_hints", []))
        if "inconsistent_text_case" not in hints and "inconsistent_categories" not in hints:
            removed_reasons.append(
                f"Removed normalize_text for '{column}': local profiling did not detect casing/category inconsistency."
            )
            continue
        kept.append(action)
    return kept, removed_reasons
