"""Local category standardization without sending cell values to the AI."""
from __future__ import annotations

import difflib
import re
from typing import Any
import pandas as pd
from schemas import CleaningAction


def generate_local_category_standardizations(
    df: pd.DataFrame,
    profile: dict[str, Any],
    similarity_threshold: float = 0.88,
    max_unique_categories: int = 50,
) -> list[CleaningAction]:
    """Identify category inconsistencies locally and generate standardize_categories actions.

    Groups values by case-insensitivity and safe typo matching, mapping variants
    to the most frequent canonical spelling in the dataset.
    Never alters values differing in numbers, dates, or identifiers.
    """
    actions: list[CleaningAction] = []
    column_profiles = profile.get("column_profiles", {})

    for column in df.columns:
        col_lower = str(column).lower()
        # Skip date/time/temporal/identifier columns
        if any(term in col_lower for term in ("date", "time", "year", "month", "day", "period", "url", "id", "code", "title", "name", "desc")):
            continue

        series = df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        details = column_profiles.get(column, {})
        semantic_type = details.get("semantic_type", "")

        # Only process categorical fields, skip free-text/PII fields
        if semantic_type in {
            "descriptive_text",
            "email",
            "phone_number",
            "government_or_identity_id",
            "person_name",
            "address",
            "date_or_timestamp",
            "salary_or_income",
            "url",
        }:
            continue

        non_null = series.dropna().astype(str).str.strip()
        non_empty = non_null[non_null != ""]
        if non_empty.empty:
            continue

        unique_vals = non_empty.unique()
        if len(unique_vals) <= 1 or len(unique_vals) > max_unique_categories:
            continue

        counts = non_empty.value_counts()
        val_list = sorted(list(unique_vals), key=lambda x: counts.get(x, 0), reverse=True)

        mapping: dict[str, str] = {}
        assigned: set[str] = set()

        for canonical in val_list:
            if canonical in assigned:
                continue
            assigned.add(canonical)

            canon_digits = re.findall(r"\d+", canonical)

            for candidate in val_list:
                if candidate in assigned:
                    continue

                # Never merge categories that contain different numbers
                cand_digits = re.findall(r"\d+", candidate)
                if cand_digits != canon_digits:
                    continue

                is_case_match = candidate.lower() == canonical.lower()

                # Typo match requires similar length and high similarity
                ratio = difflib.SequenceMatcher(None, candidate.lower(), canonical.lower()).ratio()
                is_close_typo = (
                    ratio >= similarity_threshold
                    and abs(len(candidate) - len(canonical)) <= 2
                    and len(candidate) >= 4
                )

                if is_case_match or is_close_typo:
                    mapping[candidate] = canonical
                    assigned.add(candidate)

        if mapping:
            reason = (
                f"Standardize {len(mapping)} inconsistent category spelling/casing variant(s) "
                f"to their canonical representation in '{column}'."
            )
            actions.append(
                CleaningAction(
                    action="standardize_categories",
                    column=column,
                    params={"mapping": mapping},
                    reasoning=reason,
                )
            )

    return actions
