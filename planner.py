import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from actions import execute_action
from ai_privacy import build_ai_safe_profile, remap_plan_columns
from category_standardizer import generate_local_category_standardizations
from profiler import profile_dataset
from recommendation_guardrails import filter_unsafe_text_normalization
from approval_selection import labels_for_count, parse_selection
from schemas import CleaningAction, CleaningPlan


load_dotenv()


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_planner_prompt(ai_safe_profile: dict) -> str:
    return f"""
You are an expert data quality analyst and cleaning planner.

Analyze the privacy-safe dataset profile and determine which data-cleaning operations are
actually necessary to improve the dataset.

COLUMN CONTRACT (MANDATORY): The only valid column names are the exact keys in
DATASET PROFILE -> column_profiles. Copy column names character-for-character.
Never invent, infer, correct, or substitute a column name. If no safe action applies,
return an empty actions list. Before returning JSON, verify every action column exists
exactly in that key set. Original column names and locally inferred semantic
types are provided to preserve context (for example, email, phone number, or date of birth).
Never request or infer raw cell values, sample rows, or category labels. Do not use
standardize_categories: category mappings must be created locally without sending labels.

Your job is to:
1. Identify data-quality problems from the supplied profile.
2. Select appropriate supported cleaning operations.
3. Provide parameters required to execute those operations safely.
4. Avoid unnecessary transformations.
5. Never invent information that is not supported by the profile.
6. Preserve columns unless there is a clear data-quality or analysis reason to remove them.
7. A constant column (one non-null unique value) may be recommended for removal because
   it cannot distinguish records in most analyses. Explain the exact profile evidence;
   user approval is still required. Never remove it automatically.
8. Preserve descriptive_text columns by default. Sentence/description columns can be useful
   for text analytics, search, classification, sentiment, and topic analysis. Flag them as
   high-cardinality/descriptive rather than deleting them merely because they contain sentences.

You may ONLY use these actions:

- impute_missing
- drop_missing_rows
- remove_duplicates
- remove_column
- rename_column
- convert_to_date
- convert_to_numeric
- convert_to_boolean
- normalize_text
- normalize_whitespace
- normalize_column_names
- replace_empty_strings
- remove_invalid_values
- clip_outliers
- clean_phone_number

Return ONLY valid JSON.

Do NOT return Markdown.
Do NOT use code fences.
Do NOT include explanations outside the JSON.

The JSON must have exactly this structure:

{{
  "summary": "short description of the detected data-quality issues",
  "actions": [
    {{
      "action": "impute_missing",
      "column": "Age",
      "params": {{
        "method": "median"
      }},
      "reasoning": "Age contains missing numerical values and median imputation is appropriate."
    }}
  ]
}}

============================================================
ACTION RULES
============================================================

1. impute_missing

Use ONLY when the profile shows missing values.

The column must actually contain missing values. Choose the method based on data type:
- Numeric measures: median is the robust default; mean is acceptable when appropriate.
- Categorical/text/boolean columns: use mode. Never use mean or median for categories.
- Descriptive text columns: use mode only when a small number of repeated values makes
  that reasonable; otherwise recommend review or a clearly justified constant, not an
  invented sentence.
- Identifiers, email, phone, government/identity IDs, and person names should not be
  imputed with mean/median. Prefer mode only if justified, otherwise leave missing values.

The params object MUST contain one of the following:

{{
  "method": "median"
}}

or:

{{
  "method": "mean"
}}

or:

{{
  "method": "mode"
}}

or:

{{
  "method": "constant",
  "value": "..."
}}

Guidelines:

- Prefer median for numerical columns when appropriate.
- Prefer mean when the distribution makes mean appropriate.
- Prefer mode for categorical values when appropriate.
- Use constant only when the profile provides strong evidence for a suitable constant.
- Never invent a constant value.
- Do not impute columns with no missing values.
- Do not impute missing values when the column's meaning is unclear.

============================================================

2. remove_duplicates

Use ONLY when:

duplicate_rows > 0

in the dataset profile.

Use:

{{
  "params": {{}}
}}

Do not remove rows when no duplicates are detected.

============================================================

3. remove_column

Use ONLY when the profile provides strong evidence that a column is
objectively unnecessary.

The profile may contain these indicators:

- empty_column
- constant_column

Rules:

- An entirely empty column may be proposed for removal, but it still requires
  user approval.
- A constant column may be proposed for removal after considering its role and profile.
  It must still be shown to the user and approved before any change. Explain that it has
  no variation and may not help distinguish records. Preserve it if it is meaningful
  metadata, a business rule, a label, or required for traceability.
- When recommending removal of a constant column, do NOT describe it dismissively as
  "useless" or having "no value". Explicitly acknowledge its metadata, provenance, audit,
  or business meaning (e.g. data source lineage, ingestion timestamp, or uniform business flag).
  State that while it lacks variance across records in this specific dataset slice, it may
  be essential for lineage, auditing, or multi-source merging.
- Do not remove the last remaining column.
- Do not remove a column merely because it may be irrelevant to a particular
  analysis or machine-learning task.
- Do not assume identifiers, URLs, dates, names, or raw columns are useless.
- Do not remove columns based only on their names.
- Do not assume two columns are redundant just because their names or values
  appear related.
- Preserve potentially useful information by default.
- Do not remove columns when the evidence is ambiguous.

For each removal, provide the exact column name in "column".

Use:

{{
  "params": {{}}
}}

The reasoning MUST explain the profile evidence that justifies removal.

If no column can safely be removed, do not generate a remove_column action.

============================================================

4. convert_to_date

Use ONLY when the profile contains the:

"possible_date"

semantic hint.

Use:

{{
  "params": {{}}
}}

Do not convert a column to dates merely because its name looks like a date.

============================================================

5. convert_to_numeric

Use ONLY when the profile contains the:

"possible_numeric"

semantic hint.

Use:

{{
  "params": {{}}
}}

Do not convert arbitrary categorical or textual columns to numeric.

Do not convert a column marked with "possible_identifier" unless the profile
provides strong evidence that it represents a genuine numerical measurement.

============================================================

6. convert_to_boolean

Use ONLY when the profile contains the:

"possible_boolean"

semantic hint.

Use:

{{
  "params": {{}}
}}

Boolean-like values may include:

- true / false
- yes / no
- y / n
- 1 / 0

Do not convert arbitrary categorical columns to boolean.

============================================================

Category labels are intentionally withheld from the model. Do not propose category mappings.

============================================================

8. normalize_text

Use ONLY when the profile contains:

"inconsistent_text_case"

Allowed values for params.case:

- lower
- upper
- title
- none

Example:

{{
  "action": "normalize_text",
  "column": "Age",
  "params": {{
    "case": "title"
  }},
  "reasoning": "The profile shows inconsistent capitalization in text values."
}}

Choose the case that best preserves the apparent intended representation.

Do not normalize free-form text, descriptions, job postings, salary text, skill
lists, company names, product names, acronyms, identifiers, or other fields where
capitalization may carry meaning. In particular, do not blindly alter SQL, AWS,
Power BI, company branding, or raw salary descriptions. If the profile does not
provide evidence of inconsistent casing, do not propose this action.

============================================================

9. normalize_whitespace

Use ONLY when the profile contains:

"whitespace_issues"

Use:

{{
  "params": {{}}
}}

This operation removes unnecessary leading/trailing whitespace and collapses
repeated whitespace.

Do not use it when no whitespace problem is supported by the profile.

============================================================

10. normalize_column_names

Use ONLY when:

"column_name_issues"

is non-empty.

Use:

{{
  "params": {{}}
}}

Normalize names into consistent machine-friendly names.

The profile provides the original and suggested column names.

IMPORTANT:

If this action is selected, it MUST be executed before any other action that
references a renamed column.

============================================================

11. replace_empty_strings

Use ONLY when the profile contains:

"empty_strings"

If the problem is specific to one column, provide that column.

If the problem is dataset-wide, column may be null.

Use:

{{
  "params": {{}}
}}

Empty or whitespace-only strings should become missing values.

============================================================

12. remove_invalid_values

Use ONLY when the profile provides clear evidence that particular values
are invalid.

The params object MUST contain:

{{
  "invalid_values": [...]
}}

Every value in the list MUST be supported by actual profile evidence.

Do not invent invalid values.
Do NOT propose remove_invalid_values with an empty invalid_values list. If specific invalid values cannot be determined from the profile metadata, do NOT propose remove_invalid_values.

Replace invalid values with missing values.

Do not treat unusual but potentially valid values as invalid without evidence.

============================================================

13. clip_outliers

Use ONLY when the profile provides evidence of meaningful numerical outliers.

The column must be numerical.

Use:

{{
  "params": {{}}
}}

Do NOT propose clip_outliers for this project. Statistical outliers must be
flagged for human review, not automatically changed. An extreme value may be
legitimate business data, and clipping can silently corrupt it. If outliers are
present, describe them in the summary without adding a repair action.

============================================================
GENERAL SAFETY RULES
============================================================

- Every action MUST be supported by evidence in the dataset profile.
- Use the smallest set of operations necessary to improve data quality.
- Do not perform unnecessary transformations.
- Do not invent values, categories, invalid values, or missing information.
- Do not modify ambiguous data.
- Preserve valid user data whenever possible.
- Do not generate Python code.
- Do not generate SQL.
- Do not request arbitrary code execution.
- Do not use unsupported actions.
- Every action MUST include reasoning.
- Every action that operates on a column MUST specify the column.
- Parameters MUST contain everything required to safely execute the action.
- If no cleaning action is justified, return an empty actions list.
- Do not create an action simply because an operation is available.
- Do not create multiple actions that perform the same transformation on the
  same column unless each action has a distinct, justified purpose.

============================================================
ACTION ORDER
============================================================

When multiple operations are required, prefer this general order:

1. normalize_column_names
2. rename_column
3. replace_empty_strings
4. normalize_whitespace
5. normalize_text
6. convert_to_numeric
7. convert_to_boolean
8. convert_to_date
9. clean_phone_number
10. remove_invalid_values
11. impute_missing
12. drop_missing_rows
13. clip_outliers
14. remove_duplicates
15. remove_column

Only include operations that are actually necessary.

Column removal MUST occur after all other operations so that a column is not
removed before another planned action can use it.

============================================================
DATASET PROFILE
============================================================

{json.dumps(ai_safe_profile, indent=2, default=str)}

Before returning the plan, check each action's column against the exact
column_profiles keys above. If a column is absent, remove that action. Never
use column names remembered from examples or other datasets.
"""


# ============================================================
# LOCAL GUARDRAILS & PLAN SUMMARY SYNCHRONIZATION
# ============================================================

def sync_plan_summary_with_guarded_actions(plan: CleaningPlan, pruned_actions: list[dict] = None) -> CleaningPlan:
    """Ensure the displayed plan summary matches the final guarded plan actions."""
    if not plan.summary:
        return plan

    import re

    active_action_types = {act.action for act in plan.actions}
    raw_sentences = re.split(r"(?<=[.!?])\s+", plan.summary.strip())
    kept_sentences = []

    for sentence in raw_sentences:
        s_lower = sentence.lower()
        drop = False

        # If sentence is an observation or preservation statement, keep it
        is_negation_or_preservation = bool(
            re.search(r"\b(will not|do not|cannot|should not|not be|preserved without|no .* will be)\b", s_lower)
        )

        if not is_negation_or_preservation:
            # Check date conversion claim
            if "convert_to_date" not in active_action_types:
                if (
                    re.search(r"\b(date|datetime)\b", s_lower)
                    and re.search(r"\b(convert|converted|converting|conversion|proper date|parse|format)\b", s_lower)
                ):
                    drop = True

            # Check numeric conversion claim
            if "convert_to_numeric" not in active_action_types:
                if (
                    re.search(r"\b(numeric|number|float|integer)\b", s_lower)
                    and re.search(r"\b(convert|converted|converting|conversion)\b", s_lower)
                ):
                    drop = True

            # Check boolean conversion claim
            if "convert_to_boolean" not in active_action_types:
                if re.search(r"\b(convert|converted|converting|conversion)\b.*\bboolean\b", s_lower):
                    drop = True

            # Check column removal claim
            if "remove_column" not in active_action_types:
                if (
                    re.search(r"\b(remove|drop|delet)(d|ed|ing|e)?\b.*\b(column|constant)\b", s_lower)
                    or "columns can be removed" in s_lower
                    or "columns that are constant" in s_lower
                ):
                    drop = True

            # Check duplicate removal claim
            if "remove_duplicates" not in active_action_types:
                if re.search(r"\b(remove|drop|delet)(d|ed|ing|e)?\b.*\bduplicate", s_lower):
                    drop = True

            # Check text normalization claim
            if "normalize_text" not in active_action_types:
                if (
                    re.search(r"\b(normaliz|standardiz)(e|ed|ing)?\b.*\b(text|casing|case)\b", s_lower)
                    or "inconsistent casing that could benefit from normalization" in s_lower
                ):
                    drop = True

        if not drop:
            kept_sentences.append(sentence)

    if kept_sentences:
        plan.summary = " ".join(kept_sentences).strip()
    else:
        if not plan.actions:
            plan.summary = "All proposed actions were excluded by safety guardrails. Original dataset is preserved without changes."
        else:
            action_desc = [f"{act.action} on '{act.column}'" if act.column else act.action for act in plan.actions]
            plan.summary = f"Cleaning plan includes: {', '.join(action_desc)}."

    return plan


def apply_local_guardrails(plan: CleaningPlan, df: pd.DataFrame, profile: dict) -> CleaningPlan:
    pruned_actions = []

    # ============================================================
    # APPLY LOCAL TEXT-NORMALIZATION GUARDRAILS
    # ============================================================
    initial_actions_count = len(plan.actions)
    plan.actions, text_guardrail_messages = filter_unsafe_text_normalization(
        plan.actions, df, profile
    )
    for guardrail_message in text_guardrail_messages:
        print(f"Safety guardrail: {guardrail_message}")
        pruned_actions.append({"action": "normalize_text", "reason": guardrail_message})

    # ============================================================
    # APPLY LOCAL IMPUTATION-METHOD GUARDRAILS
    # ============================================================
    for action in plan.actions:
        if action.action != "impute_missing" or action.column not in df.columns:
            continue
        series = df[action.column]
        semantic_type = profile.get("column_profiles", {}).get(action.column, {}).get("semantic_type", "")
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        requested_method = action.params.get("method", "median")
        if not is_numeric and requested_method in {"mean", "median"}:
            action.params["method"] = "mode"
            action.reasoning += " Local safety rule changed imputation to mode because this column is categorical/text or boolean."
        elif is_numeric and requested_method == "mode":
            pass
        if semantic_type in {"email", "phone_number", "government_or_identity_id", "person_name", "address"} and action.params.get("method") in {"mean", "median"}:
            action.params["method"] = "mode"
            action.reasoning += " Local semantic rule prevents mean/median imputation for this identifier/contact field."

    # ============================================================
    # AUDIT OTHER PROPOSED ACTIONS (DATES, NUMERICS, COLUMNS, INVALID VALUES)
    # ============================================================
    audited_actions = []
    column_profiles = profile.get("column_profiles", {})

    for action in plan.actions:
        col = action.column
        details = column_profiles.get(col, {}) if col else {}
        hints = set(details.get("semantic_hints", []))

        if action.action == "convert_to_date":
            if col not in df.columns:
                pruned_actions.append({"action": "convert_to_date", "column": col, "reason": "Column not in dataframe"})
                continue
            series = df[col].dropna().astype(str).str.strip()
            non_empty = series[series != ""]
            if non_empty.empty:
                print(f"Safety guardrail: Removed convert_to_date for '{col}': column is empty.")
                pruned_actions.append({"action": "convert_to_date", "column": col, "reason": "Column is empty"})
                continue
            if "possible_date" not in hints:
                parsed = pd.to_datetime(non_empty, format="mixed", errors="coerce")
                if parsed.notna().mean() < 0.5:
                    print(f"Safety guardrail: Removed convert_to_date for '{col}': values are not parseable dates.")
                    pruned_actions.append({"action": "convert_to_date", "column": col, "reason": "Values are not parseable dates"})
                    continue

        elif action.action == "convert_to_numeric":
            if col not in df.columns:
                pruned_actions.append({"action": "convert_to_numeric", "column": col, "reason": "Column not in dataframe"})
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                continue
            series = df[col].dropna().astype(str).str.strip()
            cleaned = (
                series.str.replace(r"[$€£¥₹%]", "", regex=True)
                .str.replace(r"(?i)\b(usd|eur|gbp|inr)\b", "", regex=True)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            parsed = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().mean() < 0.5:
                print(f"Safety guardrail: Removed convert_to_numeric for '{col}': values cannot be converted to numeric.")
                pruned_actions.append({"action": "convert_to_numeric", "column": col, "reason": "Values cannot be converted to numeric"})
                continue

        elif action.action == "remove_column":
            if col not in df.columns:
                pruned_actions.append({"action": "remove_column", "column": col, "reason": "Column not in dataframe"})
                continue
            if len(df.columns) <= 1:
                print(f"Safety guardrail: Preserved column '{col}': cannot remove the last remaining column.")
                pruned_actions.append({"action": "remove_column", "column": col, "reason": "Cannot remove the last remaining column"})
                continue
            is_empty = details.get("empty_column", False)
            is_const = details.get("constant_column", False)
            if not is_empty and not is_const:
                print(f"Safety guardrail: Preserved column '{col}': column is neither empty nor constant.")
                pruned_actions.append({"action": "remove_column", "column": col, "reason": "Column is neither empty nor constant"})
                continue

            # Improve constant column wording and safety checks (Finding 3)
            if is_const:
                col_lower = str(col).lower()
                provenance_keywords = {
                    "source", "origin", "provider", "scraped", "scrape",
                    "timestamp", "created", "updated", "batch", "run_id", "version", "audit"
                }
                if any(k in col_lower for k in provenance_keywords) or details.get("semantic_type") == "date_or_timestamp":
                    action.reasoning = (
                        f"The '{col}' column is constant with a single value across all records. Although it contains zero variance "
                        f"and does not differentiate records in this dataset slice, it represents data provenance/audit metadata. "
                        f"Removal is optional and recommended only if record origin/lineage is unneeded for downstream analysis."
                    )
                elif details.get("semantic_type") == "possible_boolean" or "bool" in str(details.get("dtype", "")).lower() or col_lower.startswith(("is_", "has_", "salary_")):
                    action.reasoning = (
                        f"The '{col}' column has a single uniform value across all records. While it offers no variance for distinguishing "
                        f"records within this dataset slice, it represents a uniform business attribute or flag. Removal is proposed "
                        f"for analysis simplification only if this fixed business rule attribute is not needed downstream."
                    )
                else:
                    dismissive_phrases = ["not useful", "useless", "adds no value", "provides no value", "no discriminative value"]
                    if any(phrase in action.reasoning.lower() for phrase in dismissive_phrases):
                        action.reasoning = (
                            f"The '{col}' column contains a single constant value across all records. It does not provide variance "
                            f"for distinguishing records in this dataset, but may represent fixed context or metadata. Removal is "
                            f"optional depending on whether this context is needed downstream."
                        )

        elif action.action == "remove_invalid_values":
            if col not in df.columns:
                pruned_actions.append({"action": "remove_invalid_values", "column": col, "reason": "Column not in dataframe"})
                continue
            inv_vals = action.params.get("invalid_values", [])
            if not inv_vals:
                print(f"Safety guardrail: Removed remove_invalid_values for '{col}': no specific invalid values provided.")
                pruned_actions.append({"action": "remove_invalid_values", "column": col, "reason": "No specific invalid values provided"})
                continue
            if not df[col].isin(inv_vals).any():
                print(f"Safety guardrail: Removed remove_invalid_values for '{col}': specified invalid values do not exist in column.")
                pruned_actions.append({"action": "remove_invalid_values", "column": col, "reason": "Specified invalid values do not exist in column"})
                continue

        audited_actions.append(action)

    plan.actions = audited_actions

    # ============================================================
    # PROPOSE LOCAL CATEGORY HARMONIZATION
    # ============================================================
    try:
        local_category_actions = generate_local_category_standardizations(df, profile)
        for cat_action in local_category_actions:
            if not any(a.action == "standardize_categories" and a.column == cat_action.column for a in plan.actions):
                plan.actions.append(cat_action)
                print(f"Local category assistant: proposed {cat_action.action} for '{cat_action.column}'")
    except Exception as err:
        print(f"Notice: local category harmonization skipped: {err}")

    # ============================================================
    # SYNC PLAN SUMMARY WITH FINAL GUARDED ACTIONS
    # ============================================================
    sync_plan_summary_with_guarded_actions(plan, pruned_actions)

    return plan


# ============================================================
# EXECUTE AI-GENERATED ACTIONS
# ============================================================

def execute_cleaning_plan(df: pd.DataFrame, actions: list) -> tuple[pd.DataFrame, list[dict]]:
    executed_actions = []
    column_aliases = {column: column for column in df.columns}

    # Column-name normalization must happen before column-specific actions.
    sorted_actions = sorted(
        actions,
        key=lambda action: (action.action != "normalize_column_names")
    )

    for action in sorted_actions:
        resolved_column = action.column
        if resolved_column is not None:
            resolved_column = column_aliases.get(resolved_column, resolved_column)

        try:
            before_rows = len(df)
            before_missing = df.isna().sum().to_dict()
            before_dtypes = df.dtypes.astype(str).to_dict()
            before_columns = list(df.columns)

            df.attrs.pop("values_replaced", None)
            df.attrs.pop("rows_dropped", None)

            original_action_column = action.column
            if resolved_column is not None:
                action.column = resolved_column

            df = execute_action(df, action)

            values_replaced = int(df.attrs.pop("values_replaced", 0) or 0)
            rows_dropped = int(df.attrs.pop("rows_dropped", 0) or 0)

            if action.action != "normalize_column_names" and original_action_column is not None:
                action.column = resolved_column

            after_rows = len(df)
            after_missing = df.isna().sum().to_dict()
            after_dtypes = df.dtypes.astype(str).to_dict()
            after_columns = list(df.columns)

            result = {
                "id": action.id,
                "action": action.action,
                "column": action.column,
                "params": action.params,
                "reasoning": action.reasoning,
                "status": "success",
            }

            if action.action == "impute_missing":
                before = before_missing.get(action.column, 0)
                after = after_missing.get(action.column, 0)
                result["result"] = {
                    "missing_before": before,
                    "missing_after": after,
                    "missing_values_filled": max(0, before - after)
                }

            elif action.action == "drop_missing_rows":
                result["result"] = {
                    "rows_before": before_rows,
                    "rows_after": after_rows,
                    "rows_dropped": before_rows - after_rows
                }

            elif action.action == "remove_duplicates":
                result["result"] = {
                    "rows_before": before_rows,
                    "rows_after": after_rows,
                    "duplicates_removed": before_rows - after_rows
                }

            elif action.action == "convert_to_date":
                result["result"] = {
                    "dtype_before": before_dtypes.get(action.column),
                    "dtype_after": after_dtypes.get(action.column)
                }

            elif action.action == "convert_to_numeric":
                result["result"] = {
                    "dtype_before": before_dtypes.get(action.column),
                    "dtype_after": after_dtypes.get(action.column)
                }

            elif action.action == "convert_to_boolean":
                result["result"] = {
                    "dtype_before": before_dtypes.get(action.column),
                    "dtype_after": after_dtypes.get(action.column)
                }

            elif action.action == "clean_phone_number":
                result["result"] = {
                    "column": action.column
                }

            elif action.action == "standardize_categories":
                result["result"] = {
                    "mapping": action.params.get("mapping", {})
                }

            elif action.action == "normalize_text":
                result["result"] = {
                    "case": action.params.get("case", "lower")
                }

            elif action.action == "normalize_whitespace":
                result["result"] = {
                    "column": action.column
                }

            elif action.action == "rename_column":
                new_name = action.params.get("new_name")
                if new_name and action.column:
                    column_aliases[action.column] = new_name
                result["result"] = {
                    "columns_before": before_columns,
                    "columns_after": after_columns,
                    "new_name": new_name
                }

            elif action.action == "normalize_column_names":
                result["result"] = {
                    "columns_before": before_columns,
                    "columns_after": after_columns
                }
                if len(before_columns) == len(after_columns):
                    column_aliases = {
                        old: new
                        for old, new in zip(before_columns, after_columns)
                    }

            elif action.action == "replace_empty_strings":
                result["result"] = {
                    "column": action.column,
                    "values_replaced": values_replaced
                }

            elif action.action == "remove_invalid_values":
                result["result"] = {
                    "invalid_values": action.params.get("invalid_values", []),
                    "values_replaced": values_replaced
                }

            elif action.action == "clip_outliers":
                result["result"] = {
                    "column": action.column
                }

            elif action.action == "remove_column":
                result["result"] = {
                    "columns_before": before_columns,
                    "columns_after": after_columns,
                    "column_removed": (action.column not in after_columns)
                }

            executed_actions.append(result)

        except Exception as error:
            executed_actions.append({
                "id": action.id,
                "action": action.action,
                "column": resolved_column,
                "params": action.params,
                "reasoning": action.reasoning,
                "status": "failed",
                "error": str(error)
            })
            print(f"Action failed: {action.action} on column '{resolved_column}': {error}")

    return df, executed_actions


# ============================================================
# VALIDATE CLEANING RESULTS
# ============================================================

def validate_cleaning_results(
    df_before: pd.DataFrame,
    df: pd.DataFrame,
    executed_actions: list[dict],
    original_missing: dict[str, int],
    cleaned_missing: dict[str, int],
) -> dict:
    original_rows = len(df_before)
    cleaned_rows = len(df)

    successful_actions = [
        action for action in executed_actions if action["status"] == "success"
    ]
    failed_actions = [
        action for action in executed_actions if action["status"] == "failed"
    ]

    duplicate_removal_requested = any(
        action["action"] == "remove_duplicates" for action in executed_actions
    )
    row_drop_requested = any(
        action["action"] == "drop_missing_rows" for action in executed_actions
    )

    duplicates_before = int(df_before.duplicated().sum())
    duplicates_after = int(df.duplicated().sum())

    if duplicate_removal_requested or row_drop_requested:
        rows_valid = (
            (not duplicate_removal_requested or duplicates_after == 0)
            and cleaned_rows <= original_rows
        )
    else:
        rows_valid = cleaned_rows == original_rows

    action_results_valid = True

    for action in executed_actions:
        if action["status"] != "success":
            action_results_valid = False
            continue

        result = action.get("result", {})
        action_name = action["action"]

        if action_name == "impute_missing":
            if result.get("missing_after", 0) != 0:
                action_results_valid = False
            if result.get("missing_values_filled", 0) < 0:
                action_results_valid = False

        elif action_name == "drop_missing_rows":
            if result.get("rows_after", 0) > result.get("rows_before", 0):
                action_results_valid = False
            if result.get("rows_dropped", 0) < 0:
                action_results_valid = False

        elif action_name == "remove_duplicates":
            if result.get("rows_after", 0) > result.get("rows_before", 0):
                action_results_valid = False
            if duplicates_after != 0:
                action_results_valid = False
            if result.get("duplicates_removed", 0) < 0:
                action_results_valid = False

        elif action_name == "convert_to_date":
            dtype_after = result.get("dtype_after", "")
            if "datetime" not in dtype_after.lower():
                action_results_valid = False

        elif action_name == "convert_to_numeric":
            dtype_after = result.get("dtype_after", "")
            if not any(numeric_type in dtype_after.lower() for numeric_type in ["int", "float", "decimal"]):
                action_results_valid = False

        elif action_name == "convert_to_boolean":
            dtype_after = result.get("dtype_after", "")
            if "bool" not in dtype_after.lower():
                action_results_valid = False

        elif action_name == "clean_phone_number":
            if action.get("column") not in df.columns:
                action_results_valid = False

        elif action_name == "standardize_categories":
            mapping = result.get("mapping", {})
            if not isinstance(mapping, dict) or not mapping:
                action_results_valid = False

        elif action_name == "rename_column":
            new_name = action.get("params", {}).get("new_name")
            if not new_name or new_name not in df.columns:
                action_results_valid = False

        elif action_name == "normalize_column_names":
            columns_after = result.get("columns_after", [])
            if not columns_after or len(columns_after) != len(set(columns_after)):
                action_results_valid = False

        elif action_name == "replace_empty_strings":
            values_replaced = result.get("values_replaced")
            if not isinstance(values_replaced, int) or values_replaced < 0:
                action_results_valid = False
            column = result.get("column")
            if column is not None and column not in df.columns:
                action_results_valid = False

        elif action_name == "remove_invalid_values":
            invalid_values = result.get("invalid_values", [])
            values_replaced = result.get("values_replaced")
            if not isinstance(invalid_values, list) or not invalid_values:
                action_results_valid = False
            if not isinstance(values_replaced, int) or values_replaced < 0:
                action_results_valid = False
            column = action.get("column")
            if column is not None and column not in df.columns:
                action_results_valid = False

        elif action_name == "clip_outliers":
            if action["column"] not in df.columns:
                action_results_valid = False

        elif action_name == "remove_column":
            column_removed = result.get("column_removed", False)
            if not column_removed or action["column"] in df.columns:
                action_results_valid = False

    missing_values_before = int(sum(original_missing.values()))
    missing_values_after = int(sum(cleaned_missing.values()))

    intentional_missing_increase = 0
    for action in executed_actions:
        if action["status"] != "success":
            continue
        action_name = action["action"]
        result = action.get("result", {})
        if action_name in ["remove_invalid_values", "replace_empty_strings"]:
            values_replaced = result.get("values_replaced", 0)
            if isinstance(values_replaced, int) and values_replaced > 0:
                intentional_missing_increase += values_replaced

    unexpected_missing_increase = (
        missing_values_after > missing_values_before + intentional_missing_increase
    )
    missing_values_reduced = missing_values_after < missing_values_before
    missing_values_improved_or_unchanged = not unexpected_missing_increase
    all_actions_succeeded = len(failed_actions) == 0

    # Distinguish approved imputation status from unresolved missing values (Finding 1)
    approved_imputations = [a for a in executed_actions if a["action"] == "impute_missing"]
    approved_imputations_succeeded = all(
        a["status"] == "success" and a.get("result", {}).get("missing_after", 0) == 0
        for a in approved_imputations
    ) if approved_imputations else True

    missing_reduction_requested = any(
        a["action"] in ["impute_missing", "drop_missing_rows"] for a in executed_actions
    )

    unresolved_missing_columns = {
        col: count for col, count in cleaned_missing.items() if count > 0
    }

    validation = {
        "rows_valid": rows_valid,
        "missing_values_reduced": missing_values_reduced,
        "missing_reduction_requested": missing_reduction_requested,
        "missing_values_improved_or_unchanged": missing_values_improved_or_unchanged,
        "approved_imputations_succeeded": approved_imputations_succeeded,
        "unexpected_missing_increase": unexpected_missing_increase,
        "intentional_missing_increase": intentional_missing_increase,
        "action_results_valid": action_results_valid,
        "all_actions_succeeded": all_actions_succeeded,
        "successful_actions": len(successful_actions),
        "failed_actions": len(failed_actions),
        "duplicates_before": duplicates_before,
        "duplicates_after": duplicates_after,
        "unresolved_missing_values_count": missing_values_after,
        "unresolved_missing_columns": unresolved_missing_columns,
    }

    validation["overall"] = all([
        validation["rows_valid"],
        validation["missing_values_improved_or_unchanged"],
        validation["action_results_valid"],
        validation["all_actions_succeeded"],
        validation["approved_imputations_succeeded"],
    ])

    return validation


# ============================================================
# BUILD CLEANING REPORT & RECORD UNRESOLVED ISSUES
# ============================================================

def build_cleaning_report(
    run_id: str,
    input_file: str,
    summary: str,
    executed_actions: list[dict],
    validation: dict,
    df_before: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    profile: dict,
    original_missing: dict[str, int],
    cleaned_missing: dict[str, int],
) -> dict:
    original_rows = len(df_before)
    cleaned_rows = len(df_cleaned)

    all_columns = list(
        dict.fromkeys(
            list(original_missing.keys())
            + list(cleaned_missing.keys())
        )
    )

    missing_value_report = {}
    for column in all_columns:
        before = original_missing.get(column, 0)
        after = cleaned_missing.get(column, 0)
        if before > 0 or after > 0:
            missing_value_report[column] = {
                "before": before,
                "after": after,
                "status": "resolved" if after == 0 else ("reduced" if after < before else ("unresolved" if after == before else "increased")),
            }

    duplicates_before = int(df_before.duplicated().sum())
    duplicates_after = int(df_cleaned.duplicated().sum())
    successful_actions = [a for a in executed_actions if a.get("status") == "success"]
    failed_actions = [a for a in executed_actions if a.get("status") == "failed"]

    # Record unresolved data-quality issues without inventing values (Finding 4)
    unresolved_issues = []
    column_profiles = profile.get("column_profiles", {}) if profile else {}

    # 1. Unresolved missing values (URLs, contacts, identifiers, or unapproved)
    for col, count in cleaned_missing.items():
        if count > 0:
            sem_type = column_profiles.get(col, {}).get("semantic_type", "")
            if sem_type == "url" or "url" in col.lower():
                detail = f"{count} missing URL values preserved without synthetic imputation to avoid inventing fictitious web addresses."
            elif sem_type in {"email", "phone_number", "government_or_identity_id", "person_name"}:
                detail = f"{count} missing contact/identity values preserved without synthetic imputation to prevent fabricating personal data."
            else:
                detail = f"{count} missing values remain unresolved; no imputation was approved for this column."
            unresolved_issues.append({
                "issue_type": "missing_values",
                "column": col,
                "count": count,
                "status": "unresolved_preserved",
                "details": detail,
            })

    # 2. Flagged numerical outliers (preserved for human review)
    for col, details in column_profiles.items():
        if col in df_cleaned.columns:
            stats = details.get("statistics", {})
            outlier_count = stats.get("outlier_count", 0)
            if outlier_count > 0:
                unresolved_issues.append({
                    "issue_type": "numerical_outliers",
                    "column": col,
                    "count": outlier_count,
                    "status": "flagged_for_review",
                    "details": f"{outlier_count} statistical outlier(s) detected via IQR; preserved without clipping to avoid silent data corruption."
                })

    # 3. Raw relative dates
    for col, details in column_profiles.items():
        if col in df_cleaned.columns:
            hints = set(details.get("semantic_hints", []))
            if "possible_date" in hints or "date" in col.lower():
                if not pd.api.types.is_datetime64_any_dtype(df_cleaned[col]):
                    unresolved_issues.append({
                        "issue_type": "relative_or_unparsed_dates",
                        "column": col,
                        "status": "preserved_raw",
                        "details": f"Raw date/temporal strings in '{col}' preserved in original format; relative or unstructured dates cannot be safely parsed without reference context."
                    })

    # 4. Inconsistent text casing / categories not approved or pruned
    executed_col_actions = {(a.get("column"), a.get("action")) for a in executed_actions}
    for col, details in column_profiles.items():
        if col in df_cleaned.columns:
            hints = set(details.get("semantic_hints", []))
            if "inconsistent_text_case" in hints and (col, "normalize_text") not in executed_col_actions:
                unresolved_issues.append({
                    "issue_type": "inconsistent_casing",
                    "column": col,
                    "status": "preserved_raw",
                    "details": f"Text casing variations in '{col}' preserved to protect acronyms, proper names, or domain terminology from destructive normalization."
                })

    report = {
        "run_id": run_id,
        "input_file": input_file,
        "summary": summary,
        "actions": executed_actions,
        "validation": validation,
        "rows": {
            "before": original_rows,
            "after": cleaned_rows
        },
        "missing_values": missing_value_report,
        "unresolved_issues": unresolved_issues,
        "monitoring": {
            "rows_changed": original_rows != cleaned_rows,
            "missing_values_before": int(sum(original_missing.values())),
            "missing_values_after": int(sum(cleaned_missing.values())),
            "missing_values_delta": int(sum(cleaned_missing.values()) - sum(original_missing.values())),
            "unresolved_missing_values": int(sum(cleaned_missing.values())),
            "unresolved_issues_count": len(unresolved_issues),
            "duplicate_rows_before": int(duplicates_before),
            "duplicate_rows_after": int(duplicates_after),
            "successful_actions": len(successful_actions),
            "failed_actions": len(failed_actions),
            "overall_validation_passed": bool(validation["overall"]),
        }
    }
    return report


# ============================================================
# PLAN GENERATION WITH BOUNDED RETRIES
# ============================================================

def generate_validated_plan(
    client: genai.Client,
    initial_prompt: str,
    token_to_column: dict[str, str],
    max_retries: int = 2,
) -> tuple[CleaningPlan, str]:
    """Request cleaning plan with bounded retries and validation feedback."""
    current_prompt = initial_prompt
    last_error = None
    response_text = ""
    raw_plan_data = None

    for attempt in range(max_retries + 1):
        try:
            try:
                # First attempt with Gemini structured outputs
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=current_prompt,
                    config=types.GenerateContentConfig(
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                        response_mime_type="application/json",
                        response_schema=CleaningPlan,
                    ),
                )
            except Exception:
                # Fallback to standard request
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=current_prompt,
                    config=types.GenerateContentConfig(
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        )
                    ),
                )
        except Exception as error:
            if attempt < max_retries and "503" in str(error):
                time.sleep(2 * (attempt + 1))
                continue
            raise error

        response_text = response.text.strip() if response.text else ""
        if not response_text:
            if attempt < max_retries:
                current_prompt = (
                    f"{initial_prompt}\n\n"
                    "CRITICAL: The previous model response was empty. "
                    "Return ONLY valid JSON matching the CleaningPlan schema."
                )
                continue
            raise ValueError("Gemini returned an empty response")

        try:
            raw_plan_data = json.loads(response_text)
            local_plan_data = remap_plan_columns(raw_plan_data, token_to_column)
            plan = CleaningPlan.model_validate(local_plan_data)
            return plan, response_text
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < max_retries:
                print(f"Plan validation failed (attempt {attempt + 1}/{max_retries + 1}): {error}")
                print("Requesting correction with validation feedback...")
                current_prompt = (
                    f"{initial_prompt}\n\n"
                    f"CRITICAL VALIDATION FEEDBACK FROM PREVIOUS ATTEMPT:\n"
                    f"Your previous response failed schema validation with error:\n{error}\n\n"
                    f"Mandatory corrections:\n"
                    f"1. If proposing 'remove_invalid_values', 'params.invalid_values' MUST be a non-empty list of specific values supported by the profile. If specific invalid values cannot be determined from the profile metadata, do NOT propose 'remove_invalid_values'.\n"
                    f"2. Every column referenced MUST exist character-for-character in the profile.\n"
                    f"3. Return ONLY valid JSON conforming to the CleaningPlan schema."
                )
            else:
                # Retries exhausted: attempt safe recovery by pruning invalid actions if possible
                try:
                    if isinstance(raw_plan_data, dict) and isinstance(raw_plan_data.get("actions"), list):
                        valid_actions = []
                        for act in raw_plan_data["actions"]:
                            try:
                                valid_act = CleaningAction.model_validate(act)
                                valid_actions.append(valid_act)
                            except Exception as act_err:
                                print(f"Pruned unrecoverable invalid action {act.get('action')} on {act.get('column')}: {act_err}")
                        if valid_actions or raw_plan_data.get("summary"):
                            plan = CleaningPlan(
                                summary=raw_plan_data.get("summary", "Cleaning plan"),
                                actions=valid_actions,
                            )
                            return plan, response_text
                except Exception:
                    pass
                raise last_error

    raise last_error


# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AI-powered data cleaning pipeline"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="data/Messy_Employee_dataset.csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip the interactive approval gate and execute the validated plan",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional destination path for the cleaned CSV file",
    )
    args = parser.parse_args(argv)
    input_file = args.input_file

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run: {run_id}")

    # ============================================================
    # CONNECT TO GEMINI
    # ============================================================
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is not configured.")
        error_report = {
            "run_id": run_id,
            "status": "failed",
            "stage": "configuration",
            "error": "GEMINI_API_KEY is not configured"
        }
        with open(os.path.join(run_dir, "error_report.json"), "w", encoding="utf-8") as file:
            json.dump(error_report, file, indent=2)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # ============================================================
    # PROFILE DATASET
    # ============================================================
    try:
        df, profile = profile_dataset(input_file)
    except Exception as error:
        print(f"Dataset profiling failed: {error}")
        error_report = {
            "run_id": run_id,
            "status": "failed",
            "stage": "profiling",
            "input_file": input_file,
            "error": str(error)
        }
        with open(os.path.join(run_dir, "error_report.json"), "w", encoding="utf-8") as file:
            json.dump(error_report, file, indent=2)
        sys.exit(1)

    profile_path = os.path.join(run_dir, "profile.json")
    with open(profile_path, "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    # ============================================================
    # CREATE PRIVACY-SAFE GEMINI PROMPT
    # ============================================================
    ai_safe_profile, token_to_column = build_ai_safe_profile(profile)
    prompt = build_planner_prompt(ai_safe_profile)

    # ============================================================
    # ASK GEMINI FOR CLEANING PLAN (WITH BOUNDED RETRIES)
    # ============================================================
    try:
        plan, response_text = generate_validated_plan(
            client=client,
            initial_prompt=prompt,
            token_to_column=token_to_column,
            max_retries=2,
        )
    except Exception as error:
        print(f"Gemini planning failed: {error}")
        error_report = {
            "run_id": run_id,
            "status": "failed",
            "stage": "planning",
            "input_file": input_file,
            "error": str(error)
        }
        with open(os.path.join(run_dir, "error_report.json"), "w", encoding="utf-8") as file:
            json.dump(error_report, file, indent=2)
        sys.exit(1)

    raw_plan_path = os.path.join(run_dir, "plan_raw.txt")
    with open(raw_plan_path, "w", encoding="utf-8") as file:
        file.write(response_text)

    # Apply guardrails
    plan = apply_local_guardrails(plan, df, profile)

    plan_path = os.path.join(run_dir, "plan.json")
    with open(plan_path, "w", encoding="utf-8") as file:
        json.dump(plan.model_dump(), file, indent=2)

    # ============================================================
    # APPROVE INDIVIDUAL RECOMMENDATIONS
    # ============================================================
    print("\nProposed cleaning plan:")
    print("Explanation:", plan.summary)

    all_actions = list(plan.actions)
    labels = labels_for_count(len(all_actions))

    if not all_actions:
        print("No cleaning actions were recommended. Nothing to execute.")
        with open(os.path.join(run_dir, "approval.json"), "w", encoding="utf-8") as file:
            json.dump({
                "run_id": run_id,
                "status": "no_actions",
                "stage": "approve",
                "selected": [],
                "rejected": [],
            }, file, indent=2)
        sys.exit(0)

    for label, action in zip(labels, all_actions):
        print(f"\n[{label}] {action.action} | column={action.column!r} | params={action.params}")
        print(f"    Reason: {action.reasoning}")

    if args.auto_approve:
        selected_labels = set(labels)
        selection_command = "ALL"
    else:
        print("\nSelect recommendations to execute:")
        print("  Enter letters to select individual actions (e.g., ABC or A,C)")
        print("  ALL     Approve all recommendations")
        print("  NONE    Reject all; leave the original dataset unchanged")
        print("  Q       Quit without changes")

        while True:
            try:
                raw_selection = input("\nYour choice: ")
            except EOFError:
                raw_selection = "Q"
            try:
                parsed_selection = parse_selection(raw_selection, labels)
            except ValueError as error:
                print(f"Invalid selection: {error}")
                continue

            if parsed_selection == "Q":
                approval_report = {
                    "run_id": run_id,
                    "status": "quit",
                    "stage": "approve",
                    "selection": raw_selection.strip(),
                    "selected": [],
                    "rejected": labels,
                    "message": "User quit before applying any repair.",
                    "plan_path": plan_path,
                }
                with open(os.path.join(run_dir, "approval.json"), "w", encoding="utf-8") as file:
                    json.dump(approval_report, file, indent=2)
                print("Exited without changes. Original dataset remains unchanged.")
                sys.exit(0)

            if parsed_selection == "NONE":
                approval_report = {
                    "run_id": run_id,
                    "status": "not_approved",
                    "stage": "approve",
                    "selection": "NONE",
                    "selected": [],
                    "rejected": labels,
                    "message": "No actions selected; original dataset remains unchanged.",
                    "plan_path": plan_path,
                }
                with open(os.path.join(run_dir, "approval.json"), "w", encoding="utf-8") as file:
                    json.dump(approval_report, file, indent=2)
                print("No actions selected. Original dataset remains unchanged.")
                sys.exit(0)

            if parsed_selection == "ALL":
                selected_labels = set(labels)
                selection_command = "ALL"
            else:
                selected_labels = set(parsed_selection)
                selection_command = ",".join(label for label in labels if label in selected_labels)
            break

    selected_actions = [
        action for label, action in zip(labels, all_actions)
        if label in selected_labels
    ]

    plan.actions = selected_actions
    approval_report = {
        "run_id": run_id,
        "status": "approved",
        "stage": "approve",
        "selection": selection_command,
        "selected": [
            {"label": label, "action": action.action, "column": action.column, "params": action.params}
            for label, action in zip(labels, all_actions) if label in selected_labels
        ],
        "rejected": [
            {"label": label, "action": action.action, "column": action.column, "params": action.params}
            for label, action in zip(labels, all_actions) if label not in selected_labels
        ],
        "plan_path": plan_path,
    }
    with open(os.path.join(run_dir, "approval.json"), "w", encoding="utf-8") as file:
        json.dump(approval_report, file, indent=2)

    print(f"\nSelected {len(selected_actions)} of {len(all_actions)} recommendation(s).")
    print("Only selected actions will be executed.")

    # ============================================================
    # CAPTURE BEFORE-CLEANING STATE
    # ============================================================
    original_rows = len(df)
    original_missing = df.isna().sum().to_dict()
    df_before = df.copy()

    print("\nBefore cleaning:")
    print("Rows:", len(df))
    print("Missing values by column:")
    print(df.isna().sum())
    print("\nData types:")
    print(df.dtypes)

    # ============================================================
    # EXECUTE APPROVED ACTIONS
    # ============================================================
    df, executed_actions = execute_cleaning_plan(df, plan.actions)

    cleaned_rows = len(df)
    cleaned_missing = df.isna().sum().to_dict()

    # ============================================================
    # VALIDATE CLEANING RESULTS
    # ============================================================
    validation = validate_cleaning_results(
        df_before=df_before,
        df=df,
        executed_actions=executed_actions,
        original_missing=original_missing,
        cleaned_missing=cleaned_missing,
    )

    report = build_cleaning_report(
        run_id=run_id,
        input_file=input_file,
        summary=plan.summary,
        executed_actions=executed_actions,
        validation=validation,
        df_before=df_before,
        df_cleaned=df,
        profile=profile,
        original_missing=original_missing,
        cleaned_missing=cleaned_missing,
    )

    print("\nCleaning report:")
    print("\nValidation:")
    for key, value in validation.items():
        print(f"  {key}: {value}")

    print("\nRows:")
    print(f"  Before: {original_rows}")
    print(f"  After:  {cleaned_rows}")

    print("\nAfter cleaning:")
    print("Rows:", len(df))
    print("Missing values by column:")
    print(df.isna().sum())
    print("\nData types:")
    print(df.dtypes)

    cleaned_data_path = args.output if args.output else os.path.join(run_dir, "cleaned_data.csv")
    df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned dataset saved to {cleaned_data_path}")

    report_path = os.path.join(run_dir, "cleaning_report.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Cleaning report saved to {report_path}")


if __name__ == "__main__":
    main()
