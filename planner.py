import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from actions import execute_action
from profiler import profile_dataset
from schemas import CleaningPlan


load_dotenv()


# ============================================================
# INPUT CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(
    description="AI-powered data cleaning pipeline"
)

parser.add_argument(
    "input_file",
    nargs="?",
    default="data/Messy_Employee_dataset.csv",
    help="Path to the input CSV file"
)

args = parser.parse_args()
input_file = args.input_file


# ============================================================
# RUN SETUP
# ============================================================

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

    with open(
        os.path.join(run_dir, "error_report.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(error_report, file, indent=2)

    raise SystemExit(1)


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

    with open(
        os.path.join(run_dir, "error_report.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(error_report, file, indent=2)

    raise SystemExit(1)


# Save profile
profile_path = os.path.join(
    run_dir,
    "profile.json"
)

with open(
    profile_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        profile,
        file,
        indent=2
    )


# ============================================================
# CREATE GEMINI PROMPT
# ============================================================
prompt = f"""
You are an expert data quality analyst and cleaning planner.

Analyze the dataset profile and determine which data-cleaning operations are
actually necessary to improve the dataset.

Your job is to:
1. Identify data-quality problems from the supplied profile.
2. Select appropriate supported cleaning operations.
3. Provide parameters required to execute those operations safely.
4. Avoid unnecessary transformations.
5. Never invent information that is not supported by the profile.

You may ONLY use these actions:

- impute_missing
- remove_duplicates
- convert_to_date
- convert_to_numeric
- convert_to_boolean
- standardize_categories
- normalize_text
- normalize_whitespace
- normalize_column_names
- replace_empty_strings
- remove_invalid_values
- clip_outliers

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
      "reasoning": "Age contains missing numerical values and median imputation is appropriate.",
      "confidence": 0.95
    }}
  ]
}}

============================================================
ACTION RULES
============================================================

1. impute_missing

Use ONLY when the profile shows missing values.

The column must actually contain missing values.

The params object MUST contain:

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

3. convert_to_date

Use ONLY when the profile contains the:

"possible_date"

semantic hint.

Use:

{{
  "params": {{}}
}}

Do not convert a column to dates merely because its name looks like a date.

============================================================

4. convert_to_numeric

Use ONLY when the profile contains the:

"possible_numeric"

semantic hint.

Use:

{{
  "params": {{}}
}}

Do not convert arbitrary categorical or textual columns to numeric.

============================================================

5. convert_to_boolean

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

6. standardize_categories

Use ONLY when the profile contains:

"inconsistent_categories"

The mapping MUST contain actual values supported by the profile.

The mapping MUST be non-empty.

Example:

{{
  "action": "standardize_categories",
  "column": "Membership_Status",
  "params": {{
    "mapping": {{
      "Active": "active",
      "ACTIVE": "active",
      "Inactive": "inactive",
      "INACTIVE": "inactive"
    }}
  }},
  "reasoning": "The profile shows multiple casing variants representing the same categories.",
  "confidence": 0.99
}}

Rules:

- Never invent categories.
- Use actual values from the profile.
- Do not create an empty mapping.
- Do not standardize categories when there is no evidence of inconsistency.

============================================================

7. normalize_text

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
  "column": "Customer_Name",
  "params": {{
    "case": "title"
  }},
  "reasoning": "The profile shows inconsistent capitalization in text values.",
  "confidence": 0.90
}}

Choose the case that best preserves the apparent intended representation.

Do not normalize text merely because different capitalization exists when
capitalization may carry meaning.

============================================================

8. normalize_whitespace

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

9. normalize_column_names

Use ONLY when:

"column_name_issues"

is non-empty.

Use:

{{
  "params": {{}}
}}

Normalize names into consistent machine-friendly names.

The profile provides the original and suggested names.

IMPORTANT:

If this action is selected, it MUST be executed before any other action that
references a renamed column.

============================================================

10. replace_empty_strings

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

11. remove_invalid_values

Use ONLY when the profile provides clear evidence that particular values
are invalid.

The params object MUST contain:

{{
  "invalid_values": [...]
}}

Every value in the list MUST be supported by actual profile evidence.

Do not invent invalid values.

Replace invalid values with missing values.

============================================================

12. clip_outliers

Use ONLY when the profile provides evidence of meaningful numerical outliers.

The column must be numerical.

Use:

{{
  "params": {{}}
}}

Do NOT automatically modify every statistical outlier.

An extreme value may be legitimate business data.

Only use this action when the profile provides sufficient evidence that the
values are likely erroneous or unsuitable for the dataset.

============================================================
GENERAL SAFETY RULES
============================================================

- Every action MUST be supported by evidence in the dataset profile.
- Use the smallest set of operations necessary to improve data quality.
- Do not perform unnecessary transformations.
- Do not invent values.
- Do not invent categories.
- Do not invent invalid values.
- Do not invent missing information.
- Do not modify ambiguous data.
- Preserve valid user data whenever possible.
- Do not generate Python code.
- Do not generate SQL.
- Do not request arbitrary code execution.
- Do not use unsupported actions.
- confidence MUST be between 0.0 and 1.0.
- Every action MUST include reasoning.
- Every action that operates on a column MUST specify the column.
- Parameters MUST contain everything required to safely execute the action.
- If no cleaning action is justified, return an empty actions list.
- Do not create an action simply because an operation is available.

============================================================
ACTION ORDER
============================================================

When multiple operations are required, prefer this general order:

1. normalize_column_names
2. replace_empty_strings
3. normalize_whitespace
4. normalize_text
5. convert_to_numeric
6. convert_to_boolean
7. convert_to_date
8. standardize_categories
9. remove_invalid_values
10. impute_missing
11. clip_outliers
12. remove_duplicates

Only include operations that are actually necessary.

============================================================
DATASET PROFILE
============================================================

{json.dumps(profile, indent=2)}
"""

# ============================================================
# ASK GEMINI FOR CLEANING PLAN
# ============================================================

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )
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

    with open(
        os.path.join(run_dir, "error_report.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            error_report,
            file,
            indent=2
        )

    raise SystemExit(1)


# ============================================================
# VALIDATE GEMINI RESPONSE
# ============================================================

response_text = response.text.strip() if response.text else ""

if not response_text:
    print("Gemini returned an empty response.")

    error_report = {
        "run_id": run_id,
        "status": "failed",
        "stage": "planning",
        "input_file": input_file,
        "error": "Gemini returned an empty response"
    }

    with open(
        os.path.join(run_dir, "error_report.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            error_report,
            file,
            indent=2
        )

    raise SystemExit(1)


# Save original Gemini response
raw_plan_path = os.path.join(
    run_dir,
    "plan_raw.txt"
)

with open(
    raw_plan_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(response_text)


try:
    plan = CleaningPlan.model_validate_json(
        response_text
    )

except (ValidationError, ValueError) as error:
    print(f"Gemini returned an invalid cleaning plan: {error}")

    error_report = {
        "run_id": run_id,
        "status": "failed",
        "stage": "plan_validation",
        "input_file": input_file,
        "error": str(error)
    }

    with open(
        os.path.join(run_dir, "error_report.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            error_report,
            file,
            indent=2
        )

    raise SystemExit(1)


# ============================================================
# SAVE VALIDATED PLAN
# ============================================================

plan_path = os.path.join(
    run_dir,
    "plan.json"
)

with open(
    plan_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        plan.model_dump(),
        file,
        indent=2
    )


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
# EXECUTE AI-GENERATED ACTIONS
# ============================================================

original_dtypes = df.dtypes.astype(str).to_dict()
executed_actions = []

# Track column renames so later AI actions can still find
# their intended columns after normalize_column_names.
column_aliases = {
    column: column
    for column in df.columns
}

# Column-name normalization must happen before column-specific actions.
plan.actions.sort(
    key=lambda action: (
        action.action != "normalize_column_names"
    )
)

for action in plan.actions:

    # Resolve a column name if an earlier action renamed it.
    resolved_column = action.column

    if resolved_column is not None:
        resolved_column = column_aliases.get(
            resolved_column,
            resolved_column
        )

    try:
        before_rows = len(df)
        before_missing = df.isna().sum().to_dict()
        before_dtypes = df.dtypes.astype(str).to_dict()
        before_columns = list(df.columns)

        # Temporarily use the resolved column name for execution.
        original_action_column = action.column

        if resolved_column is not None:
            action.column = resolved_column

        df = execute_action(
            df,
            action
        )

        # Restore the original AI-generated column name for
        # the report where appropriate.
        if (
            action.action != "normalize_column_names"
            and original_action_column is not None
        ):
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
            "confidence": action.confidence,
            "status": "success",
        }

        # ----------------------------------------------------
        # Action-specific results
        # ----------------------------------------------------

        if action.action == "impute_missing":

            before = before_missing.get(
                action.column,
                0
            )

            after = after_missing.get(
                action.column,
                0
            )

            result["result"] = {
                "missing_before": before,
                "missing_after": after,
                "missing_values_filled": max(
                    0,
                    before - after
                )
            }

        elif action.action == "remove_duplicates":

            result["result"] = {
                "rows_before": before_rows,
                "rows_after": after_rows,
                "duplicates_removed": (
                    before_rows - after_rows
                )
            }

        elif action.action == "convert_to_date":

            result["result"] = {
                "dtype_before": before_dtypes.get(
                    action.column
                ),
                "dtype_after": after_dtypes.get(
                    action.column
                )
            }

        elif action.action == "convert_to_numeric":

            result["result"] = {
                "dtype_before": before_dtypes.get(
                    action.column
                ),
                "dtype_after": after_dtypes.get(
                    action.column
                )
            }

        elif action.action == "convert_to_boolean":

            result["result"] = {
                "dtype_before": before_dtypes.get(
                    action.column
                ),
                "dtype_after": after_dtypes.get(
                    action.column
                )
            }

        elif action.action == "standardize_categories":

            result["result"] = {
                "mapping": action.params.get(
                    "mapping",
                    {}
                )
            }

        elif action.action == "normalize_text":

            result["result"] = {
                "case": action.params.get(
                    "case",
                    "lower"
                )
            }

        elif action.action == "normalize_whitespace":

            result["result"] = {
                "column": action.column
            }

        elif action.action == "normalize_column_names":

            result["result"] = {
                "columns_before": before_columns,
                "columns_after": after_columns
            }

            # Update aliases after column names change.
            if len(before_columns) == len(after_columns):
                column_aliases = {
                    old: new
                    for old, new in zip(
                        before_columns,
                        after_columns
                    )
                }

        elif action.action == "replace_empty_strings":

            result["result"] = {
                "column": action.column
            }

        elif action.action == "remove_invalid_values":

            result["result"] = {
                "invalid_values": action.params.get(
                    "invalid_values",
                    []
                )
            }

        elif action.action == "clip_outliers":

            result["result"] = {
                "column": action.column
            }

        executed_actions.append(result)

    except Exception as error:

        executed_actions.append({
            "id": action.id,
            "action": action.action,
            "column": resolved_column,
            "params": action.params,
            "reasoning": action.reasoning,
            "confidence": action.confidence,
            "status": "failed",
            "error": str(error)
        })

        print(
            f"Action failed: {action.action} "
            f"on column '{resolved_column}': {error}"
        )


# ============================================================
# CAPTURE AFTER-CLEANING STATE
# ============================================================

cleaned_rows = len(df)

cleaned_missing = df.isna().sum().to_dict()


# ============================================================
# VALIDATE CLEANING RESULTS
# ============================================================

successful_actions = [
    action
    for action in executed_actions
    if action["status"] == "success"
]

failed_actions = [
    action
    for action in executed_actions
    if action["status"] == "failed"
]

duplicate_removal_requested = any(
    action["action"] == "remove_duplicates"
    for action in executed_actions
)

duplicates_before = int(
    df_before.duplicated().sum()
)

duplicates_after = int(
    df.duplicated().sum()
)


# ------------------------------------------------------------
# Row validation
# ------------------------------------------------------------

if duplicate_removal_requested:
    rows_valid = (
        duplicates_after == 0
        and cleaned_rows <= original_rows
    )
else:
    rows_valid = cleaned_rows == original_rows


# ------------------------------------------------------------
# Action-specific validation
# ------------------------------------------------------------

action_results_valid = True

for action in executed_actions:

    if action["status"] != "success":
        action_results_valid = False
        continue

    result = action.get(
        "result",
        {}
    )

    action_name = action["action"]

    if action_name == "impute_missing":

        if result.get("missing_after", 0) != 0:
            action_results_valid = False

    elif action_name == "remove_duplicates":

        if result.get("rows_after", 0) > result.get(
            "rows_before",
            0
        ):
            action_results_valid = False

        if duplicates_after != 0:
            action_results_valid = False

    elif action_name == "convert_to_date":

        dtype_after = result.get(
            "dtype_after",
            ""
        )

        if "datetime" not in dtype_after.lower():
            action_results_valid = False

    elif action_name == "convert_to_numeric":

        dtype_after = result.get(
            "dtype_after",
            ""
        )

        if not any(
            numeric_type in dtype_after.lower()
            for numeric_type in [
                "int",
                "float",
                "decimal"
            ]
        ):
            action_results_valid = False

    elif action_name == "convert_to_boolean":

        dtype_after = result.get(
            "dtype_after",
            ""
        )

        if "bool" not in dtype_after.lower():
            action_results_valid = False

    elif action_name == "standardize_categories":

        mapping = result.get(
            "mapping",
            {}
        )

        if not mapping:
            action_results_valid = False

    elif action_name == "normalize_column_names":

        columns_after = result.get(
            "columns_after",
            []
        )

        if not columns_after:
            action_results_valid = False

        if len(columns_after) != len(
            set(columns_after)
        ):
            action_results_valid = False

    elif action_name == "remove_invalid_values":

        invalid_values = result.get(
            "invalid_values",
            []
        )

        if not isinstance(
            invalid_values,
            list
        ) or not invalid_values:
            action_results_valid = False

    elif action_name == "clip_outliers":

        if action["column"] not in df.columns:
            action_results_valid = False


# ------------------------------------------------------------
# Missing-value validation
# ------------------------------------------------------------

missing_values_before = int(
    sum(original_missing.values())
)

missing_values_after = int(
    sum(cleaned_missing.values())
)

missing_values_reduced = (
    missing_values_after < missing_values_before
)

# Missing values do NOT have to decrease for every valid
# cleaning operation. This is only informational.
missing_values_improved_or_unchanged = (
    missing_values_after <= missing_values_before
)


# ------------------------------------------------------------
# Overall validation
# ------------------------------------------------------------

all_actions_succeeded = (
    len(failed_actions) == 0
)

validation = {
    "rows_valid": rows_valid,
    "missing_values_reduced": missing_values_reduced,
    "missing_values_improved_or_unchanged": (
        missing_values_improved_or_unchanged
    ),
    "action_results_valid": action_results_valid,
    "all_actions_succeeded": all_actions_succeeded,
    "successful_actions": len(successful_actions),
    "failed_actions": len(failed_actions),
    "duplicates_before": duplicates_before,
    "duplicates_after": duplicates_after
}

validation["overall"] = all([
    validation["rows_valid"],
    validation["missing_values_improved_or_unchanged"],
    validation["action_results_valid"],
    validation["all_actions_succeeded"]
])


# ============================================================
# BUILD CLEANING REPORT
# ============================================================

# Build missing-value report safely even when column names
# were changed during cleaning.
all_columns = list(
    dict.fromkeys(
        list(original_missing.keys())
        + list(cleaned_missing.keys())
    )
)

missing_value_report = {}

for column in all_columns:

    before = original_missing.get(
        column,
        0
    )

    after = cleaned_missing.get(
        column,
        0
    )

    if before != after:

        missing_value_report[column] = {
            "before": before,
            "after": after
        }


report = {
    "run_id": run_id,
    "input_file": input_file,
    "summary": plan.summary,
    "actions": executed_actions,
    "validation": validation,
    "rows": {
        "before": original_rows,
        "after": cleaned_rows
    },
    "missing_values": missing_value_report
}

# ============================================================
# PRINT CLEANING REPORT
# ============================================================

print("\nCleaning report:")

print("\nValidation:")
print(
    "  Rows valid:",
    validation["rows_valid"]
)

print(
    "  Missing values reduced:",
    validation["missing_values_reduced"]
)

print(
    "  Action results valid:",
    validation["action_results_valid"]
)

print(
    "  All actions succeeded:",
    validation["all_actions_succeeded"]
)

print(
    "  Overall:",
    validation["overall"]
)

print("\nRows:")
print(
    f"  Before: {original_rows}"
)

print(
    f"  After:  {cleaned_rows}"
)

print("\nMissing values:")

for column in cleaned_missing:

    before = original_missing.get(
        column,
        0
    )

    after = cleaned_missing[column]

    if before != after:

        print(
            f"  {column}: {before} -> {after}"
        )

print("\nAfter cleaning:")

print(
    "Rows:",
    len(df)
)

print("Missing values by column:")
print(df.isna().sum())

print("\nData types:")
print(df.dtypes)


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

cleaned_data_path = os.path.join(
    run_dir,
    "cleaned_data.csv"
)

df.to_csv(
    cleaned_data_path,
    index=False
)

print(
    f"Cleaned dataset saved to {cleaned_data_path}"
)


# ============================================================
# SAVE CLEANING REPORT
# ============================================================

report_path = os.path.join(
    run_dir,
    "cleaning_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=2
    )

print(
    f"Cleaning report saved to {report_path}"
)