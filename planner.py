import argparse
import json
import os

import pandas as pd
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from actions import execute_action
from ai_privacy import build_ai_safe_profile, remap_plan_columns
from profiler import profile_dataset
from recommendation_guardrails import filter_unsafe_text_normalization
from approval_selection import labels_for_count, parse_selection
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

parser.add_argument(
    "--auto-approve",
    action="store_true",
    help="Skip the interactive approval gate and execute the validated plan",
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
# CREATE PRIVACY-SAFE GEMINI PROMPT
# ============================================================
ai_safe_profile, token_to_column = build_ai_safe_profile(profile)

prompt = f"""
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
- remove_duplicates
- remove_column
- convert_to_date
- convert_to_numeric
- convert_to_boolean
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
  "reasoning": "The profile shows inconsistent capitalization in text values.",
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
2. replace_empty_strings
3. normalize_whitespace
4. normalize_text
5. convert_to_numeric
6. convert_to_boolean
7. convert_to_date
9. remove_invalid_values
10. impute_missing
11. clip_outliers
12. remove_duplicates
13. remove_column

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
    raw_plan_data = json.loads(response_text)
    local_plan_data = remap_plan_columns(raw_plan_data, token_to_column)
    plan = CleaningPlan.model_validate(local_plan_data)

except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as error:
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
# APPLY LOCAL TEXT-NORMALIZATION GUARDRAILS
# ============================================================
# Do not trust the model alone to decide whether lower/title/upper casing is safe.
# Descriptions, free text, skills, titles, salaries, brands, and company names are
# preserved. Only locally evidenced categorical casing cleanup may remain.
plan.actions, text_guardrail_messages = filter_unsafe_text_normalization(
    plan.actions, df, profile
)
for guardrail_message in text_guardrail_messages:
    print(f"Safety guardrail: {guardrail_message}")

# ============================================================
# APPLY LOCAL IMPUTATION-METHOD GUARDRAILS
# ============================================================
# Gemini proposes actions, but local dtype/semantic rules decide which statistical
# imputation methods are valid. Categorical data must use mode, never mean/median.
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
        # Mode is valid for numeric data too; retain it if explicitly selected.
        pass
    if semantic_type in {"email", "phone_number", "government_or_identity_id", "person_name", "address"} and action.params.get("method") in {"mean", "median"}:
        action.params["method"] = "mode"
        action.reasoning += " Local semantic rule prevents mean/median imputation for this identifier/contact field."

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
    raise SystemExit(0)

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
            raise SystemExit(0)

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
            raise SystemExit(0)

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
rejected_labels = [label for label in labels if label not in selected_labels]

# Only selected, already validated actions reach the execution loop. Keep the
# full original plan in plan.json for auditability.
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

        # Clear any replacement count left by a previous action.
        df.attrs.pop("values_replaced", None)

        # Temporarily use the resolved column name for execution.
        original_action_column = action.column

        if resolved_column is not None:
            action.column = resolved_column

        df = execute_action(
            df,
            action
        )

        # Capture replacement count immediately after execution.
        values_replaced = int(
            df.attrs.pop("values_replaced", 0) or 0
        )

        # Keep the resolved column name in the report.
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
                "column": action.column,
                "values_replaced": values_replaced
            }

        elif action.action == "remove_invalid_values":

            result["result"] = {
                "invalid_values": action.params.get(
                    "invalid_values",
                    []
                ),
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
                "column_removed": (
                    action.column not in after_columns
                )
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

        if result.get("missing_values_filled", 0) < 0:
            action_results_valid = False

    elif action_name == "remove_duplicates":

        if result.get("rows_after", 0) > result.get(
            "rows_before",
            0
        ):
            action_results_valid = False

        if duplicates_after != 0:
            action_results_valid = False

        if result.get("duplicates_removed", 0) < 0:
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

        if not isinstance(mapping, dict) or not mapping:
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

        if columns_after != list(df.columns):
            action_results_valid = False

    elif action_name == "replace_empty_strings":

        values_replaced = result.get(
            "values_replaced"
        )

        if (
            not isinstance(values_replaced, int)
            or values_replaced < 0
        ):
            action_results_valid = False

        column = result.get("column")

        if column is not None and column not in df.columns:
            action_results_valid = False

    elif action_name == "remove_invalid_values":

        invalid_values = result.get(
            "invalid_values",
            []
        )

        values_replaced = result.get(
            "values_replaced"
        )

        if not isinstance(invalid_values, list) or not invalid_values:
            action_results_valid = False

        if (
            not isinstance(values_replaced, int)
            or values_replaced < 0
        ):
            action_results_valid = False

        column = action.get("column")

        if column is not None and column not in df.columns:
            action_results_valid = False

    elif action_name == "clip_outliers":

        if action["column"] not in df.columns:
            action_results_valid = False

    elif action_name == "remove_column":

        column_removed = result.get(
            "column_removed",
            False
        )

        if not column_removed:
            action_results_valid = False

        if action["column"] in df.columns:
            action_results_valid = False

        columns_after = result.get(
            "columns_after",
            []
        )

        if columns_after != list(df.columns):
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

# Count missing values intentionally introduced by actions
# that replace empty strings or explicitly invalid values.
intentional_missing_increase = 0

for action in executed_actions:

    if action["status"] != "success":
        continue

    action_name = action["action"]
    result = action.get(
        "result",
        {}
    )

    if action_name in [
        "remove_invalid_values",
        "replace_empty_strings"
    ]:

        values_replaced = result.get(
            "values_replaced",
            0
        )

        if isinstance(values_replaced, int) and values_replaced > 0:
            intentional_missing_increase += values_replaced


# Missing values may increase when previously hidden empty or
# invalid values are converted to explicit NA values.
unexpected_missing_increase = (
    missing_values_after
    > missing_values_before + intentional_missing_increase
)

missing_values_reduced = (
    missing_values_after < missing_values_before
)

missing_values_improved_or_unchanged = (
    not unexpected_missing_increase
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
    "unexpected_missing_increase": unexpected_missing_increase,
    "intentional_missing_increase": intentional_missing_increase,
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
    "missing_values": missing_value_report,
    "monitoring": {
        "rows_changed": original_rows != cleaned_rows,
        "missing_values_before": int(sum(original_missing.values())),
        "missing_values_after": int(sum(cleaned_missing.values())),
        "missing_values_delta": int(sum(cleaned_missing.values()) - sum(original_missing.values())),
        "duplicate_rows_before": int(duplicates_before),
        "duplicate_rows_after": int(duplicates_after),
        "successful_actions": len(successful_actions),
        "failed_actions": len(failed_actions),
        "overall_validation_passed": bool(validation["overall"]),
    }
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
    "  Missing values improved or unchanged:",
    validation["missing_values_improved_or_unchanged"]
)

print(
    "  Unexpected missing increase:",
    validation["unexpected_missing_increase"]
)

print(
    "  Intentional missing increase:",
    validation["intentional_missing_increase"]
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
