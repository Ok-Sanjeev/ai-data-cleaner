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
You are a data cleaning planner.

Analyze the dataset profile and create a cleaning plan.

You may ONLY use these actions:

- impute_missing
- remove_duplicates
- convert_to_date
- standardize_categories

Return ONLY valid JSON.

The JSON must have exactly this structure:

{{
  "summary": "short description of the data quality issues",
  "actions": [
    {{
      "action": "impute_missing",
      "column": "Age",
      "params": {{
        "method": "median"
      }},
      "reasoning": "why this action is appropriate",
      "confidence": 0.95
    }}
  ]
}}

Rules:

- Do not use Markdown.
- Do not use code fences.
- Do not include explanations outside the JSON.
- Do not invent actions outside the allowed action list.
- confidence must be between 0.0 and 1.0.
- Use an empty object {{}} for params only when an action genuinely requires no parameters.
- Only propose actions that are supported by the dataset profile.
- Do not propose an action merely because a column could theoretically be cleaned.
- Only propose an action when the profile provides evidence that cleaning is needed.

Rules for impute_missing:

- The column must contain missing values according to the profile.
- The "params" object MUST contain a valid "method".
- The method must be one of: "median", "mean", or "mode".
- Prefer "median" for numerical columns when it is appropriate.

Rules for remove_duplicates:

- Only propose this action when the profile provides evidence of duplicate rows or duplicate records.
- If duplicate rows are identified, use an empty params object {{}}.
- Do not remove rows unless the profile provides evidence that duplicates exist.

Rules for convert_to_date:

- Only propose this action when the profile provides evidence that a column contains date values stored in an inappropriate format or type.
- Use an empty params object {{}}.

Rules for standardize_categories:

- Only propose this action when the profile provides evidence of inconsistent categorical values.
- The "params" object MUST contain a non-empty "mapping" object.
- The mapping must contain the actual category values found in the dataset that need standardization.
- Map each inconsistent value to its intended standardized value.
- Do not invent category values that are not supported by the profile.
- Do not use an empty mapping.
- Do not propose standardize_categories if the profile does not provide enough information to construct a safe mapping.

For example, if the profile contains these values:

"Active", "active", "ACTIVE", "Inactive", "inactive", "INACTIVE"

then a valid action could be:

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

Dataset profile:

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


print("\nBefore cleaning:")
print("Rows:", len(df))

print("Missing values by column:")
print(df.isna().sum())

print("\nData types:")
print(df.dtypes)


# ============================================================
# EXECUTE AI-GENERATED ACTIONS
# ============================================================

executed_actions = []

for action in plan.actions:

    try:
        before_rows = len(df)
        before_missing = df.isna().sum().to_dict()
        before_dtypes = df.dtypes.astype(str).to_dict()

        df = execute_action(
            df,
            action
        )

        after_rows = len(df)
        after_missing = df.isna().sum().to_dict()
        after_dtypes = df.dtypes.astype(str).to_dict()

        result = {
            "id": action.id,
            "action": action.action,
            "column": action.column,
            "params": action.params,
            "reasoning": action.reasoning,
            "confidence": action.confidence,
            "status": "success"
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
                "missing_values_filled": before - after
            }


        elif action.action == "remove_duplicates":

            result["result"] = {
                "rows_before": before_rows,
                "rows_after": after_rows,
                "duplicates_removed": before_rows - after_rows
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


        elif action.action == "standardize_categories":

            result["result"] = {
                "mapping": action.params.get(
                    "mapping",
                    {}
                )
            }


        executed_actions.append(result)


    except Exception as error:

        executed_actions.append({
            "id": action.id,
            "action": action.action,
            "column": action.column,
            "params": action.params,
            "reasoning": action.reasoning,
            "confidence": action.confidence,
            "status": "failed",
            "error": str(error)
        })

        print(
            f"Action failed: {action.action} "
            f"on column '{action.column}': {error}"
        )


# ============================================================
# CAPTURE AFTER-CLEANING STATE
# ============================================================

cleaned_rows = len(df)

cleaned_missing = df.isna().sum().to_dict()


# ============================================================
# VALIDATE CLEANING RESULTS
# ============================================================

duplicate_removal_requested = any(
    action["action"] == "remove_duplicates"
    and action["status"] == "success"
    for action in executed_actions
)


rows_valid = (
    cleaned_rows == original_rows
    or duplicate_removal_requested
)


missing_values_reduced = (
    sum(cleaned_missing.values())
    < sum(original_missing.values())
)


# Validate each successful action independently
action_results_valid = True

for action in executed_actions:

    if action["status"] != "success":
        continue

    result = action.get(
        "result",
        {}
    )


    if action["action"] == "impute_missing":

        if result.get(
            "missing_after",
            0
        ) != 0:

            action_results_valid = False


    elif action["action"] == "remove_duplicates":

        if result.get(
            "rows_after",
            0
        ) > result.get(
            "rows_before",
            0
        ):

            action_results_valid = False


    elif action["action"] == "convert_to_date":

        dtype_after = result.get(
            "dtype_after",
            ""
        )

        if "datetime" not in dtype_after.lower():

            action_results_valid = False


    elif action["action"] == "standardize_categories":

        mapping = result.get(
            "mapping",
            {}
        )

        if not mapping:

            action_results_valid = False


all_actions_succeeded = all(
    action["status"] == "success"
    for action in executed_actions
)


validation = {
    "rows_valid": rows_valid,
    "missing_values_reduced": missing_values_reduced,
    "action_results_valid": action_results_valid,
    "all_actions_succeeded": all_actions_succeeded
}


validation["overall"] = all(
    validation.values()
)


# ============================================================
# BUILD CLEANING REPORT
# ============================================================

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
    "missing_values": {
        column: {
            "before": original_missing[column],
            "after": cleaned_missing[column]
        }
        for column in df.columns
        if original_missing[column]
        != cleaned_missing[column]
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

for column in df.columns:

    before = original_missing[column]
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