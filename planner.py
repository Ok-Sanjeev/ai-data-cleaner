import json
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from actions import execute_action
from profiler import profile_dataset
from schemas import CleaningPlan


load_dotenv()


# Create a unique folder for this run
run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
run_dir = os.path.join("runs", run_id)

os.makedirs(run_dir, exist_ok=True)

print(f"Run: {run_id}")


# Connect to Gemini
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# Profile dataset
df, profile = profile_dataset(
    "data/Messy_Employee_dataset.csv"
)


# Save profile inside this run
profile_path = os.path.join(run_dir, "profile.json")

with open(profile_path, "w", encoding="utf-8") as file:
    json.dump(profile, file, indent=2)


# Ask Gemini to create a cleaning plan
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
- Use an empty object {{}} for params when an action needs no parameters.
- Only propose actions that are supported by the dataset profile.
- Do not propose an action merely because a column could theoretically be cleaned.
- Only propose an action when the profile provides evidence that cleaning is needed.

Dataset profile:

{json.dumps(profile, indent=2)}
"""


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        )
    )
)


# Validate Gemini's response
plan = CleaningPlan.model_validate_json(response.text)


# Capture before-cleaning state
original_rows = len(df)
original_missing = df.isna().sum().to_dict()


print("\nBefore cleaning:")
print("Rows:", len(df))
print("Age missing:", df["Age"].isna().sum())
print("Salary missing:", df["Salary"].isna().sum())
print("Join_Date dtype:", df["Join_Date"].dtype)


# Execute AI-generated actions
executed_actions = []

for action in plan.actions:
    try:
        df = execute_action(df, action)

        executed_actions.append({
            "id": action.id,
            "action": action.action,
            "column": action.column,
            "params": action.params,
            "reasoning": action.reasoning,
            "confidence": action.confidence,
            "status": "success"
        })

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


# Capture after-cleaning state
cleaned_rows = len(df)
cleaned_missing = df.isna().sum().to_dict()


# Build cleaning report
report = {
    "run_id": run_id,
    "summary": plan.summary,
    "actions": executed_actions,
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
        if original_missing[column] != cleaned_missing[column]
    }
}


# Print cleaning report
print("\nCleaning report:")

print("Rows:")
print(f"  Before: {original_rows}")
print(f"  After:  {cleaned_rows}")

print("\nMissing values:")

for column in df.columns:
    before = original_missing[column]
    after = cleaned_missing[column]

    if before != after:
        print(f"  {column}: {before} -> {after}")


# Save cleaned dataset
print("\nAfter cleaning:")
print("Rows:", len(df))
print("Age missing:", df["Age"].isna().sum())
print("Salary missing:", df["Salary"].isna().sum())
print("Join_Date dtype:", df["Join_Date"].dtype)


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


# Save cleaning report
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