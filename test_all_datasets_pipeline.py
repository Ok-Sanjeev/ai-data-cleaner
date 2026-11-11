import os
import json
import pandas as pd
import pytest
from unittest.mock import MagicMock

from profiler import profile_dataset
from ai_privacy import build_ai_safe_profile, remap_plan_columns
from schemas import CleaningPlan, CleaningAction
from planner import (
    apply_local_guardrails,
    build_cleaning_report,
    execute_cleaning_plan,
    validate_cleaning_results,
)


def run_pipeline_for_dataset(input_file: str, tmp_path):
    """Run full pipeline for a dataset using deterministic, privacy-preserving planning."""
    # 1. Profile
    df, profile = profile_dataset(input_file)
    assert profile["rows"] == len(df)
    assert profile["columns"] == len(df.columns)

    # 2. Build AI-safe profile (verify no raw cell values leaked)
    ai_safe, token_map = build_ai_safe_profile(profile)
    serialized = json.dumps(ai_safe)
    assert "sample_values" not in serialized
    assert "value_counts" not in serialized

    # 3. Simulate realistic Gemini proposed plan tailored to dataset profile evidence
    actions = []
    col_profiles = profile.get("column_profiles", {})

    # Propose imputations where missing values exist
    for col, details in col_profiles.items():
        if details.get("missing_count", 0) > 0 and col in df.columns:
            is_num = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
            method = "median" if is_num else "mode"
            actions.append(
                CleaningAction(
                    action="impute_missing",
                    column=col,
                    params={"method": method},
                    reasoning=f"Impute missing values in {col}",
                )
            )

    # Propose duplicate removal if duplicates exist
    if profile.get("duplicate_rows", 0) > 0:
        actions.append(
            CleaningAction(
                action="remove_duplicates",
                params={},
                reasoning="Remove duplicate rows",
            )
        )

    # Propose constant column removal
    for col, details in col_profiles.items():
        if details.get("constant_column", False) and len(df.columns) > 1:
            actions.append(
                CleaningAction(
                    action="remove_column",
                    column=col,
                    params={},
                    reasoning=f"Remove constant column {col}",
                )
            )

    raw_plan_dict = {
        "summary": f"Deterministic test cleaning plan for {os.path.basename(input_file)}",
        "actions": [a.model_dump() for a in actions],
    }

    # Validate schema
    plan = CleaningPlan.model_validate(raw_plan_dict)

    # 4. Apply local guardrails
    plan = apply_local_guardrails(plan, df, profile)

    # 5. Capture before state
    df_before = df.copy()
    orig_missing = df.isna().sum().to_dict()

    # 6. Execute approved actions
    df_cleaned, executed_actions = execute_cleaning_plan(df, plan.actions)

    # 7. Validate results
    cleaned_missing = df_cleaned.isna().sum().to_dict()
    validation = validate_cleaning_results(
        df_before=df_before,
        df=df_cleaned,
        executed_actions=executed_actions,
        original_missing=orig_missing,
        cleaned_missing=cleaned_missing,
    )

    # 8. Save outputs to isolated test directory
    out_dir = tmp_path / os.path.basename(input_file).replace(".csv", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = out_dir / "cleaned_data.csv"
    report_path = out_dir / "cleaning_report.json"

    df_cleaned.to_csv(cleaned_path, index=False)
    report = build_cleaning_report(
        run_id="test_run",
        input_file=input_file,
        summary=plan.summary,
        executed_actions=executed_actions,
        validation=validation,
        df_before=df_before,
        df_cleaned=df_cleaned,
        profile=profile,
        original_missing=orig_missing,
        cleaned_missing=cleaned_missing,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return validation, len(df_before), len(df_cleaned), executed_actions


def test_pipeline_on_test_customers(tmp_path):
    val, before, after, actions = run_pipeline_for_dataset("data/test_customers.csv", tmp_path)
    assert val["overall"] is True
    assert after <= before


def test_pipeline_on_messy_employee_dataset(tmp_path):
    val, before, after, actions = run_pipeline_for_dataset("data/Messy_Employee_dataset.csv", tmp_path)
    assert val["overall"] is True
    assert after <= before


def test_pipeline_on_cleaned_data(tmp_path):
    target_path = "data/cleaned_data.csv"
    if not os.path.exists(target_path):
        clean_df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "score": [88.5, 92.0, 79.5, 95.0],
            "status": ["Active", "Active", "Inactive", "Pending"],
        })
        target_path = str(tmp_path / "cleaned_data.csv")
        clean_df.to_csv(target_path, index=False)

    val, before, after, actions = run_pipeline_for_dataset(target_path, tmp_path)
    assert val["overall"] is True
    assert after == before


def test_pipeline_on_indian_tech_jobs(tmp_path):
    val, before, after, actions = run_pipeline_for_dataset("data/indian_tech_jobs_2026.csv", tmp_path)
    assert val["overall"] is True
    assert before == 23201
    assert after <= before
    # Constant columns data_source, scraped_at, salary_negotiable should be safely handled
    assert val["all_actions_succeeded"] is True


@pytest.mark.skipif(
    not os.path.exists("data/ott_viewer_dropoff_retention_us_v1.0.csv"),
    reason="Large git-ignored OTT dataset not present locally",
)
def test_pipeline_on_ott_viewer_dropoff(tmp_path):
    val, before, after, actions = run_pipeline_for_dataset("data/ott_viewer_dropoff_retention_us_v1.0.csv", tmp_path)
    assert val["overall"] is True
    assert before == 33171
    assert after <= before
    assert val["all_actions_succeeded"] is True


def test_pipeline_on_ott_schema_fixture(tmp_path):
    """Self-contained CI test ensuring OTT streaming schema executes safely without external files."""
    ott_fixture = pd.DataFrame({
        "show_id": [101, 102, 103, 104, 105],
        "title": ["Show A", "Show B", "Show C", "Show A", "Show D"],
        "platform": ["Netflix", "Hulu", "Netflix ", None, "Prime"],
        "episode_duration_min": [45, 50, None, 45, 60],
        "drop_off_probability": [0.22, 0.45, 0.65, 0.22, 0.15],
        "dataset_version": ["v1.0", "v1.0", "v1.0", "v1.0", "v1.0"],
    })
    csv_path = str(tmp_path / "ott_fixture.csv")
    ott_fixture.to_csv(csv_path, index=False)
    val, before, after, actions = run_pipeline_for_dataset(csv_path, tmp_path)
    assert val["overall"] is True
    assert before == 5
    assert after == 5


@pytest.mark.parametrize("bad_file,expected_error", [
    ("data/robustness_tests/empty.csv", "Input CSV file is empty"),
    ("data/robustness_tests/malformed.csv", "Malformed CSV structure"),
    ("data/robustness_tests/no_data.csv", "no data rows"),
])
def test_pipeline_robustness_on_invalid_files(bad_file, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        profile_dataset(bad_file)
