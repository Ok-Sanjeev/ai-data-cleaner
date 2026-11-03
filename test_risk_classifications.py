import pandas as pd
import pytest

from actions import ACTION_METADATA
from planner import apply_local_guardrails, assign_action_risk, build_cleaning_report
from schemas import CleaningAction, CleaningPlan


def test_risk_classification_defaults():
    profile = {"column_profiles": {"col1": {"constant_column": False, "empty_column": False}}}

    # Green
    act_green = CleaningAction(action="trim_whitespace", column="col1", params={}, reasoning="Clean edges")
    act_green = assign_action_risk(act_green, profile)
    assert act_green.risk_level == "green"
    assert act_green.risk_label == "Safe Repair"

    # Yellow
    act_yellow = CleaningAction(action="impute_missing", column="col1", params={"method": "median"}, reasoning="Fill missing")
    act_yellow = assign_action_risk(act_yellow, profile)
    assert act_yellow.risk_level == "yellow"
    assert act_yellow.risk_label == "Needs Review"

    # Red
    act_red = CleaningAction(action="remove_column", column="col1", params={}, reasoning="Remove column")
    act_red = assign_action_risk(act_red, profile)
    assert act_red.risk_level == "red"
    assert act_red.risk_label == "Potentially Destructive"
    assert act_red.warning is not None


def test_constant_column_retained_as_red_in_guardrails():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "data_source": ["linkedin", "linkedin", "linkedin"],
        "age": [20, 25, 30],
    })
    profile = {
        "column_profiles": {
            "id": {"constant_column": False, "empty_column": False},
            "data_source": {"constant_column": True, "empty_column": False, "semantic_type": "categorical_or_text"},
            "age": {"constant_column": False, "empty_column": False},
        }
    }
    plan = CleaningPlan(
        summary="Test plan",
        actions=[
            CleaningAction(action="remove_column", column="data_source", params={}, reasoning="Constant column"),
        ]
    )

    guarded_plan = apply_local_guardrails(plan, df, profile)
    # Action should NOT be pruned! It should be retained with RED risk
    assert len(guarded_plan.actions) == 1
    action = guarded_plan.actions[0]
    assert action.column == "data_source"
    assert action.risk_level == "red"
    assert action.risk_label == "Potentially Destructive"
    assert "provenance" in action.warning.lower() or "audit" in action.warning.lower() or "constant" in action.warning.lower()


def test_missing_value_observations_have_full_statistics():
    df_before = pd.DataFrame({"job_url": [None, None, "https://example.com"], "id": [1, 2, 3]})
    df_cleaned = df_before.copy()
    profile = {
        "column_profiles": {
            "job_url": {"semantic_type": "url", "missing_count": 2},
            "id": {"semantic_type": "identifier", "missing_count": 0},
        }
    }
    validation = {
        "rows_valid": True,
        "missing_values_improved_or_unchanged": True,
        "action_results_valid": True,
        "all_actions_succeeded": True,
        "approved_imputations_succeeded": True,
        "overall": True,
    }

    report = build_cleaning_report(
        run_id="run_1",
        input_file="test.csv",
        summary="Summary",
        executed_actions=[],
        validation=validation,
        df_before=df_before,
        df_cleaned=df_cleaned,
        profile=profile,
        original_missing={"job_url": 2, "id": 0},
        cleaned_missing={"job_url": 2, "id": 0},
    )

    unres = report["unresolved_issues"]
    assert len(unres) >= 1
    url_obs = next(o for o in unres if o["column"] == "job_url")
    assert url_obs["missing_count"] == 2
    assert url_obs["missing_percentage"] == 66.67
    assert url_obs["total_rows"] == 3
    assert url_obs["suggested_remediation"] is not None
    assert url_obs["risk_level"] == "yellow"
    assert url_obs["explanation"] is not None
    assert "undefined" not in str(url_obs)
