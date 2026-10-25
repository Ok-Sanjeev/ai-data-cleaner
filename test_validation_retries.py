import json
import pandas as pd
import pytest
from unittest.mock import MagicMock

from planner import (
    apply_local_guardrails,
    build_cleaning_report,
    generate_validated_plan,
    sync_plan_summary_with_guarded_actions,
    validate_cleaning_results,
)
from schemas import CleaningAction, CleaningPlan


def test_generate_validated_plan_retries_on_empty_invalid_values():
    client = MagicMock()

    # First attempt: Gemini produces remove_invalid_values with empty invalid_values list (fails schema validation)
    bad_plan_json = json.dumps({
        "summary": "Bad plan",
        "actions": [{
            "action": "remove_invalid_values",
            "column": "job_url",
            "params": {"invalid_values": []},
            "reasoning": "Should fail schema validation"
        }]
    })

    # Second attempt: Gemini corrects it after receiving validation feedback
    good_plan_json = json.dumps({
        "summary": "Good plan",
        "actions": [{
            "action": "replace_empty_strings",
            "column": "job_url",
            "params": {},
            "reasoning": "Proper replacement"
        }]
    })

    bad_resp = MagicMock()
    bad_resp.text = bad_plan_json
    good_resp = MagicMock()
    good_resp.text = good_plan_json

    client.models.generate_content.side_effect = [bad_resp, good_resp]

    plan, raw_text = generate_validated_plan(
        client=client,
        initial_prompt="Clean data",
        token_to_column={"job_url": "job_url"},
        max_retries=2,
    )

    assert plan.summary == "Good plan"
    assert len(plan.actions) == 1
    assert plan.actions[0].action == "replace_empty_strings"
    # Verify generate_content was called twice (initial + 1 retry)
    assert client.models.generate_content.call_count == 2
    # Check that retry prompt contained the validation feedback
    second_call_prompt = client.models.generate_content.call_args_list[1][1]["contents"]
    assert "CRITICAL VALIDATION FEEDBACK FROM PREVIOUS ATTEMPT" in second_call_prompt
    assert "remove_invalid_values requires a non-empty invalid_values list" in second_call_prompt


def test_guardrails_prevent_unsupported_date_and_numeric_conversions():
    df = pd.DataFrame({
        "relative_date": ["3+ weeks ago", "Not Available", "2 days ago"],
        "person_name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
        "rich_column": [10, 20, 30],
    })
    profile = {
        "column_profiles": {
            "relative_date": {
                "semantic_type": "categorical_or_text",
                "semantic_hints": [],
                "empty_column": False,
                "constant_column": False,
            },
            "person_name": {
                "semantic_type": "person_name",
                "semantic_hints": [],
                "empty_column": False,
                "constant_column": False,
            },
            "rich_column": {
                "semantic_type": "numeric_measure_or_identifier",
                "empty_column": False,
                "constant_column": False,
            },
        }
    }

    plan = CleaningPlan(
        summary="Test safety guardrails",
        actions=[
            CleaningAction(
                action="convert_to_date",
                column="relative_date",
                params={},
                reasoning="Erroneous date conversion attempt",
            ),
            CleaningAction(
                action="convert_to_numeric",
                column="person_name",
                params={},
                reasoning="Erroneous numeric conversion attempt",
            ),
            CleaningAction(
                action="remove_column",
                column="rich_column",
                params={},
                reasoning="Erroneous column deletion attempt",
            ),
        ],
    )

    guarded_plan = apply_local_guardrails(plan, df, profile)
    # All 3 erroneous actions should be safely blocked by local guardrails
    assert len(guarded_plan.actions) == 0


def test_sync_plan_summary_removes_pruned_date_conversion_mentions():
    """Ensure plan explanation matches guarded plan when date conversion is pruned (Finding 2)."""
    df = pd.DataFrame({
        "posted_date_raw": ["3+ weeks ago", "2 days ago", "1 month ago"],
        "location": ["bangalore", "mumbai", "delhi"],
        "data_source": ["naukri.com", "naukri.com", "naukri.com"],
    })
    profile = {
        "column_profiles": {
            "posted_date_raw": {
                "semantic_type": "categorical_or_text",
                "semantic_hints": [],
                "empty_column": False,
                "constant_column": False,
            },
            "location": {
                "semantic_type": "categorical_or_text",
                "semantic_hints": ["inconsistent_text_case"],
                "empty_column": False,
                "constant_column": False,
            },
            "data_source": {
                "semantic_type": "categorical_or_text",
                "semantic_hints": [],
                "empty_column": False,
                "constant_column": True,
            },
        }
    }

    initial_summary = (
        "The dataset contains textual columns with inconsistent casing that could benefit from normalization. "
        "There are constant columns providing no discriminative value across records, which can be removed. "
        "One column contains date-like strings that should be converted to a proper date format. "
        "Missing values in 'job_url' are present but will not be imputed due to the column's nature as a URL."
    )

    plan = CleaningPlan(
        summary=initial_summary,
        actions=[
            CleaningAction(
                action="normalize_text",
                column="location",
                params={"case": "title"},
                reasoning="Standardize location casing",
            ),
            CleaningAction(
                action="convert_to_date",
                column="posted_date_raw",
                params={},
                reasoning="Convert relative dates",
            ),
            CleaningAction(
                action="remove_column",
                column="data_source",
                params={},
                reasoning="Remove constant data source",
            ),
        ],
    )

    guarded_plan = apply_local_guardrails(plan, df, profile)

    # convert_to_date should be pruned by date guardrails
    assert not any(a.action == "convert_to_date" for a in guarded_plan.actions)
    # The summary should no longer mention date conversion
    assert "proper date format" not in guarded_plan.summary
    assert "converted to a proper date" not in guarded_plan.summary
    # But valid parts of the plan explanation should remain intact
    assert "normalization" in guarded_plan.summary
    assert "constant columns" in guarded_plan.summary
    assert "job_url" in guarded_plan.summary


def test_constant_column_recommendation_wording_and_metadata_safety():
    """Ensure constant column wording recognizes metadata/provenance and avoids dismissive phrasing (Finding 3)."""
    df = pd.DataFrame({
        "data_source": ["naukri.com", "naukri.com"],
        "scraped_at": ["2026-10-25", "2026-10-25"],
        "salary_negotiable": [False, False],
        "other_const": ["fixed", "fixed"],
    })
    profile = {
        "column_profiles": {
            "data_source": {"constant_column": True, "empty_column": False, "semantic_type": "categorical_or_text"},
            "scraped_at": {"constant_column": True, "empty_column": False, "semantic_type": "date_or_timestamp"},
            "salary_negotiable": {"constant_column": True, "empty_column": False, "semantic_type": "possible_boolean"},
            "other_const": {"constant_column": True, "empty_column": False, "semantic_type": "categorical_or_text"},
        }
    }

    plan = CleaningPlan(
        summary="Test constant columns",
        actions=[
            CleaningAction(action="remove_column", column="data_source", params={}, reasoning="Column is useless and adds no value."),
            CleaningAction(action="remove_column", column="scraped_at", params={}, reasoning="Column is not useful for most analyses."),
            CleaningAction(action="remove_column", column="salary_negotiable", params={}, reasoning="Provides no value across records."),
            CleaningAction(action="remove_column", column="other_const", params={}, reasoning="This column is useless."),
        ],
    )

    guarded_plan = apply_local_guardrails(plan, df, profile)

    for action in guarded_plan.actions:
        # None of the reasoning should contain dismissive phrasing
        assert "useless" not in action.reasoning.lower()
        assert "not useful" not in action.reasoning.lower()
        assert "adds no value" not in action.reasoning.lower()
        assert "provides no value" not in action.reasoning.lower()

    # Provenance columns should explicitly mention provenance / audit metadata
    data_source_action = next(a for a in guarded_plan.actions if a.column == "data_source")
    assert "provenance/audit metadata" in data_source_action.reasoning
    assert "optional" in data_source_action.reasoning

    scraped_at_action = next(a for a in guarded_plan.actions if a.column == "scraped_at")
    assert "provenance/audit metadata" in scraped_at_action.reasoning

    # Business flag column should mention business attribute / flag
    flag_action = next(a for a in guarded_plan.actions if a.column == "salary_negotiable")
    assert "business attribute or flag" in flag_action.reasoning


def test_validation_distinguishes_unresolved_missing_values_from_action_failures():
    """Ensure validation accurately distinguishes unresolved missing values from action failures (Finding 1)."""
    df_before = pd.DataFrame({
        "job_id": [1, 2, 3],
        "job_url": ["http://a.com", None, None],
        "data_source": ["web", "web", "web"],
    })
    df_cleaned = pd.DataFrame({
        "job_id": [1, 2, 3],
        "job_url": ["http://a.com", None, None],
    })

    orig_missing = {"job_id": 0, "job_url": 2, "data_source": 0}
    cleaned_missing = {"job_id": 0, "job_url": 2}

    executed_actions = [{
        "action": "remove_column",
        "column": "data_source",
        "params": {},
        "status": "success",
        "result": {"column_removed": True},
    }]

    validation = validate_cleaning_results(
        df_before=df_before,
        df=df_cleaned,
        executed_actions=executed_actions,
        original_missing=orig_missing,
        cleaned_missing=cleaned_missing,
    )

    # missing_values_reduced is False because missing values remain unchanged (2 == 2)
    assert validation["missing_values_reduced"] is False
    # But it is explicitly recognized that missing reduction was not requested
    assert validation["missing_reduction_requested"] is False
    assert validation["approved_imputations_succeeded"] is True
    assert validation["unresolved_missing_values_count"] == 2
    assert validation["unresolved_missing_columns"] == {"job_url": 2}
    # Run passes overall because approved action succeeded and no unexpected missing values were added
    assert validation["overall"] is True


def test_unresolved_data_quality_issues_recorded_without_inventing_values():
    """Ensure unresolved data-quality issues, including missing URLs, are recorded accurately (Finding 4)."""
    df_before = pd.DataFrame({
        "job_url": [None] * 53 + ["https://naukri.com/job/1"],
        "salary": [10, 12, 11, 500, 14] + [10] * 49,
        "posted_raw": ["3+ weeks ago", "2 days ago", "1 month ago", "yesterday", "5 hours ago"] + ["1 day ago"] * 49,
    })
    df_cleaned = df_before.copy()

    profile = {
        "column_profiles": {
            "job_url": {"semantic_type": "url", "missing_count": 53},
            "salary": {"statistics": {"outlier_count": 1}},
            "posted_raw": {"semantic_hints": ["possible_date"]},
        }
    }
    orig_missing = {"job_url": 53, "salary": 0, "posted_raw": 0}
    cleaned_missing = {"job_url": 53, "salary": 0, "posted_raw": 0}

    validation = validate_cleaning_results(
        df_before=df_before,
        df=df_cleaned,
        executed_actions=[],
        original_missing=orig_missing,
        cleaned_missing=cleaned_missing,
    )

    report = build_cleaning_report(
        run_id="test_run_123",
        input_file="test.csv",
        summary="Test summary",
        executed_actions=[],
        validation=validation,
        df_before=df_before,
        df_cleaned=df_cleaned,
        profile=profile,
        original_missing=orig_missing,
        cleaned_missing=cleaned_missing,
    )

    # 1. Missing values report must not be empty; it must show job_url unresolved status
    assert "job_url" in report["missing_values"]
    assert report["missing_values"]["job_url"]["before"] == 53
    assert report["missing_values"]["job_url"]["after"] == 53
    assert report["missing_values"]["job_url"]["status"] == "unresolved"

    # 2. Unresolved issues must record missing URL preservation
    unresolved_types = {issue["issue_type"] for issue in report["unresolved_issues"]}
    assert "missing_values" in unresolved_types
    assert "numerical_outliers" in unresolved_types
    assert "relative_or_unparsed_dates" in unresolved_types

    url_issue = next(i for i in report["unresolved_issues"] if i["column"] == "job_url")
    assert url_issue["count"] == 53
    assert "fictitious web addresses" in url_issue["details"]
    assert report["monitoring"]["unresolved_missing_values"] == 53
