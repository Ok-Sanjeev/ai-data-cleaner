"""
Unit tests for DataRefine JSON recipe management and sequential replay.
"""

import pandas as pd
import pytest

from workflow_recipe import WorkflowRecipeManager


def test_recipe_export_and_validation():
    history = [
        {"action": "INITIAL_STATE", "step": 0},
        {"action": "trim_whitespace", "column": "name", "step": 1, "params": {}},
        {"action": "normalize_text", "column": "name", "step": 2, "params": {"case": "title"}},
    ]

    recipe = WorkflowRecipeManager.export_recipe_from_history(history)
    assert recipe["operations_count"] == 2
    assert recipe["format"] == "openrefine_datarefine_v2"

    valid, errors = WorkflowRecipeManager.validate_recipe(recipe)
    assert valid is True
    assert len(errors) == 0


def test_recipe_validation_invalid_action():
    bad_recipe = {
        "operations": [
            {"action": "unsupported_hack_operation", "column": "col"}
        ]
    }
    valid, errors = WorkflowRecipeManager.validate_recipe(bad_recipe)
    assert valid is False
    assert any("unsupported" in e for e in errors)


def test_recipe_replay():
    df = pd.DataFrame({
        "name": [" alice ", "  bob  ", "CHARLIE"],
        "age": [20, 30, 40],
    })

    recipe = {
        "operations": [
            {"action": "trim_whitespace", "column": "name", "params": {}},
            {"action": "normalize_text", "column": "name", "params": {"case": "lower"}},
        ]
    }

    cleaned_df, report = WorkflowRecipeManager.replay_recipe(df, recipe)
    assert list(cleaned_df["name"]) == ["alice", "bob", "charlie"]
    assert report["executed_count"] == 2


def test_recipe_replay_missing_column_and_partial_failure():
    # Dataset lacks 'missing_col' referenced in recipe step 2
    df = pd.DataFrame({
        "name": [" alice ", "  bob  "],
    })
    recipe = {
        "operations": [
            {"action": "trim_whitespace", "column": "name", "params": {}},
            {"action": "normalize_text", "column": "nonexistent_column", "params": {"case": "upper"}},
            {"action": "normalize_text", "column": "name", "params": {"case": "upper"}},
        ]
    }

    # With stop_on_error=False, it should execute steps 1 & 3, reporting failure on step 2
    cleaned_df, report = WorkflowRecipeManager.replay_recipe(df, recipe, stop_on_error=False)
    assert list(cleaned_df["name"]) == ["ALICE", "BOB"]
    assert report["successful_steps"] == 2
    assert report["failed_steps"] == 1
    assert report["step_results"][1]["status"] == "failed"
    assert "error" in report["step_results"][1]

    # With stop_on_error=True, it should stop after step 2 failure
    df2 = pd.DataFrame({"name": [" alice ", "  bob  "]})
    cleaned_df2, report2 = WorkflowRecipeManager.replay_recipe(df2, recipe, stop_on_error=True)
    assert report2["successful_steps"] == 1
    assert report2["failed_steps"] == 1
    assert len(report2["step_results"]) == 2  # Step 3 was never attempted

