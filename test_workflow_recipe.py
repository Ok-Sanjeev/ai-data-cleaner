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

