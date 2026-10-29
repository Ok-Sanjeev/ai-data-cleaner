"""
Unit & Integration Tests for Data Cleaning Quality Evaluator.
"""

import numpy as np
import pandas as pd
import pytest

from evaluator import evaluate_cleaning, format_evaluation_report, values_match
from schemas import CleaningAction
from actions import execute_action


def test_values_match_handles_diverse_types():
    # Null equivalence
    assert values_match(np.nan, "")
    assert values_match(None, np.nan)
    assert values_match("<NA>", None)
    assert not values_match(np.nan, "0")

    # Boolean equivalence
    assert values_match(True, "TRUE")
    assert values_match("yes", True)
    assert values_match(False, "0")
    assert values_match(0, False)
    assert not values_match(True, False)

    # Numeric equivalence (currency, commas, float vs int)
    assert values_match(25, 25.0)
    assert values_match("$50,000", 50000.0)
    assert values_match("45.50", 45.5)
    assert not values_match(25, 26)

    # Date equivalence
    assert values_match("2023-01-15", "2023-01-15 00:00:00")
    assert values_match("2023/01/15", "2023-01-15")


def test_evaluator_perfect_cleaning_scores_100():
    dirty = pd.DataFrame({
        "id": [1, 2, 3],
        "name": [" alice ", "BOB", "charlie"],
        "age": [25, None, 35],
    })
    truth = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
    })
    cleaned = truth.copy()  # perfect cleaning

    scores = evaluate_cleaning(dirty, cleaned, truth, id_column="id")
    o = scores["overall"]

    assert o["precision"] == 1.0
    assert o["recall"] == 1.0
    assert o["f1"] == 1.0
    assert o["false_positives"] == 0
    assert o["false_negatives"] == 0
    assert o["true_positives"] == 4  # 3 name fixes + 1 age fix
    assert o["cell_accuracy_after"] == 1.0
    assert o["row_accuracy_after"] == 1.0


def test_evaluator_detects_overcleaning_and_missed_errors():
    dirty = pd.DataFrame({
        "id": [1, 2],
        "val": ["correct", "dirty_val"],
    })
    truth = pd.DataFrame({
        "id": [1, 2],
        "val": ["correct", "fixed_val"],
    })
    # Overcleaned: corrupted "correct" into "wrong", and left "dirty_val" untouched
    cleaned = pd.DataFrame({
        "id": [1, 2],
        "val": ["corrupted_change", "dirty_val"],
    })

    scores = evaluate_cleaning(dirty, cleaned, truth, id_column="id")
    o = scores["overall"]

    # 1 False Positive (corrupted cell 1)
    assert o["false_positives"] == 1
    # 1 False Negative (missed dirty_val in cell 2)
    assert o["false_negatives"] == 1
    assert o["true_positives"] == 0
    assert o["precision"] == 0.0
    assert o["recall"] == 0.0
    assert o["f1"] == 0.0


def test_benchmark_dataset_cleaning_and_scoring():
    """Run full cleaning operations on data/benchmark/dirty_sample.csv and verify high quality score against ground truth."""
    dirty_path = "data/benchmark/dirty_sample.csv"
    truth_path = "data/benchmark/ground_truth.csv"

    df = pd.read_csv(dirty_path)
    df_truth = pd.read_csv(truth_path)

    # 1. Remove duplicates
    df = execute_action(df, CleaningAction(action="remove_duplicates", params={}, reasoning="Remove duplicate rows"))

    # 2. Normalize whitespace & casing on name
    df = execute_action(df, CleaningAction(action="normalize_whitespace", column="name", params={}, reasoning="Trim whitespace"))
    df = execute_action(df, CleaningAction(action="normalize_text", column="name", params={"case": "title"}, reasoning="Title case name"))

    # 3. Impute missing age with median
    df = execute_action(df, CleaningAction(action="impute_missing", column="age", params={"method": "median"}, reasoning="Median age imputation"))

    # 4. Convert join_date to date
    df = execute_action(df, CleaningAction(action="convert_to_date", column="join_date", params={}, reasoning="Convert to date"))

    # 5. Convert salary to numeric
    df = execute_action(df, CleaningAction(action="convert_to_numeric", column="salary", params={}, reasoning="Convert to numeric"))

    # 6. Clean phone number
    df = execute_action(df, CleaningAction(action="clean_phone_number", column="phone", params={}, reasoning="Clean phone number"))

    # 7. Convert is_member to boolean
    df = execute_action(df, CleaningAction(action="convert_to_boolean", column="is_member", params={}, reasoning="Convert to boolean"))

    # 8. Remove invalid values in status
    df = execute_action(df, CleaningAction(action="remove_invalid_values", column="status", params={"invalid_values": ["N/A", "missing", "-999"]}, reasoning="Remove invalid tokens"))

    # Evaluate cleaned data against ground truth
    df_dirty = pd.read_csv(dirty_path)
    scores = evaluate_cleaning(df_dirty, df, df_truth, id_column="customer_id")
    o = scores["overall"]

    # Verify score metrics
    assert o["precision"] >= 0.95
    assert o["recall"] >= 0.95
    assert o["f1"] >= 0.95
    assert o["duplicates_removed"] == 1
    assert o["cell_accuracy_after"] == 1.0

    # Ensure ASCII scorecard formats without errors
    scorecard = format_evaluation_report(scores)
    assert "QUALITY EVALUATION SCORECARD" in scorecard
    assert "PERFECT" in scorecard
