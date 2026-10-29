"""
Unit & Integration Tests for CleaningSession, Rollback, and Step-by-Step Undo.
"""

import pandas as pd
import pytest

from planner import execute_cleaning_plan
from schemas import CleaningAction
from session import CleaningSession


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": [1, 2, 2, 3],
        "name": [" alice ", "BOB", "BOB", "charlie"],
        "salary": ["$50,000", "$60,000", "$60,000", "$70,000"],
        "temp_col": ["delete_me", "delete_me", "delete_me", "delete_me"],
    })


def test_session_initialization(sample_df):
    session = CleaningSession(sample_df)
    assert session.current_step == 0
    assert not session.can_undo()
    assert not session.can_redo()
    assert len(session.get_current_df()) == 4
    assert len(session.checkpoints) == 1
    assert session.checkpoints[0]["action_name"] == "INITIAL_STATE"


def test_step_by_step_undo_and_redo(sample_df):
    session = CleaningSession(sample_df)

    # Step 1: Remove duplicates (drops 1 duplicate row)
    session.apply_action(CleaningAction(action="remove_duplicates", params={}, reasoning="Remove duplicate records"))
    assert session.current_step == 1
    assert len(session.get_current_df()) == 3
    assert session.can_undo()
    assert not session.can_redo()

    # Step 2: Remove temp_col
    session.apply_action(CleaningAction(action="remove_column", column="temp_col", params={}, reasoning="Remove temp column"))
    assert session.current_step == 2
    assert "temp_col" not in session.get_current_df().columns
    assert session.can_undo()

    # Step 3: Normalize whitespace on name
    session.apply_action(CleaningAction(action="normalize_whitespace", column="name", params={}, reasoning="Normalize whitespace"))
    assert session.current_step == 3
    assert session.get_current_df()["name"].iloc[0] == "alice"

    # UNDO Step 3 (revert whitespace normalization)
    undo_res = session.undo()
    assert undo_res["status"] == "success"
    assert session.current_step == 2
    assert session.get_current_df()["name"].iloc[0] == " alice "
    assert session.can_redo()

    # UNDO Step 2 (revert column deletion)
    undo_res2 = session.undo()
    assert undo_res2["status"] == "success"
    assert session.current_step == 1
    assert "temp_col" in session.get_current_df().columns
    assert len(session.get_current_df()) == 3

    # REDO Step 2 (re-delete column)
    redo_res = session.redo()
    assert redo_res["status"] == "success"
    assert session.current_step == 2
    assert "temp_col" not in session.get_current_df().columns

    # REDO Step 3 (re-apply whitespace normalization)
    redo_res2 = session.redo()
    assert redo_res2["status"] == "success"
    assert session.current_step == 3
    assert session.get_current_df()["name"].iloc[0] == "alice"


def test_rollback_to_specific_step(sample_df):
    session = CleaningSession(sample_df)

    # Apply 4 actions
    session.apply_action(CleaningAction(action="remove_duplicates", params={}, reasoning="Deduplicate"))
    session.apply_action(CleaningAction(action="remove_column", column="temp_col", params={}, reasoning="Drop temp"))
    session.apply_action(CleaningAction(action="normalize_whitespace", column="name", params={}, reasoning="Trim space"))
    session.apply_action(CleaningAction(action="normalize_text", column="name", params={"case": "title"}, reasoning="Title case"))

    assert session.current_step == 4
    assert session.get_current_df()["name"].iloc[0] == "Alice"

    # Rollback to step 1 (keep only duplicate removal, revert steps 2, 3, 4)
    res = session.rollback_to(1)
    assert res["status"] == "success"
    assert session.current_step == 1
    assert "temp_col" in session.get_current_df().columns
    assert session.get_current_df()["name"].iloc[0] == " alice "
    assert len(session.get_current_df()) == 3


def test_rollback_all_reverts_to_pristine_state(sample_df):
    session = CleaningSession(sample_df)

    session.apply_action(CleaningAction(action="remove_duplicates", params={}, reasoning="Deduplicate"))
    session.apply_action(CleaningAction(action="remove_column", column="temp_col", params={}, reasoning="Drop temp"))

    assert session.current_step == 2
    assert len(session.get_current_df()) == 3

    # Rollback all
    res = session.rollback_all()
    assert res["status"] == "success"
    assert session.current_step == 0
    assert not session.can_undo()

    # Current df must match initial df exactly
    pd.testing.assert_frame_equal(session.get_current_df(), sample_df)


def test_session_diff(sample_df):
    session = CleaningSession(sample_df)
    session.apply_action(CleaningAction(action="remove_duplicates", params={}, reasoning="Deduplicate"))
    session.apply_action(CleaningAction(action="remove_column", column="temp_col", params={}, reasoning="Drop temp"))

    diff = session.diff(0, 2)
    assert diff["rows_before"] == 4
    assert diff["rows_after"] == 3
    assert diff["rows_delta"] == -1
    assert "temp_col" in diff["columns_removed"]


def test_undo_redo_boundaries(sample_df):
    session = CleaningSession(sample_df)

    # Undo when at step 0
    res = session.undo()
    assert res["status"] == "noop"
    assert "Cannot undo" in res["message"]

    # Redo when empty
    res2 = session.redo()
    assert res2["status"] == "noop"
    assert "Cannot redo" in res2["message"]

    # Rollback to out of bounds
    res3 = session.rollback_to(99)
    assert res3["status"] == "error"


def test_execute_cleaning_plan_records_checkpoints_in_session(sample_df):
    actions = [
        CleaningAction(action="remove_duplicates", params={}, reasoning="Deduplicate"),
        CleaningAction(action="remove_column", column="temp_col", params={}, reasoning="Drop temp"),
    ]

    session = CleaningSession(sample_df)
    df_cleaned, executed = execute_cleaning_plan(sample_df, actions, session=session)

    assert len(executed) == 2
    assert session.current_step == 2
    assert len(session.get_history()) == 3  # Initial + 2 actions
