import pytest
from pydantic import ValidationError

from schemas import CleaningAction, CleaningPlan


def test_valid_imputation_action():
    action = CleaningAction(
        action="impute_missing",
        column="Age",
        params={"method": "median"},
        reasoning="Age contains missing values."
    )

    assert action.action == "impute_missing"
    assert action.params["method"] == "median"


def test_invalid_action_name():
    with pytest.raises(ValidationError):
        CleaningAction(
            action="drop_column",
            column="Age",
            params={},
            reasoning="Remove the column."
        )



def test_missing_required_column():
    with pytest.raises(ValidationError):
        CleaningAction(
            action="impute_missing",
            params={"method": "median"},
            reasoning="Fill missing values."
        )


def test_invalid_imputation_method():
    with pytest.raises(ValidationError):
        CleaningAction(
            action="impute_missing",
            column="Age",
            params={"method": "random"},
            reasoning="Fill missing ages."
        )


def test_missing_imputation_method():
    with pytest.raises(ValidationError):
        CleaningAction(
            action="impute_missing",
            column="Age",
            params={},
            reasoning="Fill missing ages."
        )


def test_empty_category_mapping():
    with pytest.raises(ValidationError):
        CleaningAction(
            action="standardize_categories",
            column="Status",
            params={"mapping": {}},
            reasoning="Standardize category values."
        )


def test_missing_category_mapping():
    with pytest.raises(ValidationError):
        CleaningAction(
            action="standardize_categories",
            column="Status",
            params={},
            reasoning="Standardize category values."
        )


def test_valid_category_mapping():
    action = CleaningAction(
        action="standardize_categories",
        column="Status",
        params={
            "mapping": {
                "Active": "active",
                "ACTIVE": "active",
                "Inactive": "inactive"
            }
        },
        reasoning="Normalize inconsistent capitalization."
    )

    assert action.params["mapping"]["ACTIVE"] == "active"


def test_valid_remove_duplicates():
    action = CleaningAction(
        action="remove_duplicates",
        params={},
        reasoning="Duplicate rows were detected."
    )

    assert action.action == "remove_duplicates"


def test_valid_convert_to_date():
    action = CleaningAction(
        action="convert_to_date",
        column="Registration_Date",
        params={},
        reasoning="The column contains date values stored as strings."
    )

    assert action.column == "Registration_Date"


def test_cleaning_plan_validation():
    plan = CleaningPlan(
        summary="Clean missing values.",
        actions=[
            CleaningAction(
                action="impute_missing",
                column="Age",
                params={"method": "median"},
                reasoning="Fill missing ages."
            )
        ]
    )

    assert len(plan.actions) == 1

def test_confidence_field_is_rejected_as_removed_from_schema():
    with pytest.raises(ValidationError):
        CleaningAction(
            action="remove_duplicates",
            params={},
            reasoning="Duplicate rows were detected.",
            confidence=0.95,
        )
