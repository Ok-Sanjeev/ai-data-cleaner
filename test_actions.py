import pandas as pd
import pytest

from actions import (
    execute_action,
    impute_missing,
    drop_missing_rows,
    remove_duplicates,
    remove_column,
    rename_column,
    convert_to_date,
    convert_to_numeric,
    convert_to_boolean,
    standardize_categories,
    normalize_text,
    normalize_whitespace,
    normalize_column_names,
    replace_empty_strings,
    remove_invalid_values,
    clip_outliers,
    clean_phone_number,
)
from schemas import CleaningAction


def test_impute_missing():
    df = pd.DataFrame({"age": [20.0, 30.0, None, 40.0]})
    result = impute_missing(df, "age", method="median")
    assert result["age"].isna().sum() == 0
    assert result["age"].iloc[2] == 30.0


def test_drop_missing_rows():
    df = pd.DataFrame({
        "id": [1, 2, None, 4],
        "name": ["Alice", "Bob", "Charlie", None]
    })
    res_col = drop_missing_rows(df, column="id")
    assert len(res_col) == 3
    assert res_col.attrs.get("rows_dropped") == 1

    res_all = drop_missing_rows(df, how="any")
    assert len(res_all) == 2


def test_rename_column():
    df = pd.DataFrame({"old_col": [1, 2, 3]})
    res = rename_column(df, "old_col", "new_col")
    assert "new_col" in res.columns
    assert "old_col" not in res.columns


def test_clean_phone_number():
    df = pd.DataFrame({"phone": ["+1 (555) 123-4567", "555.987.6543", None]})
    res = clean_phone_number(df, "phone")
    assert res["phone"].iloc[0] == "15551234567"
    assert res["phone"].iloc[1] == "5559876543"
    assert pd.isna(res["phone"].iloc[2])


def test_execute_action_integration():
    df = pd.DataFrame({
        "Age": [25, 30, None, 35],
        "Status": ["active", "ACTIVE", "inactive", "Active"],
        "Join_Date": ["4/2/2021", "7/10/2020", "12/7/2023", "1/5/2022"],
        "Phone": ["(555) 111-2222", "+1-555-333-4444", "5555555555", None],
    })

    actions = [
        CleaningAction(
            action="impute_missing",
            column="Age",
            params={"method": "median"},
            reasoning="Fill missing Age values."
        ),
        CleaningAction(
            action="standardize_categories",
            column="Status",
            params={
                "mapping": {
                    "active": "Active",
                    "ACTIVE": "Active",
                    "inactive": "Inactive"
                }
            },
            reasoning="Standardize Status values."
        ),
        CleaningAction(
            action="convert_to_date",
            column="Join_Date",
            params={},
            reasoning="Convert Join_Date to datetime."
        ),
        CleaningAction(
            action="clean_phone_number",
            column="Phone",
            params={},
            reasoning="Clean phone number digits."
        )
    ]

    for action in actions:
        df = execute_action(df, action)

    assert df["Age"].isna().sum() == 0
    assert set(df["Status"].unique()) == {"Active", "Inactive"}
    assert "datetime" in str(df["Join_Date"].dtype).lower()
    assert df["Phone"].iloc[0] == "5551112222"


def test_convert_to_numeric_with_currency_and_percent():
    df = pd.DataFrame({
        "Salary": ["$1,000", "$2,500.50", "€3,200", "$4,000"],
        "Discount": ["10%", "25.5%", "0%", "50%"],
        "Balance": ["(500.00)", "1,200.00", "(25.50)", "100 USD"],
    })
    res = convert_to_numeric(df, "Salary")
    res = convert_to_numeric(res, "Discount")
    res = convert_to_numeric(res, "Balance")

    assert pd.api.types.is_numeric_dtype(res["Salary"])
    assert res["Salary"].iloc[0] == 1000.0
    assert pd.api.types.is_numeric_dtype(res["Discount"])
    assert res["Discount"].iloc[1] == 25.5
    assert pd.api.types.is_numeric_dtype(res["Balance"])
    assert res["Balance"].iloc[0] == -500.0
    assert res["Balance"].iloc[2] == -25.5
    assert res["Balance"].iloc[3] == 100.0


def test_convert_to_boolean_extended():
    df = pd.DataFrame({"flag": ["on", "off", "enabled", "disabled", "t", "f"]})
    res = convert_to_boolean(df, "flag")
    assert res["flag"].tolist() == [True, False, True, False, True, False]


def test_normalize_column_names_multilingual_and_duplicate():
    df = pd.DataFrame({
        "Total (USD)": [1],
        "Total (EUR)": [2],
        "Prénom": ["Alice"],
        "Total (USD)": [4],
    })
    res = normalize_column_names(df)
    # Check that columns were normalized without error and are unique
    assert len(res.columns) == len(set(res.columns))
    assert all(isinstance(c, str) and c for c in res.columns)

