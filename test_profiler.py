import pandas as pd
import pytest

from profiler import profile_dataset


def test_missing_file():
    with pytest.raises(
        FileNotFoundError,
        match="Input CSV file not found"
    ):
        profile_dataset(
            "data/robustness_tests/does_not_exist.csv"
        )


def test_empty_file():
    with pytest.raises(
        ValueError,
        match="Input CSV file contains no columns"
    ):
        profile_dataset(
            "data/robustness_tests/empty.csv"
        )

def test_header_without_data():
    with pytest.raises(
        ValueError,
        match="no data rows"
    ):
        profile_dataset(
            "data/robustness_tests/no_data.csv"
        )


def test_malformed_csv():
    with pytest.raises(
        ValueError,
        match="Malformed CSV structure"
    ):
        profile_dataset(
            "data/robustness_tests/malformed.csv"
        )


def test_valid_customer_dataset():
    df, profile = profile_dataset(
        "data/test_customers.csv"
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 205
    assert profile["rows"] == 205
    assert profile["columns"] == 7


def test_profile_contains_column_profiles():
    _, profile = profile_dataset(
        "data/test_customers.csv"
    )

    assert "column_profiles" in profile

    assert "Age_Years" in profile["column_profiles"]
    assert "Income" in profile["column_profiles"]
    assert "Registration_Date" in profile["column_profiles"]