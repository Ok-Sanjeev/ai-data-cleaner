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
        match="Input CSV file is empty"
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


def test_semicolon_delimited_csv(tmp_path):
    tsv_file = tmp_path / "test_semicolon.csv"
    tsv_file.write_text("name;age;city\nAlice;30;New York\nBob;25;London\n", encoding="utf-8")

    df, profile = profile_dataset(str(tsv_file))
    assert len(df) == 2
    assert profile["delimiter"] == ";"
    assert list(df.columns) == ["name", "age", "city"]


def test_latin1_encoded_csv(tmp_path):
    latin_file = tmp_path / "test_latin.csv"
    latin_file.write_bytes("name,city\nRen\xe9,Montr\xe9al\n".encode("latin-1"))

    df, profile = profile_dataset(str(latin_file))
    assert len(df) == 1
    assert profile["encoding"] in ["latin-1", "cp1252"]


def test_csv_with_unnamed_and_currency_columns(tmp_path):
    csv_file = tmp_path / "test_excel_export.csv"
    csv_file.write_text(',Product,Price\n0,Laptop,"$1,200"\n1,Mouse,$25\n', encoding="utf-8")

    df, profile = profile_dataset(str(csv_file))
    assert len(df) == 2
    # First column should be auto-repaired to unnamed_column_1 instead of crashing
    assert "unnamed_column_1" in df.columns
    assert "Price" in profile["column_profiles"]
    assert "possible_numeric" in profile["column_profiles"]["Price"].get("semantic_hints", [])

