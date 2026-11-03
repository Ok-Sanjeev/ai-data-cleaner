import pandas as pd
import pytest

from actions import (
    ACTION_REGISTRY,
    convert_percentage,
    deduplicate_by_columns,
    detect_invalid_categories,
    execute_action,
    extract_date_components,
    extract_email,
    extract_numeric_values,
    map_values,
    merge_columns,
    normalize_unicode,
    remove_constant_columns,
    remove_empty_rows,
    remove_high_missing_columns,
    remove_non_printable_characters,
    remove_special_characters,
    replace_negative_values,
    round_numeric,
    split_column,
    standardize_currency,
    standardize_date_format,
    trim_whitespace,
    validate_email,
    validate_numeric_ranges,
    validate_url,
)
from schemas import CleaningAction


def test_trim_whitespace():
    df = pd.DataFrame({"text": ["  hello  ", "world   ", "  clean"]})
    res = trim_whitespace(df, "text")
    assert list(res["text"]) == ["hello", "world", "clean"]


def test_remove_special_characters():
    df = pd.DataFrame({"code": ["item #123!", "user@domain", "normal 45"]})
    res = remove_special_characters(df, "code")
    assert list(res["code"]) == ["item 123", "userdomain", "normal 45"]


def test_remove_non_printable_characters():
    df = pd.DataFrame({"txt": ["hello\x00world", "clean\u200btext\x1f"]})
    res = remove_non_printable_characters(df, "txt")
    assert list(res["txt"]) == ["helloworld", "cleantext"]


def test_normalize_unicode():
    df = pd.DataFrame({"name": ["Café", "naïve", "résumé"]})
    res = normalize_unicode(df, "name", form="NFKD")
    assert len(res) == 3


def test_standardize_date_format():
    df = pd.DataFrame({"dt": ["2023/05/12", "12-05-2023", "May 12, 2023", None]})
    res = standardize_date_format(df, "dt", "%Y-%m-%d")
    assert res["dt"].iloc[0] == "2023-05-12"
    assert pd.isna(res["dt"].iloc[3])


def test_extract_date_components():
    df = pd.DataFrame({"dt": ["2024-03-15", "2023-11-20"]})
    res = extract_date_components(df, "dt", ["year", "month", "day"])
    assert "dt_year" in res.columns
    assert "dt_month" in res.columns
    assert "dt_day" in res.columns
    assert res["dt_year"].iloc[0] == 2024
    assert res["dt_month"].iloc[0] == 3
    assert res["dt_day"].iloc[0] == 15


def test_standardize_currency():
    df = pd.DataFrame({"price": ["$1,250.50", "€450.00", "₹12,000", "(50.00)"]})
    res = standardize_currency(df, "price")
    assert res["price"].iloc[0] == 1250.50
    assert res["price"].iloc[1] == 450.00
    assert res["price"].iloc[2] == 12000.00
    assert res["price"].iloc[3] == -50.00


def test_extract_numeric_values():
    df = pd.DataFrame({"info": ["Weight: 72.5 kg", "Approx 100 units", "-15.2 celsius", "none"]})
    res = extract_numeric_values(df, "info")
    assert res["info"].iloc[0] == 72.5
    assert res["info"].iloc[1] == 100.0
    assert res["info"].iloc[2] == -15.2
    assert pd.isna(res["info"].iloc[3])


def test_round_numeric():
    df = pd.DataFrame({"val": [3.14159, 2.71828, 1.41421]})
    res = round_numeric(df, "val", decimals=2)
    assert list(res["val"]) == [3.14, 2.72, 1.41]


def test_validate_numeric_ranges():
    df = pd.DataFrame({"age": [25, 150, -5, 40]})
    res = validate_numeric_ranges(df, "age", min_val=0, max_val=120, action="replace_with_na")
    assert res["age"].iloc[0] == 25
    assert pd.isna(res["age"].iloc[1])
    assert pd.isna(res["age"].iloc[2])
    assert res["age"].iloc[3] == 40


def test_replace_negative_values():
    df = pd.DataFrame({"salary": [50000, -100, 65000, -500]})
    res_nan = replace_negative_values(df, "salary", method="nan")
    assert pd.isna(res_nan["salary"].iloc[1])
    assert pd.isna(res_nan["salary"].iloc[3])

    res_zero = replace_negative_values(df, "salary", method="zero")
    assert res_zero["salary"].iloc[1] == 0

    res_abs = replace_negative_values(df, "salary", method="abs")
    assert res_abs["salary"].iloc[1] == 100


def test_convert_percentage():
    df = pd.DataFrame({"rate": ["45%", "12.5%", "0.85%"]})
    res = convert_percentage(df, "rate", to_decimal=True)
    assert res["rate"].iloc[0] == 0.45
    assert res["rate"].iloc[1] == 0.125


def test_validate_email():
    df = pd.DataFrame({"email": ["valid@example.com", "not-an-email", "user@sub.domain.co"]})
    res = validate_email(df, "email", action="replace_with_na")
    assert res["email"].iloc[0] == "valid@example.com"
    assert pd.isna(res["email"].iloc[1])
    assert res["email"].iloc[2] == "user@sub.domain.co"


def test_extract_email():
    df = pd.DataFrame({"bio": ["Contact me at john@doe.org for inquiries", "No email here"]})
    res = extract_email(df, "bio", target_column="clean_email")
    assert res["clean_email"].iloc[0] == "john@doe.org"
    assert pd.isna(res["clean_email"].iloc[1])


def test_validate_url():
    df = pd.DataFrame({"site": ["https://google.com", "ftp://invalid-scheme", "htp:/bad.url"]})
    res = validate_url(df, "site", action="replace_with_na")
    assert res["site"].iloc[0] == "https://google.com"
    assert pd.isna(res["site"].iloc[1])
    assert pd.isna(res["site"].iloc[2])


def test_remove_empty_rows():
    df = pd.DataFrame({
        "a": [1, None, None, 4],
        "b": ["x", None, "   ", "z"],
    })
    res = remove_empty_rows(df)
    assert len(res) == 2
    assert list(res["a"]) == [1, 4]


def test_detect_invalid_categories():
    df = pd.DataFrame({"dept": ["Engineering", "Marketing", "FakeDept", "Sales"]})
    res = detect_invalid_categories(df, "dept", allowed_categories=["Engineering", "Marketing", "Sales"], action="replace_with_na")
    assert res["dept"].iloc[0] == "Engineering"
    assert pd.isna(res["dept"].iloc[2])


def test_split_column():
    df = pd.DataFrame({"full_name": ["John Doe", "Jane Smith"]})
    res = split_column(df, "full_name", delimiter=" ", new_columns=["first_name", "last_name"])
    assert list(res["first_name"]) == ["John", "Jane"]
    assert list(res["last_name"]) == ["Doe", "Smith"]


def test_merge_columns():
    df = pd.DataFrame({"city": ["San Francisco", "New York"], "country": ["USA", "USA"]})
    res = merge_columns(df, columns=["city", "country"], separator=", ", new_column="location")
    assert list(res["location"]) == ["San Francisco, USA", "New York, USA"]


def test_deduplicate_by_columns():
    df = pd.DataFrame({
        "emp_id": [101, 101, 102],
        "name": ["Alice V1", "Alice V2", "Bob"],
    })
    res = deduplicate_by_columns(df, subset=["emp_id"], keep="first")
    assert len(res) == 2
    assert list(res["name"]) == ["Alice V1", "Bob"]


def test_remove_high_missing_columns():
    df = pd.DataFrame({
        "good": [1, 2, 3, 4],
        "bad": [None, None, None, 1],  # 75% missing
    })
    res = remove_high_missing_columns(df, threshold=0.7)
    assert "good" in res.columns
    assert "bad" not in res.columns


def test_remove_constant_columns():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "constant_flag": ["ACTIVE", "ACTIVE", "ACTIVE"],
        "mixed": ["A", "B", "C"],
    })
    res = remove_constant_columns(df)
    assert "id" in res.columns
    assert "mixed" in res.columns
    assert "constant_flag" not in res.columns


def test_map_values():
    df = pd.DataFrame({"status": ["Y", "N", "Y", "P"]})
    res = map_values(df, "status", {"Y": "Approved", "N": "Rejected", "P": "Pending"})
    assert list(res["status"]) == ["Approved", "Rejected", "Approved", "Pending"]


def test_execute_action_integration_on_new_actions():
    df = pd.DataFrame({
        "salary": ["$1,000", "$2,000"],
        "name": ["  john  ", "  sarah  "],
    })

    action1 = CleaningAction(
        action="trim_whitespace",
        column="name",
        params={},
        reasoning="Trim name whitespace"
    )
    df = execute_action(df, action1)
    assert list(df["name"]) == ["john", "sarah"]

    action2 = CleaningAction(
        action="standardize_currency",
        column="salary",
        params={},
        reasoning="Parse currency"
    )
    df = execute_action(df, action2)
    assert list(df["salary"]) == [1000.0, 2000.0]
