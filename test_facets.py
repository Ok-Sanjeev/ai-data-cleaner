"""
Unit tests for multi-column facets and dataset exploration in DataRefine.
"""

import pandas as pd
import pytest

from facets import (
    compute_text_facet,
    compute_numeric_facet,
    compute_date_facet,
    compute_facet_auto,
    apply_filters,
    explore_dataset,
)


def test_text_facet():
    series = pd.Series(["US", "US", "CA", "UK", None, ""])
    res = compute_text_facet(series)
    assert res["facet_type"] == "text"
    assert res["total_rows"] == 6
    assert res["null_count"] == 1
    assert res["blank_count"] == 1
    # Check choices
    top_choice = res["choices"][0]
    assert top_choice["value"] == "US"
    assert top_choice["count"] == 2


def test_numeric_facet():
    series = pd.Series([10, 20, 30, 40, 50, None])
    res = compute_numeric_facet(series, bins_count=5)
    assert res["facet_type"] == "numeric"
    assert res["stats"]["min"] == 10.0
    assert res["stats"]["max"] == 50.0
    assert len(res["bins"]) > 0


def test_date_facet():
    series = pd.Series(["2024-01-01", "2024-02-01", "2024-03-01", None])
    res = compute_date_facet(series)
    assert res["facet_type"] == "date"
    assert res["valid_date_count"] == 3
    assert res["null_count"] == 1


def test_multi_column_filter_and_explore():
    df = pd.DataFrame({
        "country": ["US", "US", "CA", "UK", "IN"],
        "age": [25, 45, 30, 60, 22],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    })

    # Text filter on country + numeric filter on age
    filters = {
        "country": {"type": "text", "selected": ["US", "CA"]},
        "age": {"type": "numeric", "min": 25, "max": 40},
    }

    filtered_df = apply_filters(df, filters=filters)
    assert len(filtered_df) == 2  # Alice (US, 25) and Charlie (CA, 30)

    # Explore with sorting and pagination
    exp = explore_dataset(df, filters=filters, sort_by="age", sort_ascending=False)
    assert exp["matched_rows"] == 2
    assert exp["records"][0]["name"] == "Charlie"  # 30 > 25
