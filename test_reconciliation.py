"""
Unit tests for entity reconciliation against reference catalogs in DataRefine.
"""

import pandas as pd
import pytest

from reconciliation import reconcile_column_values, REFERENCE_CATALOGS


def test_reconcile_countries():
    series = pd.Series(["USA", "U.S.", "United Kingdom", "england", "Bharat", "UnknownLand"])
    res = reconcile_column_values(series, catalog_name="countries")

    assert res["catalog"] == "countries"
    assert res["total_unique"] == 6

    candidates = {c["original"]: c for c in res["candidates"]}

    # Exact aliases
    assert candidates["USA"]["matched_canonical"] == "United States"
    assert candidates["USA"]["score"] >= 0.95

    assert candidates["england"]["matched_canonical"] == "United Kingdom"
    assert candidates["Bharat"]["matched_canonical"] == "India"

    # Unknown value has no match
    assert candidates["UnknownLand"]["matched_canonical"] is None


def test_reconcile_tech_roles():
    series = pd.Series(["swe", "ds", "react developer", "Chef"])
    res = reconcile_column_values(series, catalog_name="tech_roles")
    candidates = {c["original"]: c for c in res["candidates"]}

    assert candidates["swe"]["matched_canonical"] == "Software Engineer"
    assert candidates["ds"]["matched_canonical"] == "Data Scientist"
    assert candidates["react developer"]["matched_canonical"] == "Frontend Developer"
    assert candidates["Chef"]["matched_canonical"] is None


def test_reconcile_custom_catalog():
    series = pd.Series(["Front-end", "Back-end", "Chef"])
    custom_roles = ["Frontend", "Backend", "Data Science"]

    res = reconcile_column_values(series, catalog_name="custom", custom_entities=custom_roles, match_threshold=0.7)
    candidates = {c["original"]: c for c in res["candidates"]}

    assert candidates["Front-end"]["matched_canonical"] == "Frontend"
    assert candidates["Back-end"]["matched_canonical"] == "Backend"
    assert candidates["Chef"]["matched_canonical"] is None

