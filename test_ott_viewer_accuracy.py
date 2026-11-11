import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import app, SESSIONS
from profiler import profile_dataset
from actions import normalize_whitespace, remove_column, standardize_categories, ACTION_REGISTRY
from session import CleaningSession
from schemas import CleaningAction


OTT_FILE = "data/ott_viewer_dropoff_retention_us_v1.0.csv"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.skipif(
    not os.path.exists(OTT_FILE),
    reason="Large git-ignored OTT reference CSV not present locally",
)
def test_reference_csv_profiling_and_quality_categories():
    """Verify OTT viewer reference dataset has exact 33,171 rows, zero missing, and numerical dtypes."""
    assert os.path.exists(OTT_FILE), f"Reference CSV not found at {OTT_FILE}"

    df, profile = profile_dataset(OTT_FILE)

    # 1. Structural assertions
    assert profile["rows"] == 33171
    assert profile["columns"] == 23
    assert profile["duplicate_rows"] == 0

    # 2. Zero missing values across all columns in reference CSV
    for col, p in profile["column_profiles"].items():
        assert p["missing_count"] == 0, f"Column '{col}' should have 0 missing values, got {p['missing_count']}"
        assert p["quality_categories"]["missing_values"]["count"] == 0

    # 3. Verify actual numerical data types
    assert pd.api.types.is_integer_dtype(df["show_id"])
    assert pd.api.types.is_integer_dtype(df["episode_number"])
    assert pd.api.types.is_integer_dtype(df["drop_off"])
    assert pd.api.types.is_float_dtype(df["drop_off_probability"])

    # 4. Verify Statistical Outliers are separated from missing values
    show_id_prof = profile["column_profiles"]["show_id"]
    assert show_id_prof["quality_categories"]["statistical_outliers"]["count"] == 2444
    assert show_id_prof["quality_categories"]["missing_values"]["count"] == 0
    assert show_id_prof["quality_categories"]["statistical_outliers"]["category"] == "Statistical Outliers"

    drop_off_prof = profile["column_profiles"]["drop_off"]
    assert drop_off_prof["quality_categories"]["statistical_outliers"]["count"] == 4804
    assert drop_off_prof["quality_categories"]["missing_values"]["count"] == 0

    # 5. Verify quality categories summary
    summary = profile["quality_categories_summary"]
    assert summary["missing_values"]["total_count"] == 0
    assert summary["statistical_outliers"]["total_count"] > 0
    assert "show_id" in summary["statistical_outliers"]["affected_columns"]


@pytest.mark.skipif(
    not os.path.exists(OTT_FILE),
    reason="Large git-ignored OTT reference CSV not present locally",
)
def test_cleaning_pipeline_accuracy_on_reference_dataset(client):
    """Run full session cleaning on reference dataset and verify metrics accuracy."""
    assert os.path.exists(OTT_FILE)

    # Profile via API
    prof_res = client.post("/api/profile", json={"filepath": OTT_FILE})
    assert prof_res.status_code == 200
    prof_data = prof_res.json()
    session_id = prof_data["session_id"]
    assert prof_data["profile"]["rows"] == 33171

    # Plan with custom actions:
    # 0: normalize_whitespace on platform (GREEN)
    # 1: remove_column on season_number (RED)
    # 2: remove_column on dataset_version (RED)
    # 3: standardize_categories on platform (YELLOW)
    plan_actions = [
        CleaningAction(
            action="normalize_whitespace",
            column="platform",
            params={},
            reasoning="Normalize whitespace in platform names",
            risk_level="green",
        ),
        CleaningAction(
            action="remove_column",
            column="season_number",
            params={},
            reasoning="Constant column with zero variance",
            risk_level="red",
        ),
        CleaningAction(
            action="remove_column",
            column="dataset_version",
            params={},
            reasoning="Constant metadata column",
            risk_level="red",
        ),
        CleaningAction(
            action="standardize_categories",
            column="platform",
            params={"mapping": {"AMC+ Amazon Channel": "MGM+ Amazon Channel"}},
            reasoning="Standardize platform category naming",
            risk_level="yellow",
        ),
    ]

    SESSIONS[session_id]["plan"] = type("Plan", (), {
        "summary": "Verified reference cleaning plan",
        "actions": plan_actions,
    })()

    # Execute all 4 actions
    clean_res = client.post(
        "/api/clean",
        json={"session_id": session_id, "approved_action_ids": [0, 1, 2, 3]},
    )
    assert clean_res.status_code == 200
    clean_data = clean_res.json()

    # Verify summary metrics
    diff_summary = clean_data["diff_summary"]
    assert diff_summary["rows_before"] == 33171
    assert diff_summary["rows_after"] == 33171
    assert diff_summary["rows_delta"] == 0
    assert diff_summary["rows_retained"] == 33171
    assert diff_summary["rows_retained_pct"] == 100.0

    # Modified cells count must be calculated across ALL 33,171 rows, NOT 0!
    assert diff_summary["modified_cells_count"] > 0

    # Total missing must be 0 before and 0 after
    assert diff_summary["total_missing_before"] == 0
    assert diff_summary["total_missing_after"] == 0
    assert diff_summary["missing_resolved"] == 0

    # Exactly 2 columns removed
    assert diff_summary["columns_removed_count"] == 2
    assert "season_number" in diff_summary["columns_removed"]
    assert "dataset_version" in diff_summary["columns_removed"]

    # Final columns should be 21
    assert len(clean_data["columns"]) == 21

    # Verify unresolved issues:
    # Must NOT report any missing values!
    unresolved = clean_data["unresolved_issues"]
    missing_issues = [iss for iss in unresolved if iss.get("category") == "Missing Values" or iss.get("issue_type") == "missing_values"]
    assert len(missing_issues) == 0, f"Expected 0 unresolved missing issues, found: {missing_issues}"

    # Outlier issues must be explicitly categorized as Statistical Outliers with real numerical dtypes
    outlier_issues = [iss for iss in unresolved if iss.get("category") == "Statistical Outliers" or iss.get("issue_type") == "numerical_outliers"]
    assert len(outlier_issues) > 0

    show_id_issue = next((iss for iss in outlier_issues if iss["column"] == "show_id"), None)
    assert show_id_issue is not None
    assert show_id_issue["category"] == "Statistical Outliers"
    assert show_id_issue["count"] == 2444
    assert show_id_issue["dtype"] == "int64"
    assert "outlier" in show_id_issue["details"].lower()
    assert "missing" not in show_id_issue["details"].lower()

    # Verify visual diff table produces non-null real preview values
    row0 = clean_data["diff_table"][0]
    assert row0["cells"]["show_id"]["cleaned"] is not None
    assert row0["cells"]["show_id"]["value"] is not None
    assert row0["cells"]["title"]["value"] is not None

    # Verify cleaned CSV export exists on disk
    cleaned_path = SESSIONS[session_id]["cleaned_csv_path"]
    assert os.path.exists(cleaned_path)
    df_exported = pd.read_csv(cleaned_path)
    assert df_exported.shape == (33171, 21)
    assert df_exported.isna().sum().sum() == 0


def test_self_contained_outlier_vs_missing_separation(tmp_path):
    """Ensure profiler separates IQR statistical outliers from missing values cleanly without external files."""
    fixture_df = pd.DataFrame({
        "show_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "platform": ["Netflix", "Netflix", "Hulu", "Prime", "Netflix", "Hulu", "Prime", "Netflix", "Hulu", "Netflix"],
        "watch_pct": [65.0, 65.0, 68.0, 68.0, 70.0, 70.0, 71.0, 71.0, 66.0, 9999.0],
    })
    csv_file = str(tmp_path / "ott_outlier_check.csv")
    fixture_df.to_csv(csv_file, index=False)

    df, profile = profile_dataset(csv_file)
    assert profile["rows"] == 10
    watch_prof = profile["column_profiles"]["watch_pct"]
    assert watch_prof["missing_count"] == 0
    assert watch_prof["quality_categories"]["missing_values"]["count"] == 0
    assert watch_prof["quality_categories"]["statistical_outliers"]["count"] == 1
    assert profile["quality_categories_summary"]["missing_values"]["total_count"] == 0
    assert profile["quality_categories_summary"]["statistical_outliers"]["total_count"] >= 1

