import json
import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import app, SESSIONS


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "gemini" in data["supported_providers"]
    assert "ollama" in data["supported_providers"]
    assert "openai" in data["supported_providers"]


def test_api_datasets(client):
    res = client.get("/api/datasets")
    assert res.status_code == 200
    data = res.json()
    datasets = data.get("datasets", [])
    assert len(datasets) > 0
    # Must find benchmark and test datasets
    paths = [d["path"] for d in datasets]
    assert any("dirty_sample.csv" in p for p in paths)
    assert any("test_customers.csv" in p for p in paths)


def test_api_dashboard_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "AI Data Cleaner" in res.text
    assert "Side-by-Side Visual Diff" in res.text
    assert "Benchmark Evaluation" in res.text


def test_full_pipeline_via_api(client):
    # 1. Profile dataset
    dirty_path = "data/benchmark/dirty_sample.csv"
    res = client.post("/api/profile", json={"filepath": dirty_path})
    assert res.status_code == 200
    prof_data = res.json()
    session_id = prof_data["session_id"]
    assert prof_data["summary"]["rows"] == 11
    assert prof_data["summary"]["duplicate_rows"] == 1
    assert len(prof_data["columns"]) == 8

    # Verify column distributions
    first_col = prof_data["columns"][0]
    assert "missing_count" in first_col
    assert "unique_count" in first_col

    # 2. Generate Plan with Mock Provider
    plan_res = client.post(
        "/api/plan",
        json={
            "session_id": session_id,
            "provider": "mock",
            "model": "mock-model",
        },
    )
    assert plan_res.status_code == 200
    plan_data = plan_res.json()
    assert plan_data["provider"] == "mock"

    # Set up realistic actions in session's plan for cleaning
    session = SESSIONS[session_id]
    from schemas import CleaningAction, CleaningPlan
    session["plan"] = CleaningPlan(
        summary="API cleaning test plan",
        actions=[
            CleaningAction(action="remove_duplicates", params={}, reasoning="Drop dup"),
            CleaningAction(action="normalize_whitespace", column="name", params={}, reasoning="Clean space"),
            CleaningAction(action="normalize_text", column="name", params={"case": "title"}, reasoning="Title case name"),
            CleaningAction(action="impute_missing", column="age", params={"method": "median"}, reasoning="Median age"),
            CleaningAction(action="clean_phone_number", column="phone", params={}, reasoning="Format phone"),
        ]
    )

    # 3. Execute approved cleaning
    clean_res = client.post(
        "/api/clean",
        json={
            "session_id": session_id,
            "approved_action_ids": [0, 1, 2, 3, 4],
        },
    )
    assert clean_res.status_code == 200
    clean_data = clean_res.json()
    assert clean_data["diff_summary"]["rows_before"] == 11
    assert clean_data["diff_summary"]["rows_after"] == 10
    assert clean_data["diff_summary"]["rows_delta"] == -1
    assert clean_data["diff_summary"]["modified_cells_count"] > 0
    assert len(clean_data["diff_table"]) > 0

    # Verify checkpoints
    checkpoints = clean_data["checkpoints"]
    assert len(checkpoints) >= 6  # step 0 (initial) + 5 actions

    # 4. Undo step
    undo_res = client.post("/api/session/undo", json={"session_id": session_id})
    assert undo_res.status_code == 200
    undo_data = undo_res.json()
    assert undo_data["status"] == "success"
    assert undo_data["current_step"] == 4

    # 5. Redo step
    redo_res = client.post("/api/session/redo", json={"session_id": session_id})
    assert redo_res.status_code == 200
    redo_data = redo_res.json()
    assert redo_data["status"] == "success"
    assert redo_data["current_step"] == 5

    # 6. Quality benchmark evaluation against ground truth
    gt_path = "data/benchmark/ground_truth.csv"
    eval_res = client.post(
        "/api/evaluate",
        json={
            "session_id": session_id,
            "ground_truth_path": gt_path,
        },
    )
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert "overall" in eval_data
    assert eval_data["overall"]["precision"] > 0.8
    assert eval_data["overall"]["recall"] > 0.0

    # 7. Downloads
    down_csv = client.get(f"/api/download/cleaned?session_id={session_id}")
    assert down_csv.status_code == 200
    assert "text/csv" in down_csv.headers["content-type"]
    assert len(down_csv.text.splitlines()) == 11  # header + 10 rows

    down_rep = client.get(f"/api/download/report?session_id={session_id}")
    assert down_rep.status_code == 200
    assert "application/json" in down_rep.headers["content-type"]
    rep_obj = down_rep.json()
    assert "run_id" in rep_obj
    assert "validation" in rep_obj

    # 8. Rollback all to original
    rb_res = client.post("/api/session/rollback", json={"session_id": session_id})
    assert rb_res.status_code == 200
    rb_data = rb_res.json()
    assert rb_data["current_step"] == 0
    assert rb_data["diff_summary"]["rows_after"] == 11


def test_diff_with_column_name_normalization(client):
    """Test that when normalize_column_names runs, diff table cells are properly mapped and populated."""
    dirty_path = "data/Messy_Employee_dataset.csv"
    res = client.post("/api/profile", json={"filepath": dirty_path})
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    session = SESSIONS[session_id]
    from schemas import CleaningAction, CleaningPlan
    session["plan"] = CleaningPlan(
        summary="Test column normalization diff",
        actions=[
            CleaningAction(action="normalize_column_names", params={}, reasoning="snake_case headers"),
            CleaningAction(action="impute_missing", column="Age", params={"method": "median"}, reasoning="Median age"),
        ]
    )

    clean_res = client.post(
        "/api/clean",
        json={"session_id": session_id, "approved_action_ids": [0, 1]},
    )
    assert clean_res.status_code == 200
    clean_data = clean_res.json()

    # Columns after should be snake_case
    assert "employee_id" in clean_data["columns"]
    assert "age" in clean_data["columns"]

    # Diff table should have non-empty mapped cells
    row0 = clean_data["diff_table"][0]
    assert "employee_id" in row0["cells"]
    assert row0["cells"]["employee_id"]["original"] is not None
    assert row0["cells"]["employee_id"]["cleaned"] is not None
    # Compatibility aliases for frontend
    assert row0["cells"]["employee_id"]["value"] is not None
    assert row0["cells"]["employee_id"]["original_value"] is not None
    assert "status" in row0["cells"]["employee_id"]


def test_visual_diff_table_values_not_null_on_jobs_dataset():
    """Ensure visual diff table produces populated, non-null cells for real datasets."""
    from server import compute_side_by_side_diff

    # Sample dataset mirroring the user's screenshot
    df_before = pd.DataFrame({
        "job_id": [1, 2, 3],
        "job_title": ["Data Scientist", "Backend Dev", "ML Engineer"],
        "company_name": ["Cisco", "Google", "Microsoft"],
        "company_rating": [4.1, 4.5, 4.3],
        "location": ["Bengaluru", "Hyderabad", "Pune"],
    })

    # Cleaned DataFrame with title whitespace normalized
    df_after = df_before.copy()
    df_after["job_title"] = ["Data Scientist", "Backend Developer", "ML Engineer"]

    diff = compute_side_by_side_diff(df_before, df_after)

    assert diff["rows_before"] == 3
    assert diff["rows_after"] == 3
    assert len(diff["row_diffs"]) == 3

    # Row 0: Unchanged
    row0 = diff["row_diffs"][0]
    assert row0["cells"]["job_title"]["cleaned"] == "Data Scientist"
    assert row0["cells"]["job_title"]["value"] == "Data Scientist"
    assert row0["cells"]["job_title"]["original"] == "Data Scientist"
    assert row0["cells"]["job_title"]["original_value"] == "Data Scientist"
    assert row0["cells"]["job_title"]["changed"] is False
    assert row0["cells"]["job_title"]["status"] == "unchanged"
    assert row0["cells"]["company_name"]["value"] == "Cisco"

    # Row 1: Modified
    row1 = diff["row_diffs"][1]
    assert row1["cells"]["job_title"]["cleaned"] == "Backend Developer"
    assert row1["cells"]["job_title"]["value"] == "Backend Developer"
    assert row1["cells"]["job_title"]["original"] == "Backend Dev"
    assert row1["cells"]["job_title"]["original_value"] == "Backend Dev"
    assert row1["cells"]["job_title"]["changed"] is True
    assert row1["cells"]["job_title"]["status"] == "modified"
    assert row1["status"] == "modified"

    # Test dropped row handling with index alignment
    df_dropped = df_before.drop(index=[1])
    diff_drop = compute_side_by_side_diff(df_before, df_dropped)
    assert len(diff_drop["row_diffs"]) == 3
    # Row index 1 should be explicitly marked as dropped
    row_dropped = [r for r in diff_drop["row_diffs"] if r["row_index"] == 1][0]
    assert row_dropped["status"] == "dropped"
    assert row_dropped["cells"]["job_title"]["status"] == "dropped"
    assert row_dropped["cells"]["job_title"]["original"] == "Backend Dev"
    assert row_dropped["cells"]["job_title"]["cleaned"] is None


def test_add_and_remove_action_api(client):
    """Test dynamically adding and removing operations via API."""
    res = client.post("/api/profile", json={"filepath": "data/benchmark/dirty_sample.csv"})
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    # 1. Add action
    add_res = client.post("/api/plan/add_action", json={
        "session_id": session_id,
        "action": "trim_whitespace",
        "column": "name",
        "params": {},
        "reasoning": "Trim customer names",
    })
    assert add_res.status_code == 200
    add_data = add_res.json()
    assert len(add_data["actions"]) == 1
    assert add_data["actions"][0]["action"] == "trim_whitespace"
    assert add_data["actions"][0]["column"] == "name"
    assert add_data["actions"][0]["risk_level"] == "green"

    # Add second action (destructive)
    add_res2 = client.post("/api/plan/add_action", json={
        "session_id": session_id,
        "action": "remove_column",
        "column": "phone",
        "params": {},
        "reasoning": "Discard phone column",
    })
    assert add_res2.status_code == 200
    assert len(add_res2.json()["actions"]) == 2
    assert add_res2.json()["actions"][1]["risk_level"] == "red"

    # 2. Remove first action
    del_res = client.post("/api/plan/remove_action", json={
        "session_id": session_id,
        "action_index": 0,
    })
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert len(del_data["actions"]) == 1
    assert del_data["actions"][0]["action"] == "remove_column"


def test_clean_with_zero_approved_actions(client):
    """Test that executing clean with zero approved actions transitions smoothly without errors."""
    res = client.post("/api/profile", json={"filepath": "data/benchmark/dirty_sample.csv"})
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    clean_res = client.post("/api/clean", json={
        "session_id": session_id,
        "approved_action_ids": [],
    })
    assert clean_res.status_code == 200
    clean_data = clean_res.json()
    assert clean_data["diff_summary"]["actions_executed"] == 0
    assert clean_data["diff_summary"]["modified_cells_count"] == 0
    assert clean_data["diff_summary"]["rows_retained"] == 11
    assert len(clean_data["diff_table"]) > 0


def test_openrefine_api_endpoints(client):
    """Test clustering, GREL transformation, facets, reconciliation, and recipe replay API routes."""
    # Initialize session
    prof_res = client.post("/api/profile", json={"filepath": "data/benchmark/dirty_sample.csv"})
    assert prof_res.status_code == 200
    session_id = prof_res.json()["session_id"]

    # 1. Facet Compute
    facet_res = client.post("/api/facets/compute", json={
        "session_id": session_id,
        "column": "status",
        "facet_type": "text",
    })
    assert facet_res.status_code == 200
    f_data = facet_res.json()
    assert f_data["status"] == "success"
    assert f_data["facet"]["facet_type"] == "text"
    assert len(f_data["facet"]["choices"]) > 0

    # 2. Dataset Explore
    explore_res = client.post("/api/dataset/explore", json={
        "session_id": session_id,
        "global_search": "active",
        "limit": 10,
    })
    assert explore_res.status_code == 200
    exp_data = explore_res.json()
    assert exp_data["matched_rows"] >= 1
    assert len(exp_data["records"]) <= 10

    # 3. GREL Preview & Apply
    grel_prev = client.post("/api/grel/preview", json={
        "session_id": session_id,
        "column": "name",
        "expression": "value.trim().toTitlecase()",
        "sample_limit": 5,
    })
    assert grel_prev.status_code == 200
    assert grel_prev.json()["success"] is True
    assert len(grel_prev.json()["rows"]) == 5

    grel_app = client.post("/api/grel/apply", json={
        "session_id": session_id,
        "column": "name",
        "expression": "value.trim().toTitlecase()",
    })
    assert grel_app.status_code == 200
    assert grel_app.json()["status"] == "success"
    assert grel_app.json()["current_step"] == 1

    # 4. Cluster Detect & Apply
    clust_det = client.post("/api/cluster/detect", json={
        "session_id": session_id,
        "column": "status",
        "method": "fingerprint",
    })
    assert clust_det.status_code == 200
    c_data = clust_det.json()
    assert c_data["status"] == "success"
    assert "clusters" in c_data

    # 5. Entity Reconciliation Match
    rec_match = client.post("/api/reconcile/match", json={
        "session_id": session_id,
        "column": "status",
        "catalog": "countries",
    })
    assert rec_match.status_code == 200
    assert rec_match.json()["status"] == "success"

    # 6. Workflow Recipe History & Replay
    hist_res = client.get(f"/api/workflow/history?session_id={session_id}")
    assert hist_res.status_code == 200
    recipe = hist_res.json()["recipe"]
    assert recipe["operations_count"] >= 1

    replay_res = client.post("/api/workflow/apply_recipe", json={
        "session_id": session_id,
        "recipe": recipe,
    })
    assert replay_res.status_code == 200
    assert replay_res.json()["status"] == "success"



