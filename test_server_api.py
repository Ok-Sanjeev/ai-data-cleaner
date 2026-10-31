import json
import os
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
