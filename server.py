"""
FastAPI Service & Interactive Web Dashboard for AI Data Cleaner.

Provides:
- Complete REST API for profiling, planning, approval, cleaning, undo/rollback, and evaluation.
- Responsive, modern Web UI displaying distributions, one-click approvals, and side-by-side visual diffs.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from actions import execute_action
from ai_privacy import build_ai_safe_profile
from evaluator import evaluate_cleaning
from llm_providers import BaseLLMProvider, get_llm_provider
from planner import (
    apply_local_guardrails,
    build_cleaning_report,
    execute_cleaning_plan,
    generate_validated_plan,
    validate_cleaning_results,
)
from profiler import profile_dataset
from schemas import CleaningAction, CleaningPlan
from session import CleaningSession


app = FastAPI(
    title="AI Data Cleaner API & Dashboard",
    description="Privacy-Aware AI Planning, Deterministic Data Cleaning, Visual Diffs & Checkpoint Rollback",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for active cleaning sessions
SESSIONS: Dict[str, Dict[str, Any]] = {}
UPLOAD_DIR = os.path.join("data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sanitize_value_for_json(val: Any) -> Any:
    """Ensure numpy scalars, floating NaNs, and timestamps serialize cleanly to JSON."""
    if pd.isna(val):
        return None
    import numpy as np
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val) if not np.isnan(val) else None
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return str(val) if not isinstance(val, (str, list, dict)) else val


def compute_column_distributions(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute summary distributions (top categories, numeric stats) for each column."""
    distributions = {}
    for col in df.columns:
        series = df[col]
        total = len(series)
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total * 100), 2) if total > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))

        dist: Dict[str, Any] = {
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": unique_count,
            "is_numeric": bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)),
        }

        if dist["is_numeric"]:
            clean_s = series.dropna()
            if not clean_s.empty:
                dist["numeric_stats"] = {
                    "min": sanitize_value_for_json(clean_s.min()),
                    "max": sanitize_value_for_json(clean_s.max()),
                    "mean": round(float(clean_s.mean()), 2),
                    "median": sanitize_value_for_json(clean_s.median()),
                }

        # Top 5 most frequent values
        top_counts = series.value_counts(dropna=True).head(5)
        dist["top_values"] = [
            {"value": str(val), "count": int(count), "percentage": round((count / total * 100), 1)}
            for val, count in top_counts.items()
        ]
        distributions[col] = dist
    return distributions


def compute_side_by_side_diff(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    max_preview_rows: int = 100,
) -> Dict[str, Any]:
    """Compute cell-level diff and alignment between original and current DataFrame."""
    rows_before = len(df_before)
    rows_after = len(df_after)
    common_cols = [c for c in df_before.columns if c in df_after.columns]

    modified_cells_count = 0
    row_diffs = []

    preview_limit = min(rows_before, max_preview_rows)
    for idx in range(preview_limit):
        row_has_changes = False
        cells = {}

        if idx < rows_after:
            for col in common_cols:
                v_orig = df_before.iloc[idx][col]
                v_clean = df_after.iloc[idx][col]
                orig_null = pd.isna(v_orig)
                clean_null = pd.isna(v_clean)

                if orig_null and clean_null:
                    changed = False
                elif orig_null != clean_null:
                    changed = True
                else:
                    changed = str(v_orig) != str(v_clean)

                if changed:
                    row_has_changes = True
                    modified_cells_count += 1

                cells[col] = {
                    "original": sanitize_value_for_json(v_orig),
                    "cleaned": sanitize_value_for_json(v_clean),
                    "changed": changed,
                }
            status = "modified" if row_has_changes else "unchanged"
        else:
            status = "dropped"
            for col in df_before.columns:
                cells[col] = {
                    "original": sanitize_value_for_json(df_before.iloc[idx][col]),
                    "cleaned": None,
                    "changed": True,
                }

        row_diffs.append({
            "row_index": idx,
            "status": status,
            "cells": cells,
        })

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_delta": rows_after - rows_before,
        "columns_before": list(df_before.columns),
        "columns_after": list(df_after.columns),
        "total_cells_evaluated": preview_limit * len(common_cols),
        "modified_cells_count": modified_cells_count,
        "row_diffs": row_diffs,
    }


# ============================================================
# API MODELS
# ============================================================

class ProfileRequest(BaseModel):
    filepath: Optional[str] = None
    session_id: Optional[str] = None


class PlanRequest(BaseModel):
    session_id: str
    provider: Optional[str] = "gemini"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class CleanRequest(BaseModel):
    session_id: str
    approved_action_ids: List[int]


class EvaluateRequest(BaseModel):
    session_id: str
    ground_truth_path: Optional[str] = "data/benchmark/ground_truth.csv"


# ============================================================
# REST API ENDPOINTS
# ============================================================

@app.get("/api/health")
def get_health():
    """Health check endpoint and list of supported LLM providers."""
    return {
        "status": "healthy",
        "service": "AI Data Cleaner",
        "version": "2.0.0",
        "supported_providers": ["gemini", "openai", "anthropic", "ollama", "mock"],
        "active_sessions": len(SESSIONS),
    }


@app.get("/api/datasets")
def list_datasets():
    """Discover available CSV datasets in data/ and data/benchmark/."""
    datasets = []
    base_data = "data"
    for root, _, files in os.walk(base_data):
        for f in files:
            if f.endswith(".csv") and not f.startswith("cleaned_"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, start=".")
                try:
                    size_kb = round(os.path.getsize(full_path) / 1024, 1)
                    datasets.append({
                        "name": f,
                        "path": rel_path.replace("\\", "/"),
                        "size_kb": size_kb,
                        "is_benchmark": "benchmark" in rel_path,
                    })
                except Exception:
                    pass
    return {"datasets": sorted(datasets, key=lambda d: (not d["is_benchmark"], d["name"]))}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset and create an active session."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    file_id = uuid.uuid4().hex[:8]
    safe_name = f"{file_id}_{os.path.basename(file.filename)}"
    dest_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "Upload successful", "filepath": dest_path.replace("\\", "/")}


@app.post("/api/profile")
def api_profile(req: ProfileRequest):
    """Profile a dataset, compute distributions, and initialize session state."""
    filepath = req.filepath
    if not filepath and req.session_id and req.session_id in SESSIONS:
        filepath = SESSIONS[req.session_id]["filepath"]

    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")

    try:
        df, profile = profile_dataset(filepath)
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Profiling failed: {err}")

    session_id = req.session_id or str(uuid.uuid4())
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    cleaning_session = CleaningSession(df)
    distributions = compute_column_distributions(df)

    # Save to session cache
    SESSIONS[session_id] = {
        "session_id": session_id,
        "run_id": run_id,
        "run_dir": run_dir,
        "filepath": filepath,
        "df_original": df.copy(deep=True),
        "df_current": df.copy(deep=True),
        "profile": profile,
        "cleaning_session": cleaning_session,
        "plan": None,
        "executed_actions": [],
        "report": None,
    }

    # Format column preview list
    col_profiles = profile.get("column_profiles", {})
    columns_summary = []
    for col in df.columns:
        p = col_profiles.get(col, {})
        d = distributions.get(col, {})
        columns_summary.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "semantic_type": p.get("semantic_type", "unspecified"),
            "semantic_hints": p.get("semantic_hints", []),
            "missing_count": d.get("null_count", 0),
            "missing_pct": d.get("null_percentage", 0.0),
            "unique_count": d.get("unique_count", 0),
            "is_constant": p.get("constant_column", False),
            "is_empty": p.get("empty_column", False),
            "top_values": d.get("top_values", []),
            "numeric_stats": d.get("numeric_stats", None),
        })

    # Prepare preview rows
    preview_df = df.head(15).fillna("")
    preview_rows = preview_df.to_dict(orient="records")

    return {
        "session_id": session_id,
        "filepath": filepath,
        "dataset_name": os.path.basename(filepath),
        "summary": {
            "rows": len(df),
            "columns": len(df.columns),
            "duplicate_rows": profile.get("duplicate_rows", 0),
            "total_missing_cells": sum(d.get("null_count", 0) for d in distributions.values()),
        },
        "columns": columns_summary,
        "preview_rows": preview_rows,
    }


@app.post("/api/plan")
def api_plan(req: PlanRequest):
    """Generate a privacy-safe cleaning plan using the requested LLM provider."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Profile a dataset first.")

    session = SESSIONS[req.session_id]
    profile = session["profile"]
    df = session["df_original"]

    ai_safe_profile, token_to_column = build_ai_safe_profile(profile)
    from planner import build_planner_prompt
    prompt = build_planner_prompt(ai_safe_profile)

    try:
        provider = get_llm_provider(
            provider_name=req.provider,
            model_name=req.model,
            api_key=req.api_key,
            base_url=req.base_url,
        )
        plan, response_text = generate_validated_plan(
            provider=provider,
            initial_prompt=prompt,
            token_to_column=token_to_column,
            max_retries=2,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Planning failed ({req.provider}): {err}")

    # Apply guardrails
    plan = apply_local_guardrails(plan, df, profile)
    session["plan"] = plan

    actions_list = []
    for idx, action in enumerate(plan.actions):
        actions_list.append({
            "id": idx,
            "action": action.action,
            "column": action.column,
            "params": action.params,
            "reasoning": action.reasoning,
            "approved": True,  # Default approved
        })

    return {
        "session_id": req.session_id,
        "provider": provider.provider_name,
        "model": provider.model_name,
        "summary": plan.summary,
        "actions": actions_list,
    }


@app.post("/api/clean")
def api_clean(req: CleanRequest):
    """Execute user-approved cleaning actions and return visual diff."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = SESSIONS[req.session_id]
    plan = session.get("plan")
    if not plan:
        raise HTTPException(status_code=400, detail="No cleaning plan available. Generate a plan first.")

    df_original = session["df_original"]
    cleaning_session = CleaningSession(df_original)

    # Filter actions to approved only
    approved_set = set(req.approved_action_ids)
    selected_actions = [
        act for idx, act in enumerate(plan.actions)
        if idx in approved_set
    ]

    df_cleaned, executed_actions = execute_cleaning_plan(
        df=df_original.copy(deep=True),
        actions=selected_actions,
        session=cleaning_session,
    )

    session["df_current"] = df_cleaned
    session["cleaning_session"] = cleaning_session
    session["executed_actions"] = executed_actions

    # Compute validation & audit report
    df_before = df_original
    original_missing = df_before.isna().sum().to_dict()
    cleaned_missing = df_cleaned.isna().sum().to_dict()

    validation = validate_cleaning_results(
        df_before=df_before,
        df=df_cleaned,
        executed_actions=executed_actions,
        original_missing=original_missing,
        cleaned_missing=cleaned_missing,
        session_history=cleaning_session.get_history(),
    )

    report = build_cleaning_report(
        run_id=session["run_id"],
        input_file=session["filepath"],
        summary=plan.summary,
        executed_actions=executed_actions,
        validation=validation,
        df_before=df_before,
        df_cleaned=df_cleaned,
        profile=session["profile"],
        original_missing=original_missing,
        cleaned_missing=cleaned_missing,
        session_history=cleaning_session.get_history(),
    )
    session["report"] = report

    # Save cleaned file & report to disk
    cleaned_csv_path = os.path.join(session["run_dir"], "cleaned_data.csv")
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    session["cleaned_csv_path"] = cleaned_csv_path

    report_json_path = os.path.join(session["run_dir"], "cleaning_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    diff_data = compute_side_by_side_diff(df_before, df_cleaned)

    return {
        "session_id": req.session_id,
        "run_id": session["run_id"],
        "diff_summary": {
            "rows_before": diff_data["rows_before"],
            "rows_after": diff_data["rows_after"],
            "rows_delta": diff_data["rows_delta"],
            "modified_cells_count": diff_data["modified_cells_count"],
            "total_missing_before": sum(original_missing.values()),
            "total_missing_after": sum(cleaned_missing.values()),
        },
        "diff_table": diff_data["row_diffs"],
        "columns": diff_data["columns_after"],
        "checkpoints": cleaning_session.get_history(),
        "unresolved_issues": report.get("unresolved_issues", []),
        "validation": validation,
    }


@app.post("/api/session/undo")
def api_session_undo(req: Dict[str, str]):
    """Undo the last cleaning action step in the active session."""
    session_id = req.get("session_id")
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    sess = SESSIONS[session_id]["cleaning_session"]
    result = sess.undo()
    df_current = sess.get_current_df()
    SESSIONS[session_id]["df_current"] = df_current

    diff_data = compute_side_by_side_diff(SESSIONS[session_id]["df_original"], df_current)
    return {
        "status": result["status"],
        "message": result["message"],
        "current_step": sess.current_step,
        "checkpoints": sess.get_history(),
        "diff_summary": {
            "rows_before": diff_data["rows_before"],
            "rows_after": diff_data["rows_after"],
            "modified_cells_count": diff_data["modified_cells_count"],
        },
        "diff_table": diff_data["row_diffs"],
    }


@app.post("/api/session/redo")
def api_session_redo(req: Dict[str, str]):
    """Redo an undone step."""
    session_id = req.get("session_id")
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    sess = SESSIONS[session_id]["cleaning_session"]
    result = sess.redo()
    df_current = sess.get_current_df()
    SESSIONS[session_id]["df_current"] = df_current

    diff_data = compute_side_by_side_diff(SESSIONS[session_id]["df_original"], df_current)
    return {
        "status": result["status"],
        "message": result["message"],
        "current_step": sess.current_step,
        "checkpoints": sess.get_history(),
        "diff_summary": {
            "rows_before": diff_data["rows_before"],
            "rows_after": diff_data["rows_after"],
            "modified_cells_count": diff_data["modified_cells_count"],
        },
        "diff_table": diff_data["row_diffs"],
    }


@app.post("/api/session/rollback")
def api_session_rollback(req: Dict[str, str]):
    """Rollback all changes to Step 0 (original data state)."""
    session_id = req.get("session_id")
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    sess = SESSIONS[session_id]["cleaning_session"]
    result = sess.rollback_all()
    df_current = sess.get_current_df()
    SESSIONS[session_id]["df_current"] = df_current

    diff_data = compute_side_by_side_diff(SESSIONS[session_id]["df_original"], df_current)
    return {
        "status": result["status"],
        "message": result["message"],
        "current_step": 0,
        "checkpoints": sess.get_history(),
        "diff_summary": {
            "rows_before": diff_data["rows_before"],
            "rows_after": diff_data["rows_after"],
            "modified_cells_count": 0,
        },
        "diff_table": diff_data["row_diffs"],
    }


@app.post("/api/evaluate")
def api_evaluate(req: EvaluateRequest):
    """Run quality evaluation against a ground truth benchmark CSV."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = SESSIONS[req.session_id]
    gt_path = req.ground_truth_path or "data/benchmark/ground_truth.csv"
    if not os.path.exists(gt_path):
        raise HTTPException(status_code=404, detail=f"Ground truth CSV not found: {gt_path}")

    df_dirty = session["df_original"]
    df_cleaned = session["df_current"]
    try:
        df_truth = pd.read_csv(gt_path)
        eval_results = evaluate_cleaning(df_dirty=df_dirty, df_cleaned=df_cleaned, df_ground_truth=df_truth)
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {err}")

    session["evaluation"] = eval_results
    return eval_results


@app.get("/api/download/cleaned")
def download_cleaned(session_id: str = Query(...)):
    """Download cleaned CSV dataset."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[session_id]
    df = session["df_current"]

    temp_path = os.path.join(session["run_dir"], "cleaned_download.csv")
    df.to_csv(temp_path, index=False)
    return FileResponse(
        temp_path,
        media_type="text/csv",
        filename=f"cleaned_{os.path.basename(session['filepath'])}",
    )


@app.get("/api/download/report")
def download_report(session_id: str = Query(...)):
    """Download audit report JSON."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[session_id]
    report = session.get("report")
    if not report:
        raise HTTPException(status_code=400, detail="No cleaning report generated yet.")

    report_path = os.path.join(session["run_dir"], "cleaning_report.json")
    return FileResponse(
        report_path,
        media_type="application/json",
        filename=f"report_{session['run_id']}.json",
    )


# ============================================================
# INTERACTIVE SINGLE-PAGE DASHBOARD (HTML/CSS/JS)
# ============================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Data Cleaner - Interactive Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #0b0f19; color: #f3f4f6; }
    code, pre { font-family: 'JetBrains Mono', monospace; }
    .glass { background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .glass-card { background: rgba(31, 41, 55, 0.5); border: 1px solid rgba(255, 255, 255, 0.06); }
    .cell-modified { background-color: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #6ee7b7; font-weight: 600; }
    .cell-dropped { background-color: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
  </style>
</head>
<body class="min-h-screen flex flex-col">

  <!-- NAVBAR -->
  <header class="glass sticky top-0 z-50 px-6 py-4 flex items-center justify-between border-b border-gray-800">
    <div class="flex items-center space-x-3">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-purple-500 flex items-center justify-center text-xl shadow-lg shadow-indigo-500/20">
        🤖
      </div>
      <div>
        <h1 class="text-xl font-bold bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400 bg-clip-text text-transparent">
          AI Data Cleaner
        </h1>
        <p class="text-xs text-gray-400">Privacy-Safe AI Planning · Side-by-Side Diffs · Stepwise Rollback</p>
      </div>
    </div>
    <div class="flex items-center space-x-4">
      <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-950 text-emerald-400 border border-emerald-800">
        <span class="w-2 h-2 mr-2 rounded-full bg-emerald-400 animate-pulse"></span>
        Zero Raw-Data Leakage
      </span>
      <span id="session-badge" class="text-xs text-gray-400 hidden">Session: <span class="text-indigo-400 font-mono" id="current-session-id"></span></span>
    </div>
  </header>

  <!-- MAIN CONTAINER -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">

    <!-- TAB NAVIGATION -->
    <div class="flex border-b border-gray-800 space-x-2">
      <button onclick="switchTab('profile')" id="tab-btn-profile" class="px-5 py-2.5 font-medium text-sm rounded-t-lg transition border-b-2 border-indigo-500 text-indigo-400 bg-gray-900/40">
        1. Select & Profile
      </button>
      <button onclick="switchTab('plan')" id="tab-btn-plan" class="px-5 py-2.5 font-medium text-sm rounded-t-lg transition border-b-2 border-transparent text-gray-400 hover:text-gray-200">
        2. AI Plan & One-Click Approval
      </button>
      <button onclick="switchTab('diff')" id="tab-btn-diff" class="px-5 py-2.5 font-medium text-sm rounded-t-lg transition border-b-2 border-transparent text-gray-400 hover:text-gray-200">
        3. Visual Diff & Rollback
      </button>
      <button onclick="switchTab('eval')" id="tab-btn-eval" class="px-5 py-2.5 font-medium text-sm rounded-t-lg transition border-b-2 border-transparent text-gray-400 hover:text-gray-200">
        4. Benchmark Evaluation
      </button>
    </div>

    <!-- TAB 1: DATASET SELECT & PROFILE -->
    <section id="tab-profile" class="space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <!-- Dataset Picker -->
        <div class="glass p-5 rounded-2xl md:col-span-2 space-y-4">
          <h2 class="text-lg font-semibold text-white flex items-center">
            <span class="mr-2">📁</span> Choose CSV Dataset
          </h2>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label class="block text-xs text-gray-400 mb-1">Select Existing Dataset</label>
              <select id="dataset-select" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500">
                <option value="">Loading datasets...</option>
              </select>
            </div>
            <div>
              <label class="block text-xs text-gray-400 mb-1">Or Upload Custom CSV</label>
              <input type="file" id="file-upload" accept=".csv" class="w-full text-xs text-gray-400 file:mr-3 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-700 cursor-pointer">
            </div>
          </div>
          <button onclick="profileSelectedDataset()" id="btn-profile" class="w-full py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium rounded-xl text-sm transition shadow-lg shadow-indigo-600/20">
            📊 Analyze & Profile Dataset
          </button>
        </div>

        <!-- Quick Summary Card -->
        <div class="glass p-5 rounded-2xl flex flex-col justify-between">
          <h3 class="text-sm font-semibold text-gray-300">Dataset Overview</h3>
          <div class="grid grid-cols-2 gap-3 my-2" id="overview-metrics">
            <div class="glass-card p-3 rounded-xl">
              <span class="text-xs text-gray-400">Total Rows</span>
              <p class="text-xl font-bold text-indigo-300" id="stat-rows">-</p>
            </div>
            <div class="glass-card p-3 rounded-xl">
              <span class="text-xs text-gray-400">Columns</span>
              <p class="text-xl font-bold text-indigo-300" id="stat-cols">-</p>
            </div>
            <div class="glass-card p-3 rounded-xl">
              <span class="text-xs text-gray-400">Duplicates</span>
              <p class="text-xl font-bold text-amber-300" id="stat-dups">-</p>
            </div>
            <div class="glass-card p-3 rounded-xl">
              <span class="text-xs text-gray-400">Missing Cells</span>
              <p class="text-xl font-bold text-rose-300" id="stat-missing">-</p>
            </div>
          </div>
          <p class="text-xs text-gray-500">Local profiling with type inference and IQR outliers.</p>
        </div>
      </div>

      <!-- Column Health & Distribution Table -->
      <div class="glass rounded-2xl p-5 space-y-4" id="columns-section" style="display:none;">
        <div class="flex items-center justify-between">
          <h3 class="text-base font-semibold text-white">Column Health & Distributions</h3>
          <span class="text-xs text-gray-400">Click column for distribution details</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-sm text-gray-300">
            <thead class="text-xs uppercase bg-gray-950/60 text-gray-400 border-b border-gray-800">
              <tr>
                <th class="px-4 py-3">Column Name</th>
                <th class="px-4 py-3">Type</th>
                <th class="px-4 py-3">Inferred Semantic Type</th>
                <th class="px-4 py-3">Missing Values</th>
                <th class="px-4 py-3">Unique</th>
                <th class="px-4 py-3">Top Values / Range</th>
              </tr>
            </thead>
            <tbody id="columns-table-body" class="divide-y divide-gray-800/60"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 2: AI PLAN & ONE-CLICK APPROVAL -->
    <section id="tab-plan" class="space-y-6 hidden">
      <!-- Provider Config Panel -->
      <div class="glass p-5 rounded-2xl space-y-4">
        <h3 class="text-base font-semibold text-white flex items-center">
          <span class="mr-2">⚙️</span> LLM Planning Engine (Provider-Agnostic)
        </h3>
        <div class="grid grid-cols-1 sm:grid-cols-4 gap-4">
          <div>
            <label class="block text-xs text-gray-400 mb-1">Provider</label>
            <select id="llm-provider-select" onchange="updateModelDefaults()" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:outline-none focus:border-indigo-500">
              <option value="gemini">Google Gemini (Default)</option>
              <option value="openai">OpenAI (GPT-4o / Groq / vLLM)</option>
              <option value="anthropic">Anthropic Claude</option>
              <option value="ollama">Ollama (Offline / Air-Gapped)</option>
              <option value="mock">Deterministic Mock (Test Mode)</option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-gray-400 mb-1">Model Name</label>
            <input type="text" id="llm-model-input" value="gemini-2.5-flash" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:outline-none">
          </div>
          <div>
            <label class="block text-xs text-gray-400 mb-1">Custom Base URL (vLLM / Ollama)</label>
            <input type="text" id="llm-base-url" placeholder="http://localhost:11434" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:outline-none">
          </div>
          <div>
            <label class="block text-xs text-gray-400 mb-1">API Key Override (Optional)</label>
            <input type="password" id="llm-api-key" placeholder="Uses .env if blank" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:outline-none">
          </div>
        </div>
        <button onclick="generatePlan()" id="btn-generate-plan" class="w-full py-3 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-medium rounded-xl text-sm transition shadow-lg shadow-purple-600/20">
          🤖 Generate Cleaning Plan (Privacy-Preserved)
        </button>
      </div>

      <!-- Plan Recommendations & Approval Gate -->
      <div id="plan-section" class="glass p-5 rounded-2xl space-y-5 hidden">
        <div class="flex items-center justify-between border-b border-gray-800 pb-3">
          <div>
            <h3 class="text-base font-semibold text-white">Recommended Cleaning Actions</h3>
            <p id="plan-summary-text" class="text-xs text-gray-400 mt-1"></p>
          </div>
          <div class="flex space-x-2">
            <button onclick="toggleAllApprovals(true)" class="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-xs text-indigo-300 rounded-lg">Approve All</button>
            <button onclick="toggleAllApprovals(false)" class="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-xs text-gray-400 rounded-lg">Reject All</button>
          </div>
        </div>

        <div id="actions-container" class="space-y-3"></div>

        <button onclick="executeCleaning()" id="btn-execute-clean" class="w-full py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-medium rounded-xl text-sm transition shadow-lg shadow-emerald-600/20">
          🚀 Execute Approved Cleaning Plan
        </button>
      </div>
    </section>

    <!-- TAB 3: VISUAL DIFF & ROLLBACK -->
    <section id="tab-diff" class="space-y-6 hidden">
      <!-- Diff KPI Header -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4" id="diff-kpi-bar">
        <div class="glass p-4 rounded-xl">
          <span class="text-xs text-gray-400">Row Count Delta</span>
          <p class="text-lg font-bold text-white" id="diff-rows-delta">-</p>
        </div>
        <div class="glass p-4 rounded-xl">
          <span class="text-xs text-gray-400">Modified Cells</span>
          <p class="text-lg font-bold text-emerald-400" id="diff-modified-cells">-</p>
        </div>
        <div class="glass p-4 rounded-xl">
          <span class="text-xs text-gray-400">Missing Values Reduced</span>
          <p class="text-lg font-bold text-blue-400" id="diff-missing-delta">-</p>
        </div>
        <div class="glass p-4 rounded-xl flex items-center justify-around">
          <button onclick="downloadCleanedCsv()" class="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold rounded-lg text-white">📥 Download Cleaned CSV</button>
          <button onclick="downloadReportJson()" class="px-3 py-2 bg-gray-800 hover:bg-gray-700 text-xs font-semibold rounded-lg text-gray-300">📄 Report JSON</button>
        </div>
      </div>

      <!-- Rollback & Stepwise Undo Engine Controls -->
      <div class="glass p-4 rounded-2xl flex flex-wrap items-center justify-between gap-4">
        <div class="flex items-center space-x-2">
          <span class="text-xs font-semibold uppercase text-gray-400">Session Checkpoints:</span>
          <div id="checkpoints-timeline" class="flex items-center space-x-1.5 overflow-x-auto py-1"></div>
        </div>
        <div class="flex items-center space-x-2">
          <button onclick="undoStep()" class="px-3 py-1.5 bg-amber-950 hover:bg-amber-900 text-amber-300 text-xs rounded-lg border border-amber-800">
            ⏮ Undo Step
          </button>
          <button onclick="redoStep()" class="px-3 py-1.5 bg-indigo-950 hover:bg-indigo-900 text-indigo-300 text-xs rounded-lg border border-indigo-800">
            ⏭ Redo Step
          </button>
          <button onclick="rollbackAll()" class="px-3 py-1.5 bg-rose-950 hover:bg-rose-900 text-rose-300 text-xs rounded-lg border border-rose-800">
            ⏪ Rollback All
          </button>
        </div>
      </div>

      <!-- Side-by-Side Diff Table -->
      <div class="glass rounded-2xl p-5 space-y-4">
        <div class="flex items-center justify-between">
          <h3 class="text-base font-semibold text-white">Side-by-Side Visual Diff (First 50 Rows)</h3>
          <span class="text-xs text-emerald-400 bg-emerald-950 px-2.5 py-1 rounded-full border border-emerald-800">
            Green cells indicate validated fixes
          </span>
        </div>
        <div class="overflow-x-auto max-h-[550px] overflow-y-auto">
          <table class="w-full text-left text-xs text-gray-300 border-collapse">
            <thead class="sticky top-0 bg-gray-950 text-gray-400 uppercase text-[11px] border-b border-gray-800">
              <tr id="diff-table-header"></tr>
            </thead>
            <tbody id="diff-table-body" class="divide-y divide-gray-800/60 font-mono"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 4: BENCHMARK QUALITY EVALUATION -->
    <section id="tab-eval" class="space-y-6 hidden">
      <div class="glass p-5 rounded-2xl space-y-4">
        <h3 class="text-base font-semibold text-white flex items-center">
          <span class="mr-2">🎯</span> Ground-Truth Cleaning Quality Evaluation
        </h3>
        <p class="text-xs text-gray-400">
          Scores cleaning precision, recall, and F1 score against a known clean ground-truth benchmark. Penalizes both over-cleaning (false positives) and missed errors (false negatives).
        </p>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div class="sm:col-span-2">
            <label class="block text-xs text-gray-400 mb-1">Ground Truth Dataset</label>
            <input type="text" id="eval-gt-path" value="data/benchmark/ground_truth.csv" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:outline-none">
          </div>
          <div class="flex items-end">
            <button onclick="runBenchmarkEvaluation()" id="btn-run-eval" class="w-full py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-medium rounded-xl text-sm transition shadow-lg shadow-emerald-600/20">
              Evaluate Quality Scorecard
            </button>
          </div>
        </div>
      </div>

      <!-- Evaluation Scorecard Results -->
      <div id="eval-results-card" class="glass p-6 rounded-2xl space-y-6 hidden">
        <h4 class="text-sm font-semibold uppercase text-gray-400">Cleaning Quality Scorecard</h4>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div class="glass-card p-4 rounded-xl text-center">
            <span class="text-xs text-gray-400">Precision</span>
            <p class="text-3xl font-extrabold text-emerald-400 my-1" id="eval-precision">-%</p>
            <span class="text-[10px] text-gray-500">Repairs / (Repairs + Over-cleaned)</span>
          </div>
          <div class="glass-card p-4 rounded-xl text-center">
            <span class="text-xs text-gray-400">Recall</span>
            <p class="text-3xl font-extrabold text-indigo-400 my-1" id="eval-recall">-%</p>
            <span class="text-[10px] text-gray-500">Repairs / Total Errors</span>
          </div>
          <div class="glass-card p-4 rounded-xl text-center">
            <span class="text-xs text-gray-400">F1 Score</span>
            <p class="text-3xl font-extrabold text-purple-400 my-1" id="eval-f1">-%</p>
            <span class="text-[10px] text-gray-500">Harmonic Mean</span>
          </div>
          <div class="glass-card p-4 rounded-xl text-center">
            <span class="text-xs text-gray-400">Accuracy Delta</span>
            <p class="text-3xl font-extrabold text-teal-400 my-1" id="eval-acc-delta">-%</p>
            <span class="text-[10px] text-gray-500">Overall Accuracy Gain</span>
          </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <div class="p-3 bg-emerald-950/40 rounded-xl border border-emerald-900/50">
            <span class="text-gray-400">True Positives (Repairs):</span>
            <p class="text-base font-bold text-emerald-300" id="eval-tp">-</p>
          </div>
          <div class="p-3 bg-rose-950/40 rounded-xl border border-rose-900/50">
            <span class="text-gray-400">False Positives (Over-cleaned):</span>
            <p class="text-base font-bold text-rose-300" id="eval-fp">-</p>
          </div>
          <div class="p-3 bg-amber-950/40 rounded-xl border border-amber-900/50">
            <span class="text-gray-400">False Negatives (Missed):</span>
            <p class="text-base font-bold text-amber-300" id="eval-fn">-</p>
          </div>
          <div class="p-3 bg-blue-950/40 rounded-xl border border-blue-900/50">
            <span class="text-gray-400">True Negatives (Preserved):</span>
            <p class="text-base font-bold text-blue-300" id="eval-tn">-</p>
          </div>
        </div>
      </div>
    </section>

  </main>

  <!-- JAVASCRIPT APP LOGIC -->
  <script>
    let currentSessionId = null;
    let currentActions = [];

    // On Load: fetch datasets
    window.addEventListener('DOMContentLoaded', async () => {
      try {
        const res = await fetch('/api/datasets');
        const data = await res.json();
        const sel = document.getElementById('dataset-select');
        sel.innerHTML = '<option value="">-- Choose a CSV Dataset --</option>';
        data.datasets.forEach(d => {
          const opt = document.createElement('option');
          opt.value = d.path;
          opt.textContent = `${d.is_benchmark ? '⭐ [Benchmark] ' : ''}${d.name} (${d.size_kb} KB)`;
          sel.appendChild(opt);
        });
      } catch (err) {
        console.error('Failed to load datasets:', err);
      }
    });

    function switchTab(tab) {
      ['profile', 'plan', 'diff', 'eval'].forEach(t => {
        document.getElementById(`tab-${t}`).classList.add('hidden');
        const btn = document.getElementById(`tab-btn-${t}`);
        btn.classList.remove('border-indigo-500', 'text-indigo-400', 'bg-gray-900/40');
        btn.classList.add('border-transparent', 'text-gray-400');
      });
      document.getElementById(`tab-${tab}`).classList.remove('hidden');
      const activeBtn = document.getElementById(`tab-btn-${tab}`);
      activeBtn.classList.remove('border-transparent', 'text-gray-400');
      activeBtn.classList.add('border-indigo-500', 'text-indigo-400', 'bg-gray-900/40');
    }

    function updateModelDefaults() {
      const p = document.getElementById('llm-provider-select').value;
      const m = document.getElementById('llm-model-input');
      const b = document.getElementById('llm-base-url');
      if (p === 'gemini') { m.value = 'gemini-2.5-flash'; b.placeholder = ''; }
      else if (p === 'openai') { m.value = 'gpt-4o-mini'; b.placeholder = 'https://api.openai.com/v1 (or Groq/vLLM)'; }
      else if (p === 'anthropic') { m.value = 'claude-3-5-haiku-20241022'; b.placeholder = ''; }
      else if (p === 'ollama') { m.value = 'llama3'; b.value = 'http://localhost:11434'; }
      else if (p === 'mock') { m.value = 'mock-planner'; }
    }

    async function profileSelectedDataset() {
      const fileInput = document.getElementById('file-upload');
      let filepath = document.getElementById('dataset-select').value;

      if (fileInput.files.length > 0) {
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        const uploadRes = await fetch('/api/upload', { method: 'POST', body: formData });
        const uploadData = await uploadRes.json();
        filepath = uploadData.filepath;
      }

      if (!filepath) {
        alert('Please select or upload a CSV file.');
        return;
      }

      const btn = document.getElementById('btn-profile');
      btn.textContent = '⏳ Profiling Dataset...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/profile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filepath })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Profiling failed');

        currentSessionId = data.session_id;
        document.getElementById('current-session-id').textContent = currentSessionId.slice(0, 8);
        document.getElementById('session-badge').classList.remove('hidden');

        // Overview stats
        document.getElementById('stat-rows').textContent = data.summary.rows.toLocaleString();
        document.getElementById('stat-cols').textContent = data.summary.columns;
        document.getElementById('stat-dups').textContent = data.summary.duplicate_rows;
        document.getElementById('stat-missing').textContent = data.summary.total_missing_cells.toLocaleString();

        // Columns table
        const tbody = document.getElementById('columns-table-body');
        tbody.innerHTML = '';
        data.columns.forEach(col => {
          const tr = document.createElement('tr');
          tr.className = 'hover:bg-gray-900/40 transition';
          const badgeColor = col.semantic_type !== 'unspecified' ? 'bg-indigo-950 text-indigo-300 border-indigo-800' : 'bg-gray-800 text-gray-400 border-gray-700';

          tr.innerHTML = `
            <td class="px-4 py-3 font-medium text-white">${col.name}</td>
            <td class="px-4 py-3"><span class="px-2 py-0.5 rounded text-[11px] bg-gray-900 text-gray-400 border border-gray-800">${col.dtype}</span></td>
            <td class="px-4 py-3"><span class="px-2 py-0.5 rounded text-[11px] border ${badgeColor}">${col.semantic_type}</span></td>
            <td class="px-4 py-3">
              <div class="flex items-center space-x-2">
                <div class="w-16 bg-gray-800 rounded-full h-1.5 overflow-hidden">
                  <div class="bg-rose-500 h-1.5 rounded-full" style="width: ${Math.min(col.missing_pct, 100)}%"></div>
                </div>
                <span class="text-xs text-gray-400">${col.missing_count} (${col.missing_pct}%)</span>
              </div>
            </td>
            <td class="px-4 py-3 text-xs text-gray-400">${col.unique_count}</td>
            <td class="px-4 py-3 text-xs text-gray-400">
              ${col.numeric_stats ? `[${col.numeric_stats.min} ... ${col.numeric_stats.max}], med: ${col.numeric_stats.median}` : (col.top_values.map(v => v.value).slice(0, 3).join(', ') || '-')}
            </td>
          `;
          tbody.appendChild(tr);
        });
        document.getElementById('columns-section').style.display = 'block';

      } catch (err) {
        alert(err.message);
      } finally {
        btn.textContent = '📊 Analyze & Profile Dataset';
        btn.disabled = false;
      }
    }

    async function generatePlan() {
      if (!currentSessionId) {
        alert('Please profile a dataset first.');
        return;
      }
      const provider = document.getElementById('llm-provider-select').value;
      const model = document.getElementById('llm-model-input').value;
      const baseUrl = document.getElementById('llm-base-url').value;
      const apiKey = document.getElementById('llm-api-key').value;

      const btn = document.getElementById('btn-generate-plan');
      btn.textContent = '⏳ Asking LLM for Cleaning Plan...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: currentSessionId,
            provider,
            model,
            base_url: baseUrl || null,
            api_key: apiKey || null,
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Plan generation failed');

        currentActions = data.actions;
        document.getElementById('plan-summary-text').textContent = data.summary;

        const container = document.getElementById('actions-container');
        container.innerHTML = '';

        if (currentActions.length === 0) {
          container.innerHTML = '<div class="p-4 bg-gray-900 rounded-xl text-xs text-gray-400">No actions recommended. Dataset is clean!</div>';
        } else {
          currentActions.forEach(act => {
            const card = document.createElement('div');
            card.className = 'glass-card p-4 rounded-xl flex items-start space-x-3 transition hover:border-gray-700';
            card.innerHTML = `
              <input type="checkbox" id="action-chk-${act.id}" checked class="mt-1 w-4 h-4 text-indigo-600 rounded bg-gray-900 border-gray-700 focus:ring-0 cursor-pointer">
              <div class="flex-1 space-y-1">
                <div class="flex items-center space-x-2">
                  <span class="px-2 py-0.5 rounded text-xs font-semibold bg-indigo-950 text-indigo-300 border border-indigo-800">${act.action}</span>
                  ${act.column ? `<span class="px-2 py-0.5 rounded text-xs bg-gray-900 text-gray-300 font-mono">${act.column}</span>` : ''}
                  <span class="text-[11px] text-gray-500 font-mono">${JSON.stringify(act.params)}</span>
                </div>
                <p class="text-xs text-gray-400">${act.reasoning}</p>
              </div>
            `;
            container.appendChild(card);
          });
        }
        document.getElementById('plan-section').classList.remove('hidden');
      } catch (err) {
        alert(err.message);
      } finally {
        btn.textContent = '🤖 Generate Cleaning Plan (Privacy-Preserved)';
        btn.disabled = false;
      }
    }

    function toggleAllApprovals(checked) {
      currentActions.forEach(act => {
        const el = document.getElementById(`action-chk-${act.id}`);
        if (el) el.checked = checked;
      });
    }

    async function executeCleaning() {
      const approvedIds = currentActions
        .filter(act => {
          const el = document.getElementById(`action-chk-${act.id}`);
          return el && el.checked;
        })
        .map(act => act.id);

      const btn = document.getElementById('btn-execute-clean');
      btn.textContent = '⏳ Executing Approved Cleaning...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/clean', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: currentSessionId,
            approved_action_ids: approvedIds,
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Cleaning execution failed');

        renderDiffView(data);
        switchTab('diff');
      } catch (err) {
        alert(err.message);
      } finally {
        btn.textContent = '🚀 Execute Approved Cleaning Plan';
        btn.disabled = false;
      }
    }

    function renderDiffView(data) {
      document.getElementById('diff-rows-delta').textContent = `${data.diff_summary.rows_before} ➔ ${data.diff_summary.rows_after}`;
      document.getElementById('diff-modified-cells').textContent = data.diff_summary.modified_cells_count.toLocaleString();
      document.getElementById('diff-missing-delta').textContent = `${data.diff_summary.total_missing_before} ➔ ${data.diff_summary.total_missing_after}`;

      // Checkpoint timeline
      const timeline = document.getElementById('checkpoints-timeline');
      timeline.innerHTML = '';
      (data.checkpoints || []).forEach(cp => {
        const pill = document.createElement('span');
        pill.className = 'px-2 py-0.5 rounded text-[11px] bg-gray-900 border border-gray-800 text-gray-300';
        pill.textContent = cp.step === 0 ? 'Step 0 (Original)' : `Step ${cp.step}: ${cp.action_name}`;
        timeline.appendChild(pill);
      });

      // Diff table header
      const thead = document.getElementById('diff-table-header');
      thead.innerHTML = '<th class="px-3 py-2">#</th>';
      data.columns.forEach(col => {
        const th = document.createElement('th');
        th.className = 'px-3 py-2 font-mono';
        th.textContent = col;
        thead.appendChild(th);
      });

      // Diff table body
      const tbody = document.getElementById('diff-table-body');
      tbody.innerHTML = '';
      data.diff_table.forEach(r => {
        const tr = document.createElement('tr');
        tr.className = r.status === 'modified' ? 'bg-emerald-950/20' : (r.status === 'dropped' ? 'bg-rose-950/20' : '');

        let rowHtml = `<td class="px-3 py-1.5 text-gray-500">${r.row_index}</td>`;
        data.columns.forEach(col => {
          const cell = r.cells[col] || { original: '', cleaned: '', changed: false };
          if (cell.changed) {
            rowHtml += `<td class="px-3 py-1.5 cell-modified" title="Original: ${cell.original}">${cell.cleaned !== null ? cell.cleaned : 'NaN'}</td>`;
          } else {
            rowHtml += `<td class="px-3 py-1.5 text-gray-300">${cell.cleaned !== null ? cell.cleaned : ''}</td>`;
          }
        });
        tr.innerHTML = rowHtml;
        tbody.appendChild(tr);
      });
    }

    async function undoStep() {
      if (!currentSessionId) return;
      try {
        const res = await fetch('/api/session/undo', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: currentSessionId })
        });
        const data = await res.json();
        if (data.status === 'undone') {
          alert(data.message);
          renderDiffView(data);
        } else {
          alert(data.message);
        }
      } catch (err) {
        alert(err.message);
      }
    }

    async function redoStep() {
      if (!currentSessionId) return;
      try {
        const res = await fetch('/api/session/redo', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: currentSessionId })
        });
        const data = await res.json();
        if (data.status === 'redone') {
          alert(data.message);
          renderDiffView(data);
        } else {
          alert(data.message);
        }
      } catch (err) {
        alert(err.message);
      }
    }

    async function rollbackAll() {
      if (!currentSessionId || !confirm('Revert all actions back to original dataset?')) return;
      try {
        const res = await fetch('/api/session/rollback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: currentSessionId })
        });
        const data = await res.json();
        alert(data.message);
        renderDiffView(data);
      } catch (err) {
        alert(err.message);
      }
    }

    async function runBenchmarkEvaluation() {
      if (!currentSessionId) {
        alert('Please profile and clean a dataset first.');
        return;
      }
      const gtPath = document.getElementById('eval-gt-path').value;
      const btn = document.getElementById('btn-run-eval');
      btn.textContent = '⏳ Evaluating...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/evaluate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: currentSessionId,
            ground_truth_path: gtPath,
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Evaluation failed');

        const ov = data.overall;
        document.getElementById('eval-precision').textContent = `${(ov.precision * 100).toFixed(1)}%`;
        document.getElementById('eval-recall').textContent = `${(ov.recall * 100).toFixed(1)}%`;
        document.getElementById('eval-f1').textContent = `${(ov.f1 * 100).toFixed(1)}%`;
        document.getElementById('eval-acc-delta').textContent = `+${(ov.accuracy_delta * 100).toFixed(1)}%`;

        document.getElementById('eval-tp').textContent = ov.true_positives;
        document.getElementById('eval-fp').textContent = ov.false_positives;
        document.getElementById('eval-fn').textContent = ov.false_negatives;
        document.getElementById('eval-tn').textContent = ov.true_negatives;

        document.getElementById('eval-results-card').classList.remove('hidden');
      } catch (err) {
        alert(err.message);
      } finally {
        btn.textContent = 'Evaluate Quality Scorecard';
        btn.disabled = false;
      }
    }

    function downloadCleanedCsv() {
      if (!currentSessionId) return;
      window.location.href = `/api/download/cleaned?session_id=${currentSessionId}`;
    }

    function downloadReportJson() {
      if (!currentSessionId) return;
      window.location.href = `/api/download/report?session_id=${currentSessionId}`;
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def index_page():
    """Serve the interactive single-page web dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AI Data Cleaner Web Dashboard & API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  AI Data Cleaner Dashboard & API Server")
    print(f"  Running at: http://{args.host}:{args.port}")
    print(f"=======================================================\n")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
