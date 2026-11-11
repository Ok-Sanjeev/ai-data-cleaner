"""
FastAPI Service & Interactive Web Dashboard for AI Data Cleaner.

Provides:
- Complete REST API for profiling, planning, approval, cleaning, undo/rollback, and evaluation.
- Responsive, modern Web UI displaying distributions, one-click approvals, and side-by-side visual diffs.
"""

import json
import os
import re
import secrets
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
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
from schemas import (
    CleaningAction,
    CleaningPlan,
    ClusterDetectRequest,
    ClusterApplyRequest,
    GRELPreviewRequest,
    GRELApplyRequest,
    FacetComputeRequest,
    ExploreRequest,
    ReconcileRequest,
    ReconcileApplyRequest,
    RecipeApplyRequest,
)
from session import CleaningSession
from openrefine_clustering import detect_clusters, SUPPORTED_CLUSTERING_METHODS
from grel_engine import preview_grel, execute_grel_transform
from facets import (
    compute_text_facet,
    compute_numeric_facet,
    compute_date_facet,
    compute_facet_auto,
    explore_dataset,
    apply_filters,
)
from reconciliation import reconcile_column_candidates, REFERENCE_CATALOGS
from workflow_recipe import WorkflowRecipeManager


app = FastAPI(
    title="DataRefine — AI-Powered Data Preparation & Quality Platform",
    description="Industrial Data Wrangling, Built Natively in Python. Deterministic Data Cleaning, Visual Diffs & Checkpoint Rollback",
    version="2.0.0",
)

# Security Constants & Approved Base Directories
BASE_DIR = Path(__file__).resolve().parent
ALLOWED_DATA_DIRS = [
    (BASE_DIR / "data").resolve(),
    (BASE_DIR / "data" / "uploads").resolve(),
    (BASE_DIR / "data" / "benchmark").resolve(),
    (BASE_DIR / "data" / "robustness_tests").resolve(),
]

MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_DATASET_ROWS = 500_000
MAX_DATASET_COLUMNS = 500
SESSION_MAX_AGE_SECONDS = 4 * 3600  # 4 hours
MAX_ACTIVE_SESSIONS = 50

# Standards-compliant CORS configuration
ALLOWED_ORIGINS = [orig.strip() for orig in os.getenv("ALLOWED_ORIGINS", "*").split(",") if orig.strip()]
ALLOW_CREDENTIALS = True if "*" not in ALLOWED_ORIGINS else False

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# In-memory store for active cleaning sessions
SESSIONS: Dict[str, Dict[str, Any]] = {}
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

UPLOAD_DIR = os.path.join("data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def resolve_safe_dataset_path(client_path: str) -> Path:
    """
    Safely resolve a client-supplied dataset filepath.
    Enforces strict confinement to approved dataset directories.
    Prevents path traversal, arbitrary file reading, and invalid extensions.
    """
    if not client_path or not isinstance(client_path, str):
        raise HTTPException(status_code=400, detail="Invalid filepath provided.")

    if "\x00" in client_path:
        raise HTTPException(status_code=400, detail="Path contains invalid characters.")

    raw_path = Path(client_path.strip())
    if not raw_path.is_absolute():
        resolved_path = (BASE_DIR / raw_path).resolve()
    else:
        resolved_path = raw_path.resolve()

    # Jail check: must reside inside one of the approved directories
    is_allowed = False
    for allowed_root in ALLOWED_DATA_DIRS:
        try:
            resolved_path.relative_to(allowed_root)
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        raise HTTPException(
            status_code=403,
            detail="Access forbidden: Path traverses outside approved data directories."
        )

    if not resolved_path.exists() or not resolved_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {client_path}")

    if resolved_path.suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    return resolved_path


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent directory traversal and unsafe characters."""
    base = os.path.basename(filename)
    sanitized = re.sub(r"[^\w\.\-]", "_", base)
    if not sanitized.lower().endswith(".csv"):
        sanitized += ".csv"
    return sanitized


def _cleanup_stale_sessions() -> None:
    """Prune expired sessions to prevent unbounded memory growth."""
    now = datetime.now().timestamp()
    expired = []
    for sid, sdata in SESSIONS.items():
        created_time = sdata.get("created_timestamp", now)
        if now - created_time > SESSION_MAX_AGE_SECONDS:
            expired.append(sid)
    for sid in expired:
        SESSIONS.pop(sid, None)

    if len(SESSIONS) > MAX_ACTIVE_SESSIONS:
        sorted_sessions = sorted(
            SESSIONS.items(),
            key=lambda item: item[1].get("created_timestamp", 0)
        )
        for sid, _ in sorted_sessions[:len(SESSIONS) - MAX_ACTIVE_SESSIONS]:
            SESSIONS.pop(sid, None)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sanitize_value_for_json(val: Any) -> Any:
    """Ensure numpy scalars, floating NaNs, and timestamps serialize cleanly to JSON."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val) if not np.isnan(val) else None
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return str(val) if not isinstance(val, (str, list, dict)) else val


def sanitize_obj_for_json(obj: Any) -> Any:
    """Recursively sanitize dicts, lists, numpy types, and float NaNs for JSON compliance."""
    if isinstance(obj, dict):
        return {
            ("null" if (pd.isna(k) or k is None) else str(k)): sanitize_obj_for_json(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [sanitize_obj_for_json(item) for item in obj]
    return sanitize_value_for_json(obj)


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

    # Build column mapping between df_after and df_before
    # Handles column renaming (e.g. normalize_column_names to snake_case)
    col_map = {}
    cols_before_norm = {
        str(c).lower().strip().replace(" ", "_").replace("-", "_"): c
        for c in df_before.columns
    }
    for idx, col_after in enumerate(df_after.columns):
        norm_key = str(col_after).lower().strip().replace(" ", "_").replace("-", "_")
        if col_after in df_before.columns:
            col_map[col_after] = col_after
        elif norm_key in cols_before_norm:
            col_map[col_after] = cols_before_norm[norm_key]
        elif len(df_after.columns) == len(df_before.columns) and idx < len(df_before.columns):
            col_map[col_after] = df_before.columns[idx]
        else:
            col_map[col_after] = None

    modified_cells_count = 0
    row_diffs = []

    # Check whether we can use index alignment for exact row tracking
    can_use_index = (
        len(df_after) <= len(df_before)
        and df_after.index.isin(df_before.index).all()
        and df_before.index.is_unique
        and df_after.index.is_unique
    )

    preview_limit = min(rows_before, max_preview_rows)
    for idx in range(preview_limit):
        row_has_changes = False
        cells = {}

        if can_use_index:
            row_key = df_before.index[idx]
            row_orig = df_before.loc[row_key]
            is_dropped = row_key not in df_after.index
            row_clean = df_after.loc[row_key] if not is_dropped else None
            row_display_idx = int(row_key) if isinstance(row_key, (int, np.integer)) else idx
        else:
            row_orig = df_before.iloc[idx]
            is_dropped = idx >= rows_after
            row_clean = df_after.iloc[idx] if not is_dropped else None
            row_display_idx = idx

        if not is_dropped and row_clean is not None:
            for col_after in df_after.columns:
                col_before = col_map.get(col_after)
                v_orig = (
                    row_orig[col_before]
                    if (col_before is not None and col_before in row_orig.index)
                    else None
                )
                v_clean = row_clean[col_after]

                orig_null = pd.isna(v_orig) or v_orig is None
                clean_null = pd.isna(v_clean) or v_clean is None

                if orig_null and clean_null:
                    changed = False
                elif orig_null != clean_null:
                    changed = True
                else:
                    try:
                        if isinstance(v_orig, (float, np.floating)) and isinstance(v_clean, (float, np.floating)):
                            changed = not np.isclose(v_orig, v_clean)
                        else:
                            changed = str(v_orig).strip() != str(v_clean).strip()
                    except Exception:
                        changed = str(v_orig) != str(v_clean)

                if changed:
                    row_has_changes = True
                    modified_cells_count += 1

                orig_json = sanitize_value_for_json(v_orig)
                clean_json = sanitize_value_for_json(v_clean)

                cells[col_after] = {
                    "original": orig_json,
                    "cleaned": clean_json,
                    "changed": changed,
                    "value": clean_json,
                    "original_value": orig_json,
                    "status": "modified" if changed else "unchanged",
                }
            status = "modified" if row_has_changes else "unchanged"
        else:
            status = "dropped"
            for col_after in df_after.columns:
                col_before = col_map.get(col_after)
                v_orig = (
                    row_orig[col_before]
                    if (col_before is not None and col_before in row_orig.index)
                    else None
                )
                orig_json = sanitize_value_for_json(v_orig)
                cells[col_after] = {
                    "original": orig_json,
                    "cleaned": None,
                    "changed": True,
                    "value": None,
                    "original_value": orig_json,
                    "status": "dropped",
                }

        row_diffs.append({
            "row_index": row_display_idx,
            "status": status,
            "cells": cells,
        })

    # Exact calculation of total modified cells across the ENTIRE dataset (not just preview rows)
    dataset_total_modified_cells = 0
    for col_after, col_before in col_map.items():
        if col_before is not None and col_before in df_before.columns and col_after in df_after.columns:
            s_before = df_before[col_before]
            s_after = df_after[col_after]
            if can_use_index:
                s_before_aligned = s_before.loc[df_after.index]
                s_after_aligned = s_after
            else:
                common_len = min(len(s_before), len(s_after))
                s_before_aligned = s_before.iloc[:common_len].reset_index(drop=True)
                s_after_aligned = s_after.iloc[:common_len].reset_index(drop=True)

            na_both = s_before_aligned.isna() & s_after_aligned.isna()
            if s_before_aligned.dtype == s_after_aligned.dtype and pd.api.types.is_numeric_dtype(s_before_aligned):
                diff = (s_before_aligned != s_after_aligned) & ~na_both
            else:
                diff = (s_before_aligned.astype(str) != s_after_aligned.astype(str)) & ~na_both
            dataset_total_modified_cells += int(diff.sum())

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_delta": rows_after - rows_before,
        "columns_before": list(df_before.columns),
        "columns_after": list(df_after.columns),
        "total_cells_evaluated": rows_before * len(df_after.columns),
        "modified_cells_count": dataset_total_modified_cells,
        "preview_modified_cells_count": modified_cells_count,
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


class AddActionRequest(BaseModel):
    session_id: str
    action: str
    column: Optional[str] = None
    params: Optional[Dict[str, Any]] = {}
    reasoning: Optional[str] = "User-configured operation"


class RemoveActionRequest(BaseModel):
    session_id: str
    action_index: int


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
        "service": "DataRefine (AI Data Cleaner)",
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
    """Upload a CSV dataset with strict size, filename, and format security validation."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are supported.")

    safe_name = sanitize_filename(file.filename)
    file_id = secrets.token_hex(6)
    dest_filename = f"{file_id}_{safe_name}"
    dest_path = (Path(UPLOAD_DIR) / dest_filename).resolve()

    bytes_read = 0
    chunk_size = 1024 * 1024  # 1 MB chunk
    try:
        with open(dest_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > MAX_UPLOAD_SIZE_BYTES:
                    buffer.close()
                    if dest_path.exists():
                        dest_path.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds maximum allowed size ({MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB)."
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Validate that it is parseable as CSV
    try:
        sample_df = pd.read_csv(dest_path, nrows=5)
        if sample_df.empty and sample_df.columns.empty:
            dest_path.unlink()
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
    except Exception as err:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted CSV format: {err}")

    rel_path = str(dest_path.relative_to(BASE_DIR)).replace("\\", "/")
    return {
        "message": "Upload successful",
        "filepath": rel_path,
        "filename": safe_name,
        "size_bytes": bytes_read,
    }


@app.post("/api/profile")
def api_profile(req: ProfileRequest):
    """Profile a dataset, compute distributions, and initialize session state."""
    filepath_str = req.filepath
    if not filepath_str and req.session_id and req.session_id in SESSIONS:
        filepath_str = SESSIONS[req.session_id]["filepath"]

    if not filepath_str:
        raise HTTPException(status_code=400, detail="Filepath or active session_id is required.")

    # Validate and securely resolve path within approved dataset directories
    safe_path = resolve_safe_dataset_path(filepath_str)

    try:
        df, profile = profile_dataset(str(safe_path))
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Profiling failed: {err}")

    # Enforce row and column safety limits
    if len(df) > MAX_DATASET_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset has {len(df):,} rows, exceeding maximum supported limit of {MAX_DATASET_ROWS:,} rows."
        )
    if len(df.columns) > MAX_DATASET_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset has {len(df.columns)} columns, exceeding maximum supported limit of {MAX_DATASET_COLUMNS} columns."
        )

    _cleanup_stale_sessions()

    session_id = req.session_id or str(uuid.uuid4())
    session_token = secrets.token_urlsafe(32)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    cleaning_session = CleaningSession(df.copy(deep=True))
    distributions = compute_column_distributions(df)

    # Save to session cache
    SESSIONS[session_id] = {
        "session_id": session_id,
        "session_token": session_token,
        "created_timestamp": datetime.now().timestamp(),
        "run_id": run_id,
        "run_dir": run_dir,
        "filepath": str(safe_path),
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

    # Prepare preview rows (clean serialization)
    preview_df = df.head(10).fillna("")
    preview_records = []
    for _, row in preview_df.iterrows():
        preview_records.append({col: sanitize_value_for_json(val) for col, val in row.items()})

    total_missing = sum(d.get("null_count", 0) for d in distributions.values())
    total_cells = len(df) * len(df.columns)
    missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0

    return {
        "session_id": session_id,
        "filepath": str(safe_path).replace("\\", "/"),
        "dataset_name": safe_path.name,
        "summary": {
            "rows": len(df),
            "columns": len(df.columns),
            "duplicate_rows": profile.get("duplicate_rows", 0),
            "total_missing_cells": total_missing,
            "missing_cells_percentage": missing_pct,
            "delimiter": profile.get("delimiter", ","),
            "encoding": profile.get("encoding", "utf-8"),
        },
        "columns": columns_summary,
        "preview_rows": preview_records,
        "profile": sanitize_obj_for_json(profile),
        "quality_categories_summary": sanitize_obj_for_json(profile.get("quality_categories_summary", {})),
    }


@app.get("/api/actions/registry")
def api_actions_registry():
    """Return the complete metadata and categories of all supported deterministic cleaning actions."""
    from actions import ACTION_METADATA
    return {
        "total_actions": len(ACTION_METADATA),
        "actions": ACTION_METADATA,
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
        is_red = getattr(action, "risk_level", "yellow") == "red"
        actions_list.append({
            "id": idx,
            "action": action.action,
            "column": action.column,
            "params": action.params,
            "reasoning": action.reasoning,
            "risk_level": getattr(action, "risk_level", "yellow"),
            "risk_label": getattr(action, "risk_label", "Needs Review"),
            "warning": getattr(action, "warning", None),
            "expected_impact": getattr(action, "expected_impact", None),
            "approved": not is_red,  # RED actions are unchecked by default
        })

    return {
        "session_id": req.session_id,
        "provider": provider.provider_name,
        "model": provider.model_name,
        "summary": plan.summary,
        "actions": actions_list,
    }


@app.post("/api/plan/add_action")
def api_plan_add_action(req: AddActionRequest):
    """Manually add a custom cleaning operation to the current session plan."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = SESSIONS[req.session_id]
    if not session.get("plan"):
        session["plan"] = CleaningPlan(summary="User-configured plan", actions=[])

    plan = session["plan"]

    try:
        from schemas import CleaningAction
        act_dict = {
            "action": req.action,
            "column": req.column,
            "params": req.params or {},
            "reasoning": req.reasoning or "User-configured operation",
        }
        action_obj = CleaningAction(**act_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action parameters: {e}")

    from planner import assign_action_risk
    assign_action_risk(action_obj, session.get("profile", {}), session.get("df_original"))
    plan.actions.append(action_obj)

    actions_list = []
    for idx, act in enumerate(plan.actions):
        is_red = getattr(act, "risk_level", "yellow") == "red"
        actions_list.append({
            "id": idx,
            "action": act.action,
            "column": act.column,
            "params": act.params,
            "reasoning": act.reasoning,
            "risk_level": getattr(act, "risk_level", "yellow"),
            "risk_label": getattr(act, "risk_label", "Needs Review"),
            "warning": getattr(act, "warning", None),
            "expected_impact": getattr(act, "expected_impact", None),
            "approved": not is_red,
        })

    return {
        "status": "success",
        "summary": plan.summary,
        "actions": actions_list,
    }


@app.post("/api/plan/remove_action")
def api_plan_remove_action(req: RemoveActionRequest):
    """Remove an operation from the plan by index."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = SESSIONS[req.session_id]
    plan = session.get("plan")
    if not plan or req.action_index < 0 or req.action_index >= len(plan.actions):
        raise HTTPException(status_code=400, detail="Invalid action index.")

    plan.actions.pop(req.action_index)

    actions_list = []
    for idx, act in enumerate(plan.actions):
        is_red = getattr(act, "risk_level", "yellow") == "red"
        actions_list.append({
            "id": idx,
            "action": act.action,
            "column": act.column,
            "params": act.params,
            "reasoning": act.reasoning,
            "risk_level": getattr(act, "risk_level", "yellow"),
            "risk_label": getattr(act, "risk_label", "Needs Review"),
            "warning": getattr(act, "warning", None),
            "expected_impact": getattr(act, "expected_impact", None),
            "approved": not is_red,
        })

    return {
        "status": "success",
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
        from schemas import CleaningPlan
        plan = CleaningPlan(summary="Direct dataset review", actions=[])
        session["plan"] = plan

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
            "rows_retained": diff_data["rows_after"],
            "rows_retained_pct": round((diff_data["rows_after"] / max(1, diff_data["rows_before"])) * 100, 1),
            "modified_cells_count": diff_data["modified_cells_count"],
            "total_missing_before": sum(original_missing.values()),
            "total_missing_after": sum(cleaned_missing.values()),
            "missing_resolved": max(0, sum(original_missing.values()) - sum(cleaned_missing.values())),
            "duplicates_removed": max(0, int(df_before.duplicated().sum() - df_cleaned.duplicated().sum())),
            "columns_removed": [c for c in diff_data["columns_before"] if c not in diff_data["columns_after"]],
            "columns_removed_count": max(0, len(diff_data["columns_before"]) - len(diff_data["columns_after"])),
            "actions_executed": len(executed_actions),
            "actions_proposed": len(plan.actions) if hasattr(plan, "actions") else 0,
            "actions_approved": len(req.approved_action_ids),
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
            "rows_delta": diff_data["rows_delta"],
            "rows_retained": diff_data["rows_after"],
            "rows_retained_pct": round((diff_data["rows_after"] / max(1, diff_data["rows_before"])) * 100, 1),
            "modified_cells_count": diff_data["modified_cells_count"],
            "total_missing_before": int(SESSIONS[session_id]["df_original"].isna().sum().sum()),
            "total_missing_after": int(df_current.isna().sum().sum()),
            "missing_resolved": max(0, int(SESSIONS[session_id]["df_original"].isna().sum().sum()) - int(df_current.isna().sum().sum())),
            "duplicates_removed": max(0, int(SESSIONS[session_id]["df_original"].duplicated().sum() - df_current.duplicated().sum())),
            "columns_removed": [c for c in diff_data["columns_before"] if c not in diff_data["columns_after"]],
            "columns_removed_count": max(0, len(diff_data["columns_before"]) - len(diff_data["columns_after"])),
        },
        "diff_table": diff_data["row_diffs"],
        "columns": diff_data["columns_after"],
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
            "rows_delta": diff_data["rows_delta"],
            "rows_retained": diff_data["rows_after"],
            "rows_retained_pct": round((diff_data["rows_after"] / max(1, diff_data["rows_before"])) * 100, 1),
            "modified_cells_count": diff_data["modified_cells_count"],
            "total_missing_before": int(SESSIONS[session_id]["df_original"].isna().sum().sum()),
            "total_missing_after": int(df_current.isna().sum().sum()),
            "missing_resolved": max(0, int(SESSIONS[session_id]["df_original"].isna().sum().sum()) - int(df_current.isna().sum().sum())),
            "duplicates_removed": max(0, int(SESSIONS[session_id]["df_original"].duplicated().sum() - df_current.duplicated().sum())),
            "columns_removed": [c for c in diff_data["columns_before"] if c not in diff_data["columns_after"]],
            "columns_removed_count": max(0, len(diff_data["columns_before"]) - len(diff_data["columns_after"])),
        },
        "diff_table": diff_data["row_diffs"],
        "columns": diff_data["columns_after"],
    }


@app.post("/api/session/rollback")
@app.post("/api/session/rollback_all")
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
            "rows_delta": 0,
            "rows_retained": diff_data["rows_after"],
            "rows_retained_pct": 100.0,
            "modified_cells_count": 0,
            "total_missing_before": int(SESSIONS[session_id]["df_original"].isna().sum().sum()),
            "total_missing_after": int(df_current.isna().sum().sum()),
            "missing_resolved": 0,
            "duplicates_removed": 0,
            "columns_removed": [],
            "columns_removed_count": 0,
        },
        "diff_table": diff_data["row_diffs"],
        "columns": diff_data["columns_after"],
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
# DATAREFINE STUDIO ENGINE API ENDPOINTS
# ============================================================

@app.post("/api/cluster/detect")
def api_cluster_detect(req: ClusterDetectRequest):
    """Detect similar string clusters in a column using DataRefine clustering algorithms."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[req.session_id]
    df = session["df_current"]
    if req.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.column}' not in dataset.")

    try:
        clusters = detect_clusters(
            series=df[req.column],
            method=req.method,
            params=req.params,
        )
        return {
            "status": "success",
            "column": req.column,
            "method": req.method,
            "cluster_count": len(clusters),
            "clusters": clusters,
        }
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Clustering failed: {err}")


@app.post("/api/cluster/apply")
def api_cluster_apply(req: ClusterApplyRequest):
    """Apply approved cluster merges to harmonize text variants to canonical forms."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[req.session_id]
    sess: CleaningSession = session["cleaning_session"]

    action = CleaningAction(
        action="merge_clusters",
        column=req.column,
        params={"clusters": req.clusters, "mapping": req.mapping},
        reasoning=f"Harmonize text variants in '{req.column}' using DataRefine clustering",
    )
    res = sess.apply_action(action)
    df_current = sess.get_current_df()
    session["df_current"] = df_current

    diff_data = compute_side_by_side_diff(session["df_original"], df_current)
    return {
        "status": "success",
        "message": f"Successfully merged clusters in '{req.column}'",
        "current_step": sess.current_step,
        "checkpoints": sess.get_history(),
        "diff_summary": {
            "rows_before": diff_data["rows_before"],
            "rows_after": diff_data["rows_after"],
            "rows_delta": diff_data["rows_delta"],
            "modified_cells_count": diff_data["modified_cells_count"],
        },
        "diff_table": diff_data["row_diffs"],
        "columns": diff_data["columns_after"],
    }


@app.post("/api/grel/preview")
def api_grel_preview(req: GRELPreviewRequest):
    """Preview a safe, restricted GREL-inspired expression on sample rows."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[req.session_id]
    df = session["df_current"]
    if req.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.column}' not in dataset.")

    preview_res = preview_grel(
        df=df,
        column=req.column,
        expression=req.expression,
        sample_limit=req.sample_limit,
    )
    return preview_res


@app.post("/api/grel/apply")
def api_grel_apply(req: GRELApplyRequest):
    """Apply a safe GREL transformation to a column and record a checkpoint."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[req.session_id]
    sess: CleaningSession = session["cleaning_session"]

    action = CleaningAction(
        action="grel_transform",
        column=req.column,
        params={"expression": req.expression, "on_error": req.on_error},
        reasoning=f"GREL Transformation on '{req.column}': {req.expression}",
    )
    res = sess.apply_action(action)
    df_current = sess.get_current_df()
    session["df_current"] = df_current

    diff_data = compute_side_by_side_diff(session["df_original"], df_current)
    return {
        "status": "success",
        "message": f"Successfully applied GREL transformation to '{req.column}'",
        "current_step": sess.current_step,
        "checkpoints": sess.get_history(),
        "diff_summary": {
            "rows_before": diff_data["rows_before"],
            "rows_after": diff_data["rows_after"],
            "rows_delta": diff_data["rows_delta"],
            "modified_cells_count": diff_data["modified_cells_count"],
        },
        "diff_table": diff_data["row_diffs"],
        "columns": diff_data["columns_after"],
    }


@app.post("/api/facets/compute")
def api_facets_compute(req: FacetComputeRequest):
    """Compute text, numeric, or date facets with value counts and distribution."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[req.session_id]
    df = session["df_current"]
    if req.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.column}' not in dataset.")

    filtered_df = apply_filters(df, filters=req.active_filters)
    series = filtered_df[req.column]

    facet = compute_facet_auto(
        series=series,
        facet_type=req.facet_type,
        top_n=req.top_n,
        bins_count=req.bins,
        search_query=req.search,
    )
    return {
        "status": "success",
        "column": req.column,
        "facet": facet,
    }


@app.post("/api/dataset/explore")
def api_dataset_explore(req: ExploreRequest):
    """Explore dataset with multi-column facet filtering, global search, sorting, and pagination."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[req.session_id]
    df = session["df_current"]

    exp_res = explore_dataset(
        df=df,
        filters=req.filters,
        global_search=req.global_search,
        sort_by=req.sort_by,
        sort_ascending=req.sort_ascending,
        offset=req.offset,
        limit=req.limit,
    )
    return exp_res


@app.post("/api/reconcile/match")
def api_reconcile_match(req: ReconcileRequest):
    """Match messy column values against reference entity catalogs and compute similarity scores."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[req.session_id]
    df = session["df_current"]
    if req.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.column}' not in dataset.")

    try:
        res = reconcile_column_candidates(
            series=df[req.column],
            catalog_name=req.catalog,
            custom_entities=req.custom_entities,
            match_threshold=req.threshold,
        )
        return {
            "status": "success",
            "column": req.column,
            "catalog": req.catalog,
            "result": res,
        }
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Reconciliation matching failed: {err}")


@app.post("/api/reconcile/apply")
def api_reconcile_apply(req: ReconcileApplyRequest):
    """Apply approved entity mappings to standardize column values."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[req.session_id]
    sess: CleaningSession = session["cleaning_session"]

    action = CleaningAction(
        action="reconcile_column",
        column=req.column,
        params={"mapping": req.mapping, "catalog": req.catalog},
        reasoning=f"Entity reconciliation against catalog '{req.catalog}' for '{req.column}'",
    )
    res = sess.apply_action(action)
    df_current = sess.get_current_df()
    session["df_current"] = df_current

    diff_data = compute_side_by_side_diff(session["df_original"], df_current)
    return {
        "status": "success",
        "message": f"Successfully reconciled '{req.column}' with {len(req.mapping)} canonical entities",
        "current_step": sess.current_step,
        "checkpoints": sess.get_history(),
        "diff_summary": {
            "rows_before": diff_data["rows_before"],
            "rows_after": diff_data["rows_after"],
            "rows_delta": diff_data["rows_delta"],
            "modified_cells_count": diff_data["modified_cells_count"],
        },
        "diff_table": diff_data["row_diffs"],
        "columns": diff_data["columns_after"],
    }


@app.get("/api/workflow/history")
def api_workflow_history(session_id: str = Query(...)):
    """Retrieve executed operations serialized in DataRefine Workflow JSON recipe format."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[session_id]
    sess: CleaningSession = session["cleaning_session"]
    recipe = WorkflowRecipeManager.export_recipe_from_history(sess.get_history())
    return {
        "status": "success",
        "session_id": session_id,
        "recipe": recipe,
    }


@app.post("/api/workflow/export_recipe")
def api_workflow_export_recipe(req: Dict[str, str]):
    """Export the current session's operation history as a downloadable JSON recipe."""
    session_id = req.get("session_id")
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[session_id]
    sess: CleaningSession = session["cleaning_session"]
    recipe = WorkflowRecipeManager.export_recipe_from_history(sess.get_history())
    return JSONResponse(
        content=recipe,
        headers={"Content-Disposition": f"attachment; filename=recipe_{session['run_id']}.json"},
    )


@app.post("/api/workflow/apply_recipe")
def api_workflow_apply_recipe(req: RecipeApplyRequest):
    """Import and sequentially replay a DataRefine Workflow JSON recipe onto the active dataset."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[req.session_id]
    sess: CleaningSession = session["cleaning_session"]

    valid, errors = WorkflowRecipeManager.validate_recipe(req.recipe)
    if not valid:
        raise HTTPException(status_code=400, detail=f"Recipe validation failed: {'; '.join(errors)}")

    operations = req.recipe.get("operations", [])
    applied_count = 0
    errors_encountered = []

    for idx, op in enumerate(operations):
        try:
            action_obj = CleaningAction(
                action=op["action"],
                column=op.get("column"),
                params=op.get("params", {}),
                reasoning=op.get("reasoning", f"Replayed recipe step {idx + 1}"),
            )
            sess.apply_action(action_obj)
            applied_count += 1
        except Exception as e:
            errors_encountered.append(f"Step {idx + 1} ({op.get('action')}): {str(e)}")
            if req.stop_on_error:
                break

    df_current = sess.get_current_df()
    session["df_current"] = df_current
    diff_data = compute_side_by_side_diff(session["df_original"], df_current)

    return {
        "status": "success" if not errors_encountered else "partial_success",
        "applied_count": applied_count,
        "total_operations": len(operations),
        "errors": errors_encountered,
        "current_step": sess.current_step,
        "checkpoints": sess.get_history(),
        "diff_summary": {
            "rows_before": diff_data["rows_before"],
            "rows_after": diff_data["rows_after"],
            "rows_delta": diff_data["rows_delta"],
            "modified_cells_count": diff_data["modified_cells_count"],
        },
        "diff_table": diff_data["row_diffs"],
        "columns": diff_data["columns_after"],
    }



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
    body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background-color: #0b0f19; color: #f3f4f6; margin: 0; }
    code, pre { font-family: 'JetBrains Mono', monospace; }
    .glass { background: rgba(17, 24, 39, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .glass-card { background: rgba(31, 41, 55, 0.5); border: 1px solid rgba(255, 255, 255, 0.06); }
    .cell-modified { background-color: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.5); color: #6ee7b7; font-weight: 600; }
    .cell-dropped { background-color: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.5); color: #fca5a5; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
    /* Fallback styles */
    .hidden { display: none !important; }
  </style>
</head>
<body class="min-h-screen flex flex-col bg-[#0b0f19] text-gray-100">

  <!-- TOAST NOTIFICATION -->
  <div id="toast" class="fixed top-5 right-5 z-50 px-4 py-3 rounded-xl shadow-2xl transition-all duration-300 transform translate-y-[-100px] opacity-0 text-sm font-medium flex items-center space-x-2"></div>

  <!-- NAVBAR -->
  <header class="glass sticky top-0 z-40 px-6 py-4 flex items-center justify-between border-b border-gray-800">
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
    <div class="flex border-b border-gray-800 space-x-2 overflow-x-auto pb-1">
      <button onclick="switchTab('profile')" id="tab-btn-profile" class="px-5 py-2.5 font-medium text-sm rounded-t-lg transition border-b-2 border-indigo-500 text-indigo-400 bg-gray-900/40 whitespace-nowrap">
        1. Select & Profile
      </button>
      <button onclick="switchTab('plan')" id="tab-btn-plan" class="px-5 py-2.5 font-medium text-sm rounded-t-lg transition border-b-2 border-transparent text-gray-400 hover:text-gray-200 whitespace-nowrap">
        2. AI Plan & One-Click Approval
      </button>
      <button onclick="switchTab('diff')" id="tab-btn-diff" class="px-5 py-2.5 font-medium text-sm rounded-t-lg transition border-b-2 border-transparent text-gray-400 hover:text-gray-200 whitespace-nowrap">
        3. Visual Diff & Rollback
      </button>
      <button onclick="switchTab('eval')" id="tab-btn-eval" class="px-5 py-2.5 font-medium text-sm rounded-t-lg transition border-b-2 border-transparent text-gray-400 hover:text-gray-200 whitespace-nowrap">
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
              <select id="dataset-select" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 text-gray-200">
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
          <span class="text-xs text-gray-400">Click any column row for detailed stats</span>
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
            <tbody id="columns-table-body" class="divide-y divide-gray-800/60 cursor-pointer"></tbody>
          </table>
        </div>
      </div>

      <!-- Raw Data Preview (First 10 Rows) -->
      <div class="glass rounded-2xl p-5 space-y-4" id="preview-section" style="display:none;">
        <div class="flex items-center justify-between">
          <h3 class="text-base font-semibold text-white">Original Data Preview (First 10 Rows)</h3>
          <span class="text-xs text-indigo-400 font-mono" id="preview-row-count"></span>
        </div>
        <div class="overflow-x-auto max-h-[350px] overflow-y-auto">
          <table class="w-full text-left text-xs text-gray-300 border-collapse">
            <thead class="sticky top-0 bg-gray-950 text-gray-400 uppercase text-[11px] border-b border-gray-800">
              <tr id="preview-table-header"></tr>
            </thead>
            <tbody id="preview-table-body" class="divide-y divide-gray-800/60 font-mono"></tbody>
          </table>
        </div>
        <!-- Proceed Button -->
        <div class="flex justify-end pt-2">
          <button onclick="switchTab('plan')" class="px-6 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-semibold rounded-xl text-sm transition shadow-lg shadow-purple-600/20">
            ➡️ Proceed to Step 2: AI Planning & Approvals
          </button>
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
            <select id="llm-provider-select" onchange="updateModelDefaults()" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:outline-none focus:border-indigo-500 text-gray-200">
              <option value="gemini">Google Gemini (Default)</option>
              <option value="openai">OpenAI (GPT-4o / Groq / vLLM)</option>
              <option value="anthropic">Anthropic Claude</option>
              <option value="ollama">Ollama (Offline / Air-Gapped)</option>
              <option value="mock">Deterministic Mock (Test Mode)</option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-gray-400 mb-1">Model Name</label>
            <input type="text" id="llm-model-input" value="gemini-2.5-flash" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:outline-none text-gray-200">
          </div>
          <div>
            <label class="block text-xs text-gray-400 mb-1">Custom Base URL (vLLM / Ollama)</label>
            <input type="text" id="llm-base-url" placeholder="http://localhost:11434" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:outline-none text-gray-200">
          </div>
          <div>
            <label class="block text-xs text-gray-400 mb-1">API Key Override (Optional)</label>
            <input type="password" id="llm-api-key" placeholder="Uses .env if blank" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:outline-none text-gray-200">
          </div>
        </div>
        <button onclick="generatePlan()" id="btn-generate-plan" class="w-full py-3 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-medium rounded-xl text-sm transition shadow-lg shadow-purple-600/20">
          🤖 Generate Cleaning Plan (Privacy-Preserved)
        </button>
      </div>

      <!-- Plan Recommendations & Approval Gate -->
      <div id="plan-section" class="glass p-5 rounded-2xl space-y-5 hidden">
        <div class="flex items-center justify-between border-b border-gray-800 pb-3 flex-wrap gap-2">
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
        <div class="glass p-4 rounded-xl flex items-center justify-around gap-2">
          <button onclick="downloadCleanedCsv()" class="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold rounded-lg text-white">📥 Cleaned CSV</button>
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
        <div class="flex items-center justify-between flex-wrap gap-2">
          <h3 class="text-base font-semibold text-white">Side-by-Side Visual Diff (First 100 Rows)</h3>
          <span class="text-xs text-emerald-400 bg-emerald-950 px-2.5 py-1 rounded-full border border-emerald-800">
            Green cells indicate modified/repaired values
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
            <input type="text" id="eval-gt-path" value="data/benchmark/ground_truth.csv" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-4 py-2 text-sm focus:outline-none text-gray-200">
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

  <!-- DISTRIBUTION MODAL -->
  <div id="dist-modal" class="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 hidden">
    <div class="glass max-w-lg w-full p-6 rounded-2xl space-y-4 border border-gray-700 shadow-2xl">
      <div class="flex items-center justify-between border-b border-gray-800 pb-3">
        <h3 class="text-base font-bold text-white" id="dist-modal-title">Column Distribution</h3>
        <button onclick="closeDistModal()" class="text-gray-400 hover:text-white text-lg">&times;</button>
      </div>
      <div id="dist-modal-body" class="space-y-3 text-sm"></div>
      <div class="flex justify-end pt-2">
        <button onclick="closeDistModal()" class="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-xs rounded-xl text-gray-200">Close</button>
      </div>
    </div>
  </div>

  <!-- JAVASCRIPT APP LOGIC -->
  <script>
    let currentSessionId = null;
    let currentActions = [];
    let currentColumns = [];

    function showToast(message, isError = false) {
      const t = document.getElementById('toast');
      t.textContent = message;
      t.className = isError
        ? 'fixed top-5 right-5 z-50 px-4 py-3 rounded-xl shadow-2xl transition-all duration-300 transform bg-rose-950 text-rose-200 border border-rose-800 text-sm font-medium flex items-center space-x-2'
        : 'fixed top-5 right-5 z-50 px-4 py-3 rounded-xl shadow-2xl transition-all duration-300 transform bg-emerald-950 text-emerald-200 border border-emerald-800 text-sm font-medium flex items-center space-x-2';
      t.style.transform = 'translateY(0)';
      t.style.opacity = '1';
      setTimeout(() => {
        t.style.transform = 'translateY(-100px)';
        t.style.opacity = '0';
      }, 3500);
    }

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
        const el = document.getElementById(`tab-${t}`);
        if (el) el.classList.add('hidden');
        const btn = document.getElementById(`tab-btn-${t}`);
        if (btn) {
          btn.classList.remove('border-indigo-500', 'text-indigo-400', 'bg-gray-900/40');
          btn.classList.add('border-transparent', 'text-gray-400');
        }
      });
      const activeEl = document.getElementById(`tab-${tab}`);
      if (activeEl) activeEl.classList.remove('hidden');
      const activeBtn = document.getElementById(`tab-btn-${tab}`);
      if (activeBtn) {
        activeBtn.classList.remove('border-transparent', 'text-gray-400');
        activeBtn.classList.add('border-indigo-500', 'text-indigo-400', 'bg-gray-900/40');
      }
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
        if (!uploadRes.ok) {
          showToast(uploadData.detail || 'Upload failed', true);
          return;
        }
        filepath = uploadData.filepath;
      }

      if (!filepath) {
        showToast('Please select or upload a CSV file.', true);
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
        currentColumns = data.columns;
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
        data.columns.forEach((col, idx) => {
          const tr = document.createElement('tr');
          tr.className = 'hover:bg-gray-900/60 transition';
          tr.onclick = () => openDistModal(col);
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

        // Render Data Preview Table
        if (data.preview_rows && data.preview_rows.length > 0) {
          const pCols = Object.keys(data.preview_rows[0]);
          const pHeader = document.getElementById('preview-table-header');
          pHeader.innerHTML = '<th class="px-3 py-2">#</th>' + pCols.map(c => `<th class="px-3 py-2 font-mono">${c}</th>`).join('');

          const pBody = document.getElementById('preview-table-body');
          pBody.innerHTML = '';
          data.preview_rows.forEach((r, idx) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td class="px-3 py-1.5 text-gray-500">${idx}</td>` + pCols.map(c => `<td class="px-3 py-1.5 text-gray-300">${r[c] !== null ? r[c] : '<span class="text-rose-400">NaN</span>'}</td>`).join('');
            pBody.appendChild(tr);
          });
          document.getElementById('preview-row-count').textContent = `Showing 10 of ${data.summary.rows} rows`;
          document.getElementById('preview-section').style.display = 'block';
        }

        showToast(`Dataset profiled: ${data.summary.rows} rows, ${data.summary.columns} cols`);

      } catch (err) {
        showToast(err.message, true);
      } finally {
        btn.textContent = '📊 Analyze & Profile Dataset';
        btn.disabled = false;
      }
    }

    function openDistModal(col) {
      document.getElementById('dist-modal-title').textContent = `Distribution: ${col.name} (${col.dtype})`;
      const body = document.getElementById('dist-modal-body');
      let html = `
        <div class="grid grid-cols-2 gap-2 text-xs">
          <div class="glass-card p-2 rounded-lg"><span class="text-gray-400">Semantic Type:</span> <span class="text-indigo-300 font-semibold">${col.semantic_type}</span></div>
          <div class="glass-card p-2 rounded-lg"><span class="text-gray-400">Missing Values:</span> <span class="text-rose-300 font-semibold">${col.missing_count} (${col.missing_pct}%)</span></div>
          <div class="glass-card p-2 rounded-lg"><span class="text-gray-400">Unique Values:</span> <span class="text-gray-200 font-semibold">${col.unique_count}</span></div>
          <div class="glass-card p-2 rounded-lg"><span class="text-gray-400">Empty / Constant:</span> <span class="text-gray-200">${col.is_empty ? 'Empty' : (col.is_constant ? 'Constant' : 'Normal')}</span></div>
        </div>
      `;

      if (col.numeric_stats) {
        html += `
          <div class="mt-3">
            <h4 class="text-xs font-semibold text-gray-400 uppercase mb-1">Numeric Summary</h4>
            <div class="grid grid-cols-4 gap-2 text-center text-xs">
              <div class="glass-card p-2 rounded-lg"><span class="text-gray-400 text-[10px]">Min</span><p class="font-bold text-gray-200">${col.numeric_stats.min}</p></div>
              <div class="glass-card p-2 rounded-lg"><span class="text-gray-400 text-[10px]">Max</span><p class="font-bold text-gray-200">${col.numeric_stats.max}</p></div>
              <div class="glass-card p-2 rounded-lg"><span class="text-gray-400 text-[10px]">Mean</span><p class="font-bold text-gray-200">${col.numeric_stats.mean}</p></div>
              <div class="glass-card p-2 rounded-lg"><span class="text-gray-400 text-[10px]">Median</span><p class="font-bold text-gray-200">${col.numeric_stats.median}</p></div>
            </div>
          </div>
        `;
      }

      if (col.top_values && col.top_values.length > 0) {
        html += `
          <div class="mt-3">
            <h4 class="text-xs font-semibold text-gray-400 uppercase mb-1">Top Frequent Values</h4>
            <div class="space-y-1.5">
              ${col.top_values.map(v => `
                <div class="flex items-center justify-between text-xs glass-card px-3 py-1.5 rounded-lg">
                  <span class="font-mono text-gray-300 truncate max-w-[200px]">${v.value}</span>
                  <span class="text-indigo-400 font-medium">${v.count} (${v.percentage}%)</span>
                </div>
              `).join('')}
            </div>
          </div>
        `;
      }

      body.innerHTML = html;
      document.getElementById('dist-modal').classList.remove('hidden');
    }

    function closeDistModal() {
      document.getElementById('dist-modal').classList.add('hidden');
    }

    async function generatePlan() {
      if (!currentSessionId) {
        showToast('Please profile a dataset first.', true);
        switchTab('profile');
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
                <div class="flex items-center space-x-2 flex-wrap gap-1">
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
        showToast(`Cleaning plan generated with ${currentActions.length} action(s)`);
      } catch (err) {
        showToast(err.message, true);
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
        showToast('Dataset cleaned successfully!');
      } catch (err) {
        showToast(err.message, true);
      } finally {
        btn.textContent = '🚀 Execute Approved Cleaning Plan';
        btn.disabled = false;
      }
    }

    function renderDiffView(data) {
      if (!data || !data.diff_summary) return;

      document.getElementById('diff-rows-delta').textContent = `${data.diff_summary.rows_before} ➔ ${data.diff_summary.rows_after}`;
      document.getElementById('diff-modified-cells').textContent = data.diff_summary.modified_cells_count.toLocaleString();
      document.getElementById('diff-missing-delta').textContent = `${data.diff_summary.total_missing_before} ➔ ${data.diff_summary.total_missing_after}`;

      // Checkpoint timeline
      const timeline = document.getElementById('checkpoints-timeline');
      timeline.innerHTML = '';
      (data.checkpoints || []).forEach(cp => {
        const pill = document.createElement('span');
        pill.className = 'px-2 py-0.5 rounded text-[11px] bg-gray-900 border border-gray-800 text-gray-300 whitespace-nowrap';
        pill.textContent = cp.step === 0 ? 'Step 0 (Original)' : `Step ${cp.step}: ${cp.action_name}`;
        timeline.appendChild(pill);
      });

      // Diff table header
      const thead = document.getElementById('diff-table-header');
      thead.innerHTML = '<th class="px-3 py-2">#</th>';
      const cols = data.columns || [];
      cols.forEach(col => {
        const th = document.createElement('th');
        th.className = 'px-3 py-2 font-mono whitespace-nowrap';
        th.textContent = col;
        thead.appendChild(th);
      });

      // Diff table body
      const tbody = document.getElementById('diff-table-body');
      tbody.innerHTML = '';
      const diffRows = data.diff_table || [];
      diffRows.forEach(r => {
        const tr = document.createElement('tr');
        tr.className = r.status === 'modified' ? 'bg-emerald-950/20' : (r.status === 'dropped' ? 'bg-rose-950/20' : '');

        let rowHtml = `<td class="px-3 py-1.5 text-gray-500">${r.row_index}</td>`;
        cols.forEach(col => {
          const cell = (r.cells && r.cells[col]) ? r.cells[col] : { original: '', cleaned: '', changed: false };
          if (cell.changed) {
            rowHtml += `<td class="px-3 py-1.5 cell-modified whitespace-nowrap" title="Original: ${cell.original !== null ? cell.original : 'NaN'}">${cell.cleaned !== null ? cell.cleaned : 'NaN'}</td>`;
          } else {
            rowHtml += `<td class="px-3 py-1.5 text-gray-300 whitespace-nowrap">${cell.cleaned !== null ? cell.cleaned : ''}</td>`;
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
        if (data.status === 'success') {
          showToast(data.message);
          renderDiffView(data);
        } else {
          showToast(data.message, true);
        }
      } catch (err) {
        showToast(err.message, true);
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
        if (data.status === 'success') {
          showToast(data.message);
          renderDiffView(data);
        } else {
          showToast(data.message, true);
        }
      } catch (err) {
        showToast(err.message, true);
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
        showToast(data.message);
        renderDiffView(data);
      } catch (err) {
        showToast(err.message, true);
      }
    }

    async function runBenchmarkEvaluation() {
      if (!currentSessionId) {
        showToast('Please profile and clean a dataset first.', true);
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
        showToast('Quality benchmark evaluation complete!');
      } catch (err) {
        showToast(err.message, true);
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
def home_page():
    """Serve the public-facing DataRefine homepage with no-cache headers."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = DASHBOARD_HTML
    return HTMLResponse(
        content=content,
        status_code=200,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/app", response_class=HTMLResponse)
@app.get("/app/review", response_class=HTMLResponse)
@app.get("/app/results", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def workspace_page():
    """Serve the DataRefine application workspace with no-cache headers."""
    app_path = os.path.join(STATIC_DIR, "app.html")
    if os.path.exists(app_path):
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = DASHBOARD_HTML
    return HTMLResponse(
        content=content,
        status_code=200,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.exception_handler(404)
async def custom_404_handler(request, exc):
    """Clean, branded 404 page for unknown routes."""
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <title>404 — Page Not Found | DataRefine</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-[#090a0f] text-zinc-300 min-h-screen flex items-center justify-center p-6 font-sans">
  <div class="max-w-md w-full bg-[#11141e] border border-[#1f2536] rounded-xl p-8 text-center space-y-4">
    <span class="text-xs font-mono text-blue-500 uppercase font-semibold">404 Error</span>
    <h1 class="text-xl font-bold text-white">Page Not Found</h1>
    <p class="text-xs text-zinc-400">The route you requested does not exist in DataRefine.</p>
    <div class="pt-2 flex justify-center space-x-3">
      <a href="/" class="px-4 py-2 rounded-lg bg-[#171c28] border border-[#1f2536] hover:bg-zinc-800 text-xs font-medium text-zinc-200 transition">Homepage</a>
      <a href="/app" class="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-medium text-white transition">Workspace</a>
    </div>
  </div>
</body>
</html>""",
        status_code=404,
    )


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AI Data Cleaner Web Dashboard & API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  DataRefine — Data Quality & Cleaning Studio")
    print(f"  Running at: http://{args.host}:{args.port}")
    print(f"=======================================================\n")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
