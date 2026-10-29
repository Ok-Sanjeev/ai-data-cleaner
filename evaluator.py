"""
Evaluation Module for AI Data Cleaner.

Calculates cell-level and row-level cleaning quality metrics (Precision, Recall, F1-Score,
and Accuracy Gain) by comparing original dirty data, cleaned data, and ground truth reference data.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================
# VALUE COMPARISON & NORMALIZATION
# ============================================================

def is_cell_null(value: Any) -> bool:
    """Check if a value represents null, missing, NaN, or blank."""
    if value is None:
        return True
    if pd.isna(value):
        return True
    s = str(value).strip().lower()
    return s in {"", "nan", "<na>", "nat", "none", "null"}


def values_match(val1: Any, val2: Any, tolerance: float = 1e-4) -> bool:
    """Determine whether two cell values match semantically."""
    null1 = is_cell_null(val1)
    null2 = is_cell_null(val2)
    if null1 and null2:
        return True
    if null1 != null2:
        return False

    # Boolean equivalence
    str1 = str(val1).strip().lower()
    str2 = str(val2).strip().lower()
    bool_true = {"true", "1", "1.0", "yes", "y", "t"}
    bool_false = {"false", "0", "0.0", "no", "n", "f"}
    if (str1 in (bool_true | bool_false)) and (str2 in (bool_true | bool_false)):
        return (str1 in bool_true and str2 in bool_true) or (str1 in bool_false and str2 in bool_false)

    # Numeric equivalence (handles integer vs float, e.g. 25 vs 25.0)
    try:
        clean_num1 = str(val1).strip().replace(",", "").replace("$", "").replace("£", "").replace("€", "").replace("₹", "")
        clean_num2 = str(val2).strip().replace(",", "").replace("$", "").replace("£", "").replace("€", "").replace("₹", "")
        f1 = float(clean_num1)
        f2 = float(clean_num2)
        return abs(f1 - f2) <= tolerance
    except (ValueError, TypeError):
        pass

    # Datetime equivalence
    try:
        d1 = pd.to_datetime(val1, format="mixed")
        d2 = pd.to_datetime(val2, format="mixed")
        return d1 == d2
    except Exception:
        pass

    # Text comparison: exact string equality (casing and whitespace count)
    return str(val1) == str(val2)


# ============================================================
# EVALUATION METRICS ENGINE
# ============================================================

def evaluate_cleaning(
    df_dirty: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    df_ground_truth: pd.DataFrame,
    id_column: Optional[str] = "customer_id",
) -> Dict[str, Any]:
    """Score the cleaning quality against a ground truth reference dataset.

    Returns Precision, Recall, F1-Score, accuracy deltas, per-column breakdowns,
    and a list of all repairs and missed errors.
    """
    # 1. Align rows by id_column if available and present in all datasets
    use_id_alignment = (
        id_column is not None
        and id_column in df_dirty.columns
        and id_column in df_cleaned.columns
        and id_column in df_ground_truth.columns
    )

    # 2. Determine common data columns to evaluate (excluding the alignment ID)
    eval_columns = [
        col for col in df_ground_truth.columns
        if col in df_dirty.columns and col in df_cleaned.columns
        and (not use_id_alignment or col != id_column)
    ]

    if use_id_alignment:
        # Deduplicate dirty and cleaned by ID prioritizing the first occurrence for comparison
        clean_indexed = df_cleaned.drop_duplicates(subset=[id_column]).set_index(id_column)
        dirty_indexed = df_dirty.drop_duplicates(subset=[id_column]).set_index(id_column)
        truth_indexed = df_ground_truth.drop_duplicates(subset=[id_column]).set_index(id_column)

        common_ids = [k for k in truth_indexed.index if k in clean_indexed.index and k in dirty_indexed.index]
        clean_aligned = clean_indexed.loc[common_ids]
        dirty_aligned = dirty_indexed.loc[common_ids]
        truth_aligned = truth_indexed.loc[common_ids]
    else:
        # Align by position / index for the minimum shared rows
        min_rows = min(len(df_dirty), len(df_cleaned), len(df_ground_truth))
        clean_aligned = df_cleaned.iloc[:min_rows].reset_index(drop=True)
        dirty_aligned = df_dirty.iloc[:min_rows].reset_index(drop=True)
        truth_aligned = df_ground_truth.iloc[:min_rows].reset_index(drop=True)
        common_ids = list(range(min_rows))

    total_cells = len(common_ids) * len(eval_columns)

    # Global counters
    tp_total = 0  # Repaired cells that match ground truth
    fp_total = 0  # Repaired cells that DO NOT match ground truth (over-cleaning or wrong fix)
    fn_total = 0  # Dirty cells that were not fixed or remain wrong
    tn_total = 0  # Clean cells correctly left unchanged

    cells_correct_before = 0
    cells_correct_after = 0

    per_column: Dict[str, Dict[str, Any]] = {}
    repairs_detail: List[Dict[str, Any]] = []
    missed_errors_detail: List[Dict[str, Any]] = []

    for col in eval_columns:
        col_tp = 0
        col_fp = 0
        col_fn = 0
        col_tn = 0
        col_before_correct = 0
        col_after_correct = 0

        for row_id in common_ids:
            dirty_val = dirty_aligned.at[row_id, col]
            cleaned_val = clean_aligned.at[row_id, col]
            truth_val = truth_aligned.at[row_id, col]

            was_error_before = not values_match(dirty_val, truth_val)
            was_changed = not values_match(cleaned_val, dirty_val)
            matches_truth = values_match(cleaned_val, truth_val)

            if not was_error_before:
                col_before_correct += 1

            if matches_truth:
                col_after_correct += 1

            # Classification
            if was_changed and matches_truth:
                col_tp += 1
                repairs_detail.append({
                    "row_id": str(row_id),
                    "column": col,
                    "before": str(dirty_val),
                    "after": str(cleaned_val),
                    "ground_truth": str(truth_val),
                    "status": "CORRECT_FIX",
                })
            elif was_changed and not matches_truth:
                col_fp += 1
                repairs_detail.append({
                    "row_id": str(row_id),
                    "column": col,
                    "before": str(dirty_val),
                    "after": str(cleaned_val),
                    "ground_truth": str(truth_val),
                    "status": "INCORRECT_FIX",
                })
                # If it was an error before and repaired into something still wrong, it's also a missed fix
                if was_error_before:
                    col_fn += 1

            elif not was_changed:
                if was_error_before:
                    col_fn += 1
                    missed_errors_detail.append({
                        "row_id": str(row_id),
                        "column": col,
                        "before": str(dirty_val),
                        "ground_truth": str(truth_val),
                        "status": "MISSED_ERROR",
                    })
                else:
                    col_tn += 1

        # Per-column precision, recall, F1
        col_prec = col_tp / (col_tp + col_fp) if (col_tp + col_fp) > 0 else (1.0 if col_fn == 0 else 0.0)
        col_rec = col_tp / (col_tp + col_fn) if (col_tp + col_fn) > 0 else 1.0
        col_f1 = (2 * col_prec * col_rec / (col_prec + col_rec)) if (col_prec + col_rec) > 0 else 0.0

        per_column[col] = {
            "precision": round(col_prec, 4),
            "recall": round(col_rec, 4),
            "f1": round(col_f1, 4),
            "tp": col_tp,
            "fp": col_fp,
            "fn": col_fn,
            "tn": col_tn,
            "total_errors_before": col_tp + col_fn,
            "errors_remaining": col_fp + col_fn,
            "accuracy_before": round(col_before_correct / max(len(common_ids), 1), 4),
            "accuracy_after": round(col_after_correct / max(len(common_ids), 1), 4),
        }

        tp_total += col_tp
        fp_total += col_fp
        fn_total += col_fn
        tn_total += col_tn
        cells_correct_before += col_before_correct
        cells_correct_after += col_after_correct

    # Global precision, recall, F1
    overall_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else (1.0 if fn_total == 0 else 0.0)
    overall_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 1.0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0.0

    # Duplicates handling
    dups_dirty = int(df_dirty.duplicated().sum())
    dups_cleaned = int(df_cleaned.duplicated().sum())
    duplicates_removed = dups_dirty - dups_cleaned

    # Row-level exact matches
    rows_clean_before = 0
    rows_clean_after = 0
    for row_id in common_ids:
        all_match_before = all(values_match(dirty_aligned.at[row_id, c], truth_aligned.at[row_id, c]) for c in eval_columns)
        all_match_after = all(values_match(clean_aligned.at[row_id, c], truth_aligned.at[row_id, c]) for c in eval_columns)
        if all_match_before:
            rows_clean_before += 1
        if all_match_after:
            rows_clean_after += 1

    total_rows = len(common_ids)

    return {
        "overall": {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(overall_f1, 4),
            "true_positives": tp_total,
            "false_positives": fp_total,
            "false_negatives": fn_total,
            "true_negatives": tn_total,
            "total_cells_evaluated": total_cells,
            "total_errors_in_dirty": tp_total + fn_total,
            "total_repairs_applied": tp_total + fp_total,
            "correct_repairs": tp_total,
            "incorrect_repairs": fp_total,
            "missed_errors": fn_total,
            "cell_accuracy_before": round(cells_correct_before / max(total_cells, 1), 4),
            "cell_accuracy_after": round(cells_correct_after / max(total_cells, 1), 4),
            "cell_accuracy_gain": round((cells_correct_after - cells_correct_before) / max(total_cells, 1), 4),
            "row_accuracy_before": round(rows_clean_before / max(total_rows, 1), 4),
            "row_accuracy_after": round(rows_clean_after / max(total_rows, 1), 4),
            "row_accuracy_gain": round((rows_clean_after - rows_clean_before) / max(total_rows, 1), 4),
            "duplicate_rows_in_dirty": dups_dirty,
            "duplicate_rows_in_cleaned": dups_cleaned,
            "duplicates_removed": duplicates_removed,
        },
        "per_column": per_column,
        "repairs": repairs_detail,
        "missed_errors": missed_errors_detail,
    }


# ============================================================
# SCORECARD FORMATTER
# ============================================================

def format_evaluation_report(results: Dict[str, Any]) -> str:
    """Format evaluation results into a clear, readable ASCII scorecard."""
    o = results["overall"]
    p = results["per_column"]

    lines = [
        "=" * 80,
        "                 AI DATA CLEANER QUALITY EVALUATION SCORECARD",
        "=" * 80,
        f"  Overall Precision : {o['precision'] * 100:>6.1f}%  (Correct Repairs / Total Changes)",
        f"  Overall Recall    : {o['recall'] * 100:>6.1f}%  (Fixed Errors / Total Dirty Errors)",
        f"  Overall F1-Score  : {o['f1'] * 100:>6.1f}%  (Harmonic Mean of Precision & Recall)",
        "-" * 80,
        f"  Cell Accuracy     : {o['cell_accuracy_before'] * 100:>5.1f}% -> {o['cell_accuracy_after'] * 100:>5.1f}% ({o['cell_accuracy_gain'] * 100:+5.1f}% gain)",
        f"  Row Accuracy      : {o['row_accuracy_before'] * 100:>5.1f}% -> {o['row_accuracy_after'] * 100:>5.1f}% ({o['row_accuracy_gain'] * 100:+5.1f}% gain)",
        f"  Repairs Applied   : {o['total_repairs_applied']} total ({o['correct_repairs']} correct, {o['incorrect_repairs']} incorrect)",
        f"  Duplicates Dropped: {o['duplicates_removed']} of {o['duplicate_rows_in_dirty']}",
        "-" * 80,
        f"{'Column':<18} {'Precision':>9} {'Recall':>9} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'Status':>10}",
        "-" * 80,
    ]

    for col, m in p.items():
        if m["f1"] == 1.0:
            status = "PERFECT"
        elif m["f1"] >= 0.8:
            status = "HIGH"
        elif m["f1"] >= 0.5:
            status = "MODERATE"
        else:
            status = "LOW/REVIEW"

        lines.append(
            f"{col:<18} {m['precision'] * 100:>8.1f}% {m['recall'] * 100:>8.1f}% {m['f1'] * 100:>7.1f}% {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {status:>10}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate data cleaning quality against ground truth.")
    parser.add_argument("--dirty", "-d", default="data/benchmark/dirty_sample.csv", help="Path to dirty CSV")
    parser.add_argument("--cleaned", "-c", default=None, help="Path to cleaned CSV (or runs/run_xxx/cleaned_data.csv)")
    parser.add_argument("--ground-truth", "-g", default="data/benchmark/ground_truth.csv", help="Path to ground truth CSV")
    parser.add_argument("--id-column", default="customer_id", help="Primary key column for row alignment (default: customer_id)")
    parser.add_argument("--json", action="store_true", help="Output full report as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.dirty):
        print(f"Error: Dirty CSV not found: {args.dirty}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth CSV not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    cleaned_path = args.cleaned
    if not cleaned_path:
        # Default to dirty vs ground truth baseline if no cleaned file given
        print("Notice: No cleaned file specified. Comparing dirty sample against ground truth baseline...")
        cleaned_path = args.dirty

    df_dirty = pd.read_csv(args.dirty)
    df_cleaned = pd.read_csv(cleaned_path)
    df_truth = pd.read_csv(args.ground_truth)

    results = evaluate_cleaning(
        df_dirty=df_dirty,
        df_cleaned=df_cleaned,
        df_ground_truth=df_truth,
        id_column=args.id_column if args.id_column in df_truth.columns else None,
    )

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(format_evaluation_report(results))


if __name__ == "__main__":
    main()
