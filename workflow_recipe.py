"""
OpenRefine-Compatible Workflow Recipe & Replay Engine for DataRefine.

Supports:
- Exporting session cleaning history as a portable, serializable JSON recipe.
- Importing and replaying recipes onto the current dataset or new datasets.
- Schema validation of recipe steps against the trusted deterministic action registry.
- Generating before/after execution reports and checkpoint rollback points.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from actions import execute_action
from schemas import CleaningAction


# Map DataRefine actions to OpenRefine operation types
ACTION_TO_OP_TYPE = {
    "trim_whitespace": "core/text-transform",
    "normalize_whitespace": "core/text-transform",
    "normalize_text": "core/text-transform",
    "standardize_case": "core/text-transform",
    "normalize_unicode": "core/text-transform",
    "remove_html_tags": "core/text-transform",
    "grel_transform": "core/grel-transform",
    "standardize_categories": "core/mass-edit",
    "merge_clusters": "core/mass-edit",
    "impute_missing": "core/impute",
    "fill_with_constant": "core/impute",
    "drop_missing_rows": "core/row-removal",
    "remove_duplicates": "core/row-removal",
    "remove_column": "core/column-removal",
    "rename_column": "core/column-rename",
    "reconcile_column": "core/reconciliation",
}


class WorkflowRecipeManager:
    """Manages creation, serialization, validation, and execution of JSON cleaning recipes."""

    @staticmethod
    def export_recipe_from_history(history: List[Dict[str, Any]], title: str = "DataRefine Recipe") -> Dict[str, Any]:
        """Convert session execution history into an OpenRefine-compatible JSON recipe."""
        operations = []
        for step in history:
            action_name = step.get("action")
            if not action_name or action_name == "INITIAL_STATE":
                continue

            op_type = ACTION_TO_OP_TYPE.get(action_name, "core/custom-operation")
            op_dict = {
                "op": op_type,
                "description": step.get("description", f"Execute {action_name}"),
                "action": action_name,
                "column": step.get("column"),
                "params": step.get("params", {}),
                "reasoning": step.get("reasoning", "User-approved or automated transformation"),
            }
            operations.append(op_dict)

        return {
            "title": title,
            "version": "2.0.0",
            "format": "openrefine_datarefine_v2",
            "created_at": datetime.now().isoformat(),
            "operations_count": len(operations),
            "operations": operations,
        }

    @staticmethod
    def validate_recipe(recipe: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate recipe format and verify all actions belong to allowed registry."""
        errors = []
        operations = recipe.get("operations")
        if operations is None or not isinstance(operations, list):
            return False, ["Recipe must contain an 'operations' list."]

        from schemas import ALLOWED_ACTIONS
        allowed_set = set(ALLOWED_ACTIONS.__args__)

        for i, op in enumerate(operations):
            if not isinstance(op, dict):
                errors.append(f"Operation #{i + 1} is not a valid JSON object.")
                continue

            action = op.get("action")
            if not action:
                errors.append(f"Operation #{i + 1} is missing required 'action' field.")
            elif action not in allowed_set:
                errors.append(f"Operation #{i + 1} references unsupported action '{action}'.")

        return len(errors) == 0, errors

    @classmethod
    def replay_recipe(
        cls,
        df: pd.DataFrame,
        recipe: Dict[str, Any],
        stop_on_error: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Replay an entire recipe onto a DataFrame sequentially.
        Returns the transformed DataFrame and execution report.
        """
        valid, errors = cls.validate_recipe(recipe)
        if not valid:
            raise ValueError(f"Invalid recipe: {'; '.join(errors)}")

        operations = recipe["operations"]
        current_df = df.copy(deep=True)
        step_results = []

        initial_rows = len(current_df)
        initial_cols = len(current_df.columns)

        for idx, op in enumerate(operations):
            action_name = op["action"]
            column = op.get("column")
            params = op.get("params", {})
            reasoning = op.get("reasoning", f"Replaying recipe step {idx + 1}")

            try:
                action_obj = CleaningAction(
                    action=action_name,
                    column=column,
                    params=params,
                    reasoning=reasoning,
                )

                before_r = len(current_df)
                before_missing = int(current_df.isna().sum().sum())

                current_df = execute_action(current_df, action_obj)

                after_r = len(current_df)
                after_missing = int(current_df.isna().sum().sum())

                step_results.append({
                    "step": idx + 1,
                    "action": action_name,
                    "column": column,
                    "status": "success",
                    "rows_affected": abs(after_r - before_r),
                    "missing_delta": after_missing - before_missing,
                    "error": None,
                })
            except Exception as e:
                step_results.append({
                    "step": idx + 1,
                    "action": action_name,
                    "column": column,
                    "status": "failed",
                    "error": str(e),
                })
                if stop_on_error:
                    break

        report = {
            "title": recipe.get("title", "Replayed Recipe"),
            "total_steps": len(operations),
            "successful_steps": sum(1 for s in step_results if s["status"] == "success"),
            "executed_count": sum(1 for s in step_results if s["status"] == "success"),
            "failed_steps": sum(1 for s in step_results if s["status"] == "failed"),
            "initial_rows": initial_rows,
            "final_rows": len(current_df),
            "initial_cols": initial_cols,
            "final_cols": len(current_df.columns),
            "step_results": step_results,
            "executed_at": datetime.now().isoformat(),
        }

        return current_df, report
