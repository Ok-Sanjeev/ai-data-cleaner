"""
Cleaning Session Manager for DataRefine.

Provides step-by-step undo, redo, checkpointing, and rollback capabilities
for interactive cleaning sessions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

from actions import execute_action
from schemas import CleaningAction


class CleaningSession:
    """
    Maintains a full checkpoint history of DataFrame states, allowing step-by-step
    undo/redo, jump-to-step rollback, and delta inspections.
    """

    def __init__(self, df: pd.DataFrame, session_name: str = "default"):
        self.session_name = session_name
        self.initial_df = df.copy(deep=True)
        self.current_df = df.copy(deep=True)

        # Step 0 is the initial pristine state
        self.checkpoints: List[Dict[str, Any]] = [
            {
                "step": 0,
                "action": None,
                "action_name": "INITIAL_STATE",
                "column": None,
                "params": {},
                "df": df.copy(deep=True),
                "timestamp": datetime.now().isoformat(),
                "rows": len(df),
                "columns": list(df.columns),
                "description": "Initial dataset before any cleaning operations",
            }
        ]

        # Stack for redoing undone steps
        self.redo_stack: List[Dict[str, Any]] = []

        # Audit log of all actions applied
        self.executed_history: List[Dict[str, Any]] = []

    @property
    def current_step(self) -> int:
        return len(self.checkpoints) - 1

    def can_undo(self) -> bool:
        return len(self.checkpoints) > 1

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def get_current_df(self) -> pd.DataFrame:
        return self.current_df

    def apply_action(self, action: Any) -> Dict[str, Any]:
        """Execute a cleaning action, record before/after states, and push a checkpoint."""
        if isinstance(action, dict):
            action_obj = CleaningAction(**action)
        else:
            action_obj = action

        before_rows = len(self.current_df)
        before_cols = list(self.current_df.columns)
        before_missing = int(self.current_df.isna().sum().sum())

        # Execute deterministic action
        transformed_df = execute_action(self.current_df, action_obj)

        after_rows = len(transformed_df)
        after_cols = list(transformed_df.columns)
        after_missing = int(transformed_df.isna().sum().sum())

        new_step = self.current_step + 1
        action_name = getattr(action_obj, "action", str(action_obj))
        target_col = getattr(action_obj, "column", None)
        col_desc = f" on '{target_col}'" if target_col else ""
        formatted_name = action_name.replace("_", " ").title()

        checkpoint = {
            "step": new_step,
            "action": action_obj.model_dump() if hasattr(action_obj, "model_dump") else action_obj,
            "action_name": action_name,
            "column": target_col,
            "params": getattr(action_obj, "params", {}),
            "df": transformed_df.copy(deep=True),
            "timestamp": datetime.now().isoformat(),
            "rows": after_rows,
            "columns": after_cols,
            "description": f"{formatted_name}{col_desc}",
            "diff": {
                "rows_delta": after_rows - before_rows,
                "cols_delta": len(after_cols) - len(before_cols),
                "missing_delta": after_missing - before_missing,
            },
        }

        self.current_df = transformed_df
        self.checkpoints.append(checkpoint)
        self.redo_stack.clear()

        success_record = {
            "step": new_step,
            "action": action_name,
            "action_name": formatted_name,
            "column": target_col,
            "params": getattr(action_obj, "params", {}),
            "reasoning": getattr(action_obj, "reasoning", ""),
            "status": "success",
            "rows_before": before_rows,
            "rows_after": after_rows,
            "timestamp": checkpoint["timestamp"],
            "description": checkpoint["description"],
        }
        self.executed_history.append(success_record)

        return success_record

    def undo(self) -> Dict[str, Any]:
        """Undo the most recent cleaning operation."""
        if not self.can_undo():
            return {
                "status": "noop",
                "message": "Cannot undo: already at initial dataset state.",
                "current_step": self.current_step,
            }

        last_checkpoint = self.checkpoints.pop()
        self.redo_stack.append(last_checkpoint)

        # Restore previous DataFrame state
        previous_checkpoint = self.checkpoints[-1]
        self.current_df = previous_checkpoint["df"].copy(deep=True)

        return {
            "status": "success",
            "undone_step": last_checkpoint["step"],
            "undone_action": last_checkpoint["action_name"],
            "undone_column": last_checkpoint["column"],
            "current_step": previous_checkpoint["step"],
            "remaining_checkpoints": len(self.checkpoints),
            "message": f"Undid step {last_checkpoint['step']}: {last_checkpoint['action_name']} on '{last_checkpoint['column']}'.",
            "rows": len(self.current_df),
            "columns": list(self.current_df.columns),
        }

    def redo(self) -> Dict[str, Any]:
        """Redo the most recently undone cleaning operation."""
        if not self.can_redo():
            return {
                "status": "noop",
                "message": "Cannot redo: redo stack is empty.",
                "current_step": self.current_step,
            }

        checkpoint_to_restore = self.redo_stack.pop()
        self.checkpoints.append(checkpoint_to_restore)
        self.current_df = checkpoint_to_restore["df"].copy(deep=True)

        return {
            "status": "success",
            "redone_step": checkpoint_to_restore["step"],
            "redone_action": checkpoint_to_restore["action_name"],
            "redone_column": checkpoint_to_restore["column"],
            "current_step": checkpoint_to_restore["step"],
            "remaining_redo": len(self.redo_stack),
            "message": f"Redid step {checkpoint_to_restore['step']}: {checkpoint_to_restore['action_name']} on '{checkpoint_to_restore['column']}'.",
            "rows": len(self.current_df),
            "columns": list(self.current_df.columns),
        }

    def rollback_to(self, target_step: int) -> Dict[str, Any]:
        """Roll back DataFrame to a specific checkpoint step index."""
        if target_step < 0 or target_step >= len(self.checkpoints):
            return {
                "status": "error",
                "message": f"Invalid target step {target_step}. Valid range is 0 to {len(self.checkpoints) - 1}.",
                "current_step": self.current_step,
            }

        if target_step == self.current_step:
            return {
                "status": "noop",
                "current_step": self.current_step,
                "message": f"Already at target step {target_step}.",
            }

        # Pop into redo_stack until we reach target_step
        while len(self.checkpoints) - 1 > target_step:
            popped = self.checkpoints.pop()
            self.redo_stack.append(popped)

        # Restore target DataFrame state
        self.current_df = self.checkpoints[-1]["df"].copy(deep=True)

        return {
            "status": "success",
            "restored_step": target_step,
            "message": f"Successfully rolled back to step {target_step}.",
            "rows": len(self.current_df),
            "columns": list(self.current_df.columns),
            "total_checkpoints": len(self.checkpoints),
        }

    def rollback_all(self) -> Dict[str, Any]:
        """Roll back all cleaning actions and restore the pristine initial DataFrame."""
        return self.rollback_to(0)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return a chronological summary of all checkpoints in the session."""
        history = []
        for cp in self.checkpoints:
            act_name = cp.get("action_name") or "Initial Dataset"
            desc = cp.get("description") or (f"Step {cp['step']}: {act_name}" if act_name != "INITIAL_STATE" else "Initial Dataset")
            history.append({
                "step": cp["step"],
                "action": act_name,
                "action_name": act_name,
                "column": cp.get("column"),
                "params": cp.get("params", {}),
                "rows": cp.get("rows", 0),
                "columns_count": len(cp.get("columns", [])),
                "timestamp": cp.get("timestamp"),
                "description": desc,
            })
        return history

    def diff(self, step_a: int = 0, step_b: Optional[int] = None) -> Dict[str, Any]:
        """Compute structural and cell-level difference between two session checkpoints."""
        if step_b is None:
            step_b = self.current_step

        if step_a < 0 or step_a >= len(self.checkpoints) or step_b < 0 or step_b >= len(self.checkpoints):
            raise ValueError("Step index out of range for diff computation.")

        df_a = self.checkpoints[step_a]["df"]
        df_b = self.checkpoints[step_b]["df"]

        cols_a = set(df_a.columns)
        cols_b = set(df_b.columns)

        common_cols = [c for c in df_b.columns if c in cols_a]
        added_cols = list(cols_b - cols_a)
        removed_cols = list(cols_a - cols_b)

        # Row comparison
        rows_a = len(df_a)
        rows_b = len(df_b)

        # Cell comparison on common columns and minimum rows (aligned with reset_index)
        common_len = min(rows_a, rows_b)
        cell_diffs_count = 0

        if common_len > 0 and common_cols:
            sub_a = df_a.iloc[:common_len][common_cols].reset_index(drop=True)
            sub_b = df_b.iloc[:common_len][common_cols].reset_index(drop=True)
            diff_mask = (sub_a != sub_b) & ~(sub_a.isna() & sub_b.isna())
            cell_diffs_count = int(diff_mask.sum().sum())

        return {
            "step_a": step_a,
            "step_b": step_b,
            "rows_before": rows_a,
            "rows_after": rows_b,
            "rows_delta": rows_b - rows_a,
            "columns_before": list(df_a.columns),
            "columns_after": list(df_b.columns),
            "columns_added": added_cols,
            "columns_removed": removed_cols,
            "modified_cells_count": cell_diffs_count,
            "total_missing_before": int(df_a.isna().sum().sum()),
            "total_missing_after": int(df_b.isna().sum().sum()),
        }
