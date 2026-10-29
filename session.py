"""
Cleaning Session & Rollback Management Module.

Provides step-by-step undo, redo, checkpointing, and rollback capabilities
for data cleaning operations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from actions import execute_action
from schemas import CleaningAction


class CleaningSession:
    """Manages an active data cleaning session with reversible transformations.

    Maintains a full checkpoint history of DataFrame states, allowing step-by-step
    undo, redo, jump-to-step rollback, and full rollback.
    """

    def __init__(self, df: pd.DataFrame, session_name: str = "default_session"):
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

        # Action execution history
        self.executed_history: List[Dict[str, Any]] = []

    @property
    def current_step(self) -> int:
        """Return the current step number (0 = initial)."""
        return len(self.checkpoints) - 1

    def can_undo(self) -> bool:
        """Check if an undo operation is possible."""
        return len(self.checkpoints) > 1

    def can_redo(self) -> bool:
        """Check if a redo operation is possible."""
        return len(self.redo_stack) > 0

    def get_current_df(self) -> pd.DataFrame:
        """Return a copy of the current active DataFrame."""
        return self.current_df.copy(deep=True)

    def apply_action(self, action: Union[CleaningAction, Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a cleaning action, record before/after states, and push a checkpoint.

        Clears the redo stack upon applying a new action.
        """
        # Convert dict to CleaningAction if needed
        if isinstance(action, dict):
            action_obj = CleaningAction.model_validate(action)
        else:
            action_obj = action

        before_rows = len(self.current_df)
        before_cols = list(self.current_df.columns)
        before_missing = int(self.current_df.isna().sum().sum())

        try:
            # Execute on working copy
            transformed_df = execute_action(self.current_df.copy(deep=True), action_obj)
        except Exception as error:
            # Leave current_df untouched on failure
            failure_record = {
                "step": self.current_step + 1,
                "action": action_obj.action,
                "column": action_obj.column,
                "params": action_obj.params,
                "status": "failed",
                "error": str(error),
                "timestamp": datetime.now().isoformat(),
            }
            self.executed_history.append(failure_record)
            raise error

        after_rows = len(transformed_df)
        after_cols = list(transformed_df.columns)
        after_missing = int(transformed_df.isna().sum().sum())

        new_step = self.current_step + 1
        checkpoint = {
            "step": new_step,
            "action": action_obj.model_dump(),
            "action_name": action_obj.action,
            "column": action_obj.column,
            "params": action_obj.params,
            "df": transformed_df.copy(deep=True),
            "timestamp": datetime.now().isoformat(),
            "rows": after_rows,
            "columns": after_cols,
            "description": f"{action_obj.action} on column '{action_obj.column}'",
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
            "action": action_obj.action,
            "column": action_obj.column,
            "params": action_obj.params,
            "reasoning": getattr(action_obj, "reasoning", ""),
            "status": "success",
            "rows_before": before_rows,
            "rows_after": after_rows,
            "timestamp": checkpoint["timestamp"],
        }
        self.executed_history.append(success_record)

        return success_record

    def undo(self) -> Dict[str, Any]:
        """Revert the most recent cleaning action and restore previous DataFrame state."""
        if not self.can_undo():
            return {
                "status": "noop",
                "message": "Cannot undo: already at initial dataset state.",
                "current_step": 0,
            }

        last_checkpoint = self.checkpoints.pop()
        self.redo_stack.append(last_checkpoint)

        # Restore previous state
        previous_checkpoint = self.checkpoints[-1]
        self.current_df = previous_checkpoint["df"].copy(deep=True)

        return {
            "status": "success",
            "undone_step": last_checkpoint["step"],
            "undone_action": last_checkpoint["action_name"],
            "undone_column": last_checkpoint["column"],
            "current_step": previous_checkpoint["step"],
            "rows_now": len(self.current_df),
            "message": f"Undid step {last_checkpoint['step']}: {last_checkpoint['action_name']} on '{last_checkpoint['column']}'.",
        }

    def redo(self) -> Dict[str, Any]:
        """Re-apply the last undone action from the redo stack."""
        if not self.can_redo():
            return {
                "status": "noop",
                "message": "Cannot redo: no undone actions available.",
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
            "rows_now": len(self.current_df),
            "message": f"Redid step {checkpoint_to_restore['step']}: {checkpoint_to_restore['action_name']} on '{checkpoint_to_restore['column']}'.",
        }

    def rollback_to(self, target_step: int) -> Dict[str, Any]:
        """Roll back the session to an arbitrary earlier step index."""
        if target_step < 0 or target_step > self.current_step:
            return {
                "status": "error",
                "message": f"Target step {target_step} is out of bounds (0 to {self.current_step}).",
            }

        if target_step == self.current_step:
            return {
                "status": "noop",
                "message": f"Session is already at step {target_step}.",
                "current_step": self.current_step,
            }

        reverted_actions = []
        while self.current_step > target_step:
            popped = self.checkpoints.pop()
            self.redo_stack.append(popped)
            reverted_actions.append(f"Step {popped['step']}: {popped['action_name']}")

        self.current_df = self.checkpoints[-1]["df"].copy(deep=True)

        return {
            "status": "success",
            "target_step": target_step,
            "steps_reverted": len(reverted_actions),
            "reverted_actions": reverted_actions,
            "rows_now": len(self.current_df),
            "message": f"Rolled back to step {target_step}. Reverted {len(reverted_actions)} action(s).",
        }

    def rollback_all(self) -> Dict[str, Any]:
        """Roll back all cleaning operations, restoring the original dataset."""
        return self.rollback_to(0)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return a chronological summary of all checkpoints in the session."""
        history = []
        for cp in self.checkpoints:
            history.append({
                "step": cp["step"],
                "action": cp["action_name"],
                "column": cp["column"],
                "params": cp["params"],
                "rows": cp["rows"],
                "columns_count": len(cp["columns"]),
                "timestamp": cp["timestamp"],
                "description": cp["description"],
            })
        return history

    def diff(self, step_a: int = 0, step_b: Optional[int] = None) -> Dict[str, Any]:
        """Compute structural and cell-level difference between two session checkpoints."""
        if step_b is None:
            step_b = self.current_step

        if step_a < 0 or step_a >= len(self.checkpoints) or step_b < 0 or step_b >= len(self.checkpoints):
            raise ValueError(f"Step indices ({step_a}, {step_b}) must be within 0 and {self.current_step}.")

        df_a = self.checkpoints[step_a]["df"]
        df_b = self.checkpoints[step_b]["df"]

        cols_added = [c for c in df_b.columns if c not in df_a.columns]
        cols_removed = [c for c in df_a.columns if c not in df_b.columns]
        common_cols = [c for c in df_a.columns if c in df_b.columns]

        rows_a = len(df_a)
        rows_b = len(df_b)

        # Cell comparison on common columns and overlapping rows
        min_rows = min(rows_a, rows_b)
        cells_modified = 0
        if min_rows > 0 and common_cols:
            arr_a = df_a[common_cols].iloc[:min_rows].to_numpy()
            arr_b = df_b[common_cols].iloc[:min_rows].to_numpy()
            null_a = pd.isna(arr_a)
            null_b = pd.isna(arr_b)
            both_null = null_a & null_b
            either_null = null_a | null_b
            diff_mask = (either_null & ~both_null) | (~either_null & (arr_a != arr_b))
            cells_modified = int(diff_mask.sum())

        return {
            "step_a": step_a,
            "step_b": step_b,
            "rows_before": rows_a,
            "rows_after": rows_b,
            "rows_delta": rows_b - rows_a,
            "columns_added": cols_added,
            "columns_removed": cols_removed,
            "cells_modified_in_shared_rows": cells_modified,
        }
