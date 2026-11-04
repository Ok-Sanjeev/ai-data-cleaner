"""
Safe, Restricted GREL-Inspired (General Refine Expression Language) Engine for DataRefine.

Provides an AST-parsed, sandboxed expression evaluator without using arbitrary
`eval` or `exec`. Strictly validates allowed nodes and forbids dunder attributes,
imports, and system access.

Supports OpenRefine-style syntax:
- `value.toLowercase()`
- `value.trim()`
- `value.replace('foo', 'bar')`
- `value.split(',')[0]`
- `value.toNumber() * 1.18`
- `if(value == 'N/A', null, value.toTitlecase())`
- `cells['other_col']` / `row['other_col']`
"""

from __future__ import annotations

import ast
import math
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class GRELSyntaxError(Exception):
    """Raised when an expression has invalid syntax."""
    pass


class GRELSecurityError(Exception):
    """Raised when an expression contains forbidden nodes or attributes."""
    pass


class GRELExecutionError(Exception):
    """Raised when evaluation fails during execution."""
    pass


# Allowed AST nodes for safe arithmetic, comparisons, attribute calls, and conditions
ALLOWED_AST_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.IfExp,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Attribute,
    ast.Call,
    ast.keyword,
    ast.Load,
)

FORBIDDEN_ATTRIBUTES = {
    "__class__",
    "__globals__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__import__",
    "__code__",
    "__closure__",
    "__builtins__",
    "__dict__",
    "mro",
}


# =====================================================================
# SAFE WRAPPER OBJECTS FOR GREL METHOD CHAINING
# =====================================================================

class GRELValueWrapper:
    """Wraps cell values to provide GREL-style method chaining safely."""

    def __init__(self, val: Any):
        if pd.isna(val) or val is None:
            self._val = None
        else:
            self._val = val

    @property
    def raw(self) -> Any:
        return self._val

    def __repr__(self) -> str:
        return f"GRELValue({self._val!r})"

    def __str__(self) -> str:
        return "" if self._val is None else str(self._val)

    def __eq__(self, other: Any) -> bool:
        other_val = other._val if isinstance(other, GRELValueWrapper) else other
        return self._val == other_val

    def __bool__(self) -> bool:
        return bool(self._val)

    def __add__(self, other: Any) -> Any:
        other_val = other._val if isinstance(other, GRELValueWrapper) else other
        if self._val is None or other_val is None:
            return None
        return self._val + other_val

    def __sub__(self, other: Any) -> Any:
        other_val = other._val if isinstance(other, GRELValueWrapper) else other
        if self._val is None or other_val is None:
            return None
        return self._val - other_val

    def __mul__(self, other: Any) -> Any:
        other_val = other._val if isinstance(other, GRELValueWrapper) else other
        if self._val is None or other_val is None:
            return None
        return self._val * other_val

    def __truediv__(self, other: Any) -> Any:
        other_val = other._val if isinstance(other, GRELValueWrapper) else other
        if self._val is None or other_val is None:
            return None
        return self._val / other_val

    # String transformations
    def toLowercase(self) -> str:
        return "" if self._val is None else str(self._val).lower()

    def toUppercase(self) -> str:
        return "" if self._val is None else str(self._val).upper()

    def toTitlecase(self) -> str:
        return "" if self._val is None else str(self._val).title()

    def trim(self) -> str:
        return "" if self._val is None else str(self._val).strip()

    def strip(self) -> str:
        return self.trim()

    def length(self) -> int:
        if self._val is None:
            return 0
        if isinstance(self._val, (str, list, tuple, dict)):
            return len(self._val)
        return len(str(self._val))

    def split(self, sep: Optional[str] = None) -> List[str]:
        if self._val is None:
            return []
        return str(self._val).split(sep)

    def replace(self, old: str, new: str) -> str:
        if self._val is None:
            return ""
        return str(self._val).replace(str(old), str(new))

    def replaceRegex(self, pattern: str, repl: str) -> str:
        if self._val is None:
            return ""
        return re.sub(str(pattern), str(repl), str(self._val))

    def match(self, pattern: str) -> List[str]:
        if self._val is None:
            return []
        m = re.findall(str(pattern), str(self._val))
        return [str(x) for x in m]

    def substring(self, start: int, end: Optional[int] = None) -> str:
        if self._val is None:
            return ""
        s = str(self._val)
        start = int(start)
        end = int(end) if end is not None else len(s)
        return s[start:end]

    def slice(self, start: int, end: Optional[int] = None) -> str:
        return self.substring(start, end)

    def contains(self, sub: str) -> bool:
        if self._val is None:
            return False
        return str(sub) in str(self._val)

    def startsWith(self, prefix: str) -> bool:
        if self._val is None:
            return False
        return str(self._val).startswith(str(prefix))

    def endsWith(self, suffix: str) -> bool:
        if self._val is None:
            return False
        return str(self._val).endswith(str(suffix))

    def toNumber(self) -> Optional[Union[int, float]]:
        if self._val is None:
            return None
        cleaned = re.sub(r"[^\d\.-]", "", str(self._val))
        if not cleaned:
            return None
        try:
            if "." in cleaned:
                return float(cleaned)
            return int(cleaned)
        except (ValueError, TypeError):
            return None

    def toString(self) -> str:
        return "" if self._val is None else str(self._val)

    def toDate(self, fmt: Optional[str] = None) -> Optional[str]:
        if self._val is None:
            return None
        try:
            if fmt:
                dt = datetime.strptime(str(self._val), fmt)
            else:
                dt = pd.to_datetime(self._val, errors="coerce")
                if pd.isna(dt):
                    return None
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    def formatDate(self, fmt: str = "%Y-%m-%d") -> Optional[str]:
        return self.toDate(fmt)

    def isNull(self) -> bool:
        return self._val is None

    def isNotNull(self) -> bool:
        return self._val is not None

    def isEmpty(self) -> bool:
        if self._val is None:
            return True
        return len(str(self._val).strip()) == 0


# =====================================================================
# AST VISITOR & SAFE EVALUATOR
# =====================================================================

class SafeGRELVisitor:
    """Evaluates an AST expression safely within a restricted environment."""

    def __init__(self, context: Dict[str, Any]):
        self.context = context

    def eval(self, node: ast.AST) -> Any:
        method_name = f"eval_{type(node).__name__}"
        visitor = getattr(self, method_name, None)
        if visitor is None:
            raise GRELSecurityError(f"AST node '{type(node).__name__}' is forbidden in GREL expressions.")
        return visitor(node)

    def eval_Expression(self, node: ast.Expression) -> Any:
        return self.eval(node.body)

    def eval_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def eval_Name(self, node: ast.Name) -> Any:
        name = node.id
        if name in FORBIDDEN_ATTRIBUTES:
            raise GRELSecurityError(f"Access to '{name}' is forbidden.")
        if name in self.context:
            return self.context[name]
        if name == "null" or name == "nil":
            return None
        if name == "true":
            return True
        if name == "false":
            return False
        raise GRELExecutionError(f"Undefined variable '{name}'. Available: {list(self.context.keys())}")

    def eval_UnaryOp(self, node: ast.UnaryOp) -> Any:
        val = self.eval(node.operand)
        if isinstance(node.op, ast.Not):
            return not val
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
        raise GRELExecutionError(f"Unsupported unary operator: {node.op}")

    def eval_BinOp(self, node: ast.BinOp) -> Any:
        left = self.eval(node.left)
        right = self.eval(node.right)

        # Unbox wrappers if needed
        left_val = left._val if isinstance(left, GRELValueWrapper) else left
        right_val = right._val if isinstance(right, GRELValueWrapper) else right

        if left_val is None or right_val is None:
            if isinstance(node.op, ast.Add) and (isinstance(left_val, str) or isinstance(right_val, str)):
                return str("" if left_val is None else left_val) + str("" if right_val is None else right_val)
            return None

        if isinstance(node.op, ast.Add):
            return left_val + right_val
        if isinstance(node.op, ast.Sub):
            return left_val - right_val
        if isinstance(node.op, ast.Mult):
            return left_val * right_val
        if isinstance(node.op, ast.Div):
            return left_val / right_val if right_val != 0 else None
        if isinstance(node.op, ast.FloorDiv):
            return left_val // right_val if right_val != 0 else None
        if isinstance(node.op, ast.Mod):
            return left_val % right_val if right_val != 0 else None
        if isinstance(node.op, ast.Pow):
            if abs(right_val) > 1000:
                raise GRELExecutionError("Exponent too large")
            return left_val ** right_val

        raise GRELExecutionError(f"Unsupported binary operator: {node.op}")

    def eval_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            for v in node.values:
                res = self.eval(v)
                if not res:
                    return res
            return res
        if isinstance(node.op, ast.Or):
            for v in node.values:
                res = self.eval(v)
                if res:
                    return res
            return res
        raise GRELExecutionError(f"Unsupported boolean operator: {node.op}")

    def eval_Compare(self, node: ast.Compare) -> bool:
        left = self.eval(node.left)
        for op, comp in zip(node.ops, node.comparators):
            right = self.eval(comp)
            l = left._val if isinstance(left, GRELValueWrapper) else left
            r = right._val if isinstance(right, GRELValueWrapper) else right

            if isinstance(op, ast.Eq):
                matched = (l == r)
            elif isinstance(op, ast.NotEq):
                matched = (l != r)
            elif isinstance(op, ast.Lt):
                matched = (l < r) if (l is not None and r is not None) else False
            elif isinstance(op, ast.LtE):
                matched = (l <= r) if (l is not None and r is not None) else False
            elif isinstance(op, ast.Gt):
                matched = (l > r) if (l is not None and r is not None) else False
            elif isinstance(op, ast.GtE):
                matched = (l >= r) if (l is not None and r is not None) else False
            elif isinstance(op, ast.In):
                matched = (l in r) if r is not None else False
            elif isinstance(op, ast.NotIn):
                matched = (l not in r) if r is not None else True
            elif isinstance(op, ast.Is):
                matched = (l is r)
            elif isinstance(op, ast.IsNot):
                matched = (l is not r)
            else:
                raise GRELExecutionError(f"Unsupported comparison: {op}")

            if not matched:
                return False
            left = right
        return True

    def eval_IfExp(self, node: ast.IfExp) -> Any:
        test = self.eval(node.test)
        if test:
            return self.eval(node.body)
        return self.eval(node.orelse)

    def eval_Subscript(self, node: ast.Subscript) -> Any:
        val = self.eval(node.value)
        val = val._val if isinstance(val, GRELValueWrapper) else val
        slice_val = self.eval(node.slice)
        if val is None:
            return None
        try:
            return val[slice_val]
        except (IndexError, KeyError):
            return None

    def eval_Slice(self, node: ast.Slice) -> slice:
        lower = self.eval(node.lower) if node.lower else None
        upper = self.eval(node.upper) if node.upper else None
        step = self.eval(node.step) if node.step else None
        return slice(lower, upper, step)

    def eval_List(self, node: ast.List) -> List[Any]:
        return [self.eval(e) for e in node.elts]

    def eval_Tuple(self, node: ast.Tuple) -> Tuple[Any, ...]:
        return tuple(self.eval(e) for e in node.elts)

    def eval_Dict(self, node: ast.Dict) -> Dict[Any, Any]:
        return {self.eval(k): self.eval(v) for k, v in zip(node.keys, node.values)}

    def eval_Attribute(self, node: ast.Attribute) -> Any:
        attr = node.attr
        if attr in FORBIDDEN_ATTRIBUTES or attr.startswith("_"):
            raise GRELSecurityError(f"Access to private attribute '{attr}' is forbidden.")

        target = self.eval(node.value)

        # Automatic wrapping for GREL chaining
        if not isinstance(target, GRELValueWrapper):
            if isinstance(target, dict):
                return target.get(attr)
            target = GRELValueWrapper(target)

        if hasattr(target, attr):
            return getattr(target, attr)

        raise GRELExecutionError(f"Unknown GREL property or method '{attr}'.")

    def eval_Call(self, node: ast.Call) -> Any:
        func = self.eval(node.func)
        args = [self.eval(arg) for arg in node.args]
        kwargs = {kw.arg: self.eval(kw.value) for kw in node.keywords}

        if not callable(func):
            raise GRELExecutionError(f"Target '{func}' is not callable.")

        try:
            # Unwrap GRELValueWrapper if native function expects primitive
            unwrapped_args = [a._val if isinstance(a, GRELValueWrapper) else a for a in args]
            return func(*unwrapped_args, **kwargs)
        except Exception as e:
            raise GRELExecutionError(f"Error calling function: {e}")


# =====================================================================
# STANDALONE GREL BUILT-IN FUNCTIONS
# =====================================================================

def grel_if(condition: Any, true_val: Any, false_val: Any) -> Any:
    return true_val if condition else false_val

def grel_is_null(val: Any) -> bool:
    return val is None or pd.isna(val)

def grel_is_not_null(val: Any) -> bool:
    return not grel_is_null(val)

def grel_is_empty(val: Any) -> bool:
    if grel_is_null(val):
        return True
    return len(str(val).strip()) == 0

def grel_to_number(val: Any) -> Optional[Union[int, float]]:
    return GRELValueWrapper(val).toNumber()

def grel_to_string(val: Any) -> str:
    return GRELValueWrapper(val).toString()

def grel_to_date(val: Any, fmt: Optional[str] = None) -> Optional[str]:
    return GRELValueWrapper(val).toDate(fmt)

def grel_length(val: Any) -> int:
    return GRELValueWrapper(val).length()

def grel_round(val: Any, decimals: int = 0) -> Optional[Union[int, float]]:
    num = grel_to_number(val)
    if num is None:
        return None
    if decimals == 0:
        return round(num)
    return round(num, decimals)

def grel_floor(val: Any) -> Optional[int]:
    num = grel_to_number(val)
    return math.floor(num) if num is not None else None

def grel_ceil(val: Any) -> Optional[int]:
    num = grel_to_number(val)
    return math.ceil(num) if num is not None else None

def grel_abs(val: Any) -> Optional[Union[int, float]]:
    num = grel_to_number(val)
    return abs(num) if num is not None else None

def grel_coalesce(*args: Any) -> Any:
    for a in args:
        if a is not None and not pd.isna(a):
            return a
    return None


GREL_BUILTINS: Dict[str, Any] = {
    "if": grel_if,
    "_grel_if": grel_if,
    "isNull": grel_is_null,
    "isNotNull": grel_is_not_null,
    "isEmpty": grel_is_empty,
    "toNumber": grel_to_number,
    "toString": grel_to_string,
    "toDate": grel_to_date,
    "length": grel_length,
    "round": grel_round,
    "floor": grel_floor,
    "ceil": grel_ceil,
    "abs": grel_abs,
    "min": min,
    "max": max,
    "coalesce": grel_coalesce,
}


# =====================================================================
# COMPILED GREL EXPRESSION
# =====================================================================

class CompiledGRELExpression:
    """Pre-compiles and validates an AST expression for fast, safe evaluation."""

    def __init__(self, expression: str):
        self.raw_expression = expression.strip()
        if not self.raw_expression:
            raise GRELSyntaxError("Expression cannot be empty.")

        # OpenRefine compatibility normalization:
        # e.g., 'null' -> 'None', 'true' -> 'True', 'false' -> 'False'
        normalized = self._normalize_syntax(self.raw_expression)

        try:
            self.parsed_ast = ast.parse(normalized, mode="eval")
        except SyntaxError as e:
            raise GRELSyntaxError(f"Syntax error in GREL expression: {e}")

        self._validate_ast(self.parsed_ast)

    def _normalize_syntax(self, expr: str) -> str:
        """Translate OpenRefine idiosyncrasies into standard parseable Python AST."""
        # Replace unquoted null, true, false with Python equivalents
        expr = re.sub(r"\bnull\b", "None", expr)
        expr = re.sub(r"\btrue\b", "True", expr)
        expr = re.sub(r"\bfalse\b", "False", expr)
        expr = re.sub(r"\bif\s*\(", "_grel_if(", expr)
        return expr

    def _validate_ast(self, node: ast.AST):
        """Recursively ensure every node belongs to the whitelist and has no dunder access."""
        for subnode in ast.walk(node):
            if not isinstance(subnode, ALLOWED_AST_NODES):
                raise GRELSecurityError(
                    f"Forbidden syntax node '{type(subnode).__name__}'. Only safe expressions are allowed."
                )
            if isinstance(subnode, ast.Attribute):
                if subnode.attr in FORBIDDEN_ATTRIBUTES or subnode.attr.startswith("__"):
                    raise GRELSecurityError(f"Forbidden attribute '{subnode.attr}'.")
            if isinstance(subnode, ast.Name):
                if subnode.id in FORBIDDEN_ATTRIBUTES or subnode.id.startswith("__"):
                    raise GRELSecurityError(f"Forbidden identifier '{subnode.id}'.")

    def evaluate(self, value: Any, row: Optional[Dict[str, Any]] = None, index: int = 0) -> Any:
        """Evaluate expression on a single value and optional row context."""
        row_dict = row or {}
        context = {
            **GREL_BUILTINS,
            "value": GRELValueWrapper(value),
            "v": GRELValueWrapper(value),
            "row": row_dict,
            "cells": row_dict,
            "index": index,
            "rowIndex": index + 1,
        }
        visitor = SafeGRELVisitor(context)
        res = visitor.eval(self.parsed_ast)
        if isinstance(res, GRELValueWrapper):
            return res._val
        return res


# =====================================================================
# BATCH EXECUTION & PREVIEW APIS
# =====================================================================

def preview_grel(
    df: pd.DataFrame,
    column: str,
    expression: str,
    sample_limit: int = 15,
) -> Dict[str, Any]:
    """
    Evaluate GREL expression on a sample of rows and return before/after preview.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in dataset.")

    try:
        compiled = CompiledGRELExpression(expression)
    except Exception as e:
        return {
            "success": False,
            "error_message": str(e),
            "rows": [],
        }

    sample_df = df.head(sample_limit)
    preview_rows = []

    for idx, (original_idx, row) in enumerate(sample_df.iterrows()):
        val = row[column]
        row_dict = row.to_dict()
        try:
            transformed = compiled.evaluate(val, row=row_dict, index=idx)
            # Ensure JSON-compliant scalar
            if pd.isna(transformed) or transformed is None:
                transformed = None
            elif isinstance(transformed, (np.integer, int)):
                transformed = int(transformed)
            elif isinstance(transformed, (np.floating, float)):
                transformed = float(transformed)
            else:
                transformed = str(transformed)

            preview_rows.append({
                "row_index": idx,
                "original": None if pd.isna(val) else str(val),
                "transformed": transformed,
                "error": None,
            })
        except Exception as e:
            preview_rows.append({
                "row_index": idx,
                "original": None if pd.isna(val) else str(val),
                "transformed": None,
                "error": str(e),
            })

    return {
        "success": True,
        "error_message": None,
        "rows": preview_rows,
    }


def execute_grel_transform(
    df: pd.DataFrame,
    column: str,
    expression: str,
    on_error: str = "keep",
) -> pd.DataFrame:
    """
    Apply GREL expression to an entire DataFrame column and return modified copy.
    on_error options: 'keep' (retain original), 'set_null', 'set_empty'
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    compiled = CompiledGRELExpression(expression)
    res_df = df.copy()

    transformed_vals = []
    for idx, (_, row) in enumerate(df.iterrows()):
        val = row[column]
        try:
            new_val = compiled.evaluate(val, row=row.to_dict(), index=idx)
            transformed_vals.append(new_val)
        except Exception:
            if on_error == "set_null":
                transformed_vals.append(None)
            elif on_error == "set_empty":
                transformed_vals.append("")
            else:
                transformed_vals.append(val)

    res_df[column] = transformed_vals
    return res_df
