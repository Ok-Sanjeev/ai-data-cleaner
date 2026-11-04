"""
Unit tests for the safe, restricted GREL-inspired expression engine in DataRefine.
"""

import pandas as pd
import pytest

from grel_engine import (
    CompiledGRELExpression,
    GRELSecurityError,
    GRELSyntaxError,
    execute_grel_transform,
    preview_grel,
)


def test_grel_string_operations():
    c = CompiledGRELExpression("value.toLowercase()")
    assert c.evaluate("HELLO") == "hello"

    c = CompiledGRELExpression("value.trim()")
    assert c.evaluate("  hello world  ") == "hello world"

    c = CompiledGRELExpression("value.toTitlecase()")
    assert c.evaluate("san francisco") == "San Francisco"

    c = CompiledGRELExpression("value.replace('old', 'new')")
    assert c.evaluate("old car") == "new car"

    c = CompiledGRELExpression("value.split(',')[0]")
    assert c.evaluate("first,second,third") == "first"

    c = CompiledGRELExpression("value.length()")
    assert c.evaluate("hello") == 5


def test_grel_numeric_and_arithmetic():
    c = CompiledGRELExpression("value.toNumber() * 2")
    assert c.evaluate("21") == 42.0

    c = CompiledGRELExpression("round(value.toNumber() * 1.18, 2)")
    assert c.evaluate(100) == 118.0


def test_grel_conditional_if():
    c = CompiledGRELExpression("if(value == 'N/A', null, value.toUppercase())")
    assert c.evaluate("N/A") is None
    assert c.evaluate("valid") == "VALID"


def test_grel_row_cells_access():
    c = CompiledGRELExpression("cells['first_name'] + ' ' + cells['last_name']")
    row = {"first_name": "Alan", "last_name": "Turing"}
    assert c.evaluate(None, row=row) == "Alan Turing"


def test_grel_security_blocking():
    # Blocks dunder and reflection
    with pytest.raises(GRELSecurityError):
        CompiledGRELExpression("value.__class__")

    # Blocks import statements
    with pytest.raises(GRELSyntaxError):
        CompiledGRELExpression("import os")

    # Blocks __builtins__
    with pytest.raises(GRELSecurityError):
        CompiledGRELExpression("__builtins__")


def test_grel_preview_and_execute():
    df = pd.DataFrame({
        "city": [" new york ", "LONDON", "  paris  ", "tokyo"],
        "code": [1, 2, 3, 4],
    })

    preview = preview_grel(df, "city", "value.trim().toTitlecase()", sample_limit=3)
    assert preview["success"] is True
    assert len(preview["rows"]) == 3
    assert preview["rows"][0]["transformed"] == "New York"
    assert preview["rows"][1]["transformed"] == "London"

    df_clean = execute_grel_transform(df, "city", "value.trim().toLowercase()")
    assert list(df_clean["city"]) == ["new york", "london", "paris", "tokyo"]
