from pathlib import Path


def test_planner_prompt_has_safety_and_usefulness_rules():
    source = Path(__file__).with_name("planner.py").read_text(encoding="utf-8")
    assert "Never invent, infer, correct, or substitute a column name" in source
    assert "A constant column may be proposed for removal" in source
    assert "Preserve descriptive_text columns by default" in source
    assert "Never use mean or median for categories" in source
    assert "Do NOT propose clip_outliers for this project" in source
    assert "salary text, skill" in source


def test_local_text_guardrail_is_applied_before_plan_save():
    source = Path(__file__).with_name("planner.py").read_text(encoding="utf-8")
    assert "filter_unsafe_text_normalization" in source
    assert "APPLY LOCAL TEXT-NORMALIZATION GUARDRAILS" in source
