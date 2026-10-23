import pandas as pd
from types import SimpleNamespace

from recommendation_guardrails import filter_unsafe_text_normalization, is_free_text_or_case_sensitive_field


def action(column, case="lower"):
    return SimpleNamespace(action="normalize_text", column=column, params={"case": case})


def test_description_semantic_type_blocks_lowercase_recommendation():
    df = pd.DataFrame({"description": ["SQL and Power BI Experience", "AWS Cloud Solutions"]})
    profile = {"column_profiles": {"description": {"semantic_type": "descriptive_text", "semantic_hints": ["inconsistent_text_case"]}}}
    kept, removed = filter_unsafe_text_normalization([action("description")], df, profile)
    assert kept == []
    assert "free text" in removed[0]


def test_description_and_skills_names_are_preserved_even_without_semantic_hint():
    assert is_free_text_or_case_sensitive_field("job_description")
    assert is_free_text_or_case_sensitive_field("skills_required")
    assert is_free_text_or_case_sensitive_field("company_name")


def test_categorical_case_normalization_requires_local_evidence():
    df = pd.DataFrame({"membership_status": ["Gold", "gold", "GOLD"]})
    action_obj = action("membership_status")
    profile = {"column_profiles": {"membership_status": {"semantic_type": "categorical_or_text", "semantic_hints": ["inconsistent_text_case"]}}}
    kept, removed = filter_unsafe_text_normalization([action_obj], df, profile)
    assert kept == [action_obj]
    assert removed == []


def test_no_casing_issue_means_normalization_is_removed():
    df = pd.DataFrame({"membership_status": ["Gold", "Silver"]})
    profile = {"column_profiles": {"membership_status": {"semantic_type": "categorical_or_text", "semantic_hints": []}}}
    kept, removed = filter_unsafe_text_normalization([action("membership_status")], df, profile)
    assert kept == []
    assert "did not detect" in removed[0]


def test_non_text_column_normalization_is_removed():
    df = pd.DataFrame({"age": [20, 21]})
    profile = {"column_profiles": {"age": {"semantic_type": "numeric_measure_or_identifier", "semantic_hints": ["inconsistent_text_case"]}}}
    kept, removed = filter_unsafe_text_normalization([action("age")], df, profile)
    assert kept == []
    assert "not a text/category dtype" in removed[0]
