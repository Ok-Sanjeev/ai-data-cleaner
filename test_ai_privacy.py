import json

import pytest

from ai_privacy import build_ai_safe_profile, remap_plan_columns


def test_profile_keeps_column_names_but_excludes_real_values_and_category_labels():
    profile = {
        "rows": 4,
        "columns": 2,
        "duplicate_rows": 0,
        "column_name_issues": [],
        "column_profiles": {
            "customer_email": {
                "dtype": "object", "missing_count": 0, "missing_percentage": 0.0,
                "unique_count": 4, "empty_column": False, "constant_column": False,
                "sample_values": ["person@example.com", "secret@example.com"],
                "value_counts": {"person@example.com": 1, "secret@example.com": 1},
                "semantic_type": "email", "sensitive_field": True,
            },
            "employment_status": {
                "dtype": "object", "missing_count": 1, "missing_percentage": 25.0,
                "unique_count": 2, "empty_column": False, "constant_column": False,
                "sample_values": ["Top Secret", "Active"],
                "value_counts": {"Top Secret": 1, "Active": 2},
            },
        },
    }
    safe, identity_map = build_ai_safe_profile(profile)
    serialized = json.dumps(safe)
    assert "customer_email" in serialized
    assert "employment_status" in serialized
    assert "semantic_type" in serialized
    for forbidden in [
        "person@example.com", "secret@example.com", "Top Secret", "Active",
        "sample_values", "value_counts",
    ]:
        assert forbidden not in serialized
    assert identity_map == {"customer_email": "customer_email", "employment_status": "employment_status"}


def test_plan_keeps_original_column_names():
    plan = {"summary": "Email column has missing values.", "actions": [{
        "action": "impute_missing", "column": "customer_email",
        "params": {"method": "mode"}, "reasoning": "customer_email has missing values.",
    }]}
    mapped = remap_plan_columns(plan, {"customer_email": "customer_email"})
    assert mapped["actions"][0]["column"] == "customer_email"


def test_unknown_column_is_rejected():
    with pytest.raises(ValueError, match="unknown column"):
        remap_plan_columns({"actions": [{"column": "not_a_real_column"}]}, {"email": "email"})


def test_column_name_issue_values_are_not_forwarded():
    profile = {
        "rows": 1, "columns": 1, "duplicate_rows": 0,
        "column_name_issues": [{"original": "private_name_123", "suggested": "private_name_123"}],
        "column_profiles": {"email": {"dtype": "object", "unique_count": 1}},
    }
    safe, _ = build_ai_safe_profile(profile)
    serialized = json.dumps(safe)
    assert "private_name_123" not in serialized
    assert safe["column_name_issue_count"] == 1


def test_remap_plan_columns_resolves_case_only_mismatch():
    from ai_privacy import remap_plan_columns
    plan = {"actions": [{"action": "convert_to_date", "column": "join_date"}]}
    result = remap_plan_columns(plan, {"Join_Date": "Join_Date"})
    assert result["actions"][0]["column"] == "Join_Date"


def test_remap_plan_columns_rejects_unknown_column():
    import pytest
    from ai_privacy import remap_plan_columns
    plan = {"actions": [{"action": "convert_to_date", "column": "not_a_column"}]}
    with pytest.raises(ValueError, match="unknown column"):
        remap_plan_columns(plan, {"Join_Date": "Join_Date"})
