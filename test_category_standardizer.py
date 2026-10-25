import pandas as pd
from category_standardizer import generate_local_category_standardizations


def test_category_standardizer_groups_casing_and_typos():
    df = pd.DataFrame({
        "status": ["Active", "active", "ACTIVE", "Inactive", "inactive"],
        "region": ["North", "north", "South", "south", "East"],
    })
    profile = {
        "column_profiles": {
            "status": {
                "semantic_type": "categorical_or_text",
                "semantic_hints": ["inconsistent_categories"],
            },
            "region": {
                "semantic_type": "categorical_or_text",
                "semantic_hints": ["inconsistent_categories"],
            },
        }
    }

    actions = generate_local_category_standardizations(df, profile)
    assert len(actions) == 2

    status_action = next(a for a in actions if a.column == "status")
    mapping = status_action.params["mapping"]
    assert "active" in mapping or "ACTIVE" in mapping

    region_action = next(a for a in actions if a.column == "region")
    assert "north" in region_action.params["mapping"]


def test_category_standardizer_skips_sensitive_fields():
    df = pd.DataFrame({
        "email": ["user@example.com", "user@example.com"],
        "description": ["A long descriptive text sentence.", "Another sentence here."],
    })
    profile = {
        "column_profiles": {
            "email": {"semantic_type": "email"},
            "description": {"semantic_type": "descriptive_text"},
        }
    }

    actions = generate_local_category_standardizations(df, profile)
    assert actions == []
