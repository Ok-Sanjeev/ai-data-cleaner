import pandas as pd

from profiler import detect_semantic_type


def test_semantic_type_detected_from_column_name_without_sharing_values():
    assert detect_semantic_type("customer_email", pd.Series(["a@example.com", "b@example.com"]))["semantic_type"] == "email"
    assert detect_semantic_type("phone_number", pd.Series(["9876543210", "9123456789"]))["semantic_type"] == "phone_number"
    assert detect_semantic_type("date_of_birth", pd.Series(["2002-05-12", "2001-01-01"]))["semantic_type"] == "date_of_birth"
    assert detect_semantic_type("salary", pd.Series([45000, 50000]))["semantic_type"] == "salary_or_income"


def test_semantic_type_detected_from_values_when_header_is_generic():
    result = detect_semantic_type("contact", pd.Series(["9876543210", "9123456789", "9988776655"]))
    assert result["semantic_type"] == "phone_number"
    assert result["sensitive_field"] is True


def test_long_sentence_column_is_classified_as_descriptive_text():
    import pandas as pd
    from profiler import detect_semantic_type
    series = pd.Series([
        "This is a long descriptive sentence intended to explain a customer experience and provide context for analysis.",
        "Another long descriptive sentence intended to explain a product feature and provide context for analysis.",
    ])
    result = detect_semantic_type("description", series)
    assert result["semantic_type"] == "descriptive_text"
