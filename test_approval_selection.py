import pytest

from approval_selection import label_for_index, labels_for_count, parse_selection


def test_labels_support_more_than_26_actions():
    assert label_for_index(0) == "A"
    assert label_for_index(25) == "Z"
    assert label_for_index(26) == "AA"
    assert label_for_index(27) == "AB"


def test_parse_comma_separated_selection():
    assert parse_selection(" a, C ", ["A", "B", "C"]) == {"A", "C"}


@pytest.mark.parametrize("value,expected", [("ALL", "ALL"), ("none", "NONE"), ("q", "Q"), ("quit", "Q")])
def test_parse_commands(value, expected):
    assert parse_selection(value, ["A", "B"]) == expected


def test_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unknown selection"):
        parse_selection("A,D", ["A", "B", "C"])


def test_rejects_empty_selection():
    with pytest.raises(ValueError):
        parse_selection("", ["A", "B"])


def test_labels_for_count():
    assert labels_for_count(3) == ["A", "B", "C"]
