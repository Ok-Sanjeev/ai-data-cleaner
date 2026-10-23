"""Helpers for selecting validated cleaning recommendations interactively."""
from __future__ import annotations

import string
from typing import Iterable


def label_for_index(index: int) -> str:
    """Return spreadsheet-style labels: A..Z, AA, AB, ... for a zero-based index."""
    if index < 0:
        raise ValueError("index must be non-negative")
    alphabet = string.ascii_uppercase
    number = index + 1
    label = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        label = alphabet[remainder] + label
    return label


def labels_for_count(count: int) -> list[str]:
    if count < 0:
        raise ValueError("count must be non-negative")
    return [label_for_index(i) for i in range(count)]


def parse_selection(user_input: str, labels: Iterable[str]) -> set[str] | str:
    """Accept ABC, A,B,C, A, B, C, ALL, NONE, or Q."""
    valid_labels = {label.upper() for label in labels}
    value = user_input.strip().upper()

    if value in {"ALL", "NONE", "Q", "QUIT"}:
        return "Q" if value == "QUIT" else value

    if not value:
        raise ValueError("Enter letters, ALL, NONE, or Q.")

    # Comma-separated input: A,B,C or A, B, C
    if "," in value:
        requested = [part.strip() for part in value.split(",")]
        if any(not part for part in requested):
            raise ValueError("Invalid selection. Example: A,B or ABC.")
    # Space-separated input: A B C
    elif " " in value:
        requested = value.split()
    # Consecutive letters: ABC, when all labels are single letters
    elif value in valid_labels:
        requested = [value]
    elif all(len(label) == 1 for label in valid_labels):
        requested = list(value)
    else:
        requested = [value]

    selected = set(requested)
    invalid = sorted(selected - valid_labels)

    if invalid:
        raise ValueError(f"Unknown selection: {', '.join(invalid)}")

    return selected