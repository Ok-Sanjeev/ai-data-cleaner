"""Pure helpers for recording per-action approval decisions."""

def approval_status(selected_count: int, rejected_count: int, proposed_count: int) -> str:
    if proposed_count == 0:
        return "no_actions"
    if selected_count == 0:
        return "not_approved"
    if rejected_count:
        return "partially_approved"
    return "approved"
