from approval_utils import approval_status

def test_all_selected_is_approved():
    assert approval_status(3, 0, 3) == "approved"

def test_some_selected_is_partially_approved():
    assert approval_status(2, 1, 3) == "partially_approved"

def test_none_selected_is_not_approved():
    assert approval_status(0, 3, 3) == "not_approved"

def test_no_recommendations_is_no_actions():
    assert approval_status(0, 0, 0) == "no_actions"
