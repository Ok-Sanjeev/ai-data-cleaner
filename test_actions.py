import pandas as pd

from actions import execute_action
from schemas import CleaningAction


df = pd.DataFrame({
    "Age": [25, 30, None, 35],
    "Status": ["active", "ACTIVE", "inactive", "Active"],
    "Join_Date": ["4/2/2021", "7/10/2020", "12/7/2023", "1/5/2022"]
})


actions = [
    CleaningAction(
        action="impute_missing",
        column="Age",
        params={"method": "median"},
        reasoning="Fill missing Age values.",
        confidence=0.95
    ),

    CleaningAction(
        action="standardize_categories",
        column="Status",
        params={
            "mapping": {
                "active": "Active",
                "ACTIVE": "Active",
                "inactive": "Inactive"
            }
        },
        reasoning="Standardize Status values.",
        confidence=0.96
    ),

    CleaningAction(
        action="convert_to_date",
        column="Join_Date",
        params={},
        reasoning="Convert Join_Date to datetime.",
        confidence=0.99
    )
]


for action in actions:
    df = execute_action(df, action)


print(df)
print("\nMissing Age:", df["Age"].isna().sum())
print("Status values:", df["Status"].unique().tolist())
print("Join_Date dtype:", df["Join_Date"].dtype)