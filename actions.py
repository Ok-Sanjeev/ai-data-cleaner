import pandas as pd


ALLOWED_ACTIONS = {
    "impute_missing",
    "remove_duplicates",
    "convert_to_date",
    "standardize_categories"
}


def remove_duplicates(df):
    return df.drop_duplicates().copy()


def impute_missing(df, column, method):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    if method == "median":
        df[column] = df[column].fillna(df[column].median())

    elif method == "mean":
        df[column] = df[column].fillna(df[column].mean())

    elif method == "mode":
        mode = df[column].mode()

        if mode.empty:
            raise ValueError(
                f"Cannot calculate mode for column '{column}'"
            )

        df[column] = df[column].fillna(mode.iloc[0])

    else:
        raise ValueError(
            f"Unsupported imputation method: {method}"
        )

    return df

def convert_to_date(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    original_non_null = df[column].notna()

    converted = pd.to_datetime(
        df[column],
        format="mixed",
        errors="coerce"
    )

    invalid_values = original_non_null & converted.isna()

    if invalid_values.any():
        invalid_count = invalid_values.sum()

        raise ValueError(
            f"Date conversion failed for {invalid_count} "
            f"non-null value(s) in column '{column}'"
        )

    df[column] = converted

    return df


def standardize_categories(df, column, mapping):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    if not mapping:
        raise ValueError(
            f"A non-empty mapping is required to standardize "
            f"categories in column '{column}'"
        )

    if not isinstance(mapping, dict):
        raise ValueError(
            f"Mapping for column '{column}' must be a dictionary"
        )

    df[column] = df[column].replace(mapping)

    return df


ACTION_REGISTRY = {
    "remove_duplicates": remove_duplicates,
    "impute_missing": impute_missing,
    "convert_to_date": convert_to_date,
    "standardize_categories": standardize_categories
}


def execute_action(df, action):
    if action.action not in ACTION_REGISTRY:
        raise ValueError(
            f"Unsupported action: {action.action}"
        )

    function = ACTION_REGISTRY[action.action]

    if action.action == "remove_duplicates":
        return function(df)

    if action.action == "impute_missing":
        method = action.params.get("method")

        if not method:
            raise ValueError(
                "A valid imputation method is required"
            )

        return function(
            df,
            action.column,
            method
        )

    if action.action == "convert_to_date":
        return function(
            df,
            action.column
        )

    if action.action == "standardize_categories":
        mapping = action.params.get("mapping")

        return function(
            df,
            action.column,
            mapping
        )

    raise ValueError(
        f"No execution logic defined for: {action.action}"
    )