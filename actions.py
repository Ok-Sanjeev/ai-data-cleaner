import re

import pandas as pd


ALLOWED_ACTIONS = {
    "impute_missing",
    "remove_duplicates",
    "convert_to_date",
    "convert_to_numeric",
    "convert_to_boolean",
    "standardize_categories",
    "normalize_text",
    "normalize_whitespace",
    "normalize_column_names",
    "replace_empty_strings",
    "remove_invalid_values",
    "clip_outliers"
}


def impute_missing(df, column, method, value=None):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    if method == "median":
        value = df[column].median()

    elif method == "mean":
        value = df[column].mean()

    elif method == "mode":
        mode = df[column].mode()

        if mode.empty:
            raise ValueError(
                f"Cannot calculate mode for column '{column}'"
            )

        value = mode.iloc[0]

    elif method == "constant":
        if value is None:
            raise ValueError(
                "Constant imputation requires a value"
            )

    else:
        raise ValueError(
            f"Unsupported imputation method: {method}"
        )

    if pd.isna(value):
        raise ValueError(
            f"Cannot impute column '{column}': "
            "replacement value is missing"
        )

    df[column] = df[column].fillna(value)

    return df


def remove_duplicates(df):
    return df.drop_duplicates().copy()


def convert_to_date(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    original_non_null = df[column].notna()

    converted = pd.to_datetime(
        df[column],
        format="mixed",
        errors="coerce"
    )

    invalid = (
        original_non_null
        & converted.isna()
    )

    if invalid.any():
        raise ValueError(
            f"Date conversion failed for "
            f"{int(invalid.sum())} non-null value(s) "
            f"in column '{column}'"
        )

    df[column] = converted

    return df


def convert_to_numeric(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    original_non_null = df[column].notna()

    converted = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    invalid = (
        original_non_null
        & converted.isna()
    )

    if invalid.any():
        raise ValueError(
            f"Numeric conversion failed for "
            f"{int(invalid.sum())} non-null value(s) "
            f"in column '{column}'"
        )

    df[column] = converted

    return df


def convert_to_boolean(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    mapping = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "1": True,
        "0": False
    }

    original_non_null = df[column].notna()

    normalized = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    converted = normalized.map(mapping)

    invalid = (
        original_non_null
        & converted.isna()
    )

    if invalid.any():
        raise ValueError(
            f"Boolean conversion failed for "
            f"{int(invalid.sum())} non-null value(s) "
            f"in column '{column}'"
        )

    df[column] = converted.astype("boolean")

    return df


def standardize_categories(df, column, mapping):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(
            "A non-empty category mapping is required"
        )

    df[column] = df[column].replace(mapping)

    return df


def normalize_text(df, column, case="lower"):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    series = df[column].astype("string")

    if case == "lower":
        series = series.str.lower()

    elif case == "upper":
        series = series.str.upper()

    elif case == "title":
        series = series.str.title()

    elif case != "none":
        raise ValueError(
            f"Unsupported text case: {case}"
        )

    df[column] = series

    return df


def normalize_whitespace(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )

    return df


def normalize_column_names(df):
    df = df.copy()

    new_columns = []

    for column in df.columns:

        normalized = str(column).strip().lower()

        normalized = re.sub(
            r"[^a-z0-9]+",
            "_",
            normalized
        )

        normalized = normalized.strip("_")

        if not normalized:
            raise ValueError(
                f"Column '{column}' cannot be normalized"
            )

        new_columns.append(normalized)

    if len(new_columns) != len(set(new_columns)):
        raise ValueError(
            "Column normalization would create "
            "duplicate column names"
        )

    df.columns = new_columns

    return df


def replace_empty_strings(df, column=None):
    df = df.copy()

    columns = (
        [column]
        if column is not None
        else list(df.columns)
    )

    for current_column in columns:

        if current_column not in df.columns:
            raise ValueError(
                f"Column '{current_column}' does not exist"
            )

        if (
            pd.api.types.is_object_dtype(
                df[current_column]
            )
            or pd.api.types.is_string_dtype(
                df[current_column]
            )
        ):
            df[current_column] = (
                df[current_column]
                .replace(
                    r"^\s*$",
                    pd.NA,
                    regex=True
                )
            )

    return df


def remove_invalid_values(
    df,
    column,
    invalid_values
):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    if (
        not isinstance(invalid_values, list)
        or not invalid_values
    ):
        raise ValueError(
            "invalid_values must be a non-empty list"
        )

    df[column] = df[column].replace(
        invalid_values,
        pd.NA
    )

    return df


def clip_outliers(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    if not pd.api.types.is_numeric_dtype(
        df[column]
    ):
        raise ValueError(
            f"Column '{column}' must be numeric"
        )

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[column] = df[column].clip(
        lower=lower,
        upper=upper
    )

    return df


ACTION_REGISTRY = {
    "impute_missing": impute_missing,
    "remove_duplicates": remove_duplicates,
    "convert_to_date": convert_to_date,
    "convert_to_numeric": convert_to_numeric,
    "convert_to_boolean": convert_to_boolean,
    "standardize_categories": standardize_categories,
    "normalize_text": normalize_text,
    "normalize_whitespace": normalize_whitespace,
    "normalize_column_names": normalize_column_names,
    "replace_empty_strings": replace_empty_strings,
    "remove_invalid_values": remove_invalid_values,
    "clip_outliers": clip_outliers
}


def execute_action(df, action):

    if action.action not in ACTION_REGISTRY:
        raise ValueError(
            f"Unsupported action: {action.action}"
        )

    function = ACTION_REGISTRY[action.action]

    if action.action == "remove_duplicates":
        return function(df)

    if action.action == "normalize_column_names":
        return function(df)

    if action.action == "impute_missing":
        return function(
            df,
            action.column,
            action.params["method"],
            action.params.get("value")
        )

    if action.action in {
        "convert_to_date",
        "convert_to_numeric",
        "convert_to_boolean",
        "normalize_whitespace",
        "clip_outliers"
    }:
        return function(
            df,
            action.column
        )

    if action.action == "standardize_categories":
        return function(
            df,
            action.column,
            action.params["mapping"]
        )

    if action.action == "normalize_text":
        return function(
            df,
            action.column,
            action.params.get(
                "case",
                "lower"
            )
        )

    if action.action == "replace_empty_strings":
        return function(
            df,
            action.column
        )

    if action.action == "remove_invalid_values":
        return function(
            df,
            action.column,
            action.params["invalid_values"]
        )

    raise ValueError(
        f"No execution logic defined for "
        f"{action.action}"
    )