import re

import pandas as pd


ALLOWED_ACTIONS = {
    "impute_missing",
    "remove_duplicates",
    "remove_column",
    "convert_to_date",
    "convert_to_numeric",
    "convert_to_boolean",
    "standardize_categories",
    "normalize_text",
    "normalize_whitespace",
    "normalize_column_names",
    "replace_empty_strings",
    "remove_invalid_values",
    "clip_outliers",
}


# ============================================================
# MISSING VALUE IMPUTATION
# ============================================================

def impute_missing(df, column, method, value=None):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    df = df.copy()

    if method == "median":
        value = df[column].median()

    elif method == "mean":
        value = df[column].mean()

    elif method == "mode":
        mode = df[column].mode(dropna=True)

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


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    return df.drop_duplicates().copy()


# ============================================================
# REMOVE COLUMN
# ============================================================

def remove_column(df, column):
    """Remove a column after the planner determines it is safe."""

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    if len(df.columns) <= 1:
        raise ValueError(
            "Cannot remove the last remaining column"
        )

    return df.drop(columns=[column]).copy()


# ============================================================
# CONVERT TO DATE
# ============================================================

def convert_to_date(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    df = df.copy()

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


# ============================================================
# CONVERT TO NUMERIC
# ============================================================

def convert_to_numeric(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    df = df.copy()

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


# ============================================================
# CONVERT TO BOOLEAN
# ============================================================

def convert_to_boolean(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    df = df.copy()

    mapping = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "1": True,
        "0": False,
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


# ============================================================
# STANDARDIZE CATEGORIES
# ============================================================

def standardize_categories(df, column, mapping):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(
            "A non-empty category mapping is required"
        )

    df = df.copy()

    df[column] = df[column].replace(mapping)

    return df


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(df, column, case="lower"):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    df = df.copy()

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


# ============================================================
# NORMALIZE WHITESPACE
# ============================================================

def normalize_whitespace(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    df = df.copy()

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


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

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


# ============================================================
# REPLACE EMPTY STRINGS
# ============================================================

def replace_empty_strings(df, column=None):
    """
    Replace empty or whitespace-only strings with pandas NA.

    Returns a dataframe with an attrs entry named
    'values_replaced' containing the number of replacements.
    """

    df = df.copy()

    columns = (
        [column]
        if column is not None
        else list(df.columns)
    )

    values_replaced = 0

    for current_column in columns:

        if current_column not in df.columns:
            raise ValueError(
                f"Column '{current_column}' does not exist"
            )

        if (
            pd.api.types.is_object_dtype(df[current_column])
            or pd.api.types.is_string_dtype(df[current_column])
        ):
            empty_mask = (
                df[current_column]
                .astype("string")
                .str.fullmatch(r"\s*", na=False)
            )

            values_replaced += int(empty_mask.sum())

            df.loc[empty_mask, current_column] = pd.NA

    df.attrs["values_replaced"] = values_replaced

    return df


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

def remove_invalid_values(df, column, invalid_values):
    """
    Replace explicitly specified invalid values with pandas NA.

    Returns a dataframe with an attrs entry named
    'values_replaced' containing the number of replacements.
    """

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

    df = df.copy()

    invalid_mask = df[column].isin(invalid_values)

    values_replaced = int(invalid_mask.sum())

    df.loc[invalid_mask, column] = pd.NA

    df.attrs["values_replaced"] = values_replaced

    return df


# ============================================================
# CLIP OUTLIERS
# ============================================================

def clip_outliers(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' must be numeric"
        )

    df = df.copy()

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return df

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[column] = df[column].clip(
        lower=lower,
        upper=upper
    )

    return df


# ============================================================
# ACTION REGISTRY
# ============================================================

ACTION_REGISTRY = {
    "impute_missing": impute_missing,
    "remove_duplicates": remove_duplicates,
    "remove_column": remove_column,
    "convert_to_date": convert_to_date,
    "convert_to_numeric": convert_to_numeric,
    "convert_to_boolean": convert_to_boolean,
    "standardize_categories": standardize_categories,
    "normalize_text": normalize_text,
    "normalize_whitespace": normalize_whitespace,
    "normalize_column_names": normalize_column_names,
    "replace_empty_strings": replace_empty_strings,
    "remove_invalid_values": remove_invalid_values,
    "clip_outliers": clip_outliers,
}


# ============================================================
# ACTION EXECUTION
# ============================================================

def execute_action(df, action):

    if action.action not in ACTION_REGISTRY:
        raise ValueError(
            f"Unsupported action: {action.action}"
        )

    function = ACTION_REGISTRY[action.action]

    if action.action == "remove_duplicates":
        return function(df)

    if action.action == "remove_column":
        return function(df, action.column)

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
        "clip_outliers",
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