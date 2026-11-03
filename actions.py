import re
import unicodedata
from typing import Any, Dict, List, Optional

import pandas as pd


ALLOWED_ACTIONS = {
    "impute_missing",
    "drop_missing_rows",
    "remove_duplicates",
    "remove_column",
    "rename_column",
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
    "clean_phone_number",
    # Text Cleaning
    "trim_whitespace",
    "remove_special_characters",
    "remove_non_printable_characters",
    "normalize_unicode",
    # Data Type & Format Standardization
    "standardize_date_format",
    "extract_date_components",
    "standardize_currency",
    "extract_numeric_values",
    # Numeric Cleaning
    "round_numeric",
    "validate_numeric_ranges",
    "replace_negative_values",
    "convert_percentage",
    # Data Validation
    "validate_email",
    "extract_email",
    "validate_url",
    "remove_empty_rows",
    "detect_invalid_categories",
    # Structural Cleaning
    "split_column",
    "merge_columns",
    "deduplicate_by_columns",
    "remove_high_missing_columns",
    "remove_constant_columns",
    # Value Mapping
    "map_values",
}


# ============================================================
# 1. MISSING VALUE IMPUTATION
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
# 2. DROP MISSING ROWS
# ============================================================

def drop_missing_rows(df, column=None, how="any"):
    """Drop rows with missing values in the specified column or all columns."""
    if column is not None and column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )
    if how not in {"any", "all"}:
        raise ValueError("how must be 'any' or 'all'")

    df = df.copy()
    subset = [column] if column is not None else None
    before_rows = len(df)
    df = df.dropna(subset=subset, how=how).copy()
    df.attrs["rows_dropped"] = before_rows - len(df)
    return df


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    return df.drop_duplicates().copy()


# ============================================================
# 4. REMOVE COLUMN
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
# 5. RENAME COLUMN
# ============================================================

def rename_column(df, column, new_name):
    """Rename a specific column to a new name."""
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )
    if not isinstance(new_name, str) or not new_name.strip():
        raise ValueError("new_name must be a non-empty string")
    if new_name in df.columns and new_name != column:
        raise ValueError(f"Column '{new_name}' already exists")

    df = df.copy()
    return df.rename(columns={column: new_name})


# ============================================================
# 6. CONVERT TO DATE
# ============================================================

def convert_to_date(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    df = df.copy()

    original_non_null = df[column].notna()
    if original_non_null.sum() == 0:
        df[column] = pd.to_datetime(df[column], errors="coerce")
        return df

    if pd.api.types.is_datetime64_any_dtype(df[column]):
        return df

    cleaned = df[column].astype("string").str.strip()

    converted = pd.to_datetime(
        cleaned,
        format="mixed",
        errors="coerce"
    )

    valid_count = converted.notna().sum()
    total_non_null = original_non_null.sum()

    if total_non_null > 0 and (valid_count / total_non_null) < 0.5:
        invalid = (
            original_non_null
            & converted.isna()
        )
        raise ValueError(
            f"Date conversion failed for "
            f"{int(invalid.sum())} non-null value(s) "
            f"in column '{column}'"
        )

    df[column] = converted

    return df


# ============================================================
# 7. CONVERT TO NUMERIC
# ============================================================

def convert_to_numeric(df, column):
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    df = df.copy()

    original_non_null = df[column].notna()
    if original_non_null.sum() == 0:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
        return df

    cleaned = (
        df[column]
        .astype("string")
        .str.strip()
        .str.replace(r"[$€£¥₹%]", "", regex=True)
        .str.replace(r"(?i)\b(usd|eur|gbp|inr|cad|aud)\b", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    # Handle accounting parenthesized negative values: (123.45) -> -123.45
    cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True).str.strip()

    converted = pd.to_numeric(
        cleaned,
        errors="coerce"
    )

    valid_count = converted.notna().sum()
    total_non_null = original_non_null.sum()

    if total_non_null > 0 and (valid_count / total_non_null) < 0.5:
        invalid = (
            original_non_null
            & converted.isna()
        )
        raise ValueError(
            f"Numeric conversion failed for "
            f"{int(invalid.sum())} non-null value(s) "
            f"in column '{column}'"
        )

    df[column] = converted

    return df


# ============================================================
# 8. CONVERT TO BOOLEAN
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
        "t": True,
        "f": False,
        "on": True,
        "off": False,
        "enabled": True,
        "disabled": False,
        "active": True,
        "inactive": False,
        "pass": True,
        "fail": False,
    }

    original_non_null = df[column].notna()
    if original_non_null.sum() == 0:
        df[column] = df[column].astype("boolean")
        return df

    normalized = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    converted = normalized.map(mapping)

    valid_count = converted.notna().sum()
    total_non_null = original_non_null.sum()

    if total_non_null > 0 and (valid_count / total_non_null) < 0.5:
        invalid = (
            original_non_null
            & converted.isna()
        )
        raise ValueError(
            f"Boolean conversion failed for "
            f"{int(invalid.sum())} non-null value(s) "
            f"in column '{column}'"
        )

    df[column] = converted.astype("boolean")

    return df


# ============================================================
# 9. STANDARDIZE CATEGORIES
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
# 10. NORMALIZE TEXT
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
# 11. NORMALIZE WHITESPACE
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
# 12. NORMALIZE COLUMN NAMES
# ============================================================

def normalize_column_names(df):
    df = df.copy()

    new_columns = []
    seen = {}

    for i, column in enumerate(df.columns, start=1):
        col_str = str(column).strip().lower()

        normalized = re.sub(
            r"[^\w]+",
            "_",
            col_str,
            flags=re.UNICODE
        )

        normalized = normalized.strip("_")

        if not normalized:
            normalized = f"column_{i}"

        if normalized in seen:
            seen[normalized] += 1
            unique_name = f"{normalized}_{seen[normalized]}"
        else:
            seen[normalized] = 0
            unique_name = normalized

        new_columns.append(unique_name)

    df.columns = new_columns
    return df


# ============================================================
# 13. REPLACE EMPTY STRINGS
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
# 14. REMOVE INVALID VALUES
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
# 15. CLIP OUTLIERS
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
# 16. CLEAN PHONE NUMBER
# ============================================================

def clean_phone_number(df, column):
    """Standardize phone numbers by stripping non-digit characters while preserving missing values."""
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist"
        )

    df = df.copy()
    series = df[column].astype("string")
    digits = series.str.replace(r"\D", "", regex=True)
    df[column] = digits.where(series.notna() & (series != ""), pd.NA)
    return df


# ============================================================
# 17. TRIM WHITESPACE (Targeted edge trim only)
# ============================================================

def trim_whitespace(df, column):
    """Remove leading and trailing whitespace from string values without altering internal spaces."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    df = df.copy()
    df[column] = df[column].astype("string").str.strip()
    return df


# ============================================================
# 18. REMOVE SPECIAL CHARACTERS
# ============================================================

def remove_special_characters(df, column, pattern=r"[^\w\s]", replacement=""):
    """Remove explicitly configured unwanted characters (default: non-alphanumeric and non-space characters)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    df = df.copy()
    df[column] = df[column].astype("string").str.replace(pattern, replacement, regex=True)
    return df


# ============================================================
# 19. REMOVE NON-PRINTABLE CHARACTERS
# ============================================================

def remove_non_printable_characters(df, column):
    """Remove non-printable ASCII and control characters including zero-width spaces."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    df = df.copy()
    pattern = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200d\ufeff]")

    def _clean(val):
        if pd.isna(val):
            return val
        return pattern.sub("", str(val))

    df[column] = df[column].map(_clean).astype("string")
    return df


# ============================================================
# 20. NORMALIZE UNICODE
# ============================================================

def normalize_unicode(df, column, form="NFKD"):
    """Normalize Unicode text representations (e.g. convert accents, ligatures, full-width characters)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"Unsupported Unicode normalization form: {form}")

    df = df.copy()

    def _norm(val):
        if pd.isna(val):
            return val
        return unicodedata.normalize(form, str(val))

    df[column] = df[column].map(_norm).astype("string")
    return df


# ============================================================
# 21. STANDARDIZE DATE FORMAT
# ============================================================

def standardize_date_format(df, column, target_format="%Y-%m-%d"):
    """Convert valid date values into a standardized string format (e.g. ISO 8601 %Y-%m-%d)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    df = df.copy()
    dt = pd.to_datetime(df[column], format="mixed", errors="coerce")
    df[column] = dt.dt.strftime(target_format).where(dt.notna(), pd.NA).astype("string")
    return df


# ============================================================
# 22. EXTRACT DATE COMPONENTS
# ============================================================

def extract_date_components(df, column, components=None):
    """Extract year, month, and day into separate integer columns."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")
    if components is None:
        components = ["year", "month", "day"]

    df = df.copy()
    dt = pd.to_datetime(df[column], format="mixed", errors="coerce")

    for comp in components:
        comp_lower = comp.lower().strip()
        target_col = f"{column}_{comp_lower}"
        if comp_lower == "year":
            df[target_col] = dt.dt.year.astype("Int64")
        elif comp_lower == "month":
            df[target_col] = dt.dt.month.astype("Int64")
        elif comp_lower == "day":
            df[target_col] = dt.dt.day.astype("Int64")
        elif comp_lower == "day_name":
            df[target_col] = dt.dt.day_name().astype("string")
        elif comp_lower == "hour":
            df[target_col] = dt.dt.hour.astype("Int64")

    return df


# ============================================================
# 23. STANDARDIZE CURRENCY
# ============================================================

def standardize_currency(df, column):
    """Strip currency symbols, commas, and currency labels and parse into clean numeric values."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    df = df.copy()
    cleaned = (
        df[column]
        .astype("string")
        .str.strip()
        .str.replace(r"[$€£¥₹\s]", "", regex=True)
        .str.replace(r"(?i)\b(usd|eur|gbp|inr|cad|aud|jpy)\b", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        .str.strip()
    )
    df[column] = pd.to_numeric(cleaned, errors="coerce")
    return df


# ============================================================
# 24. EXTRACT NUMERIC VALUES
# ============================================================

def extract_numeric_values(df, column):
    """Extract embedded numeric measurements or values from unstructured text (e.g. '75.5 kg' -> 75.5)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    df = df.copy()
    extracted = df[column].astype("string").str.extract(r"(-?\d+(?:\.\d+)?)")[0]
    df[column] = pd.to_numeric(extracted, errors="coerce")
    return df


# ============================================================
# 25. ROUND NUMERIC
# ============================================================

def round_numeric(df, column, decimals=2):
    """Round numeric column to a specified number of decimal places."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")

    df = df.copy()
    df[column] = df[column].round(int(decimals))
    return df


# ============================================================
# 26. VALIDATE NUMERIC RANGES
# ============================================================

def validate_numeric_ranges(df, column, min_val=None, max_val=None, action="flag"):
    """
    Validate numeric values within [min_val, max_val].
    action='flag' creates a {column}_out_of_range boolean column.
    action='replace_with_na' sets invalid values to NA.
    action='clamp' clips values to [min_val, max_val].
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")

    df = df.copy()
    mask = pd.Series(False, index=df.index)
    if min_val is not None:
        mask = mask | (df[column] < float(min_val))
    if max_val is not None:
        mask = mask | (df[column] > float(max_val))

    if action == "replace_with_na":
        df.loc[mask, column] = pd.NA
        df.attrs["values_replaced"] = int(mask.sum())
    elif action == "clamp":
        df[column] = df[column].clip(lower=min_val, upper=max_val)
        df.attrs["values_replaced"] = int(mask.sum())
    elif action == "flag":
        df[f"{column}_out_of_range"] = mask
    else:
        raise ValueError(f"Unsupported action for validate_numeric_ranges: {action}")

    return df


# ============================================================
# 27. REPLACE NEGATIVE VALUES
# ============================================================

def replace_negative_values(df, column, method="nan"):
    """
    Replace impossible negative values when domain rules prohibit them.
    method='nan' sets negative values to NA.
    method='zero' replaces negative values with 0.
    method='abs' takes the absolute value.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")

    df = df.copy()
    neg_mask = df[column] < 0
    df.attrs["values_replaced"] = int(neg_mask.sum())

    if method == "nan":
        df.loc[neg_mask, column] = pd.NA
    elif method == "zero":
        df.loc[neg_mask, column] = 0
    elif method == "abs":
        df[column] = df[column].abs()
    else:
        raise ValueError(f"Unsupported method: {method}")

    return df


# ============================================================
# 28. CONVERT PERCENTAGE
# ============================================================

def convert_percentage(df, column, to_decimal=True):
    """Normalize percentage strings (e.g. '45%', '0.45') to standard decimal (0.45) or percentage float (45.0)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    df = df.copy()
    s = df[column].astype("string").str.strip()
    has_pct = s.str.endswith("%")
    cleaned = s.str.replace("%", "", regex=False).str.strip()
    vals = pd.to_numeric(cleaned, errors="coerce")

    if to_decimal:
        res = vals.where(~has_pct, vals / 100.0)
        # If all non-null values are > 1, assume 0-100 scale and divide
        non_null_res = res.dropna()
        if len(non_null_res) > 0 and (non_null_res > 1.0).all():
            res = res / 100.0
        df[column] = res
    else:
        res = vals.where(has_pct, vals * 100.0)
        df[column] = res

    return df


# ============================================================
# 29. VALIDATE EMAIL
# ============================================================

def validate_email(df, column, action="flag"):
    """
    Validate email address formats using standard RFC pattern.
    action='flag' creates a {column}_is_valid_email boolean column.
    action='replace_with_na' replaces malformed emails with NA.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    df = df.copy()
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    s = df[column].astype("string").str.strip()
    is_valid = s.str.match(pattern, na=False)
    is_non_null = s.notna() & (s != "")
    invalid_mask = is_non_null & (~is_valid)

    if action == "replace_with_na":
        df.loc[invalid_mask, column] = pd.NA
        df.attrs["values_replaced"] = int(invalid_mask.sum())
    elif action == "flag":
        df[f"{column}_is_valid_email"] = is_valid.where(is_non_null, pd.NA)
    else:
        raise ValueError(f"Unsupported action: {action}")

    return df


# ============================================================
# 30. EXTRACT EMAIL
# ============================================================

def extract_email(df, column, target_column=None):
    """Extract email addresses from unstructured or messy text into target column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    df = df.copy()
    target = target_column if target_column else column
    extracted = df[column].astype("string").str.extract(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")[0]
    df[target] = extracted
    return df


# ============================================================
# 31. VALIDATE URL
# ============================================================

def validate_url(df, column, action="flag"):
    """
    Validate URL structure (http/https).
    action='flag' creates a {column}_is_valid_url column.
    action='replace_with_na' replaces malformed URLs with NA.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    df = df.copy()
    pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    s = df[column].astype("string").str.strip()
    is_valid = s.str.match(pattern, na=False)
    is_non_null = s.notna() & (s != "")
    invalid_mask = is_non_null & (~is_valid)

    if action == "replace_with_na":
        df.loc[invalid_mask, column] = pd.NA
        df.attrs["values_replaced"] = int(invalid_mask.sum())
    elif action == "flag":
        df[f"{column}_is_valid_url"] = is_valid.where(is_non_null, pd.NA)
    else:
        raise ValueError(f"Unsupported action: {action}")

    return df


# ============================================================
# 32. REMOVE EMPTY ROWS
# ============================================================

def remove_empty_rows(df, columns=None):
    """Remove rows where all checked columns are completely blank or whitespace-only."""
    df = df.copy()
    before_rows = len(df)
    cols_to_check = columns if columns is not None else list(df.columns)

    for col in cols_to_check:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist")

    is_empty_cell = pd.DataFrame(index=df.index)
    for c in cols_to_check:
        s = df[c]
        is_empty_cell[c] = s.isna() | (s.astype("string").str.strip() == "")

    all_empty = is_empty_cell.all(axis=1)
    df = df[~all_empty].copy()
    df.attrs["rows_dropped"] = before_rows - len(df)
    return df


# ============================================================
# 33. DETECT INVALID CATEGORIES
# ============================================================

def detect_invalid_categories(df, column, allowed_categories, action="flag"):
    """
    Detect values outside an explicitly configured allowed category set.
    action='flag' creates a {column}_invalid_category boolean column.
    action='replace_with_na' replaces invalid values with NA.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")
    if not isinstance(allowed_categories, list) or not allowed_categories:
        raise ValueError("allowed_categories must be a non-empty list")

    df = df.copy()
    s = df[column]
    is_invalid = s.notna() & (~s.isin(allowed_categories))

    if action == "replace_with_na":
        df.loc[is_invalid, column] = pd.NA
        df.attrs["values_replaced"] = int(is_invalid.sum())
    elif action == "flag":
        df[f"{column}_invalid_category"] = is_invalid
    else:
        raise ValueError(f"Unsupported action: {action}")

    return df


# ============================================================
# 34. SPLIT COLUMN
# ============================================================

def split_column(df, column, delimiter=",", new_columns=None):
    """Split a column into multiple new columns using a delimiter."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")
    if not delimiter:
        raise ValueError("delimiter cannot be empty")

    df = df.copy()
    splits = df[column].astype("string").str.split(delimiter, expand=True)
    num_splits = splits.shape[1]

    if new_columns is None or len(new_columns) != num_splits:
        new_columns = [f"{column}_part_{i+1}" for i in range(num_splits)]

    for i, col_name in enumerate(new_columns):
        if col_name in df.columns and col_name != column:
            raise ValueError(f"Destination column '{col_name}' already exists")
        df[col_name] = splits[i].str.strip()

    return df


# ============================================================
# 35. MERGE COLUMNS
# ============================================================

def merge_columns(df, columns, separator=" ", new_column="merged"):
    """Combine multiple columns with a separator into a new column."""
    if not isinstance(columns, list) or len(columns) < 2:
        raise ValueError("merge_columns requires at least two columns")
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist")
    if not new_column or not str(new_column).strip():
        raise ValueError("new_column must be a non-empty string")

    df = df.copy()
    series_list = [df[c].fillna("").astype(str) for c in columns]
    result = series_list[0]
    for s in series_list[1:]:
        result = result + separator + s

    df[new_column] = result.str.strip(separator).astype("string")
    return df


# ============================================================
# 36. DEDUPLICATE BY COLUMNS
# ============================================================

def deduplicate_by_columns(df, subset, keep="first"):
    """Remove duplicate rows considering only a key subset of columns."""
    if not isinstance(subset, list) or not subset:
        raise ValueError("subset must be a non-empty list of columns")
    for col in subset:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist")
    if keep not in {"first", "last", False}:
        raise ValueError("keep must be 'first', 'last', or False")

    df = df.copy()
    before_rows = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep).copy()
    df.attrs["rows_dropped"] = before_rows - len(df)
    return df


# ============================================================
# 37. REMOVE HIGH MISSING COLUMNS
# ============================================================

def remove_high_missing_columns(df, threshold=0.8):
    """Remove columns whose missing value percentage exceeds the threshold (e.g. 0.8 for 80%)."""
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold must be between 0.0 and 1.0")

    df = df.copy()
    total_rows = len(df)
    if total_rows == 0:
        return df

    cols_to_drop = [c for c in df.columns if (df[c].isna().sum() / total_rows) >= threshold]
    if len(cols_to_drop) == len(df.columns):
        # Prevent removing all columns
        cols_to_drop = cols_to_drop[:-1]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop).copy()

    return df


# ============================================================
# 38. REMOVE CONSTANT COLUMNS
# ============================================================

def remove_constant_columns(df, columns=None):
    """Remove columns that have only a single non-null uniform value."""
    df = df.copy()
    cols_to_check = columns if columns is not None else list(df.columns)

    for col in cols_to_check:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist")

    cols_to_drop = []
    for col in cols_to_check:
        if df[col].nunique(dropna=True) <= 1:
            cols_to_drop.append(col)

    if len(cols_to_drop) == len(df.columns):
        # Prevent removing all columns
        cols_to_drop = cols_to_drop[:-1]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop).copy()

    return df


# ============================================================
# 39. MAP VALUES
# ============================================================

def map_values(df, column, mapping):
    """Map specific values in a column using a standardized dictionary mapping."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("mapping must be a non-empty dictionary")

    df = df.copy()
    df[column] = df[column].replace(mapping)
    return df


# ============================================================
# ACTION METADATA & RISK REGISTRY
# ============================================================

ACTION_METADATA: Dict[str, Dict[str, Any]] = {
    # Text Cleaning (7 actions)
    "normalize_text": {
        "category": "Text Cleaning",
        "risk": "green",
        "label": "Normalize Text Case",
        "description": "Standardize casing across text entries (lower, upper, title)."
    },
    "normalize_whitespace": {
        "category": "Text Cleaning",
        "risk": "green",
        "label": "Normalize Whitespace",
        "description": "Trim outer whitespace and collapse repeated internal spaces."
    },
    "trim_whitespace": {
        "category": "Text Cleaning",
        "risk": "green",
        "label": "Trim Whitespace",
        "description": "Remove leading and trailing whitespace from string values."
    },
    "remove_special_characters": {
        "category": "Text Cleaning",
        "risk": "green",
        "label": "Remove Special Characters",
        "description": "Strip unwanted punctuation and special characters."
    },
    "remove_non_printable_characters": {
        "category": "Text Cleaning",
        "risk": "green",
        "label": "Remove Non-Printable Chars",
        "description": "Clean control characters, zero-width spaces, and invisible glyphs."
    },
    "normalize_unicode": {
        "category": "Text Cleaning",
        "risk": "green",
        "label": "Normalize Unicode",
        "description": "Standardize Unicode representations and diacritics."
    },
    "replace_empty_strings": {
        "category": "Text Cleaning",
        "risk": "yellow",
        "label": "Convert Empty Strings to Null",
        "description": "Convert blank or whitespace-only text into standard null indicators."
    },

    # Data Type & Format Standardization (8 actions)
    "convert_to_date": {
        "category": "Data Type & Formatting",
        "risk": "yellow",
        "label": "Convert to Date",
        "description": "Parse string representations into datetime objects."
    },
    "standardize_date_format": {
        "category": "Data Type & Formatting",
        "risk": "green",
        "label": "Standardize Date Format",
        "description": "Format dates to ISO 8601 (%Y-%m-%d) or a specified pattern."
    },
    "extract_date_components": {
        "category": "Data Type & Formatting",
        "risk": "green",
        "label": "Extract Date Components",
        "description": "Deconstruct dates into year, month, and day integer columns."
    },
    "convert_to_numeric": {
        "category": "Data Type & Formatting",
        "risk": "yellow",
        "label": "Convert to Numeric",
        "description": "Parse numbers formatted with symbols or commas into float/int."
    },
    "standardize_currency": {
        "category": "Data Type & Formatting",
        "risk": "green",
        "label": "Standardize Currency",
        "description": "Strip currency codes and symbols into clean numeric amounts."
    },
    "extract_numeric_values": {
        "category": "Data Type & Formatting",
        "risk": "green",
        "label": "Extract Numeric Values",
        "description": "Extract embedded numbers from descriptive text fields."
    },
    "convert_to_boolean": {
        "category": "Data Type & Formatting",
        "risk": "yellow",
        "label": "Convert to Boolean",
        "description": "Map Yes/No, 1/0, and True/False strings to native boolean flags."
    },
    "clean_phone_number": {
        "category": "Data Type & Formatting",
        "risk": "yellow",
        "label": "Clean Phone Numbers",
        "description": "Standardize phone numbers by stripping non-digit formatting."
    },

    # Numeric Cleaning (5 actions)
    "round_numeric": {
        "category": "Numeric Cleaning",
        "risk": "green",
        "label": "Round Numeric",
        "description": "Round numbers to a specified decimal precision."
    },
    "validate_numeric_ranges": {
        "category": "Numeric Cleaning",
        "risk": "yellow",
        "label": "Validate Numeric Ranges",
        "description": "Identify or constrain values outside allowed [min, max] bounds."
    },
    "replace_negative_values": {
        "category": "Numeric Cleaning",
        "risk": "yellow",
        "label": "Replace Negative Values",
        "description": "Correct domain-prohibited negative values with null, zero, or abs."
    },
    "convert_percentage": {
        "category": "Numeric Cleaning",
        "risk": "green",
        "label": "Convert Percentage",
        "description": "Normalize percentage strings to decimals (0.45) or percentages (45.0)."
    },
    "clip_outliers": {
        "category": "Numeric Cleaning",
        "risk": "red",
        "label": "Clip Outliers",
        "description": "Constrain statistical outliers to interquartile range (IQR) boundaries."
    },

    # Missing Value Handling (4 actions)
    "impute_missing": {
        "category": "Missing Values",
        "risk": "yellow",
        "label": "Impute Missing Values",
        "description": "Fill null values with median, mean, mode, or a constant."
    },
    "drop_missing_rows": {
        "category": "Missing Values",
        "risk": "red",
        "label": "Drop Missing Rows",
        "description": "Permanently drop records that contain missing values."
    },
    "remove_empty_rows": {
        "category": "Missing Values",
        "risk": "green",
        "label": "Remove Blank Rows",
        "description": "Drop completely blank rows where all columns are empty."
    },
    "remove_high_missing_columns": {
        "category": "Missing Values",
        "risk": "red",
        "label": "Remove High-Missing Columns",
        "description": "Drop columns where missing value percentage exceeds the threshold."
    },

    # Data Validation (4 actions)
    "validate_email": {
        "category": "Data Validation",
        "risk": "yellow",
        "label": "Validate Email Format",
        "description": "Validate email addresses against RFC standards."
    },
    "extract_email": {
        "category": "Data Validation",
        "risk": "green",
        "label": "Extract Email",
        "description": "Extract valid email addresses from messy text fields."
    },
    "validate_url": {
        "category": "Data Validation",
        "risk": "yellow",
        "label": "Validate URL Format",
        "description": "Validate web addresses for valid URI structure."
    },
    "detect_invalid_categories": {
        "category": "Data Validation",
        "risk": "yellow",
        "label": "Detect Invalid Categories",
        "description": "Flag or replace categorical values outside the approved list."
    },

    # Structural Cleaning (6 actions)
    "remove_column": {
        "category": "Structural Cleaning",
        "risk": "red",
        "label": "Remove Column",
        "description": "Permanently remove a column from the dataset."
    },
    "remove_constant_columns": {
        "category": "Structural Cleaning",
        "risk": "red",
        "label": "Remove Constant Columns",
        "description": "Remove columns having zero variance or a single uniform value."
    },
    "rename_column": {
        "category": "Structural Cleaning",
        "risk": "yellow",
        "label": "Rename Column",
        "description": "Rename a column to a cleaner or more descriptive header."
    },
    "normalize_column_names": {
        "category": "Structural Cleaning",
        "risk": "green",
        "label": "Normalize Column Names",
        "description": "Convert all column headers to consistent snake_case format."
    },
    "split_column": {
        "category": "Structural Cleaning",
        "risk": "yellow",
        "label": "Split Column",
        "description": "Split a composite column into multiple new columns by delimiter."
    },
    "merge_columns": {
        "category": "Structural Cleaning",
        "risk": "yellow",
        "label": "Merge Columns",
        "description": "Combine multiple columns into one using a separator."
    },

    # Value Mapping (3 actions)
    "standardize_categories": {
        "category": "Value Mapping",
        "risk": "yellow",
        "label": "Standardize Categories",
        "description": "Consolidate casing variations and synonyms into standard labels."
    },
    "remove_invalid_values": {
        "category": "Value Mapping",
        "risk": "red",
        "label": "Remove Invalid Values",
        "description": "Replace explicitly identified invalid entries with null."
    },
    "map_values": {
        "category": "Value Mapping",
        "risk": "yellow",
        "label": "Map Values",
        "description": "Apply an explicit replacement dictionary to clean column entries."
    },

    # Deduplication (2 actions)
    "remove_duplicates": {
        "category": "Deduplication",
        "risk": "green",
        "label": "Remove Exact Duplicates",
        "description": "Drop identical records across all columns."
    },
    "deduplicate_by_columns": {
        "category": "Deduplication",
        "risk": "yellow",
        "label": "Deduplicate by Key Columns",
        "description": "Drop duplicate records based on a critical subset of columns."
    },
}


# ============================================================
# ACTION REGISTRY
# ============================================================

ACTION_REGISTRY = {
    # Original 16 actions
    "impute_missing": impute_missing,
    "drop_missing_rows": drop_missing_rows,
    "remove_duplicates": remove_duplicates,
    "remove_column": remove_column,
    "rename_column": rename_column,
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
    "clean_phone_number": clean_phone_number,
    # Newly expanded actions
    "trim_whitespace": trim_whitespace,
    "remove_special_characters": remove_special_characters,
    "remove_non_printable_characters": remove_non_printable_characters,
    "normalize_unicode": normalize_unicode,
    "standardize_date_format": standardize_date_format,
    "extract_date_components": extract_date_components,
    "standardize_currency": standardize_currency,
    "extract_numeric_values": extract_numeric_values,
    "round_numeric": round_numeric,
    "validate_numeric_ranges": validate_numeric_ranges,
    "replace_negative_values": replace_negative_values,
    "convert_percentage": convert_percentage,
    "validate_email": validate_email,
    "extract_email": extract_email,
    "validate_url": validate_url,
    "remove_empty_rows": remove_empty_rows,
    "detect_invalid_categories": detect_invalid_categories,
    "split_column": split_column,
    "merge_columns": merge_columns,
    "deduplicate_by_columns": deduplicate_by_columns,
    "remove_high_missing_columns": remove_high_missing_columns,
    "remove_constant_columns": remove_constant_columns,
    "map_values": map_values,
}


# ============================================================
# ACTION EXECUTION DISPATCHER
# ============================================================

def execute_action(df, action):
    if action.action not in ACTION_REGISTRY:
        raise ValueError(
            f"Unsupported action: {action.action}"
        )

    function = ACTION_REGISTRY[action.action]

    # Deduplication & Column Name Actions
    if action.action == "remove_duplicates":
        return function(df)

    if action.action == "normalize_column_names":
        return function(df)

    if action.action == "remove_empty_rows":
        return function(df, action.params.get("columns"))

    if action.action == "remove_high_missing_columns":
        return function(df, float(action.params.get("threshold", 0.8)))

    if action.action == "remove_constant_columns":
        return function(df, action.params.get("columns"))

    if action.action == "deduplicate_by_columns":
        return function(
            df,
            action.params.get("subset", []),
            action.params.get("keep", "first")
        )

    if action.action == "merge_columns":
        return function(
            df,
            action.params.get("columns", []),
            action.params.get("separator", " "),
            action.params.get("new_column", "merged")
        )

    # Column-level removal and renaming
    if action.action == "remove_column":
        return function(df, action.column)

    if action.action == "rename_column":
        return function(df, action.column, action.params["new_name"])

    if action.action == "drop_missing_rows":
        return function(
            df,
            action.column,
            action.params.get("how", "any")
        )

    # Imputation
    if action.action == "impute_missing":
        return function(
            df,
            action.column,
            action.params["method"],
            action.params.get("value")
        )

    # Standard single-column transformers
    if action.action in {
        "convert_to_date",
        "convert_to_numeric",
        "convert_to_boolean",
        "normalize_whitespace",
        "clip_outliers",
        "clean_phone_number",
        "trim_whitespace",
        "remove_non_printable_characters",
        "standardize_currency",
        "extract_numeric_values",
        "replace_empty_strings",
    }:
        return function(df, action.column)

    # Text & Unicode Actions
    if action.action == "normalize_text":
        return function(
            df,
            action.column,
            action.params.get("case", "lower")
        )

    if action.action == "remove_special_characters":
        return function(
            df,
            action.column,
            action.params.get("pattern", r"[^\w\s]"),
            action.params.get("replacement", "")
        )

    if action.action == "normalize_unicode":
        return function(
            df,
            action.column,
            action.params.get("form", "NFKD")
        )

    # Date formatting & component extraction
    if action.action == "standardize_date_format":
        return function(
            df,
            action.column,
            action.params.get("target_format", "%Y-%m-%d")
        )

    if action.action == "extract_date_components":
        return function(
            df,
            action.column,
            action.params.get("components", ["year", "month", "day"])
        )

    # Numeric Actions
    if action.action == "round_numeric":
        return function(
            df,
            action.column,
            int(action.params.get("decimals", 2))
        )

    if action.action == "validate_numeric_ranges":
        return function(
            df,
            action.column,
            action.params.get("min_val"),
            action.params.get("max_val"),
            action.params.get("action", "flag")
        )

    if action.action == "replace_negative_values":
        return function(
            df,
            action.column,
            action.params.get("method", "nan")
        )

    if action.action == "convert_percentage":
        return function(
            df,
            action.column,
            action.params.get("to_decimal", True)
        )

    # Validation & Extraction Actions
    if action.action == "validate_email":
        return function(
            df,
            action.column,
            action.params.get("action", "flag")
        )

    if action.action == "extract_email":
        return function(
            df,
            action.column,
            action.params.get("target_column")
        )

    if action.action == "validate_url":
        return function(
            df,
            action.column,
            action.params.get("action", "flag")
        )

    if action.action == "detect_invalid_categories":
        return function(
            df,
            action.column,
            action.params.get("allowed_categories", []),
            action.params.get("action", "flag")
        )

    # Value Mapping Actions
    if action.action == "standardize_categories":
        return function(
            df,
            action.column,
            action.params["mapping"]
        )

    if action.action == "map_values":
        return function(
            df,
            action.column,
            action.params.get("mapping", {})
        )

    if action.action == "remove_invalid_values":
        return function(
            df,
            action.column,
            action.params["invalid_values"]
        )

    if action.action == "split_column":
        return function(
            df,
            action.column,
            action.params.get("delimiter", ","),
            action.params.get("new_columns")
        )

    raise ValueError(
        f"No execution logic defined for {action.action}"
    )
