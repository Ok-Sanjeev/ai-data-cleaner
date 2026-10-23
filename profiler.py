import csv
import json
import re

import pandas as pd


FILE_PATH = "data/Messy_Employee_dataset.csv"


BOOLEAN_VALUES = {
    "true",
    "false",
    "yes",
    "no",
    "y",
    "n",
    "1",
    "0"
}


def detect_semantic_hints(series):
    hints = []

    non_null = series.dropna()

    # --------------------------------------------------------
    # Empty / whitespace strings
    # --------------------------------------------------------

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        string_values = series.astype("string")

        if string_values.str.strip().eq("").any():
            hints.append("empty_strings")

        whitespace_issue = (
            string_values.notna()
            & (
                string_values
                != string_values.str.strip()
            )
        )

        repeated_whitespace = (
            string_values.notna()
            & string_values.str.contains(
                r"\s{2,}",
                regex=True
            )
        )

        if (
            whitespace_issue.any()
            or repeated_whitespace.any()
        ):
            hints.append("whitespace_issues")

    # --------------------------------------------------------
    # Possible date
    # --------------------------------------------------------

    if (
        len(non_null) > 0
        and (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        )
    ):
        parsed = pd.to_datetime(
            non_null,
            format="mixed",
            errors="coerce"
        )

        if (
            parsed.notna().mean() >= 0.8
            and not pd.api.types.is_datetime64_any_dtype(
                series
            )
        ):
            hints.append("possible_date")

    # --------------------------------------------------------
    # Possible email
    # --------------------------------------------------------

    if len(non_null) > 0 and (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        email_pattern = non_null.astype("string").str.contains(
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            regex=True
        )

        if email_pattern.mean() >= 0.8:
            hints.append("possible_email")

    # --------------------------------------------------------
    # Possible boolean
    # --------------------------------------------------------

    if len(non_null) > 0 and (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        normalized = (
            non_null.astype("string")
            .str.strip()
            .str.lower()
        )

        if (
            normalized.isin(BOOLEAN_VALUES).mean()
            >= 0.8
        ):
            hints.append("possible_boolean")

    # --------------------------------------------------------
    # Numeric-looking strings
    # --------------------------------------------------------

    if len(non_null) > 0 and (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        numeric = pd.to_numeric(
            non_null.astype("string").str.replace(
                ",",
                "",
                regex=False
            ),
            errors="coerce"
        )

        if numeric.notna().mean() >= 0.8:
            hints.append("possible_numeric")

    # --------------------------------------------------------
    # Identifier
    # --------------------------------------------------------

    if (
        series.nunique() == len(series)
        and series.notna().all()
    ):
        hints.append("possible_identifier")

    # --------------------------------------------------------
    # Negative values in identifier-like columns
    # --------------------------------------------------------

    if (
        "possible_identifier" in hints
        and pd.api.types.is_numeric_dtype(series)
        and (series.dropna() < 0).any()
    ):
        hints.append("invalid_negative_values")

    # --------------------------------------------------------
    # Mixed category casing
    # --------------------------------------------------------

    if len(non_null) > 0 and (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        values = non_null.astype("string")

        normalized = values.str.strip().str.lower()

        if normalized.nunique() < values.nunique():
            hints.append("inconsistent_categories")

        casing_variants = (
            values.str.lower().nunique()
            < values.nunique()
        )

        if casing_variants:
            hints.append("inconsistent_text_case")

    return list(dict.fromkeys(hints))



def detect_semantic_type(column_name, series):
    """Classify a field locally from its header and value patterns.

    Only the classification is eligible for the Gemini profile; the values used
    to infer it are never copied into the AI-safe payload.
    """
    name = re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower()).strip("_")
    compact = name.replace("_", "")
    values = series.dropna().astype("string").str.strip()
    nonempty = values[values.ne("")]

    rules = [
        (("email", "e_mail", "email_address"), "email", True),
        (("phone", "phone_number", "mobile", "telephone", "contact_number", "tel"), "phone_number", True),
        (("dob", "date_of_birth", "birth_date", "birthdate"), "date_of_birth", True),
        (("ssn", "social_security", "national_id", "passport", "tax_id"), "government_or_identity_id", True),
        (("salary", "wage", "income", "compensation", "pay_amount"), "salary_or_income", True),
        (("address", "street_address", "postal_address"), "address", True),
        (("first_name", "last_name", "full_name", "customer_name", "employee_name"), "person_name", True),
        (("date", "_at", "_on", "timestamp", "created_time", "updated_time"), "date_or_timestamp", False),
    ]
    for needles, semantic_type, sensitive in rules:
        if any(needle in name for needle in needles) or any(needle.replace("_", "") in compact for needle in needles if len(needle) >= 4):
            result = {"semantic_type": semantic_type, "sensitive_field": sensitive}
            if semantic_type == "phone_number" and len(nonempty):
                digits = nonempty.str.replace(r"\D", "", regex=True)
                valid = digits.str.len().between(7, 15)
                result["invalid_format_count"] = int((~valid).sum())
            elif semantic_type == "email" and len(nonempty):
                valid = nonempty.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
                result["invalid_format_count"] = int((~valid).sum())
            return result

    if len(nonempty):
        email_match = nonempty.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
        if email_match.mean() >= 0.8:
            return {"semantic_type": "email", "sensitive_field": True, "invalid_format_count": int((~email_match).sum())}
        digits = nonempty.str.replace(r"\D", "", regex=True)
        phone_like = digits.str.len().between(7, 15) & nonempty.str.contains(r"[0-9]", regex=True)
        if phone_like.mean() >= 0.8 and (series.nunique(dropna=True) / max(len(nonempty), 1)) >= 0.5:
            return {"semantic_type": "phone_number", "sensitive_field": True, "invalid_format_count": int((~phone_like).sum())}

    if pd.api.types.is_datetime64_any_dtype(series):
        return {"semantic_type": "date_or_timestamp", "sensitive_field": False}
    if pd.api.types.is_numeric_dtype(series):
        return {"semantic_type": "numeric_measure_or_identifier", "sensitive_field": False}
    # Length-based classification only: no text samples are included in the AI payload.
    if len(nonempty):
        lengths = nonempty.str.len()
        if float(lengths.median()) >= 80 or float((lengths >= 40).mean()) >= 0.75:
            return {"semantic_type": "descriptive_text", "sensitive_field": False}
    return {"semantic_type": "categorical_or_text", "sensitive_field": False}

def compute_iqr_statistics(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound)
        | (series > upper_bound)
    ]

    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(round(series.mean(), 2)),
        "median": float(series.median()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "outlier_count": int(len(outliers))
    }

def profile_column(series):
    semantic_hints = detect_semantic_hints(series)

    column_profile = {
        "dtype": str(series.dtype),
        "missing_count": int(
            series.isna().sum()
        ),
        "missing_percentage": float(
            round(
                series.isna().mean() * 100,
                2
            )
        ),
        "unique_count": int(
            series.nunique()
        ),
        "empty_column": bool(
            series.isna().all()
        ),
        "constant_column": bool(
            series.nunique(dropna=True) <= 1
        ),
        "sample_values": (
            series
            .dropna()
            .head(5)
            .tolist()
        )
    }

    if semantic_hints:
        column_profile[
            "semantic_hints"
        ] = semantic_hints

    if series.nunique() <= 20:
        column_profile["value_counts"] = (
            series
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

    if (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
        and "possible_identifier"
        not in semantic_hints
    ):
        column_profile[
            "statistics"
        ] = compute_iqr_statistics(series)

    return column_profile

def build_profile(df):
    profile = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "column_name_issues": [],
        "column_profiles": {}
    }

    for column in df.columns:

        column_string = str(column)

        normalized = re.sub(
            r"[^a-z0-9]+",
            "_",
            column_string.strip().lower()
        ).strip("_")

        if (
            column_string != normalized
            and column_string.strip()
        ):
            profile[
                "column_name_issues"
            ].append(
                {
                    "original": column_string,
                    "suggested": normalized
                }
            )

        column_profile = profile_column(df[column])
        column_profile.update(detect_semantic_type(column_string, df[column]))
        profile["column_profiles"][column] = column_profile

    return profile


def validate_csv_structure(file_path):
    try:
        with open(
            file_path,
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.reader(file)

            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError(
                    f"Input CSV file is empty: {file_path}"
                ) from error

            if not header:
                raise ValueError(
                    f"Input CSV file is empty (contains no columns): "
                    f"{file_path}"
                )

            expected_fields = len(header)

            for line_number, row in enumerate(
                reader,
                start=2
            ):
                if not row:
                    continue

                if len(row) != expected_fields:
                    raise ValueError(
                        f"Malformed CSV structure at line "
                        f"{line_number}: expected "
                        f"{expected_fields} fields, found "
                        f"{len(row)}"
                    )

    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Input CSV file not found: {file_path}"
        ) from error

    except UnicodeDecodeError as error:
        raise ValueError(
            "Input CSV file has an unsupported "
            f"text encoding: {file_path}"
        ) from error

    except PermissionError as error:
        raise PermissionError(
            f"Permission denied while reading "
            f"input file: {file_path}"
        ) from error

    except OSError as error:
        raise OSError(
            f"Could not read input CSV file "
            f"'{file_path}': {error}"
        ) from error


def profile_dataset(file_path):

    if not file_path:
        raise ValueError(
            "No input file was provided"
        )

    if not isinstance(file_path, str):
        raise TypeError(
            "Input file path must be a string"
        )

    validate_csv_structure(file_path)

    try:
        df = pd.read_csv(
            file_path,
            on_bad_lines="error"
        )

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"Input CSV file is empty: {file_path}"
        ) from error

    except pd.errors.ParserError as error:
        raise ValueError(
            f"Input CSV file could not be parsed: "
            f"{file_path}"
        ) from error

    except UnicodeDecodeError as error:
        raise ValueError(
            f"Input CSV file has an unsupported "
            f"text encoding: {file_path}"
        ) from error

    if df.shape[1] == 0:
        raise ValueError(
            f"Input CSV file is empty (contains no columns): "
            f"{file_path}"
        )

    if df.empty:
        raise ValueError(
            f"Input CSV file contains no data rows: "
            f"{file_path}"
        )

    invalid_columns = [
        column
        for column in df.columns
        if (
            not str(column).strip()
            or str(column).startswith("Unnamed:")
        )
    ]

    if invalid_columns:
        raise ValueError(
            "Input CSV contains invalid or unnamed "
            f"columns: {invalid_columns}"
        )

    return df, build_profile(df)


if __name__ == "__main__":

    df, profile = profile_dataset(
        FILE_PATH
    )

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nMissing value percentage:")
    print(
        (df.isnull().mean() * 100).round(2)
    )

    print(
        "\nDuplicate rows:",
        df.duplicated().sum()
    )

    with open(
        "profile.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            profile,
            file,
            indent=2
        )

    print("\nProfile saved to profile.json")