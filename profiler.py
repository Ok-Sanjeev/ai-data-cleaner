import csv
import json

import pandas as pd


FILE_PATH = "data/Messy_Employee_dataset.csv"


def detect_semantic_hints(series):
    hints = []

    # Possible date
    if series.dtype == "str":
        parsed = pd.to_datetime(
            series,
            format="mixed",
            errors="coerce"
        )

        valid_percentage = parsed.notna().mean() * 100

        if valid_percentage >= 80:
            hints.append("possible_date")

    # Possible email
    if series.dtype == "str":
        non_null = series.dropna()

        if len(non_null) > 0:
            email_pattern = non_null.str.contains(
                r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
                regex=True
            )

            if email_pattern.mean() >= 0.8:
                hints.append("possible_email")

    # Possible identifier
    if (
        series.nunique() == len(series)
        and series.notna().all()
    ):
        hints.append("possible_identifier")

    # Negative values in identifier-like numeric columns
    if (
        "possible_identifier" in hints
        and pd.api.types.is_numeric_dtype(series)
        and (series.dropna() < 0).any()
    ):
        hints.append("invalid_negative_values")

    return hints


def compute_iqr_statistics(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
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
    column_profile = {
        "dtype": str(series.dtype),
        "missing_count": int(series.isna().sum()),
        "missing_percentage": float(
            round(series.isna().mean() * 100, 2)
        ),
        "unique_count": int(series.nunique()),
        "sample_values": series.dropna().head(5).tolist()
    }

    semantic_hints = detect_semantic_hints(series)

    if semantic_hints:
        column_profile["semantic_hints"] = semantic_hints

    # Store value frequencies only for low-cardinality columns
    if series.nunique() <= 20:
        column_profile["value_counts"] = (
            series
            .value_counts(dropna=False)
            .to_dict()
        )

    # Don't calculate numerical statistics for identifier-like columns
    if (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
        and "possible_identifier" not in semantic_hints
    ):
        column_profile["statistics"] = compute_iqr_statistics(series)

    return column_profile


def build_profile(df):
    profile = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_profiles": {}
    }

    for column in df.columns:
        profile["column_profiles"][column] = profile_column(
            df[column]
        )

    return profile


def validate_csv_structure(file_path):
    """
    Validate the raw CSV structure before pandas parses it.

    Every non-empty data row must contain the same number
    of fields as the header.
    """

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
                    f"Input CSV file contains no columns: {file_path}"
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
            "Input CSV file has an unsupported text encoding: "
            f"{file_path}"
        ) from error

    except PermissionError as error:
        raise PermissionError(
            "Permission denied while reading input file: "
            f"{file_path}"
        ) from error

    except OSError as error:
        raise OSError(
            f"Could not read input CSV file '{file_path}': {error}"
        ) from error


def profile_dataset(file_path):
    """
    Load, validate, and profile a CSV dataset.
    """

    if not file_path:
        raise ValueError(
            "No input file was provided"
        )

    if not isinstance(file_path, str):
        raise TypeError(
            "Input file path must be a string"
        )

    # Validate raw CSV structure before pandas parsing.
    validate_csv_structure(file_path)

    # Parse the validated CSV.
    try:
        df = pd.read_csv(
            file_path,
            on_bad_lines="error"
        )

    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Input CSV file not found: {file_path}"
        ) from error

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"Input CSV file is empty: {file_path}"
        ) from error

    except pd.errors.ParserError as error:
        raise ValueError(
            f"Input CSV file could not be parsed: {file_path}"
        ) from error

    except UnicodeDecodeError as error:
        raise ValueError(
            "Input CSV file has an unsupported text encoding: "
            f"{file_path}"
        ) from error

    except PermissionError as error:
        raise PermissionError(
            "Permission denied while reading input file: "
            f"{file_path}"
        ) from error

    except OSError as error:
        raise OSError(
            f"Could not read input CSV file '{file_path}': {error}"
        ) from error

    # No columns
    if df.shape[1] == 0:
        raise ValueError(
            f"Input CSV file contains no columns: {file_path}"
        )

    # Header exists but no data rows
    if df.empty:
        raise ValueError(
            f"Input CSV file contains no data rows: {file_path}"
        )

    # Reject blank or automatically generated unnamed columns.
    invalid_columns = [
        column
        for column in df.columns
        if not str(column).strip()
        or str(column).startswith("Unnamed:")
    ]

    if invalid_columns:
        raise ValueError(
            "Input CSV contains invalid or unnamed columns: "
            f"{invalid_columns}"
        )

    return df, build_profile(df)


if __name__ == "__main__":
    df, profile = profile_dataset(FILE_PATH)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nMissing value percentage:")
    print((df.isnull().mean() * 100).round(2))

    print("\nDuplicate rows:", df.duplicated().sum())

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