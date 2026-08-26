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


def profile_dataset(file_path):
    df = pd.read_csv(file_path)

    profile = build_profile(df)

    return df, profile


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

    with open("profile.json", "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    print("\nProfile saved to profile.json")