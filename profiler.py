import csv
import json
import os
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
    "0",
    "t",
    "f",
    "on",
    "off",
    "enabled",
    "disabled",
    "active",
    "inactive",
    "pass",
    "fail",
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
        cleaned_num = (
            non_null.astype("string")
            .str.replace(r"[$€£¥₹%]", "", regex=True)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        numeric = pd.to_numeric(
            cleaned_num,
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
        (("url", "website", "web_url", "link", "uri", "href", "job_url"), "url", False),
        (("phone", "phone_number", "mobile", "telephone", "contact_number", "tel"), "phone_number", True),
        (("dob", "date_of_birth", "birth_date", "birthdate"), "date_of_birth", True),
        (("ssn", "social_security", "national_id", "passport", "tax_id", "aadhaar", "pan"), "government_or_identity_id", True),
        (("salary", "wage", "income", "compensation", "pay_amount", "revenue", "price", "cost"), "salary_or_income", True),
        (("address", "street_address", "postal_address", "city", "state", "pincode", "zipcode", "zip_code"), "address", True),
        (("first_name", "last_name", "full_name", "customer_name", "employee_name", "client_name"), "person_name", True),
        (("date", "_at", "_on", "timestamp", "created_time", "updated_time", "hire_date", "joining_date"), "date_or_timestamp", False),
    ]
    for needles, semantic_type, sensitive in rules:
        if any(needle in name for needle in needles) or any(needle.replace("_", "") in compact for needle in needles if len(needle) >= 4):
            # For date types, verify that non-empty values actually resemble dates, not relative text
            if semantic_type == "date_or_timestamp" and len(nonempty):
                parsed_dates = pd.to_datetime(nonempty, format="mixed", errors="coerce")
                if parsed_dates.notna().mean() < 0.5:
                    continue

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
        url_match = nonempty.str.match(r"^https?://[^\s]+$")
        if url_match.mean() >= 0.7:
            return {"semantic_type": "url", "sensitive_field": False}

        email_match = nonempty.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
        if email_match.mean() >= 0.8:
            return {"semantic_type": "email", "sensitive_field": True, "invalid_format_count": int((~email_match).sum())}

        has_words = nonempty.str.contains(r"[a-zA-Z]{3,}", regex=True).mean() >= 0.2
        if not has_words:
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

def compute_column_quality_categories(series, column_name, semantic_hints=None):
    """Compute the 5 independent data-quality categories for a column:
    A. Missing Values
    B. Statistical Outliers
    C. Invalid Values
    D. Data Type Inconsistencies
    E. Formatting Issues
    """
    total = len(series)
    non_null = series.dropna()
    categories = {}
    col_str = str(column_name).lower() if column_name is not None else ""
    hints = set(semantic_hints or [])

    # A. Missing Values
    null_count = int(series.isna().sum())
    empty_str_count = 0
    marker_count = 0
    missing_markers = {"nan", "null", "none", "na", "n/a", "n.a.", "-", "--", "?", "unknown", "missing", "undefined"}

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        str_s = series.dropna().astype(str).str.strip()
        empty_str_count = int((str_s == "").sum())
        marker_count = int(str_s.str.lower().isin(missing_markers).sum())

    total_missing = null_count + empty_str_count + marker_count
    categories["missing_values"] = {
        "category": "Missing Values",
        "count": total_missing,
        "percentage": round((total_missing / max(1, total)) * 100, 2),
        "null_count": null_count,
        "empty_string_count": empty_str_count,
        "marker_count": marker_count,
        "status": "detected" if total_missing > 0 else "clean",
        "explanation": f"{total_missing} missing value(s) ({round((total_missing/max(1,total))*100,1)}%)" if total_missing > 0 else "No missing values."
    }

    # B. Statistical Outliers (Numeric columns only, never identifier)
    outlier_count = 0
    outlier_stats = {}
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and "possible_identifier" not in hints:
        clean_num = series.dropna()
        if len(clean_num) >= 4:
            outlier_stats = compute_iqr_statistics(clean_num)
            outlier_count = outlier_stats.get("outlier_count", 0)

    categories["statistical_outliers"] = {
        "category": "Statistical Outliers",
        "count": outlier_count,
        "percentage": round((outlier_count / max(1, total)) * 100, 2),
        "stats": outlier_stats,
        "status": "preserved_for_review" if outlier_count > 0 else "clean",
        "explanation": f"{outlier_count} statistical outlier(s) detected via IQR; statistical variation preserved without clipping." if outlier_count > 0 else "No statistical outliers detected."
    }

    # C. Invalid Values
    invalid_count = 0
    invalid_reasons = []
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        non_neg_hints = {"id", "age", "count", "number", "duration", "rating", "score", "year", "price", "salary"}
        if any(h in col_str for h in non_neg_hints) and (series.dropna() < 0).any():
            neg_count = int((series.dropna() < 0).sum())
            invalid_count += neg_count
            invalid_reasons.append(f"{neg_count} negative value(s) in non-negative column")

    if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and len(non_null) > 0:
        str_nonempty = series.dropna().astype(str).str.strip()
        str_nonempty = str_nonempty[str_nonempty != ""]
        if "email" in col_str and len(str_nonempty) > 0:
            valid_email = str_nonempty.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
            bad_email = int((~valid_email).sum())
            if bad_email > 0:
                invalid_count += bad_email
                invalid_reasons.append(f"{bad_email} malformed email address(es)")
        if ("phone" in col_str or "mobile" in col_str) and len(str_nonempty) > 0:
            digits = str_nonempty.str.replace(r"\D", "", regex=True)
            valid_phone = digits.str.len().between(7, 15)
            bad_phone = int((~valid_phone).sum())
            if bad_phone > 0:
                invalid_count += bad_phone
                invalid_reasons.append(f"{bad_phone} malformed phone number(s)")

    categories["invalid_values"] = {
        "category": "Invalid Values",
        "count": invalid_count,
        "percentage": round((invalid_count / max(1, total)) * 100, 2),
        "reasons": invalid_reasons,
        "status": "detected" if invalid_count > 0 else "clean",
        "explanation": "; ".join(invalid_reasons) if invalid_reasons else "No constraint violations or invalid values detected."
    }

    # D. Data Type Inconsistencies
    type_count = 0
    type_notes = []
    if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and len(non_null) > 0:
        str_s = series.dropna().astype(str).str.strip()
        num_clean = str_s.str.replace(r"[$€£¥₹%]", "", regex=True).str.replace(",", "", regex=False)
        converted_num = pd.to_numeric(num_clean, errors="coerce")
        if converted_num.notna().mean() >= 0.8 and not converted_num.notna().all():
            unparsed = int((converted_num.isna()).sum())
            type_count += unparsed
            type_notes.append(f"Numeric column with {unparsed} non-numeric string value(s)")
        lower_s = str_s.str.lower()
        if lower_s.isin(BOOLEAN_VALUES).mean() >= 0.8 and not lower_s.isin(BOOLEAN_VALUES).all():
            bad_bool = int((~lower_s.isin(BOOLEAN_VALUES)).sum())
            type_count += bad_bool
            type_notes.append(f"Mixed boolean representation with {bad_bool} unparseable value(s)")
        if ("date" in col_str or "time" in col_str or "_at" in col_str) and not pd.api.types.is_datetime64_any_dtype(series):
            parsed_d = pd.to_datetime(str_s, format="mixed", errors="coerce")
            if parsed_d.notna().mean() >= 0.5:
                unparsed_d = int(parsed_d.isna().sum())
                if unparsed_d > 0:
                    type_count += unparsed_d
                    type_notes.append(f"{unparsed_d} unparseable temporal/date string(s)")

    categories["type_inconsistencies"] = {
        "category": "Data Type Inconsistencies",
        "count": type_count,
        "percentage": round((type_count / max(1, total)) * 100, 2),
        "notes": type_notes,
        "status": "detected" if type_count > 0 else "clean",
        "explanation": "; ".join(type_notes) if type_notes else "Consistent data types throughout column."
    }

    # E. Formatting Issues
    format_count = 0
    format_notes = []
    if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and len(non_null) > 0:
        str_s = series.dropna().astype(str)
        ws_issue = (str_s != str_s.str.strip()) | str_s.str.contains(r"\s{2,}", regex=True)
        ws_cnt = int(ws_issue.sum())
        if ws_cnt > 0:
            format_count += ws_cnt
            format_notes.append(f"{ws_cnt} leading/trailing or repeated whitespace issue(s)")
        lower_nunique = str_s.str.strip().str.lower().nunique()
        raw_nunique = str_s.str.strip().nunique()
        if lower_nunique < raw_nunique:
            diff_cnt = raw_nunique - lower_nunique
            format_count += diff_cnt
            format_notes.append(f"{diff_cnt} category casing variation(s)")

    categories["formatting_issues"] = {
        "category": "Formatting Issues",
        "count": format_count,
        "percentage": round((format_count / max(1, total)) * 100, 2),
        "notes": format_notes,
        "status": "detected" if format_count > 0 else "clean",
        "explanation": "; ".join(format_notes) if format_notes else "Consistent formatting throughout column."
    }

    return categories


def profile_column(series, column_name=None):
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

        column_profile = profile_column(df[column], column_name=column_string)
        column_profile.update(detect_semantic_type(column_string, df[column]))
        column_profile["quality_categories"] = compute_column_quality_categories(
            df[column], column_name=column_string, semantic_hints=column_profile.get("semantic_hints", [])
        )
        profile["column_profiles"][column] = column_profile

    # Dataset-level quality summary
    profile["quality_categories_summary"] = {
        "missing_values": {
            "total_count": sum(cp.get("quality_categories", {}).get("missing_values", {}).get("count", 0) for cp in profile["column_profiles"].values()),
            "affected_columns": [c for c, cp in profile["column_profiles"].items() if cp.get("quality_categories", {}).get("missing_values", {}).get("count", 0) > 0],
        },
        "statistical_outliers": {
            "total_count": sum(cp.get("quality_categories", {}).get("statistical_outliers", {}).get("count", 0) for cp in profile["column_profiles"].values()),
            "affected_columns": [c for c, cp in profile["column_profiles"].items() if cp.get("quality_categories", {}).get("statistical_outliers", {}).get("count", 0) > 0],
        },
        "invalid_values": {
            "total_count": sum(cp.get("quality_categories", {}).get("invalid_values", {}).get("count", 0) for cp in profile["column_profiles"].values()),
            "affected_columns": [c for c, cp in profile["column_profiles"].items() if cp.get("quality_categories", {}).get("invalid_values", {}).get("count", 0) > 0],
        },
        "type_inconsistencies": {
            "total_count": sum(cp.get("quality_categories", {}).get("type_inconsistencies", {}).get("count", 0) for cp in profile["column_profiles"].values()),
            "affected_columns": [c for c, cp in profile["column_profiles"].items() if cp.get("quality_categories", {}).get("type_inconsistencies", {}).get("count", 0) > 0],
        },
        "formatting_issues": {
            "total_count": sum(cp.get("quality_categories", {}).get("formatting_issues", {}).get("count", 0) for cp in profile["column_profiles"].values()),
            "affected_columns": [c for c, cp in profile["column_profiles"].items() if cp.get("quality_categories", {}).get("formatting_issues", {}).get("count", 0) > 0],
        },
    }

    return profile


def detect_encoding_and_delimiter(file_path):
    if not isinstance(file_path, str):
        raise TypeError("Input file path must be a string")
    if not file_path:
        raise ValueError("No input file was provided")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input CSV file not found: {file_path}")

    try:
        size = os.path.getsize(file_path)
    except OSError as error:
        raise OSError(f"Could not read input CSV file '{file_path}': {error}") from error

    if size == 0:
        raise ValueError(f"Input CSV file is empty: {file_path}")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    valid_encoding = None
    sample_text = ""
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, newline="") as file:
                sample_text = file.read(8192)
                valid_encoding = enc
                break
        except UnicodeDecodeError:
            continue
        except PermissionError as error:
            raise PermissionError(f"Permission denied while reading input file: {file_path}") from error
        except OSError as error:
            raise OSError(f"Could not read input CSV file '{file_path}': {error}") from error

    if not valid_encoding:
        raise ValueError(f"Input CSV file has an unsupported text encoding: {file_path}")

    if not sample_text.strip():
        raise ValueError(f"Input CSV file is empty: {file_path}")

    delimiter = ","
    try:
        lines = [line for line in sample_text.splitlines() if line.strip()]
        if lines:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff("\n".join(lines[:5]), delimiters=",\t;|")
            delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    return valid_encoding, delimiter


def validate_csv_structure(file_path, encoding=None, delimiter=None):
    if not isinstance(file_path, str):
        raise TypeError("Input file path must be a string")
    if not file_path:
        raise ValueError("No input file was provided")

    if encoding is None or delimiter is None:
        det_enc, det_delim = detect_encoding_and_delimiter(file_path)
        if encoding is None:
            encoding = det_enc
        if delimiter is None:
            delimiter = det_delim

    try:
        with open(
            file_path,
            "r",
            encoding=encoding,
            newline=""
        ) as file:

            reader = csv.reader(file, delimiter=delimiter)

            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError(
                    f"Input CSV file is empty: {file_path}"
                ) from error

            if not header or not any(str(field).strip() for field in header):
                raise ValueError(
                    f"Input CSV file is empty (contains no columns): "
                    f"{file_path}"
                )

            expected_fields = len(header)

            for line_number, row in enumerate(
                reader,
                start=2
            ):
                if not row or not any(str(field).strip() for field in row):
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

    encoding, delimiter = detect_encoding_and_delimiter(file_path)
    validate_csv_structure(file_path, encoding=encoding, delimiter=delimiter)

    try:
        df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
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

    # Auto-repair blank or Unnamed columns so the dataset can be processed
    renamed_columns = {}
    for i, col in enumerate(df.columns, start=1):
        col_name = str(col).strip()
        if not col_name or col_name.startswith("Unnamed:"):
            renamed_columns[col] = f"unnamed_column_{i}"

    if renamed_columns:
        df = df.rename(columns=renamed_columns)

    profile = build_profile(df)
    profile["encoding"] = encoding
    profile["delimiter"] = delimiter
    return df, profile


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