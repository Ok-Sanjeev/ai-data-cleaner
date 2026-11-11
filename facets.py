"""
Multi-Column Faceting and Data Exploration Engine for DataRefine.

Supports:
- Text Facet: Value counts, frequency distribution, blank counts, search, and selection filters.
- Numeric Facet: Range min/max, summary statistics, uniform histogram bin distribution.
- Date Facet: Temporal min/max, null counts, timeline period distribution.
- Multi-Column Filter Engine: Combines text, numeric, date, and text search filters.
- Paginated & Sorted Preview: Returns filtered slice of data with row indices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_text_facet(
    series: pd.Series,
    top_n: int = 100,
    search_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute text facet value counts and distribution for a column.
    """
    total = len(series)
    null_count = int(series.isna().sum())
    blank_count = int((series.astype(str).str.strip() == "").sum()) if total > 0 else 0

    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]

    if search_query:
        query = search_query.lower()
        non_null = non_null[non_null.str.lower().str.contains(query, regex=False)]

    counts = non_null.value_counts()
    unique_count = len(counts)

    col_name = str(series.name) if series.name is not None else ""
    choices = [
        {
            "value": val,
            "count": int(cnt),
            "percentage": round((cnt / total) * 100, 1) if total > 0 else 0.0,
        }
        for val, cnt in counts.head(top_n).items()
    ]

    return {
        "facet_type": "text",
        "column": col_name,
        "total_rows": total,
        "null_count": null_count,
        "blank_count": blank_count,
        "unique_count": unique_count,
        "choices": choices,
    }


def compute_numeric_facet(
    series: pd.Series,
    bins_count: int = 15,
) -> Dict[str, Any]:
    """
    Compute numeric facet statistics and histogram distribution for a column.
    """
    total = len(series)
    null_count = int(series.isna().sum())

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    valid_count = len(numeric_series)

    col_name = str(series.name) if series.name is not None else ""
    if valid_count == 0:
        return {
            "facet_type": "numeric",
            "column": col_name,
            "total_rows": total,
            "null_count": null_count,
            "valid_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "bins": [],
        }

    min_val = float(numeric_series.min())
    max_val = float(numeric_series.max())
    mean_val = float(numeric_series.mean())
    median_val = float(numeric_series.median())

    # Build histogram bins
    bins = []
    if min_val == max_val or bins_count <= 1:
        bins = [{
            "from": min_val,
            "to": max_val,
            "count": valid_count,
            "label": f"{min_val:g}",
        }]
    else:
        counts, bin_edges = np.histogram(numeric_series, bins=bins_count)
        for i in range(len(counts)):
            b_from = float(bin_edges[i])
            b_to = float(bin_edges[i + 1])
            bins.append({
                "from": round(b_from, 2),
                "to": round(b_to, 2),
                "count": int(counts[i]),
                "label": f"{b_from:.1f} - {b_to:.1f}",
            })

    stats = {
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
    }

    return {
        "facet_type": "numeric",
        "column": col_name,
        "total_rows": total,
        "null_count": null_count,
        "valid_count": valid_count,
        "stats": stats,
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "bins": bins,
    }


def compute_date_facet(
    series: pd.Series,
    bins_count: int = 10,
) -> Dict[str, Any]:
    """
    Compute date timeline facet for a column.
    """
    total = len(series)
    null_count = int(series.isna().sum())

    date_series = pd.to_datetime(series, errors="coerce").dropna()
    valid_count = len(date_series)

    col_name = str(series.name) if series.name is not None else ""
    if valid_count == 0:
        return {
            "facet_type": "date",
            "column": col_name,
            "total_rows": total,
            "null_count": null_count,
            "valid_count": 0,
            "min_date": None,
            "max_date": None,
            "timeline": [],
            "years": [],
        }

    min_date = date_series.min()
    max_date = date_series.max()

    # Form timeline distribution by year or month
    time_span_days = (max_date - min_date).days
    if time_span_days > 730:
        # Group by Year
        period_counts = date_series.dt.year.value_counts().sort_index()
    elif time_span_days > 60:
        # Group by Year-Month
        period_counts = date_series.dt.to_period("M").value_counts().sort_index()
    else:
        # Group by Day
        period_counts = date_series.dt.to_period("D").value_counts().sort_index()

    timeline = [
        {"period": str(period), "count": int(count)}
        for period, count in period_counts.items()
    ]

    return {
        "facet_type": "date",
        "column": col_name,
        "total_rows": total,
        "null_count": null_count,
        "valid_count": valid_count,
        "valid_date_count": valid_count,
        "min_date": min_date.strftime("%Y-%m-%d"),
        "max_date": max_date.strftime("%Y-%m-%d"),
        "timeline": timeline[:50],
        "years": [{"year": item["period"], "count": item["count"]} for item in timeline[:50]],
    }


# =====================================================================
# MULTI-COLUMN FILTER ENGINE
# =====================================================================

def apply_filters(
    df: pd.DataFrame,
    filters: Optional[Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
    global_search: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter a DataFrame according to multiple column facets and search queries.
    Seamlessly accepts either dictionary-format or list-format filter specifications.
    """
    if df.empty:
        return df

    # Normalize list-based filter specifications (from frontend API calls)
    normalized_filters: Dict[str, Dict[str, Any]] = {}
    if isinstance(filters, list):
        for f_item in filters:
            if isinstance(f_item, dict) and "column" in f_item:
                col = f_item["column"]
                vals = f_item.get("values") or f_item.get("selected") or []
                mode = f_item.get("mode", "include")
                normalized_filters[col] = {
                    "type": f_item.get("type", "text"),
                    "selected": vals,
                    "invert": (mode == "exclude") or f_item.get("invert", False),
                    "include_null": f_item.get("include_null", False),
                    "include_blank": f_item.get("include_blank", False),
                    "min": f_item.get("min"),
                    "max": f_item.get("max"),
                    "start_date": f_item.get("start_date"),
                    "end_date": f_item.get("end_date"),
                    "search": f_item.get("search"),
                }
    elif isinstance(filters, dict):
        normalized_filters = filters

    mask = pd.Series(True, index=df.index)

    # 1. Global text search across all columns
    if global_search and str(global_search).strip():
        q = str(global_search).strip().lower()
        search_mask = pd.Series(False, index=df.index)
        for col in df.columns:
            col_str = df[col].astype(str).str.lower()
            search_mask |= col_str.str.contains(q, regex=False, na=False)
        mask &= search_mask

    # 2. Specific column facet filters
    if normalized_filters:
        for col, flt in normalized_filters.items():
            if col not in df.columns:
                continue

            series = df[col]
            flt_type = flt.get("type", "text")

            # Column search
            if flt.get("search"):
                sq = str(flt["search"]).strip().lower()
                c_mask = series.astype(str).str.lower().str.contains(sq, regex=False, na=False)
                mask &= c_mask

            if flt_type == "text":
                selected = flt.get("selected", [])
                if selected:
                    selected_set = {str(x) for x in selected}
                    col_str = series.astype(str).str.strip()
                    c_mask = col_str.isin(selected_set)

                    if flt.get("include_null"):
                        c_mask |= series.isna()
                    if flt.get("include_blank"):
                        c_mask |= (col_str == "")

                    if flt.get("invert", False):
                        c_mask = ~c_mask
                    mask &= c_mask

            elif flt_type == "numeric":
                num_series = pd.to_numeric(series, errors="coerce")
                c_mask = pd.Series(True, index=df.index)

                if flt.get("min") is not None:
                    min_val = float(flt["min"])
                    c_mask &= (num_series >= min_val)

                if flt.get("max") is not None:
                    max_val = float(flt["max"])
                    c_mask &= (num_series <= max_val)

                if flt.get("include_null"):
                    c_mask |= series.isna()
                else:
                    c_mask &= num_series.notna()

                mask &= c_mask

            elif flt_type == "date":
                dt_series = pd.to_datetime(series, errors="coerce")
                c_mask = pd.Series(True, index=df.index)

                if flt.get("start_date"):
                    s_date = pd.to_datetime(flt["start_date"], errors="coerce")
                    if pd.notna(s_date):
                        c_mask &= (dt_series >= s_date)

                if flt.get("end_date"):
                    e_date = pd.to_datetime(flt["end_date"], errors="coerce")
                    if pd.notna(e_date):
                        c_mask &= (dt_series <= e_date)

                if flt.get("include_null"):
                    c_mask |= series.isna()
                else:
                    c_mask &= dt_series.notna()

                mask &= c_mask

    return df[mask]


def explore_dataset(
    df: pd.DataFrame,
    filters: Optional[Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
    global_search: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_ascending: bool = True,
    offset: int = 0,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Complete interactive exploration: filters, sorts, paginates, and returns
    exploration summary with matching row counts and preview records.
    """
    total_rows = len(df)
    filtered_df = apply_filters(df, filters=filters, global_search=global_search)
    matched_rows = len(filtered_df)

    # Sort if requested
    if sort_by and sort_by in filtered_df.columns:
        try:
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=sort_ascending)
        except Exception:
            pass

    # Paginate
    paged_df = filtered_df.iloc[offset : offset + limit]

    # Convert preview records cleanly
    records = []
    for orig_idx, row in paged_df.iterrows():
        rec = {"_row_id": int(orig_idx) if isinstance(orig_idx, (int, np.integer)) else str(orig_idx)}
        for col, val in row.items():
            if pd.isna(val) or val is None:
                rec[col] = None
            elif isinstance(val, (np.integer, int)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                rec[col] = float(val) if not np.isnan(val) else None
            else:
                rec[col] = str(val)
        records.append(rec)

    matched_pct = round((matched_rows / total_rows * 100), 1) if total_rows > 0 else 0.0

    return {
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "matched_percentage": matched_pct,
        "offset": offset,
        "limit": limit,
        "columns": list(df.columns),
        "records": records,
    }


def compute_facet_auto(
    series: pd.Series,
    facet_type: str = "auto",
    top_n: int = 100,
    bins_count: int = 15,
    search_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute appropriate facet (text, numeric, or date) dynamically based on facet_type
    or column data type and contents.
    """
    ft = (facet_type or "auto").lower().strip()

    if ft == "text":
        return compute_text_facet(series, top_n=top_n, search_query=search_query)
    elif ft == "numeric":
        return compute_numeric_facet(series, bins_count=bins_count)
    elif ft in ("date", "datetime"):
        return compute_date_facet(series)

    # Auto-detection
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return compute_numeric_facet(series, bins_count=bins_count)
    elif pd.api.types.is_datetime64_any_dtype(series):
        return compute_date_facet(series)
    else:
        clean_s = series.dropna().astype(str).str.strip()
        clean_s = clean_s[clean_s != ""]
        if not clean_s.empty:
            num_sample = pd.to_numeric(clean_s.head(50), errors="coerce")
            if num_sample.notna().mean() >= 0.8:
                return compute_numeric_facet(series, bins_count=bins_count)
        return compute_text_facet(series, top_n=top_n, search_query=search_query)

