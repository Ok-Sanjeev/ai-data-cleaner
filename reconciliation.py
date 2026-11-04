"""
Entity Reconciliation Framework for DataRefine.

Matches messy column values against reference entity catalogs (countries,
currencies, US states, tech job roles, or custom user lists) and suggests
canonical entities with confidence scores.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from openrefine_clustering import fingerprint, levenshtein_distance, similarity_score


# =====================================================================
# BUILT-IN REFERENCE CATALOGS
# =====================================================================

REFERENCE_CATALOGS: Dict[str, Dict[str, List[str]]] = {
    "countries": {
        "United States": ["usa", "us", "u.s.", "u.s.a.", "united states of america", "america"],
        "United Kingdom": ["uk", "u.k.", "great britain", "britain", "england"],
        "India": ["in", "ind", "bharat", "republic of india"],
        "Canada": ["ca", "can"],
        "Germany": ["de", "deu", "deutschland"],
        "France": ["fr", "fra", "french republic"],
        "Japan": ["jp", "jpn", "nippon"],
        "Australia": ["au", "aus"],
        "China": ["cn", "chn", "prc", "people's republic of china"],
        "Singapore": ["sg", "sgp", "republic of singapore"],
        "Brazil": ["br", "bra", "brasil"],
        "Netherlands": ["nl", "nld", "holland"],
        "Spain": ["es", "esp", "espana"],
        "Italy": ["it", "ita", "italia"],
        "Switzerland": ["ch", "che", "swiss"],
    },
    "us_states": {
        "California": ["ca", "calif", "calif."],
        "New York": ["ny", "n.y."],
        "Texas": ["tx", "tex."],
        "Washington": ["wa", "wash."],
        "Florida": ["fl", "fla."],
        "Illinois": ["il", "ill."],
        "Massachusetts": ["ma", "mass."],
        "Colorado": ["co", "colo."],
        "Georgia": ["ga"],
        "North Carolina": ["nc", "n.c."],
    },
    "currencies": {
        "USD": ["$", "dollar", "us dollar", "usd"],
        "EUR": ["€", "euro", "eur"],
        "GBP": ["£", "pound", "british pound", "gbp"],
        "INR": ["₹", "rupee", "indian rupee", "inr", "rs", "rs."],
        "JPY": ["¥", "yen", "japanese yen", "jpy"],
        "CAD": ["c$", "cad", "canadian dollar"],
        "AUD": ["a$", "aud", "australian dollar"],
        "CHF": ["chf", "swiss franc"],
    },
    "tech_roles": {
        "Software Engineer": ["swe", "software developer", "sde", "software dev", "programmer", "sw engineer"],
        "Data Scientist": ["ds", "data science practitioner", "applied scientist", "ml data scientist"],
        "Data Analyst": ["da", "analytics analyst", "bi analyst", "business intelligence analyst"],
        "Data Engineer": ["de", "big data engineer", "data platform engineer", "etl developer"],
        "Machine Learning Engineer": ["mle", "ml engineer", "ai engineer", "deep learning engineer"],
        "Product Manager": ["pm", "product lead", "technical product manager", "tpm"],
        "DevOps Engineer": ["devops", "sre", "site reliability engineer", "infrastructure engineer"],
        "Frontend Developer": ["frontend engineer", "front end dev", "ui developer", "react developer"],
        "Backend Developer": ["backend engineer", "back end dev", "server developer", "api engineer"],
        "Full Stack Developer": ["fullstack engineer", "full stack dev", "full-stack engineer"],
    },
}


# =====================================================================
# RECONCILIATION MATCHER
# =====================================================================

def reconcile_column_values(
    series: pd.Series,
    catalog_name: str = "countries",
    custom_entities: Optional[List[str]] = None,
    match_threshold: float = 0.75,
    auto_match_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Reconcile messy values in a Series against a reference catalog.

    Returns:
    - total_unique: count of unique values evaluated
    - matched_count: count of values with a high-confidence match
    - candidates: list of candidate matches per unique value:
        {
          "original": "U.S.A.",
          "count": 142,
          "matched_canonical": "United States",
          "score": 1.0,
          "auto_match": True,
          "alternatives": [{"name": "...", "score": ...}]
        }
    """
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]

    if non_null.empty:
        return {
            "catalog": catalog_name,
            "total_unique": 0,
            "matched_count": 0,
            "candidates": [],
        }

    value_counts = non_null.value_counts().to_dict()

    # Build reference lookup table: alias/canonical -> canonical
    canonicals: Set[str] = set()
    alias_lookup: Dict[str, str] = {}
    fingerprint_lookup: Dict[str, str] = {}

    if catalog_name == "custom" and custom_entities:
        for entity in custom_entities:
            ent = str(entity).strip()
            if ent:
                canonicals.add(ent)
                alias_lookup[ent.lower()] = ent
                fp = fingerprint(ent)
                if fp:
                    fingerprint_lookup[fp] = ent
    elif catalog_name in REFERENCE_CATALOGS:
        for canonical, aliases in REFERENCE_CATALOGS[catalog_name].items():
            canonicals.add(canonical)
            alias_lookup[canonical.lower()] = canonical
            fp = fingerprint(canonical)
            if fp:
                fingerprint_lookup[fp] = canonical

            for alias in aliases:
                alias_lookup[alias.lower()] = canonical
                afp = fingerprint(alias)
                if afp:
                    fingerprint_lookup[afp] = canonical
    else:
        raise ValueError(
            f"Unknown catalog '{catalog_name}'. Available: {list(REFERENCE_CATALOGS.keys())} or 'custom'."
        )

    canonical_list = sorted(list(canonicals))
    results = []

    for orig_val, count in value_counts.items():
        val_clean = str(orig_val).strip()
        val_lower = val_clean.lower()
        val_fp = fingerprint(val_clean)

        # 1. Exact alias match (100% confidence)
        if val_lower in alias_lookup:
            matched_canonical = alias_lookup[val_lower]
            results.append({
                "original": orig_val,
                "count": count,
                "matched_canonical": matched_canonical,
                "score": 1.0,
                "auto_match": True,
                "alternatives": [],
            })
            continue

        # 2. Fingerprint match (95% confidence)
        if val_fp and val_fp in fingerprint_lookup:
            matched_canonical = fingerprint_lookup[val_fp]
            results.append({
                "original": orig_val,
                "count": count,
                "matched_canonical": matched_canonical,
                "score": 0.95,
                "auto_match": True,
                "alternatives": [],
            })
            continue

        # 3. Fuzzy similarity search across canonicals
        scored_candidates = []
        for canon in canonical_list:
            sim = similarity_score(val_lower, canon.lower())
            if sim >= match_threshold:
                scored_candidates.append((canon, round(sim, 3)))

        # Also search aliases for fuzzy hits
        for alias, canon in alias_lookup.items():
            sim = similarity_score(val_lower, alias)
            if sim >= match_threshold:
                scored_candidates.append((canon, round(sim * 0.95, 3)))

        if scored_candidates:
            # Sort by score descending
            scored_candidates.sort(key=lambda x: -x[1])
            # Deduplicate by canonical
            seen = set()
            deduped = []
            for c, s in scored_candidates:
                if c not in seen:
                    seen.add(c)
                    deduped.append({"name": c, "score": s})

            best = deduped[0]
            is_auto = best["score"] >= auto_match_threshold

            results.append({
                "original": orig_val,
                "count": count,
                "matched_canonical": best["name"],
                "score": best["score"],
                "auto_match": is_auto,
                "alternatives": deduped[1:4],
            })
        else:
            results.append({
                "original": orig_val,
                "count": count,
                "matched_canonical": None,
                "score": 0.0,
                "auto_match": False,
                "alternatives": [],
            })

    # Sort results: auto-matched first, then by frequency descending
    results.sort(key=lambda x: (-x["score"], -x["count"]))

    matched_count = sum(1 for r in results if r["matched_canonical"] is not None)

    return {
        "catalog": catalog_name,
        "total_unique": len(value_counts),
        "matched_count": matched_count,
        "candidates": results,
    }


# Alias for backward/naming compatibility
reconcile_column_candidates = reconcile_column_values

