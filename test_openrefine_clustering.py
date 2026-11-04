"""
Unit tests for OpenRefine clustering and merging algorithms in DataRefine.
"""

import pandas as pd
import pytest

from openrefine_clustering import (
    fingerprint,
    ngram_fingerprint,
    soundex,
    metaphone,
    levenshtein_distance,
    damerau_levenshtein_distance,
    similarity_score,
    detect_clusters,
)


def test_fingerprint():
    # Casing, whitespace, punctuation, and token ordering
    assert fingerprint("New York") == "new york"
    assert fingerprint("   new   york   ") == "new york"
    assert fingerprint("York, New") == "new york"
    assert fingerprint("New-York!") == "new york"
    # Accents removal
    assert fingerprint("Montréal") == "montreal"
    assert fingerprint("München") == "munchen"
    # Empty / none
    assert fingerprint("") == ""
    assert fingerprint(None) == ""


def test_ngram_fingerprint():
    # 2-grams: "paris" -> "pa", "ar", "ri", "is" -> sorted unique joined: "arispari"
    fp1 = ngram_fingerprint("paris", n=2)
    fp2 = ngram_fingerprint("  Paris  ", n=2)
    assert fp1 == fp2
    assert len(fp1) > 0


def test_phonetic_soundex():
    # Standard soundex phonetic collisions
    assert soundex("Robert") == soundex("Rupert")
    assert soundex("Smith") == soundex("Smyth")
    assert soundex("") == ""
    assert soundex(None) == ""


def test_phonetic_metaphone():
    # Metaphone phonetic collisions
    assert metaphone("Smith") == metaphone("Smyth")
    assert metaphone("Stephen") == metaphone("Steven")
    assert metaphone("") == ""


def test_edit_distances():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("apple", "apple") == 0
    assert levenshtein_distance("", "test") == 4

    # Damerau handles transpositions as 1 edit
    assert damerau_levenshtein_distance("ca", "ac") == 1
    assert levenshtein_distance("ca", "ac") == 2

    # Normalized similarity score (0.0 - 1.0)
    assert similarity_score("apple", "apple") == 1.0
    assert 0.0 <= similarity_score("kitten", "sitting") <= 1.0
    assert similarity_score("", "apple") == 0.0


def test_detect_clusters_fingerprint():
    data = pd.Series([
        "Google Inc.",
        "google inc",
        "Inc, Google",
        "Microsoft Corp",
        "microsoft corp.",
        "Amazon",
    ])
    clusters = detect_clusters(data, method="fingerprint")
    assert len(clusters) == 2  # Google cluster and Microsoft cluster

    google_cluster = next(c for c in clusters if any("google" in m["value"].lower() for m in c["members"]))
    assert len(google_cluster["members"]) == 3
    assert google_cluster["canonical"] in ["Google Inc.", "google inc", "Inc, Google"]


def test_detect_clusters_levenshtein():
    data = pd.Series([
        "Data Science",
        "Data Sciene",
        "Data Scienc",
        "Web Developer",
    ])
    clusters = detect_clusters(data, method="levenshtein", params={"max_distance": 2, "min_similarity": 0.8})
    assert len(clusters) == 1
    assert len(clusters[0]["members"]) == 3
