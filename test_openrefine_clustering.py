"""
Unit tests for key-collision and edit-distance clustering in DataRefine.
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
    # Standard Soundex phonetic collisions
    assert soundex("Robert") == soundex("Rupert")
    assert soundex("Smith") == soundex("Smyth")
    assert soundex("") == ""
    assert soundex(None) == ""

    # Authoritative NARA reference test cases
    assert soundex("Ashcraft") == "A261"  # H does not reset code: S(2)-H-C(2) collapses, R(6), F(1) -> A261
    assert soundex("Tymczak") == "T522"   # Vowel 'a' separates Z(2) and K(2) -> T522
    assert soundex("Pfister") == "P236"   # P(1) and F(1) collapse -> P236
    assert soundex("Jackson") == "J250"
    assert soundex("Gutierrez") == "G362"
    assert soundex("Honeyman") == "H555"
    assert soundex("Euler") == "E460"
    assert soundex("Gauss") == "G200"
    assert soundex("Hilbert") == "H416"
    assert soundex("Knuth") == "K530"


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
    # Check consistent cluster schema contract
    assert "values" in clusters[0]
    assert "size" in clusters[0]
    assert clusters[0]["size"] == 3
    assert clusters[0]["canonical"] == clusters[0]["proposed_canonical"]


def test_distance_clustering_prevents_transitive_drift():
    # "cat" ~ "bat" (dist 1) and "bat" ~ "bet" (dist 1), but "cat" ~ "bet" (dist 2)
    # Under complete-linkage with max_distance=1, "cat" and "bet" cannot be in the same cluster
    data = pd.Series([
        "cat",
        "bat",
        "bet",
        "bit",
    ])
    clusters = detect_clusters(data, method="levenshtein", params={"max_distance": 1, "min_similarity": 0.6})
    for cl in clusters:
        members = [m["value"] for m in cl["members"]]
        # In any cluster, verify all pairs have edit distance <= 1
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                assert levenshtein_distance(members[i], members[j]) <= 1
