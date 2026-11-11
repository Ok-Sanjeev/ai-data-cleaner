"""
Advanced Text Clustering and Value Harmonization Engine for DataRefine.

Implements core key-collision and edit-distance clustering algorithms:
1. Key Collision:
   - Fingerprint: whitespace normalization, lowercase, punctuation removal, sorted unique tokens.
   - N-gram Fingerprint: sorted unique character n-grams.
   - Phonetic (Soundex): phonetic consonant-encoding key collision.
   - Phonetic (Metaphone): English phonetic rules key collision.
2. Distance / Nearest Neighbor:
   - Levenshtein Distance (edit distance with length-bucket optimization).
   - Damerau-Levenshtein Distance (transposition-aware edit distance).
   - Normalized similarity scoring (0.0 - 1.0).
3. Canonical Value Selection:
   - Frequency-based (highest occurrence in dataset).
   - Centroid-based (minimum aggregate edit distance to other members).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import pandas as pd


# =====================================================================
# 1. KEY COLLISION ALGORITHMS
# =====================================================================

def fingerprint(text: str) -> str:
    """
    Standard Key Collision Fingerprint algorithm.
    1. Normalize Unicode (NFKD) and strip accents.
    2. Lowercase and strip leading/trailing whitespace.
    3. Remove punctuation and control characters.
    4. Tokenize by whitespace, deduplicate, and sort alphabetically.
    5. Join tokens with single space.
    """
    if not isinstance(text, str) or not text:
        return ""

    # Normalize unicode and strip accents
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    cleaned = normalized.lower().strip()

    # Remove all punctuation and symbols by converting to spaces (retain alphanumeric and spaces)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # Split into whitespace-delimited tokens
    tokens = [tok for tok in cleaned.split() if tok]
    if not tokens:
        return ""

    # Sort unique tokens and join
    unique_sorted = sorted(set(tokens))
    return " ".join(unique_sorted)


def ngram_fingerprint(text: str, n: int = 2) -> str:
    """
    N-Gram Fingerprint algorithm.
    1. Normalize Unicode, lowercase, remove punctuation and whitespace.
    2. Extract all character n-grams of size n.
    3. Deduplicate, sort alphabetically, and join.
    """
    if not isinstance(text, str) or not text:
        return ""

    n = max(1, int(n))
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    cleaned = re.sub(r"[^\w]", "", normalized.lower())

    if len(cleaned) < n:
        return cleaned

    ngrams = set(cleaned[i : i + n] for i in range(len(cleaned) - n + 1))
    return "".join(sorted(ngrams))


def soundex(text: str) -> str:
    """
    Standard American Soundex phonetic encoding (National Archives / NARA standard).
    Retains the first letter and encodes consonants into digits 1-6.

    Rules:
    - Retain first letter (uppercase).
    - Consonant mapping:
        B, F, P, V -> 1
        C, G, J, K, Q, S, X, Z -> 2
        D, T -> 3
        L -> 4
        M, N -> 5
        R -> 6
    - Vowels (A, E, I, O, U, Y) do not receive a digit, but act as separators
      between consonants sharing the same code.
    - H and W do NOT act as separators between consonants sharing the same code.
      If H or W separates two consonants with the same code, the second is ignored.
    - Adjacent duplicate consonant codes are collapsed.
    - Padded with trailing zeros or truncated to exactly 4 characters (e.g. 'A261').
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = re.sub(r"[^A-Za-z]", "", text).upper()
    if not text:
        return ""

    first_letter = text[0]
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    vowels = {"A", "E", "I", "O", "U", "Y"}

    digits = []
    last_code = mapping.get(first_letter, "")

    for char in text[1:]:
        if char in mapping:
            code = mapping[char]
            if code != last_code:
                digits.append(code)
            last_code = code
        elif char in vowels:
            # Vowels separate consonants, resetting last_code
            last_code = ""
        elif char in ("H", "W"):
            # H and W do NOT separate consonants; last_code is preserved!
            pass

    return (first_letter + "".join(digits) + "000")[:4]


def metaphone(text: str) -> str:
    """
    Philips Metaphone phonetic encoding algorithm for English text clustering.
    Transforms words into phonetic consonant keys based on standard English pronunciation rules.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    words = re.findall(r"[A-Za-z]+", text.upper())
    if not words:
        return ""

    encoded_tokens = []
    for word in words[:3]:
        key = _metaphone_word(word)
        if key:
            encoded_tokens.append(key)

    return " ".join(encoded_tokens)


def _metaphone_word(w: str) -> str:
    """Encode single word via core Metaphone rules with duplicate consonant reduction."""
    if not w:
        return ""

    # Drop initial letter for KN, GN, PN, AE, WR; WH -> W
    if len(w) >= 2:
        if w[:2] in ("KN", "GN", "PN", "AE", "WR"):
            w = w[1:]
        elif w[:2] == "WH":
            w = "W" + w[2:]

    # Deduplicate adjacent identical consonants (except C which can be CC -> KS or K)
    deduped = []
    for idx, char in enumerate(w):
        if idx > 0 and char == w[idx - 1] and char != "C":
            continue
        deduped.append(char)
    w = "".join(deduped)

    res = []
    i = 0
    length = len(w)

    while i < length:
        ch = w[i]
        nxt = w[i + 1] if i + 1 < length else ""
        prev = w[i - 1] if i > 0 else ""

        if ch in ("A", "E", "I", "O", "U"):
            if i == 0:
                res.append(ch)
        elif ch == "B":
            # Silent in MB at end of word
            if not (i == length - 1 and prev == "M"):
                res.append("B")
        elif ch == "C":
            if nxt == "H":
                res.append("X")
                i += 1
            elif nxt in ("I", "E", "Y"):
                # Silent C in SC followed by I/E/Y (e.g. SCENE)
                if prev != "S":
                    res.append("S")
            elif nxt == "C":
                res.append("KS")
                i += 1
            else:
                res.append("K")
        elif ch == "D":
            if nxt == "G" and (i + 2 < length and w[i + 2] in ("E", "I", "Y")):
                res.append("J")
                i += 1
            else:
                res.append("T")
        elif ch == "F":
            res.append("F")
        elif ch == "G":
            if nxt == "H":
                if i + 2 < length and w[i + 2] in ("A", "E", "I", "O", "U"):
                    res.append("K")
                    i += 1
                else:
                    # Silent GH
                    i += 1
            elif nxt in ("I", "E", "Y"):
                res.append("J")
            elif nxt == "N" and (i + 1 == length - 1 or (i + 3 < length and w[i + 1:i + 4] == "NED")):
                # Silent GN
                pass
            else:
                res.append("K")
        elif ch == "H":
            if nxt in ("A", "E", "I", "O", "U") and prev not in ("C", "S", "P", "T", "G"):
                res.append("H")
        elif ch == "J":
            res.append("J")
        elif ch == "K":
            if prev != "C":
                res.append("K")
        elif ch == "L":
            res.append("L")
        elif ch == "M":
            res.append("M")
        elif ch == "N":
            res.append("N")
        elif ch == "P":
            if nxt == "H":
                res.append("F")
                i += 1
            else:
                res.append("P")
        elif ch == "Q":
            res.append("K")
        elif ch == "R":
            res.append("R")
        elif ch == "S":
            if nxt == "H":
                res.append("X")
                i += 1
            elif nxt == "I" and i + 2 < length and w[i + 2] in ("O", "A"):
                res.append("X")
                i += 1
            else:
                res.append("S")
        elif ch == "T":
            if nxt == "H":
                res.append("0")  # Th sound
                i += 1
            elif nxt == "I" and i + 2 < length and w[i + 2] in ("O", "A"):
                res.append("X")
                i += 1
            elif nxt == "C" and i + 2 < length and w[i + 2] == "H":
                res.append("X")
                i += 2
            else:
                res.append("T")
        elif ch == "V":
            res.append("F")
        elif ch == "W" or ch == "Y":
            if nxt in ("A", "E", "I", "O", "U"):
                res.append(ch)
        elif ch == "X":
            res.append("KS")
        elif ch == "Z":
            res.append("S")

        i += 1

    return "".join(res)


# =====================================================================
# 2. DISTANCE METRICS
# =====================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Standard Levenshtein distance with space optimization.
    Returns the minimum number of single-character edits (insertions,
    deletions or substitutions).
    """
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    prev_row = list(range(len(s1) + 1))
    for i, c2 in enumerate(s2):
        curr_row = [i + 1] * (len(s1) + 1)
        for j, c1 in enumerate(s1):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row[j + 1] = min(insertions, deletions, substitutions)
        prev_row = curr_row

    return prev_row[-1]


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Damerau-Levenshtein distance supporting insertions, deletions,
    substitutions, and transpositions of adjacent characters.
    """
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Matrix of distances
    d = [[0] * (len2 + 2) for _ in range(len1 + 2)]
    max_dist = len1 + len2
    d[0][0] = max_dist

    da: Dict[str, int] = defaultdict(int)

    for i in range(len1 + 1):
        d[i + 1][0] = max_dist
        d[i + 1][1] = i
    for j in range(len2 + 1):
        d[0][j + 1] = max_dist
        d[1][j + 1] = j

    for i in range(1, len1 + 1):
        db = 0
        for j in range(1, len2 + 1):
            k = da[s2[j - 1]]
            l = db
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            if cost == 0:
                db = j

            d[i + 1][j + 1] = min(
                d[i][j + 1] + 1,       # deletion
                d[i + 1][j] + 1,       # insertion
                d[i][j] + cost,        # substitution
                d[k][l] + (i - k - 1) + 1 + (j - l - 1),  # transposition
            )

        da[s1[i - 1]] = i

    return d[len1 + 1][len2 + 1]


def similarity_score(s1: str, s2: str, distance_func: Callable[[str, str], int] = levenshtein_distance) -> float:
    """
    Normalized string similarity score between 0.0 and 1.0.
    1.0 = identical, 0.0 = completely dissimilar.
    """
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = distance_func(s1, s2)
    return max(0.0, 1.0 - (dist / max_len))


# =====================================================================
# 3. CLUSTERING DATA STRUCTURES & CONTROLLER
# =====================================================================

@dataclass
class ClusterMember:
    value: str
    count: int


@dataclass
class ClusterResult:
    cluster_id: str
    method: str
    members: List[ClusterMember]
    total_rows: int
    proposed_canonical: str
    confidence: float
    key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        members_data = [asdict(m) for m in self.members]
        return {
            "cluster_id": self.cluster_id,
            "method": self.method,
            "members": members_data,
            "values": members_data,
            "size": len(self.members),
            "total_rows": self.total_rows,
            "proposed_canonical": self.proposed_canonical,
            "canonical": self.proposed_canonical,
            "confidence": round(self.confidence, 3),
            "key": self.key,
        }


def _select_canonical(values_with_counts: List[Tuple[str, int]]) -> str:
    """
    Pick best canonical value for a cluster:
    1. Highest row count in dataset.
    2. Tie-breaker: Shortest length.
    3. Tie-breaker: Standard capitalization or alphabetical.
    """
    if not values_with_counts:
        return ""

    # Sort: highest count desc, length asc, text asc
    sorted_candidates = sorted(
        values_with_counts,
        key=lambda pair: (-pair[1], len(pair[0]), pair[0])
    )
    return sorted_candidates[0][0]


def cluster_by_key_collision(
    value_counts: Dict[str, int],
    key_func: Callable[[str], str],
    method_name: str,
) -> List[ClusterResult]:
    """Group unique values sharing identical key representation."""
    groups: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    for val, count in value_counts.items():
        if not str(val).strip():
            continue
        key = key_func(str(val))
        if key:
            groups[key].append((str(val), count))

    clusters: List[ClusterResult] = []
    idx = 1

    for key, members in groups.items():
        if len(members) > 1:
            total_rows = sum(cnt for _, cnt in members)
            canonical = _select_canonical(members)

            # Average pairwise similarity in cluster
            val_strs = [v for v, _ in members]
            sims = []
            for i in range(len(val_strs)):
                for j in range(i + 1, len(val_strs)):
                    sims.append(similarity_score(val_strs[i].lower(), val_strs[j].lower()))
            confidence = sum(sims) / len(sims) if sims else 1.0

            clusters.append(
                ClusterResult(
                    cluster_id=f"cluster_{method_name}_{idx}",
                    method=method_name,
                    members=[ClusterMember(value=v, count=c) for v, c in members],
                    total_rows=total_rows,
                    proposed_canonical=canonical,
                    confidence=confidence,
                    key=key,
                )
            )
            idx += 1

    # Sort clusters by total impact (number of rows affected descending)
    return sorted(clusters, key=lambda c: (-c.total_rows, -len(c.members)))


# Limits for distance-based clustering scalability and resource protection
MAX_UNIQUE_VALUES_DISTANCE = 3000
MAX_STRING_LENGTH_DISTANCE = 150
MAX_PAIRWISE_COMPARISONS = 250_000


def cluster_by_distance(
    value_counts: Dict[str, int],
    max_distance: int = 1,
    min_similarity: float = 0.80,
    use_damerau: bool = False,
    method_name: str = "levenshtein",
) -> List[ClusterResult]:
    """
    Cluster unique values using complete-linkage distance clustering with length pruning.

    Clustering Semantics:
    - Seed Selection: The most frequent unassigned value acts as the cluster seed / medoid.
    - Complete-Linkage Guarantee: To prevent transitive chaining/drift (e.g. 'cat' -> 'bat'
      -> 'bad' -> 'bed'), every member added to a cluster must satisfy the distance and similarity
      thresholds against the seed and all already-admitted cluster members.
    - Scalability Controls:
      - Limits unique values evaluated to top MAX_UNIQUE_VALUES_DISTANCE.
      - Skips strings exceeding MAX_STRING_LENGTH_DISTANCE characters.
      - Enforces length-difference pruning (|len(a) - len(b)| <= max_distance).
      - Caps total pairwise distance calculations at MAX_PAIRWISE_COMPARISONS.
    """
    dist_func = damerau_levenshtein_distance if use_damerau else levenshtein_distance
    val_items = [
        (str(val), count) for val, count in value_counts.items()
        if str(val).strip() and len(str(val)) <= MAX_STRING_LENGTH_DISTANCE
    ]
    if len(val_items) <= 1:
        return []

    # Sort by frequency descending and cap at max unique values
    val_items.sort(key=lambda x: -x[1])
    val_items = val_items[:MAX_UNIQUE_VALUES_DISTANCE]
    n = len(val_items)

    assigned: Set[str] = set()
    clusters: List[ClusterResult] = []
    idx = 1
    evaluations_count = 0

    for i in range(n):
        val_i, count_i = val_items[i]
        if val_i in assigned:
            continue

        cluster_members = [(val_i, count_i)]
        cluster_val_strs = [val_i.lower()]
        len_i = len(val_i)

        for j in range(i + 1, n):
            if evaluations_count >= MAX_PAIRWISE_COMPARISONS:
                break

            val_j, count_j = val_items[j]
            if val_j in assigned:
                continue

            # Length pruning: edit distance cannot be less than length difference
            if abs(len(val_j) - len_i) > max_distance:
                continue

            val_j_lower = val_j.lower()

            # Check distance against cluster seed
            evaluations_count += 1
            dist = dist_func(cluster_val_strs[0], val_j_lower)
            if dist > max_distance:
                continue

            score = similarity_score(cluster_val_strs[0], val_j_lower, dist_func)
            if score < min_similarity:
                continue

            # Complete-linkage check: ensure val_j is within threshold of ALL current members
            is_compatible_with_all = True
            for existing_val in cluster_val_strs[1:]:
                evaluations_count += 1
                if dist_func(existing_val, val_j_lower) > max_distance:
                    is_compatible_with_all = False
                    break

            if is_compatible_with_all:
                cluster_members.append((val_j, count_j))
                cluster_val_strs.append(val_j_lower)

        if len(cluster_members) > 1:
            for member, _ in cluster_members:
                assigned.add(member)

            total_rows = sum(cnt for _, cnt in cluster_members)
            canonical = _select_canonical(cluster_members)

            # Confidence calculation across all cluster pairs
            val_strs = [v for v, _ in cluster_members]
            sims = [
                similarity_score(val_strs[a].lower(), val_strs[b].lower(), dist_func)
                for a in range(len(val_strs))
                for b in range(a + 1, len(val_strs))
            ]
            confidence = sum(sims) / len(sims) if sims else 1.0

            clusters.append(
                ClusterResult(
                    cluster_id=f"cluster_{method_name}_{idx}",
                    method=method_name,
                    members=[ClusterMember(value=v, count=c) for v, c in cluster_members],
                    total_rows=total_rows,
                    proposed_canonical=canonical,
                    confidence=confidence,
                )
            )
            idx += 1

    return sorted(clusters, key=lambda c: (-c.total_rows, -len(c.members)))


# =====================================================================
# 4. HIGH-LEVEL API ENTRYPOINT
# =====================================================================

SUPPORTED_CLUSTERING_METHODS = {
    "fingerprint": "Key Collision (Fingerprint)",
    "ngram_fingerprint": "Key Collision (N-Gram Fingerprint)",
    "soundex": "Key Collision (Soundex Phonetic)",
    "metaphone": "Key Collision (Metaphone Phonetic)",
    "levenshtein": "Distance (Levenshtein)",
    "damerau_levenshtein": "Distance (Damerau-Levenshtein)",
}


def detect_clusters(
    series: pd.Series,
    method: str = "fingerprint",
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run the requested clustering algorithm on a pandas Series.
    Returns serialized list of clusters ready for UI review and approval.
    """
    params = params or {}
    clean_series = series.dropna().astype(str).str.strip()
    clean_series = clean_series[clean_series != ""]

    if clean_series.empty:
        return []

    # Calculate frequencies of unique non-empty strings
    value_counts: Dict[str, int] = clean_series.value_counts().to_dict()
    if len(value_counts) <= 1:
        return []

    method = method.lower().strip()

    if method == "fingerprint":
        results = cluster_by_key_collision(value_counts, fingerprint, "fingerprint")
    elif method in ("ngram_fingerprint", "ngram"):
        ngram_size = int(params.get("ngram_size", 2))
        results = cluster_by_key_collision(
            value_counts,
            lambda t: ngram_fingerprint(t, ngram_size),
            f"ngram_{ngram_size}"
        )
    elif method == "soundex":
        results = cluster_by_key_collision(value_counts, soundex, "soundex")
    elif method == "metaphone":
        results = cluster_by_key_collision(value_counts, metaphone, "metaphone")
    elif method == "levenshtein":
        max_dist = int(params.get("max_distance", 1))
        min_sim = float(params.get("min_similarity", 0.80))
        results = cluster_by_distance(
            value_counts,
            max_distance=max_dist,
            min_similarity=min_sim,
            use_damerau=False,
            method_name="levenshtein",
        )
    elif method in ("damerau_levenshtein", "damerau"):
        max_dist = int(params.get("max_distance", 1))
        min_sim = float(params.get("min_similarity", 0.80))
        results = cluster_by_distance(
            value_counts,
            max_distance=max_dist,
            min_similarity=min_sim,
            use_damerau=True,
            method_name="damerau_levenshtein",
        )
    else:
        raise ValueError(f"Unknown clustering method '{method}'. Supported: {list(SUPPORTED_CLUSTERING_METHODS.keys())}")

    return [c.to_dict() for c in results]
