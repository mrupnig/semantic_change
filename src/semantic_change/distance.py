"""Cosine-distance computation and word ranking for grammatical profiles."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _to_vector(counter: Counter, keys: list[str]) -> np.ndarray:
    return np.array([counter.get(k, 0) for k in keys], dtype=float)


def profile_distance(
    profile_t1: Counter,
    profile_t2: Counter,
    normalize: bool = True,
) -> float:
    """Cosine distance between two grammatical profiles (0 = identical, 1 = orthogonal).

    Parameters
    ----------
    normalize:
        If True, convert raw counts to relative frequencies before computing
        the distance (recommended so that corpus-size differences don't distort
        the result).
    """
    all_keys = sorted(set(profile_t1) | set(profile_t2))
    if not all_keys:
        return 0.0

    v1 = _to_vector(profile_t1, all_keys)
    v2 = _to_vector(profile_t2, all_keys)

    if normalize:
        s1, s2 = v1.sum(), v2.sum()
        if s1 > 0:
            v1 = v1 / s1
        if s2 > 0:
            v2 = v2 / s2

    sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
    return float(1 - sim)


def profile_distance_by_category(
    profile_t1: Counter,
    profile_t2: Counter,
    normalize: bool = True,
) -> dict[str, float]:
    """Compute cosine distance separately for each feature category (e.g. dep, Tense).

    Returns a dict ``{category: distance}``.
    """
    # Group keys by category prefix
    all_keys = set(profile_t1) | set(profile_t2)
    categories: dict[str, list[str]] = {}
    for key in all_keys:
        prefix = key.split(":")[0] if ":" in key else key
        categories.setdefault(prefix, []).append(key)

    result: dict[str, float] = {}
    for cat, keys in categories.items():
        c1 = Counter({k: profile_t1.get(k, 0) for k in keys})
        c2 = Counter({k: profile_t2.get(k, 0) for k in keys})
        result[cat] = profile_distance(c1, c2, normalize=normalize)

    return result


def rank_words(
    profiles: dict[str, dict[str, Counter]],
    slice_labels: tuple[str, str] = ("T1", "T2"),
    normalize: bool = True,
) -> pd.DataFrame:
    """Rank all target words by their cosine distance between two slices.

    Parameters
    ----------
    profiles:
        Output of ``GrammaticalProfiler.build_profiles()``.
    slice_labels:
        Which two slices to compare.

    Returns
    -------
    DataFrame with columns ``word``, ``distance``, and one column per feature category.
    """
    label1, label2 = slice_labels
    p1 = profiles.get(label1, {})
    p2 = profiles.get(label2, {})

    words = sorted(set(p1) | set(p2))
    rows = []
    for word in words:
        c1 = p1.get(word, Counter())
        c2 = p2.get(word, Counter())
        dist = profile_distance(c1, c2, normalize=normalize)
        cat_dists = profile_distance_by_category(c1, c2, normalize=normalize)
        rows.append({"word": word, "distance": dist, **cat_dists})

    df = pd.DataFrame(rows).sort_values("distance", ascending=False).reset_index(drop=True)
    return df
