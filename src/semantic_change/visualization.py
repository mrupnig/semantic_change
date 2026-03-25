"""Visualization helpers for grammatical profiling results."""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_profiles(
    word: str,
    profile_t1: Counter,
    profile_t2: Counter,
    label1: str = "T1",
    label2: str = "T2",
    top_n: int = 20,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
) -> plt.Figure:
    """Bar chart comparing the top-N grammatical features of a word between two slices.

    Analogous to Fig. 1 in Giulianelli et al. (2021).
    """
    all_keys = sorted(set(profile_t1) | set(profile_t2))

    # Normalize to relative frequencies
    def normalize(counter: Counter) -> dict[str, float]:
        total = sum(counter.values()) or 1
        return {k: counter.get(k, 0) / total for k in all_keys}

    n1 = normalize(profile_t1)
    n2 = normalize(profile_t2)

    # Select top_n features by combined frequency
    combined = {k: n1.get(k, 0) + n2.get(k, 0) for k in all_keys}
    top_keys = sorted(combined, key=combined.get, reverse=True)[:top_n]

    x = np.arange(len(top_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, [n1[k] for k in top_keys], width, label=label1, alpha=0.8)
    ax.bar(x + width / 2, [n2[k] for k in top_keys], width, label=label2, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(top_keys, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Relative frequency")
    ax.set_title(title or f"Grammatical profile: '{word}' ({label1} vs {label2})")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_heatmap(
    ranking_df: pd.DataFrame,
    category_cols: list[str] | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Cosine distances by feature category",
) -> plt.Figure:
    """Heatmap of per-category cosine distances across all target words.

    Parameters
    ----------
    ranking_df:
        Output of ``rank_words()``.
    category_cols:
        Column names to include. If None, all columns except ``word`` and
        ``distance`` are used.
    """
    if category_cols is None:
        category_cols = [c for c in ranking_df.columns if c not in ("word", "distance")]

    if not category_cols:
        raise ValueError("No category columns found. Pass explicit category_cols.")

    heat_data = ranking_df.set_index("word")[category_cols].fillna(0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        heat_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Feature category")
    ax.set_ylabel("Word")
    fig.tight_layout()
    return fig


def plot_ranking(
    ranking_df: pd.DataFrame,
    col: str = "distance",
    figsize: tuple[float, float] = (8, 5),
    title: str | None = None,
    color: str = "steelblue",
) -> plt.Figure:
    """Horizontal bar chart of words ranked by semantic-change score."""
    df = ranking_df.sort_values(col, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df["word"], df[col], color=color, alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("Cosine distance (grammatical profile)")
    ax.set_title(title or f"Word ranking by '{col}'")
    ax.set_xlim(0, min(1.05, df[col].max() * 1.2 + 0.05))
    fig.tight_layout()
    return fig


def plot_feature_diff(
    word: str,
    profile_t1: Counter,
    profile_t2: Counter,
    label1: str = "T1",
    label2: str = "T2",
    top_n: int = 15,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Diverging bar chart showing which features increased or decreased."""
    all_keys = sorted(set(profile_t1) | set(profile_t2))

    def normalize(counter: Counter) -> dict[str, float]:
        total = sum(counter.values()) or 1
        return {k: counter.get(k, 0) / total for k in all_keys}

    n1 = normalize(profile_t1)
    n2 = normalize(profile_t2)
    diffs = {k: n2[k] - n1[k] for k in all_keys}

    # Take top_n by absolute change
    top_keys = sorted(diffs, key=lambda k: abs(diffs[k]), reverse=True)[:top_n]
    vals = [diffs[k] for k in top_keys]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top_keys[::-1], vals[::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Δ relative frequency ({label2} − {label1})")
    ax.set_title(f"Feature shift for '{word}'")
    fig.tight_layout()
    return fig
