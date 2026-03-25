"""CLI entry point for the semantic change analysis pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from semantic_change import (
    Corpus,
    GrammaticalProfiler,
    rank_words,
    plot_ranking,
    plot_heatmap,
)

TARGET_WORDS = [
    "steam", "electric", "race", "progress", "science", "machine",
    "labour", "industry", "telegraph", "evolution",
    "empire", "reform", "capital", "engine",
]


def main(max_occurrences: int = 50) -> None:
    print("=== Semantic Change – Grammatical Profiling ===\n")

    corpus = Corpus()
    texts = corpus.load()
    print("Token counts:", corpus.token_counts(), "\n")

    profiler = GrammaticalProfiler(target_words=TARGET_WORDS, feature_type="both")
    profiles = profiler.build_profiles(texts, max_occurrences=max_occurrences)

    ranking = rank_words(profiles)
    print("\nRanking (semantischer Wandel T1 → T2):\n")
    print(ranking[["word", "distance"]].to_string(index=False))

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    fig = plot_ranking(ranking)
    fig.savefig(out_dir / "ranking.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot gespeichert: {out_dir / 'ranking.png'}")

    cat_cols = [c for c in ranking.columns if c not in ("word", "distance")]
    if cat_cols:
        fig2 = plot_heatmap(ranking)
        fig2.savefig(out_dir / "heatmap.png", dpi=150, bbox_inches="tight")
        print(f"Heatmap gespeichert: {out_dir / 'heatmap.png'}")

    ranking.to_csv(out_dir / "ranking.csv", index=False)
    print(f"Tabelle gespeichert: {out_dir / 'ranking.csv'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grammatical Profiling Pipeline")
    parser.add_argument(
        "--max-occurrences",
        type=int,
        default=50,
        help="Max token occurrences per word per slice (default: 50)",
    )
    args = parser.parse_args()
    main(max_occurrences=args.max_occurrences)
