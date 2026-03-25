"""Corpus loading and time-slice partitioning."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parents[2] / "data"

# Source directory names as found on disk
_SOURCE_DIRS: dict[str, str] = {
    "london gazette": "gazette",
    "project gutenberg": "project_gutenberg",
    "justia": "justia",
    "dracor": "dracor",
    "corpus": "corpus",
}


def _resolve_path(row: pd.Series) -> Path | None:
    """Resolve the actual path of a text file from a metadata row."""
    filename: str = row["filename"]
    source: str = str(row.get("source", "")).strip().lower()

    # Try source-based subdirectory first
    for key, subdir in _SOURCE_DIRS.items():
        if key in source:
            candidate = DATA_DIR / subdir / filename
            if candidate.exists():
                return candidate

    # Fall back: search all subdirectories
    for candidate in DATA_DIR.rglob(filename):
        return candidate

    return None


class Corpus:
    """Loads texts from the data directory and splits them into time slices.

    Parameters
    ----------
    t1_range:
        (start_year, end_year) inclusive for the first time slice.
    t2_range:
        (start_year, end_year) inclusive for the second time slice.
    metadata_path:
        Path to metadata.csv. Defaults to ``data/metadata.csv``.
    max_files_per_slice:
        Cap the number of files loaded per slice (useful for quick tests).
    """

    DEFAULT_SLICES = {
        "T1": (1800, 1860),
        "T2": (1861, 1920),
    }

    def __init__(
        self,
        t1_range: tuple[int, int] = (1800, 1860),
        t2_range: tuple[int, int] = (1861, 1920),
        metadata_path: Path | None = None,
        max_files_per_slice: int | None = None,
    ) -> None:
        self.slices = {"T1": t1_range, "T2": t2_range}
        self.max_files = max_files_per_slice

        meta_path = metadata_path or DATA_DIR / "metadata.csv"
        self._meta = pd.read_csv(meta_path)
        self._meta["year"] = pd.to_numeric(self._meta["year"], errors="coerce")
        self._meta = self._meta.dropna(subset=["year"])
        self._meta["year"] = self._meta["year"].astype(int)

        self._texts: dict[str, list[str]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict[str, list[str]]:
        """Return ``{slice_label: [text, ...]}`` dict (cached after first call)."""
        if self._texts is not None:
            return self._texts

        self._texts = {}
        for label, (start, end) in self.slices.items():
            subset = self._meta[
                (self._meta["year"] >= start) & (self._meta["year"] <= end)
            ]
            if self.max_files is not None:
                subset = subset.head(self.max_files)

            texts: list[str] = []
            missing = 0
            for _, row in subset.iterrows():
                path = _resolve_path(row)
                if path is None:
                    missing += 1
                    continue
                try:
                    texts.append(path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    missing += 1

            print(
                f"[Corpus] {label} ({start}–{end}): "
                f"loaded {len(texts)} files, {missing} not found."
            )
            self._texts[label] = texts

        return self._texts

    def stats(self) -> pd.DataFrame:
        """Return a summary DataFrame with file counts per year."""
        rows = []
        for label, (start, end) in self.slices.items():
            subset = self._meta[
                (self._meta["year"] >= start) & (self._meta["year"] <= end)
            ]
            for year, grp in subset.groupby("year"):
                rows.append(
                    {"slice": label, "year": year, "n_files": len(grp)}
                )
        return pd.DataFrame(rows)

    def token_counts(self) -> dict[str, int]:
        """Rough token count per slice (whitespace split)."""
        texts = self.load()
        return {
            label: sum(len(re.split(r"\s+", t)) for t in txts)
            for label, txts in texts.items()
        }
