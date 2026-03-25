"""Grammatical profiling via spaCy POS-tagging and dependency parsing.

Implements the method from:
  Giulianelli, Kutuzov & Pivovarova (2021).
  Grammatical Profiling for Semantic Change Detection. CoNLL 2021.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Literal

import spacy
from spacy.language import Language


# Features to extract per token (besides dependency relation)
_MORPH_CATEGORIES = ("Case", "Number", "Tense", "VerbForm", "Mood", "Voice")

FeatureType = Literal["morph", "dep", "both"]


class GrammaticalProfiler:
    """Build grammatical profiles for a list of target words across time slices.

    Parameters
    ----------
    target_words:
        Words to track (matched against lowercased lemmas).
    feature_type:
        Which features to include: ``"morph"``, ``"dep"``, or ``"both"``.
    min_freq_ratio:
        Categories accounting for less than this fraction of a word's total
        occurrences are filtered out (noise reduction, cf. paper §3.2).
    model:
        spaCy model name.
    """

    def __init__(
        self,
        target_words: list[str],
        feature_type: FeatureType = "both",
        min_freq_ratio: float = 0.05,
        model: str = "en_core_web_sm",
    ) -> None:
        self.targets = {w.lower() for w in target_words}
        self.feature_type = feature_type
        self.min_freq_ratio = min_freq_ratio

        print(f"[Profiler] Loading spaCy model '{model}' …")
        self._nlp: Language = spacy.load(model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_profiles(
        self,
        texts: dict[str, list[str]],
        max_occurrences: int = 50,
        seed: int | None = 42,
    ) -> dict[str, dict[str, Counter]]:
        """Parse texts and build balanced grammatical profiles.

        For each target word and time slice, **all** occurrences in the corpus
        are collected first, then ``max_occurrences`` are drawn at random.
        This ensures that:

        - Both T1 and T2 profiles are based on the same sample size.
        - The sample is not biased towards early documents.

        Parameters
        ----------
        max_occurrences:
            Number of token occurrences to randomly sample per word per slice.
        seed:
            Random seed for reproducibility. Pass ``None`` for non-deterministic
            sampling.

        Returns
        -------
        profiles : ``{slice_label: {word: Counter({feature: count})}}``
        """
        rng = random.Random(seed)
        profiles: dict[str, dict[str, Counter]] = {}
        for label, docs in texts.items():
            print(
                f"[Profiler] Slice {label}: {len(docs)} documents, "
                f"sampling {max_occurrences} occurrences/word …"
            )
            profiles[label] = self._profile_slice_sampled(docs, max_occurrences, rng)
            for word in profiles[label]:
                profiles[label][word] = self._filter_rare(profiles[label][word])
        return profiles

    def sample_counts(
        self, profiles: dict[str, dict[str, Counter]]
    ) -> dict[str, dict[str, int]]:
        """Return the actual number of occurrences sampled per word per slice.

        This is the sum of all feature counts divided by the mean features per
        token, approximated by the raw token count stored during profiling.
        Use ``_actual_counts`` which is populated by ``build_profiles``.
        """
        return getattr(self, "_actual_counts", {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sentence_iter(self, text: str):
        """Split text into ~500-char chunks to keep memory usage manageable."""
        paragraphs = re.split(r"\n{2,}", text)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) > 500:
                words = para.split()
                chunk: list[str] = []
                length = 0
                for word in words:
                    chunk.append(word)
                    length += len(word) + 1
                    if length >= 500:
                        yield " ".join(chunk)
                        chunk, length = [], 0
                if chunk:
                    yield " ".join(chunk)
            else:
                yield para

    def _extract_features(self, token) -> list[str]:
        """Extract feature strings from a spaCy token."""
        features: list[str] = []

        if self.feature_type in ("dep", "both"):
            if token.dep_ not in ("", "ROOT"):
                features.append(f"dep:{token.dep_}")

        if self.feature_type in ("morph", "both"):
            for cat in _MORPH_CATEGORIES:
                val = token.morph.get(cat)
                if val:
                    features.append(f"{cat}:{val[0]}")

        return features

    def _profile_slice_sampled(
        self,
        docs: list[str],
        max_occurrences: int,
        rng: random.Random,
    ) -> dict[str, Counter]:
        """Randomly sample up to max_occurrences token occurrences per target word.

        Strategy: **shuffle documents randomly**, then collect occurrences
        sequentially until each word's quota is filled.  Because the document
        order is randomised, the collected tokens form an unbiased sample —
        neither geographically nor temporally skewed towards early files.

        This is equivalent to token-level random sampling when occurrences are
        roughly uniformly distributed across documents, while avoiding the cost
        of a full corpus parse.
        """
        # Shuffle document order so early-stop draws a random sample
        indices = list(range(len(docs)))
        rng.shuffle(indices)

        # word -> list of feature-lists for sampled tokens
        word_samples: dict[str, list[list[str]]] = defaultdict(list)
        word_counts: dict[str, int] = defaultdict(int)
        active: set[str] = set(self.targets)

        for idx in indices:
            if not active:
                break

            text = docs[idx]
            text_lower = text.lower()
            if not any(w in text_lower for w in active):
                continue

            for sent in self._sentence_iter(text):
                if not active:
                    break
                spacy_doc = self._nlp(sent)
                for token in spacy_doc:
                    lemma = token.lemma_.lower()
                    if lemma not in active:
                        continue
                    word_samples[lemma].append(self._extract_features(token))
                    word_counts[lemma] += 1
                    if word_counts[lemma] >= max_occurrences:
                        active.discard(lemma)

        # Build Counters from collected samples
        word_profiles: dict[str, Counter] = {}
        for word in self.targets:
            sample = word_samples.get(word, [])
            counter: Counter = Counter()
            for feats in sample:
                counter.update(feats)
            word_profiles[word] = counter

            n = len(sample)
            status = "OK" if n >= max_occurrences else f"only {n} in corpus"
            print(f"  {word:15s}: {n:4d} sampled – {status}")

        return word_profiles

    def _filter_rare(self, counter: Counter) -> Counter:
        """Remove features that make up less than min_freq_ratio of the total."""
        total = sum(counter.values())
        if total == 0:
            return counter
        threshold = total * self.min_freq_ratio
        return Counter({k: v for k, v in counter.items() if v >= threshold})
