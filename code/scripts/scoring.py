"""Turning model token probabilities into report-level sentiment.

Pure Python and torch-free, so the scoring rules can be unit-tested on a laptop
without a GPU stack. The tensor work lives in :mod:`sentiment_extraction`.

Two details differ deliberately from the earlier project code:

* Token probabilities arrive as a *list of pairs*, not a dict. Several raw
  tokens normalize to the same word (``" good"``, ``"good"``, ``"good,"``), and
  their probability mass must be **summed**; a dict keyed by the normalized word
  would keep only the last one.
* Alongside ``polarity = positive - negative`` we keep ``matched_mass`` and
  ``polarity_norm = polarity / matched_mass``. Raw polarity mixes *direction*
  with *how much* of the nucleus the verbalizer captured; the normalized version
  separates the two, and ``matched_mass`` doubles as a coverage diagnostic.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from prompts import normalize_token

#: Columns produced per report, in output order.
SCORE_COLUMNS = (
    "positive",
    "negative",
    "polarity",
    "matched_mass",
    "polarity_norm",
)


def nucleus_cutoff(probs_desc: Sequence[float], top_p: float) -> int:
    """How many tokens the nucleus keeps, given probabilities in descending order.

    The smallest prefix whose cumulative mass reaches ``top_p`` (always at least
    one token). Reference implementation for the vectorized version used on GPU.
    """
    if top_p is None or top_p >= 1.0:
        return len(probs_desc)
    total = 0.0
    for i, prob in enumerate(probs_desc):
        total += prob
        if total >= top_p:
            return i + 1
    return len(probs_desc)


def score_token_probs(
    token_probs: Iterable[tuple[str, float]],
    verbalizer: Mapping[str, frozenset[str]],
) -> dict[str, float]:
    """Score one model readout against a verbalizer.

    Parameters
    ----------
    token_probs:
        ``(raw_token, probability)`` pairs, e.g. the nucleus of the next-token
        distribution. Raw tokens are normalized here.
    verbalizer:
        Mapping with ``"positive"`` and ``"negative"`` word sets.
    """
    positive = 0.0
    negative = 0.0
    for token, prob in token_probs:
        word = normalize_token(token)
        if word in verbalizer["positive"]:
            positive += prob
        elif word in verbalizer["negative"]:
            negative += prob

    matched = positive + negative
    return {
        "positive": positive,
        "negative": negative,
        "polarity": positive - negative,
        "matched_mass": matched,
        "polarity_norm": (positive - negative) / matched if matched > 0 else math.nan,
    }


def aggregate_chunk_scores(chunk_scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Combine chunk-level scores into one report-level score.

    ``positive`` and ``negative`` are averaged across chunks; ``polarity`` and
    ``polarity_norm`` are then recomputed from those averages so the row stays
    internally consistent. ``polarity_chunk_std`` reports how much the chunks
    disagreed, which is a useful flag for long filings.
    """
    if not chunk_scores:
        raise ValueError("no chunk scores to aggregate")

    n = len(chunk_scores)
    positive = sum(s["positive"] for s in chunk_scores) / n
    negative = sum(s["negative"] for s in chunk_scores) / n
    matched = positive + negative

    polarities = [s["polarity"] for s in chunk_scores]
    if n > 1:
        mean_pol = sum(polarities) / n
        variance = sum((p - mean_pol) ** 2 for p in polarities) / (n - 1)
        chunk_std = math.sqrt(variance)
    else:
        chunk_std = 0.0

    return {
        "positive": positive,
        "negative": negative,
        "polarity": positive - negative,
        "matched_mass": matched,
        "polarity_norm": (positive - negative) / matched if matched > 0 else math.nan,
        "n_chunks": n,
        "polarity_chunk_std": chunk_std,
    }
