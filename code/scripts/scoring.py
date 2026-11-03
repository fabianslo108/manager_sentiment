"""Turning model token probabilities into report-level sentiment.

Pure Python and torch-free, so the scoring rules can be unit-tested on a laptop
without a GPU stack. The tensor work lives in :mod:`sentiment_extraction`.

A readout scores as ``positive``, ``negative`` and their difference
``polarity``.

"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from prompts import normalize_token

#: Columns produced per report, in output order.
SCORE_COLUMNS = (
    "positive",
    "negative",
    "polarity",
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

    return {
        "positive": positive,
        "negative": negative,
        "polarity": positive - negative,
    }


def aggregate_chunk_scores(chunk_scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Combine chunk-level scores into one report-level score.

    ``positive`` and ``negative`` are averaged across chunks and ``polarity`` is
    then recomputed from those averages, so the row stays internally consistent.
    """
    if not chunk_scores:
        raise ValueError("no chunk scores to aggregate")

    n = len(chunk_scores)
    positive = sum(s["positive"] for s in chunk_scores) / n
    negative = sum(s["negative"] for s in chunk_scores) / n

    return {
        "positive": positive,
        "negative": negative,
        "polarity": positive - negative,
    }
