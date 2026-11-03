"""Tests for the parts of the pipeline that need no model.

Run with plain Python -- no GPU stack, no pytest required::

    python code/scripts/tests/test_scoring_rules.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import prompts  # noqa: E402
import reports  # noqa: E402
import scoring  # noqa: E402

FIXTURES = {
    "20130726_10-K_edgar_data_1750_0001047469-13-007797_item7.txt": "Revenues grew strongly. " * 40,
    "20140717_10-K_edgar_data_1750_0001047469-14-006243_item7.txt": "Margins declined sharply. " * 40,
    "20130726_10-K_edgar_data_1750_0001047469-13-007797_item1a.txt": "Risk factors text. " * 40,
    # Same filing as the first 10-K, stored the way the sample is: no _item7 suffix.
    "20130726_10-K_edgar_data_1750_0001047469-13-007797.txt": "Plain copy that must lose to item7. " * 40,
    "20150115_10-K_edgar_data_1750_0001047469-15-000111.txt": "Item 7 text stored without a suffix. " * 40,
    "20030414_10-Q_edgar_data_1750_0001104659-03-006403.txt": "Quarterly results were solid. " * 40,
    "20031014_10-Q_edgar_data_1750_0001104659-03-022660.txt": "short",
    "not_a_filing.txt": "ignore me",
}


def write_fixtures(root: Path) -> None:
    for name, body in FIXTURES.items():
        (root / name).write_text(body)


# ---------------------------------------------------------------------------
# file naming and discovery
# ---------------------------------------------------------------------------


def test_filename_round_trip() -> None:
    for section in (None, "item7", "item1a"):
        name = reports.report_filename("1750", "20130726", "0001047469-13-007797", "10-K", section)
        parsed = reports.parse_report_filename(name)
        assert parsed is not None, name
        assert parsed["CIK"] == "1750"
        assert parsed["FILING_DATE"] == "20130726"
        assert parsed["ACC_NUM"] == "0001047469-13-007797"
        assert parsed["FORM_TYPE"] == "10-K"
        assert parsed["SECTION"] == section
    assert reports.parse_report_filename("not_a_filing.txt") is None


def test_default_sections_per_form_type(root: Path) -> None:
    """10-Ks are Item 7 (suffix optional); 10-Qs are the whole document."""
    found = reports.discover_reports(root)
    ten_k = found[found["FORM_TYPE"] == "10-K"]
    ten_q = found[found["FORM_TYPE"] == "10-Q"]

    # The labeled extract wins when both names exist; a suffix-free 10-K is kept.
    assert set(ten_k["ACC_NUM"]) == {"0001047469-13-007797", "0001047469-14-006243", "0001047469-15-000111"}
    labeled = ten_k[ten_k["ACC_NUM"] == "0001047469-13-007797"]
    assert len(labeled) == 1
    assert labeled["SECTION"].iloc[0] == "item7"
    assert labeled["path"].iloc[0].endswith("_item7.txt")
    plain = ten_k[ten_k["ACC_NUM"] == "0001047469-15-000111"]
    assert plain["SECTION"].isna().all() or plain["SECTION"].iloc[0] is None

    assert ten_q["SECTION"].isna().all() or ten_q["SECTION"].iloc[0] is None
    assert len(found) == 5  # item1a, the duplicate plain 10-K, and the non-filing are excluded

    assert len(reports.discover_reports(root, "10-K", "item1a")) == 1
    assert len(reports.discover_reports(root, "10-K", None)) == 2  # plain names only
    assert len(reports.discover_reports(root, section=reports.SECTION_ANY)) == 7


def test_dataset_flags_bad_filings_instead_of_dropping(root: Path) -> None:
    found = reports.discover_reports(root)
    dataset = reports.ReportDataset(found, max_chars=200, min_chars=200)
    items = list(dataset)

    assert len(items) == len(found)
    too_short = [i for i in items if not i.ok]
    assert len(too_short) == 1
    assert "too_short" in too_short[0].error

    scored = [i for i in items if i.ok]
    assert all(i.n_chars_scored == 200 for i in scored), "max_chars must cap the scored text"
    assert all(i.n_chars > i.n_chars_scored for i in scored), "n_chars must stay the full length"
    assert all(len(i.key) == 4 for i in items), "filing key is (CIK, FILING_DATE, ACC_NUM, FORM_TYPE)"


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_prompt_endings() -> None:
    growth = prompts.GROWTH
    assert growth.build_prompt("text", "causal").endswith("appear to be")
    assert "[MASK]" in growth.build_prompt("text", "mlm", "[MASK]")

    try:
        growth.ending("mlm")
    except ValueError:
        pass
    else:
        raise AssertionError("an mlm ending without a mask token must be refused")

    try:
        prompts.GROWTH_LEGACY_CAUSAL.ending("mlm")
    except KeyError:
        pass
    else:
        raise AssertionError("a strategy without an mlm ending must be refused")


def test_token_normalization() -> None:
    assert prompts.normalize_token(" optimistic") == "optimistic"
    assert prompts.normalize_token("Optimistic") == "optimistic"
    assert prompts.normalize_token("modest:") == "modest"
    assert prompts.normalize_token(" lower,") == "lower"
    assert prompts.normalize_token("   ") == ""


def test_verbalizer_polarities_are_disjoint() -> None:
    for strategy in prompts.STRATEGIES.values():
        overlap = strategy.verbalizer["positive"] & strategy.verbalizer["negative"]
        assert not overlap, f"{strategy.name}: {sorted(overlap)}"


def test_risk_sign_convention() -> None:
    """Higher financial-risk polarity must mean *less* risk (good news)."""
    risk = prompts.FINANCIAL_RISK
    assert risk.higher_is_good
    low = scoring.score_token_probs([(" manageable", 0.6), (" high", 0.1)], risk.verbalizer)
    high = scoring.score_token_probs([(" manageable", 0.1), (" significant", 0.6)], risk.verbalizer)
    assert low["polarity"] > 0 > high["polarity"]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def test_duplicate_surface_forms_are_summed() -> None:
    """Several tokens can normalize to one word; their mass must add up."""
    pairs = [(" optimistic", 0.30), ("optimistic", 0.10), (" cautious", 0.15), ("uncertain,", 0.05)]
    score = scoring.score_token_probs(pairs, prompts.GROWTH.verbalizer)
    assert math.isclose(score["positive"], 0.40)
    assert math.isclose(score["negative"], 0.20)
    assert math.isclose(score["polarity"], 0.20)
    assert set(score) == set(scoring.SCORE_COLUMNS), "no extra score columns"


def test_unmatched_readout_scores_zero() -> None:
    """A readout with no verbalizer word scores a flat zero.

    Same convention as the earlier project: such a filing is indistinguishable
    from one where positive and negative mass genuinely balanced out.
    """
    score = scoring.score_token_probs([(" the", 0.9), (" of", 0.1)], prompts.GROWTH.verbalizer)
    assert score["positive"] == 0.0
    assert score["negative"] == 0.0
    assert score["polarity"] == 0.0


def test_nucleus_cutoff() -> None:
    probs = [0.5, 0.25, 0.15, 0.05, 0.05]
    assert scoring.nucleus_cutoff(probs, 0.5) == 1
    assert scoring.nucleus_cutoff(probs, 0.9) == 3
    assert scoring.nucleus_cutoff(probs, 1.0) == len(probs)


def test_chunk_aggregation_is_internally_consistent() -> None:
    chunks = [
        {"positive": 0.40, "negative": 0.10, "polarity": 0.30},
        {"positive": 0.10, "negative": 0.30, "polarity": -0.20},
        {"positive": 0.20, "negative": 0.20, "polarity": 0.00},
    ]
    agg = scoring.aggregate_chunk_scores(chunks)
    assert math.isclose(agg["positive"], (0.40 + 0.10 + 0.20) / 3)
    assert math.isclose(agg["polarity"], agg["positive"] - agg["negative"])
    assert set(agg) == {"positive", "negative", "polarity"}


def test_vectorized_scoring_matches_the_reference() -> None:
    """The GPU path's arithmetic, mirrored in numpy, must equal scoring.py.

    Guards the tensor implementation in ``ScoringBackend.score_prompts`` against
    drifting away from the reference rule in ``scoring.py``.
    """
    import numpy as np

    vocab = [" optimistic", "optimistic", " confident", " cautious", "cautious,",
             " uncertain", " the", " of", " and", " company", " strong", " weak", "xx"]
    verbalizer = prompts.GROWTH.verbalizer

    classes = np.zeros(len(vocab), dtype=np.int8)
    for i, token in enumerate(vocab):
        word = prompts.normalize_token(token)
        if word in verbalizer["positive"]:
            classes[i] = 1
        elif word in verbalizer["negative"]:
            classes[i] = 2

    rng = np.random.default_rng(0)
    raw = rng.random((100, len(vocab))) ** 3
    probs = raw / raw.sum(-1, keepdims=True)

    for top_p in (0.5, 0.9, 0.99, 1.0):
        order = np.argsort(-probs, axis=-1, kind="stable")
        sorted_probs = np.take_along_axis(probs, order, axis=-1)
        if top_p >= 1.0:
            keep = np.ones_like(sorted_probs, dtype=bool)
        else:
            keep = (sorted_probs.cumsum(-1) - sorted_probs) < top_p
        token_class = classes[order]
        kept = sorted_probs * keep
        positive = (kept * (token_class == 1)).sum(-1)
        negative = (kept * (token_class == 2)).sum(-1)

        for row in range(probs.shape[0]):
            n_keep = scoring.nucleus_cutoff(sorted_probs[row].tolist(), top_p)
            pairs = [(vocab[order[row, i]], float(sorted_probs[row, i])) for i in range(n_keep)]
            expected = scoring.score_token_probs(pairs, verbalizer)
            assert math.isclose(positive[row], expected["positive"], abs_tol=1e-12)
            assert math.isclose(negative[row], expected["negative"], abs_tol=1e-12)
            assert int(keep[row].sum()) == n_keep


# ---------------------------------------------------------------------------


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_fixtures(root)

        tests = [
            (test_filename_round_trip, ()),
            (test_default_sections_per_form_type, (root,)),
            (test_dataset_flags_bad_filings_instead_of_dropping, (root,)),
            (test_prompt_endings, ()),
            (test_token_normalization, ()),
            (test_verbalizer_polarities_are_disjoint, ()),
            (test_risk_sign_convention, ()),
            (test_duplicate_surface_forms_are_summed, ()),
            (test_unmatched_readout_scores_zero, ()),
            (test_nucleus_cutoff, ()),
            (test_chunk_aggregation_is_internally_consistent, ()),
            (test_vectorized_scoring_matches_the_reference, ()),
        ]
        for func, args in tests:
            func(*args)
            print(f"  [ok] {func.__name__}")

    print(f"\n{len(tests)} test(s) passed.")


if __name__ == "__main__":
    main()
