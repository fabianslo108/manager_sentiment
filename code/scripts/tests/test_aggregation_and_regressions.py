"""Tests for the time-aggregation and analysis stages.

Synthetic data with known answers, so the conventions that are easy to get
wrong -- the return horizon, the HAC lag length, weighted aggregation, which
sample a marginal-effect plot describes -- are pinned down.

Run with plain Python::

    python code/scripts/tests/test_aggregation_and_regressions.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import econometrics as ec  # noqa: E402
import macro_data as md  # noqa: E402
import time_aggregation as ta  # noqa: E402

# ---------------------------------------------------------------------------
# return horizons
# ---------------------------------------------------------------------------


def _index_from_returns(returns: list[float]) -> pd.DataFrame:
    """Build a Goyal-Welch-shaped frame whose pct_change equals ``returns``."""
    level = [100.0]
    for r in returns:
        level.append(level[-1] * (1 + r))
    months = pd.date_range("2000-01-01", periods=len(level), freq="MS")
    return pd.DataFrame({ta.MONTH_COL: months, "Index": level, "Rfree": 0.0})


def test_horizon_returns_start_after_the_predictor() -> None:
    returns = [0.10, 0.20, 0.30, 0.40, 0.50]
    frame = md.build_excess_market_returns(
        _index_from_returns(returns), horizons=(1, 3), skip_months=1
    )

    # row 0 is the month whose pct_change is undefined; the predictor lives at
    # row t and the target compounds months t+1 onwards.
    row = 1  # excess_return[1] == returns[0]
    assert math.isclose(frame["excess_return"].iloc[row], returns[0], abs_tol=1e-12)

    expected_1 = returns[1]
    expected_3 = (1 + returns[1]) * (1 + returns[2]) * (1 + returns[3]) - 1
    assert math.isclose(frame["net_cum_return1"].iloc[row], expected_1, abs_tol=1e-12), (
        "net_cum_return1 must be the NEXT month's return only"
    )
    assert math.isclose(frame["net_cum_return3"].iloc[row], expected_3, abs_tol=1e-12), (
        "net_cum_return3 must compound exactly 3 months, starting next month"
    )


def test_horizon_returns_exclude_the_contemporaneous_month() -> None:
    """The predictor's own month must not appear in its target."""
    returns = [0.10, 0.20, 0.30, 0.40]
    frame = md.build_excess_market_returns(_index_from_returns(returns), horizons=(1,), skip_months=1)
    row = 1
    contemporaneous = frame["excess_return"].iloc[row]
    assert not math.isclose(frame["net_cum_return1"].iloc[row], (1 + contemporaneous) * (1 + returns[1]) - 1)


def test_skip_months_zero_includes_the_current_month() -> None:
    returns = [0.10, 0.20, 0.30, 0.40]
    frame = md.build_excess_market_returns(_index_from_returns(returns), horizons=(2,), skip_months=0)
    row = 1
    expected = (1 + returns[0]) * (1 + returns[1]) - 1  # months t and t+1
    assert math.isclose(frame["net_cum_return2"].iloc[row], expected, abs_tol=1e-12)


def test_compounding_refuses_to_bridge_a_gap() -> None:
    frame = _index_from_returns([0.1, 0.2, 0.3, 0.4])
    frame.loc[3, "Index"] = np.nan
    try:
        md.build_excess_market_returns(frame, horizons=(1,))
    except ValueError as exc:
        assert "missing month" in str(exc)
        return
    raise AssertionError("a hole in the return series must be refused, not compounded over")


# ---------------------------------------------------------------------------
# monthly aggregation
# ---------------------------------------------------------------------------


def _reports() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FILING_DATE": ["20200115", "20200120", "20200210", "20200215"],
            "FORM_TYPE": ["10-K", "10-Q", "10-K", "10-Q"],
            "polarity": [1.0, 3.0, 2.0, 5.0],
            "mktcap": [100.0, 900.0, 1.0, 1.0],
        }
    )


def test_equal_and_weighted_aggregation() -> None:
    equal = ta.aggregate_to_monthly(_reports(), ["polarity"], standardize_by=None)
    assert list(equal["n_obs"]) == [2, 2]
    assert math.isclose(equal["polarity"].iloc[0], 2.0)  # (1 + 3) / 2

    weighted = ta.aggregate_to_monthly(
        _reports(), ["polarity"], weight_col="mktcap", standardize_by=None
    )
    expected = (1.0 * 100 + 3.0 * 900) / 1000
    assert math.isclose(weighted["polarity"].iloc[0], expected), "weights must be applied"
    assert weighted["polarity"].iloc[0] > equal["polarity"].iloc[0]


def test_missing_weights_do_not_poison_a_month() -> None:
    reports = _reports()
    reports.loc[0, "mktcap"] = np.nan
    monthly = ta.aggregate_to_monthly(reports, ["polarity"], weight_col="mktcap", standardize_by=None)
    assert math.isclose(monthly["polarity"].iloc[0], 3.0), "only the weighted observation remains"


def test_missing_weight_column_is_refused() -> None:
    try:
        ta.aggregate_to_monthly(_reports(), ["polarity"], weight_col="no_such_column")
    except KeyError as exc:
        assert "market capitalization" in str(exc)
        return
    raise AssertionError("a missing weight column must be refused, not silently ignored")


def test_standardizing_within_form_type_removes_the_level_difference() -> None:
    """10-K and 10-Q scores on different scales must not tilt the monthly mean."""
    rng = np.random.default_rng(0)
    n = 400
    months = np.where(np.arange(n) % 2 == 0, "20200215", "20200515")
    form = np.where(months == "20200215", "10-K", "10-Q")
    # 10-Ks are centred higher and far more dispersed than 10-Qs
    polarity = np.where(form == "10-K", rng.normal(0.5, 0.30, n), rng.normal(0.1, 0.05, n))
    reports = pd.DataFrame({"FILING_DATE": months, "FORM_TYPE": form, "polarity": polarity})

    pooled = ta.aggregate_to_monthly(reports, ["polarity"], standardize_by=None)
    within = ta.aggregate_to_monthly(reports, ["polarity"], standardize_by="FORM_TYPE")

    pooled_gap = abs(pooled["polarity"].iloc[0] - pooled["polarity"].iloc[1])
    within_gap = abs(within["polarity"].iloc[0] - within["polarity"].iloc[1])
    assert pooled_gap > 0.3, "the raw composition effect should be large here"
    assert within_gap < 0.05, f"standardizing must remove it, got {within_gap:.3f}"


def test_centering_removes_the_level_but_keeps_the_spread() -> None:
    rng = np.random.default_rng(5)
    n = 600
    form = np.tile(["10-K", "10-Q"], n // 2)
    polarity = np.where(form == "10-K", rng.normal(0.5, 0.30, n), rng.normal(0.1, 0.05, n))
    reports = pd.DataFrame({"FILING_DATE": ["20200215"] * n, "FORM_TYPE": form, "polarity": polarity})

    centered = ta.standardize_reports(reports, ["polarity"], by="FORM_TYPE", method="center")
    zscored = ta.standardize_reports(reports, ["polarity"], by="FORM_TYPE", method="zscore")

    by_form_centered = centered.groupby("FORM_TYPE")["polarity"]
    assert np.allclose(by_form_centered.mean(), 0.0, atol=1e-12), "centering removes the level"
    spreads = by_form_centered.std()
    assert spreads["10-K"] / spreads["10-Q"] > 3, "centering must leave the scale difference alone"

    # utils.zscore divides by the population std (ddof=0), as StandardScaler does.
    zscored_spreads = zscored.groupby("FORM_TYPE")["polarity"].std(ddof=0)
    assert np.allclose(zscored_spreads, 1.0, atol=1e-9), "z-scoring equalizes the scale too"


def test_unknown_standardize_method_is_refused() -> None:
    try:
        ta.standardize_reports(_reports(), ["polarity"], by="FORM_TYPE", method="demean")
    except ValueError as exc:
        assert "zscore" in str(exc)
        return
    raise AssertionError("an unknown standardization method must be refused")


def test_per_group_counts_are_reported() -> None:
    monthly = ta.aggregate_to_monthly(_reports(), ["polarity"], standardize_by="FORM_TYPE")
    assert {"n_obs_10-K", "n_obs_10-Q"} <= set(monthly.columns)
    assert monthly["n_obs_10-K"].iloc[0] == 1


# ---------------------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------------------


def test_moving_average_is_backward_looking() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    smoothed = ta.smooth(series, "ma", window=2)
    assert math.isnan(smoothed.iloc[0]), "no value before the window is full"
    assert math.isclose(smoothed.iloc[1], 1.5)
    assert math.isclose(smoothed.iloc[4], 4.5), "must average t-1 and t, never t+1"


def test_hp_two_sided_uses_the_future_and_one_sided_does_not() -> None:
    rng = np.random.default_rng(1)
    series = pd.Series(np.cumsum(rng.normal(size=80)))

    two_sided = ta.smooth(series, "hp", lamb=ta.HP_LAMBDA_MONTHLY)
    one_sided = ta.smooth(series, "hp", lamb=ta.HP_LAMBDA_MONTHLY, one_sided=True, min_periods=24)

    # Changing only the last observation must move the two-sided trend at an
    # earlier date, and must leave the one-sided trend there untouched.
    shocked = series.copy()
    shocked.iloc[-1] += 50.0
    two_sided_shocked = ta.smooth(shocked, "hp", lamb=ta.HP_LAMBDA_MONTHLY)
    one_sided_shocked = ta.smooth(shocked, "hp", lamb=ta.HP_LAMBDA_MONTHLY, one_sided=True, min_periods=24)

    mid = 40
    assert abs(two_sided_shocked.iloc[mid] - two_sided.iloc[mid]) > 1e-6, (
        "the two-sided HP trend is look-ahead by construction"
    )
    assert math.isclose(one_sided_shocked.iloc[mid], one_sided.iloc[mid], abs_tol=1e-9), (
        "the one-sided HP trend must not react to a later observation"
    )
    assert one_sided.iloc[:23].isna().all(), "one-sided HP is undefined before min_periods"


def test_build_manager_sentiment_column_names() -> None:
    rng = np.random.default_rng(2)
    months = pd.date_range("2005-01-01", periods=60, freq="MS")
    reports = pd.DataFrame(
        {
            "FILING_DATE": np.repeat(months.strftime("%Y%m%d"), 5),
            "FORM_TYPE": np.tile(["10-K", "10-Q", "10-Q", "10-Q", "10-Q"], 60),
            "polarity_growth": rng.normal(size=300),
        }
    )
    series = ta.build_manager_sentiment(reports, ["polarity_growth"], standardize_by="FORM_TYPE")

    for column in ("polarity_growth", "polarity_growth_ma4", "polarity_growth_hp", "polarity_growth_hp1s"):
        assert column in series.columns, column
    assert len(series) == 60


# ---------------------------------------------------------------------------
# regressions
# ---------------------------------------------------------------------------


def _panel(n: int = 240, beta: float = 0.02, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sentiment = rng.normal(size=n)
    noise = rng.normal(scale=0.04, size=n)
    data = pd.DataFrame({ta.MONTH_COL: pd.date_range("2003-01-01", periods=n, freq="MS")})
    data["sentiment"] = sentiment
    data["vix"] = rng.normal(size=n)
    for horizon in (1, 12):
        data[f"net_cum_return{horizon}"] = beta * horizon * sentiment + noise * math.sqrt(horizon)
    return data


def test_hac_lags_scale_with_the_horizon() -> None:
    assert ec.resolve_maxlags("horizon", 1) == 1
    assert ec.resolve_maxlags("horizon", 36) == 36
    assert ec.resolve_maxlags(12, 36) == 12, "an explicit lag length must be respected"

    _, table = ec.predictive_regression(_panel(), "sentiment", horizons=(1, 12))
    assert list(table["maxlags"]) == [1, 12], "the table must record the lag length used"


def test_predictive_table_shape_and_content() -> None:
    models, table = ec.predictive_regression(_panel(), "sentiment", horizons=(1, 12))

    assert list(table.index) == ["net_cum_return1", "net_cum_return12"]
    assert list(table.columns) == list(ec.TABLE_COLUMNS)
    assert (table["nobs"] > 0).all(), "sample size belongs in every row"
    assert set(models) == {1, 12}
    assert "***" in table.loc["net_cum_return1", "coef"], "a planted effect should show up"


def test_controls_are_included() -> None:
    data = _panel()
    data["control"] = np.random.default_rng(4).normal(size=len(data))
    _, plain = ec.predictive_regression(data, "sentiment", horizons=(1,))
    models, with_control = ec.predictive_regression(data, "sentiment", horizons=(1,), controls=["control"])
    assert "control" in models[1].model.exog_names
    assert plain.loc["net_cum_return1", "nobs"] == with_control.loc["net_cum_return1", "nobs"]


def test_interaction_table_reports_the_interaction() -> None:
    data = _panel()
    name = md.add_interaction(data, "sentiment", "vix")
    assert name == "sentiment_x_vix"
    models, table = ec.interaction_regression(data, "sentiment", "vix", horizons=(12,))
    assert table.loc["net_cum_return12", "coef"] is not None
    assert name in models[12].model.exog_names


def test_missing_interaction_column_is_refused() -> None:
    try:
        ec.interaction_regression(_panel(), "sentiment", "vix", horizons=(1,))
    except KeyError as exc:
        assert "add_interaction" in str(exc)
        return
    raise AssertionError("the interaction term must be created explicitly")


def test_marginal_effect_uses_the_regression_sample() -> None:
    data = _panel()
    md.add_interaction(data, "sentiment", "vix")
    models, _ = ec.interaction_regression(data, "sentiment", "vix", horizons=(12,))
    results = models[12]

    effects = ec.marginal_effect(results, "sentiment", "vix", "sentiment_x_vix", quantiles=(10, 90))
    used = results.model.data.frame

    levels = sorted(effects["moderator_level"].unique())
    expected = sorted(np.percentile(used["vix"].dropna(), [10, 90]))
    assert np.allclose(levels, expected), "moderator levels must be percentiles of the fitted sample"
    assert effects["sentiment"].min() >= used["sentiment"].min() - 1e-9
    assert effects["sentiment"].max() <= used["sentiment"].max() + 1e-9
    assert effects["predicted"].notna().all()


def test_correlation_table_reports_pairwise_n() -> None:
    data = _panel()
    data.loc[:50, "vix"] = np.nan
    table = ec.correlation_with_pvalues(data, ["sentiment", "vix"])
    assert table.loc["sentiment", "sentiment"] == "1.00"
    assert "n=" in table.loc["sentiment", "vix"]


# ---------------------------------------------------------------------------


def main() -> None:
    tests = [
        test_horizon_returns_start_after_the_predictor,
        test_horizon_returns_exclude_the_contemporaneous_month,
        test_skip_months_zero_includes_the_current_month,
        test_compounding_refuses_to_bridge_a_gap,
        test_equal_and_weighted_aggregation,
        test_missing_weights_do_not_poison_a_month,
        test_missing_weight_column_is_refused,
        test_standardizing_within_form_type_removes_the_level_difference,
        test_centering_removes_the_level_but_keeps_the_spread,
        test_unknown_standardize_method_is_refused,
        test_per_group_counts_are_reported,
        test_moving_average_is_backward_looking,
        test_hp_two_sided_uses_the_future_and_one_sided_does_not,
        test_build_manager_sentiment_column_names,
        test_hac_lags_scale_with_the_horizon,
        test_predictive_table_shape_and_content,
        test_controls_are_included,
        test_interaction_table_reports_the_interaction,
        test_missing_interaction_column_is_refused,
        test_marginal_effect_uses_the_regression_sample,
        test_correlation_table_reports_pairwise_n,
    ]
    for func in tests:
        func()
        print(f"  [ok] {func.__name__}")
    print(f"\n{len(tests)} test(s) passed.")


if __name__ == "__main__":
    main()
