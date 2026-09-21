"""Report-level sentiment -> monthly manager-sentiment time series.

Both steps are pluggable, as `code/time aggregation/contents.md` asks:

*aggregation*
    equal-weighted or weighted by any report-level column (market capitalization
    being the intended one), optionally standardizing within groups first;

*smoothing*
    a backward moving average, or a Hodrick-Prescott trend with the monthly
    smoothing parameter.

Two things worth knowing before choosing options.

**Standardizing within form type.** 10-K and 10-Q filings do not arrive evenly
over the calendar: February and March are almost entirely 10-Ks, May, August and
November almost entirely 10-Qs. If the two form types have different score
levels or spreads, a plain pooled monthly mean inherits that calendar pattern as
a mechanical annual cycle. ``standardize_by="FORM_TYPE"`` equalizes the groups
before pooling -- both level and scale with ``standardize_method="zscore"``, the
level only with ``"center"``. Use ``standardize_by=None`` for a plain pooled
mean.

**The HP filter looks ahead.** It is two-sided: the trend at month *t* uses
observations after *t*. That is fine for describing the series and for plots,
but a two-sided trend must not be used as a *predictor* of later returns -- it
leaks future information into the forecast. Use the raw series, the backward
moving average, or ``one_sided=True`` (HP re-estimated on an expanding window,
taking only the last point) for anything predictive.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from utils import get_logger, zscore

logger = get_logger(__name__)

MONTH_COL = "year_month"

#: Hodrick-Prescott smoothing parameter for monthly data, by the Ravn-Uhlig
#: frequency rule: 1600 (quarterly) * 3**4.
HP_LAMBDA_MONTHLY = 129_600

#: Window of the backward moving average used in the earlier project.
DEFAULT_MA_WINDOW = 4


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def to_month(filing_date: pd.Series) -> pd.Series:
    """``YYYYMMDD`` filing dates -> timestamps at the start of the month."""
    parsed = pd.to_datetime(filing_date.astype(str).str[:8], format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        n_bad = int(parsed.isna().sum())
        logger.warning("%d filing date(s) could not be parsed and become NaT", n_bad)
    return parsed.dt.to_period("M").dt.to_timestamp()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


#: How :func:`standardize_reports` equalizes groups before pooling.
STANDARDIZE_METHODS = ("zscore", "center")


def standardize_reports(
    reports: pd.DataFrame,
    value_cols: Sequence[str],
    by: str | Sequence[str] | None,
    method: str = "zscore",
) -> pd.DataFrame:
    """Equalize ``value_cols`` across groups at the report level.

    Applied before monthly aggregation so that pooling groups with different
    score distributions (10-K vs 10-Q) does not inject the calendar composition
    into the monthly index.

    ``method="zscore"`` removes both the level and the scale difference.
    Removing the scale is a real change to the measure -- a group whose scores
    are tightly clustered gets stretched more than a dispersed one -- but it is
    what stops a month dominated by the dispersed group from being mechanically
    more volatile than the next. ``method="center"`` subtracts the group mean
    only, leaving each group's spread as it is.
    """
    if by is None:
        return reports
    if method not in STANDARDIZE_METHODS:
        raise ValueError(f"method must be one of {STANDARDIZE_METHODS}, got {method!r}")

    out = reports.copy()
    groups = [by] if isinstance(by, str) else list(by)
    missing = [c for c in groups if c not in out.columns]
    if missing:
        raise KeyError(f"cannot standardize within missing column(s): {missing}")

    for col in value_cols:
        grouped = out.groupby(groups, dropna=False)[col]
        if method == "zscore":
            out[col] = grouped.transform(zscore)
        else:
            out[col] = out[col] - grouped.transform("mean")

    sizes = out.groupby(groups, dropna=False).size()
    logger.info(
        "%s %s within %s (group sizes: %s)",
        "standardized" if method == "zscore" else "centered",
        list(value_cols),
        groups,
        sizes.to_dict(),
    )
    return out


def _weighted_mean(values: pd.Series, weights: pd.Series | None) -> float:
    ok = values.notna()
    if weights is None:
        return values[ok].mean() if ok.any() else np.nan
    ok &= weights.notna() & (weights > 0)
    if not ok.any():
        return np.nan
    return float((values[ok] * weights[ok]).sum() / weights[ok].sum())


def aggregate_to_monthly(
    reports: pd.DataFrame,
    value_cols: Sequence[str],
    weight_col: str | None = None,
    standardize_by: str | Sequence[str] | None = "FORM_TYPE",
    standardize_method: str = "zscore",
    date_col: str = "FILING_DATE",
    min_obs: int = 1,
) -> pd.DataFrame:
    """Collapse report-level scores into one row per calendar month.

    Parameters
    ----------
    reports:
        Report-level table, e.g. the concatenated extraction output.
    value_cols:
        Sentiment columns to aggregate.
    weight_col:
        Report-level weight, typically market capitalization. ``None``
        equal-weights. Missing or non-positive weights drop out of that month's
        weighted mean.
    standardize_by:
        Group column(s) to equalize before pooling; ``None`` disables it.
    standardize_method:
        ``"zscore"`` (level and scale) or ``"center"`` (level only). See
        :func:`standardize_reports`.
    min_obs:
        Months with fewer filings than this are dropped.

    Returns
    -------
    DataFrame with ``year_month``, the aggregated ``value_cols``, ``n_obs`` and
    per-group counts when ``standardize_by`` is a single column.
    """
    missing = [c for c in value_cols if c not in reports.columns]
    if missing:
        raise KeyError(f"missing sentiment column(s): {missing}")
    if weight_col is not None and weight_col not in reports.columns:
        raise KeyError(
            f"weight column {weight_col!r} is not in the report table; "
            "merge market capitalization in first, or pass weight_col=None"
        )

    frame = reports.copy()
    frame[MONTH_COL] = to_month(frame[date_col])
    frame = frame.dropna(subset=[MONTH_COL])

    frame = standardize_reports(frame, value_cols, standardize_by, standardize_method)

    records = []
    for month, group in frame.groupby(MONTH_COL, sort=True):
        weights = group[weight_col] if weight_col else None
        row = {MONTH_COL: month, "n_obs": len(group)}
        for col in value_cols:
            row[col] = _weighted_mean(group[col], weights)
        if isinstance(standardize_by, str) and standardize_by in group.columns:
            for label, count in group[standardize_by].value_counts().items():
                row[f"n_obs_{label}"] = int(count)
        records.append(row)

    monthly = pd.DataFrame(records).sort_values(MONTH_COL).reset_index(drop=True)
    for col in [c for c in monthly.columns if c.startswith("n_obs_")]:
        monthly[col] = monthly[col].fillna(0).astype(int)

    thin = monthly["n_obs"] < min_obs
    if thin.any():
        logger.warning("dropping %d month(s) with fewer than %d filing(s)", int(thin.sum()), min_obs)
        monthly = monthly[~thin].reset_index(drop=True)

    logger.info(
        "aggregated %d filing(s) into %d month(s) (%s weighting)",
        len(frame),
        len(monthly),
        f"weighted by {weight_col}" if weight_col else "equal",
    )
    return monthly


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def smooth_none(series: pd.Series) -> pd.Series:
    return series


def smooth_ma(series: pd.Series, window: int = DEFAULT_MA_WINDOW, min_periods: int | None = None) -> pd.Series:
    """Backward moving average -- uses only months up to and including *t*."""
    return series.rolling(window=window, min_periods=min_periods or window).mean()


def smooth_hp(
    series: pd.Series,
    lamb: float = HP_LAMBDA_MONTHLY,
    one_sided: bool = False,
    min_periods: int = 24,
) -> pd.Series:
    """Hodrick-Prescott trend.

    ``one_sided=False`` (default) is the usual two-sided filter: the trend at
    *t* uses the whole sample, so it must not be used to predict anything dated
    after *t*. ``one_sided=True`` re-runs the filter on each expanding window and
    keeps only its last point, which is safe for predictive use but slower and
    undefined for the first ``min_periods`` months.
    """
    import warnings

    from scipy.sparse import SparseEfficiencyWarning
    from statsmodels.tsa.filters.hp_filter import hpfilter

    # statsmodels builds the filter matrix in a format scipy grumbles about on
    # every call; the result is fine and the one-sided loop makes it very noisy.
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

    values = series.astype(float)
    observed = values.notna()
    trend = pd.Series(np.nan, index=series.index, dtype=float)
    if observed.sum() < 4:
        logger.warning("too few observations (%d) for an HP trend", int(observed.sum()))
        return trend

    if not one_sided:
        _, fitted = hpfilter(values[observed], lamb=lamb)
        trend.loc[observed[observed].index] = fitted
        return trend

    index = list(values[observed].index)
    for i in range(min_periods - 1, len(index)):
        window = values.loc[index[: i + 1]]
        _, fitted = hpfilter(window, lamb=lamb)
        trend.loc[index[i]] = fitted.iloc[-1]
    return trend


#: Smoothers available by name, each taking a series and keyword options.
SMOOTHERS: Mapping[str, Callable[..., pd.Series]] = {
    "none": smooth_none,
    "ma": smooth_ma,
    "hp": smooth_hp,
}


def smooth(series: pd.Series, method: str, **options) -> pd.Series:
    try:
        smoother = SMOOTHERS[method]
    except KeyError:
        raise KeyError(f"unknown smoother {method!r}; available: {sorted(SMOOTHERS)}") from None
    return smoother(series, **options)


def _suffix(method: str, options: Mapping[str, object]) -> str:
    if method == "none":
        return ""
    if method == "ma":
        return f"_ma{options.get('window', DEFAULT_MA_WINDOW)}"
    if method == "hp":
        return "_hp1s" if options.get("one_sided") else "_hp"
    return f"_{method}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

#: Aggregate raw, plus the backward MA used in the earlier project, plus both HP
#: variants. Override to change what lands in the output file.
DEFAULT_SMOOTHING: Sequence[tuple[str, dict]] = (
    ("none", {}),
    ("ma", {"window": DEFAULT_MA_WINDOW}),
    ("hp", {}),
    ("hp", {"one_sided": True}),
)


def build_manager_sentiment(
    reports: pd.DataFrame,
    value_cols: Sequence[str],
    weight_col: str | None = None,
    standardize_by: str | Sequence[str] | None = "FORM_TYPE",
    standardize_method: str = "zscore",
    smoothing: Sequence[tuple[str, dict]] = DEFAULT_SMOOTHING,
    date_col: str = "FILING_DATE",
    min_obs: int = 1,
) -> pd.DataFrame:
    """Full monthly series: aggregate, then add one column per smoother.

    Column names are the measure plus a suffix: no suffix for the raw monthly
    mean, ``_ma4`` for the 4-month backward average, ``_hp`` for the two-sided
    HP trend and ``_hp1s`` for its one-sided counterpart.
    """
    monthly = aggregate_to_monthly(
        reports,
        value_cols=value_cols,
        weight_col=weight_col,
        standardize_by=standardize_by,
        standardize_method=standardize_method,
        date_col=date_col,
        min_obs=min_obs,
    ).set_index(MONTH_COL)

    for col in value_cols:
        for method, options in smoothing:
            suffix = _suffix(method, options)
            if not suffix:
                continue
            monthly[f"{col}{suffix}"] = smooth(monthly[col], method, **options)

    return monthly.reset_index()


def save_time_series(frame: pd.DataFrame, path) -> None:
    """Write the monthly series, creating the directory if needed."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("wrote %d month(s) to %s", len(frame), path)
