"""Report-level sentiment -> monthly manager-sentiment time series.

Both steps are pluggable, as `code/time aggregation/contents.md` asks:

*aggregation*
    equal-weighted or weighted by any report-level column (market capitalization
    being the intended one), optionally standardizing within groups first;

*smoothing*
    a backward moving average, or a Hodrick-Prescott trend with the monthly
    smoothing parameter.


**The HP filter looks ahead.** It is two-sided: the trend at month *t* uses
observations after *t*. That is fine for describing the series and for plots,
but a two-sided trend must not be used as a *predictor* of later returns -- it
leaks future information into the forecast. Use the raw series, the backward
moving average, or ``one_sided=True`` (HP re-estimated on an expanding window,
taking only the last point) for anything predictive.
"""

from __future__ import annotations

import argparse
from pathlib import Path
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
    standardize_by: str | Sequence[str] | None = None,
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
        Group column(s) to equalize before pooling. ``None`` (the default) pools
        the raw scores, as the Jiang monthly mean does. ``"FORM_TYPE"`` is the
        optional equalization of 10-K and 10-Q before pooling.
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


def standardize_series(series: pd.Series) -> pd.Series:
    """Full-sample ``StandardScaler``: subtract the mean and divide by the population std.

    NaNs stay NaN and are left out of the mean and the scale, which is what
    ``sklearn.preprocessing.StandardScaler`` does on a ``dropna``'d column.
    The mean and scale use the whole observed sample, including later months.
    """
    out = pd.Series(np.nan, index=series.index, dtype=float)
    observed = series.notna()
    if not observed.any():
        return out
    out.loc[observed] = zscore(series.loc[observed]).to_numpy()
    return out


def smooth_ma(
    series: pd.Series,
    window: int = DEFAULT_MA_WINDOW,
    min_periods: int | None = None,
    normalize: bool = False,
) -> pd.Series:
    """Backward moving average -- uses only months up to and including *t*.

    ``normalize=True`` then applies :func:`standardize_series` to that average,
    the default in the earlier Jiang notebook.
    """
    averaged = series.rolling(window=window, min_periods=min_periods or window).mean()
    if normalize:
        return standardize_series(averaged)
    return averaged


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
        suffix = f"_ma{options.get('window', DEFAULT_MA_WINDOW)}"
        return f"{suffix}_norm" if options.get("normalize") else suffix
    if method == "hp":
        return "_hp1s" if options.get("one_sided") else "_hp"
    return f"_{method}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

#: Default from the Jiang notebook: a backward 4-month average, then a
#: full-sample StandardScaler of that average. The raw monthly mean is always
#: kept as well. Pass other ``smoothing`` entries for the unnormalized average
#: or either HP variant.
DEFAULT_SMOOTHING: Sequence[tuple[str, dict]] = (
    ("ma", {"window": DEFAULT_MA_WINDOW, "normalize": True}),
)


def build_manager_sentiment(
    reports: pd.DataFrame,
    value_cols: Sequence[str],
    weight_col: str | None = None,
    standardize_by: str | Sequence[str] | None = None,
    standardize_method: str = "zscore",
    smoothing: Sequence[tuple[str, dict]] = DEFAULT_SMOOTHING,
    date_col: str = "FILING_DATE",
    min_obs: int = 1,
) -> pd.DataFrame:
    """Full monthly series: aggregate, then add one column per smoother.

    The default smoother is the Jiang one: a backward 4-month average,
    standardized over the whole sample (``_ma4_norm``). The raw monthly mean
    has no suffix. Other suffixes, when requested: ``_ma4`` for the
    unnormalized average, ``_hp`` for the two-sided HP trend and ``_hp1s`` for
    its one-sided counterpart.
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


def save_time_series(frame: pd.DataFrame, path) -> Path:
    """Write the monthly series as parquet (or csv, if the path says so)."""
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    logger.info("wrote %d month(s) to %s", len(frame), path)
    return path


# ---------------------------------------------------------------------------
# Report-level input
# ---------------------------------------------------------------------------

#: Filing key used to match text files to a sentiment table. Form type is
#: stored as ``10-K`` / ``10-Q`` whichever spelling the source used.
_MATCH_COLUMNS = ("_DATE", "_ACC", "_FORM")


def _normalize_form(value: object) -> str:
    compact = str(value).upper().replace("-", "")
    return {"10K": "10-K", "10Q": "10-Q"}.get(compact, str(value))


def _filing_date_str(series: pd.Series) -> pd.Series:
    """``YYYYMMDD`` as an 8-character string, from ints or strings."""
    if pd.api.types.is_numeric_dtype(series):
        return series.round().astype("Int64").astype(str)
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.slice(0, 8)


def _with_match_keys(reports: pd.DataFrame) -> pd.DataFrame:
    if "FORM_TYPE" in reports.columns:
        form = reports["FORM_TYPE"]
    elif "REPORT_TYPE" in reports.columns:
        form = reports["REPORT_TYPE"]
    else:
        raise KeyError("report table needs FORM_TYPE or REPORT_TYPE")
    missing = [c for c in ("FILING_DATE", "ACC_NUM") if c not in reports.columns]
    if missing:
        raise KeyError(f"report table is missing column(s): {missing}")
    out = reports.copy()
    out["_DATE"] = _filing_date_str(out["FILING_DATE"])
    out["_ACC"] = out["ACC_NUM"].astype(str)
    out["_FORM"] = form.map(_normalize_form)
    return out


#: Report-level score columns. A bare name such as ``growth_llama`` means the
#: polarity column; ``positive_`` and ``negative_`` have to be written in full.
SCORE_PREFIXES = ("polarity_", "positive_", "negative_")


def score_columns(reports: pd.DataFrame) -> list[str]:
    """``polarity_*``, ``positive_*`` and ``negative_*`` columns, unsmoothed."""
    return [
        column
        for column in reports.columns
        if column.startswith(SCORE_PREFIXES)
        and "_ma" not in column
        and not column.endswith(("_hp", "_hp1s"))
    ]


def polarity_columns(reports: pd.DataFrame) -> list[str]:
    """Polarity columns only. Smoothed monthly suffixes are excluded."""
    return [column for column in score_columns(reports) if column.startswith("polarity_")]


def _as_score_columns(names: Sequence[str]) -> list[str]:
    """Resolve measure names. A name without a prefix is a polarity column."""
    if len(names) == 0:
        raise ValueError("polarities is empty; pass None to keep every polarity column")
    resolved: list[str] = []
    for name in names:
        column = str(name)
        if not column.startswith(SCORE_PREFIXES):
            column = f"polarity_{column}"
        if column not in resolved:
            resolved.append(column)
    return resolved


def _split_score(column: str) -> tuple[str, str]:
    """``positive_growth_llama`` -> ``("positive", "growth_llama")``."""
    for prefix in SCORE_PREFIXES:
        if column.startswith(prefix):
            return prefix[:-1], column[len(prefix):]
    return "polarity", column


def _pivot_scores(frame: pd.DataFrame, requested: list[str] | None) -> pd.DataFrame:
    """Long extraction rows (one strategy per row) -> one row per filing."""
    if requested is None:
        kinds = ["polarity"]
        strategies = None
    else:
        parts = [_split_score(column) for column in requested]
        kinds = list(dict.fromkeys(kind for kind, _ in parts))
        strategies = {strategy for _, strategy in parts}
        frame = frame[frame["strategy"].isin(strategies)]
        if frame.empty:
            raise KeyError(f"none of {requested} are in the strategy column")
    missing = [kind for kind in kinds if kind not in frame.columns]
    if missing:
        raise KeyError(f"long table has no column(s) {missing}")

    wide = frame.pivot_table(
        index=["CIK", "_DATE", "_ACC", "_FORM"],
        columns="strategy",
        values=kinds[0] if len(kinds) == 1 else kinds,
        aggfunc="mean",
    )
    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = [f"{kind}_{strategy}" for kind, strategy in wide.columns]
    else:
        wide.columns = [f"{kinds[0]}_{strategy}" for strategy in wide.columns]
    wide = wide.reset_index().rename(
        columns={"_DATE": "FILING_DATE", "_ACC": "ACC_NUM", "_FORM": "FORM_TYPE"}
    )
    wide.columns.name = None
    if requested is not None:
        missing = [column for column in requested if column not in wide.columns]
        if missing:
            raise KeyError(f"score column(s) not in the table: {missing}")
        keys = [column for column in wide.columns if column not in score_columns(wide)]
        wide = wide[keys + requested]
    return wide


def prepare_reports(
    reports: pd.DataFrame,
    polarities: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per filing, with ``FORM_TYPE`` and the selected score columns.

    Accepts either the wide table (``polarity_growth_llama``, ``positive_*``,
    ``negative_*`` and the other measures) or the long extraction output (one
    row per ``strategy``, with ``positive``, ``negative`` and ``polarity``).

    ``polarities=None`` keeps every polarity column. A list keeps only the named
    measures, in that order. Each name is a ``polarity_*``, ``positive_*`` or
    ``negative_*`` column; a name without a prefix is read as a polarity.
    """
    requested = None if polarities is None else _as_score_columns(polarities)
    if "strategy" in reports.columns and "polarity" in reports.columns and not score_columns(reports):
        frame = _with_match_keys(reports)
        frame["CIK"] = frame["CIK"].astype(str).str.replace(r"\.0$", "", regex=True)
        return _pivot_scores(frame, requested)

    frame = _with_match_keys(reports)
    available = score_columns(frame)
    if not available:
        raise KeyError(
            "no score columns found: expected polarity_*/positive_*/negative_* "
            "or a long table with strategy and polarity"
        )
    columns = polarity_columns(frame) if requested is None else requested
    if requested is not None:
        missing = [column for column in requested if column not in available]
        if missing:
            raise KeyError(f"score column(s) not in the table: {missing}. available: {available}")
        columns = requested
    frame["CIK"] = frame["CIK"].astype(str).str.replace(r"\.0$", "", regex=True)
    before = len(frame)
    grouped = (
        frame.groupby(["CIK", "_DATE", "_ACC", "_FORM"], dropna=False, as_index=False)[columns]
        .mean()
        .rename(columns={"_DATE": "FILING_DATE", "_ACC": "ACC_NUM", "_FORM": "FORM_TYPE"})
    )
    if len(grouped) < before:
        logger.warning(
            "averaged %d extra duplicate row(s); %d filing(s) remain",
            before - len(grouped),
            len(grouped),
        )
    logger.info(
        "prepared %d filing(s), measures: %s",
        len(grouped),
        ", ".join(columns),
    )
    return grouped


def select_filings(reports: pd.DataFrame, filings_dir: str | Path) -> pd.DataFrame:
    """Keep sentiment rows whose filing is among the texts under ``filings_dir``.

    Match is on filing date, accession number and form type. The returned frame
    keeps the source columns, so it can be written back out as an extraction table.
    """
    from reports import discover_reports

    found = discover_reports(filings_dir)
    if found.empty:
        raise RuntimeError(f"no filings discovered under {filings_dir}")

    keyed = _with_match_keys(reports)
    wanted = _with_match_keys(found)[list(_MATCH_COLUMNS)].drop_duplicates()
    present = keyed[list(_MATCH_COLUMNS)].drop_duplicates()
    n_hit = wanted.merge(present, on=list(_MATCH_COLUMNS), how="inner").shape[0]
    n_wanted = len(wanted)
    logger.info("%d of %d discovered filing(s) have a sentiment row", n_hit, n_wanted)

    matched = keyed.merge(wanted, on=list(_MATCH_COLUMNS), how="inner")
    matched = matched.drop(columns=list(_MATCH_COLUMNS))
    if n_hit < n_wanted:
        logger.warning(
            "%d discovered filing(s) are not in the sentiment table and are left out",
            n_wanted - n_hit,
        )
    if matched.empty:
        raise RuntimeError(f"none of the filings under {filings_dir} appear in the sentiment table")
    return matched.reset_index(drop=True)


def load_reports(path: str | Path) -> pd.DataFrame:
    """Read one report-level sentiment table, parquet or csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported sentiment table {path.name}; use .parquet or .csv")
    logger.info("loaded %d row(s) from %s", len(frame), path.name)
    return frame


def save_table(frame: pd.DataFrame, path) -> Path:
    """Write a report-level table as parquet."""
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    logger.info("wrote %d row(s) to %s", len(frame), path)
    return path


def aggregate_file(
    input_path: str | Path,
    output_path: str | Path,
    filings_dir: str | Path | None = None,
    matched_output: str | Path | None = None,
    standardize_by: str | None = None,
    standardize_method: str = "zscore",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Load a sentiment table, optionally restrict it to on-disk filings, and save the monthly series."""
    reports = load_reports(input_path)
    if filings_dir is not None:
        reports = select_filings(reports, filings_dir)
        if matched_output is not None:
            save_table(reports, matched_output)
    prepared = prepare_reports(reports)
    series = build_manager_sentiment(
        prepared,
        value_cols=polarity_columns(prepared),
        weight_col=weight_col,
        standardize_by=standardize_by,
        standardize_method=standardize_method,
    )
    save_time_series(series, output_path)
    return series


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate report-level sentiment to a monthly series.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filings-dir", type=Path, default=None)
    parser.add_argument("--matched-output", type=Path, default=None)
    parser.add_argument("--standardize-by", default="none")
    parser.add_argument("--standardize-method", default="zscore", choices=list(STANDARDIZE_METHODS))
    parser.add_argument("--weight-col", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    standardize_by = None if args.standardize_by in ("none", "None", "") else args.standardize_by
    aggregate_file(
        args.input,
        args.output,
        filings_dir=args.filings_dir,
        matched_output=args.matched_output,
        standardize_by=standardize_by,
        standardize_method=args.standardize_method,
        weight_col=args.weight_col,
    )


if __name__ == "__main__":
    main()
