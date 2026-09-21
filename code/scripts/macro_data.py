"""Monthly market data for the predictive regressions.

Loaders for the files the earlier project used, all returning a frame with a
``year_month`` timestamp column so they can be merged with the manager-sentiment
series:

===========================================  ==========================================
file                                         contents
===========================================  ==========================================
``PredictorData<YEAR>.xlsx``                 Goyal-Welch predictors -> S&P 500 index,
                                             dividends, earnings, rates, spreads
``VIX_History.csv``                          daily CBOE VIX -> monthly mean
``nvix_and_categories_timeseries_*.xlsx``    Manela-Moreira news-implied volatility
``SENTIMENT.xlsx``                           Baker-Wurgler file, used for ``recess``
``JLMZ_MS_Index2017.xlsx``                   Jiang et al. Manager Sentiment Index
===========================================  ==========================================

On the return horizon convention
--------------------------------

``net_cum_return{h}`` at month *t* is the cumulative excess market return over
**exactly h months, starting at t + skip_months**. With the default
``skip_months=1`` the h=1 column is the return of month *t+1*, which is what a
predictive regression of returns on month-*t* sentiment needs.

This is worth stating because it is easy to get wrong. A cumulative product that
starts at month *t* itself puts the predictor and part of the "future" return in
the same month: sentiment aggregated from filings made during month *t* is then
mechanically correlated with month *t*'s market return, and the regression reads
that overlap as predictability. ``skip_months=0`` reproduces that variant if you
need to compare against it; it is not the default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from time_aggregation import MONTH_COL
from utils import get_logger

logger = get_logger(__name__)

#: Controls from Goyal-Welch, following Jiang et al. Table 2. ``log_DY`` and
#: ``log_DE`` are also built but left out by default: they are near-collinear
#: with ``log_DP`` and ``log_EP``.
GOYAL_WELCH_CONTROLS = (
    "log_DP",
    "log_EP",
    "svar",
    "bm",
    "ntis",
    "tbl",
    "lty",
    "ltr",
    "tms",
    "dfy",
    "dfr",
    "infl",
)

DEFAULT_HORIZONS = (1, 3, 6, 9, 12, 24, 36)

RETURN_PREFIX = "net_cum_return"


def _month_from_yyyymm(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(int).astype(str), format="%Y%m")


# ---------------------------------------------------------------------------
# Goyal-Welch predictors and market returns
# ---------------------------------------------------------------------------


def load_goyal_welch(path: str | Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load the Goyal-Welch monthly file and build the control variables.

    ``start`` / ``end`` are ``YYYYMM`` strings, inclusive.
    """
    raw = pd.read_excel(path)
    data = raw.rename(columns={"b/m": "bm"}).copy()

    if start is not None:
        data = data[data["yyyymm"].astype(int) >= int(start)]
    if end is not None:
        data = data[data["yyyymm"].astype(int) <= int(end)]

    data = data.sort_values("yyyymm").reset_index(drop=True)
    data[MONTH_COL] = _month_from_yyyymm(data["yyyymm"])

    # Dividend / earnings ratios use 12-month moving sums (D12, E12).
    data["log_DP"] = np.log(data["D12"]) - np.log(data["Index"])
    data["log_DY"] = np.log(data["D12"]) - np.log(data["Index"].shift(1))
    data["log_EP"] = np.log(data["E12"]) - np.log(data["Index"])
    data["log_DE"] = np.log(data["D12"]) - np.log(data["E12"])
    data["tms"] = data["lty"] - data["tbl"]
    data["dfy"] = data["BAA"] - data["AAA"]
    data["dfr"] = data["corpr"] - data["ltr"]

    logger.info(
        "loaded Goyal-Welch: %d month(s), %s to %s",
        len(data),
        data[MONTH_COL].min().date(),
        data[MONTH_COL].max().date(),
    )
    return data


def build_excess_market_returns(
    data: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    skip_months: int = 1,
    index_col: str = "Index",
    rf_col: str = "Rfree",
) -> pd.DataFrame:
    """Add the monthly excess return and cumulative returns at each horizon.

    ``excess_return`` at month *t* is the S&P 500 price return of month *t* minus
    the risk-free rate. ``net_cum_return{h}`` at *t* compounds
    ``excess_return`` over months ``t + skip_months`` .. ``t + skip_months + h - 1``.
    See the module docstring on ``skip_months``.
    """
    if skip_months < 0:
        raise ValueError(f"skip_months must be >= 0, got {skip_months}")

    out = data.sort_values(MONTH_COL).reset_index(drop=True).copy()
    # fill_method=None: a missing index level must surface as a missing return,
    # not be padded into a spurious 0% month.
    out["excess_return"] = out[index_col].pct_change(fill_method=None) - out[rf_col]

    returns = out["excess_return"]
    valid = returns.notna()
    if not valid.any():
        raise ValueError("excess_return is missing for every month; check the index and risk-free columns")

    # Only the very first month may be missing (pct_change has no predecessor).
    # Any later gap would be bridged by the cumulative product, quietly turning
    # two non-adjacent months into one compounded return.
    first = valid.idxmax()
    gaps = int((~valid.loc[first:]).sum())
    if gaps:
        months = out.loc[first:][~valid.loc[first:]][MONTH_COL].dt.strftime("%Y-%m").tolist()
        raise ValueError(
            f"excess_return has {gaps} missing month(s) after the sample starts "
            f"({', '.join(months[:6])}{'...' if gaps > 6 else ''}); compounding across a gap would "
            "silently bridge it -- trim the sample to a contiguous span first"
        )

    # Cumulative gross product with the pre-sample months contributing a factor
    # of 1, so that G_{t-1} exists for the very first return and a horizon
    # return is just a ratio of two points on this curve.
    gross = (1.0 + returns.fillna(0.0)).cumprod()
    for horizon in horizons:
        upper = gross.shift(-(skip_months + horizon - 1))
        lower = gross.shift(1 - skip_months)
        out[f"{RETURN_PREFIX}{horizon}"] = upper / lower - 1.0

    logger.info(
        "built %s for horizons %s, starting %d month(s) after the predictor",
        RETURN_PREFIX,
        list(horizons),
        skip_months,
    )
    return out


def return_columns(horizons: Sequence[int] = DEFAULT_HORIZONS) -> list[str]:
    return [f"{RETURN_PREFIX}{h}" for h in horizons]


# ---------------------------------------------------------------------------
# Uncertainty and sentiment benchmarks
# ---------------------------------------------------------------------------


def load_msi(path: str | Path) -> pd.DataFrame:
    """Jiang et al. Manager Sentiment Index (monthly, ``YYYY-MM`` dates)."""
    raw = pd.read_excel(path)
    data = raw.rename(columns={"Manager Sentiment Index": "msi"})
    data[MONTH_COL] = pd.to_datetime(data["date"].astype(str), format="%Y-%m")
    return data[[MONTH_COL, "msi"]].sort_values(MONTH_COL).reset_index(drop=True)


def load_vix(path: str | Path, price_col: str = "CLOSE") -> pd.DataFrame:
    """Daily CBOE VIX history collapsed to a monthly mean."""
    raw = pd.read_csv(path)
    raw["DATE"] = pd.to_datetime(raw["DATE"])
    raw[MONTH_COL] = raw["DATE"].dt.to_period("M").dt.to_timestamp()
    monthly = raw.groupby(MONTH_COL, as_index=False)[price_col].mean()
    return monthly.rename(columns={price_col: "vix"})


def load_nvix(path: str | Path, sheet_name: str = "news_implied_volatility") -> pd.DataFrame:
    """Manela-Moreira news-implied volatility (month-end dated)."""
    raw = pd.read_excel(path, sheet_name=sheet_name)
    raw[MONTH_COL] = pd.to_datetime(raw["Date"].astype(int).astype(str), format="%Y%m%d")
    raw[MONTH_COL] = raw[MONTH_COL].dt.to_period("M").dt.to_timestamp()
    monthly = raw.groupby(MONTH_COL, as_index=False)["NVIX"].mean()
    return monthly.rename(columns={"NVIX": "nvix"})


def load_recession(path: str | Path, flag_col: str = "recess") -> pd.DataFrame:
    """NBER recession indicator from the Baker-Wurgler sentiment workbook."""
    raw = pd.read_excel(path)
    if flag_col not in raw.columns:
        raise KeyError(f"{Path(path).name} has no {flag_col!r} column; found {list(raw.columns)[:15]}")
    data = raw[["yearmo", flag_col]].dropna(subset=["yearmo"]).copy()
    data[MONTH_COL] = _month_from_yyyymm(data["yearmo"])
    return data[[MONTH_COL, flag_col]].rename(columns={flag_col: "recession"})


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_monthly(left: pd.DataFrame, *others: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    """Left-join monthly frames on ``year_month``, warning about no-overlap joins."""
    merged = left.copy()
    for other in others:
        if MONTH_COL not in other.columns:
            raise KeyError(f"frame to merge has no {MONTH_COL!r} column")
        new_cols = [c for c in other.columns if c != MONTH_COL]
        overlap = merged[MONTH_COL].isin(other[MONTH_COL]).sum()
        if overlap == 0:
            logger.warning("merging %s adds no overlapping month at all", new_cols)
        merged = merged.merge(other, on=MONTH_COL, how=how)
        matched = merged[new_cols].notna().any(axis=1).sum()
        logger.info("merged %s: %d of %d month(s) matched", new_cols, matched, len(merged))
    return merged


def add_interaction(data: pd.DataFrame, left: str, right: str, name: str | None = None) -> str:
    """Add a product term and return its column name."""
    name = name or f"{left}_x_{right}"
    data[name] = data[left] * data[right]
    return name
