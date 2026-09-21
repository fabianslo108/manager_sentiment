"""Predictive regressions of market returns on manager sentiment.

The Jiang et al. design: regress the cumulative excess market return at horizon
*h* on month-*t* sentiment, one regression per horizon, with Newey-West standard
errors. Which months the target spans is set by
:func:`macro_data.build_excess_market_returns`.

On the lag length
-----------------

Overlapping long-horizon returns are strongly autocorrelated by construction:
consecutive 36-month returns share 35 months of data. The HAC lag length has to
grow with the horizon, otherwise the standard errors are too small and the
t-statistics too large exactly where the results tend to look strongest. So
``maxlags="horizon"`` (the default) uses *h* lags for the *h*-month return, and
a fixed integer is accepted when a specific lag length is wanted.

Every table reports ``nobs`` alongside the coefficient. Rows are estimated on
whatever months have a non-missing predictor and target, and that sample is not
the same across measures or horizons -- comparing coefficients without knowing
the sample sizes is misleading.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from macro_data import RETURN_PREFIX
from utils import get_logger, significance_stars

logger = get_logger(__name__)

TABLE_COLUMNS = ("coef", "t", "pval", "r2_adj", "nobs", "maxlags")


def horizon_maxlags(horizon: int) -> int:
    """HAC lag length for an ``horizon``-month overlapping return."""
    return max(int(horizon), 1)


def resolve_maxlags(maxlags: int | str, horizon: int) -> int:
    if isinstance(maxlags, str):
        if maxlags != "horizon":
            raise ValueError(f"maxlags must be an int or 'horizon', got {maxlags!r}")
        return horizon_maxlags(horizon)
    return max(int(maxlags), 1)


def newey_west_ols(
    data: pd.DataFrame,
    target: str,
    x_vars: Sequence[str],
    maxlags: int,
):
    """OLS with Newey-West (HAC) standard errors, fitted on complete cases.

    Formula-based so that ``results.predict(frame)`` works for marginal-effect
    plots without having to add a constant by hand.
    """
    columns = [target, *x_vars]
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise KeyError(f"missing column(s) for the regression: {missing}")

    subset = data.dropna(subset=columns)
    if subset.empty:
        raise ValueError(f"no complete observations for {target} ~ {' + '.join(x_vars)}")

    formula = f"{target} ~ " + " + ".join(x_vars)
    return smf.ols(formula, data=subset).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def _row(results, focus: str, maxlags: int) -> dict:
    pval = float(results.pvalues[focus])
    return {
        "coef": round(float(results.params[focus]), 4),
        "t": round(float(results.tvalues[focus]), 3),
        "pval": round(pval, 4),
        "r2_adj": round(100 * float(results.rsquared_adj), 3),
        "nobs": int(results.nobs),
        "maxlags": maxlags,
        "stars": significance_stars(pval),
    }


def _format_table(rows: Mapping[object, dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows).T
    table["coef"] = table["coef"].astype(str) + table["stars"]
    return table[list(TABLE_COLUMNS)]


def predictive_regression(
    data: pd.DataFrame,
    sentiment: str,
    horizons: Sequence[int],
    controls: Sequence[str] | None = None,
    maxlags: int | str = "horizon",
    return_prefix: str = RETURN_PREFIX,
) -> tuple[dict[int, object], pd.DataFrame]:
    """One regression per horizon: ``net_cum_return{h} ~ sentiment_t (+ controls)``.

    Returns the fitted models keyed by horizon and a summary table whose
    ``coef`` carries significance stars.
    """
    x_vars = [sentiment, *(controls or [])]
    models: dict[int, object] = {}
    rows: dict[object, dict] = {}

    for horizon in horizons:
        target = f"{return_prefix}{horizon}"
        lags = resolve_maxlags(maxlags, horizon)
        results = newey_west_ols(data, target, x_vars, lags)
        models[horizon] = results
        rows[target] = _row(results, sentiment, lags)

    return models, _format_table(rows)


def interaction_regression(
    data: pd.DataFrame,
    sentiment: str,
    moderator: str,
    horizons: Sequence[int],
    interaction: str | None = None,
    controls: Sequence[str] | None = None,
    maxlags: int | str = "horizon",
    return_prefix: str = RETURN_PREFIX,
    focus: str | None = None,
) -> tuple[dict[int, object], pd.DataFrame]:
    """Sentiment, a moderator and their product, one regression per horizon.

    ``focus`` picks which coefficient the table reports; it defaults to the
    interaction term, since that is what the specification is about.
    """
    interaction = interaction or f"{sentiment}_x_{moderator}"
    if interaction not in data.columns:
        raise KeyError(
            f"interaction column {interaction!r} is not in the data; "
            "create it with macro_data.add_interaction first"
        )

    x_vars = [sentiment, moderator, interaction, *(controls or [])]
    focus = focus or interaction

    models: dict[int, object] = {}
    rows: dict[object, dict] = {}
    for horizon in horizons:
        target = f"{return_prefix}{horizon}"
        lags = resolve_maxlags(maxlags, horizon)
        results = newey_west_ols(data, target, x_vars, lags)
        models[horizon] = results
        rows[target] = _row(results, focus, lags)

    return models, _format_table(rows)


def compare_measures(
    data: pd.DataFrame,
    sentiments: Sequence[str],
    horizons: Sequence[int],
    controls: Sequence[str] | None = None,
    maxlags: int | str = "horizon",
    return_prefix: str = RETURN_PREFIX,
) -> pd.DataFrame:
    """One row per measure, one column block per horizon.

    Useful for putting a new measure next to the published Manager Sentiment
    Index. Read the ``nobs`` columns: the measures rarely share a sample.
    """
    blocks = []
    for sentiment in sentiments:
        _, table = predictive_regression(
            data, sentiment, horizons, controls=controls, maxlags=maxlags, return_prefix=return_prefix
        )
        block = table.copy()
        block.index = pd.MultiIndex.from_product([[sentiment], block.index])
        blocks.append(block)
    return pd.concat(blocks)


# ---------------------------------------------------------------------------
# Marginal effects
# ---------------------------------------------------------------------------


def marginal_effect(
    results,
    sentiment: str,
    moderator: str,
    interaction: str,
    quantiles: Sequence[float] = (10, 50, 90),
    n_points: int = 100,
    controls: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Predicted return across sentiment values, at several moderator levels.

    Both the sentiment grid and the moderator levels come from the **regression's
    own sample** (``results.model.data.frame``), so the plotted percentiles
    describe the data the coefficients were estimated on. Taking them from
    another frame -- a daily VIX series when the regression used monthly averages,
    say -- overstates the range.

    Controls not given in ``controls`` are held at their in-sample mean.
    """
    sample = results.model.data.frame
    exog_names = [n for n in results.model.exog_names if n != "Intercept"]

    grid = np.linspace(sample[sentiment].min(), sample[sentiment].max(), n_points)
    held = {name: float(sample[name].mean()) for name in exog_names}
    held.update({k: float(v) for k, v in (controls or {}).items()})

    frames = []
    for q in quantiles:
        level = float(np.percentile(sample[moderator].dropna(), q))
        scenario = pd.DataFrame({name: np.full(n_points, held[name]) for name in exog_names})
        scenario[sentiment] = grid
        scenario[moderator] = level
        scenario[interaction] = grid * level
        frames.append(
            pd.DataFrame(
                {
                    sentiment: grid,
                    "moderator_q": q,
                    "moderator_level": level,
                    "predicted": results.predict(scenario).to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def plot_marginal_effect(
    effects: pd.DataFrame,
    sentiment: str,
    moderator_name: str,
    ylabel: str = "Predicted cumulative excess return",
    ax=None,
):
    """Plot the output of :func:`marginal_effect`."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    for q, group in effects.groupby("moderator_q"):
        level = group["moderator_level"].iloc[0]
        ax.plot(group[sentiment], group["predicted"], linewidth=2,
                label=f"{moderator_name} = {level:.2f} ({int(q)}th pct)")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(sentiment)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Marginal effect of sentiment at different {moderator_name} levels")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------


def correlation_with_pvalues(data: pd.DataFrame, columns: Sequence[str], method: str = "pearson") -> pd.DataFrame:
    """Pairwise correlations annotated with significance stars and pair-wise N."""
    from scipy import stats

    test = {"pearson": stats.pearsonr, "spearman": stats.spearmanr}[method]
    out = pd.DataFrame("", index=list(columns), columns=list(columns), dtype=object)

    for i, left in enumerate(columns):
        for j, right in enumerate(columns):
            if i == j:
                out.iloc[i, j] = "1.00"
                continue
            pair = data[[left, right]].dropna()
            if len(pair) < 3:
                out.iloc[i, j] = "n/a"
                continue
            corr, pval = test(pair[left], pair[right])
            out.iloc[i, j] = f"{corr:.2f}{significance_stars(pval)} (n={len(pair)})"
    return out
