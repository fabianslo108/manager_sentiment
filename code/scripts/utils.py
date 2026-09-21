"""Shared helpers for the manager-sentiment pipeline.

Holds the things every stage needs -- repository paths, logging, reproducible
seeding, memory cleanup -- so that the stage-specific modules stay focused.

Deliberately torch-free: importing this module must work on a laptop without a
GPU stack, because the time-aggregation and analysis stages need the paths too.
"""

from __future__ import annotations

import gc
import logging
import os
import random
from pathlib import Path

import numpy as np

# code/scripts/utils.py -> code/scripts -> code -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_DATA_DIR = REPO_ROOT / "input_data"
SENTIMENT_RESULTS_DIR = REPO_ROOT / "sentiment_extraction_results"
MANAGER_SENTIMENT_DIR = REPO_ROOT / "manager_sentiment_results"

#: Natural key of a filing. Used instead of a positional row index so that
#: sentiment scores stay attached to the right report no matter how the input
#: table was built or ordered.
DOC_KEY_COLUMNS = ("CIK", "FILING_DATE", "ACC_NUM", "FORM_TYPE")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that prints once per record, even when re-imported."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed every RNG the pipeline touches, including torch when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_mem() -> None:
    """Collect garbage and release cached CUDA blocks."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def significance_stars(pval: float) -> str:
    """Conventional significance markers: 1%, 5%, 10%."""
    if pval <= 0.01:
        return "***"
    if pval <= 0.05:
        return "**"
    if pval <= 0.10:
        return "*"
    return ""


def doc_key(row: dict) -> tuple:
    """Natural key of one filing, as a hashable tuple of strings."""
    return tuple(str(row[col]) for col in DOC_KEY_COLUMNS)


def zscore(values, ddof: int = 0):
    """Standardize, ignoring NaN. Returns NaN everywhere if there is no spread."""
    import pandas as pd

    series = pd.Series(values)
    std = series.std(ddof=ddof)
    if not std or np.isnan(std):
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std
