# Manager sentiment

**Authors:** Fabian Slonimczyk (ICEF, Higher School of Economics) and
Natalia Vasilyeva

This repository builds report-level measures of managers' growth outlook and
perceived financial risk from the text of SEC filings, aggregates those
measures into monthly time series, and studies whether sentiment predicts
market returns.

The workflow is organized as three notebook stages:

1. **Extract sentiment** from 10-K and 10-Q filing text using a language model.
2. **Aggregate** report scores into monthly manager-sentiment series.
3. **Analyze** the series with descriptive statistics and predictive
   regressions.

The notebooks are the main entry points; reusable processing and analysis code
lives in `code/scripts/`. Example filing text and precomputed result files are
also included. Model weights are not included.

## Repository structure

| Path | Purpose |
|---|---|
| `code/document processing/run_extraction.ipynb` | Runs report-level sentiment extraction over filing text. |
| `code/time aggregation/build_manager_sentiment.ipynb` | Builds monthly sentiment series from report-level scores. |
| `code/analysis/macro_predictive_regressions.ipynb` | Compares sentiment measures and runs return-prediction analyses. |
| `code/scripts/` | Shared implementations used by the notebooks: filing discovery and parsing, prompts and scoring, extraction, monthly aggregation, market-data preparation, econometrics, and utilities. |
| `code/scripts/tests/` | Unit and workflow tests for scoring, extraction, aggregation, and regression helpers. |
| `input_data/` | Example filing text and input datasets used for analysis, including market predictors, volatility, recession indicators, and comparison sentiment indices. |
| `sentiment_extraction_results/` | Report-level scores, word counts, and extraction metadata. |
| `manager_sentiment_results/` | Monthly sentiment time series. |
| `requirements.txt` | Python dependencies for extraction, aggregation, and analysis. |

The repository includes a small sample of filing texts for testing; see
[Input text data](#input-text-data) for its source and scope.

## Setup

Use Python with the packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The workflow is notebook-based, so open the repository in Jupyter or VS Code
with Jupyter support. If Jupyter is not already available in your environment,
install it separately. The extraction stage depends on PyTorch and Hugging
Face Transformers; choose a PyTorch build compatible with your hardware and
CUDA setup if you plan to use a GPU.

### Language models

The extraction notebook supports:

- **Llama 3.1 8B** (`meta-llama/Meta-Llama-3.1-8B`), a gated model. You must
  accept its Hugging Face license and authenticate before downloading or
  loading it.
- **ModernBERT-large** (`answerdotai/ModernBERT-large`), a public model.

Download model weights outside the repository, then set `MODEL_ID` in the
extraction notebook to either the Hugging Face model ID or the local model
directory. The optional download commands and model-specific notes are in
[`code/scripts/download_models.sh`](code/scripts/download_models.sh). The
`models/` directory is ignored by Git.

## Run the workflow

Run the notebooks from the repository checkout, in order. The notebooks add
`code/scripts/` to the Python import path and use repository-relative data and
output locations.

### 1. Extract report-level sentiment

Open and run [`run_extraction.ipynb`](code/document%20processing/run_extraction.ipynb).
Set `MODEL_ID` and `STRATEGY` in its configuration cell. Available strategies
include `growth` and `financial_risk`; growth also has a legacy causal-model
variant in the shared prompt definitions.

Each filing is split into token-budgeted chunks. The model scores a prompted
readout for each chunk, and the chunk scores are averaged into one report-level
row. The output contains positive and negative readout probabilities and
`polarity` (positive minus negative). The financial-risk strategy treats
low-risk language as positive, so higher polarity consistently indicates
better news for the firm.

Start with the notebook's small smoke test before running the full sample.
Extraction is resumable: filings already completed are skipped on subsequent
runs. Model context size, chunking, and runtime settings can be adjusted in the
notebook configuration.

### 2. Build monthly time series

Open and run
[`build_manager_sentiment.ipynb`](code/time%20aggregation/build_manager_sentiment.ipynb).
By default it reads the full report-level table and writes the monthly series.
The notebook also shows how to switch to the sample-matched table.

The default construction takes an equal-weight mean within each calendar
month, computes a backward four-month moving average, and standardizes that
average over the full sample. The raw monthly mean is retained alongside the
normalized series. The aggregation helpers also support report weights,
within-form standardization, and alternative smoothing options; the notebook
documents choices and their consequences.

### 3. Analyze sentiment and returns

Open and run
[`macro_predictive_regressions.ipynb`](code/analysis/macro_predictive_regressions.ipynb).
It uses the monthly series and market-data inputs in `input_data/` to compare
measures, calculate return horizons, and run Newey-West predictive regressions.
It also examines the original 2003–2014 sample period and interactions with
uncertainty measures, recessions, and the two sentiment dimensions.

## Data and outputs

The repository's data files fall into four stages of the workflow:

### Input text data

`input_data/` includes 100 SEC filing texts (50 10-K Item 7 sections and 50
10-Q filings) for testing the extraction workflow. They are publicly available
from the [Notre Dame Securities Research Initiative (SRAF) cleaned SEC EDGAR
10-X files](https://sraf.nd.edu/sec-edgar-data/cleaned-10x-files/). This small
test sample is not the sample used in the paper; see the paper for the larger
research sample and its selection criteria. A 10-K text file contains Item 7
even though its filename does not include an `_item7` suffix.

We thank Bill McDonald and the Notre Dame Securities Research Initiative for
making these cleaned filing texts available to researchers.

### Intermediate results: report-level scores

`sentiment_extraction_results/` contains intermediate extraction results:

- `sent_growth_fin_risk_10k_10q.parquet` — report-level scores for the larger
  scored table.
- `sample_filings.parquet` — scored rows corresponding to the filing texts
  included in `input_data/`.
- `filing_word_counts.parquet` — filing word counts from the
  Loughran–McDonald summaries.
- CSV files and their `.meta.json` sidecars — extraction outputs and run
  metadata.

### Additional sources for analysis

These datasets in `input_data/` support the market-return analysis and
comparison measures:

| File | Source and use |
|---|---|
| `PredictorData2024.xlsx` | Updated Goyal–Welch equity-premium predictor data, used to construct market returns and regression controls. See Welch and Goyal, [“A Comprehensive Look at the Empirical Performance of Equity Premium Prediction”](https://doi.org/10.1093/rfs/hhm014) (*Review of Financial Studies*, 2008). |
| `VIX_History.csv` | Daily VIX close observations from [Cboe historical data](https://www.cboe.com/tradable-products/vix/vix-historical-data/); the analysis code averages daily closes by month. |
| `nvix_and_categories_timeseries_mar2016.xlsx` | Monthly news-implied volatility (NVIX) and category series from Manela and Moreira, [“News Implied Volatility and Disaster Concerns”](https://doi.org/10.1016/j.jfineco.2016.01.032) (*Journal of Financial Economics*, 2017). |
| `SENTIMENT.xlsx` | Baker–Wurgler investor-sentiment data; this project uses its `recess` field as a recession indicator. Cite Baker and Wurgler, [“Investor Sentiment and the Cross-Section of Stock Returns”](https://doi.org/10.1111/j.1540-6261.2006.00885.x) (*Journal of Finance*, 2006). |
| `JLMZ_MS_Index2017.xlsx` | Published Manager Sentiment Index used as a comparison series. See Jiang, Lee, Martin, and Zhou, [“Manager Sentiment and Stock Returns”](https://doi.org/10.1016/j.jfineco.2018.10.001) (*Journal of Financial Economics*, 2019). |
<!-- | `finaldata_10k.sas7bdat` | Supplemental market-cap data for optional 10-K weighting (the notebook notes coverage only through 2019). It is believed to come from a replication package for a *Management Science* paper by Frankel et al.; the exact paper and package citation should be confirmed before citing or redistributing the file. It is not used in the default aggregation. | -->

### Final results: monthly sentiment series

`manager_sentiment_results/` contains the time-aggregated outputs:

- `manager_sentiment_monthly.parquet` — monthly series built from the larger
  report-level table.
- `manager_sentiment_monthly_sample.parquet` — the corresponding series for
  the 100 filing texts included for testing.

## Tests

Run the repository's tests with pytest:

```bash
python -m pytest code/scripts/tests
```

The tests cover scoring rules, report discovery and extraction flow, time
aggregation, and econometric helpers. The scoring and analysis helpers are
designed to be usable without loading model weights; running extraction on
filings does require the selected model.
