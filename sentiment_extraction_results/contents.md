# Sentiment Extraction Results

- Report level sentiment data
    - Sentiment
    - Word counts
- `sent_growth_fin_risk_10k_10q.parquet` — full scored sample
- `sample_filings.parquet` — rows of that table that match the filings in `input_data`
- `filing_word_counts.parquet` — `N_Words` and `N_Unique_Words` from the Loughran-McDonald summaries, one row per filing in the full table