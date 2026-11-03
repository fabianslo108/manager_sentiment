#!/usr/bin/env bash
# Model weights are not stored in this repository.
# Download them yourself into ./models and point the extraction notebook at that
# directory (or pass the Hugging Face id and let transformers cache the files).
#
# Llama 3.1 8B is a gated model. Accept the license at
# https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
# and log in before the download:
#
#   huggingface-cli login
#
# ModernBERT-large is public. Both checkpoints are what
# code/document processing/run_extraction.ipynb expects:
# meta-llama/Meta-Llama-3.1-8B as a causal LM, answerdotai/ModernBERT-large
# as a masked LM.
#
# From the repository root:
#
#   huggingface-cli download meta-llama/Meta-Llama-3.1-8B \
#       --local-dir models/Meta-Llama-3.1-8B
#
#   huggingface-cli download answerdotai/ModernBERT-large \
#       --local-dir models/ModernBERT-large
#
# Then set MODEL_ID in the notebook to the local directory, for example
# models/Meta-Llama-3.1-8B or models/ModernBERT-large.
