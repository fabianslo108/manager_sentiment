"""Report-level sentiment extraction -- one script for 10-Ks and 10-Qs.

Pipeline, following the chunked 10-Q procedure of the earlier project and
applying it to both form types:

    filing text -> token-budgeted chunks -> prompt per chunk -> readout of one
    token position -> nucleus (top-p) -> verbalizer mass -> mean over chunks

Two model families are supported through one code path:

``causal`` (e.g. Llama-3.1-8B)
    the ending is appended and the *next-token* distribution is read;
``mlm`` (e.g. ModernBERT-large)
    the ending contains a ``[MASK]`` and that position's distribution is read.


Run from a notebook via :func:`run_extraction`, or from the shell::

    python sentiment_extraction.py --model-id meta-llama/Meta-Llama-3.1-8B \
        --strategy growth --form-type 10-Q --limit 50
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from transformers import logging as hf_logging

from prompts import PromptStrategy, get_strategy, normalize_token
from reports import (
    SECTION_DEFAULT,
    ReportDataset,
    ReportItem,
    collate_items,
    discover_reports,
    reports_from_index,
)
from scoring import aggregate_chunk_scores
from utils import (
    DOC_KEY_COLUMNS,
    INPUT_DATA_DIR,
    SENTIMENT_RESULTS_DIR,
    clean_mem,
    get_logger,
    set_seed,
)

logger = get_logger(__name__)

CAUSAL = "causal"
MLM = "mlm"

#: Reserve left for the prompt ending inside the model window, per backend.
#: Mirrors the earlier 10-Q runs (1000 for Llama, 100 for ModernBERT).
DEFAULT_PROMPT_RESERVE = {CAUSAL: 1000, MLM: 100}

#: A causal forward pass can be asked for a single logit row, so it can take
#: more chunks at once than an MLM pass, which materializes (B, L, V) logits.
DEFAULT_CHUNKS_PER_FORWARD = {CAUSAL: 8, MLM: 2}

META_OUTPUT_COLUMNS = (
    *DOC_KEY_COLUMNS,
    "SECTION",
    "strategy",
    "model",
    "backend",
)

SCORE_OUTPUT_COLUMNS = (
    "positive",
    "negative",
    "polarity",
    "n_chars",
    "n_chars_scored",
    "decode_fallback",
)

OUTPUT_COLUMNS = (*META_OUTPUT_COLUMNS, *SCORE_OUTPUT_COLUMNS)


class MaskReadoutError(RuntimeError):
    """An MLM prompt lost its mask token (or gained a second one)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExtractionConfig:
    """Everything one extraction run needs. Serialized next to the results."""

    model_id: str
    strategy: str = "growth"
    backend: str = "auto"  # "causal" | "mlm" | "auto"

    # --- sample -----------------------------------------------------------
    input_dir: Path = INPUT_DATA_DIR
    form_type: str | None = None  # None = both 10-K and 10-Q
    section: str | None = SECTION_DEFAULT
    index_path: Path | None = None  # optional explicit sample index (CSV/TSV)
    limit: int | None = None

    # --- text handling ----------------------------------------------------
    chunk_size_tokens: int | None = None  # None -> window - prompt_reserve_tokens
    prompt_reserve_tokens: int | None = None  # None -> DEFAULT_PROMPT_RESERVE
    chunk_overlap_tokens: int = 240
    max_chars: int | None = None  # document-level character cap
    min_chars: int = 200

    # --- batching ---------------------------------------------------------
    docs_per_batch: int = 2
    max_chunks_per_forward: int | None = None  # None -> DEFAULT_CHUNKS_PER_FORWARD
    num_workers: int = 0

    # --- scoring ----------------------------------------------------------
    top_p: float = 0.9

    # --- runtime ----------------------------------------------------------
    device: str | None = None
    dtype: str = "bfloat16"
    data_parallel: bool = True
    seed: int = 42

    # --- output -----------------------------------------------------------
    output_dir: Path = SENTIMENT_RESULTS_DIR
    output_name: str | None = None
    save_every: int = 500
    resume: bool = True

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.index_path is not None:
            self.index_path = Path(self.index_path)
        if self.backend not in (CAUSAL, MLM, "auto"):
            raise ValueError(f"backend must be 'causal', 'mlm' or 'auto', got {self.backend!r}")
        if not 0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    @property
    def model_tag(self) -> str:
        """Filesystem-safe short name of the model."""
        return Path(self.model_id.rstrip("/")).name.replace(".", "_")

    def resolve_output_path(self) -> Path:
        if self.output_name:
            return self.output_dir / self.output_name
        form_tag = (self.form_type or "10-K_10-Q").replace("-", "")
        return self.output_dir / f"{self.strategy}__{self.model_tag}__{form_tag}.csv"

    def to_json(self) -> str:
        payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        return json.dumps(payload, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Verbalizer -> vocabulary map
# ---------------------------------------------------------------------------


@dataclass
class VerbalizerMap:
    """Which vocabulary ids count as positive (1) / negative (2) / neither (0)."""

    classes: torch.Tensor  # (vocab_size,) int8
    n_positive_ids: int
    n_negative_ids: int
    unreachable_words: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"{self.n_positive_ids} positive / {self.n_negative_ids} negative vocabulary ids"]
        for polarity, words in self.unreachable_words.items():
            if words:
                parts.append(f"{len(words)} {polarity} word(s) unreachable as a single token")
        return "; ".join(parts)


def build_verbalizer_map(tokenizer, verbalizer, decode_batch: int = 20_000) -> VerbalizerMap:
    """Map every vocabulary id to a verbalizer polarity.

    This is the single definition of "which tokens count", shared by the
    vectorized scorer and by :func:`scoring.score_token_probs`: both normalize
    the decoded token with :func:`prompts.normalize_token`.

    A verbalizer word that no single token decodes to is *unreachable* with a
    one-token readout and contributes nothing -- reported so it is visible
    rather than quietly ignored.
    """
    vocab_size = len(tokenizer)
    classes = torch.zeros(vocab_size, dtype=torch.int8)

    positive = verbalizer["positive"]
    negative = verbalizer["negative"]
    seen_positive: set[str] = set()
    seen_negative: set[str] = set()

    for start in range(0, vocab_size, decode_batch):
        ids = list(range(start, min(start + decode_batch, vocab_size)))
        # batch_decode on single-id sequences gives the token's surface form,
        # including its leading space, unlike convert_ids_to_tokens which keeps
        # the tokenizer's internal markers.
        for offset, decoded in enumerate(tokenizer.batch_decode([[i] for i in ids])):
            word = normalize_token(decoded)
            if not word:
                continue
            if word in positive:
                classes[ids[offset]] = 1
                seen_positive.add(word)
            elif word in negative:
                classes[ids[offset]] = 2
                seen_negative.add(word)

    return VerbalizerMap(
        classes=classes,
        n_positive_ids=int((classes == 1).sum()),
        n_negative_ids=int((classes == 2).sum()),
        unreachable_words={
            "positive": sorted(positive - seen_positive),
            "negative": sorted(negative - seen_negative),
        },
    )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def resolve_backend_kind(model_id: str, requested: str = "auto") -> str:
    """Decide whether a checkpoint is read as a causal LM or a masked LM."""
    if requested != "auto":
        return requested
    config = AutoConfig.from_pretrained(model_id)
    architectures = " ".join(getattr(config, "architectures", None) or [])
    if "ForMaskedLM" in architectures:
        return MLM
    if "ForCausalLM" in architectures:
        return CAUSAL
    raise ValueError(
        f"cannot tell the backend for {model_id!r} (architectures={architectures!r}); "
        "pass backend='causal' or backend='mlm' explicitly"
    )


class ScoringBackend:
    """Loads a model and reads the verbalizer mass at one token position."""

    def __init__(self, cfg: ExtractionConfig) -> None:
        hf_logging.set_verbosity_error()

        self.kind = resolve_backend_kind(cfg.model_id, cfg.backend)
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Causal readout takes the last position, so padding must go on the left.
        # A masked LM is read at the mask position, so right padding is natural.
        self.tokenizer.padding_side = "left" if self.kind == CAUSAL else "right"

        loader = AutoModelForCausalLM if self.kind == CAUSAL else AutoModelForMaskedLM
        model = loader.from_pretrained(cfg.model_id, torch_dtype=dtype)
        model.eval()

        config = AutoConfig.from_pretrained(cfg.model_id)
        self.window = int(config.max_position_embeddings)

        if self.kind == MLM and self.tokenizer.mask_token_id is None:
            raise ValueError(f"{cfg.model_id!r} has no mask token but backend is 'mlm'")

        self.n_gpus = torch.cuda.device_count() if self.device.startswith("cuda") else 0
        if cfg.data_parallel and self.n_gpus > 1:
            logger.info("wrapping model in DataParallel over %d GPUs", self.n_gpus)
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

        # Ask for a single logit row when the installed transformers supports it;
        # full (B, L, V) causal logits are the main source of OOM here.
        self._logits_kwarg = self._detect_logits_kwarg()
        logger.info(
            "loaded %s as %s backend on %s (window=%d, vocab=%d)",
            cfg.model_id,
            self.kind,
            self.device,
            self.window,
            len(self.tokenizer),
        )

    # -- internals --------------------------------------------------------

    def _detect_logits_kwarg(self) -> str | None:
        if self.kind != CAUSAL:
            return None
        import inspect

        target = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        params = inspect.signature(target.forward).parameters
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in params:
                return name
        return None

    @property
    def mask_token(self) -> str | None:
        return self.tokenizer.mask_token

    def _encode(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,          # pad to the longest in the batch, not to the window
            truncation=True,
            max_length=self.window,
        ).to(self.device)

    def _readout_logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Logits at the single position we score, shape (batch, vocab)."""
        if self.kind == CAUSAL:
            kwargs = {self._logits_kwarg: 1} if self._logits_kwarg else {}
            with torch.no_grad():
                logits = self.model(**inputs, **kwargs).logits
            return logits[:, -1, :]

        mask_id = self.tokenizer.mask_token_id
        is_mask = inputs["input_ids"] == mask_id
        counts = is_mask.sum(dim=1)
        if not bool((counts == 1).all()):
            bad = (counts != 1).nonzero(as_tuple=True)[0].tolist()
            raise MaskReadoutError(
                f"{len(bad)} of {counts.numel()} prompt(s) do not contain exactly one mask "
                f"(rows {bad[:10]}, counts {counts[bad[:10]].tolist()}): the prompt ending was "
                "truncated -- increase prompt_reserve_tokens or lower chunk_size_tokens"
            )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        # Integer gather, not boolean indexing: this cannot silently reorder or
        # drop rows even if the mask layout changes.
        rows = torch.arange(logits.shape[0], device=logits.device)
        cols = is_mask.float().argmax(dim=1)
        return logits[rows, cols, :]

    # -- public API -------------------------------------------------------

    def score_prompts(
        self, prompts: Sequence[str], vmap: VerbalizerMap, top_p: float
    ) -> list[dict[str, float]]:
        """Verbalizer mass inside the top-p nucleus, one dict per prompt."""
        if not prompts:
            return []

        inputs = self._encode(prompts)
        logits = self._readout_logits(inputs)
        del inputs

        probs = torch.softmax(logits.float(), dim=-1)
        del logits

        sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)
        del probs

        if top_p >= 1.0:
            keep = torch.ones_like(sorted_probs, dtype=torch.bool)
        else:
            cumulative = sorted_probs.cumsum(dim=-1)
            # Smallest prefix reaching top_p, always at least one token.
            keep = cumulative - sorted_probs < top_p

        classes = vmap.classes.to(sorted_ids.device)[sorted_ids]
        kept = sorted_probs * keep
        positive = (kept * (classes == 1)).sum(dim=-1)
        negative = (kept * (classes == 2)).sum(dim=-1)

        return [
            {"positive": pos, "negative": neg, "polarity": pos - neg}
            for pos, neg in zip(positive.tolist(), negative.tolist())
        ]

    def top_tokens(self, prompt: str, k: int = 20) -> list[tuple[str, float]]:
        """Most likely answer tokens for one prompt -- for prompt engineering."""
        inputs = self._encode([prompt])
        logits = self._readout_logits(inputs)
        probs = torch.softmax(logits.float(), dim=-1)[0]
        top = torch.topk(probs, k)
        return [
            (self.tokenizer.decode([int(i)]), float(p))
            for i, p in zip(top.indices.tolist(), top.values.tolist())
        ]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def resolve_chunk_size(cfg: ExtractionConfig, backend: ScoringBackend) -> tuple[int, int]:
    """Return ``(chunk_size_tokens, prompt_reserve_tokens)``."""
    reserve = cfg.prompt_reserve_tokens
    if reserve is None:
        reserve = DEFAULT_PROMPT_RESERVE[backend.kind]
    if cfg.chunk_size_tokens is not None:
        chunk_size = cfg.chunk_size_tokens
        if chunk_size + reserve > backend.window:
            raise ValueError(
                f"chunk_size_tokens={chunk_size} plus prompt_reserve_tokens={reserve} "
                f"exceeds the model window ({backend.window})"
            )
    else:
        chunk_size = backend.window - reserve
    if chunk_size <= 0:
        raise ValueError(f"chunk size resolved to {chunk_size}")
    return chunk_size, reserve


def make_splitter(tokenizer, chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    overlap = min(overlap, max(chunk_size // 4, 0))
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------


def _key_frame(rows: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, rows[list(DOC_KEY_COLUMNS)].astype(str).itertuples(index=False, name=None)))


def load_done_keys(output_path: Path) -> set[tuple]:
    """Filing keys already present in an earlier run's output."""
    if not output_path.exists():
        return set()
    done = pd.read_csv(output_path, dtype=str)
    missing = [c for c in DOC_KEY_COLUMNS if c not in done.columns]
    if missing:
        logger.warning("existing output %s lacks key column(s) %s; ignoring it", output_path, missing)
        return set()
    keys = _key_frame(done)
    logger.info("resume: %d filing(s) already scored in %s", len(keys), output_path.name)
    return keys


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def build_report_table(cfg: ExtractionConfig) -> pd.DataFrame:
    """Resolve the sample, either from an index table or by directory scan."""
    if cfg.index_path is not None:
        if cfg.form_type is None:
            raise ValueError("form_type is required when an index_path is given")
        sep = "\t" if "".join(cfg.index_path.suffixes).startswith(".ts") else ","
        index = pd.read_csv(cfg.index_path, sep=sep, dtype=str)
        return reports_from_index(index, cfg.input_dir, cfg.form_type, cfg.section)
    return discover_reports(cfg.input_dir, cfg.form_type, cfg.section)


def _forward_in_slices(
    backend: ScoringBackend,
    prompts: list[str],
    vmap: VerbalizerMap,
    top_p: float,
    chunks_per_forward: int,
) -> list[dict[str, float]]:
    """Score all prompts, shrinking the forward batch when CUDA runs out."""
    scores: list[dict[str, float]] = []
    start = 0
    size = max(chunks_per_forward, 1)
    while start < len(prompts):
        window = prompts[start : start + size]
        try:
            scores.extend(backend.score_prompts(window, vmap, top_p))
        except torch.cuda.OutOfMemoryError:
            clean_mem()
            if size == 1:
                raise
            size = max(size // 2, 1)
            logger.warning("CUDA OOM: retrying with %d chunk(s) per forward pass", size)
            continue
        start += len(window)
    return scores


def run_extraction(cfg: ExtractionConfig) -> pd.DataFrame:
    """Run one extraction and return the report-level results."""
    set_seed(cfg.seed)

    strategy: PromptStrategy = get_strategy(cfg.strategy)
    output_path = cfg.resolve_output_path()
    failures_path = output_path.with_name(output_path.stem + ".failures.csv")
    meta_path = output_path.with_name(output_path.stem + ".meta.json")

    reports = build_report_table(cfg)
    if reports.empty:
        raise RuntimeError(f"no filings found for {cfg.form_type or 'any form type'} in {cfg.input_dir}")

    done_keys = load_done_keys(output_path) if cfg.resume else set()
    if done_keys:
        keys = reports[list(DOC_KEY_COLUMNS)].astype(str).agg(tuple, axis=1)
        reports = reports[~keys.isin(done_keys)].reset_index(drop=True)
    if cfg.limit is not None:
        reports = reports.head(cfg.limit).reset_index(drop=True)
    if reports.empty:
        logger.info("nothing left to score; returning the existing output")
        return pd.read_csv(output_path) if output_path.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)

    logger.info("scoring %d filing(s) with strategy %r", len(reports), strategy.name)

    backend = ScoringBackend(cfg)
    ending = strategy.ending(backend.kind, backend.mask_token)
    logger.info("prompt ending: %s", ending.replace("\n", " "))

    vmap = build_verbalizer_map(backend.tokenizer, strategy.verbalizer)
    logger.info("verbalizer: %s", vmap.summary())
    for polarity, words in vmap.unreachable_words.items():
        if words:
            logger.warning("%s words with no single-token form: %s", polarity, ", ".join(words))
    if vmap.n_positive_ids == 0 or vmap.n_negative_ids == 0:
        raise RuntimeError("verbalizer matched no vocabulary ids on one side; check the tokenizer")

    chunk_size, reserve = resolve_chunk_size(cfg, backend)
    ending_tokens = len(backend.tokenizer.encode(ending, add_special_tokens=False))
    if ending_tokens > reserve:
        raise ValueError(
            f"prompt ending needs {ending_tokens} tokens but only {reserve} are reserved"
        )
    logger.info(
        "chunking at %d token(s) (window %d - reserve %d), overlap %d",
        chunk_size,
        backend.window,
        reserve,
        cfg.chunk_overlap_tokens,
    )
    splitter = make_splitter(backend.tokenizer, chunk_size, cfg.chunk_overlap_tokens)

    chunks_per_forward = cfg.max_chunks_per_forward or DEFAULT_CHUNKS_PER_FORWARD[backend.kind]

    dataset = ReportDataset(reports, max_chars=cfg.max_chars, min_chars=cfg.min_chars)
    loader = DataLoader(
        dataset,
        batch_size=cfg.docs_per_batch,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_items,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(cfg.to_json())

    rows: list[dict] = []
    failures: list[dict] = []
    since_save = 0
    started = time.time()

    for batch in loader:
        usable = [item for item in batch if item.ok]
        for item in batch:
            if not item.ok:
                failures.append({**item.meta, "reason": item.error or "empty_text"})
        if not usable:
            continue

        try:
            prompts: list[str] = []
            spans: list[tuple[ReportItem, int, int]] = []
            for item in usable:
                chunks = splitter.split_text(item.text)
                if not chunks:
                    failures.append({**item.meta, "reason": "no_chunks"})
                    continue
                start = len(prompts)
                prompts.extend(
                    strategy.build_prompt(chunk, backend.kind, backend.mask_token) for chunk in chunks
                )
                spans.append((item, start, len(prompts)))

            if not prompts:
                continue

            scores = _forward_in_slices(backend, prompts, vmap, cfg.top_p, chunks_per_forward)
            if len(scores) != len(prompts):
                raise RuntimeError(
                    f"scored {len(scores)} of {len(prompts)} chunk(s); refusing to align them"
                )

            for item, start, stop in spans:
                aggregated = aggregate_chunk_scores(scores[start:stop])
                rows.append(
                    {
                        **item.meta,
                        "strategy": strategy.name,
                        "model": cfg.model_id,
                        "backend": backend.kind,
                        **aggregated,
                        "n_chars": item.n_chars,
                        "n_chars_scored": item.n_chars_scored,
                        "decode_fallback": item.decode_fallback,
                    }
                )
                since_save += 1

        except MaskReadoutError:
            raise  # a configuration error: fail loudly rather than lose filings
        except torch.cuda.OutOfMemoryError as exc:
            clean_mem()
            for item in usable:
                failures.append({**item.meta, "reason": f"cuda_oom: {exc}"})
            logger.warning("CUDA OOM on %d filing(s); recorded and moving on", len(usable))
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            for item in usable:
                failures.append({**item.meta, "reason": f"{type(exc).__name__}: {exc}"})
            logger.warning("batch failed (%s: %s); recorded and moving on", type(exc).__name__, exc)

        if since_save >= cfg.save_every:
            _flush(rows, failures, output_path, failures_path, done_keys)
            logger.info(
                "checkpoint: %d filing(s) scored, %d failure(s), %.1f min elapsed",
                len(rows),
                len(failures),
                (time.time() - started) / 60,
            )
            since_save = 0

        clean_mem()

    result = _flush(rows, failures, output_path, failures_path, done_keys)
    logger.info(
        "done: %d filing(s) scored, %d failure(s) in %.1f min -> %s",
        len(rows),
        len(failures),
        (time.time() - started) / 60,
        output_path,
    )
    return result


def _flush(
    rows: Sequence[dict],
    failures: Sequence[dict],
    output_path: Path,
    failures_path: Path,
    done_keys: set[tuple],
) -> pd.DataFrame:
    """Merge new rows into the output file and rewrite it atomically."""
    new = pd.DataFrame(list(rows))
    if not new.empty:
        new = new.reindex(columns=list(OUTPUT_COLUMNS))

    if done_keys and output_path.exists():
        previous = pd.read_csv(output_path)
        combined = pd.concat([previous, new], ignore_index=True) if not new.empty else previous
    else:
        combined = new

    if not combined.empty:
        combined = combined.drop_duplicates(list(DOC_KEY_COLUMNS), keep="last").reset_index(drop=True)
        _write_atomic(combined, output_path)

    if failures:
        _write_atomic(pd.DataFrame(list(failures)), failures_path)

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--strategy", default="growth")
    parser.add_argument("--backend", default="auto", choices=[CAUSAL, MLM, "auto"])
    parser.add_argument("--input-dir", type=Path, default=INPUT_DATA_DIR)
    parser.add_argument("--form-type", default=None, choices=["10-K", "10-Q"])
    parser.add_argument("--section", default=SECTION_DEFAULT)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size-tokens", type=int, default=None)
    parser.add_argument("--prompt-reserve-tokens", type=int, default=None)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=240)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--docs-per-batch", type=int, default=2)
    parser.add_argument("--max-chunks-per-forward", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-data-parallel", dest="data_parallel", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=SENTIMENT_RESULTS_DIR)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    section = None if args.section in ("none", "None", "") else args.section
    cfg = ExtractionConfig(**{**vars(args), "section": section})
    run_extraction(cfg)


if __name__ == "__main__":
    main()
