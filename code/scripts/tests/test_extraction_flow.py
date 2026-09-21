"""Tests for the extraction run loop, with the model stubbed out.

Covers the parts that are easy to get quietly wrong -- chunk-to-document
alignment, resume, failure recording, OOM backoff -- without needing a GPU or
even torch. Only ``ScoringBackend``, the splitter and the verbalizer map are
faked; ``run_extraction`` itself is the real code.

Run with plain Python::

    python code/scripts/tests/test_extraction_flow.py
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Stub torch / transformers / langchain before importing the module under test.
# ---------------------------------------------------------------------------


class FakeOutOfMemoryError(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError."""


class FakeDataLoader:
    """Sequential batching plus collate_fn -- all run_extraction uses."""

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn or list

    def __iter__(self):
        batch = []
        for i in range(len(self.dataset)):
            batch.append(self.dataset[i])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch:
            yield self.collate_fn(batch)


def install_stubs() -> None:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        empty_cache=lambda: None,
        manual_seed=lambda seed: None,
        manual_seed_all=lambda seed: None,
        OutOfMemoryError=FakeOutOfMemoryError,
    )
    torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=False)
    )
    torch.manual_seed = lambda seed: None
    torch.bfloat16 = "bfloat16"
    torch.nn = types.SimpleNamespace(DataParallel=object)
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.DataLoader = FakeDataLoader
    sys.modules.update(
        {"torch": torch, "torch.utils": torch.utils, "torch.utils.data": torch.utils.data}
    )

    transformers = types.ModuleType("transformers")
    for name in ("AutoConfig", "AutoModelForCausalLM", "AutoModelForMaskedLM", "AutoTokenizer"):
        setattr(transformers, name, object)
    transformers.logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
    sys.modules["transformers"] = transformers

    splitters = types.ModuleType("langchain_text_splitters")
    splitters.RecursiveCharacterTextSplitter = object
    sys.modules["langchain_text_splitters"] = splitters


install_stubs()
sys.path.insert(0, str(SCRIPTS))

import pandas as pd  # noqa: E402

import sentiment_extraction as se  # noqa: E402

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

CHARS_PER_CHUNK = 300

FIXTURES = {
    "20130726_10-K_edgar_data_1750_0001047469-13-007797_item7.txt": "Revenues grew strongly. " * 40,
    "20140717_10-K_edgar_data_1750_0001047469-14-006243_item7.txt": "Margins declined sharply. " * 40,
    "20030414_10-Q_edgar_data_1750_0001104659-03-006403.txt": "Quarterly results were solid. " * 40,
    "20031014_10-Q_edgar_data_1750_0001104659-03-022660.txt": "short",
}


class FakeSplitter:
    def split_text(self, text: str) -> list[str]:
        return [text[i : i + CHARS_PER_CHUNK] for i in range(0, len(text), CHARS_PER_CHUNK)]


class FakeBackend:
    """Scores each chunk from its own wording, so misalignment is detectable."""

    kind = "causal"
    window = 4096
    mask_token = None
    tokenizer = types.SimpleNamespace(encode=lambda *a, **k: [0] * 5)
    calls: list[int] = []

    def __init__(self, cfg=None) -> None:
        pass

    def score_prompts(self, prompts, vmap, top_p):
        type(self).calls.append(len(prompts))
        scores = []
        for prompt in prompts:
            positive = 0.6 if ("grew strongly" in prompt or "solid" in prompt) else 0.1
            negative = 0.7 - positive
            matched = positive + negative
            scores.append(
                {
                    "positive": positive,
                    "negative": negative,
                    "polarity": positive - negative,
                    "matched_mass": matched,
                    "polarity_norm": (positive - negative) / matched,
                }
            )
        return scores


def patch_module(backend_cls=FakeBackend) -> None:
    se.ScoringBackend = backend_cls
    se.build_verbalizer_map = lambda tokenizer, verbalizer, **kw: se.VerbalizerMap(
        classes=None, n_positive_ids=41, n_negative_ids=43
    )
    se.make_splitter = lambda tokenizer, size, overlap: FakeSplitter()
    se.resolve_chunk_size = lambda cfg, backend: (backend.window - 1000, 1000)


def make_config(input_dir: Path, output_dir: Path, **overrides) -> se.ExtractionConfig:
    settings = dict(
        model_id="fake/model",
        strategy="growth",
        backend="causal",
        input_dir=input_dir,
        output_dir=output_dir,
        docs_per_batch=2,
        save_every=1000,
        min_chars=200,
    )
    settings.update(overrides)
    return se.ExtractionConfig(**settings)


def read_failures(cfg: se.ExtractionConfig) -> pd.DataFrame:
    path = cfg.resolve_output_path()
    failures = path.with_name(path.stem + ".failures.csv")
    return pd.read_csv(failures) if failures.exists() else pd.DataFrame(columns=["reason"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scores_both_form_types_in_one_run(input_dir: Path, output_dir: Path) -> None:
    patch_module()
    cfg = make_config(input_dir, output_dir)
    result = se.run_extraction(cfg)

    assert len(result) == 3, f"expected 3 scored filings, got {len(result)}"
    assert set(result["FORM_TYPE"]) == {"10-K", "10-Q"}
    assert list(result.columns) == list(se.OUTPUT_COLUMNS)
    assert result["polarity"].notna().all()
    assert (result["n_chunks"] == 4).all(), "chunk count must follow the splitter"
    assert (result["n_words"] > 0).all(), "word counts belong in the output"


def test_chunks_stay_attached_to_their_document(input_dir: Path, output_dir: Path) -> None:
    patch_module()
    result = se.run_extraction(make_config(input_dir, output_dir))
    polarity = dict(zip(result["FILING_DATE"], result["polarity"]))

    assert polarity["20130726"] > 0, "the optimistic 10-K must score positive"
    assert polarity["20140717"] < 0, "the pessimistic 10-K must score negative"
    assert polarity["20030414"] > 0, "the optimistic 10-Q must score positive"


def test_unscorable_filings_are_recorded(input_dir: Path, output_dir: Path) -> None:
    patch_module()
    cfg = make_config(input_dir, output_dir)
    se.run_extraction(cfg)

    failures = read_failures(cfg)
    assert len(failures) == 1, f"expected the short filing to fail, got {len(failures)}"
    assert "too_short" in failures["reason"].iloc[0]
    assert failures["ACC_NUM"].iloc[0] == "0001104659-03-022660"


def test_resume_is_exact_and_idempotent(input_dir: Path, output_dir: Path) -> None:
    patch_module()
    se.run_extraction(make_config(input_dir, output_dir))

    FakeBackend.calls.clear()
    again = se.run_extraction(make_config(input_dir, output_dir))
    assert FakeBackend.calls == [], "a finished run must not score anything again"
    assert len(again) == 3
    assert not again.duplicated(list(se.DOC_KEY_COLUMNS)).any()


def test_resume_fills_only_the_missing_rows(input_dir: Path, output_dir: Path) -> None:
    patch_module()
    cfg = make_config(input_dir, output_dir)
    se.run_extraction(cfg)

    output_path = cfg.resolve_output_path()
    pd.read_csv(output_path).iloc[:2].to_csv(output_path, index=False)

    FakeBackend.calls.clear()
    result = se.run_extraction(make_config(input_dir, output_dir))
    assert FakeBackend.calls, "the deleted filing must be re-scored"
    assert len(result) == 3
    assert not result.duplicated(list(se.DOC_KEY_COLUMNS)).any()


def test_oom_shrinks_the_forward_batch() -> None:
    class OomOverTwo(FakeBackend):
        calls: list[int] = []

        def score_prompts(self, prompts, vmap, top_p):
            type(self).calls.append(len(prompts))
            if len(prompts) > 2:
                raise FakeOutOfMemoryError("out of memory")
            return FakeBackend.score_prompts(self, prompts, vmap, top_p)

    backend = OomOverTwo()
    prompts = [f"chunk {i} solid" for i in range(7)]
    scores = se._forward_in_slices(backend, prompts, None, 0.9, 8)

    assert len(scores) == 7, "no chunk may be lost to an OOM retry"
    assert max(OomOverTwo.calls[1:]) <= 4, "the forward batch must shrink"


def test_lost_mask_raises_instead_of_misaligning(input_dir: Path, output_dir: Path) -> None:
    class MaskLost(FakeBackend):
        def score_prompts(self, prompts, vmap, top_p):
            raise se.MaskReadoutError("prompt ending was truncated")

    patch_module(MaskLost)
    try:
        se.run_extraction(make_config(input_dir, output_dir))
    except se.MaskReadoutError:
        return
    raise AssertionError("a lost mask must abort the run, not produce shifted scores")


def test_score_count_mismatch_is_refused(input_dir: Path, output_dir: Path) -> None:
    class DropsOneScore(FakeBackend):
        def score_prompts(self, prompts, vmap, top_p):
            return FakeBackend.score_prompts(self, prompts, vmap, top_p)[:-1]

    patch_module(DropsOneScore)
    cfg = make_config(input_dir, output_dir)
    result = se.run_extraction(cfg)

    assert result.empty, "scores that cannot be aligned must not be written"
    reasons = read_failures(cfg)["reason"]
    assert reasons.str.contains("refusing to align").any()


# ---------------------------------------------------------------------------


def main() -> None:
    needs_dirs = [
        test_scores_both_form_types_in_one_run,
        test_chunks_stay_attached_to_their_document,
        test_unscorable_filings_are_recorded,
        test_resume_is_exact_and_idempotent,
        test_resume_fills_only_the_missing_rows,
        test_lost_mask_raises_instead_of_misaligning,
        test_score_count_mismatch_is_refused,
    ]

    count = 0
    for func in needs_dirs:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()
            for name, body in FIXTURES.items():
                (input_dir / name).write_text(body)
            func(input_dir, Path(tmp) / "output")
        print(f"  [ok] {func.__name__}")
        count += 1

    test_oom_shrinks_the_forward_batch()
    print(f"  [ok] {test_oom_shrinks_the_forward_batch.__name__}")
    count += 1

    print(f"\n{count} test(s) passed.")


if __name__ == "__main__":
    main()
