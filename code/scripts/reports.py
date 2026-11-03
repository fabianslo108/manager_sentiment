"""Discovery and reading of 10-K / 10-Q filing texts.

One code path serves both form types. A filing is always identified by its
natural key ``(CIK, FILING_DATE, ACC_NUM, FORM_TYPE)``.

Expected file names:

    20130214_10-K_edgar_data_91440_0001193125-13-058193.txt
    20130726_10-K_edgar_data_1750_0001047469-13-007797_item7.txt
    20030414_10-Q_edgar_data_1750_0001104659-03-006403.txt

i.e. ``{FILING_DATE}_{FORM_TYPE}_edgar_data_{CIK}_{ACC_NUM}[_{SECTION}].txt``.

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

from utils import DOC_KEY_COLUMNS, get_logger

logger = get_logger(__name__)

META_COLUMNS = ("CIK", "FILING_DATE", "ACC_NUM", "FORM_TYPE", "SECTION")

DEFAULT_SECTION = {"10-K": "item7", "10-Q": None}

#: ``section`` sentinels accepted by the discovery helpers.
SECTION_DEFAULT = "default"
SECTION_ANY = "any"

FILENAME_RE = re.compile(
    r"^(?P<FILING_DATE>\d{8})"
    r"_(?P<FORM_TYPE>10-[KQ])"
    r"_edgar_data"
    r"_(?P<CIK>\d+)"
    r"_(?P<ACC_NUM>[0-9][0-9-]*)"
    r"(?:_(?P<SECTION>item\d+[a-z]?))?"
    r"\.txt$",
    re.IGNORECASE,
)

def parse_report_filename(filename: str) -> dict | None:
    """Parse one filing file name, or return ``None`` if it does not match."""
    match = FILENAME_RE.match(Path(filename).name)
    if match is None:
        return None
    parsed = match.groupdict()
    parsed["FORM_TYPE"] = parsed["FORM_TYPE"].upper()
    if parsed["SECTION"] is not None:
        parsed["SECTION"] = parsed["SECTION"].lower()
    return parsed


def report_filename(
    cik: str,
    filing_date: str,
    acc_num: str,
    form_type: str,
    section: str | None = None,
) -> str:
    """Inverse of :func:`parse_report_filename`."""
    stem = f"{filing_date}_{form_type}_edgar_data_{cik}_{acc_num}"
    if section:
        stem = f"{stem}_{section}"
    return f"{stem}.txt"


def _wanted_section(form_type: str, section: str | None) -> str | None:
    if section == SECTION_DEFAULT:
        return DEFAULT_SECTION.get(form_type)
    return section


def _section_accepted(parsed_section: str | None, form_type: str, section: str | None) -> bool:
    """Whether one file's section matches the requested filter."""
    if section == SECTION_ANY:
        return True
    wanted = _wanted_section(form_type, section)
    if section == SECTION_DEFAULT and form_type == "10-K" and wanted == "item7":
        return parsed_section in (None, "item7")
    return parsed_section == wanted


def _prefer_explicit_item7(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[int]] = {}
    for index, row in enumerate(rows):
        key = tuple(str(row[col]) for col in ("CIK", "FILING_DATE", "ACC_NUM", "FORM_TYPE"))
        grouped.setdefault(key, []).append(index)

    drop: set[int] = set()
    for indexes in grouped.values():
        if len(indexes) < 2:
            continue
        sections = {rows[i].get("SECTION") for i in indexes}
        if rows[indexes[0]]["FORM_TYPE"] == "10-K" and sections == {None, "item7"}:
            drop.update(i for i in indexes if rows[i].get("SECTION") is None)
    return [row for index, row in enumerate(rows) if index not in drop]


def _as_frame(rows: Sequence[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=[*META_COLUMNS, "path"])
    for col in META_COLUMNS:
        if col != "SECTION":
            frame[col] = frame[col].astype(str)
    return frame.sort_values(list(DOC_KEY_COLUMNS)).reset_index(drop=True)


def discover_reports(
    root: str | Path,
    form_type: str | Iterable[str] | None = None,
    section: str | None = SECTION_DEFAULT,
) -> pd.DataFrame:
    """Find filing texts under ``root`` by parsing their file names.

    Parameters
    ----------
    root:
        Directory to walk recursively.
    form_type:
        ``"10-K"``, ``"10-Q"``, an iterable of both, or ``None`` for any.
    section:
        ``"default"`` uses :data:`DEFAULT_SECTION` per form type. A 10-K matches
        either ``..._item7.txt`` or the plain filing name (the sample drops the
        suffix). A 10-Q matches the whole document. ``"any"`` disables the
        filter, ``None`` keeps whole-document files only, and e.g. ``"item1a"``
        keeps that section only. When a 10-K exists both ways, the ``_item7``
        file is kept.

    Returns
    -------
    DataFrame with :data:`META_COLUMNS` plus an absolute ``path`` column.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"input directory does not exist: {root}")

    if form_type is None:
        wanted_forms = None
    elif isinstance(form_type, str):
        wanted_forms = {form_type.upper()}
    else:
        wanted_forms = {f.upper() for f in form_type}

    rows: list[dict] = []
    skipped = 0
    for path in sorted(root.rglob("*.txt")):
        parsed = parse_report_filename(path.name)
        if parsed is None:
            skipped += 1
            continue
        if wanted_forms is not None and parsed["FORM_TYPE"] not in wanted_forms:
            continue
        if not _section_accepted(parsed["SECTION"], parsed["FORM_TYPE"], section):
            continue
        rows.append({**parsed, "path": str(path)})

    if section == SECTION_DEFAULT:
        rows = _prefer_explicit_item7(rows)

    if skipped:
        logger.warning("%d .txt file(s) under %s did not match the naming pattern", skipped, root)
    if not rows:
        logger.warning("no filings matched form_type=%s section=%s under %s", form_type, section, root)

    frame = _as_frame(rows)
    duplicated = frame.duplicated(list(DOC_KEY_COLUMNS), keep=False)
    if duplicated.any():
        logger.warning(
            "%d file(s) share a filing key (kept all; check for duplicate extractions)",
            int(duplicated.sum()),
        )
    logger.info("discovered %d filing(s) under %s", len(frame), root)
    return frame


def reports_from_index(
    index: pd.DataFrame,
    root: str | Path,
    form_type: str,
    section: str | None = SECTION_DEFAULT,
    require_existing: bool = True,
) -> pd.DataFrame:
    """Build the report table from an explicit index of filings.

    Use this when the sample is defined by a table (e.g. the Loughran-McDonald
    10-X index) rather than by whatever happens to sit in a directory. ``index``
    must have ``CIK``, ``FILING_DATE`` and ``ACC_NUM`` columns.
    """
    root = Path(root).expanduser().resolve()
    missing_cols = [c for c in ("CIK", "FILING_DATE", "ACC_NUM") if c not in index.columns]
    if missing_cols:
        raise KeyError(f"index is missing column(s): {missing_cols}")

    form_type = form_type.upper()
    wanted = _wanted_section(form_type, section)

    rows: list[dict] = []
    for record in index[["CIK", "FILING_DATE", "ACC_NUM"]].astype(str).to_dict("records"):
        name = report_filename(
            record["CIK"], record["FILING_DATE"], record["ACC_NUM"], form_type, wanted
        )
        candidates = [root / name]
        if wanted:
            # The extraction pipeline sometimes nests sections in subfolders.
            candidates.append(root / f"{wanted}_files" / name)
        if section == SECTION_DEFAULT and form_type == "10-K" and wanted:
            candidates.append(
                root
                / report_filename(
                    record["CIK"], record["FILING_DATE"], record["ACC_NUM"], form_type, None
                )
            )
        path = next((c for c in candidates if c.exists()), None)
        if path is None and require_existing:
            continue
        found_section = wanted
        if path is not None:
            parsed = parse_report_filename(path.name)
            if parsed is not None:
                found_section = parsed["SECTION"]
        rows.append(
            {
                **record,
                "FORM_TYPE": form_type,
                "SECTION": found_section,
                "path": str(path) if path is not None else "",
            }
        )

    frame = _as_frame(rows)
    n_missing = len(index) - len(frame)
    if n_missing:
        logger.warning("%d of %d indexed filing(s) have no text file under %s", n_missing, len(index), root)
    logger.info("resolved %d filing(s) from index under %s", len(frame), root)
    return frame


@dataclass
class ReportItem:
    """One filing handed to the scorer."""

    key: tuple
    meta: dict = field(repr=False)
    text: str | None = field(default=None, repr=False)
    n_chars: int = 0
    n_chars_scored: int = 0
    decode_fallback: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


class ReportDataset:
    """Map-style dataset over a report table produced by this module.

    Returns :class:`ReportItem` objects. Read failures are reported on the item
    rather than raised, so the extraction loop can record them explicitly
    instead of losing the filing.
    """

    def __init__(
        self,
        reports: pd.DataFrame,
        max_chars: int | None = None,
        min_chars: int = 200,
    ) -> None:
        missing = [c for c in (*META_COLUMNS, "path") if c not in reports.columns]
        if missing:
            raise KeyError(f"report table is missing column(s): {missing}")
        self.reports = reports.reset_index(drop=True)
        self.max_chars = max_chars
        self.min_chars = min_chars

    def __len__(self) -> int:
        return len(self.reports)

    def __iter__(self) -> Iterator[ReportItem]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> ReportItem:
        record = self.reports.iloc[idx].to_dict()
        meta = {col: record[col] for col in META_COLUMNS}
        key = tuple(str(record[col]) for col in DOC_KEY_COLUMNS)
        item = ReportItem(key=key, meta=meta)

        path = Path(record["path"]) if record["path"] else None
        if path is None or not path.exists():
            item.error = "missing_file"
            return item

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # EDGAR text occasionally carries stray bytes; keep the filing but
            # flag it rather than dropping it on the floor.
            text = path.read_text(encoding="utf-8", errors="replace")
            item.decode_fallback = True
        except OSError as exc:
            item.error = f"read_error: {exc}"
            return item

        item.n_chars = len(text)

        if item.n_chars < self.min_chars:
            item.error = f"too_short: {item.n_chars} chars < {self.min_chars}"
            return item

        item.text = text if self.max_chars is None else text[: self.max_chars]
        item.n_chars_scored = len(item.text)
        return item


def collate_items(batch: Sequence[ReportItem]) -> list[ReportItem]:
    """Pass-through collate: batching happens at the chunk level downstream."""
    return list(batch)
