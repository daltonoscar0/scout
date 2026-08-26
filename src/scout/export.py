"""The Excel deliverable: a shortlist as a workbook someone can email.

This is the end of the query path, so it stays on the same diet as
`scout.similarity` -- pandas, yaml and openpyxl, no torch, no sklearn. The
caller assembles `ShortlistRow`s from whichever modules it needs; this one only
knows how to lay them out.

Two things here are not cosmetic:

**The projection caveat travels with the number.** `scout.population` measured
that the open corpus cannot support a learned cross-competition projection, and
`scout.projection` reports that no predictor beats identity outside its
confidence interval. A projected per-90 in a spreadsheet with no warning
attached is the single most misreadable artifact scout can emit -- it survives
being copied into a slide deck long after the report that qualified it is gone.
So the caveat is written three times: an adjacent column, a cell comment on the
projection itself, and the provenance sheet.

**The provenance sheet is mandatory.** A workbook found on a shared drive in
six months is worthless if the reader cannot tell which vocabulary, which
checkpoint and which valuation target produced it. Recording that costs one
sheet and answers the question permanently.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SHORTLIST_SHEET = "Shortlist"
PROVENANCE_SHEET = "Provenance"

# The caveat that follows every projected number out of this repo. Kept as one
# string so the column, the cell comment and the provenance sheet cannot drift.
PROJECTION_CAVEAT = (
    "Identity projection: the player's source-competition output, unchanged. "
    "No fitted predictor beat this null outside its bootstrap confidence "
    "interval, so scout does not claim a learned cross-competition translation. "
    "Treat this as 'what they did', not 'what they will do'."
)

VALUATION_TARGET = "next_season_minutes_at_tier"
VALUATION_LICENCE = (
    "Derived from StatsBomb open lineups only. Not a transfer fee. Transfermarkt "
    "market value is not used: its terms prohibit automated access and prohibit "
    "training models on the content, and all of its competitions are men's "
    "football while much of this corpus is not."
)

# Number formats. Rates get two decimals as the spec asks; cosine similarity
# gets three, because neighbour scores for a real player cluster inside a band
# narrower than 0.01 and two decimals would print a column of identical numbers.
RATE_FORMAT = "0.00"
SCORE_FORMAT = "0.000"
COUNT_FORMAT = "#,##0"

HEADER_FILL = "1F3B4D"
HEADER_FONT_COLOUR = "FFFFFF"
BAND_FILL = "F2F5F8"

MIN_COLUMN_WIDTH = 10
MAX_COLUMN_WIDTH = 44


class ShortlistError(ValueError):
    """The shortlist file could not be read as a shortlist."""


@dataclass
class Comparable:
    name: str
    score: float


@dataclass
class ShortlistRow:
    """One player, with everything the workbook says about them.

    Every analytical field is optional. A shortlist is typed by a human before
    the index is consulted, so a name that resolved to nobody, a player with no
    neighbours above the minutes floor, and a player outside the valuation
    model's coverage all have to reach the sheet as blanks rather than as a
    crash halfway through writing it.
    """

    name: str
    player_id: int | None = None
    competition: str = ""
    season: str = ""
    position: str | None = None
    minutes: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    comparables: list[Comparable] = field(default_factory=list)
    projection: float | None = None
    projection_caveat: str = PROJECTION_CAVEAT
    valuation: float | None = None
    valuation_source: str = VALUATION_TARGET
    note: str = ""


# --- the shortlist file ---------------------------------------------------


def load_shortlist(path: str | Path) -> list[dict]:
    """Parse a YAML shortlist into `[{"name": ..., "note": ...}, ...]`.

    Both shapes a person actually writes are accepted: a bare list of names, or
    a mapping with a `players:` key whose entries are names or
    `{name:, note:}` mappings. Rejecting the terse form would mean the fastest
    way to use scout is the one that errors.

    Extra keys on an entry are passed through untouched, so a caller can carry
    its own hints (a preferred competition, a target position) without this
    function needing to know about them.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise ShortlistError(f"cannot read shortlist {path}: {exc}") from exc

    import yaml

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ShortlistError(f"{path} is not valid YAML: {exc}") from exc

    if document is None:
        raise ShortlistError(f"{path} is empty; expected a list of players")

    if isinstance(document, dict):
        if "players" not in document:
            raise ShortlistError(
                f"{path} is a mapping with no `players:` key. Give it a "
                "`players:` list, or make the whole file a list of names."
            )
        entries = document["players"]
    else:
        entries = document

    if not isinstance(entries, list):
        raise ShortlistError(f"{path}: expected a list of players, got {type(entries).__name__}")

    players: list[dict] = []
    for position, entry in enumerate(entries, start=1):
        players.append(_shortlist_entry(entry, path, position))
    if not players:
        raise ShortlistError(f"{path} lists no players")
    return players


def _shortlist_entry(entry: Any, path: Path, position: int) -> dict:
    if isinstance(entry, str):
        name = entry.strip()
        if not name:
            raise ShortlistError(f"{path}: player {position} is an empty name")
        return {"name": name, "note": ""}

    if isinstance(entry, dict):
        raw = entry.get("name")
        if not isinstance(raw, str) or not raw.strip():
            raise ShortlistError(f"{path}: player {position} has no usable `name:` (got {entry!r})")
        record = {str(k): v for k, v in entry.items()}
        record["name"] = raw.strip()
        record.setdefault("note", "")
        return record

    raise ShortlistError(
        f"{path}: player {position} should be a name or a mapping with `name:`, "
        f"got {type(entry).__name__}"
    )


# --- provenance -----------------------------------------------------------


def build_provenance(
    index_meta: dict | None = None,
    *,
    generated: dt.date | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the provenance record for a workbook.

    The sportio version is looked up defensively: sportio is an editable path
    dependency developed in another working tree, and an export must not fail
    because it drifted. An honest "unavailable" in the provenance sheet beats
    losing the workbook.
    """
    from scout import __version__

    meta = dict(index_meta or {})
    try:
        from scout import sportio_adapter

        sportio_version = sportio_adapter.sportio_version()
    except Exception as exc:  # noqa: BLE001 - any failure here is just a blank field
        sportio_version = f"unavailable ({type(exc).__name__})"

    provenance = {
        "scout_version": __version__,
        "sportio_version": sportio_version,
        "vocab_fingerprint": meta.get("vocab_fingerprint", "unknown"),
        "generated": (generated or dt.date.today()).isoformat(),
        "index": meta,
        "projection_caveat": PROJECTION_CAVEAT,
        "valuation_target": VALUATION_TARGET,
        "valuation_licence": VALUATION_LICENCE,
    }
    provenance.update(extra or {})
    return provenance


def _provenance_rows(provenance: dict) -> list[tuple[str, str]]:
    """Flatten provenance to label/value pairs, mandatory fields first.

    Defaults are filled in for anything the caller left out, so the sheet always
    answers the same questions even when it is handed a partial dict.
    """
    record = {
        "scout_version": "unknown",
        "sportio_version": "unknown",
        "vocab_fingerprint": "unknown",
        "generated": dt.date.today().isoformat(),
        "projection_caveat": PROJECTION_CAVEAT,
        "valuation_target": VALUATION_TARGET,
        "valuation_licence": VALUATION_LICENCE,
        **provenance,
    }

    rows = [
        ("scout version", record["scout_version"]),
        ("sportio version", record["sportio_version"]),
        ("vocabulary fingerprint", record["vocab_fingerprint"]),
        ("generated", record["generated"]),
    ]
    for key, value in dict(record.get("index") or {}).items():
        rows.append((f"index: {key}", value))
    rows += [
        ("projection caveat", record["projection_caveat"]),
        ("valuation target", record["valuation_target"]),
        ("valuation licence", record["valuation_licence"]),
    ]
    known = {
        "scout_version",
        "sportio_version",
        "vocab_fingerprint",
        "generated",
        "index",
        "projection_caveat",
        "valuation_target",
        "valuation_licence",
    }
    for key, value in provenance.items():
        if key not in known:
            rows.append((key.replace("_", " "), value))
    return [(label, _as_text(value)) for label, value in rows]


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


# --- the sheet layout -----------------------------------------------------


@dataclass
class _Column:
    header: str
    number_format: str | None = None


IDENTITY_COLUMNS = (
    _Column("Player"),
    _Column("Player ID", COUNT_FORMAT),
    _Column("Competition"),
    _Column("Season"),
    _Column("Position"),
    _Column("Minutes", COUNT_FORMAT),
)


def _metric_columns(rows: list[ShortlistRow]) -> list[str]:
    """Union of the metric keys present, canonical order first.

    Headers are the raw metric names rather than prettified labels, so a number
    in the sheet can be traced back to the column it came from in
    `corpus.load_metrics()` without a translation table.
    """
    from scout import metrics as metrics_module

    canonical = [f"{name}_per90" for name in metrics_module.PER_90_COLUMNS]
    seen = {key for row in rows for key in row.metrics}
    ordered = [name for name in canonical if name in seen]
    ordered += sorted(seen - set(ordered))
    return ordered


def _columns(rows: list[ShortlistRow]) -> tuple[list[_Column], list[str], int]:
    metric_names = _metric_columns(rows)
    # Always at least one comparable pair: a stable column layout means a
    # shortlist where nobody matched still reads as "no comparables" rather than
    # as a differently shaped sheet.
    n_comparables = max(1, max((len(row.comparables) for row in rows), default=1))

    columns = list(IDENTITY_COLUMNS)
    columns += [_Column(name, RATE_FORMAT) for name in metric_names]
    for i in range(1, n_comparables + 1):
        columns.append(_Column(f"Comparable {i}"))
        columns.append(_Column(f"Similarity {i}", SCORE_FORMAT))
    columns += [
        _Column("Projection", RATE_FORMAT),
        _Column("Projection caveat"),
        _Column("Implied value", COUNT_FORMAT),
        _Column("Value source"),
        _Column("Note"),
    ]
    return columns, metric_names, n_comparables


def _row_values(row: ShortlistRow, metric_names: list[str], n_comparables: int) -> list[Any]:
    values: list[Any] = [
        row.name,
        row.player_id,
        row.competition,
        row.season,
        row.position or "",
        _number(row.minutes),
    ]
    values += [_number(row.metrics.get(name)) for name in metric_names]
    for i in range(n_comparables):
        if i < len(row.comparables):
            values.append(row.comparables[i].name)
            values.append(_number(row.comparables[i].score))
        else:
            values += ["", None]
    projection = _number(row.projection)
    values += [
        projection,
        # No projection means no caveat to attach; an unconditional caveat next
        # to an empty cell reads as a warning about nothing.
        row.projection_caveat if projection is not None else "",
        _number(row.valuation),
        row.valuation_source if _number(row.valuation) is not None else "",
        row.note,
    ]
    return values


def _number(value: Any) -> float | None:
    """Coerce to a float Excel will accept, or None.

    NaN is the important case: it arrives routinely from pandas and openpyxl
    will happily write it, producing a file Excel reports as corrupt on open.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


# --- writing --------------------------------------------------------------


def write_workbook(
    path: str | Path,
    rows: list[ShortlistRow],
    *,
    provenance: dict,
) -> Path:
    """Write the shortlist and its provenance to `path`, returning the path."""
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns, metric_names, n_comparables = _columns(rows)
    table = [_row_values(row, metric_names, n_comparables) for row in rows]

    header_font = Font(bold=True, color=HEADER_FONT_COLOUR)
    header_fill = PatternFill("solid", fgColor=HEADER_FILL)
    band_fill = PatternFill("solid", fgColor=BAND_FILL)
    header_border = Border(bottom=Side(style="thin", color=HEADER_FILL))

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = SHORTLIST_SHEET

    for index, column in enumerate(columns, start=1):
        cell = sheet.cell(row=1, column=index, value=column.header)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = header_border
        cell.alignment = Alignment(vertical="center", horizontal="left")

    projection_index = next(i for i, c in enumerate(columns, start=1) if c.header == "Projection")

    for offset, (record, values) in enumerate(zip(rows, table, strict=True)):
        excel_row = offset + 2
        for index, (column, value) in enumerate(zip(columns, values, strict=True), start=1):
            cell = sheet.cell(row=excel_row, column=index, value=value)
            if column.number_format and isinstance(value, (int, float)):
                cell.number_format = column.number_format
            # Band every other row. Subtle enough to guide the eye across a wide
            # sheet without turning the export into a colour scheme.
            if offset % 2 == 1:
                cell.fill = band_fill
        if values[projection_index - 1] is not None:
            comment = Comment(record.projection_caveat, "scout")
            comment.width, comment.height = 340, 150
            sheet.cell(row=excel_row, column=projection_index).comment = comment

    sheet.freeze_panes = "A2"
    if table:
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{len(table) + 1}"
    _fit_columns(sheet, [c.header for c in columns], table, get_column_letter)

    _write_provenance(workbook, provenance, get_column_letter)
    workbook.save(path)
    return path


def _write_provenance(workbook: Any, provenance: dict, get_column_letter: Any) -> None:
    from openpyxl.styles import Font, PatternFill

    sheet = workbook.create_sheet(PROVENANCE_SHEET)
    for index, header in enumerate(("Field", "Value"), start=1):
        cell = sheet.cell(row=1, column=index, value=header)
        cell.font = Font(bold=True, color=HEADER_FONT_COLOUR)
        cell.fill = PatternFill("solid", fgColor=HEADER_FILL)

    rows = _provenance_rows(provenance)
    for offset, (label, value) in enumerate(rows, start=2):
        sheet.cell(row=offset, column=1, value=label).font = Font(bold=True)
        sheet.cell(row=offset, column=2, value=value)

    sheet.freeze_panes = "A2"
    _fit_columns(sheet, ["Field", "Value"], [list(r) for r in rows], get_column_letter)


def _fit_columns(
    sheet: Any,
    headers: list[str],
    table: list[list[Any]],
    get_column_letter: Any,
) -> None:
    """Width from the longest rendered value, clamped at both ends.

    Clamped because the caveat column is 300 characters: an honest fit would
    push every later column off the screen, and the text is readable in the
    formula bar and the cell comment anyway.
    """
    for index, header in enumerate(headers, start=1):
        longest = len(str(header))
        for values in table:
            value = values[index - 1] if index - 1 < len(values) else None
            if value is None:
                continue
            rendered = f"{value:,.2f}" if isinstance(value, float) else str(value)
            longest = max(longest, len(rendered))
        width = min(MAX_COLUMN_WIDTH, max(MIN_COLUMN_WIDTH, longest + 2))
        sheet.column_dimensions[get_column_letter(index)].width = width
