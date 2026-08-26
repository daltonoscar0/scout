"""Excel export tests, including a real openpyxl round-trip.

The acceptance criterion is that the workbook opens clean in Excel, which no
unit test can literally check. The nearest honest substitute is writing a file,
reading it back with a fresh openpyxl load, and asserting on what a reader
would actually see: sheet names, headers, cell values, number formats and the
provenance sheet.
"""

from __future__ import annotations

import datetime as dt

import pytest
from openpyxl import load_workbook

from scout import caveats, export


def make_row(name="Player One", **overrides):
    defaults = dict(
        player_id=7,
        competition="UEFA Euro",
        season="2024",
        position="Center Midfield",
        minutes=1234.6,
        metrics={"xg_per90": 0.4321, "xa_per90": 0.1987},
        comparables=[
            export.Comparable("Player Two", 0.9412),
            export.Comparable("Player Three", 0.9086),
        ],
        projection=0.55,
        valuation=2100.4,
        note="left-footed",
    )
    defaults.update(overrides)
    return export.ShortlistRow(name=name, **defaults)


def write(tmp_path, rows, provenance=None):
    path = export.write_workbook(
        tmp_path / "out.xlsx",
        rows,
        provenance=provenance or export.build_provenance({"vocab_fingerprint": "abc123"}),
    )
    return path, load_workbook(path)


# --- the shortlist file ---------------------------------------------------


def test_a_bare_list_of_names_is_accepted(tmp_path):
    path = tmp_path / "shortlist.yaml"
    path.write_text("- Player One\n- Player Two\n")
    assert [p["name"] for p in export.load_shortlist(path)] == ["Player One", "Player Two"]


def test_a_players_mapping_with_notes_is_accepted(tmp_path):
    path = tmp_path / "shortlist.yaml"
    path.write_text(
        "title: January window\n"
        "players:\n"
        "  - Player One\n"
        "  - name: Player Two\n"
        "    note: needs a left foot\n"
    )
    players = export.load_shortlist(path)
    assert [p["name"] for p in players] == ["Player One", "Player Two"]
    assert players[0]["note"] == ""
    assert players[1]["note"] == "needs a left foot"


def test_unknown_keys_on_an_entry_survive(tmp_path):
    path = tmp_path / "shortlist.yaml"
    path.write_text("players:\n  - name: Player One\n    position: Left Back\n")
    assert export.load_shortlist(path)[0]["position"] == "Left Back"


def test_broken_yaml_names_the_file(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("players:\n  - name: [unclosed\n")
    with pytest.raises(export.ShortlistError) as excinfo:
        export.load_shortlist(path)
    assert "broken.yaml" in str(excinfo.value)


def test_a_mapping_without_players_is_rejected(tmp_path):
    path = tmp_path / "nope.yaml"
    path.write_text("title: January window\n")
    with pytest.raises(export.ShortlistError, match="players"):
        export.load_shortlist(path)


def test_an_entry_with_no_name_is_rejected(tmp_path):
    path = tmp_path / "nameless.yaml"
    path.write_text("players:\n  - note: who is this\n")
    with pytest.raises(export.ShortlistError) as excinfo:
        export.load_shortlist(path)
    assert "nameless.yaml" in str(excinfo.value)
    assert "player 1" in str(excinfo.value)


def test_an_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(export.ShortlistError, match="empty"):
        export.load_shortlist(path)


def test_a_scalar_entry_that_is_not_a_name_is_rejected(tmp_path):
    path = tmp_path / "numbers.yaml"
    path.write_text("- 17\n")
    with pytest.raises(export.ShortlistError, match="int"):
        export.load_shortlist(path)


def test_a_missing_file_is_reported_as_a_shortlist_error(tmp_path):
    with pytest.raises(export.ShortlistError, match="cannot read"):
        export.load_shortlist(tmp_path / "absent.yaml")


# --- the round-trip -------------------------------------------------------


def test_both_sheets_are_present(tmp_path):
    _, book = write(tmp_path, [make_row()])
    assert book.sheetnames == [export.SHORTLIST_SHEET, export.PROVENANCE_SHEET]


def test_the_header_row_reads_as_expected(tmp_path):
    _, book = write(tmp_path, [make_row()])
    header = [cell.value for cell in book[export.SHORTLIST_SHEET][1]]
    assert header[:6] == [
        "Player",
        "Player ID",
        "Competition",
        "Season",
        "Position",
        "Minutes",
    ]
    # Metric headers stay as the raw corpus column names so a number is traceable.
    assert "xg_per90" in header
    assert header[-5:] == [
        "Projection",
        "Projection caveat",
        caveats.VALUATION_COLUMN,
        "Value source",
        "Note",
    ]


def test_the_values_round_trip(tmp_path):
    _, book = write(tmp_path, [make_row()])
    sheet = book[export.SHORTLIST_SHEET]
    header = [cell.value for cell in sheet[1]]
    row = dict(zip(header, [cell.value for cell in sheet[2]], strict=True))

    assert row["Player"] == "Player One"
    assert row["Player ID"] == 7
    assert row["Competition"] == "UEFA Euro"
    assert row["Minutes"] == pytest.approx(1234.6)
    assert row["xg_per90"] == pytest.approx(0.4321)
    assert row["Comparable 1"] == "Player Two"
    assert row["Similarity 1"] == pytest.approx(0.9412)
    assert row["Comparable 2"] == "Player Three"
    assert row["Projection"] == pytest.approx(0.55)
    assert row[caveats.VALUATION_COLUMN] == pytest.approx(2100.4)
    assert row["Value source"] == export.VALUATION_TARGET
    assert row["Note"] == "left-footed"


def test_the_number_formats_are_what_a_reader_needs(tmp_path):
    _, book = write(tmp_path, [make_row()])
    sheet = book[export.SHORTLIST_SHEET]
    header = [cell.value for cell in sheet[1]]
    cells = dict(zip(header, sheet[2], strict=True))

    assert cells["Minutes"].number_format == export.COUNT_FORMAT
    assert cells[caveats.VALUATION_COLUMN].number_format == export.COUNT_FORMAT
    assert cells["xg_per90"].number_format == export.RATE_FORMAT
    assert cells["Projection"].number_format == export.RATE_FORMAT
    assert cells["Similarity 1"].number_format == export.SCORE_FORMAT


def test_one_row_per_player(tmp_path):
    rows = [make_row("A"), make_row("B"), make_row("C")]
    _, book = write(tmp_path, rows)
    sheet = book[export.SHORTLIST_SHEET]
    assert sheet.max_row == 4
    assert [sheet.cell(row=i, column=1).value for i in (2, 3, 4)] == ["A", "B", "C"]


# --- styling --------------------------------------------------------------


def test_the_header_is_bold_filled_and_frozen(tmp_path):
    _, book = write(tmp_path, [make_row()])
    sheet = book[export.SHORTLIST_SHEET]
    assert sheet.freeze_panes == "A2"
    assert sheet["A1"].font.bold
    assert sheet["A1"].fill.fgColor.rgb.endswith(export.HEADER_FILL)


def test_alternate_rows_are_banded(tmp_path):
    _, book = write(tmp_path, [make_row("A"), make_row("B")])
    sheet = book[export.SHORTLIST_SHEET]
    assert sheet["A2"].fill.fill_type is None
    assert sheet["A3"].fill.fgColor.rgb.endswith(export.BAND_FILL)


def test_columns_are_widened_but_capped(tmp_path):
    _, book = write(tmp_path, [make_row()])
    widths = [dim.width for dim in book[export.SHORTLIST_SHEET].column_dimensions.values()]
    assert widths
    assert all(export.MIN_COLUMN_WIDTH <= w <= export.MAX_COLUMN_WIDTH for w in widths)


# --- the caveat -----------------------------------------------------------


def test_the_projection_carries_its_caveat_in_an_adjacent_column(tmp_path):
    _, book = write(tmp_path, [make_row()])
    sheet = book[export.SHORTLIST_SHEET]
    header = [cell.value for cell in sheet[1]]
    caveat = sheet.cell(row=2, column=header.index("Projection caveat") + 1).value
    assert caveat == export.PROJECTION_CAVEAT
    assert "identity projection" in caveat.lower()


def test_the_projection_cell_carries_a_comment(tmp_path):
    _, book = write(tmp_path, [make_row()])
    sheet = book[export.SHORTLIST_SHEET]
    header = [cell.value for cell in sheet[1]]
    comment = sheet.cell(row=2, column=header.index("Projection") + 1).comment
    assert comment is not None
    assert "identity projection" in comment.text.lower()


def test_an_empty_projection_gets_no_caveat_and_no_comment(tmp_path):
    _, book = write(tmp_path, [make_row(projection=None)])
    sheet = book[export.SHORTLIST_SHEET]
    header = [cell.value for cell in sheet[1]]
    assert sheet.cell(row=2, column=header.index("Projection caveat") + 1).value in (None, "")
    assert sheet.cell(row=2, column=header.index("Projection") + 1).comment is None


# --- provenance -----------------------------------------------------------


def test_the_provenance_sheet_records_what_a_later_reader_needs(tmp_path):
    provenance = export.build_provenance(
        {"vocab_fingerprint": "abc123", "checkpoint": "run-4.pt", "dim": 256, "players": 6300},
        generated=dt.date(2026, 8, 25),
    )
    _, book = write(tmp_path, [make_row()], provenance)
    sheet = book[export.PROVENANCE_SHEET]
    record = {
        sheet.cell(row=i, column=1).value: sheet.cell(row=i, column=2).value
        for i in range(2, sheet.max_row + 1)
    }

    assert record["vocabulary fingerprint"] == "abc123"
    assert record["index: checkpoint"] == "run-4.pt"
    assert record["index: dim"] == "256"
    assert record["generated"] == "2026-08-25"
    assert record["scout version"]
    assert record["sportio version"]
    assert record["projection caveat"] == export.PROJECTION_CAVEAT
    assert record["valuation target"] == export.VALUATION_TARGET
    assert "Transfermarkt" in record["valuation licence"]


def test_a_partial_provenance_dict_still_answers_every_question(tmp_path):
    _, book = write(tmp_path, [make_row()], {"vocab_fingerprint": "deadbeef"})
    sheet = book[export.PROVENANCE_SHEET]
    record = {
        sheet.cell(row=i, column=1).value: sheet.cell(row=i, column=2).value
        for i in range(2, sheet.max_row + 1)
    }
    assert record["vocabulary fingerprint"] == "deadbeef"
    assert record["scout version"] == "unknown"
    assert record["projection caveat"] == export.PROJECTION_CAVEAT


def test_extra_provenance_keys_are_carried_through(tmp_path):
    provenance = export.build_provenance({}, extra={"shortlist_file": "jan.yaml"})
    _, book = write(tmp_path, [make_row()], provenance)
    sheet = book[export.PROVENANCE_SHEET]
    labels = {sheet.cell(row=i, column=1).value for i in range(2, sheet.max_row + 1)}
    assert "shortlist file" in labels


# --- degenerate rows ------------------------------------------------------


def test_a_player_with_no_comparables_and_no_projection_still_writes(tmp_path):
    row = export.ShortlistRow(name="Unknown Player")
    path, book = write(tmp_path, [row])
    sheet = book[export.SHORTLIST_SHEET]
    header = [cell.value for cell in sheet[1]]

    assert path.exists()
    assert sheet["A2"].value == "Unknown Player"
    # The comparable columns still exist, so the sheet shape does not change.
    assert "Comparable 1" in header
    assert sheet.cell(row=2, column=header.index("Comparable 1") + 1).value in (None, "")
    assert sheet.cell(row=2, column=header.index("Projection") + 1).value is None
    assert sheet.cell(row=2, column=header.index("Value source") + 1).value in (None, "")


def test_nan_metrics_are_written_as_blanks_not_as_nan(tmp_path):
    # openpyxl will happily write float('nan'); Excel then calls the file corrupt.
    row = make_row(metrics={"xg_per90": float("nan")}, minutes=float("nan"))
    _, book = write(tmp_path, [row])
    sheet = book[export.SHORTLIST_SHEET]
    header = [cell.value for cell in sheet[1]]
    assert sheet.cell(row=2, column=header.index("xg_per90") + 1).value is None
    assert sheet.cell(row=2, column=header.index("Minutes") + 1).value is None


def test_rows_with_different_metric_sets_share_one_column_layout(tmp_path):
    rows = [
        make_row("A", metrics={"xg_per90": 0.3}),
        make_row("B", metrics={"xa_per90": 0.2, "touches_per90": 55.0}),
    ]
    _, book = write(tmp_path, rows)
    sheet = book[export.SHORTLIST_SHEET]
    header = [cell.value for cell in sheet[1]]
    for name in ("xg_per90", "xa_per90", "touches_per90"):
        assert name in header
    assert sheet.cell(row=2, column=header.index("xa_per90") + 1).value is None
    assert sheet.cell(row=3, column=header.index("xg_per90") + 1).value is None


def test_an_empty_shortlist_produces_a_readable_workbook(tmp_path):
    _, book = write(tmp_path, [])
    sheet = book[export.SHORTLIST_SHEET]
    assert sheet["A1"].value == "Player"
    assert sheet.max_row == 1
    assert export.PROVENANCE_SHEET in book.sheetnames
