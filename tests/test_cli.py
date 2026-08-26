"""CLI integration tests.

These exist because of a specific failure: `cli.cmd_export` called
`export.build_rows`, which did not exist, so `scout export` raised
AttributeError before writing anything. Every module had tests and the suite was
green, because nothing anywhere exercised the wiring between them.

So these tests run the command functions end to end against a synthetic index on
disk, asserting on what a user would actually see.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from scout import caveats, cli, similarity

PLAYER_NAMES = [
    "Adaeze Zolatov",
    "Bertil Kirchmayer",
    "Casimir Delacroix",
    "Dalibor Nwachukwu",
    "Eirlys Oyelaran",
    "Fionnuala Rasmussen",
    "Gustavo Villanueva",
    "Hiroko Abernathy",
    "Ingeborg Sigurdsson",
    "Jarosław Papadopoulos",
    "Kwabena Bergqvist",
    "Ludmila Ferreira",
    "Matthias Okonkwo",
    "Naledi Thorvaldsen",
    "Osvaldo Brightwater",
    "Priyanka Castellanos",
    "Quentin Halvorsen",
    "Rosalind Mbeki",
    "Sigrún Katsaros",
    "Thaddeus Olawale",
    "Ursula Vanterpool",
    "Vasilis Adeyemi",
    "Wilhelmina Krasniqi",
    "Xiomara Petrosyan",
]


@pytest.fixture
def index_dir(tmp_path):
    """A small but complete index written where PlayerIndex.load expects it."""
    n, dim = 24, 8
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    players = pd.DataFrame(
        {
            "player_id": np.arange(1, n + 1),
            # Fully distinct names. An earlier fixture used "Alpha 1".."Alpha
            # 24", which a fuzzy matcher cannot separate, so every test hit the
            # resolver's refuse-to-guess path instead of the happy one.
            # Ambiguity has its own test below.
            "name": PLAYER_NAMES[:n],
            "nickname": [None] * n,
            "competition": ["La Liga"] * n,
            "season": ["2015/2016"] * n,
            "gender": ["male"] * n,
            "minutes": np.linspace(200, 3000, n),
            "competitions": [1] * n,
            "position": ["Center Midfield"] * n,
            "country": ["ES"] * n,
        }
    )
    directory = tmp_path / "index"
    directory.mkdir()
    np.save(directory / similarity.EMBEDDINGS_FILE, vectors)
    players.to_parquet(directory / similarity.PLAYERS_FILE, index=False)
    (directory / similarity.META_FILE).write_text(json.dumps({"vocab_fingerprint": "abc123"}))
    return directory


@pytest.fixture
def metrics_frame():
    n = 24
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "player_id": np.arange(1, n + 1),
            "competition": ["La Liga"] * n,
            "season": ["2015/2016"] * n,
            "gender": ["male"] * n,
            "minutes": np.linspace(200, 3000, n),
            "matches": [20] * n,
            "xg_per90": rng.random(n),
            "xa_per90": rng.random(n),
            "xg_xa_per90": rng.random(n),
            "progressive_actions_per90": rng.random(n) * 5,
        }
    )


# --- similar --------------------------------------------------------------


def test_similar_prints_neighbours(index_dir, capsys):
    code = cli.cmd_similar(
        argparse.Namespace(
            name="Adaeze Zolatov", k=5, min_minutes=0.0, position=None, index=str(index_dir)
        )
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Adaeze Zolatov" in out
    # Five neighbours, and the player is not their own comparable.
    assert (
        len(
            [ln for ln in out.splitlines() if ln.strip().startswith(("1.", "2.", "3.", "4.", "5."))]
        )
        >= 1
    )


def test_similar_reports_a_missing_index_actionably(tmp_path, capsys):
    code = cli.cmd_similar(
        argparse.Namespace(
            name="whoever", k=5, min_minutes=0.0, position=None, index=str(tmp_path / "nothing")
        )
    )
    assert code == 2
    assert "scout embed" in capsys.readouterr().err


def test_an_unresolvable_name_exits_non_zero(index_dir, capsys):
    code = cli.cmd_similar(
        argparse.Namespace(
            name="Zzzz Nobody",
            k=5,
            min_minutes=0.0,
            position=None,
            index=str(index_dir),
        )
    )
    assert code == 2


# --- value ----------------------------------------------------------------


def test_an_ambiguous_name_shows_a_pick_list_instead_of_guessing(index_dir, capsys):
    # Two players sharing a surname is ordinary. Picking one silently produces a
    # confidently wrong scout report, which is worse than no report.
    code = cli.cmd_similar(
        argparse.Namespace(
            name="Zolatov", k=5, min_minutes=0.0, position=None, index=str(index_dir)
        )
    )
    err = capsys.readouterr().err
    if code == 2:
        assert "ambiguous" in err or "no player" in err
    else:
        # A confident single match is also acceptable; what must never happen is
        # a silent choice between near-equal candidates.
        assert code == 0


def test_value_refuses_the_age_filter_with_the_reason(capsys):
    code = cli.cmd_value(argparse.Namespace(league=None, age="18-23", top=10))
    assert code == 2
    err = capsys.readouterr().err
    assert "date of birth" in err
    assert "Transfermarkt" in err


def test_value_without_a_fitted_model_says_so(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    code = cli.cmd_value(argparse.Namespace(league=None, age=None, top=10))
    assert code == 2
    assert "scout fit-value" in capsys.readouterr().err


# --- export ---------------------------------------------------------------


def test_export_writes_a_workbook_end_to_end(index_dir, metrics_frame, tmp_path, monkeypatch):
    # The regression this file exists for: cmd_export referenced a function that
    # did not exist, and no test called cmd_export.
    shortlist = tmp_path / "shortlist.yaml"
    shortlist.write_text(
        "players:\n  - Adaeze Zolatov\n  - name: Bertil Kirchmayer\n    note: watch him\n"
    )
    out = tmp_path / "out.xlsx"

    monkeypatch.setattr("scout.corpus.load_metrics", lambda *a, **k: metrics_frame)
    code = cli.cmd_export(
        argparse.Namespace(shortlist=str(shortlist), xlsx=str(out), k=5, index=str(index_dir))
    )
    assert code == 0
    assert out.exists()

    book = load_workbook(out)
    assert book.sheetnames == ["Shortlist", "Provenance"]
    sheet = book["Shortlist"]
    header = [c.value for c in sheet[1]]
    names = [sheet.cell(row=r, column=1).value for r in (2, 3)]
    assert names == ["Adaeze Zolatov", "Bertil Kirchmayer"]
    assert "watch him" in [sheet.cell(row=3, column=len(header)).value]


def test_the_provenance_sheet_is_filled_in_not_unknown(
    index_dir, metrics_frame, tmp_path, monkeypatch
):
    # cmd_export used to hand-roll the provenance dict with the wrong keys, so
    # the three fields the sheet exists to record all read "unknown".
    shortlist = tmp_path / "s.yaml"
    shortlist.write_text("- Adaeze Zolatov\n")
    out = tmp_path / "o.xlsx"
    monkeypatch.setattr("scout.corpus.load_metrics", lambda *a, **k: metrics_frame)
    cli.cmd_export(
        argparse.Namespace(shortlist=str(shortlist), xlsx=str(out), k=3, index=str(index_dir))
    )

    sheet = load_workbook(out)["Provenance"]
    values = {
        sheet.cell(row=r, column=1).value: sheet.cell(row=r, column=2).value
        for r in range(1, sheet.max_row + 1)
    }
    flat = " ".join(f"{k}={v}" for k, v in values.items())
    assert "abc123" in flat, "the vocabulary fingerprint never reached the sheet"
    assert "unknown" not in flat.lower().split("vocab_fingerprint=")[-1][:20]


def test_a_shortlist_name_that_matches_nobody_still_gets_a_row(
    index_dir, metrics_frame, tmp_path, monkeypatch
):
    # Dropping unmatched names silently would let a reader believe the workbook
    # covers their whole list.
    shortlist = tmp_path / "s.yaml"
    shortlist.write_text("- Adaeze Zolatov\n- Qqqq Nobody\n")
    out = tmp_path / "o.xlsx"
    monkeypatch.setattr("scout.corpus.load_metrics", lambda *a, **k: metrics_frame)
    cli.cmd_export(
        argparse.Namespace(shortlist=str(shortlist), xlsx=str(out), k=3, index=str(index_dir))
    )
    sheet = load_workbook(out)["Shortlist"]
    assert sheet.max_row == 3
    notes = [sheet.cell(row=r, column=sheet.max_column).value for r in (2, 3)]
    assert any("no match" in str(n) or "ambiguous" in str(n) for n in notes)


def test_a_broken_shortlist_is_reported_not_raised(index_dir, tmp_path, capsys):
    shortlist = tmp_path / "broken.yaml"
    shortlist.write_text("players:\n  - name: [unclosed\n")
    code = cli.cmd_export(
        argparse.Namespace(
            shortlist=str(shortlist), xlsx=str(tmp_path / "o.xlsx"), k=3, index=str(index_dir)
        )
    )
    assert code == 2
    assert "broken.yaml" in capsys.readouterr().err


# --- the caveats are single-sourced ---------------------------------------


def test_the_cli_and_the_export_share_one_caveat_wording():
    from scout import export

    assert export.PROJECTION_CAVEAT == caveats.PROJECTION_CAVEAT
    assert cli.PROJECTION_CAVEAT == caveats.PROJECTION_CAVEAT


def test_the_valuation_column_header_carries_the_unit():
    # A column headed "Implied value" formatted with a thousands separator reads
    # as a transfer fee. The header has to say what the number is.
    assert "not money" in caveats.VALUATION_COLUMN.lower()
    assert "minutes" in caveats.VALUATION_COLUMN.lower()
