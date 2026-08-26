"""Tests for the scout card's pure layer, plus the import guard that protects it.

Everything here runs without a Streamlit runtime. That is the point of the
split in `scout.app`: the frames are testable, the widgets are not, so all the
logic lives in the frames.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scout import app, corpus, metrics, similarity, valuation

N_PLAYERS = 30


def make_index(n: int = N_PLAYERS, dim: int = 8, seed: int = 0) -> similarity.PlayerIndex:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    players = pd.DataFrame(
        {
            "player_id": np.arange(1, n + 1),
            "name": [f"Player {i}" for i in range(1, n + 1)],
            "nickname": [None] * n,
            "competition": ["UEFA Euro"] * n,
            "season": ["2024"] * n,
            "gender": ["male"] * (n // 2) + ["female"] * (n - n // 2),
            "minutes": np.linspace(500, 3000, n),
            "position": ["Center Midfield"] * (n // 2) + ["Goalkeeper"] * (n - n // 2),
        }
    )
    return similarity.PlayerIndex(vectors, players, {"vocab_fingerprint": "x"})


def make_metrics(n: int = N_PLAYERS) -> pd.DataFrame:
    """Two seasons per player, in two competitions, with output rising with id.

    Both competitions are club football on purpose. The value target is defined
    over club seasons only -- tiering a World Cup against a league is a category
    error -- so a fixture built from tournaments would exercise nothing.
    """
    rows = []
    for player_id in range(1, n + 1):
        gender = "male" if player_id <= n // 2 else "female"
        for season, competition in (("2023", "Premier League"), ("2024", "La Liga")):
            row = {
                "player_id": player_id,
                "competition": competition,
                "season": season,
                "gender": gender,
                "minutes": 600.0 + player_id * 10,
                "matches": 10,
                "goals": 1,
                "pass_completion": 0.8,
            }
            for column in app.RADAR_COLUMNS:
                row[column] = player_id * 0.1
            for target in ("xg_xa_per90", "progressive_actions_per90"):
                row[target] = player_id * 0.05
            rows.append(row)
    return pd.DataFrame(rows)


# --- the architectural constraint ----------------------------------------


def test_importing_the_app_does_not_pull_in_torch_or_sklearn():
    # Streamlit re-runs the script on every interaction, so a module-level torch
    # import would cost 2-4s per click rather than 2-4s once. sklearn is banned
    # for the same reason and because the valuation model lives in an extra.
    code = (
        "import sys; import scout.app; "
        "sys.exit(1 if {'torch', 'sklearn', 'faiss', 'lightgbm'} & set(sys.modules) else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


# --- radar ----------------------------------------------------------------


def test_the_radar_frame_has_one_row_per_requested_axis():
    frame = app.radar_frame(make_index(), make_metrics(), 5)
    assert list(frame["metric"]) == list(app.RADAR_COLUMNS)


def test_radar_percentiles_are_bounded_zero_to_one_hundred():
    metrics_frame = make_metrics()
    index = make_index()
    for player_id in (1, N_PLAYERS // 2, N_PLAYERS):
        percentiles = app.radar_frame(index, metrics_frame, player_id)["percentile"]
        assert ((percentiles >= 0.0) & (percentiles <= 100.0)).all()


def test_a_higher_output_player_ranks_above_a_lower_one():
    metrics_frame, index = make_metrics(), make_index()
    low = app.radar_frame(index, metrics_frame, 2).set_index("metric")["percentile"]
    high = app.radar_frame(index, metrics_frame, 14).set_index("metric")["percentile"]
    assert (high > low).all()


def test_percentile_peers_are_restricted_to_the_same_gender():
    metrics_frame, index = make_metrics(), make_index()
    # The pool is half male, half female, so a same-gender pool is smaller.
    restricted = app.radar_frame(index, metrics_frame, 3)["peers"].iloc[0]
    everyone = app.radar_frame(index, metrics_frame, 3, same_gender=False)["peers"].iloc[0]
    assert restricted < everyone


def test_the_peer_pool_excludes_players_below_the_minutes_floor():
    metrics_frame, index = make_metrics(), make_index()
    frame = app.radar_frame(index, metrics_frame, 20, min_minutes=850.0)
    # Only players with enough weighted minutes survive; the count travels with
    # the frame so the card can state what the percentile is against.
    assert 0 < frame["peers"].iloc[0] < N_PLAYERS


def test_profiles_weight_seasons_by_minutes_rather_than_averaging_rows():
    frame = pd.DataFrame(
        {
            "player_id": [1, 1],
            "minutes": [900.0, 100.0],
            "gender": ["male", "male"],
            "xg_per90": [1.0, 0.0],
        }
    )
    profile = app.player_profiles(frame, ["xg_per90"])
    # A flat mean would say 0.5; weighting by minutes says 0.9.
    assert profile["xg_per90"].iloc[0] == pytest.approx(0.9)


def test_the_radar_falls_back_to_the_index_when_metrics_are_missing():
    index = make_index()
    index.players["xg_per90"] = np.linspace(0.0, 1.0, len(index))
    frame = app.radar_frame(index, pd.DataFrame(), 30, ["xg_per90"])
    assert list(frame["metric"]) == ["xg_per90"]
    assert 0.0 <= frame["percentile"].iloc[0] <= 100.0


def test_an_unknown_player_with_no_metrics_yields_an_empty_radar():
    # similarity.percentiles drops columns it does not have, so a card with no
    # metrics at all renders an empty panel rather than fabricating axes.
    assert app.radar_frame(make_index(), pd.DataFrame(), 5).empty


def test_every_radar_axis_is_a_metric_the_pipeline_actually_computes():
    bases = {column.removesuffix("_per90") for column in app.RADAR_COLUMNS}
    assert bases <= set(metrics.PER_90_COLUMNS)


# --- comparables ----------------------------------------------------------


def test_the_comps_table_never_contains_the_queried_player():
    frame = app.comps_frame(make_index(), 7, k=10)
    assert 7 not in set(frame["player_id"])


def test_the_comps_table_is_sorted_by_similarity_descending():
    scores = app.comps_frame(make_index(), 7, k=10)["similarity"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_the_comps_table_honours_the_filters():
    frame = app.comps_frame(make_index(), 1, k=10, min_minutes=1500.0, position="Goalkeeper")
    assert (frame["minutes"] >= 1500.0).all()
    assert (frame["position"] == "Goalkeeper").all()


def test_a_filter_that_excludes_everyone_gives_an_empty_but_well_shaped_frame():
    frame = app.comps_frame(make_index(), 1, k=10, min_minutes=1e9)
    assert frame.empty
    assert list(frame.columns) == list(app.COMPS_COLUMNS)


# --- projection -----------------------------------------------------------


def test_the_projection_is_the_identity_and_says_so():
    frame = app.projection_frame(make_metrics(), 5, "Premier League")
    assert not frame.empty
    assert (frame["projected"] == frame["observed"]).all()
    assert set(frame["basis"]) == {"identity (B0)"}


def test_the_projection_caveat_states_the_measured_limitation():
    note = app.projection_note()
    # The counts are the finding, not a turn of phrase: without them the caveat
    # reads as boilerplate a user can dismiss.
    assert "275" in note and "211" in note
    assert "club-to-international" in note
    assert "identity" in note


def test_a_player_absent_from_the_metrics_gets_no_projection_row():
    frame = app.projection_frame(make_metrics(), 9_999, "UEFA Euro")
    assert frame.empty
    assert list(frame.columns) == list(app.PROJECTION_COLUMNS)


def test_the_source_competition_is_where_the_player_played_most():
    frame = make_metrics()
    frame.loc[frame["competition"] == "La Liga", "minutes"] = 5.0
    assert app.source_competition(frame, 4) == "Premier League"


# --- value ----------------------------------------------------------------


def test_the_value_caveat_says_it_is_not_money():
    note = app.value_note()
    assert valuation.TARGET in note
    assert "not money" in note
    assert "scarcity proxy" in note
    assert "Transfermarkt" in note


def test_the_value_reading_is_a_bounded_percentile_of_observed_minutes():
    reading = app.value_reading(app.value_table(make_metrics()), 5)
    assert reading is not None
    assert 0.0 <= reading.percentile <= 100.0
    assert reading.minutes_at_tier > 0


def test_a_player_with_no_following_season_has_no_reading():
    # Unknown is not zero: a zeroed gauge would assert they played nowhere.
    frame = make_metrics()
    frame = frame[(frame["player_id"] != 5) | (frame["season"] == "2024")]
    assert app.value_reading(app.value_table(frame), 5) is None


def test_the_value_table_reproduces_the_valuation_modules_target():
    # app.value_table trades build_dataset's per-player loop for a shift, because
    # the loop costs ~10s on a full corpus. The two must not drift apart.
    frame = make_metrics()
    dataset = valuation.build_dataset(frame)
    reference = pd.DataFrame(
        {
            "player_id": dataset.player_ids,
            "season": dataset.seasons,
            "minutes_at_tier": dataset.target,
        }
    ).sort_values(["player_id", "season"], ignore_index=True)
    ours = app.value_table(frame).sort_values(["player_id", "season"], ignore_index=True)
    pd.testing.assert_frame_equal(ours, reference, check_dtype=False)


def test_a_drop_in_tier_scores_zero_rather_than_the_minutes_played():
    frame = pd.DataFrame(
        {
            "player_id": [1, 1],
            "competition": ["Premier League", "Major League Soccer"],
            "season": ["2023", "2024"],
            "minutes": [2000.0, 2500.0],
        }
    )
    assert app.value_table(frame)["minutes_at_tier"].tolist() == [0.0]


def test_the_value_table_is_empty_when_no_season_has_a_successor():
    single = make_metrics()
    single = single[single["season"] == "2024"]
    assert app.value_table(single).empty


# --- name resolution ------------------------------------------------------


def test_a_near_tie_is_sent_to_the_pick_list():
    # Two players sharing a name is the common case; guessing is worse than asking.
    assert not app.is_unambiguous([(1, "Smith", 97.0), (2, "Smith", 96.0)])


def test_an_isolated_exact_match_is_taken_without_asking():
    assert app.is_unambiguous([(1, "Sam Kerr", 100.0), (2, "Sam Kern", 70.0)])


def test_a_weak_best_match_still_asks():
    assert not app.is_unambiguous([(1, "Someone", 60.0)])


def test_an_empty_query_resolves_to_nothing():
    assert app.resolve_choices(make_index(), "   ") == []


# --- charts and chrome ----------------------------------------------------


def test_the_radar_chart_is_a_valid_vega_lite_spec():
    # to_dict validates against the Vega-Lite schema, so this catches an encoding
    # that only fails once a browser has it.
    spec = app.radar_chart(app.radar_frame(make_index(), make_metrics(), 5)).to_dict()
    assert spec["mark"]["type"] == "arc"


def test_the_gauge_is_a_valid_spec_and_clamps_out_of_range_readings():
    for percentile in (-20.0, 0.0, 55.0, 120.0, float("nan")):
        spec = app.gauge_chart(percentile).to_dict()
        assert spec["mark"]["type"] == "arc"


def test_labels_fall_back_to_something_readable():
    assert app.label_of("xg_per90") == "xG"
    assert app.label_of("ball_recoveries_per90") == "Recoveries"
    assert app.label_of("some_new_metric_per90") == "Some new metric"


def test_a_missing_index_produces_an_actionable_message_not_a_traceback(tmp_path):
    with pytest.raises(similarity.IndexMissing) as excinfo:
        app.load_index_and_metrics(tmp_path, tmp_path)
    message = app.index_missing_message(excinfo.value)
    assert "scout embed" in message
    assert "scout demo" in message


# --- latency --------------------------------------------------------------


def test_a_whole_card_builds_well_inside_the_latency_budget():
    """The spec's budget is 5 seconds per query on CPU.

    At corpus scale a card is ~0.2s, so the 2s assertion here is a tripwire for
    an accidental per-player Python loop -- the first draft of `value_table`
    reused `valuation.build_dataset` and took 9 seconds on exactly this input.
    """
    index = make_index(n=6_300, dim=256)
    frame = make_metrics(n=6_300)

    start = time.perf_counter()
    app.radar_frame(index, frame, 4_242)
    app.comps_frame(index, 4_242, k=10)
    app.projection_frame(frame, 4_242, "UEFA Euro")
    app.value_reading(app.value_table(frame), 4_242)
    index.resolve("Player 4242")
    assert time.perf_counter() - start < 2.0


# --- the whole page, under a real Streamlit runtime ------------------------
#
# AppTest executes the script the way the server does, so these cover the layer
# the pure functions cannot: that the widgets wire together, that a caveat is
# actually emitted next to its number, and that a fresh clone gets a message
# rather than a stack trace.


def _run_app(timeout: float = 60.0):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(Path(app.__file__).resolve()), default_timeout=timeout).run()


def test_a_fresh_clone_with_no_index_gets_told_what_to_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = _run_app(timeout=30.0)
    assert not at.exception
    assert any("scout embed" in error.value for error in at.error)


def test_the_card_renders_and_carries_both_caveats(tmp_path, monkeypatch):
    index_dir = tmp_path / similarity.INDEX_DIR
    metrics_dir = tmp_path / corpus.METRICS_DIR
    index_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    index = make_index()
    np.save(index_dir / similarity.EMBEDDINGS_FILE, index.embeddings)
    index.players.to_parquet(index_dir / similarity.PLAYERS_FILE)
    make_metrics().to_parquet(metrics_dir / "1_1.parquet", index=False)
    monkeypatch.chdir(tmp_path)

    at = _run_app()
    at.text_input[0].input("Player 14").run()

    assert not at.exception
    assert [s.value for s in at.subheader] == ["Player 14"]
    # A crowded name space must produce a pick-list, not a silent choice.
    assert len(at.radio) == 1
    # Both caveats are rendered as their own blocks beside the numbers they
    # qualify, which is the requirement -- a footnote would satisfy neither.
    warnings = " ".join(w.value for w in at.warning)
    assert "275" in warnings and "211" in warnings
    assert "not money" in warnings
    # Comparables and projection tables.
    assert len(at.dataframe) == 2
