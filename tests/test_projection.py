"""Projection baseline tests.

The load-bearing test here is `test_random_embeddings_do_not_beat_identity`. An
earlier version of `evaluate` fitted and scored on the same pairs, and under that
scheme a ten-parameter ridge beat identity using embeddings drawn from pure
noise. Any conclusion drawn from that table would have been wrong in the exact
direction that flatters the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scout import projection


def metrics_frame(n_players=120, seed=0, gender="male"):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_players + 1):
        level = rng.normal(0.4, 0.15)
        for competition, season in (("Premier League", "2015/2016"), ("FIFA World Cup", "2018")):
            rows.append(
                {
                    "player_id": pid,
                    "competition": competition,
                    "season": season,
                    "gender": gender,
                    "minutes": 1000.0,
                    "xg_xa_per90": max(level + rng.normal(0, 0.05), 0.0),
                    "progressive_actions_per90": max(rng.normal(6, 2), 0.0),
                }
            )
    return pd.DataFrame(rows)


# --- pair construction ----------------------------------------------------


def test_pairs_are_built_across_competitions():
    pairs = projection.build_pairs(metrics_frame(n_players=3))
    assert len(pairs) == 3
    assert all(p.source_competition != p.target_competition for p in pairs)


def test_same_competition_pairs_are_excluded_by_default():
    # Euro 2020 to Euro 2024 is a real pair, but a model fit on it learns
    # "four years older", not how a player translates between competitions.
    frame = pd.DataFrame(
        [
            {
                "player_id": 1,
                "competition": "UEFA Euro",
                "season": season,
                "gender": "male",
                "minutes": 500.0,
                "xg_xa_per90": 0.3,
                "progressive_actions_per90": 5.0,
            }
            for season in ("2020", "2024")
        ]
    )
    assert projection.build_pairs(frame) == []
    assert len(projection.build_pairs(frame, cross_competition_only=False)) == 1


def test_the_minutes_floor_excludes_thin_seasons():
    frame = metrics_frame(n_players=2)
    frame.loc[frame["competition"] == "FIFA World Cup", "minutes"] = 10.0
    assert projection.build_pairs(frame) == []


def test_an_empty_frame_yields_no_pairs():
    assert projection.build_pairs(pd.DataFrame()) == []


# --- the honesty guard ----------------------------------------------------


def test_random_embeddings_do_not_beat_identity():
    # Fitted out of sample, an embedding carrying no information must not
    # resolve as an improvement. In sample, it did.
    rng = np.random.default_rng(1)
    pairs = projection.build_pairs(metrics_frame())
    noise = {p.player_id: rng.normal(size=32).astype(np.float32) for p in pairs}
    for target in projection.TARGETS:
        comparison = projection.evaluate(pairs, target, "male", noise)
        assert not comparison.resolved(), f"noise resolved as signal on {target}"


def test_an_informative_embedding_can_beat_identity():
    # The mirror: if the guard rejected everything, it would be useless. Here the
    # embedding literally encodes the target, so it must win.
    frame = metrics_frame()
    pairs = projection.build_pairs(frame)
    informative = {
        p.player_id: np.array([p.target["xg_xa_per90"]] * 8, dtype=np.float32) for p in pairs
    }
    comparison = projection.evaluate(pairs, "xg_xa_per90", "male", informative)
    assert comparison.resolved()


def test_folds_never_split_a_player_across_train_and_test():
    pairs = projection.build_pairs(metrics_frame(n_players=40))
    folds = projection._player_folds(pairs, 5, seed=0)
    for fold in folds:
        held_players = {pairs[i].player_id for i in fold}
        others = {pairs[i].player_id for i in range(len(pairs)) if i not in set(fold.tolist())}
        assert not (held_players & others)


def test_every_pair_is_predicted_exactly_once_across_folds():
    pairs = projection.build_pairs(metrics_frame(n_players=40))
    folds = projection._player_folds(pairs, 5, seed=0)
    covered = sorted(i for fold in folds for i in fold.tolist())
    assert covered == list(range(len(pairs)))


# --- scoring --------------------------------------------------------------


def test_identity_reproduces_the_source_value():
    pairs = projection.build_pairs(metrics_frame(n_players=5))
    predicted = projection.predict_identity(pairs, "xg_xa_per90")
    assert predicted == pytest.approx([p.source["xg_xa_per90"] for p in pairs])


def test_a_score_carries_a_confidence_interval_around_its_mae():
    comparison = projection.evaluate(projection.build_pairs(metrics_frame()), "xg_xa_per90", "male")
    for score in comparison.scores:
        assert score.low <= score.mae <= score.high


def test_beats_requires_non_overlapping_intervals():
    better = projection.Score("better", mae=1.0, rmse=1.0, low=0.9, high=1.1, n=10)
    worse = projection.Score("worse", mae=2.0, rmse=2.0, low=1.0, high=3.0, n=10)
    # Lower MAE but the intervals overlap: not resolvable at this sample size.
    assert not better.beats(worse)
    clearly_worse = projection.Score("worse", mae=5.0, rmse=5.0, low=4.0, high=6.0, n=10)
    assert better.beats(clearly_worse)


def test_an_empty_gender_stratum_scores_nothing_rather_than_crashing():
    comparison = projection.evaluate(
        projection.build_pairs(metrics_frame()), "xg_xa_per90", "female"
    )
    assert comparison.n_pairs == 0
    assert comparison.scores == []


# --- report ---------------------------------------------------------------


def test_the_report_states_the_missing_age_feature():
    pairs = projection.build_pairs(metrics_frame())
    text = projection.report([projection.evaluate(pairs, "xg_xa_per90", "male")])
    assert "Age is absent" in text
    assert "out of sample" in text


def test_the_report_says_so_when_nothing_beats_identity():
    rng = np.random.default_rng(2)
    pairs = projection.build_pairs(metrics_frame())
    noise = {p.player_id: rng.normal(size=32).astype(np.float32) for p in pairs}
    text = projection.report([projection.evaluate(pairs, "xg_xa_per90", "male", noise)])
    assert "No predictor beats identity" in text
