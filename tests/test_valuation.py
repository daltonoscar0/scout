"""Valuation tests.

Two things here are easy to get silently wrong and expensive if you do: the
next-season join (off by one and every label belongs to the wrong season) and the
train/test split (random rows leak a player across both sides).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scout import valuation

FEATURE_DEFAULTS = {
    "matches": 20,
    "xg_per90": 0.2,
    "xa_per90": 0.1,
    "shots_per90": 1.5,
    "progressive_passes_per90": 4.0,
    "progressive_carries_per90": 3.0,
    "passes_per90": 40.0,
    "key_passes_per90": 1.0,
    "pressures_per90": 15.0,
    "duels_won_per90": 3.0,
    "interceptions_per90": 1.2,
    "ball_recoveries_per90": 5.0,
    "dribbles_completed_per90": 1.0,
    "touches_per90": 60.0,
    "pass_completion": 0.8,
}


def row(player_id, season, competition, minutes, **overrides):
    return {
        "player_id": player_id,
        "competition": competition,
        "season": season,
        "gender": "male",
        "minutes": minutes,
        **FEATURE_DEFAULTS,
        **overrides,
    }


def frame(rows):
    return pd.DataFrame(rows)


# --- tiers ----------------------------------------------------------------


def test_known_competitions_get_their_declared_tier():
    assert valuation.tier_of("Premier League") < valuation.tier_of("Major League Soccer")
    assert valuation.tier_of("FIFA World Cup") == 0


def test_an_unlisted_competition_falls_back_rather_than_raising():
    assert valuation.tier_of("Sunday League") == valuation.DEFAULT_TIER


# --- season ordering ------------------------------------------------------


def test_both_season_label_shapes_sort_correctly():
    seasons = ["2023", "2015/2016", "2019/2020"]
    assert sorted(seasons, key=valuation.season_sort_key) == ["2015/2016", "2019/2020", "2023"]


def test_an_unparseable_season_does_not_raise():
    assert valuation.season_sort_key("not a season")[0] == 0


# --- the next-season join -------------------------------------------------


def test_the_target_is_the_following_seasons_minutes():
    data = frame(
        [
            row(1, "2015/2016", "Premier League", 1000),
            row(1, "2016/2017", "Premier League", 2500),
        ]
    )
    dataset = valuation.build_dataset(data)
    assert len(dataset) == 1
    assert dataset.target[0] == 2500
    # The features must come from the *earlier* season, not the labelled one.
    assert dataset.features.iloc[0]["minutes"] == 1000


def test_a_players_last_season_is_excluded_not_zero_filled():
    # "We do not know what happened next" and "they played nowhere" are
    # different. Zero-filling would teach the model every career ends.
    data = frame([row(1, "2015/2016", "Premier League", 1000)])
    assert len(valuation.build_dataset(data)) == 0


def test_dropping_a_tier_scores_zero():
    data = frame(
        [
            row(1, "2015/2016", "Premier League", 2000),
            row(1, "2016/2017", "Indian Super league", 3000),
        ]
    )
    dataset = valuation.build_dataset(data)
    assert dataset.target[0] == 0.0


def test_moving_up_a_tier_keeps_the_minutes():
    data = frame(
        [
            row(1, "2015/2016", "Major League Soccer", 2000),
            row(1, "2016/2017", "Premier League", 1800),
        ]
    )
    assert valuation.build_dataset(data).target[0] == 1800


def test_three_seasons_yield_two_rows():
    data = frame(
        [
            row(1, "2015/2016", "Premier League", 1000),
            row(1, "2016/2017", "Premier League", 2000),
            row(1, "2017/2018", "Premier League", 3000),
        ]
    )
    dataset = valuation.build_dataset(data)
    assert len(dataset) == 2
    assert list(dataset.target) == [2000, 3000]


def test_an_empty_frame_yields_an_empty_dataset():
    assert len(valuation.build_dataset(pd.DataFrame())) == 0


# --- the split ------------------------------------------------------------


def test_the_split_holds_out_the_most_recent_seasons():
    rows = []
    for pid in range(1, 20):
        for season in ("2015/2016", "2016/2017", "2017/2018", "2018/2019"):
            rows.append(row(pid, season, "Premier League", 1500))
    dataset = valuation.build_dataset(frame(rows))
    train, test = valuation.split_by_season(dataset)
    held = {s for s, keep in zip(dataset.seasons, test, strict=True) if keep}
    trained = {s for s, keep in zip(dataset.seasons, train, strict=True) if keep}
    assert held and trained
    # Every held-out season is later than every trained season.
    assert min(map(valuation.season_sort_key, held)) > max(map(valuation.season_sort_key, trained))


def test_the_split_is_a_partition():
    rows = [
        row(pid, season, "Premier League", 1500)
        for pid in range(1, 10)
        for season in ("2015/2016", "2016/2017", "2017/2018")
    ]
    dataset = valuation.build_dataset(frame(rows))
    train, test = valuation.split_by_season(dataset)
    assert (train | test).all()
    assert not (train & test).any()


# --- fitting --------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(1, 300):
        quality = rng.normal()
        for season in ("2015/2016", "2016/2017", "2017/2018"):
            rows.append(
                row(
                    pid,
                    season,
                    "Premier League",
                    max(200 + 900 * quality + rng.normal(0, 150), 50),
                    xg_per90=max(0.2 + 0.15 * quality, 0),
                )
            )
    dataset = valuation.build_dataset(frame(rows))
    return dataset, valuation.fit(dataset)


def test_the_model_beats_predicting_the_mean_when_there_is_signal(fitted):
    _, model = fitted
    assert model.metrics["mae"] < model.metrics["baseline_mae"]


def test_the_model_reports_its_holdout_size(fitted):
    _, model = fitted
    assert model.metrics["n_train"] > 0
    assert model.metrics["n_test"] > 0


def test_residuals_line_up_with_their_rows(fitted):
    dataset, model = fitted
    result = valuation.residuals(model, dataset)
    assert len(result) == len(dataset)
    assert list(result["player_id"]) == list(dataset.player_ids)
    assert result["residual"].to_numpy() == pytest.approx(
        result["actual"].to_numpy() - result["predicted"].to_numpy()
    )


# --- the question the spec insists on having answered ---------------------


def test_pure_noise_residuals_are_reported_as_no_signal():
    rng = np.random.default_rng(1)
    result = pd.DataFrame(
        {
            "player_id": np.repeat(np.arange(200), 3),
            "season": ["2015/2016", "2016/2017", "2017/2018"] * 200,
            "residual": rng.normal(size=600),
            "predicted": 0.0,
            "actual": 0.0,
        }
    )
    correlation, answer = valuation.residual_predicts_next_window(result)
    assert abs(correlation) < 0.15
    assert answer.startswith("no")


def test_a_persistent_residual_is_reported_as_weak_signal():
    # The mirror: if the check said "no" to everything it would be useless.
    players = np.repeat(np.arange(200), 3)
    level = np.repeat(np.random.default_rng(2).normal(size=200), 3)
    result = pd.DataFrame(
        {
            "player_id": players,
            "season": ["2015/2016", "2016/2017", "2017/2018"] * 200,
            "residual": level,
            "predicted": 0.0,
            "actual": 0.0,
        }
    )
    correlation, answer = valuation.residual_predicts_next_window(result)
    assert correlation > 0.5
    assert answer.startswith("yes")


def test_too_few_pairs_is_reported_as_unknown_not_as_no():
    result = pd.DataFrame(
        {
            "player_id": [1, 1],
            "season": ["2015/2016", "2016/2017"],
            "residual": [0.5, -0.5],
            "predicted": 0.0,
            "actual": 0.0,
        }
    )
    _, answer = valuation.residual_predicts_next_window(result)
    assert "too few" in answer


# --- report ---------------------------------------------------------------


def test_the_report_says_the_target_is_not_money(fitted):
    dataset, model = fitted
    text = valuation.report(model, "no. Spearman +0.01", 0.01)
    assert "not money" in text
    assert valuation.TARGET in text


def test_the_report_names_every_feature_open_data_cannot_supply(fitted):
    _, model = fitted
    text = valuation.report(model, "no", 0.0)
    for name in valuation.MISSING_FEATURES:
        assert name in text
