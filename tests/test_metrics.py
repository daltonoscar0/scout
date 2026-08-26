"""Metric tests.

The xA join is the subtle one: it credits a shot's xG to whoever made the pass
the shot's `key_pass_id` points at. Getting that wrong would not crash, it would
just attribute chance creation to the wrong player.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scout import metrics


def events(rows):
    base = {
        "match_id": 1,
        "competition": "C",
        "season": "S",
        "team_id": 10,
        "team": "T",
        "player_id": 100,
        "player": "P",
        "t": 0.0,
        "period": 1,
        "type": "pass",
        "subtype": None,
        "x": 50.0,
        "y": 34.0,
        "end_x": 60.0,
        "end_y": 34.0,
        "outcome": "complete",
        "pressure": False,
        "body_part": "right_foot",
        "extra": "{}",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def totals_for(frame):
    totals = metrics.empty_totals()
    metrics.accumulate(frame, totals)
    return totals


# --- progressive ----------------------------------------------------------


def test_a_long_ball_toward_goal_is_progressive():
    assert metrics.is_progressive(20.0, 34.0, 60.0, 34.0)


def test_a_backward_pass_is_not_progressive():
    assert not metrics.is_progressive(60.0, 34.0, 20.0, 34.0)


def test_a_sideways_pass_is_not_progressive():
    assert not metrics.is_progressive(50.0, 10.0, 50.0, 58.0)


def test_a_short_pass_near_the_box_is_not_progressive_on_ratio_alone():
    # 25% of a short remaining distance is a couple of metres. The absolute
    # gate exists to stop that counting.
    assert not metrics.is_progressive(99.0, 34.0, 101.0, 34.0)


def test_entering_the_penalty_area_always_counts():
    # In the last 16 metres the relative test stops discriminating, so a ball
    # played into the box counts regardless.
    assert metrics.is_progressive(90.0, 10.0, 95.0, 30.0)


def test_a_pass_already_inside_the_box_is_judged_on_distance():
    assert not metrics.is_progressive(95.0, 30.0, 96.0, 31.0)


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_an_action_with_no_end_point_is_not_progressive(missing):
    assert not metrics.is_progressive(50.0, 34.0, missing, 34.0)


# --- accumulation ---------------------------------------------------------


def test_a_completed_pass_counts_once():
    totals = totals_for(events([{}]))
    assert totals[100].passes == 1
    assert totals[100].passes_completed == 1


def test_an_incomplete_pass_is_not_counted_as_completed():
    totals = totals_for(events([{"outcome": "incomplete"}]))
    assert totals[100].passes == 1
    assert totals[100].passes_completed == 0
    assert totals[100].progressive_passes == 0


def test_a_missing_outcome_does_not_raise():
    # sportio's columns are nullable, so outcome can be pd.NA. Comparing NA to a
    # string yields NA, and branching on that raises.
    totals = totals_for(events([{"type": "duel", "outcome": pd.NA}]))
    assert totals[100].duels_won == 0


def test_shots_accumulate_expected_goals():
    frame = events(
        [
            {"type": "shot", "outcome": "goal", "extra": json.dumps({"statsbomb_xg": 0.4})},
            {"type": "shot", "outcome": "saved", "extra": json.dumps({"statsbomb_xg": 0.1})},
        ]
    )
    totals = totals_for(frame)
    assert totals[100].shots == 2
    assert totals[100].goals == 1
    assert totals[100].xg == pytest.approx(0.5)


def test_a_shot_with_no_xg_field_contributes_zero_rather_than_crashing():
    totals = totals_for(events([{"type": "shot", "outcome": "goal"}]))
    assert totals[100].xg == 0.0
    assert totals[100].goals == 1


# --- the xA join ----------------------------------------------------------


def test_expected_assists_are_credited_to_the_passer():
    frame = events(
        [
            {"player_id": 7, "type": "pass", "extra": json.dumps({"id": "pass-1"})},
            {
                "player_id": 9,
                "type": "shot",
                "outcome": "goal",
                "extra": json.dumps({"statsbomb_xg": 0.3, "key_pass_id": "pass-1"}),
            },
        ]
    )
    totals = totals_for(frame)
    assert totals[7].xa == pytest.approx(0.3)
    assert totals[7].key_passes == 1
    # The shooter keeps the xG and earns no xA for their own shot.
    assert totals[9].xg == pytest.approx(0.3)
    assert totals[9].xa == 0.0


def test_an_unassisted_shot_credits_nobody_with_an_assist():
    frame = events(
        [
            {"player_id": 9, "type": "shot", "extra": json.dumps({"statsbomb_xg": 0.2})},
        ]
    )
    totals = totals_for(frame)
    assert all(record.xa == 0.0 for record in totals.values())


def test_a_key_pass_pointing_at_an_unknown_event_is_dropped_not_misattributed():
    frame = events(
        [
            {"player_id": 7, "type": "pass", "extra": json.dumps({"id": "pass-1"})},
            {
                "player_id": 9,
                "type": "shot",
                "extra": json.dumps({"statsbomb_xg": 0.3, "key_pass_id": "nonexistent"}),
            },
        ]
    )
    totals = totals_for(frame)
    assert totals[7].xa == 0.0


# --- per 90 ---------------------------------------------------------------


def test_rates_scale_to_ninety_minutes():
    totals = totals_for(events([{"type": "shot", "extra": json.dumps({"statsbomb_xg": 0.5})}]))
    frame = metrics.to_frame(totals, {100: 45.0})
    assert float(frame.iloc[0]["xg_per90"]) == pytest.approx(1.0)


def test_a_player_with_no_minutes_is_dropped_rather_than_divided_by_zero():
    totals = totals_for(events([{}]))
    assert metrics.to_frame(totals, {100: 0.0}).empty


def test_every_declared_per90_column_is_produced():
    totals = totals_for(events([{}]))
    frame = metrics.to_frame(totals, {100: 90.0})
    for name in metrics.PER_90_COLUMNS:
        assert f"{name}_per90" in frame.columns


def test_xg_xa_is_the_sum_of_its_parts():
    frame = events(
        [
            {"player_id": 7, "type": "pass", "extra": json.dumps({"id": "p1"})},
            {
                "player_id": 7,
                "type": "shot",
                "extra": json.dumps({"statsbomb_xg": 0.2, "key_pass_id": "p1"}),
            },
        ]
    )
    totals = totals_for(frame)
    row = metrics.to_frame(totals, {7: 90.0}).iloc[0]
    assert float(row["xg_xa"]) == pytest.approx(float(row["xg"]) + float(row["xa"]))


def test_an_empty_frame_produces_no_rows():
    assert metrics.to_frame(metrics.empty_totals(), {}).empty
