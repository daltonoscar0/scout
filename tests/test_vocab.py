"""The vocabulary is a contract. These tests pin the parts that, if they drifted
silently, would produce plausible-looking but wrong nearest neighbours."""

from __future__ import annotations

import math

import pytest

from scout import vocab


def test_pad_is_index_zero():
    # A zero-filled array must be a valid padded sequence, so that the loss mask
    # is exactly `tokens != 0`. If PAD ever moves, padding silently becomes a
    # real token and the model trains on it.
    assert vocab.PAD_ID == 0
    assert vocab.ACTIONS[0] == "PAD"


def test_every_field_size_matches_its_table():
    assert len(vocab.ACTIONS) == vocab.ACTION_SIZE
    assert vocab.ZONE_SIZE == vocab.ZONE_COLS * vocab.ZONE_ROWS + 1
    assert vocab.DELTA_SECTORS * (len(vocab.DELTA_LENGTH_EDGES) + 1) + 1 == vocab.DELTA_SIZE
    assert 2 * 2 * len(vocab.BODY_PARTS) == vocab.FLAGS_SIZE
    assert len(vocab.DT_EDGES) + 3 == vocab.DT_SIZE


def test_action_names_are_unique():
    assert len(set(vocab.ACTIONS)) == len(vocab.ACTIONS)


def test_fingerprint_is_stable_across_calls():
    assert vocab.vocabulary_fingerprint() == vocab.vocabulary_fingerprint()


def test_fingerprint_changes_when_the_vocabulary_changes(monkeypatch):
    before = vocab.vocabulary_fingerprint()
    monkeypatch.setattr(vocab, "BODY_PARTS", ("none", "right_foot"))
    assert vocab.vocabulary_fingerprint() != before


# --- zones ----------------------------------------------------------------


def test_zone_corners_map_to_opposite_cells():
    bottom_left = vocab.zone_index(0.0, 0.0)
    top_right = vocab.zone_index(vocab.PITCH_LENGTH, vocab.PITCH_WIDTH)
    assert bottom_left == 0
    assert top_right == vocab.ZONE_COLS * vocab.ZONE_ROWS - 1
    assert bottom_left != top_right


def test_zone_clamps_rather_than_dropping_off_pitch_events():
    # Real events leave the pitch: a shot crossing the goal line, a throw-in
    # from behind the touchline. sportio's own bounds allow x in [-10, 115].
    assert vocab.zone_index(-5.0, -3.0) == 0
    assert vocab.zone_index(112.0, 71.0) == vocab.ZONE_COLS * vocab.ZONE_ROWS - 1


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_zone_handles_missing_location(bad):
    # ~28 events per match genuinely have no $.location.
    assert vocab.zone_index(bad, 30.0) == vocab.ZONE_NULL
    assert vocab.zone_index(50.0, bad) == vocab.ZONE_NULL


def test_every_pitch_coordinate_yields_a_valid_zone():
    for x in range(0, 106, 7):
        for y in range(0, 69, 5):
            assert 0 <= vocab.zone_index(float(x), float(y)) < vocab.ZONE_SIZE


# --- deltas ---------------------------------------------------------------


def test_forward_pass_lands_in_sector_zero():
    assert vocab.delta_index(20.0, 0.0) // (len(vocab.DELTA_LENGTH_EDGES) + 1) == 0


def test_opposite_directions_differ():
    forward = vocab.delta_index(20.0, 0.0)
    backward = vocab.delta_index(-20.0, 0.0)
    assert forward != backward


def test_sector_zero_is_centred_on_forward_not_straddling_it():
    # A slightly-left and slightly-right forward pass should share a sector.
    eps = math.radians(10)
    left = vocab.delta_index(20 * math.cos(eps), 20 * math.sin(eps))
    right = vocab.delta_index(20 * math.cos(-eps), 20 * math.sin(-eps))
    assert left == right


def test_longer_passes_land_in_higher_length_buckets():
    span = len(vocab.DELTA_LENGTH_EDGES) + 1
    short = vocab.delta_index(3.0, 0.0) % span
    medium = vocab.delta_index(20.0, 0.0) % span
    long = vocab.delta_index(70.0, 0.0) % span
    assert short < medium < long


def test_a_touch_in_place_is_not_a_movement():
    assert vocab.delta_index(0.0, 0.0) == vocab.DELTA_NONE
    assert vocab.delta_index(None, None) == vocab.DELTA_NONE


def test_every_delta_is_in_range():
    for dx in range(-60, 61, 6):
        for dy in range(-30, 31, 6):
            assert 0 <= vocab.delta_index(float(dx), float(dy)) < vocab.DELTA_SIZE


# --- flags ----------------------------------------------------------------


def test_flags_are_injective():
    seen = {
        vocab.flags_index(p, c, b)
        for p in (True, False)
        for c in (True, False)
        for b in vocab.BODY_PARTS
    }
    assert len(seen) == vocab.FLAGS_SIZE


def test_absent_statsbomb_booleans_read_as_false():
    # StatsBomb sparse booleans are present-or-absent, never literally false,
    # so callers use .get(key, False) and None must not become a third state.
    assert vocab.flags_index(None, None, "none") == vocab.flags_index(False, False, "none")


def test_unknown_body_part_falls_back_to_other():
    assert vocab.flags_index(False, False, "left_shoulder") == vocab.flags_index(
        False, False, "other"
    )


# --- dt -------------------------------------------------------------------


def test_period_start_outranks_any_measured_gap():
    assert vocab.dt_index(0.5, period_start=True) == vocab.DT_PERIOD_START


def test_dt_is_monotonic():
    assert vocab.dt_index(0.01) <= vocab.dt_index(1.0) <= vocab.dt_index(45.0)


@pytest.mark.parametrize("bad", [None, float("nan"), -1.0])
def test_dt_rejects_nonsense_without_raising(bad):
    assert vocab.dt_index(bad) == vocab.DT_UNKNOWN


def test_every_dt_is_in_range():
    for dt in (0.0, 0.05, 0.3, 1.2, 4.0, 12.0, 90.0, 3600.0):
        assert 0 <= vocab.dt_index(dt) < vocab.DT_SIZE
