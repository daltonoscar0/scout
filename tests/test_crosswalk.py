"""Minutes arithmetic decides which players are eligible for similarity and how
big the projection population is. A silent bug here would not crash anything --
it would just quietly produce the wrong answer to the most important question in
the project."""

from __future__ import annotations

import pytest

from scout import crosswalk


def player(positions, player_id=1, name="A Player"):
    return {
        "player_id": player_id,
        "player_name": name,
        "player_nickname": None,
        "country": {"id": 1, "name": "Nowhere"},
        "cards": [],
        "positions": positions,
    }


def spell(start, end, position="Center Midfield"):
    return {"position": position, "from": start, "to": end, "from_period": 1, "to_period": 2}


# --- clock parsing --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("00:00", 0), ("45:00", 2700), ("89:31", 5371), ("120:00", 7200), ("01:02:03", 3723)],
)
def test_parse_clock(text, seconds):
    assert crosswalk.parse_clock(text) == seconds


@pytest.mark.parametrize("bad", [None, "", "not a clock", "12:xy"])
def test_parse_clock_rejects_nonsense_without_raising(bad):
    assert crosswalk.parse_clock(bad) is None


def test_the_clock_runs_continuously_across_periods():
    # A second-half substitution reads 89:31, not 44:31. If this were ever
    # per-period, every second-half spell would be credited 45 extra minutes.
    assert crosswalk.parse_clock("89:31") > crosswalk.parse_clock("45:00")


# --- minutes --------------------------------------------------------------


def test_a_named_substitute_who_never_played_has_no_minutes():
    # 20 of 52 players in a sampled match had no positions entry. Counting them
    # would inflate the projection population by roughly a third.
    assert crosswalk.player_minutes(player([]), 5580) == 0
    assert crosswalk.player_minutes(player(None), 5580) == 0


def test_a_full_match_uses_the_estimated_end_when_to_is_null():
    p = player([spell("00:00", None)])
    assert crosswalk.player_minutes(p, 5580) == 5580


def test_a_substituted_player_is_credited_only_their_spell():
    p = player([spell("00:00", "60:00")])
    assert crosswalk.player_minutes(p, 5580) == 3600


def test_a_tactical_shift_does_not_double_count():
    # Two consecutive spells meeting at the same timestamp is the normal shape
    # for a tactical shift. Summing them naively is correct here, but merging
    # keeps it correct when they overlap too.
    p = player([spell("00:00", "89:31"), spell("89:31", None)])
    assert crosswalk.player_minutes(p, 5580) == 5580


def test_overlapping_spells_are_merged_not_summed():
    # A malformed file must not be able to produce a 200-minute appearance.
    p = player([spell("00:00", "60:00"), spell("30:00", "90:00")])
    assert crosswalk.player_minutes(p, 5580) == 5400


def test_disjoint_spells_are_both_counted():
    p = player([spell("00:00", "20:00"), spell("70:00", "90:00")])
    assert crosswalk.player_minutes(p, 5580) == 1200 + 1200


def test_a_zero_length_spell_contributes_nothing():
    p = player([spell("45:00", "45:00")])
    assert crosswalk.player_minutes(p, 5580) == 0


def test_a_spell_with_no_start_is_skipped_rather_than_crashing():
    p = player([{"position": "X", "from": None, "to": "90:00"}])
    assert crosswalk.player_minutes(p, 5580) == 0


# --- match end ------------------------------------------------------------


def test_match_end_uses_the_latest_timestamp_seen():
    teams = [{"lineup": [player([spell("00:00", "97:12")])]}]
    assert crosswalk.match_end_seconds(teams) == 97 * 60 + 12


def test_match_end_is_floored_so_a_quiet_match_is_not_shortened():
    # If nobody was substituted, the latest timestamp is 00:00. Without the
    # floor every starter would be credited zero minutes.
    teams = [{"lineup": [player([spell("00:00", None)])]}]
    assert crosswalk.match_end_seconds(teams) == crosswalk.DEFAULT_MATCH_END


def test_extra_time_is_picked_up_from_the_timestamps():
    teams = [{"lineup": [player([spell("00:00", "118:40")])]}]
    assert crosswalk.match_end_seconds(teams) > 110 * 60


# --- multi-competition selection ------------------------------------------


def season(pid, cid, sid, minutes):
    return crosswalk.PlayerSeason(
        player_id=pid,
        player_name="x",
        player_nickname=None,
        country=None,
        competition_id=cid,
        season_id=sid,
        competition="C",
        season="S",
        gender="male",
        team_id=1,
        team="T",
        seconds=int(minutes * 60),
    )


def test_multi_competition_applies_the_minutes_floor_to_every_side():
    rows = [
        season(1, 10, 1, 500),  # clears
        season(1, 20, 2, 500),  # clears -> player 1 qualifies
        season(2, 10, 1, 500),  # clears
        season(2, 20, 2, 10),  # does not clear -> player 2 must not qualify
    ]
    qualifying = crosswalk.multi_competition(rows, min_minutes=270)
    assert set(qualifying) == {1}


def test_two_seasons_of_one_competition_still_count_as_two_rows():
    # Deliberate: Euro 2020 -> Euro 2024 is a real pair of comp-seasons. Whether
    # it is a useful *domain shift* is a separate question the report answers;
    # this function only reports membership.
    rows = [season(1, 55, 43, 500), season(1, 55, 282, 500)]
    assert set(crosswalk.multi_competition(rows, 270)) == {1}


def test_main_position_is_the_most_frequent_one():
    row = season(1, 10, 1, 500)
    row.positions = {"Left Back": 3, "Left Wing": 9}
    assert row.main_position == "Left Wing"


def test_main_position_is_none_when_the_player_never_took_the_pitch():
    assert season(1, 10, 1, 0).main_position is None
