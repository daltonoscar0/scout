"""Tests for the projection-power measurement.

This is the most consequential number the project produces -- it decides whether
the projection module is a model or a documented null result -- so the counting
rules are pinned rather than trusted.
"""

from __future__ import annotations

from scout import population
from scout.crosswalk import PlayerSeason


def season(pid, comp, sid, minutes, gender="male"):
    return PlayerSeason(
        player_id=pid,
        player_name=f"p{pid}",
        player_nickname=None,
        country=None,
        competition_id=hash(comp) % 1000,
        season_id=sid,
        competition=comp,
        season=str(sid),
        gender=gender,
        team_id=1,
        team="T",
        seconds=int(minutes * 60),
    )


# --- classification -------------------------------------------------------


def test_national_team_competitions_are_international():
    assert population.is_international("FIFA World Cup")
    assert population.is_international("UEFA Euro")
    assert population.is_international("Women's World Cup")


def test_club_continental_competitions_are_not_international():
    # The Champions League is club football: team, teammates and league context
    # carry over, so it is not the domain shift the projection head needs.
    assert not population.is_international("Champions League")
    assert not population.is_international("UEFA Europa League")
    assert not population.is_international("Copa del Rey")
    assert not population.is_international("Premier League")


# --- pair decomposition ---------------------------------------------------


def test_two_editions_of_one_competition_are_a_same_competition_pair():
    # Euro 2020 to Euro 2024 is a pair, but a head fit on it learns "four years
    # older", not a domain map.
    rows = [season(1, "UEFA Euro", 43, 500), season(1, "UEFA Euro", 282, 500)]
    counts = population.analyse(rows, floors=(270,)).pairs[270]
    assert counts.same_competition == 1
    assert counts.club_to_international == 0
    assert counts.cross_league == 0


def test_a_club_to_international_move_is_counted_as_such():
    rows = [season(1, "Premier League", 1, 500), season(1, "FIFA World Cup", 2, 500)]
    counts = population.analyse(rows, floors=(270,)).pairs[270]
    assert counts.club_to_international == 1
    assert counts.cross_league == 0


def test_two_different_leagues_are_a_cross_league_pair():
    rows = [season(1, "Premier League", 1, 500), season(1, "La Liga", 2, 500)]
    counts = population.analyse(rows, floors=(270,)).pairs[270]
    assert counts.cross_league == 1
    assert counts.club_to_international == 0


def test_the_minutes_floor_is_applied_to_both_sides_of_a_pair():
    rows = [season(1, "Premier League", 1, 500), season(1, "FIFA World Cup", 2, 10)]
    assert population.analyse(rows, floors=(270,)).multi[270] == 0


def test_squad_membership_counts_more_players_than_real_minutes_do():
    rows = [
        season(1, "Premier League", 1, 500),
        season(1, "FIFA World Cup", 2, 500),
        season(2, "Premier League", 1, 500),
        season(2, "FIFA World Cup", 2, 0),  # named substitute, never played
    ]
    result = population.analyse(rows, floors=(0, 270))
    assert result.multi[0] == 2
    assert result.multi[270] == 1


# --- the verdict ----------------------------------------------------------


def make(per_gender):
    return population.Population(
        players=0,
        rows=0,
        rows_with_minutes=0,
        multi={},
        pairs={},
        league_pair_counts=population.Counter(),
        per_gender=per_gender,
    )


def test_the_verdict_is_assessed_per_gender_not_on_the_pooled_total():
    # 211 + 275 clears 300 pooled, but no head is ever fit on the pooled set:
    # essentially no player appears in both men's and women's competitions.
    # Summing them would be a bigger number no estimator gets to spend.
    ok, message = population.verdict(make({"male": 211, "female": 275}), threshold=300)
    assert not ok
    assert "211" in message and "275" in message


def test_a_stratum_clearing_the_bar_alone_still_fails():
    ok, message = population.verdict(make({"male": 400, "female": 100}), threshold=300)
    assert not ok
    assert "asymmetric" in message


def test_all_strata_clearing_the_bar_passes():
    ok, message = population.verdict(make({"male": 400, "female": 350}), threshold=300)
    assert ok
    assert "bootstrap" in message


def test_an_empty_stratum_does_not_veto_the_others():
    # A corpus with no women's competitions at all should be judged on the
    # stratum that exists, not failed for the one that does not.
    ok, _ = population.verdict(make({"male": 400, "female": 0}), threshold=300)
    assert ok


# --- report ---------------------------------------------------------------


def test_the_report_states_the_verdict_and_the_similarity_contrast():
    rows = [season(1, "Premier League", 1, 500), season(1, "FIFA World Cup", 2, 500)]
    text = population.report(population.analyse(rows))
    assert "INSUFFICIENT DATA" in text
    assert "similarity" in text.lower()
