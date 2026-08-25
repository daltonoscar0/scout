"""The player/minutes index, built from StatsBomb lineups.

sportio does not expose one yet, and scout needs it in three places: to apply a
minutes floor before offering a player for similarity, to supply the
`player_nickname` field that makes any external name join workable, and to
answer the question that decides whether the projection module is worth
building -- how many players genuinely appear, with real minutes, in more than
one competition.

Why minutes and not squad membership: a named substitute who never comes on
appears in the lineup and emits no events. In one sampled match, 20 of 52 listed
players had no `positions` entry at all. Counting them would inflate the
projection population by roughly a third with players who contribute nothing to
train on.

Match-clock note: `from`/`to` are "MM:SS" strings on a clock that runs
continuously across periods (a second-half substitution reads "89:31", not
"44:31"), so a duration is a plain subtraction. `to` is null when the player was
on at the final whistle.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scout import fetch

# Floor for the assumed match end when every position on a match ends null.
# Regulation plus a nominal stoppage allowance; see match_end_seconds().
DEFAULT_MATCH_END = 93 * 60


def parse_clock(value: str | None) -> int | None:
    """Parse a "MM:SS" (or "HH:MM:SS") match clock into seconds."""
    if not value:
        return None
    parts = value.split(":")
    try:
        numbers = [int(p) for p in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    return None


def match_end_seconds(teams: list[dict[str, Any]]) -> int:
    """Estimate when the match ended, in match-clock seconds.

    Lineups alone do not record a final whistle. The latest timestamp appearing
    anywhere in the match -- a substitution off, a tactical shift -- is a lower
    bound on it, and in practice a tight one for any match with substitutions.
    Floored at DEFAULT_MATCH_END so a match with no recorded events late on does
    not silently credit everyone with 60 minutes.

    This is an approximation, and it is the one soft number in this module. It
    biases minutes slightly low for matches whose last substitution came early,
    and it handles extra time correctly because those matches carry timestamps
    past 90 minutes. Errors of a minute or two do not move a 270-minute
    threshold, which is all this feeds.
    """
    latest = 0
    for team in teams:
        for player in team.get("lineup", []):
            for position in player.get("positions") or []:
                for key in ("to", "from"):
                    seconds = parse_clock(position.get(key))
                    if seconds is not None:
                        latest = max(latest, seconds)
    return max(latest, DEFAULT_MATCH_END)


def player_minutes(player: dict[str, Any], end_seconds: int) -> int:
    """Seconds on the pitch for one player in one match.

    Overlapping position spells are not double counted -- a tactical shift ends
    one spell and starts another at the same timestamp -- but the intervals are
    merged rather than summed, so a malformed file cannot produce a 200-minute
    appearance.
    """
    spans: list[tuple[int, int]] = []
    for position in player.get("positions") or []:
        start = parse_clock(position.get("from"))
        if start is None:
            continue
        finish = parse_clock(position.get("to"))
        if finish is None:
            finish = end_seconds
        if finish > start:
            spans.append((start, finish))
    if not spans:
        return 0

    spans.sort()
    merged_total = 0
    current_start, current_end = spans[0]
    for start, finish in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, finish)
        else:
            merged_total += current_end - current_start
            current_start, current_end = start, finish
    merged_total += current_end - current_start
    return merged_total


@dataclass
class PlayerSeason:
    player_id: int
    player_name: str
    player_nickname: str | None
    country: str | None
    competition_id: int
    season_id: int
    competition: str
    season: str
    gender: str | None
    team_id: int
    team: str
    seconds: int = 0
    matches: int = 0
    appearances: int = 0  # matches with more than zero minutes
    positions: dict[str, int] = field(default_factory=dict)

    @property
    def minutes(self) -> float:
        return self.seconds / 60.0

    @property
    def main_position(self) -> str | None:
        if not self.positions:
            return None
        return max(self.positions.items(), key=lambda kv: kv[1])[0]


def build(
    spine_path: Path | str = "data/raw/spine.json",
    root: Path | None = None,
    *,
    require_cached: bool = True,
) -> list[PlayerSeason]:
    """Aggregate every cached lineup into per-(player, competition-season) rows.

    With require_cached=True this reads only what is already on disk and reports
    what is missing, so a partial fetch yields a partial but honest index rather
    than silently triggering thousands of requests.
    """
    root = fetch.cache_root() if root is None else root
    spine = json.loads(Path(spine_path).read_text())

    rows: dict[tuple[int, int, int], PlayerSeason] = {}
    missing = 0
    seen_matches = 0

    for comp_season in spine:
        cid, sid = comp_season["cid"], comp_season["sid"]
        for match in fetch.matches(cid, sid, root=root):
            match_id = match["match_id"]
            if require_cached:
                teams = fetch.read_cached(root, f"lineups/{match_id}.json")
                if teams is None:
                    missing += 1
                    continue
            else:
                teams = fetch.lineup(match_id, root=root)

            seen_matches += 1
            end_seconds = match_end_seconds(teams)
            for team in teams:
                for player in team.get("lineup", []):
                    key = (player["player_id"], cid, sid)
                    row = rows.get(key)
                    if row is None:
                        row = PlayerSeason(
                            player_id=player["player_id"],
                            player_name=player.get("player_name", ""),
                            player_nickname=player.get("player_nickname"),
                            country=(player.get("country") or {}).get("name"),
                            competition_id=cid,
                            season_id=sid,
                            competition=comp_season["comp"],
                            season=comp_season["season"],
                            gender=comp_season.get("gender"),
                            team_id=team.get("team_id", -1),
                            team=team.get("team_name", ""),
                        )
                        rows[key] = row
                    seconds = player_minutes(player, end_seconds)
                    row.matches += 1
                    row.seconds += seconds
                    if seconds > 0:
                        row.appearances += 1
                    for position in player.get("positions") or []:
                        name = position.get("position")
                        if name:
                            row.positions[name] = row.positions.get(name, 0) + 1

    build.missing_lineups = missing  # type: ignore[attr-defined]
    build.matches_read = seen_matches  # type: ignore[attr-defined]
    return list(rows.values())


def multi_competition(
    rows: list[PlayerSeason], min_minutes: float
) -> dict[int, list[PlayerSeason]]:
    """Players clearing `min_minutes` in two or more distinct competition-seasons."""
    by_player: dict[int, list[PlayerSeason]] = defaultdict(list)
    for row in rows:
        if row.minutes >= min_minutes:
            by_player[row.player_id].append(row)
    return {pid: seasons for pid, seasons in by_player.items() if len(seasons) >= 2}
