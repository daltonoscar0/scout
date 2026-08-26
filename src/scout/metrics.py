"""Per-player output metrics from canonical events.

These are the numbers the rest of scout is measured against: the projection
targets (xG+xA per 90, progressive actions per 90), the scout card's percentile
radar, and the valuation model's features.

The embedding describes *how* a player plays. These describe *how much* they
produced. Keeping them separate matters -- a similarity model that quietly
optimised for output would just rank players by minutes.

Definitions are stated rather than assumed, because "progressive" in particular
has no single agreed meaning and every provider's number differs:

  * **xG** is StatsBomb's own `statsbomb_xg`, taken as given.
  * **xA** is the xG of shots assisted by this player's pass, joined through the
    shot's `key_pass_id` to the pass's event id. This is expected *assists*, not
    expected assisted goals -- a pass that sets up a bad shot earns little.
  * **Progressive** is scout's own definition, documented below. It is not
    FBref's or Opta's and the numbers will not match theirs.

All coordinates arrive already normalised by sportio to a 105x68 pitch with the
team in possession attacking +x, so the opponent's goal is always at (105, 34).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import pandas as pd

GOAL_X = 105.0
GOAL_Y = 34.0

# An action is progressive if it ends meaningfully closer to the opponent's goal.
# Both a relative and an absolute gate: the relative one stops a 5m pass on the
# edge of the box counting as progression, the absolute one stops a long
# sideways ball from a team's own corner flag counting because the percentages
# are large when you start far away.
PROGRESSIVE_RATIO = 0.25
PROGRESSIVE_MIN_METRES = 5.0

# The penalty area on a 105x68 pitch: 16.5m deep, 40.32m wide.
BOX_X = GOAL_X - 16.5
BOX_Y_MIN = GOAL_Y - 20.16
BOX_Y_MAX = GOAL_Y + 20.16


def distance_to_goal(x: float, y: float) -> float:
    return math.hypot(GOAL_X - x, GOAL_Y - y)


def in_box(x: float, y: float) -> bool:
    return x >= BOX_X and BOX_Y_MIN <= y <= BOX_Y_MAX


def is_progressive(x: float, y: float, end_x: float, end_y: float) -> bool:
    """Did this action move the ball meaningfully toward the opponent's goal?

    Any completed action ending in the penalty area counts, because the last 16
    metres are where the relative test becomes useless.
    """
    if any(v is None or v != v for v in (x, y, end_x, end_y)):
        return False
    if in_box(end_x, end_y) and not in_box(x, y):
        return True
    before = distance_to_goal(x, y)
    after = distance_to_goal(end_x, end_y)
    gain = before - after
    return gain >= PROGRESSIVE_MIN_METRES and gain >= PROGRESSIVE_RATIO * before


@dataclass
class PlayerTotals:
    player_id: int
    touches: int = 0
    passes: int = 0
    passes_completed: int = 0
    progressive_passes: int = 0
    carries: int = 0
    progressive_carries: int = 0
    shots: int = 0
    goals: int = 0
    xg: float = 0.0
    xa: float = 0.0
    key_passes: int = 0
    pressures: int = 0
    duels_won: int = 0
    interceptions: int = 0
    ball_recoveries: int = 0
    dribbles_completed: int = 0
    dispossessed: int = 0
    matches: set = field(default_factory=set)

    def merge(self, other: PlayerTotals) -> None:
        for key, value in asdict(other).items():
            if key in ("player_id", "matches"):
                continue
            setattr(self, key, getattr(self, key) + value)
        self.matches |= other.matches


# Columns exposed per 90 minutes. Kept explicit so the app and the export cannot
# drift from what is actually computed.
PER_90_COLUMNS = (
    "xg",
    "xa",
    "xg_xa",
    "shots",
    "progressive_passes",
    "progressive_carries",
    "progressive_actions",
    "passes",
    "key_passes",
    "pressures",
    "duels_won",
    "interceptions",
    "ball_recoveries",
    "dribbles_completed",
    "touches",
)


def _extra(value: object) -> dict:
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _text(value: object) -> str | None:
    """Coerce a possibly-NA pandas string to str or None.

    sportio's canonical columns use pandas' nullable dtypes, so `outcome` can be
    pd.NA. Comparing pd.NA to a string returns pd.NA rather than False, and
    branching on that raises "boolean value of NA is ambiguous". Every read of a
    nullable text column goes through here.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, str):
        return value or None
    return None


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def accumulate(events: pd.DataFrame, totals: dict[int, PlayerTotals]) -> None:
    """Fold one match's canonical events into per-player running totals."""
    if events.empty:
        return

    extras = events["extra"].map(_extra)

    # Pass event id -> passer, so a shot's key_pass_id can be credited as xA.
    # The link is by event id rather than by proximity, so it is exact.
    passer_of: dict[str, int] = {}
    for extra, row in zip(extras, events.itertuples(index=False), strict=True):
        if _text(getattr(row, "type", None)) == "pass":
            event_id = extra.get("id")
            player_id = _number(getattr(row, "player_id", None))
            if event_id and player_id is not None:
                passer_of[str(event_id)] = int(player_id)

    def totals_for(player_id: int) -> PlayerTotals:
        if player_id not in totals:
            totals[player_id] = PlayerTotals(player_id=player_id)
        return totals[player_id]

    for extra, row in zip(extras, events.itertuples(index=False), strict=True):
        player_id = _number(getattr(row, "player_id", None))
        event_type = _text(getattr(row, "type", None))
        if player_id is None or event_type is None:
            continue
        player_id = int(player_id)
        record = totals_for(player_id)
        record.matches.add(int(getattr(row, "match_id", 0)))

        outcome = _text(getattr(row, "outcome", None))
        x, y = _number(getattr(row, "x", None)), _number(getattr(row, "y", None))
        end_x = _number(getattr(row, "end_x", None))
        end_y = _number(getattr(row, "end_y", None))

        if event_type in ("pass", "carry", "shot", "dribble", "ball_receipt"):
            record.touches += 1

        if event_type == "pass":
            record.passes += 1
            if outcome == "complete":
                record.passes_completed += 1
                if is_progressive(x, y, end_x, end_y):
                    record.progressive_passes += 1
        elif event_type == "carry":
            record.carries += 1
            if is_progressive(x, y, end_x, end_y):
                record.progressive_carries += 1
        elif event_type == "shot":
            record.shots += 1
            xg = _number(extra.get("statsbomb_xg")) or 0.0
            record.xg += xg
            if outcome == "goal":
                record.goals += 1
            key_pass = extra.get("key_pass_id")
            if key_pass:
                assister = passer_of.get(str(key_pass))
                if assister is not None:
                    helper = totals_for(assister)
                    helper.xa += xg
                    helper.key_passes += 1
        elif event_type == "pressure":
            record.pressures += 1
        elif event_type == "duel":
            if outcome == "success":
                record.duels_won += 1
        elif event_type == "interception":
            record.interceptions += 1
        elif event_type == "ball_recovery":
            record.ball_recoveries += 1
        elif event_type == "dribble":
            if outcome == "success":
                record.dribbles_completed += 1
        elif event_type == "dispossessed":
            record.dispossessed += 1


def to_frame(totals: dict[int, PlayerTotals], minutes: dict[int, float]) -> pd.DataFrame:
    """Per-90 rates, given each player's minutes for the same competition-season.

    Players with no recorded minutes are dropped rather than divided by zero:
    they are almost always a data gap, and a per-90 built on a fraction of a
    minute is a meaningless outlier that would dominate any percentile.
    """
    rows = []
    for player_id, record in sorted(totals.items()):
        played = minutes.get(player_id, 0.0)
        if played <= 0:
            continue
        scale = 90.0 / played
        row = {
            "player_id": player_id,
            "minutes": played,
            "matches": len(record.matches),
            "goals": record.goals,
            "pass_completion": (
                record.passes_completed / record.passes if record.passes else float("nan")
            ),
        }
        raw = {
            "xg": record.xg,
            "xa": record.xa,
            "xg_xa": record.xg + record.xa,
            "shots": record.shots,
            "progressive_passes": record.progressive_passes,
            "progressive_carries": record.progressive_carries,
            "progressive_actions": record.progressive_passes + record.progressive_carries,
            "passes": record.passes,
            "key_passes": record.key_passes,
            "pressures": record.pressures,
            "duels_won": record.duels_won,
            "interceptions": record.interceptions,
            "ball_recoveries": record.ball_recoveries,
            "dribbles_completed": record.dribbles_completed,
            "touches": record.touches,
        }
        for name, value in raw.items():
            row[f"{name}_per90"] = value * scale
            row[name] = value
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["player_id", "minutes", "matches"])
    return pd.DataFrame(rows)


def empty_totals() -> dict[int, PlayerTotals]:
    return {}
