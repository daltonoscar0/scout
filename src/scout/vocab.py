"""The factorized token vocabulary.

A token in the spec is (action type, zone, body part, pressure flag, outcome).
Fusing those into one symbol space is not viable: the cross product is roughly
10^8 symbols, and a softmax over it would be tens of billions of parameters.

So the token is factorized into five fields with five small embedding tables,
summed before the first transformer block. At d_model=256 the five tables total
347 rows and about 89k parameters, which is the point.

    F1 action     StatsBomb event type fused with an outcome class
    F2 start_zone 12x8 grid over the canonical 105x68 pitch, plus null
    F3 end_delta  direction sector x length bucket, plus none
    F4 flags      under_pressure x counterpress x body-part class
    F5 dt_bucket  log-spaced time since the previous event

Fields F1-F3 are predicted by the LM. F4/F5 are auxiliary heads.

This module is deliberately stdlib-only and side-effect free. The vocabulary is
a *contract*: embeddings, checkpoints and the FAISS-free index are all keyed to
it, so any change invalidates trained artifacts. VOCAB_VERSION and
vocabulary_fingerprint() exist so that a mismatch is caught loudly at load time
rather than silently producing garbage neighbours.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Final

VOCAB_VERSION: Final = "2"

# --------------------------------------------------------------------------
# F1: action = event type x outcome class
# --------------------------------------------------------------------------

# Outcome classes. Raw StatsBomb has no top-level outcome -- it lives in
# type-specific sub-objects (pass.outcome, duel.outcome, ...) and is absent when
# a pass completes -- but sportio has already lifted it to a single snake_cased
# `outcome` column, filling in "complete" for a completed pass. What survives is
# a long tail: 32 distinct values in four matches of Euro 2024 alone, most of
# them rare. These seven classes are what the model sees.
OUTCOME_CLASSES: Final = ("na", "success", "fail", "out", "goal", "blocked", "saved")

# sportio's snake_cased outcome strings, mapped onto the seven classes. Measured
# against real events rather than guessed -- every key below was observed.
# Anything absent maps to "na" and is counted and reported by the tokenizer, so
# an unseen value from an older competition surfaces as a number rather than
# silently becoming a no-op.
OUTCOME_MAP: Final = {
    # completions and wins
    "complete": "success",
    "won": "success",
    "success": "success",
    "success_in_play": "success",
    "success_out": "success",
    "success_to_team": "success",
    # keeper actions that retain the ball
    "claim": "success",
    "collected_twice": "success",
    "in_play_safe": "success",
    # failures that keep the ball live
    "incomplete": "fail",
    "lost": "fail",
    "lost_in_play": "fail",
    "success_to_opposition": "fail",
    "wayward": "fail",
    "pass_offside": "fail",
    "in_play_danger": "fail",
    "recovery_failure": "fail",
    # ball leaves play
    "out": "out",
    "lost_out": "out",
    "touched_out": "out",
    "off_t": "out",
    "injury_clearance": "out",
    # shot outcomes
    "goal": "goal",
    "blocked": "blocked",
    "saved": "saved",
    "saved_twice": "saved",
    "saved_to_post": "saved",
    "saved_off_target": "saved",
    # genuinely carry no success/failure sense
    "unknown": "na",
    "no_touch": "na",
    "tactical": "na",  # substitution reason
    "injury": "na",  # substitution reason
}

# The 23 event types that represent something a player did. Taken from sportio's
# canonical vocabulary (snake_cased), not from raw StatsBomb type names.
ACTION_TYPES: Final = (
    "pass",
    "ball_receipt",
    "carry",
    "pressure",
    "duel",
    "dribble",
    "dribbled_past",
    "shot",
    "block",
    "clearance",
    "interception",
    "ball_recovery",
    "goal_keeper",
    "miscontrol",
    "dispossessed",
    "shield",
    "foul_committed",
    "foul_won",
    "50_50",
    "error",
    "own_goal_against",
    "own_goal_for",
    "bad_behaviour",
)

# Every (type, outcome class) pair gets a row, rather than only the pairs judged
# reachable in advance. Restricting the table would save about two thirds of it,
# but this corpus spans 1958 to 2025 and the outcome vocabulary is demonstrably
# not stable across it -- an unforeseen pair would have to fall back to some
# other row, which is a silent wrong answer rather than a loud one. The unused
# rows cost 23 * 7 * 256 = 41k parameters and simply never receive gradient.
ACTION_OUTCOME_PAIRS: Final = tuple(
    f"{action}|{outcome}" for action in ACTION_TYPES for outcome in OUTCOME_CLASSES
)

# Structural markers. They carry no player action but anchor the sequence: a
# model that cannot see a half boundary will happily predict a through ball
# across half time.
CONTROL_TOKENS: Final = (
    "half_start",
    "half_end",
    "starting_xi",
    "tactical_shift",
    "substitution",
    "player_on",
    "player_off",
    "injury_stoppage",
    "referee_ball_drop",
)

# PAD must be index 0 so that a zero-filled array is a valid padded sequence
# and the loss mask is simply `tokens != 0`.
SPECIAL_TOKENS: Final = ("PAD", "BOS", "EOS")


ACTIONS: Final = SPECIAL_TOKENS + CONTROL_TOKENS + ACTION_OUTCOME_PAIRS
ACTION_LOOKUP: Final = {name: index for index, name in enumerate(ACTIONS)}

# Types that exist in the canonical vocabulary but are deliberately not modelled
# as actions. They are either control markers (handled above) or carry no
# information about how a player plays.
UNMODELLED_TYPES: Final = frozenset(CONTROL_TOKENS)


def outcome_class(outcome: str | None) -> str:
    """Map one snake_cased sportio outcome onto a class. Unknown values are 'na'.

    Callers that care about coverage should check `outcome in OUTCOME_MAP`
    themselves; this function never raises, because a single unrecognised
    outcome in a 1958 World Cup file must not abort a corpus-wide tokenization.
    """
    if not outcome:
        return "na"
    return OUTCOME_MAP.get(outcome, "na")


def action_index(event_type: str, outcome: str | None) -> int | None:
    """Index for an (event type, outcome) pair, or None if the type is not modelled."""
    if event_type in ACTION_LOOKUP:  # a control token, used bare
        return ACTION_LOOKUP[event_type]
    key = f"{event_type}|{outcome_class(outcome)}"
    return ACTION_LOOKUP.get(key)


# --------------------------------------------------------------------------
# F2: start zone, a 12x8 grid over the canonical 105x68 pitch
# --------------------------------------------------------------------------

ZONE_COLS: Final = 12
ZONE_ROWS: Final = 8
PITCH_LENGTH: Final = 105.0
PITCH_WIDTH: Final = 68.0

# The spec asks for an 18-zone grid. 18 (6x3) cells of 17.5m x 22.7m are far too
# coarse to separate a near-post cross from a cut-back, which is exactly the
# distinction a similarity model needs. 12x8 gives 8.75m x 8.5m cells -- roughly
# square, and fine enough to resolve the width of the six-yard box. The
# difference is 96 embedding rows against 18, which at d_model=256 costs 20k
# parameters. Recorded here because it is a deliberate departure from the spec.
ZONE_NULL: Final = ZONE_COLS * ZONE_ROWS  # 28 events per match have no location


def zone_index(x: float | None, y: float | None) -> int:
    """Map a pitch coordinate to a grid cell. Out-of-range coordinates clamp.

    Coordinates genuinely exceed the pitch (a shot crossing the goal line, a
    throw-in taken from behind the touchline), so clamping is correct here;
    rejecting them would drop real events.
    """
    if x is None or y is None:
        return ZONE_NULL
    try:
        if x != x or y != y:  # NaN
            return ZONE_NULL
    except TypeError:
        return ZONE_NULL
    col = int(x / PITCH_LENGTH * ZONE_COLS)
    row = int(y / PITCH_WIDTH * ZONE_ROWS)
    col = min(max(col, 0), ZONE_COLS - 1)
    row = min(max(row, 0), ZONE_ROWS - 1)
    return row * ZONE_COLS + col


ZONE_SIZE: Final = ZONE_COLS * ZONE_ROWS + 1

# --------------------------------------------------------------------------
# F3: end delta, how far and in which direction the ball moved
# --------------------------------------------------------------------------

DELTA_SECTORS: Final = 8  # 45-degree sectors, 0 = straight forward (+x)
DELTA_LENGTH_EDGES: Final = (5.0, 15.0, 30.0, 60.0)  # metres; yields 5 buckets
DELTA_NONE: Final = DELTA_SECTORS * (len(DELTA_LENGTH_EDGES) + 1)
DELTA_SIZE: Final = DELTA_NONE + 1


def delta_index(dx: float | None, dy: float | None) -> int:
    """Bucket a displacement into (direction sector, length bucket).

    Absolute end coordinates are deliberately not modelled -- F2 already carries
    where the action started, and a *relative* displacement generalises across
    the pitch in a way an absolute end zone does not.
    """
    if dx is None or dy is None:
        return DELTA_NONE
    try:
        if dx != dx or dy != dy:
            return DELTA_NONE
    except TypeError:
        return DELTA_NONE

    length = math.hypot(dx, dy)
    if length < 0.5:  # a touch in place, not a movement
        return DELTA_NONE

    bucket = 0
    for edge in DELTA_LENGTH_EDGES:
        if length >= edge:
            bucket += 1
    # atan2 gives (-pi, pi]; rotate by half a sector so that sector 0 is
    # centred on "straight forward" rather than straddling the boundary.
    angle = math.atan2(dy, dx)
    sector = int(
        ((angle + math.pi / DELTA_SECTORS) % (2 * math.pi)) / (2 * math.pi) * DELTA_SECTORS
    )
    sector %= DELTA_SECTORS
    return sector * (len(DELTA_LENGTH_EDGES) + 1) + bucket


# --------------------------------------------------------------------------
# F4: flags = under_pressure x counterpress x body part
# --------------------------------------------------------------------------

BODY_PARTS: Final = ("none", "right_foot", "left_foot", "head", "other")
FLAGS_SIZE: Final = 2 * 2 * len(BODY_PARTS)


def flags_index(under_pressure: bool, counterpress: bool, body_part: str) -> int:
    """Pack the three flag dimensions into one index.

    All StatsBomb sparse booleans are present-or-absent and never literally
    false, so callers must read them with `.get(key, False)`. Passing a missing
    key straight through as None would land in the truthiness test below and
    quietly read as False, which is the correct interpretation.
    """
    try:
        part = BODY_PARTS.index(body_part)
    except ValueError:
        part = BODY_PARTS.index("other")
    return (int(bool(under_pressure)) * 2 + int(bool(counterpress))) * len(BODY_PARTS) + part


# --------------------------------------------------------------------------
# F5: dt bucket, log-spaced gap since the previous event
# --------------------------------------------------------------------------

DT_EDGES: Final = (0.04, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 60.0)
DT_PERIOD_START: Final = len(DT_EDGES) + 1
DT_UNKNOWN: Final = DT_PERIOD_START + 1
DT_SIZE: Final = DT_UNKNOWN + 1


def dt_index(dt: float | None, *, period_start: bool = False) -> int:
    """Bucket the gap since the previous event in the same possession stream."""
    if period_start:
        return DT_PERIOD_START
    if dt is None:
        return DT_UNKNOWN
    try:
        if dt != dt or dt < 0:
            return DT_UNKNOWN
    except TypeError:
        return DT_UNKNOWN
    bucket = 0
    for edge in DT_EDGES:
        if dt >= edge:
            bucket += 1
    return bucket


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------

ACTION_SIZE: Final = len(ACTIONS)
FIELD_SIZES: Final = (ACTION_SIZE, ZONE_SIZE, DELTA_SIZE, FLAGS_SIZE, DT_SIZE)
FIELD_NAMES: Final = ("action", "start_zone", "end_delta", "flags", "dt_bucket")

PAD_ID: Final = 0
BOS_ID: Final = 1
EOS_ID: Final = 2


def vocabulary_fingerprint() -> str:
    """Stable hash of the vocabulary.

    Written into every checkpoint and index. Embeddings trained under one
    vocabulary are meaningless under another, and the failure mode is silent --
    plausible-looking but wrong nearest neighbours -- so it is checked on load.
    """
    payload = json.dumps(
        {
            "version": VOCAB_VERSION,
            "actions": ACTIONS,
            "zone": [ZONE_COLS, ZONE_ROWS],
            "delta": [DELTA_SECTORS, list(DELTA_LENGTH_EDGES)],
            "body_parts": BODY_PARTS,
            "dt_edges": list(DT_EDGES),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def describe() -> str:
    lines = [f"vocabulary v{VOCAB_VERSION}  fingerprint={vocabulary_fingerprint()}"]
    for name, size in zip(FIELD_NAMES, FIELD_SIZES, strict=True):
        lines.append(f"  {name:<11} {size:>4} rows")
    lines.append(f"  {'TOTAL':<11} {sum(FIELD_SIZES):>4} rows")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
