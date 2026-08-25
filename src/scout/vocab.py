"""The factorized token vocabulary.

A token in the spec is (action type, zone, body part, pressure flag, outcome).
Fusing those into one symbol space is not viable: the cross product is roughly
10^8 symbols, and a softmax over it would be tens of billions of parameters.

So the token is factorized into five fields with five small embedding tables,
summed before the first transformer block. At d_model=256 the entire embedding
stack is about 71k parameters, which is the point.

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

VOCAB_VERSION: Final = "1"

# --------------------------------------------------------------------------
# F1: action = event type x outcome class
# --------------------------------------------------------------------------

# Outcome classes. StatsBomb has no top-level `outcome`; it lives in type-
# specific sub-objects (pass.outcome, duel.outcome, shot.outcome, ...) and is
# *absent* when a pass completes. The loader maps those into this small closed
# set, so the LM sees one consistent notion of "how did it go".
OUTCOME_CLASSES: Final = ("na", "success", "fail", "out", "goal", "blocked", "saved")

# Which outcome classes are actually reachable per event type. Taking the full
# cross product would waste ~2/3 of the table on combinations that never occur
# (a Pressure event cannot be a goal), and dead embedding rows never receive
# gradient. Each entry is (statsbomb_type, reachable_outcome_classes).
#
# NOT YET CALIBRATED AGAINST REAL DATA. This table is derived a priori from the
# StatsBomb spec, and it is deliberately coarser than the raw outcome
# vocabulary: StatsBomb distinguishes e.g. Duel outcomes "Lost In Play" /
# "Lost Out" / "Success In Play" / "Success Out", which collapse to fail/success
# here. That collapse may be discarding signal a similarity model wants. The
# tokenizer therefore *counts* every (type, outcome) pair it cannot map and
# reports them rather than silently bucketing, so a single pass over one
# competition tells us whether this table needs widening. Do that before
# training -- retraining after a vocabulary change is a full redo.
ACTION_TYPES: Final = (
    ("Pass", ("success", "fail", "out")),
    ("Ball Receipt*", ("success", "fail")),
    ("Carry", ("success",)),
    ("Pressure", ("na",)),
    ("Duel", ("success", "fail")),
    ("Dribble", ("success", "fail")),
    ("Shot", ("goal", "saved", "blocked", "out", "fail")),
    ("Clearance", ("na",)),
    ("Interception", ("success", "fail")),
    ("Ball Recovery", ("success", "fail")),
    ("Block", ("na",)),
    ("Foul Committed", ("na",)),
    ("Foul Won", ("na",)),
    ("Goal Keeper", ("success", "fail", "saved")),
    ("Miscontrol", ("na",)),
    ("Dispossessed", ("na",)),
    ("Dribbled Past", ("na",)),
    ("Shield", ("na",)),
    ("50/50", ("success", "fail")),
    ("Error", ("na",)),
    ("Offside", ("na",)),
    ("Own Goal Against", ("na",)),
    ("Own Goal For", ("na",)),
    ("Bad Behaviour", ("na",)),
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
    "referee_ball_drop",
)

# PAD must be index 0 so that a zero-filled array is a valid padded sequence
# and the loss mask is simply `tokens != 0`.
SPECIAL_TOKENS: Final = ("PAD", "BOS", "EOS")


def _build_actions() -> tuple[str, ...]:
    names: list[str] = list(SPECIAL_TOKENS)
    names.extend(CONTROL_TOKENS)
    for event_type, outcomes in ACTION_TYPES:
        for outcome in outcomes:
            names.append(f"{event_type}|{outcome}")
    return tuple(names)


ACTIONS: Final = _build_actions()

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
