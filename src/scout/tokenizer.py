"""Turn canonical events into factorized token sequences.

One sequence is a single team's event stream within one match, ordered by
possession and then by time, chunked into fixed windows. Chunking across
possessions rather than emitting one sequence per possession is deliberate:
possessions average well under ten events, which is far too short to give a
transformer any context to work with, and the possession boundary is preserved
in the data anyway because `dt` jumps and the action pattern changes.

Alongside the tokens the tokenizer emits an `actor` array of player ids aligned
1:1 with token positions. It is never fed to the model -- that would let the
model memorise identity instead of learning style. It is the key used later to
pool hidden states into a player embedding.

Coverage is reported, never silently absorbed. Every event type the vocabulary
does not model and every outcome string outside OUTCOME_MAP is counted and
returned, so an old competition with a different vocabulary shows up as a number
rather than as a quietly degraded corpus.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scout import vocab

SEQ_LEN = 256

# Column order in the emitted token array.
FIELDS = ("action", "start_zone", "end_delta", "flags", "dt_bucket")


@dataclass
class Coverage:
    """What the tokenizer could not map. The point is to make this visible."""

    events: int = 0
    tokenized: int = 0
    unmodelled_types: Counter = field(default_factory=Counter)
    unmapped_outcomes: Counter = field(default_factory=Counter)
    missing_location: int = 0
    missing_actor: int = 0

    @property
    def dropped(self) -> int:
        return self.events - self.tokenized

    def merge(self, other: Coverage) -> None:
        self.events += other.events
        self.tokenized += other.tokenized
        self.unmodelled_types.update(other.unmodelled_types)
        self.unmapped_outcomes.update(other.unmapped_outcomes)
        self.missing_location += other.missing_location
        self.missing_actor += other.missing_actor

    def report(self) -> str:
        share = 100.0 * self.tokenized / self.events if self.events else 0.0
        lines = [
            f"events {self.events:,}  tokenized {self.tokenized:,} ({share:.2f}%)  "
            f"dropped {self.dropped:,}",
            f"  no location: {self.missing_location:,}   no actor: {self.missing_actor:,}",
        ]
        if self.unmodelled_types:
            lines.append("  unmodelled event types:")
            for name, count in self.unmodelled_types.most_common(15):
                lines.append(f"    {name:<28} {count:,}")
        if self.unmapped_outcomes:
            lines.append("  outcomes outside OUTCOME_MAP (treated as 'na'):")
            for name, count in self.unmapped_outcomes.most_common(15):
                lines.append(f"    {name:<28} {count:,}")
        return "\n".join(lines)


@dataclass
class Corpus:
    """Tokenized sequences plus the actor alignment and provenance."""

    tokens: np.ndarray  # (n_sequences, SEQ_LEN, 5) int16
    actors: np.ndarray  # (n_sequences, SEQ_LEN) int64, -1 where no actor
    lengths: np.ndarray  # (n_sequences,) int32, real tokens before padding
    match_ids: np.ndarray  # (n_sequences,) int64
    team_ids: np.ndarray  # (n_sequences,) int64
    coverage: Coverage
    vocab_fingerprint: str

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


def _counterpress(extra: object) -> bool:
    """counterpress lives inside the JSON `extra` blob, not in a column.

    All StatsBomb sparse booleans are present-or-absent and never literally
    false, so presence of the key is the value.
    """
    if not isinstance(extra, str) or "counterpress" not in extra:
        return False
    try:
        return bool(json.loads(extra).get("counterpress", False))
    except (ValueError, TypeError):
        return False


def _possession(extra: object) -> int:
    if not isinstance(extra, str):
        return 0
    try:
        return int(json.loads(extra).get("possession", 0) or 0)
    except (ValueError, TypeError):
        return 0


def _flag(value: object) -> bool:
    """Read a nullable boolean without tripping over pandas' three-valued logic.

    `pressure` arrives as pandas' nullable BooleanDtype, so itertuples yields
    numpy.bool_ rather than Python bool -- and `np.True_ is True` is False. An
    identity check here silently pinned under_pressure to false for every event,
    which cost half the flags table and showed up only as a suspiciously low
    maximum index. pd.NA is falsy-by-exception, so it is caught explicitly.
    """
    if value is None or value is pd.NA:
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _body_part_class(value: object) -> str:
    """Collapse sportio's 11 observed body parts onto the five modelled classes."""
    if not isinstance(value, str) or not value:
        return "none"
    if value in ("right_foot", "left_foot", "head"):
        return value
    return "other"


def tokenize_events(events: pd.DataFrame) -> Corpus:
    """Tokenize a canonical events frame into fixed-length sequences."""
    coverage = Coverage(events=len(events))
    if events.empty:
        return Corpus(
            tokens=np.zeros((0, SEQ_LEN, 5), dtype=np.int16),
            actors=np.zeros((0, SEQ_LEN), dtype=np.int64),
            lengths=np.zeros((0,), dtype=np.int32),
            match_ids=np.zeros((0,), dtype=np.int64),
            team_ids=np.zeros((0,), dtype=np.int64),
            coverage=coverage,
            vocab_fingerprint=vocab.vocabulary_fingerprint(),
        )

    frame = events.copy()
    frame["possession_no"] = frame["extra"].map(_possession)
    frame["counterpress_flag"] = frame["extra"].map(_counterpress)

    # Sort within each team's stream by possession then clock. Possession number
    # is monotonic within a match, so this keeps a team's touches in the order
    # they happened while grouping each spell of control together.
    frame = frame.sort_values(
        ["match_id", "team_id", "period", "possession_no", "t"], kind="stable"
    )

    sequences: list[np.ndarray] = []
    actor_rows: list[np.ndarray] = []
    lengths: list[int] = []
    match_ids: list[int] = []
    team_ids: list[int] = []

    for (match_id, team_id), group in frame.groupby(["match_id", "team_id"], sort=False):
        tokens, actors = _tokenize_stream(group, coverage)
        if not tokens:
            continue
        for start in range(0, len(tokens), SEQ_LEN - 2):
            window = tokens[start : start + SEQ_LEN - 2]
            window_actors = actors[start : start + SEQ_LEN - 2]
            if not window:
                continue

            padded = np.zeros((SEQ_LEN, 5), dtype=np.int16)
            padded_actors = np.full(SEQ_LEN, -1, dtype=np.int64)

            # BOS and EOS occupy the action field; the other four fields stay at
            # their padding value, which is index 0 in each table.
            padded[0, 0] = vocab.BOS_ID
            body = np.asarray(window, dtype=np.int16)
            padded[1 : 1 + len(body)] = body
            padded_actors[1 : 1 + len(body)] = np.asarray(window_actors, dtype=np.int64)
            padded[1 + len(body), 0] = vocab.EOS_ID

            sequences.append(padded)
            actor_rows.append(padded_actors)
            lengths.append(len(body) + 2)
            match_ids.append(int(match_id))
            team_ids.append(int(team_id))

    if not sequences:
        stacked = np.zeros((0, SEQ_LEN, 5), dtype=np.int16)
        stacked_actors = np.zeros((0, SEQ_LEN), dtype=np.int64)
    else:
        stacked = np.stack(sequences)
        stacked_actors = np.stack(actor_rows)

    return Corpus(
        tokens=stacked,
        actors=stacked_actors,
        lengths=np.asarray(lengths, dtype=np.int32),
        match_ids=np.asarray(match_ids, dtype=np.int64),
        team_ids=np.asarray(team_ids, dtype=np.int64),
        coverage=coverage,
        vocab_fingerprint=vocab.vocabulary_fingerprint(),
    )


def _tokenize_stream(group: pd.DataFrame, coverage: Coverage) -> tuple[list[list[int]], list[int]]:
    tokens: list[list[int]] = []
    actors: list[int] = []

    previous_t: float | None = None
    previous_period: int | None = None

    for row in group.itertuples(index=False):
        event_type = getattr(row, "type", None)
        if not isinstance(event_type, str) or not event_type:
            coverage.unmodelled_types["<missing>"] += 1
            continue

        outcome = getattr(row, "outcome", None)
        outcome = outcome if isinstance(outcome, str) and outcome else None
        if outcome is not None and outcome not in vocab.OUTCOME_MAP:
            coverage.unmapped_outcomes[outcome] += 1

        action = vocab.action_index(event_type, outcome)
        if action is None:
            coverage.unmodelled_types[event_type] += 1
            continue

        x, y = _coord(row, "x"), _coord(row, "y")
        if x is None or y is None:
            coverage.missing_location += 1
        zone = vocab.zone_index(x, y)

        end_x, end_y = _coord(row, "end_x"), _coord(row, "end_y")
        if x is None or y is None or end_x is None or end_y is None:
            delta = vocab.DELTA_NONE
        else:
            delta = vocab.delta_index(end_x - x, end_y - y)

        flags = vocab.flags_index(
            _flag(getattr(row, "pressure", None)),
            _flag(getattr(row, "counterpress_flag", None)),
            _body_part_class(getattr(row, "body_part", None)),
        )

        period = _int(getattr(row, "period", None))
        t = _coord(row, "t")
        if previous_period is None or period != previous_period:
            dt_bucket = vocab.dt_index(None, period_start=True)
        elif t is None or previous_t is None:
            dt_bucket = vocab.dt_index(None)
        else:
            dt_bucket = vocab.dt_index(t - previous_t)
        previous_period = period
        if t is not None:
            previous_t = t

        player_id = _int(getattr(row, "player_id", None))
        if player_id is None:
            coverage.missing_actor += 1

        tokens.append([action, zone, delta, flags, dt_bucket])
        actors.append(player_id if player_id is not None else -1)
        coverage.tokenized += 1

    return tokens, actors


def _coord(row: object, name: str) -> float | None:
    value = getattr(row, name, None)
    if value is None or value is pd.NA:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _int(value: object) -> int | None:
    if value is None or value is pd.NA:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else int(number)
