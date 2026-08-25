"""Tokenizer tests.

The failure mode that matters here is not a crash -- it is a token stream that
looks fine and quietly drops a dimension. `under_pressure` was pinned to false
for every event by an `is True` identity check against a numpy bool, and nothing
failed; the only symptom was a suspiciously low maximum flag index. Several
tests below exist specifically to catch that shape of bug.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scout import tokenizer, vocab


def events(rows: list[dict]) -> pd.DataFrame:
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


def body(corpus: tokenizer.Corpus) -> np.ndarray:
    """Real tokens only, excluding PAD/BOS/EOS positions."""
    return corpus.tokens[corpus.tokens[:, :, 0] > 2]


# --- structure ------------------------------------------------------------


def test_an_empty_frame_yields_an_empty_corpus():
    corpus = tokenizer.tokenize_events(events([]).iloc[0:0])
    assert len(corpus) == 0
    assert corpus.tokens.shape[1:] == (tokenizer.SEQ_LEN, 5)


def test_sequences_are_bracketed_by_bos_and_eos():
    corpus = tokenizer.tokenize_events(events([{"t": float(i)} for i in range(5)]))
    assert corpus.tokens[0, 0, 0] == vocab.BOS_ID
    assert corpus.tokens[0, 6, 0] == vocab.EOS_ID
    assert corpus.lengths[0] == 7


def test_padding_is_zero_so_the_loss_mask_is_just_nonzero():
    corpus = tokenizer.tokenize_events(events([{"t": float(i)} for i in range(5)]))
    assert (corpus.tokens[0, 7:] == 0).all()
    assert (corpus.actors[0, 7:] == -1).all()


def test_a_long_stream_is_chunked_without_dropping_events():
    n = tokenizer.SEQ_LEN * 2
    corpus = tokenizer.tokenize_events(events([{"t": float(i)} for i in range(n)]))
    assert len(corpus) > 1
    assert corpus.coverage.tokenized == n
    # Every real event appears exactly once across the chunks.
    assert int((corpus.tokens[:, :, 0] > 2).sum()) == n


def test_each_team_gets_its_own_sequence():
    rows = [{"team_id": 10, "t": 1.0}, {"team_id": 20, "t": 2.0}]
    corpus = tokenizer.tokenize_events(events(rows))
    assert len(corpus) == 2
    assert set(corpus.team_ids.tolist()) == {10, 20}


def test_actors_align_one_to_one_with_token_positions():
    rows = [{"player_id": 7, "t": 0.0}, {"player_id": 9, "t": 1.0}]
    corpus = tokenizer.tokenize_events(events(rows))
    assert corpus.actors[0, 1] == 7
    assert corpus.actors[0, 2] == 9
    # BOS and EOS have no actor.
    assert corpus.actors[0, 0] == -1
    assert corpus.actors[0, 3] == -1


def test_a_missing_player_id_becomes_minus_one_not_zero():
    # Zero is a plausible player id; conflating "no actor" with a real id would
    # pool an event into the wrong player's embedding.
    corpus = tokenizer.tokenize_events(events([{"player_id": None}]))
    assert corpus.actors[0, 1] == -1
    assert corpus.coverage.missing_actor == 1


# --- the flags bug --------------------------------------------------------


@pytest.mark.parametrize("truthy", [True, np.True_, pd.array([True], dtype="boolean")[0]])
def test_pressure_is_read_from_every_boolean_flavour_pandas_produces(truthy):
    # np.True_ is not the Python singleton True. An identity check here silently
    # pinned under_pressure to false for the entire corpus.
    corpus = tokenizer.tokenize_events(events([{"pressure": truthy}]))
    flags = int(body(corpus)[0, 3])
    assert flags >= 10, "under_pressure did not survive tokenization"


def test_pressure_and_counterpress_are_independent_dimensions():
    plain = events([{"pressure": False}])
    pressed = events([{"pressure": True}])
    counter = events([{"pressure": False, "extra": json.dumps({"counterpress": True})}])
    both = events([{"pressure": True, "extra": json.dumps({"counterpress": True})}])
    indices = {
        int(body(tokenizer.tokenize_events(frame))[0, 3])
        for frame in (plain, pressed, counter, both)
    }
    assert len(indices) == 4


def test_a_missing_pressure_value_is_false_rather_than_an_exception():
    corpus = tokenizer.tokenize_events(events([{"pressure": pd.NA}]))
    assert int(body(corpus)[0, 3]) < 10


# --- field mapping --------------------------------------------------------


def test_body_parts_outside_the_modelled_set_collapse_to_other():
    keeper = tokenizer.tokenize_events(events([{"body_part": "keeper_arm"}]))
    chest = tokenizer.tokenize_events(events([{"body_part": "chest"}]))
    assert int(body(keeper)[0, 3]) == int(body(chest)[0, 3])


def test_an_unmodelled_event_type_is_dropped_and_counted():
    corpus = tokenizer.tokenize_events(events([{"type": "half_time_oranges"}]))
    assert corpus.coverage.tokenized == 0
    assert corpus.coverage.unmodelled_types["half_time_oranges"] == 1


def test_an_unmapped_outcome_is_counted_but_still_tokenized():
    # Losing a rare outcome is acceptable; losing the event is not.
    corpus = tokenizer.tokenize_events(events([{"outcome": "some_1958_outcome"}]))
    assert corpus.coverage.tokenized == 1
    assert corpus.coverage.unmapped_outcomes["some_1958_outcome"] == 1


def test_a_missing_location_is_counted_and_lands_in_the_null_zone():
    corpus = tokenizer.tokenize_events(events([{"x": None, "y": None}]))
    assert corpus.coverage.missing_location == 1
    assert int(body(corpus)[0, 1]) == vocab.ZONE_NULL


def test_an_event_with_no_end_point_has_no_delta():
    corpus = tokenizer.tokenize_events(events([{"end_x": None, "end_y": None}]))
    assert int(body(corpus)[0, 2]) == vocab.DELTA_NONE


# --- time -----------------------------------------------------------------


def test_the_first_event_of_a_period_is_marked_as_a_period_start():
    corpus = tokenizer.tokenize_events(events([{"t": 0.0}]))
    assert int(body(corpus)[0, 4]) == vocab.DT_PERIOD_START


def test_a_new_period_restarts_the_clock_rather_than_showing_a_huge_gap():
    rows = [{"t": 2700.0, "period": 1}, {"t": 0.0, "period": 2}]
    corpus = tokenizer.tokenize_events(events(rows))
    assert int(body(corpus)[1, 4]) == vocab.DT_PERIOD_START


def test_gaps_within_a_period_are_bucketed_by_size():
    rows = [{"t": 0.0}, {"t": 0.2}, {"t": 40.0}]
    corpus = tokenizer.tokenize_events(events(rows))
    quick = int(body(corpus)[1, 4])
    slow = int(body(corpus)[2, 4])
    assert quick < slow


# --- provenance -----------------------------------------------------------


def test_the_corpus_records_the_vocabulary_it_was_built_with():
    corpus = tokenizer.tokenize_events(events([{}]))
    assert corpus.vocab_fingerprint == vocab.vocabulary_fingerprint()


def test_coverage_merges_additively():
    a = tokenizer.tokenize_events(events([{}])).coverage
    b = tokenizer.tokenize_events(events([{"type": "nonsense"}])).coverage
    a.merge(b)
    assert a.events == 2
    assert a.tokenized == 1
    assert a.unmodelled_types["nonsense"] == 1
