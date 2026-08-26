"""Training-data tests.

Two failure modes here are silent and would only show up as a model that is
quietly worse than it should be: a corpus cap that truncates instead of
sampling, and a held-out split that leaks.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="train tests need the `train` extra")

from scout import tokenizer, train, vocab  # noqa: E402


def write_shard(directory, competition_id, season_id, *, n=40, first_match=1000, length=100):
    path = directory / f"{competition_id}_{season_id}.npz"
    np.savez_compressed(
        path,
        tokens=np.ones((n, tokenizer.SEQ_LEN, 5), dtype=np.int16),
        actors=np.full((n, tokenizer.SEQ_LEN), 7, dtype=np.int64),
        lengths=np.full(n, length, dtype=np.int32),
        match_ids=np.arange(first_match, first_match + n, dtype=np.int64),
        team_ids=np.full(n, 1, dtype=np.int64),
        meta=np.array(
            json.dumps(
                {
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "competition": f"comp{competition_id}",
                    "season": str(season_id),
                    "vocab_fingerprint": vocab.vocabulary_fingerprint(),
                }
            )
        ),
    )
    return path


@pytest.fixture
def corpus_dir(tmp_path):
    # Three competitions with disjoint match-id ranges, so the origin of any
    # sampled sequence is identifiable.
    for index, (competition_id, first) in enumerate([(11, 1000), (43, 2000), (55, 3000)]):
        write_shard(tmp_path, competition_id, index, first_match=first)
    return tmp_path


# --- the corpus cap -------------------------------------------------------


def test_the_whole_corpus_loads_when_there_is_no_cap(corpus_dir):
    data = train.load_corpus(corpus_dir)
    assert len(data) == 120
    assert int(data.lengths.sum()) == 12_000


def test_a_cap_samples_across_every_competition_rather_than_truncating(corpus_dir):
    # Shards are named by competition id, so reading them in order until the
    # budget fills would train only on whichever competitions sort first. The
    # corpus spans men's and women's football, 1958 to 2025 -- a truncation of it
    # is not a sample of it.
    data = train.load_corpus(corpus_dir, max_events=2_000)
    origins = {int(m) // 1000 for m in data.match_ids}
    assert origins == {1, 2, 3}


def test_a_cap_respects_its_token_budget(corpus_dir):
    data = train.load_corpus(corpus_dir, max_events=2_000)
    # One sequence of slack: the cap keeps whole sequences, so it lands on or
    # just past the budget rather than under it.
    assert 2_000 <= int(data.lengths.sum()) <= 2_100


def test_a_cap_larger_than_the_corpus_keeps_everything(corpus_dir):
    assert len(train.load_corpus(corpus_dir, max_events=10**9)) == 120


def test_the_cap_is_deterministic(corpus_dir):
    a = train.load_corpus(corpus_dir, max_events=2_000)
    b = train.load_corpus(corpus_dir, max_events=2_000)
    assert list(a.match_ids) == list(b.match_ids)


def test_an_empty_corpus_directory_says_what_to_run(tmp_path):
    with pytest.raises(SystemExit, match="scout build"):
        train.load_corpus(tmp_path)


def test_a_shard_from_another_vocabulary_is_refused(tmp_path):
    path = write_shard(tmp_path, 11, 0)
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    payload["meta"] = np.array(json.dumps({"vocab_fingerprint": "stale0000"}))
    np.savez_compressed(path, **payload)
    from scout.corpus import VocabularyMismatch

    with pytest.raises(VocabularyMismatch):
        train.load_corpus(tmp_path)


# --- the held-out split ---------------------------------------------------


def test_the_split_holds_out_whole_matches(corpus_dir):
    data = train.load_corpus(corpus_dir)
    fit, held = train.split_by_match(data, 0.2, seed=0)
    # A match that appears in training must not appear in the held-out set.
    # Sequences from one match share possessions, teams and opponents, so a row
    # split would report a perplexity flattered by leakage.
    assert not (set(fit.match_ids.tolist()) & set(held.match_ids.tolist()))


def test_the_split_keeps_every_sequence(corpus_dir):
    data = train.load_corpus(corpus_dir)
    fit, held = train.split_by_match(data, 0.2, seed=0)
    assert len(fit) + len(held) == len(data)


def test_the_split_holds_out_something(corpus_dir):
    _, held = train.split_by_match(train.load_corpus(corpus_dir), 0.2, seed=0)
    assert len(held) > 0


# --- the mask -------------------------------------------------------------


def test_the_mask_marks_exactly_the_real_tokens():
    lengths = np.array([3, 5], dtype=np.int32)
    mask = train.mask_for(lengths, 8)
    assert mask.shape == (2, 8)
    assert mask[0].sum() == 3
    assert mask[1].sum() == 5
    assert mask[0, 3] == 0


# --- the baseline ---------------------------------------------------------


def test_the_bigram_baseline_returns_a_finite_perplexity(corpus_dir):
    data = train.load_corpus(corpus_dir)
    fit, held = train.split_by_match(data, 0.2, seed=0)
    value = train.bigram_perplexity(fit, held)
    assert np.isfinite(value)
    assert value > 0
