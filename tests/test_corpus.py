"""Shard round-trip and the vocabulary guard.

The guard is the important one. Nothing about a stale shard looks wrong: the
arrays load, the shapes are right, training runs. The damage only appears as
subtly wrong nearest neighbours, which is exactly the kind of failure nobody
notices. So a fingerprint mismatch is an error rather than a warning.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scout import corpus, tokenizer, vocab


def write_shard(path, fingerprint=None, n=3):
    tokens = np.arange(n * tokenizer.SEQ_LEN * 5, dtype=np.int16).reshape(n, tokenizer.SEQ_LEN, 5)
    np.savez_compressed(
        path,
        tokens=tokens,
        actors=np.full((n, tokenizer.SEQ_LEN), 7, dtype=np.int64),
        lengths=np.full(n, tokenizer.SEQ_LEN, dtype=np.int32),
        match_ids=np.arange(n, dtype=np.int64),
        team_ids=np.full(n, 10, dtype=np.int64),
        meta=np.array(
            json.dumps(
                {
                    "competition_id": 55,
                    "season_id": 282,
                    "competition": "UEFA Euro",
                    "season": "2024",
                    "gender": "male",
                    "vocab_fingerprint": fingerprint or vocab.vocabulary_fingerprint(),
                    "vocab_version": vocab.VOCAB_VERSION,
                    "sportio_version": "0.1.0",
                }
            )
        ),
    )
    return path


def test_a_shard_round_trips(tmp_path):
    path = write_shard(tmp_path / "55_282.npz")
    loaded = corpus.load_shard(path)
    assert len(loaded) == 3
    assert loaded.tokens.shape == (3, tokenizer.SEQ_LEN, 5)
    assert loaded.actors.shape == (3, tokenizer.SEQ_LEN)
    assert loaded.vocab_fingerprint == vocab.vocabulary_fingerprint()


def test_a_shard_from_a_different_vocabulary_is_refused(tmp_path):
    path = write_shard(tmp_path / "55_282.npz", fingerprint="deadbeefdeadbeef")
    with pytest.raises(corpus.VocabularyMismatch) as excinfo:
        corpus.load_shard(path)
    # The message must name both fingerprints, or the fix is guesswork.
    assert "deadbeefdeadbeef" in str(excinfo.value)
    assert vocab.vocabulary_fingerprint() in str(excinfo.value)


def test_shard_meta_carries_provenance(tmp_path):
    path = write_shard(tmp_path / "55_282.npz")
    meta = corpus.shard_meta(path)
    assert meta["competition"] == "UEFA Euro"
    assert meta["sportio_version"] == "0.1.0"
    assert meta["vocab_version"] == vocab.VOCAB_VERSION


def test_shard_path_is_keyed_by_competition_and_season(tmp_path):
    assert corpus.shard_path(55, 282, tmp_path).name == "55_282.npz"


def test_building_skips_a_shard_that_already_exists(tmp_path):
    path = write_shard(tmp_path / "55_282.npz")
    stats = corpus.build_shard(
        {"cid": 55, "sid": 282, "comp": "UEFA Euro", "season": "2024"},
        out_dir=tmp_path,
        root=tmp_path / "nonexistent-cache",
    )
    assert stats is None
    assert path.exists()
