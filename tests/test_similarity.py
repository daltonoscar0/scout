"""Query-path tests, including the one that keeps torch out of it."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scout import similarity


def make_index(n=40, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    players = pd.DataFrame(
        {
            "player_id": np.arange(1, n + 1),
            "name": [f"Player {i}" for i in range(1, n + 1)],
            "nickname": [None] * n,
            "competition": ["UEFA Euro"] * n,
            "season": ["2024"] * n,
            "minutes": np.linspace(0, 3000, n),
            "position": ["Center Midfield"] * (n // 2) + ["Goalkeeper"] * (n - n // 2),
        }
    )
    return similarity.PlayerIndex(vectors, players, {"vocab_fingerprint": "x"})


# --- the architectural constraint ----------------------------------------


def test_the_query_path_does_not_import_torch():
    # torch costs 2-4s to import cold, which alone would eat most of the spec's
    # 5s budget. It is also fatal alongside faiss. This is the test that keeps
    # an innocent-looking `from scout.model import ...` out of the query path.
    code = (
        "import sys; import scout.similarity, scout.cli; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_faiss_is_not_installed():
    # faiss-cpu and torch each vendor libomp; importing both aborts the process
    # with rc=134 in either order. Keeping faiss out of the environment is the
    # only reliable guard, since import order is destroyed by ruff's autofix.
    with pytest.raises(ImportError):
        __import__("faiss")


# --- search ---------------------------------------------------------------


def test_a_player_is_not_their_own_nearest_neighbour():
    index = make_index()
    assert all(m.player_id != 1 for m in index.similar(1, k=5))


def test_results_come_back_in_descending_similarity():
    scores = [m.score for m in make_index().similar(1, k=10)]
    assert scores == sorted(scores, reverse=True)


def test_k_is_respected():
    assert len(make_index().similar(1, k=7)) == 7


def test_asking_for_more_neighbours_than_exist_returns_what_there_is():
    index = make_index(n=5)
    assert len(index.similar(1, k=50)) == 4


def test_an_unknown_player_id_raises_rather_than_returning_nothing():
    with pytest.raises(KeyError):
        make_index().similar(999_999)


def test_the_minutes_floor_excludes_low_minute_players():
    index = make_index()
    for match in index.similar(1, k=10, min_minutes=2000):
        assert match.minutes >= 2000


def test_a_filter_that_excludes_everyone_returns_an_empty_list():
    assert make_index().similar(1, k=10, min_minutes=1e9) == []


def test_the_position_filter_restricts_the_pool():
    index = make_index()
    for match in index.similar(1, k=5, position="Goalkeeper"):
        assert match.position == "Goalkeeper"


def test_an_identical_embedding_scores_highest():
    index = make_index(n=10)
    # Make player 5 a copy of player 1; it must come back first.
    index.embeddings[4] = index.embeddings[0]
    assert index.similar(1, k=1)[0].player_id == 5


# --- consistency and loading ---------------------------------------------


def test_a_mismatched_index_is_rejected_at_construction():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="inconsistent"):
        similarity.PlayerIndex(
            rng.normal(size=(10, 4)).astype(np.float32),
            pd.DataFrame({"player_id": [1, 2, 3]}),
            {},
        )


def test_loading_a_missing_index_explains_how_to_build_one(tmp_path):
    with pytest.raises(similarity.IndexMissing) as excinfo:
        similarity.PlayerIndex.load(tmp_path)
    assert "scout embed" in str(excinfo.value)


def test_an_index_round_trips_through_disk(tmp_path):
    index = make_index(n=12, dim=8)
    np.save(tmp_path / similarity.EMBEDDINGS_FILE, index.embeddings)
    index.players.to_parquet(tmp_path / similarity.PLAYERS_FILE)
    (tmp_path / similarity.META_FILE).write_text(json.dumps({"built": "now"}))

    loaded = similarity.PlayerIndex.load(tmp_path)
    assert len(loaded) == 12
    assert loaded.meta["built"] == "now"
    assert [m.player_id for m in loaded.similar(1, k=3)] == [
        m.player_id for m in index.similar(1, k=3)
    ]


# --- names and percentiles ------------------------------------------------


def test_a_typo_still_resolves_to_the_right_player():
    index = make_index()
    hits = index.resolve("Playr 7")
    assert 7 in [player_id for player_id, _, _ in hits]


def test_resolve_returns_candidates_rather_than_guessing():
    # Two players sharing a surname is common; silently picking one is worse
    # than showing a pick-list.
    assert len(make_index().resolve("Player")) > 1


def test_percentiles_are_bounded_and_ordered():
    index = make_index()
    lowest = similarity.percentiles(index, 1, ["minutes"])["minutes"]
    highest = similarity.percentiles(index, 40, ["minutes"])["minutes"]
    assert 0.0 <= lowest < highest <= 100.0


def test_percentiles_ignore_columns_that_are_not_present():
    assert similarity.percentiles(make_index(), 1, ["not_a_column"]) == {}
