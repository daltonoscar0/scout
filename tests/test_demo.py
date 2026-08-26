"""Tests for the committed offline demo bundle.

The index is synthetic throughout. No trained index exists yet, and a test that
needs one would be a test that skips.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scout import demo, similarity


def make_source(directory, n=60, dim=8, seed=0):
    """Write a full index to `directory` and return it with a matching metrics frame."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    # Minutes deliberately not monotonic in row order, so a build that ignored
    # them and kept the first rows would look identical to one that ranked.
    minutes = rng.permutation(np.linspace(90.0, 3000.0, n))
    players = pd.DataFrame(
        {
            "player_id": np.arange(1, n + 1),
            "name": [f"Player {i}" for i in range(1, n + 1)],
            "nickname": [None] * n,
            "competition": ["UEFA Euro"] * n,
            "season": ["2024"] * n,
            "gender": ["male"] * n,
            "minutes": minutes,
            "competitions": np.ones(n, dtype=int),
            "position": ["Center Midfield"] * (n // 2) + ["Goalkeeper"] * (n - n // 2),
            "country": ["Nowhere"] * n,
        }
    )
    meta = {"vocab_fingerprint": "abc123", "checkpoint": "models/lm.pt", "dim": dim, "players": n}

    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / similarity.EMBEDDINGS_FILE, vectors)
    players.to_parquet(directory / similarity.PLAYERS_FILE, index=False)
    (directory / similarity.META_FILE).write_text(json.dumps(meta))

    metrics = pd.DataFrame(
        {
            "player_id": np.repeat(np.arange(1, n + 1), 2),
            "competition": ["UEFA Euro"] * (2 * n),
            "season": ["2024"] * (2 * n),
            "minutes": np.repeat(minutes, 2) / 2,
            "xg_per90": rng.random(2 * n),
            "progressive_actions_per90": rng.random(2 * n),
        }
    )
    return similarity.PlayerIndex.load(directory), metrics


def build(tmp_path, *, n=60, players=20, seed=0):
    source_dir = tmp_path / "index"
    out_dir = tmp_path / "bundle"
    source, metrics = make_source(source_dir, n=n, seed=seed)
    demo.build_bundle(source_dir, metrics, out_dir, players=players)
    return source, metrics, out_dir


# --- the alignment invariant ---------------------------------------------


def test_a_bundled_players_embedding_is_their_embedding_in_the_source(tmp_path):
    # The worst available failure mode: slice the vectors one way and the rows
    # another, and every neighbour lookup silently answers for someone else.
    source, _, out_dir = build(tmp_path)
    bundle, _ = demo.load_bundle(out_dir)

    for player_id in bundle.players["player_id"]:
        bundled = bundle.embeddings[bundle.players["player_id"].to_list().index(player_id)]
        original = source.embeddings[source.players["player_id"].to_list().index(player_id)]
        np.testing.assert_array_equal(bundled, original)


def test_neighbours_of_a_bundled_player_match_the_source_restricted_to_the_bundle(tmp_path):
    source, _, out_dir = build(tmp_path)
    bundle, _ = demo.load_bundle(out_dir)
    target = int(bundle.players["player_id"].iloc[0])
    kept = set(bundle.players["player_id"].astype(int))

    from_bundle = [m.player_id for m in bundle.similar(target, k=5)]
    from_source = [
        m.player_id for m in source.similar(target, k=len(source)) if m.player_id in kept
    ]
    assert from_bundle == from_source[:5]


def test_every_bundled_row_agrees_with_the_source_row(tmp_path):
    source, _, out_dir = build(tmp_path)
    bundle, _ = demo.load_bundle(out_dir)
    merged = bundle.players.merge(source.players, on="player_id", suffixes=("_b", "_s"))
    assert len(merged) == len(bundle.players)
    assert (merged["name_b"] == merged["name_s"]).all()
    assert np.allclose(merged["minutes_b"], merged["minutes_s"])


# --- selection ------------------------------------------------------------


def test_the_bundle_keeps_the_busiest_players(tmp_path):
    source, _, out_dir = build(tmp_path, n=60, players=20)
    bundle, _ = demo.load_bundle(out_dir)
    cutoff = np.sort(source.players["minutes"].to_numpy())[-20]
    assert bundle.players["minutes"].min() >= cutoff


def test_the_player_count_is_respected(tmp_path):
    _, _, out_dir = build(tmp_path, n=60, players=20)
    assert len(demo.load_bundle(out_dir)[0]) == 20


def test_asking_for_more_players_than_the_index_has_keeps_all_of_them(tmp_path):
    _, _, out_dir = build(tmp_path, n=12, players=500)
    assert len(demo.load_bundle(out_dir)[0]) == 12


def test_a_non_positive_player_count_is_rejected(tmp_path):
    source_dir = tmp_path / "index"
    _, metrics = make_source(source_dir, n=10)
    with pytest.raises(ValueError, match="positive"):
        demo.build_bundle(source_dir, metrics, tmp_path / "bundle", players=0)


def test_rebuilding_gives_the_same_bundle(tmp_path):
    # A bundle that churns on every rebuild is a bundle nobody wants in git.
    source_dir, out_dir = tmp_path / "index", tmp_path / "bundle"
    _, metrics = make_source(source_dir, n=40)
    demo.build_bundle(source_dir, metrics, out_dir, players=15)
    first = (out_dir / similarity.EMBEDDINGS_FILE).read_bytes()
    demo.build_bundle(source_dir, metrics, out_dir, players=15)
    assert (out_dir / similarity.EMBEDDINGS_FILE).read_bytes() == first


# --- metrics --------------------------------------------------------------


def test_metrics_are_restricted_to_the_bundled_players(tmp_path):
    _, _, out_dir = build(tmp_path, n=60, players=20)
    bundle, metrics = demo.load_bundle(out_dir)
    assert set(metrics["player_id"].astype(int)) <= set(bundle.players["player_id"].astype(int))
    assert len(metrics) == 40  # two comp-season rows per kept player


def test_metric_columns_survive_the_slice(tmp_path):
    _, _, out_dir = build(tmp_path)
    _, metrics = demo.load_bundle(out_dir)
    assert {"xg_per90", "progressive_actions_per90"} <= set(metrics.columns)


def test_a_bundle_can_be_built_without_any_metrics(tmp_path):
    source_dir, out_dir = tmp_path / "index", tmp_path / "bundle"
    make_source(source_dir, n=20)
    demo.build_bundle(source_dir, None, out_dir, players=5)
    bundle, metrics = demo.load_bundle(out_dir)
    assert len(bundle) == 5
    assert metrics.empty


# --- self-containment -----------------------------------------------------


def test_the_bundle_loads_when_there_is_no_derived_index_at_all(tmp_path, monkeypatch):
    # The fresh-clone case: data/ is gitignored, so the only index on disk is
    # the bundle itself.
    _, _, out_dir = build(tmp_path)
    home = tmp_path / "clone"
    home.mkdir()
    monkeypatch.chdir(home)
    assert not similarity.INDEX_DIR.exists()

    bundle, _ = demo.load_bundle(out_dir)
    assert bundle.similar(int(bundle.players["player_id"].iloc[0]), k=3)


def test_the_bundle_needs_no_network(tmp_path, monkeypatch):
    import requests

    def refuse(*args, **kwargs):
        raise AssertionError("the demo bundle must not touch the network")

    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(requests.Session, "get", refuse)
    monkeypatch.setattr(requests, "request", refuse)

    _, _, out_dir = build(tmp_path)
    bundle, metrics = demo.load_bundle(out_dir)
    matches = bundle.similar(int(bundle.players["player_id"].iloc[0]), k=5)
    assert len(matches) == 5
    assert not metrics.empty


def test_loading_the_bundle_does_not_import_torch(tmp_path):
    # Same constraint as the rest of the query path, checked in a clean process
    # because the test session itself has already imported torch elsewhere.
    _, _, out_dir = build(tmp_path)
    code = (
        "import sys; from scout import demo; "
        f"index, metrics = demo.load_bundle({str(out_dir)!r}); "
        "index.similar(int(index.players['player_id'].iloc[0]), k=3); "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_the_bundle_is_small_enough_to_commit(tmp_path):
    # 200 players at 256 dims is the shipping shape; assert on that, not on the
    # tiny synthetic default.
    source_dir, out_dir = tmp_path / "index", tmp_path / "bundle"
    _, metrics = make_source(source_dir, n=400, dim=256)
    demo.build_bundle(source_dir, metrics, out_dir, players=200)
    size = sum(p.stat().st_size for p in out_dir.iterdir())
    assert size < demo.MAX_BUNDLE_BYTES


def test_an_oversized_bundle_fails_instead_of_being_written(tmp_path, monkeypatch):
    source_dir, out_dir = tmp_path / "index", tmp_path / "bundle"
    _, metrics = make_source(source_dir, n=40)
    monkeypatch.setattr(demo, "MAX_BUNDLE_BYTES", 16)

    with pytest.raises(ValueError, match="commit budget"):
        demo.build_bundle(source_dir, metrics, out_dir, players=20)
    # The manifest is written last, so the half-built directory is not loadable.
    assert not demo.bundle_exists(out_dir)


# --- missing bundle -------------------------------------------------------


def test_bundle_exists_is_false_before_a_build(tmp_path):
    assert not demo.bundle_exists(tmp_path / "nothing")


def test_a_missing_bundle_says_how_to_build_one(tmp_path):
    with pytest.raises(demo.BundleMissing) as excinfo:
        demo.load_bundle(tmp_path / "nothing")
    assert "scout demo --build" in str(excinfo.value)


def test_a_missing_bundle_is_also_an_index_missing(tmp_path):
    # Callers that already handle "no index" should not need a second branch.
    with pytest.raises(similarity.IndexMissing):
        demo.manifest(tmp_path / "nothing")


def test_a_partial_bundle_does_not_count_as_one(tmp_path):
    _, _, out_dir = build(tmp_path)
    (out_dir / demo.METRICS_FILE).unlink()
    assert not demo.bundle_exists(out_dir)
    with pytest.raises(demo.BundleMissing):
        demo.load_bundle(out_dir)


def test_building_from_a_missing_index_reports_the_index_not_the_bundle(tmp_path):
    with pytest.raises(similarity.IndexMissing, match="scout embed"):
        demo.build_bundle(tmp_path / "nothing", pd.DataFrame(), tmp_path / "bundle")


# --- manifest -------------------------------------------------------------


def test_the_manifest_records_what_the_bundle_is(tmp_path):
    _, _, out_dir = build(tmp_path, n=60, players=20)
    recorded = demo.manifest(out_dir)
    assert recorded["players"] == 20
    assert recorded["source_players"] == 60
    assert recorded["dim"] == 8
    assert recorded["metric_rows"] == 40
    assert recorded["bytes"] > 0
    assert recorded["bundle_version"] == demo.BUNDLE_VERSION


def test_the_manifest_carries_the_vocabulary_fingerprint(tmp_path):
    # Embeddings built under two vocabularies are not comparable and the symptom
    # is plausible-looking wrong neighbours, so the fingerprint travels with the
    # bundle rather than being inferred later.
    _, _, out_dir = build(tmp_path)
    recorded = demo.manifest(out_dir)
    assert recorded["vocab_fingerprint"] == "abc123"
    assert recorded["source_index"]["checkpoint"] == "models/lm.pt"


def test_the_manifest_records_the_build_date(tmp_path):
    from datetime import datetime

    _, _, out_dir = build(tmp_path)
    built = datetime.fromisoformat(demo.manifest(out_dir)["built"])
    assert built.tzinfo is not None


def test_the_bundles_own_meta_reports_the_bundled_player_count(tmp_path):
    # Carrying the source's count would have the app print "6,300 players" over
    # a 200-player index.
    _, _, out_dir = build(tmp_path, n=60, players=20)
    bundle, _ = demo.load_bundle(out_dir)
    assert bundle.meta["players"] == 20
    assert bundle.meta["demo"] is True
