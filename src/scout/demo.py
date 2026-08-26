"""The committed offline demo bundle: a full index shrunk to something shippable.

The spec's hardest acceptance criterion is "fresh clone + `scout demo` works
offline", and the repo gitignores data/ and models/. A fresh clone therefore has
no index, no corpus, no cache and no checkpoint, so `scout demo` cannot rebuild
any of it -- the only way the criterion is satisfiable is if the demo's inputs
are in the repository already. That is what this module produces.

The bundle is a *strict subset* of a real index, written under the same
filenames, so `similarity.PlayerIndex.load(DEMO_DIR)` loads it with no special
case anywhere on the query path. Nothing here is regenerated or approximated:
the vectors are the trained ones, byte for byte, for the players that were kept.

Size is what makes it committable. 200 players at 256 dims of float32 is about
200 KB of embeddings, so no quantisation is needed and the vectors stay
bit-identical to the source. A hard ceiling is enforced at build time rather
than left to reviewer discipline, because a bundle that quietly grows to 80 MB
is discovered by whoever clones the repo, not by whoever built it.

Query-path module: numpy and pandas only. It deliberately does not reuse
`embed.write_index`, because `scout.embed` imports torch at module level.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scout import similarity

DEMO_DIR = Path("data/demo")
METRICS_FILE = "metrics.parquet"
MANIFEST_FILE = "manifest.json"

DEFAULT_PLAYERS = 200
BUNDLE_VERSION = 1

# Well under GitHub's 50 MB per-file warning and small enough that cloning the
# repo is not a download decision. A build that would exceed it fails instead of
# committing the reviewer to a large blob in the history, where it is permanent.
MAX_BUNDLE_BYTES = 20 * 1024 * 1024


class BundleMissing(similarity.IndexMissing):
    """No demo bundle where one was expected.

    Subclasses IndexMissing so a caller that already handles "there is no index"
    handles this too, but carries its own message, since the fix is different:
    the bundle is built by `scout demo --build`, not by `scout embed`.
    """


def _bundle_files(out_dir: Path) -> tuple[Path, ...]:
    out_dir = Path(out_dir)
    return (
        out_dir / similarity.EMBEDDINGS_FILE,
        out_dir / similarity.PLAYERS_FILE,
        out_dir / METRICS_FILE,
        # The index metadata carries the vocabulary fingerprint and player count.
        # Omitting it here would let a bundle missing that file read as complete,
        # and PlayerIndex.load would then silently return meta={}.
        out_dir / similarity.META_FILE,
        out_dir / MANIFEST_FILE,
    )


def bundle_exists(out_dir: Path = DEMO_DIR) -> bool:
    """True only if every part is present.

    The manifest is written last, so a build that failed its size check or was
    interrupted leaves files behind but does not read as a bundle.
    """
    return all(path.exists() for path in _bundle_files(out_dir))


def _require(out_dir: Path) -> None:
    if not bundle_exists(out_dir):
        raise BundleMissing(
            f"no demo bundle in {out_dir}. Run `scout demo --build` to slice one "
            "from a full index, or check out a revision where the bundle is "
            "committed -- the demo is meant to run offline from the repo alone."
        )


def _select(players: pd.DataFrame, count: int) -> np.ndarray:
    """Row positions of the `count` highest-minute players, in source order.

    Returned ascending rather than by minutes so the bundle's row order is a
    deterministic function of the source index: rebuilding at a different
    `count` reorders nothing, it only adds or drops rows.
    """
    if "minutes" not in players:
        # A malformed index rather than a real one; keep the first rows so the
        # build still produces something inspectable.
        return np.arange(min(count, len(players)))
    # NaN minutes rank last rather than poisoning the comparison.
    values = np.nan_to_num(players["minutes"].to_numpy(dtype=float), nan=-1.0)
    return np.sort(np.argsort(-values, kind="stable")[:count])


def build_bundle(
    index_dir: Path,
    metrics: pd.DataFrame | None,
    out_dir: Path = DEMO_DIR,
    *,
    players: int = DEFAULT_PLAYERS,
) -> Path:
    """Slice a full index down to its `players` busiest players and write a bundle.

    `metrics` is the frame from `corpus.load_metrics()`; it is passed in rather
    than loaded here so this stays a pure function of its inputs and so the
    query path never reaches into data/derived/metrics. None or an empty frame
    is accepted: similarity still works from the bundle, the percentile radar
    simply has nothing to draw.
    """
    if players <= 0:
        raise ValueError(f"players must be positive, got {players}")

    source = similarity.PlayerIndex.load(index_dir)
    positions = _select(source.players, players)

    # One index array drives both slices. Slicing them independently -- by
    # player_id on one side and by position on the other, say -- is the single
    # worst bug available in this project: every neighbour lookup would return a
    # different player's vector, with plausible-looking output and no error.
    embeddings = np.ascontiguousarray(source.embeddings[positions], dtype=np.float32)
    kept = source.players.iloc[positions].reset_index(drop=True)

    meta = dict(source.meta)
    meta.update({"players": len(kept), "demo": True, "source_players": len(source)})
    # Constructing the index re-runs its consistency check before anything is
    # written, so a mis-sliced bundle cannot reach disk.
    bundle = similarity.PlayerIndex(embeddings, kept, meta)

    ids = set(kept["player_id"].astype(int))
    if metrics is not None and not metrics.empty and "player_id" in metrics:
        # to_numeric rather than astype(int): player_id can arrive as a pandas
        # nullable Int64 carrying pd.NA, and astype raises on that rather than
        # dropping the unusable rows.
        numeric = pd.to_numeric(metrics["player_id"], errors="coerce")
        rows = metrics[numeric.notna() & numeric.isin(ids)].reset_index(drop=True)
    else:
        rows = pd.DataFrame(columns=list(metrics.columns) if metrics is not None else [])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path, players_path, metrics_path, meta_path, manifest_path = _bundle_files(out_dir)

    # Remove the old manifest before touching any data file. A rebuild that
    # fails its size check would otherwise leave new data beside a stale
    # manifest, and bundle_exists() would keep reporting a complete bundle that
    # describes something no longer on disk.
    manifest_path.unlink(missing_ok=True)

    np.save(embeddings_path, bundle.embeddings)
    bundle.players.to_parquet(players_path, index=False)
    rows.to_parquet(metrics_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))

    written = [embeddings_path, players_path, metrics_path, meta_path]
    size = sum(path.stat().st_size for path in written)
    if size > MAX_BUNDLE_BYTES:
        raise ValueError(
            f"bundle would be {size / 1e6:.1f} MB, over the {MAX_BUNDLE_BYTES / 1e6:.0f} MB "
            f"commit budget. Rebuild with fewer players (currently {len(kept)})."
        )

    manifest_path.write_text(
        json.dumps(
            {
                "bundle_version": BUNDLE_VERSION,
                "built": datetime.now(UTC).isoformat(timespec="seconds"),
                "players": len(kept),
                "dim": int(embeddings.shape[1]),
                # Carried so a bundle built under an older vocabulary is
                # identifiable after the fact; embeddings from two vocabularies
                # are not comparable and the symptom is silent.
                "vocab_fingerprint": meta.get("vocab_fingerprint"),
                "source_index": dict(source.meta),
                "source_index_dir": str(index_dir),
                "source_players": len(source),
                "metric_rows": int(len(rows)),
                "bytes": int(size),
            },
            indent=2,
        )
    )
    return out_dir


def load_bundle(out_dir: Path = DEMO_DIR) -> tuple[similarity.PlayerIndex, pd.DataFrame]:
    """Load the bundle. No network, no torch, no dependency on data/derived.

    Everything read here is committed to the repo, which is the whole point:
    this must work in a fresh clone with no cache and no trained checkpoint.
    """
    out_dir = Path(out_dir)
    _require(out_dir)
    _, _, metrics_path, _, _ = _bundle_files(out_dir)
    return similarity.PlayerIndex.load(out_dir), pd.read_parquet(metrics_path)


def manifest(out_dir: Path = DEMO_DIR) -> dict:
    """What the bundle is and where it came from."""
    out_dir = Path(out_dir)
    _require(out_dir)
    return json.loads((out_dir / MANIFEST_FILE).read_text())
