"""The query path: nearest-neighbour search over player embeddings.

Deliberately numpy-only. No torch, no faiss, no lightgbm.

faiss is not merely unnecessary here, it is unusable: faiss-cpu and torch each
vendor their own libomp, and importing both in one process aborts with rc=134 in
either order. The usual `KMP_DUPLICATE_LIB_OK=TRUE` workaround converts the
clean abort into a segfault, and relying on import order is unshippable because
ruff's import sorter will reorder them on its next autofix.

It is also not needed. The player universe is a few thousand; a cosine search is
one matrix-vector product over a 6,300 x 256 float32 matrix -- about 1.6 MFLOP,
which numpy does in single-digit milliseconds. The spec's budget is 5 seconds.

Embeddings are memory-mapped, so a Streamlit rerun does not re-read 6 MB from
disk and startup stays well under a second.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_DIR = Path("data/derived/index")
EMBEDDINGS_FILE = "embeddings.npy"
PLAYERS_FILE = "players.parquet"
META_FILE = "index_meta.json"


class IndexMissing(RuntimeError):
    """The prebuilt index is not there."""


@dataclass
class Match:
    player_id: int
    name: str
    score: float
    competition: str
    season: str
    minutes: float
    position: str | None


class PlayerIndex:
    """Embeddings plus their metadata, with cosine search over them."""

    def __init__(self, embeddings: np.ndarray, players: pd.DataFrame, meta: dict) -> None:
        if len(players) != embeddings.shape[0]:
            raise ValueError(
                f"index is inconsistent: {embeddings.shape[0]} embeddings but "
                f"{len(players)} player rows"
            )
        self.embeddings = embeddings
        self.players = players.reset_index(drop=True)
        self.meta = meta
        self._by_id = {int(pid): i for i, pid in enumerate(self.players["player_id"])}

    # -- loading ----------------------------------------------------------

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> PlayerIndex:
        directory = Path(directory)
        embeddings_path = directory / EMBEDDINGS_FILE
        players_path = directory / PLAYERS_FILE
        if not embeddings_path.exists() or not players_path.exists():
            raise IndexMissing(
                f"no prebuilt index in {directory}. Run `scout embed` to build one, "
                "or `scout demo` to use the bundled offline index."
            )
        meta_path = directory / META_FILE
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        # mmap so a Streamlit rerun does not re-read the whole matrix.
        embeddings = np.load(embeddings_path, mmap_mode="r")
        return cls(np.asarray(embeddings), pd.read_parquet(players_path), meta)

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    # -- search -----------------------------------------------------------

    def similar(
        self,
        player_id: int,
        k: int = 10,
        *,
        min_minutes: float = 0.0,
        position: str | None = None,
        exclude_same_player: bool = True,
    ) -> list[Match]:
        """Cosine nearest neighbours.

        Embeddings are L2-normalised at build time, so a dot product is the
        cosine and no per-query normalisation is needed.
        """
        row = self._by_id.get(int(player_id))
        if row is None:
            raise KeyError(f"player_id {player_id} is not in the index")

        scores = self.embeddings @ self.embeddings[row]
        eligible = np.ones(len(self), dtype=bool)
        if exclude_same_player:
            eligible &= self.players["player_id"].to_numpy() != int(player_id)
        if min_minutes > 0:
            eligible &= self.players["minutes"].to_numpy() >= min_minutes
        if position:
            eligible &= (
                self.players["position"].fillna("").str.contains(position, case=False).to_numpy()
            )

        scores = np.where(eligible, scores, -np.inf)
        # argpartition is O(n) against a full sort's O(n log n). At n=6,300 both
        # are trivial, but this keeps the cost flat if the index grows.
        take = min(k, int(eligible.sum()))
        if take <= 0:
            return []
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        return [self._match(int(i), float(scores[i])) for i in top]

    def _match(self, row: int, score: float) -> Match:
        record = self.players.iloc[row]
        return Match(
            player_id=int(record["player_id"]),
            name=str(record["name"]),
            score=score,
            competition=str(record.get("competition", "")),
            season=str(record.get("season", "")),
            minutes=float(record.get("minutes", 0.0)),
            position=(None if pd.isna(record.get("position")) else str(record.get("position"))),
        )

    # -- name resolution --------------------------------------------------

    def resolve(self, query: str, limit: int = 8) -> list[tuple[int, str, float]]:
        """Fuzzy-match a typed name to candidate players.

        Returns (player_id, display name, score) sorted best first. The caller
        decides whether a single confident hit is enough or whether to show a
        pick-list; this function never guesses on the user's behalf, because two
        players sharing a surname is common and silently picking one is worse
        than asking.
        """
        from rapidfuzz import fuzz, process

        names = self.players["name"].astype(str).tolist()
        nicknames = self.players.get("nickname")
        haystack = list(names)
        if nicknames is not None:
            haystack += [str(n) for n in nicknames.fillna("")]

        results = process.extract(query, haystack, scorer=fuzz.WRatio, limit=limit * 2)
        seen: dict[int, tuple[int, str, float]] = {}
        for _, score, position in results:
            row = position % len(names)
            player_id = int(self.players.iloc[row]["player_id"])
            if player_id not in seen or score > seen[player_id][2]:
                seen[player_id] = (player_id, names[row], float(score))
        return sorted(seen.values(), key=lambda item: -item[2])[:limit]


def percentiles(index: PlayerIndex, player_id: int, columns: list[str]) -> dict[str, float]:
    """Where a player sits within the index on each of `columns`, as 0-100.

    Used for the scout card's radar. Percentile rank rather than raw value,
    because raw per-90 numbers are not comparable across competitions.
    """
    row = index.players[index.players["player_id"] == int(player_id)]
    if row.empty:
        raise KeyError(f"player_id {player_id} is not in the index")

    out: dict[str, float] = {}
    for column in columns:
        if column not in index.players:
            continue
        series = index.players[column].astype(float)
        value = float(row.iloc[0][column])
        out[column] = float((series < value).mean() * 100.0)
    return out
