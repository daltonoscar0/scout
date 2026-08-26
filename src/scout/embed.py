"""Player embeddings from the frozen LM, plus the contrastive refinement.

For each player, run the frozen model over every sequence they appear in and
pool the hidden states at the positions where they were the actor. The actor
array is never fed to the model -- if it were, the model would learn identity
instead of style -- it is only used here, after the fact, to decide which
positions belong to whom.

Pooling is attention over the player's own positions concatenated with a mean,
then projected and L2-normalised. Mean alone washes out a player whose
distinctive actions are rare; attention alone is unstable for players with few
touches. The concatenation is cheap insurance.

The contrastive stage is the part that actually works, and it works for a
structural reason worth stating: positives are two disjoint halves of the *same
player's same season*, split by match parity. Every player with enough matches
yields a pair, so there are thousands of them. This is the exact opposite of the
projection problem, where genuine cross-domain pairs number in the low hundreds
across the entire open corpus. Same data, same players -- but one task has
abundant supervision and the other has almost none.

This module imports torch. It runs offline and writes plain arrays; the query
path never touches it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from scout import corpus, similarity, vocab
from scout.model import ModelConfig, PossessionLM, select_device

EMBED_DIM = 256
MIN_TOUCHES = 50


class CheckpointMismatch(RuntimeError):
    """A checkpoint was trained under a different vocabulary."""


@dataclass
class EmbedConfig:
    dim: int = EMBED_DIM
    batch_size: int = 64
    min_touches: int = MIN_TOUCHES
    contrastive_epochs: int = 10
    contrastive_batch: int = 256
    temperature: float = 0.07
    learning_rate: float = 1e-3
    seed: int = 0


class PoolingHead(nn.Module):
    """Attention pooling over a player's own positions, concatenated with a mean."""

    def __init__(self, d_model: int, dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.project = nn.Sequential(
            nn.Linear(2 * d_model, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """hidden: (n, d_model) for one player's positions; mask unused, kept for symmetry."""
        scores = hidden @ self.query
        weights = torch.softmax(scores, dim=0)
        attended = (weights[:, None] * hidden).sum(dim=0)
        averaged = hidden.mean(dim=0)
        pooled = self.project(torch.cat([attended, averaged], dim=-1))
        return torch.nn.functional.normalize(pooled, dim=-1)


def load_checkpoint(path: Path, device: torch.device) -> PossessionLM:
    """Load a trained LM, refusing a vocabulary mismatch."""
    payload = torch.load(path, map_location=device, weights_only=False)
    fingerprint = payload.get("vocab_fingerprint")
    current = vocab.vocabulary_fingerprint()
    if fingerprint != current:
        raise CheckpointMismatch(
            f"{path.name} was trained under vocabulary {fingerprint} but the "
            f"current vocabulary is {current}. Retrain, or check out the "
            "vocabulary that produced this checkpoint -- embeddings across the "
            "two are not comparable."
        )
    model = PossessionLM(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_player_states(
    model: PossessionLM,
    shards: list[Path],
    device: torch.device,
    batch_size: int = 64,
    match_filter: set[int] | None = None,
) -> dict[int, list[np.ndarray]]:
    """Hidden states at each player's own token positions, gathered per player.

    Returned as float16 to keep memory sane: a few thousand players with
    hundreds of touches each at 256 dims is otherwise gigabytes.
    """
    states: dict[int, list[np.ndarray]] = defaultdict(list)

    for path in shards:
        shard = corpus.load_shard(path)
        for start in range(0, len(shard), batch_size):
            tokens_np = shard.tokens[start : start + batch_size]
            actors_np = shard.actors[start : start + batch_size]
            if match_filter is not None:
                keep = np.array(
                    [int(m) in match_filter for m in shard.match_ids[start : start + batch_size]]
                )
                if not keep.any():
                    continue
                tokens_np, actors_np = tokens_np[keep], actors_np[keep]

            tokens = torch.from_numpy(tokens_np.astype(np.int64)).to(device)
            hidden = model.hidden_states(tokens).to("cpu").numpy().astype(np.float16)

            for row in range(hidden.shape[0]):
                actors = actors_np[row]
                for player_id in np.unique(actors):
                    if player_id < 0:
                        continue
                    states[int(player_id)].append(hidden[row][actors == player_id])

    return states


def pool_all(
    states: dict[int, list[np.ndarray]],
    head: PoolingHead,
    device: torch.device,
    min_touches: int = MIN_TOUCHES,
) -> tuple[np.ndarray, list[int]]:
    """Pool each player's states into one L2-normalised vector."""
    ids: list[int] = []
    vectors: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for player_id, blocks in sorted(states.items()):
            stacked = np.concatenate(blocks)
            if stacked.shape[0] < min_touches:
                continue
            hidden = torch.from_numpy(stacked.astype(np.float32)).to(device)
            pooled = head(hidden, None).to("cpu").numpy()
            ids.append(player_id)
            vectors.append(pooled)
    if not vectors:
        return np.zeros((0, head.project[-1].out_features), dtype=np.float32), []
    return np.stack(vectors).astype(np.float32), ids


def contrastive_pairs(
    shards: list[Path], min_touches: int = MIN_TOUCHES
) -> dict[int, tuple[set[int], set[int]]]:
    """Split each player's matches by parity into two disjoint halves.

    Odd-numbered against even-numbered matches, by rank within the player's own
    match list. Splitting by parity rather than by a random half keeps both
    sides spread across the season, so the pair is not "early form vs late form".
    """
    per_player: dict[int, set[int]] = defaultdict(set)
    for path in shards:
        shard = corpus.load_shard(path)
        for row in range(len(shard)):
            match_id = int(shard.match_ids[row])
            for player_id in np.unique(shard.actors[row]):
                if player_id >= 0:
                    per_player[int(player_id)].add(match_id)

    pairs: dict[int, tuple[set[int], set[int]]] = {}
    for player_id, matches in per_player.items():
        if len(matches) < 6:
            continue
        ordered = sorted(matches)
        pairs[player_id] = (set(ordered[::2]), set(ordered[1::2]))
    return pairs


def info_nce(anchors: Tensor, positives: Tensor, temperature: float) -> Tensor:
    """Standard InfoNCE with in-batch negatives, symmetric in both directions."""
    logits = anchors @ positives.T / temperature
    labels = torch.arange(anchors.shape[0], device=anchors.device)
    forward = torch.nn.functional.cross_entropy(logits, labels)
    backward = torch.nn.functional.cross_entropy(logits.T, labels)
    return 0.5 * (forward + backward)


def write_index(
    directory: Path,
    embeddings: np.ndarray,
    players: pd.DataFrame,
    meta: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / similarity.EMBEDDINGS_FILE, embeddings.astype(np.float32))
    players.to_parquet(directory / similarity.PLAYERS_FILE, index=False)
    (directory / similarity.META_FILE).write_text(json.dumps(meta, indent=2))


def build_index(
    checkpoint: Path,
    tokens_dir: Path = corpus.TOKENS_DIR,
    out_dir: Path = similarity.INDEX_DIR,
    config: EmbedConfig | None = None,
    device_preference: str = "auto",
) -> Path:
    """Run the frozen LM, pool per player, and write the query-path index."""
    from scout import crosswalk

    config = config or EmbedConfig()
    torch.manual_seed(config.seed)
    device = select_device(device_preference)

    shards = sorted(Path(tokens_dir).glob("*.npz"))
    if not shards:
        raise SystemExit(f"no token shards in {tokens_dir}. Run `scout build` first.")

    model = load_checkpoint(checkpoint, device)
    print(f"collecting hidden states over {len(shards)} shards")
    states = collect_player_states(model, shards, device, config.batch_size)
    print(f"players with any touches: {len(states):,}")

    head = PoolingHead(model.config.d_model, config.dim).to(device)
    embeddings, ids = pool_all(states, head, device, config.min_touches)
    print(f"players clearing {config.min_touches} touches: {len(ids):,}")

    rows = crosswalk.build()
    by_player: dict[int, list] = defaultdict(list)
    for row in rows:
        by_player[row.player_id].append(row)

    records = []
    for player_id in ids:
        seasons = by_player.get(player_id, [])
        best = max(seasons, key=lambda r: r.seconds, default=None)
        records.append(
            {
                "player_id": player_id,
                "name": best.player_name if best else f"player {player_id}",
                "nickname": best.player_nickname if best else None,
                "competition": best.competition if best else "",
                "season": best.season if best else "",
                "gender": best.gender if best else None,
                "minutes": float(sum(r.minutes for r in seasons)),
                "competitions": len({(r.competition_id, r.season_id) for r in seasons}),
                "position": best.main_position if best else None,
                "country": best.country if best else None,
            }
        )

    meta = {
        "vocab_fingerprint": vocab.vocabulary_fingerprint(),
        "vocab_version": vocab.VOCAB_VERSION,
        "checkpoint": str(checkpoint),
        "dim": config.dim,
        "min_touches": config.min_touches,
        "players": len(ids),
        "shards": [p.name for p in shards],
    }
    write_index(Path(out_dir), embeddings, pd.DataFrame(records), meta)
    print(f"wrote index for {len(ids):,} players to {out_dir}")
    return Path(out_dir)
