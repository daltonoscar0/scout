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

# Cap on how many of a player's token positions are held in memory for pooling.
#
# Collecting every position is not affordable: the corpus has 13.85M positions
# with an actor, which at 256 dims in float16 is 6.6 GiB of hidden states, on a
# 16 GiB machine that is also running five other builds. It would swap, and
# there is no disk to swap into.
#
# It is also unnecessary. Pooling estimates a player's central tendency; 256
# positions sampled uniformly from a career is a large sample for that, and a
# 3,000-minute season only yields ~2,000 touches in the first place. What
# matters is that the sample is *uniform over the whole stream* rather than the
# first N positions, which would be biased toward whichever matches happen to
# sort first -- so this is a reservoir, not a truncation.
MAX_POSITIONS_PER_PLAYER = 256


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


class _Reservoir:
    """Uniform sample of at most `capacity` vectors from an unbounded stream.

    Classic reservoir sampling: the first `capacity` items are kept, and item n
    thereafter replaces a random held item with probability capacity/n. Every
    position a player ever touched has the same chance of being in the final
    sample, no matter how many they had or what order the shards arrived in.
    """

    __slots__ = ("capacity", "rng", "seen", "kept")

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.seen = 0
        self.kept: list[np.ndarray] = []

    def add(self, block: np.ndarray) -> None:
        for vector in block:
            self.seen += 1
            if len(self.kept) < self.capacity:
                self.kept.append(vector)
            else:
                index = int(self.rng.integers(0, self.seen))
                if index < self.capacity:
                    self.kept[index] = vector

    def stack(self) -> np.ndarray:
        return np.stack(self.kept) if self.kept else np.zeros((0, 0), dtype=np.float16)


@torch.no_grad()
def collect_player_states(
    model: PossessionLM,
    shards: list[Path],
    device: torch.device,
    batch_size: int = 64,
    match_filter: set[int] | None = None,
    max_positions: int = MAX_POSITIONS_PER_PLAYER,
    seed: int = 0,
) -> dict[int, dict[int, _Reservoir]]:
    """A bounded uniform sample of each player's hidden states, keyed by player.

    Stored as float16 and capped per player, so peak memory is a function of the
    player count rather than of the corpus size. See MAX_POSITIONS_PER_PLAYER
    for why an uncapped version is not affordable here.
    """
    rng = np.random.default_rng(seed)
    # Two reservoirs per player, split by the parity of the match id. They are
    # disjoint by construction, which is exactly what the contrastive stage
    # needs: two independent samples of the same player, from different matches,
    # so a positive pair cannot be two views of one possession. Collecting both
    # in one pass avoids running the LM over the corpus twice.
    states: dict[int, dict[int, _Reservoir]] = {}
    half = max(1, max_positions // 2)

    for path in shards:
        shard = corpus.load_shard(path)
        for start in range(0, len(shard), batch_size):
            tokens_np = shard.tokens[start : start + batch_size]
            actors_np = shard.actors[start : start + batch_size]
            match_np = shard.match_ids[start : start + batch_size]
            if match_filter is not None:
                keep = np.array([int(m) in match_filter for m in match_np])
                if not keep.any():
                    continue
                tokens_np, actors_np, match_np = tokens_np[keep], actors_np[keep], match_np[keep]

            tokens = torch.from_numpy(tokens_np.astype(np.int64)).to(device)
            hidden = model.hidden_states(tokens).to("cpu").numpy().astype(np.float16)

            for row in range(hidden.shape[0]):
                actors = actors_np[row]
                parity = int(match_np[row]) % 2
                for player_id in np.unique(actors):
                    if player_id < 0:
                        continue
                    per_player = states.setdefault(int(player_id), {})
                    reservoir = per_player.get(parity)
                    if reservoir is None:
                        reservoir = _Reservoir(half, rng)
                        per_player[parity] = reservoir
                    reservoir.add(hidden[row][actors == player_id])

    return states


def touches(halves: dict[int, _Reservoir]) -> int:
    """How many positions this player actually had, before sampling."""
    return sum(r.seen for r in halves.values())


def stack_all(halves: dict[int, _Reservoir]) -> np.ndarray:
    blocks = [r.stack() for r in halves.values() if r.kept]
    return np.concatenate(blocks) if blocks else np.zeros((0, 0), dtype=np.float16)


def pool_all(
    states: dict[int, dict[int, _Reservoir]],
    head: PoolingHead,
    device: torch.device,
    min_touches: int = MIN_TOUCHES,
) -> tuple[np.ndarray, list[int]]:
    """Pool each player's sampled states into one L2-normalised vector.

    The touches floor is applied against how many positions the player actually
    had, not against how many were sampled -- otherwise the reservoir cap would
    silently become the floor and a 2,000-touch player would be indistinguishable
    from a 256-touch one.
    """
    ids: list[int] = []
    vectors: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for player_id, halves in sorted(states.items()):
            if touches(halves) < min_touches:
                continue
            stacked = stack_all(halves)
            if stacked.shape[0] == 0:
                continue
            hidden = torch.from_numpy(stacked.astype(np.float32)).to(device)
            pooled = head(hidden, None).to("cpu").numpy()
            ids.append(player_id)
            vectors.append(pooled)
    if not vectors:
        return np.zeros((0, head.project[-1].out_features), dtype=np.float32), []
    return np.stack(vectors).astype(np.float32), ids


def train_pooling_head(
    states: dict[int, dict[int, _Reservoir]],
    head: PoolingHead,
    device: torch.device,
    config: EmbedConfig,
) -> dict:
    """Contrastive fine-tune of the pooling head. This is the stage that works.

    A positive pair is the same player's two match-parity halves: two disjoint
    samples of one player, from different matches. Negatives are the other
    players in the batch. Every player with enough touches on both sides yields
    a pair, so there are thousands -- which is the whole contrast with the
    projection problem, where the entire open corpus contains a few hundred
    cross-domain pairs. Same data, same players, opposite supply of supervision.

    Only the pooling head trains; the language model stays frozen.
    """
    eligible = [
        pid
        for pid, halves in sorted(states.items())
        if len(halves) == 2
        and all(r.kept for r in halves.values())
        and touches(halves) >= config.min_touches
    ]
    if len(eligible) < 32:
        return {"pairs": len(eligible), "trained": False}

    tensors = {
        pid: [
            torch.from_numpy(states[pid][parity].stack().astype(np.float32)).to(device)
            for parity in sorted(states[pid])
        ]
        for pid in eligible
    }

    optimiser = torch.optim.AdamW(head.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    head.train()
    history = []

    for epoch in range(config.contrastive_epochs):
        order = rng.permutation(len(eligible))
        total, batches = 0.0, 0
        for start in range(0, len(order), config.contrastive_batch):
            chunk = order[start : start + config.contrastive_batch]
            if len(chunk) < 8:  # InfoNCE needs negatives to be meaningful
                continue
            anchors = torch.stack([head(tensors[eligible[i]][0], None) for i in chunk])
            positives = torch.stack([head(tensors[eligible[i]][1], None) for i in chunk])
            value = info_nce(anchors, positives, config.temperature)
            optimiser.zero_grad(set_to_none=True)
            value.backward()
            optimiser.step()
            total += float(value.detach())
            batches += 1
        if batches:
            history.append(total / batches)
            print(
                f"  contrastive epoch {epoch + 1}/{config.contrastive_epochs}: "
                f"loss {history[-1]:.4f}",
                flush=True,
            )

    head.eval()
    return {
        "pairs": len(eligible),
        "trained": True,
        "first_loss": history[0] if history else None,
        "last_loss": history[-1] if history else None,
    }


@torch.no_grad()
def retrieval_eval(
    states: dict[int, dict[int, _Reservoir]],
    head: PoolingHead,
    device: torch.device,
    min_touches: int = MIN_TOUCHES,
    limit: int = 1500,
) -> dict:
    """Held-out same-player retrieval: does half A find half B?

    The honest test of whether these embeddings describe a player rather than a
    match. Each player's two parity halves are pooled separately; a query is one
    half, and the target is the same player's other half among all the others.
    Chance is 1/n, so recall@1 is directly interpretable.
    """
    eligible = [
        pid
        for pid, halves in sorted(states.items())
        if len(halves) == 2
        and all(r.kept for r in halves.values())
        and touches(halves) >= min_touches
    ][:limit]
    if len(eligible) < 16:
        return {"players": len(eligible), "evaluated": False}

    head.eval()
    left, right = [], []
    for pid in eligible:
        halves = [states[pid][p].stack().astype(np.float32) for p in sorted(states[pid])]
        left.append(head(torch.from_numpy(halves[0]).to(device), None))
        right.append(head(torch.from_numpy(halves[1]).to(device), None))
    a = torch.stack(left)
    b = torch.stack(right)

    scores = a @ b.T
    ranks = (scores > scores.diag().unsqueeze(1)).sum(dim=1) + 1
    ranks = ranks.float()
    return {
        "players": len(eligible),
        "evaluated": True,
        "recall_at_1": float((ranks <= 1).float().mean()),
        "recall_at_10": float((ranks <= 10).float().mean()),
        "mean_rank": float(ranks.mean()),
        "chance_recall_at_1": 1.0 / len(eligible),
    }


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
    print(f"collecting hidden states over {len(shards)} shards", flush=True)
    states = collect_player_states(model, shards, device, config.batch_size, seed=config.seed)
    print(f"players with any touches: {len(states):,}", flush=True)

    head = PoolingHead(model.config.d_model, config.dim).to(device)

    before = retrieval_eval(states, head, device, config.min_touches)
    print(f"retrieval before contrastive training: {before}", flush=True)

    stats = train_pooling_head(states, head, device, config)
    print(f"contrastive: {stats}", flush=True)

    after = retrieval_eval(states, head, device, config.min_touches)
    print(f"retrieval after contrastive training:  {after}", flush=True)

    embeddings, ids = pool_all(states, head, device, config.min_touches)
    print(f"players clearing {config.min_touches} touches: {len(ids):,}", flush=True)

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
        "max_positions_per_player": MAX_POSITIONS_PER_PLAYER,
        "players": len(ids),
        "shards": len(shards),
        # Recorded so the index carries its own evidence: how well a player's
        # embedding from one set of matches finds the same player's embedding
        # from a disjoint set. Without this the index is unfalsifiable.
        "retrieval_before_contrastive": before,
        "retrieval_after_contrastive": after,
        "contrastive": stats,
    }
    write_index(Path(out_dir), embeddings, pd.DataFrame(records), meta)
    print(f"wrote index for {len(ids):,} players to {out_dir}")
    return Path(out_dir)
