"""Build the tokenized corpus from the cached events, one comp-season at a time.

Output is one .npz per competition-season under data/derived/tokens/, which
makes the build resumable: an interrupted run picks up at the first comp-season
that has no output file. That matters because a full pass is thousands of
matches, and this machine has already had a long-running fetch killed once.

Each shard stores the token array, the actor alignment, sequence lengths, and
the match/team ids, plus the vocabulary fingerprint it was built under. Loading
a shard whose fingerprint does not match the current vocabulary is an error, not
a warning -- embeddings trained under one vocabulary are meaningless under
another, and the symptom would be plausible-looking wrong neighbours.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scout import fetch, metrics, sportio_adapter, tokenizer, vocab

TOKENS_DIR = Path("data/derived/tokens")
METRICS_DIR = Path("data/derived/metrics")


class VocabularyMismatch(RuntimeError):
    """A shard was built under a different vocabulary than the one loaded."""


@dataclass
class ShardStats:
    competition: str
    season: str
    matches: int
    sequences: int
    events: int
    tokenized: int


def shard_path(competition_id: int, season_id: int, out_dir: Path = TOKENS_DIR) -> Path:
    return out_dir / f"{competition_id}_{season_id}.npz"


def build_shard(
    comp_season: dict,
    *,
    out_dir: Path = TOKENS_DIR,
    metrics_dir: Path = METRICS_DIR,
    root: Path | None = None,
    coverage: tokenizer.Coverage | None = None,
    minutes: dict[int, float] | None = None,
    force: bool = False,
) -> ShardStats | None:
    """Tokenize one competition-season from cached events. Returns None if skipped.

    `minutes` maps player_id to minutes played in *this* competition-season, and
    comes from the lineup crosswalk. It is passed in rather than derived here
    because building it reads all 3,961 lineup files, which must happen once for
    the whole build rather than once per shard.
    """
    root = fetch.cache_root() if root is None else root
    cid, sid = comp_season["cid"], comp_season["sid"]
    path = shard_path(cid, sid, out_dir)
    if path.exists() and not force:
        return None

    matches = fetch.matches(cid, sid, root=root)

    # Metrics are accumulated in the same pass. Parsing 3,961 events files is by
    # far the most expensive step in the build, and doing it twice to produce
    # two artifacts from the same rows would be pure waste.
    totals = metrics.empty_totals()

    token_blocks: list[np.ndarray] = []
    actor_blocks: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    match_ids: list[np.ndarray] = []
    team_ids: list[np.ndarray] = []
    used_matches = 0
    local = tokenizer.Coverage()

    for match in matches:
        match_id = match["match_id"]
        payload = fetch.read_cached(root, f"events/{match_id}.json")
        if payload is None:
            continue
        try:
            events = sportio_adapter.load_events_from_payload(
                payload,
                match_id=match_id,
                competition=comp_season["comp"],
                season=comp_season["season"],
            )
        except Exception:
            # One unparseable match must not abandon a whole competition. It is
            # counted as an unmodelled event batch so the loss is visible.
            local.unmodelled_types["<unparseable match>"] += 1
            continue

        metrics.accumulate(events, totals)

        shard = tokenizer.tokenize_events(events)
        local.merge(shard.coverage)
        if len(shard) == 0:
            continue
        token_blocks.append(shard.tokens)
        actor_blocks.append(shard.actors)
        lengths.append(shard.lengths)
        match_ids.append(shard.match_ids)
        team_ids.append(shard.team_ids)
        used_matches += 1

    if coverage is not None:
        coverage.merge(local)
    if not token_blocks:
        return None

    if minutes is not None:
        frame = metrics.to_frame(totals, minutes)
        if not frame.empty:
            frame.insert(1, "competition_id", cid)
            frame.insert(2, "season_id", sid)
            frame.insert(3, "competition", comp_season["comp"])
            frame.insert(4, "season", comp_season["season"])
            frame.insert(5, "gender", comp_season.get("gender"))
            metrics_dir.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(metrics_dir / f"{cid}_{sid}.parquet", index=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.concatenate(token_blocks)
    np.savez_compressed(
        path,
        tokens=tokens,
        actors=np.concatenate(actor_blocks),
        lengths=np.concatenate(lengths),
        match_ids=np.concatenate(match_ids),
        team_ids=np.concatenate(team_ids),
        meta=np.array(
            json.dumps(
                {
                    "competition_id": cid,
                    "season_id": sid,
                    "competition": comp_season["comp"],
                    "season": comp_season["season"],
                    "gender": comp_season.get("gender"),
                    "vocab_fingerprint": vocab.vocabulary_fingerprint(),
                    "vocab_version": vocab.VOCAB_VERSION,
                    "sportio_version": sportio_adapter.sportio_version(),
                }
            )
        ),
    )
    return ShardStats(
        competition=comp_season["comp"],
        season=comp_season["season"],
        matches=used_matches,
        sequences=int(tokens.shape[0]),
        events=local.events,
        tokenized=local.tokenized,
    )


def load_shard(path: Path) -> tokenizer.Corpus:
    """Load one shard, refusing it if the vocabulary has moved underneath it."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        current = vocab.vocabulary_fingerprint()
        if meta.get("vocab_fingerprint") != current:
            raise VocabularyMismatch(
                f"{path.name} was tokenized under vocabulary "
                f"{meta.get('vocab_fingerprint')} but the current vocabulary is "
                f"{current}. Rebuild the corpus -- embeddings across the two are "
                "not comparable."
            )
        return tokenizer.Corpus(
            tokens=data["tokens"],
            actors=data["actors"],
            lengths=data["lengths"],
            match_ids=data["match_ids"],
            team_ids=data["team_ids"],
            coverage=tokenizer.Coverage(),
            vocab_fingerprint=meta["vocab_fingerprint"],
        )


def shard_meta(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["meta"]))


def load_metrics(metrics_dir: Path = METRICS_DIR):
    """All per-player metric rows across every built shard, as one frame."""
    import pandas as pd

    files = sorted(Path(metrics_dir).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def minutes_by_comp_season(rows) -> dict[tuple[int, int], dict[int, float]]:
    """Reshape crosswalk rows into {(competition_id, season_id): {player_id: minutes}}."""
    out: dict[tuple[int, int], dict[int, float]] = {}
    for row in rows:
        out.setdefault((row.competition_id, row.season_id), {})[row.player_id] = row.minutes
    return out
