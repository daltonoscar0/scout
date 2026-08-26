"""Training loop for the possession LM.

Karpathy-shaped and deliberately plain: load shards into one int16 array, split
held-out sequences by match, iterate shuffled batches, AdamW with warmup and
cosine decay, checkpoint every epoch.

Two things here are not boilerplate:

  * The held-out split is by match_id, never by row. Two sequences from the same
    match share possessions, teams and opponents, so a random row split leaks
    and the reported perplexity would be optimistic for reasons that have
    nothing to do with the model.

  * Perplexity is reported against a bigram baseline over the action field. An
    absolute perplexity number on a bespoke vocabulary means nothing -- there is
    nothing to compare it to. If the transformer does not clearly beat a bigram,
    the representation is not worth building embeddings on, and that is worth
    finding out after one epoch rather than after six hours.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scout import corpus, vocab
from scout.model import LOSS_WEIGHTS, ModelConfig, PossessionLM, cosine_schedule, loss

MODELS_DIR = Path("models")


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 96
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    held_out_fraction: float = 0.05
    seed: int = 0
    max_events: int | None = None  # cap the corpus; None uses everything


@dataclass
class Dataset:
    tokens: np.ndarray  # (n, seq, 5) int16
    lengths: np.ndarray
    match_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


def load_corpus(
    tokens_dir: Path = corpus.TOKENS_DIR,
    max_events: int | None = None,
    seed: int = 0,
) -> Dataset:
    """Load every shard, then subsample uniformly if a cap is set.

    The cap subsamples *sequences at random across the whole corpus*. Reading
    shards until the budget fills would be simpler and badly wrong: shards are
    named by competition id, so a 3M-event cap would train on the handful of
    competitions that sort first and never see the rest. The corpus spans men's
    and women's football, 1958 to 2025 and eight countries -- a truncation is not
    a sample of it.
    """
    shards = sorted(tokens_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"no token shards in {tokens_dir}. Run `scout build` first.")

    token_blocks, length_blocks, match_blocks = [], [], []
    for path in shards:
        shard = corpus.load_shard(path)  # raises on a vocabulary mismatch
        token_blocks.append(shard.tokens)
        length_blocks.append(shard.lengths)
        match_blocks.append(shard.match_ids)

    tokens = np.concatenate(token_blocks)
    lengths = np.concatenate(length_blocks)
    match_ids = np.concatenate(match_blocks)

    total = int(lengths.sum())
    if max_events is None or total <= max_events:
        return Dataset(tokens=tokens, lengths=lengths, match_ids=match_ids)

    # Sample whole sequences, in a stable order, until the token budget is met.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lengths))
    running = np.cumsum(lengths[order])
    keep = order[: int(np.searchsorted(running, max_events) + 1)]
    keep.sort()
    print(
        f"corpus capped: {len(keep):,} of {len(lengths):,} sequences "
        f"({int(lengths[keep].sum()):,} of {total:,} tokens), sampled across all "
        f"{len(shards)} competition-seasons",
        flush=True,
    )
    return Dataset(tokens=tokens[keep], lengths=lengths[keep], match_ids=match_ids[keep])


def split_by_match(data: Dataset, fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    """Hold out whole matches, never individual sequences."""
    rng = np.random.default_rng(seed)
    matches = np.unique(data.match_ids)
    rng.shuffle(matches)
    n_held = max(1, int(len(matches) * fraction))
    held = set(matches[:n_held].tolist())
    mask = np.array([m in held for m in data.match_ids])
    return (
        Dataset(data.tokens[~mask], data.lengths[~mask], data.match_ids[~mask]),
        Dataset(data.tokens[mask], data.lengths[mask], data.match_ids[mask]),
    )


def mask_for(lengths: np.ndarray, seq_len: int) -> np.ndarray:
    positions = np.arange(seq_len)[None, :]
    return (positions < lengths[:, None]).astype(np.float32)


def bigram_perplexity(train: Dataset, held: Dataset) -> float:
    """Perplexity of a add-one-smoothed bigram over the action field.

    The number the transformer has to beat. Anything that does not is a more
    expensive way of counting pairs.
    """
    counts: dict[int, Counter] = {}
    for row, length in zip(train.tokens, train.lengths, strict=True):
        actions = row[:length, 0]
        for previous, current in zip(actions[:-1], actions[1:], strict=True):
            counts.setdefault(int(previous), Counter())[int(current)] += 1

    vocab_size = vocab.ACTION_SIZE
    total_log_prob = 0.0
    n = 0
    for row, length in zip(held.tokens, held.lengths, strict=True):
        actions = row[:length, 0]
        for previous, current in zip(actions[:-1], actions[1:], strict=True):
            table = counts.get(int(previous))
            numerator = (table.get(int(current), 0) if table else 0) + 1
            denominator = (sum(table.values()) if table else 0) + vocab_size
            total_log_prob += float(np.log(numerator / denominator))
            n += 1
    return float(np.exp(-total_log_prob / max(n, 1)))


@torch.no_grad()
def evaluate(model: PossessionLM, data: Dataset, device: torch.device, batch_size: int) -> dict:
    model.eval()
    seq_len = data.tokens.shape[1]
    masks = mask_for(data.lengths, seq_len)
    totals: dict[str, float] = {}
    weight_total = 0.0

    for start in range(0, len(data), batch_size):
        tokens = torch.from_numpy(data.tokens[start : start + batch_size].astype(np.int64)).to(
            device
        )
        mask = torch.from_numpy(masks[start : start + batch_size]).to(device)
        _, parts = loss(model(tokens), tokens, mask)
        weight = float(mask.sum())
        weight_total += weight
        for key, value in parts.items():
            if key != "tokens":
                totals[key] = totals.get(key, 0.0) + value * weight

    model.train()
    out = {key: value / max(weight_total, 1.0) for key, value in totals.items()}
    out["action_perplexity"] = float(np.exp(out.get("action", 0.0)))
    return out


def train(
    config: TrainConfig | None = None,
    model_config: ModelConfig | None = None,
    tokens_dir: Path = corpus.TOKENS_DIR,
    out_dir: Path = MODELS_DIR,
    device_preference: str = "auto",
) -> Path:
    from scout.model import select_device

    config = config or TrainConfig()
    model_config = model_config or ModelConfig.small()
    torch.manual_seed(config.seed)

    data = load_corpus(tokens_dir, config.max_events)
    corpus_note = (
        f"Pre-training used a {int(data.lengths.sum()):,}-token sample of the corpus, "
        f"drawn uniformly across every competition-season "
        f"(cap: {config.max_events:,} events)."
        if config.max_events
        else f"Pre-training used the whole corpus: {int(data.lengths.sum()):,} tokens."
    )
    train_set, held_set = split_by_match(data, config.held_out_fraction, config.seed)
    print(
        f"sequences: {len(train_set):,} train / {len(held_set):,} held out "
        f"({len(np.unique(held_set.match_ids)):,} matches held back)",
        flush=True,
    )

    baseline = bigram_perplexity(train_set, held_set)
    print(f"bigram baseline action perplexity: {baseline:.2f}", flush=True)

    device = select_device(device_preference)
    model = PossessionLM(model_config).to(device)
    print(model.describe(), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    seq_len = train_set.tokens.shape[1]
    train_masks = mask_for(train_set.lengths, seq_len)
    steps_per_epoch = max(1, len(train_set) // config.batch_size)
    total_steps = steps_per_epoch * config.epochs

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    best = float("inf")
    best_path = out_dir / "possession_lm.pt"
    step = 0
    started = time.time()

    for epoch in range(config.epochs):
        order = rng.permutation(len(train_set))
        running = 0.0
        for index in range(steps_per_epoch):
            rows = order[index * config.batch_size : (index + 1) * config.batch_size]
            tokens = torch.from_numpy(train_set.tokens[rows].astype(np.int64)).to(device)
            mask = torch.from_numpy(train_masks[rows]).to(device)

            learning_rate = cosine_schedule(
                step, warmup=config.warmup_steps, total=total_steps, peak=config.learning_rate
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            value, _ = loss(model(tokens), tokens, mask)
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running += float(value.detach())
            step += 1

        metrics = evaluate(model, held_set, device, config.batch_size)
        elapsed = (time.time() - started) / 60
        print(
            f"epoch {epoch + 1:>3}/{config.epochs}  train {running / steps_per_epoch:.4f}  "
            f"held-out action ppl {metrics['action_perplexity']:.2f}  "
            f"({metrics['action_perplexity'] / baseline:.2f}x bigram)  {elapsed:.1f} min",
            flush=True,
        )

        if metrics["action_perplexity"] < best:
            best = metrics["action_perplexity"]
            save_checkpoint(best_path, model, config, metrics, baseline, epoch)

    write_model_card(
        out_dir, model, config, best, baseline, (time.time() - started) / 60, corpus_note
    )
    print(f"\nbest held-out action perplexity {best:.2f} vs bigram {baseline:.2f}", flush=True)
    if best >= baseline:
        print(
            "WARNING: the transformer did not beat a bigram. The representation is "
            "not worth building embeddings on -- investigate before running `scout embed`.",
            flush=True,
        )
    return best_path


def save_checkpoint(
    path: Path,
    model: PossessionLM,
    config: TrainConfig,
    metrics: dict,
    baseline: float,
    epoch: int,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model.config_dict(),
            "train_config": vars(config),
            "metrics": metrics,
            "bigram_baseline": baseline,
            "epoch": epoch,
            "vocab_fingerprint": vocab.vocabulary_fingerprint(),
            "vocab_version": vocab.VOCAB_VERSION,
        },
        path,
    )


def write_model_card(
    out_dir: Path,
    model: PossessionLM,
    config: TrainConfig,
    best: float,
    baseline: float,
    minutes: float,
    corpus_note: str = "full corpus",
) -> None:
    verdict = (
        f"beats the bigram baseline by {baseline / best:.2f}x"
        if best < baseline
        else "DOES NOT beat a bigram baseline -- do not build embeddings on this"
    )
    card = f"""# possession-lm

Next-event model over factorized StatsBomb possession tokens. Trained to produce
player embeddings, not to be a good forecaster in its own right.

## Architecture
{model.describe()}

Five embedding tables ({sum(vocab.FIELD_SIZES)} rows total) summed before the first
block; five prediction heads with loss weights {list(LOSS_WEIGHTS)}.

## Vocabulary
version {vocab.VOCAB_VERSION}, fingerprint `{vocab.vocabulary_fingerprint()}`.
Checkpoints are only valid against this fingerprint.

## Training
epochs {config.epochs}, batch {config.batch_size}, lr {config.learning_rate}
(warmup {config.warmup_steps}, cosine decay), weight decay {config.weight_decay},
grad clip {config.grad_clip}. Held-out split is {config.held_out_fraction:.0%} of
*matches*, never of rows -- sequences from one match share possessions and
opponents, so a row split leaks.
Wall clock: {minutes:.1f} minutes.

## Corpus
{corpus_note}

Player embeddings are computed over the **entire** corpus regardless of this
cap; it constrains only what the language model saw during pre-training.

## Result
held-out action perplexity {best:.2f}, bigram baseline {baseline:.2f}: {verdict}.

An absolute perplexity on a bespoke vocabulary is uninterpretable, which is why
the baseline is reported alongside it.

## Limitations
- Off-ball 360 freeze-frame players are excluded. They carry no player id, so
  they cannot be attributed to named teammates.
- Trained on open StatsBomb data only: heavily weighted toward international
  tournaments and a handful of league seasons.
"""
    (out_dir / "MODEL_CARD.md").write_text(card)
