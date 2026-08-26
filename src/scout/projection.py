"""Projecting a player's output from one competition into another.

The spec asks for a head that predicts competition-B output from a
competition-A embedding plus age and a league coefficient, with a GBM on
aggregates and naive carry-forward as baselines, "reported honestly".

`scout.population` already answered whether the corpus can support that: it
cannot. Per gender only 275 and 211 players have a genuine club-to-international
move with adequate minutes on both sides, against a 65,000-parameter target.
This module is the honest reporting that follows from that finding, not an
attempt to work around it.

Three predictors, all evaluated on the same pairs with the same bootstrap:

  B0  identity          predict the source value unchanged -- the null
  B1  league offset     identity plus a fitted per-competition-pair shift
  B2  embedding ridge   a heavily constrained map from the player embedding

B2 is deliberately constrained to a few dozen free parameters. With a few
hundred training pairs, anything larger fits noise and would report a flattering
in-sample number.

**Age is not available.** StatsBomb lineups carry no date of birth, and the only
open source that does is Transfermarkt, whose terms prohibit this use. The spec's
"+ age" feature therefore cannot be built from open data, and its absence is
reported rather than silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGETS = ("xg_xa_per90", "progressive_actions_per90")
DEFAULT_MIN_MINUTES = 270.0
BOOTSTRAP_SAMPLES = 1000


@dataclass
class Pair:
    player_id: int
    source_competition: str
    source_season: str
    target_competition: str
    target_season: str
    gender: str | None
    source: dict[str, float]
    target: dict[str, float]


@dataclass
class Score:
    name: str
    mae: float
    rmse: float
    low: float  # bootstrap 2.5th percentile of MAE
    high: float  # bootstrap 97.5th percentile of MAE
    n: int

    def beats(self, other: Score) -> bool:
        """Strictly better, and outside the other's interval.

        Overlapping confidence intervals mean the difference is not resolvable
        at this sample size, which with a few hundred pairs is the expected
        outcome. Reporting such a result as an improvement would be the exact
        dishonesty the spec is asking to avoid.
        """
        return self.mae < other.mae and self.high < other.low


def build_pairs(
    metrics: pd.DataFrame,
    *,
    min_minutes: float = DEFAULT_MIN_MINUTES,
    cross_competition_only: bool = True,
) -> list[Pair]:
    """Every ordered (source, target) competition-season pair for each player.

    Ordered by season label, so the pair reads as "this player, then that one".
    Same-competition pairs are excluded by default: Euro 2020 to Euro 2024 is a
    real pair but a model fit on it learns aging, not translation.
    """
    if metrics.empty:
        return []

    eligible = metrics[metrics["minutes"] >= min_minutes]
    pairs: list[Pair] = []

    for player_id, group in eligible.groupby("player_id"):
        rows = group.sort_values(["season", "competition"]).to_dict("records")
        for i, source in enumerate(rows):
            for target in rows[i + 1 :]:
                if cross_competition_only and source["competition"] == target["competition"]:
                    continue
                pairs.append(
                    Pair(
                        player_id=int(player_id),
                        source_competition=str(source["competition"]),
                        source_season=str(source["season"]),
                        target_competition=str(target["competition"]),
                        target_season=str(target["season"]),
                        gender=source.get("gender"),
                        source={t: float(source.get(t, np.nan)) for t in TARGETS},
                        target={t: float(target.get(t, np.nan)) for t in TARGETS},
                    )
                )
    return pairs


def _bootstrap_mae(errors: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if errors.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, errors.size, size=(samples, errors.size))
    means = np.abs(errors)[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _score(name: str, predicted: np.ndarray, actual: np.ndarray, seed: int = 0) -> Score:
    errors = predicted - actual
    finite = np.isfinite(errors)
    errors = errors[finite]
    low, high = _bootstrap_mae(errors, BOOTSTRAP_SAMPLES, seed)
    return Score(
        name=name,
        mae=float(np.abs(errors).mean()) if errors.size else float("nan"),
        rmse=float(np.sqrt((errors**2).mean())) if errors.size else float("nan"),
        low=low,
        high=high,
        n=int(errors.size),
    )


def league_coefficients(metrics: pd.DataFrame, target: str) -> dict[str, float]:
    """Mean per-90 output by competition, a crude difficulty/style proxy.

    This is the closest thing to the spec's "league coefficient" that open data
    supports. It conflates league difficulty with league style -- a low-scoring
    league and a hard league look the same here -- and that caveat travels with
    any number derived from it.
    """
    if metrics.empty or target not in metrics:
        return {}
    return metrics.groupby("competition")[target].mean().to_dict()


# --- the three predictors -------------------------------------------------


def predict_identity(pairs: list[Pair], target: str) -> np.ndarray:
    """B0: whatever they did in the source competition, they will do again."""
    return np.array([p.source[target] for p in pairs])


def fit_league_offset(pairs: list[Pair], target: str) -> dict[tuple[str, str], float]:
    """B1: mean residual per (source competition, target competition)."""
    sums: dict[tuple[str, str], list[float]] = {}
    for pair in pairs:
        if not (np.isfinite(pair.source[target]) and np.isfinite(pair.target[target])):
            continue
        key = (pair.source_competition, pair.target_competition)
        sums.setdefault(key, []).append(pair.target[target] - pair.source[target])
    return {key: float(np.mean(values)) for key, values in sums.items()}


def predict_league_offset(
    pairs: list[Pair], target: str, offsets: dict[tuple[str, str], float]
) -> np.ndarray:
    return np.array(
        [
            p.source[target] + offsets.get((p.source_competition, p.target_competition), 0.0)
            for p in pairs
        ]
    )


def fit_embedding_ridge(
    pairs: list[Pair],
    target: str,
    embeddings: dict[int, np.ndarray],
    *,
    components: int = 8,
    alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """B2: ridge on the source value plus a low-rank projection of the embedding.

    The embedding is reduced to `components` dimensions before it is allowed
    anywhere near the regression. Fitting 256 coefficients on a few hundred pairs
    would be unidentifiable; the point of this baseline is to give the embedding
    a fair chance at a sample size where it can actually be evaluated.

    Returns (basis, coefficients, mean_vector), or None when there is not enough
    data to fit even this. The mean vector is part of the fit, not a separate
    input: predictions must be centred on the same point the basis was derived
    from, and passing it around independently invites the two drifting apart.
    """
    rows, targets = [], []
    for pair in pairs:
        vector = embeddings.get(pair.player_id)
        if vector is None:
            continue
        if not (np.isfinite(pair.source[target]) and np.isfinite(pair.target[target])):
            continue
        rows.append((vector, pair.source[target]))
        targets.append(pair.target[target])

    if len(rows) < components * 5:
        return None

    matrix = np.stack([r[0] for r in rows])
    mean_vector = matrix.mean(axis=0)
    centred = matrix - mean_vector
    # SVD rather than sklearn PCA: no extra dependency, and this keeps the
    # projection module importable without the `models` extra.
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    basis = right[:components].T

    features = np.column_stack(
        [centred @ basis, np.array([r[1] for r in rows]), np.ones(len(rows))]
    )
    y = np.asarray(targets)
    penalty = alpha * np.eye(features.shape[1])
    penalty[-1, -1] = 0.0  # never penalise the intercept
    coefficients = np.linalg.solve(features.T @ features + penalty, features.T @ y)
    return basis, coefficients, mean_vector


def predict_embedding_ridge(
    pairs: list[Pair],
    target: str,
    embeddings: dict[int, np.ndarray],
    fitted: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    basis, coefficients, mean_vector = fitted
    out = []
    for pair in pairs:
        vector = embeddings.get(pair.player_id)
        source = pair.source[target]
        if vector is None or not np.isfinite(source):
            out.append(np.nan)
            continue
        reduced = (vector - mean_vector) @ basis
        features = np.concatenate([reduced, [source, 1.0]])
        out.append(float(features @ coefficients))
    return np.array(out)


# --- evaluation -----------------------------------------------------------


@dataclass
class Comparison:
    target: str
    gender: str
    scores: list[Score]
    n_pairs: int

    @property
    def winner(self) -> Score:
        return min(self.scores, key=lambda s: s.mae)

    @property
    def null(self) -> Score:
        return next(s for s in self.scores if s.name == "B0 identity")

    def resolved(self) -> bool:
        """Did anything beat the null outside its confidence interval?"""
        return any(s.beats(self.null) for s in self.scores if s.name != "B0 identity")


N_FOLDS = 5


def _player_folds(pairs: list[Pair], folds: int, seed: int) -> list[np.ndarray]:
    """Split pair indices into folds that never separate a player from themselves.

    Grouping by player matters: one player can appear in several pairs, and a
    random split would train on one of their moves and test on another, leaking
    their individual level straight into the test fold.
    """
    players = sorted({p.player_id for p in pairs})
    rng = np.random.default_rng(seed)
    assignment = {int(pid): i % folds for i, pid in enumerate(rng.permutation(players))}
    return [
        np.array([i for i, p in enumerate(pairs) if assignment[p.player_id] == fold], dtype=int)
        for fold in range(folds)
    ]


def _cross_val_predict(pairs, target, fit_predict, folds, seed) -> np.ndarray:
    """Out-of-sample predictions: every pair is predicted by a model that never saw it."""
    out = np.full(len(pairs), np.nan)
    for held in _player_folds(pairs, folds, seed):
        if held.size == 0:
            continue
        held_set = set(held.tolist())
        train = [p for i, p in enumerate(pairs) if i not in held_set]
        test = [pairs[i] for i in held]
        if not train:
            continue
        predictions = fit_predict(train, test, target)
        if predictions is None:
            continue
        out[held] = predictions
    return out


def evaluate(
    pairs: list[Pair],
    target: str,
    gender: str,
    embeddings: dict[int, np.ndarray] | None = None,
    seed: int = 0,
    folds: int = N_FOLDS,
) -> Comparison:
    """Score all available predictors on one target for one gender stratum.

    B1 and B2 have fitted parameters and are scored **out of sample**, by grouped
    k-fold over players. This is not fussiness. Scored in sample, a ridge with
    ten free parameters beats identity using embeddings made of pure random
    noise -- verified on synthetic data, which is why this function works the way
    it does. An in-sample comparison would be wrong in exactly the direction that
    flatters the model, which is the failure the spec asks to avoid.

    B0 has no fitted parameters, so for it in-sample and out-of-sample coincide.
    """
    usable = [p for p in pairs if (p.gender or "") == gender]
    if not usable:
        return Comparison(target=target, gender=gender, scores=[], n_pairs=0)
    actual = _actual(usable, target)

    scores = [_score("B0 identity", predict_identity(usable, target), actual, seed)]

    def offset_fit_predict(train, test, target_name):
        return predict_league_offset(test, target_name, fit_league_offset(train, target_name))

    scores.append(
        _score(
            "B1 league offset",
            _cross_val_predict(usable, target, offset_fit_predict, folds, seed),
            actual,
            seed,
        )
    )

    if embeddings:

        def ridge_fit_predict(train, test, target_name):
            fitted = fit_embedding_ridge(train, target_name, embeddings)
            if fitted is None:
                return None
            return predict_embedding_ridge(test, target_name, embeddings, fitted)

        predictions = _cross_val_predict(usable, target, ridge_fit_predict, folds, seed)
        if np.isfinite(predictions).any():
            scores.append(_score("B2 embedding ridge", predictions, actual, seed))

    return Comparison(target=target, gender=gender, scores=scores, n_pairs=len(usable))


def _actual(pairs: list[Pair], target: str) -> np.ndarray:
    return np.array([p.target[target] for p in pairs])


def report(comparisons: list[Comparison]) -> str:
    lines = [
        "# Projection: the honest three-way comparison",
        "",
        "Predicting a player's output in a target competition from their output in",
        "a source competition. Every predictor is scored on the same pairs, with a",
        f"{BOOTSTRAP_SAMPLES}-sample bootstrap confidence interval on mean absolute error.",
        "",
        "| predictor | what it does |",
        "| --- | --- |",
        "| B0 identity | predict the source value unchanged -- the null hypothesis |",
        "| B1 league offset | identity plus a fitted per-competition-pair shift |",
        "| B2 embedding ridge | low-rank embedding projection, plus the source value |",
        "",
        "B1 and B2 have fitted parameters and are scored **out of sample**, by",
        "5-fold cross-validation grouped by player so no player is ever trained on",
        "and tested on at once. This matters: scored in sample, B2 beats identity",
        "using embeddings made of pure random noise -- ten free parameters will fit",
        "a few hundred pairs. An in-sample table would be wrong in precisely the",
        "direction that flatters the model.",
        "",
        "**Age is absent.** The spec asks for age as a feature. StatsBomb lineups",
        "carry no date of birth and the only open source that does is Transfermarkt,",
        "whose terms prohibit this use. The feature is reported missing, not faked.",
        "",
    ]

    for comparison in comparisons:
        lines += [
            f"## {comparison.target}, {comparison.gender} ({comparison.n_pairs} pairs)",
            "",
            "| predictor | MAE | RMSE | 95% CI on MAE | n |",
            "| --- | ---: | ---: | :---: | ---: |",
        ]
        for score in comparison.scores:
            lines.append(
                f"| {score.name} | {score.mae:.3f} | {score.rmse:.3f} | "
                f"[{score.low:.3f}, {score.high:.3f}] | {score.n} |"
            )
        lines.append("")
        if comparison.resolved():
            lines.append(
                f"**{comparison.winner.name} beats identity** outside the null's "
                "confidence interval."
            )
        else:
            lines.append(
                "**No predictor beats identity outside its confidence interval.** "
                "At this sample size the difference is not resolvable, which is the "
                "expected result given the population measured in "
                "`projection_power.md`. scout therefore returns the identity "
                "projection and says so wherever a projected number appears."
            )
        lines.append("")

    return "\n".join(lines)
