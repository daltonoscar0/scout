"""Valuation: what is a player worth, and is the market wrong about them?

The spec asks to regress Transfermarkt market value on the embedding, age,
contract and league, and to treat the residual as a mispricing signal.

That cannot ship as written, for two reasons that are not technical:

  1. Transfermarkt's terms (clause 11.1) prohibit automated access **and**
     prohibit using the content to train machine learning models, by name. The
     spec also asks to commit the snapshot into `data/`, which would make this
     repo a redistribution point for scraped commercial data covering ~50,000
     identifiable people's dates of birth and home cities.
  2. Every one of Transfermarkt's competitions is men's football. A large part
     of this corpus is not. Those players would have no target variable and
     would silently vanish from the output.

So the shipped default target is **next-season minutes at tier**: how many
minutes a player goes on to play the following season, at the same or a higher
competition tier, computed entirely from StatsBomb lineups. It is legally clean,
it covers women's football, and it has thousands of labels.

It is a scarcity and trust proxy, not a price, and nothing in this module
pretends otherwise. Market value is available behind `--enable-transfermarkt`,
which fetches at build time, never commits, and prints the terms notice first.

Age and contract length, both of which the spec wants as features, are not
available from open data at all. They are reported missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scout import caveats

# Competition tiers, coarse and deliberately explicit. Used to decide whether a
# player's following season was a step up, sideways, or down. Lower is better.
# Tiering leagues is inherently arguable -- this ordering is a defensible default
# and is stated here so anyone who disagrees can see and change it, rather than
# it being buried in a model.
COMPETITION_TIER = {
    "FIFA World Cup": 0,
    "Women's World Cup": 0,
    "UEFA Euro": 0,
    "UEFA Women's Euro": 0,
    "Copa America": 0,
    "African Cup of Nations": 1,
    "Champions League": 1,
    "Premier League": 1,
    "La Liga": 1,
    "Serie A": 1,
    "1. Bundesliga": 1,
    "Ligue 1": 1,
    "UEFA Europa League": 2,
    "FA Women's Super League": 2,
    "NWSL": 2,
    "Liga F": 2,
    "Frauen Bundesliga": 2,
    "Serie A Women": 2,
    "Copa del Rey": 2,
    "Major League Soccer": 3,
    "Liga Profesional": 3,
    "Indian Super league": 4,
    "North American League": 4,
    "FIFA U20 World Cup": 4,
}
DEFAULT_TIER = 3

# One definition, shared with the CLI, the app and the Excel export. Two
# independent spellings of the same target string is how the caveats drifted
# the first time.
TARGET = caveats.VALUATION_TARGET

FEATURE_COLUMNS = (
    "minutes",
    "matches",
    "xg_per90",
    "xa_per90",
    "shots_per90",
    "progressive_passes_per90",
    "progressive_carries_per90",
    "passes_per90",
    "key_passes_per90",
    "pressures_per90",
    "duels_won_per90",
    "interceptions_per90",
    "ball_recoveries_per90",
    "dribbles_completed_per90",
    "touches_per90",
    "pass_completion",
    "tier",
)

# Features the spec asks for that open data cannot supply. Named so the gap is
# visible in the output rather than being quietly absent.
MISSING_FEATURES = {
    "age": "StatsBomb lineups carry no date of birth.",
    "contract_length": "No open source covers contracts.",
    "market_value": "Transfermarkt's terms prohibit this use; opt in with --enable-transfermarkt.",
}


def tier_of(competition: str) -> int:
    return COMPETITION_TIER.get(competition, DEFAULT_TIER)


def season_sort_key(season: str) -> tuple:
    """Order season labels that come in two shapes: '2015/2016' and '2023'."""
    text = str(season)
    head = text.split("/")[0]
    try:
        return (int(head), text)
    except ValueError:
        return (0, text)


@dataclass
class Dataset:
    features: pd.DataFrame
    target: np.ndarray
    seasons: list[str]
    player_ids: np.ndarray
    # Years between a row's season and the season its label came from. In a
    # dense corpus this is always 1; here it is not, and that turns out to
    # matter more than any modelling choice.
    gap_years: np.ndarray = field(default_factory=lambda: np.array([]))
    unmatched: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.target)


def build_dataset(metrics: pd.DataFrame) -> Dataset:
    """Join each player-season to what the player did the *following* season.

    A row is (this season's production) -> (next season's minutes at the same or
    a better tier). Players whose final season is the last one in the corpus have
    no label and are excluded, not zero-filled: "we do not know" and "they played
    nowhere" are different statements and conflating them would teach the model
    that every player's career ends.
    """
    if metrics.empty:
        return Dataset(pd.DataFrame(), np.array([]), [], np.array([]))

    # Club football only. Tiering a World Cup against a league is a category
    # error: a national-team tournament is not a rung on the club ladder, and
    # treating it as the top rung made every club season *following* a
    # tournament score zero, as if the player had dropped down. That labelled
    # 22% of the held-out rows as collapses that never happened, against 1% in
    # training -- a fabricated distribution shift that no model could fit and
    # that would have been reported as "the data has no signal".
    from scout.population import is_international

    frame = metrics[~metrics["competition"].map(is_international)].copy()
    if frame.empty:
        return Dataset(pd.DataFrame(), np.array([]), [], np.array([]))

    frame["tier"] = frame["competition"].map(tier_of)
    frame["_season_key"] = frame["season"].map(season_sort_key)

    rows, targets, seasons, ids, gaps = [], [], [], [], []
    for player_id, group in frame.groupby("player_id"):
        ordered = group.sort_values("_season_key").to_dict("records")
        for index, current in enumerate(ordered[:-1]):
            later = ordered[index + 1 :]
            gaps.append(int(later[0]["_season_key"][0]) - int(current["_season_key"][0]))
            # Minutes in the next season the player appears in, counted only if
            # it was at the same tier or better.
            following = later[0]
            # A drop in tier scores zero: the target is minutes *at tier*, so
            # falling to a weaker competition is not the same as holding a place.
            minutes = 0.0 if following["tier"] > current["tier"] else float(following["minutes"])
            rows.append(current)
            targets.append(minutes)
            seasons.append(str(current["season"]))
            ids.append(int(player_id))

    if not rows:
        return Dataset(pd.DataFrame(), np.array([]), [], np.array([]))

    features = pd.DataFrame(rows)
    available = [c for c in FEATURE_COLUMNS if c in features.columns]
    return Dataset(
        features=features[available].astype(float).fillna(0.0),
        target=np.asarray(targets, dtype=float),
        seasons=seasons,
        player_ids=np.asarray(ids, dtype=int),
        gap_years=np.asarray(gaps, dtype=int),
    )


def split_by_season(
    dataset: Dataset, holdout_fraction: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the most recent seasons, never random rows.

    A random split puts a player's 2015/16 row in train and their 2016/17 row in
    test, which leaks. Splitting on time also matches how the model would
    actually be used: fit on the past, applied to the present.
    """
    unique = sorted(set(dataset.seasons), key=season_sort_key)
    n_held = max(1, int(len(unique) * holdout_fraction))
    held = set(unique[-n_held:])
    mask = np.array([s in held for s in dataset.seasons])
    return ~mask, mask


@dataclass
class ValuationModel:
    model: object
    feature_names: list[str]
    baseline: float
    metrics: dict

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        aligned = features[self.feature_names].astype(float).fillna(0.0)
        return np.asarray(self.model.predict(aligned))


def fit(dataset: Dataset, seed: int = 0) -> ValuationModel:
    """Fit a gradient-boosted regressor, validated on held-out seasons.

    sklearn's HistGradientBoostingRegressor rather than LightGBM. LightGBM links
    Homebrew's libomp by rpath and fails at import on a clean machine or CI
    runner; sklearn is already a dependency of the `models` extra and vendors its
    own OpenMP. The accuracy difference on a few thousand rows is not worth an
    install-time failure a user cannot diagnose.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    train_mask, test_mask = split_by_season(dataset)
    features = dataset.features
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=seed)
    model.fit(features[train_mask], dataset.target[train_mask])

    predicted = model.predict(features[test_mask])
    actual = dataset.target[test_mask]
    # The baseline any model must beat: predict the training mean for everyone.
    baseline_prediction = float(dataset.target[train_mask].mean())

    def mae(p, a):
        return float(np.abs(np.asarray(p) - a).mean()) if len(a) else float("nan")

    return ValuationModel(
        model=model,
        feature_names=list(features.columns),
        baseline=baseline_prediction,
        metrics={
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "mae": mae(predicted, actual),
            "baseline_mae": mae(np.full(actual.shape, baseline_prediction), actual),
            "spearman": _spearman(predicted, actual),
        },
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without pulling in scipy."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")
    rank_a = pd.Series(a).rank().to_numpy()
    rank_b = pd.Series(b).rank().to_numpy()
    if rank_a.std() == 0 or rank_b.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def residuals(model: ValuationModel, dataset: Dataset) -> pd.DataFrame:
    """Model-implied value against observed, as a mispricing signal.

    A positive residual means the player went on to play more than their
    production predicted -- the model under-rated them.
    """
    predicted = model.predict(dataset.features)
    return pd.DataFrame(
        {
            "player_id": dataset.player_ids,
            "season": dataset.seasons,
            "predicted": predicted,
            "actual": dataset.target,
            "residual": dataset.target - predicted,
        }
    )


def residual_predicts_next_window(frame: pd.DataFrame) -> tuple[float, str]:
    """Does this season's residual predict the next season's residual?

    The spec asks this question out loud and requires the answer in the README
    "even if it is no". This is the out-of-sample check: if mispricing is a real
    signal, a player under-rated once should tend to be under-rated again. If the
    correlation is near zero, the residual is noise and saying so is the result.
    """
    ordered = frame.copy()
    ordered["_key"] = ordered["season"].map(season_sort_key)
    ordered = ordered.sort_values(["player_id", "_key"])

    current, following = [], []
    for _, group in ordered.groupby("player_id"):
        values = group["residual"].tolist()
        for a, b in zip(values[:-1], values[1:], strict=False):
            current.append(a)
            following.append(b)

    if len(current) < 30:
        return float("nan"), (
            f"only {len(current)} consecutive residual pairs, too few to answer this. "
            "Reported as unknown rather than as a null result."
        )

    correlation = _spearman(np.array(current), np.array(following))
    if not np.isfinite(correlation):
        return correlation, "the correlation is undefined; treat the residual as noise."
    if abs(correlation) < 0.1:
        return correlation, (
            f"no. Spearman {correlation:+.3f} over {len(current)} pairs: this season's "
            "residual carries essentially no information about next season's. The "
            "mispricing signal does not persist and should not be traded on."
        )
    return correlation, (
        f"yes, weakly. Spearman {correlation:+.3f} over {len(current)} pairs. The "
        "residual persists somewhat between windows, but this is a rank correlation "
        "on a proxy target, not evidence of a tradable edge."
    )


def coverage_note(dataset: Dataset) -> str:
    """How often the label really is the *next* season.

    Open StatsBomb coverage is sparse and non-contiguous: a player's next
    appearance in the corpus can be years after their last. Where that gap is
    large the label stops meaning "next season" and starts meaning "whenever we
    next happened to observe them", which is not something this season's
    production can be expected to predict.
    """
    if dataset.gap_years.size == 0:
        return "No gap information recorded."
    consecutive = float((dataset.gap_years <= 1).mean())
    return (
        f"Only {consecutive:.1%} of rows are labelled by a genuinely consecutive "
        f"season. For the rest the gap is 2 to {int(dataset.gap_years.max())} years, "
        "because open coverage of any one competition is intermittent. Where the "
        'gap is large the label means "whenever we next observed them", not "next '
        "season\", and this season's production should not be expected to predict it."
    )


def report(
    model: ValuationModel, answer: str, correlation: float, dataset: Dataset | None = None
) -> str:
    improvement = model.metrics["baseline_mae"] - model.metrics["mae"]
    verdict = (
        f"beats the predict-the-mean baseline by {improvement:.0f} minutes MAE"
        if improvement > 0
        else "DOES NOT beat predicting the training mean"
    )
    lines = [
        "# Valuation",
        "",
        f"Target: **{TARGET}** -- minutes played the following season at the same or a",
        "higher competition tier, computed from StatsBomb lineups alone.",
        "",
        "This is not money. The spec asks to model Transfermarkt market value, but",
        "Transfermarkt's terms prohibit automated access and prohibit training models",
        "on their content by name, and all 65 of their competitions are men's football",
        "while a large part of this corpus is not. A minutes-at-tier target is legally",
        "clean, covers everyone, and is a genuine scarcity proxy. Market value is",
        "available behind `--enable-transfermarkt`, fetched at build time and never",
        "committed.",
        "",
        "## Features the spec asks for that open data cannot supply",
        "",
        "| feature | why not |",
        "| --- | --- |",
    ]
    for name, reason in MISSING_FEATURES.items():
        lines.append(f"| {name} | {reason} |")

    lines += [
        "",
        "## Fit",
        "",
        "Validated on held-out **seasons**, not random rows: a random split puts a",
        "player's 2015/16 row in train and their 2016/17 row in test.",
        "",
        "| | |",
        "| --- | ---: |",
        f"| training rows | {model.metrics['n_train']:,} |",
        f"| held-out rows | {model.metrics['n_test']:,} |",
        f"| MAE (minutes) | {model.metrics['mae']:.1f} |",
        f"| baseline MAE (predict the mean) | {model.metrics['baseline_mae']:.1f} |",
        f"| rank correlation with actual | {model.metrics['spearman']:+.3f} |",
        "",
        f"The model {verdict}.",
        "",
        "## Why the fit looks like this",
        "",
        (coverage_note(dataset) if dataset is not None else ""),
        "",
        "## Does the residual predict the next window?",
        "",
        "The spec requires this answer in the README even if it is no.",
        "",
        f"**{answer}**",
        "",
    ]
    return "\n".join(lines)
