"""The scout card: one page per player, served by Streamlit.

This module is on the query path, so it is numpy/pandas/altair only. No torch,
no sklearn, no faiss -- see `scout.similarity` for why those three are fatal
here rather than merely slow.

The constraint bites harder in Streamlit than anywhere else in the codebase.
Streamlit re-executes this script top to bottom on *every* widget interaction,
so a module-level `import torch` would not cost 2-4 seconds once, it would cost
2-4 seconds per click and blow the 5-second budget on imports alone. The index
is held in `st.cache_resource` (one copy per server, never re-read) and the
metrics frame in `st.cache_data`, which leaves a rerun doing nothing but a
6,300 x 256 dot product and some pandas.

Every number on the card that is a *projection* or a *value* carries its caveat
in the same visual block, never in a footnote. That is not decoration: scout
measured that the open corpus cannot support a learned cross-competition
projection, and the valuation target is a scarcity proxy rather than money. A
caveat a user has to scroll to find is a caveat that does not exist.

The pure functions below take frames and return frames so they can be tested
without a Streamlit runtime; `render_card` and `main` are the only parts that
touch `st`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from scout import corpus, projection, similarity, valuation

# Eight axes. A radar with twenty spokes is unreadable, and the marginal spoke
# is always the one nobody scouts on. These are the per-90s that separate
# players by role rather than by minutes.
RADAR_COLUMNS: tuple[str, ...] = (
    "xg_per90",
    "xa_per90",
    "shots_per90",
    "key_passes_per90",
    "progressive_passes_per90",
    "progressive_carries_per90",
    "dribbles_completed_per90",
    "duels_won_per90",
)

# Percentile peers must have played enough for a per-90 to mean anything. A
# player with 40 minutes and one shot rates at 2.2 shots per 90 and would sit in
# the top percentile of every finishing axis on the card.
PEER_MIN_MINUTES = 450.0

LABELS = {
    "xg_per90": "xG",
    "xa_per90": "xA",
    "xg_xa_per90": "xG + xA",
    "shots_per90": "Shots",
    "key_passes_per90": "Key passes",
    "progressive_passes_per90": "Prog. passes",
    "progressive_carries_per90": "Prog. carries",
    "progressive_actions_per90": "Prog. actions",
    "dribbles_completed_per90": "Dribbles",
    "duels_won_per90": "Duels won",
    "pressures_per90": "Pressures",
    "interceptions_per90": "Interceptions",
    "ball_recoveries_per90": "Recoveries",
    "passes_per90": "Passes",
    "touches_per90": "Touches",
}

# Above this the top hit is taken as the intended player; below it, or when the
# runner-up is close behind, the user picks. `similarity.resolve` deliberately
# never guesses, and neither does this.
CONFIDENT_SCORE = 95.0
CONFIDENT_MARGIN = 8.0


def label_of(column: str) -> str:
    """Human label for a metric column, falling back to a readable derivation."""
    if column in LABELS:
        return LABELS[column]
    text = column.removesuffix("_per90").replace("_", " ")
    return text[:1].upper() + text[1:]


# --- loading --------------------------------------------------------------


def load_index_and_metrics(
    index_dir: Path = similarity.INDEX_DIR,
    metrics_dir: Path = corpus.METRICS_DIR,
    *,
    cached: bool = False,
) -> tuple[similarity.PlayerIndex, pd.DataFrame]:
    """The two artifacts the card is built from. Raises `similarity.IndexMissing`.

    `cached` routes through Streamlit's caches, which is what the app wants and
    what a test does not: the cache decorators are process-global, so a test that
    loaded a synthetic index through them would poison every later test.
    """
    if cached:
        return _cached_index(str(index_dir)), _cached_metrics(str(metrics_dir))
    return similarity.PlayerIndex.load(index_dir), corpus.load_metrics(metrics_dir)


@st.cache_resource(show_spinner="Loading the player index...")
def _cached_index(index_dir: str) -> similarity.PlayerIndex:
    # cache_resource, not cache_data: the embeddings are memory-mapped and must
    # not be deep-copied on every rerun, which is exactly what cache_data does.
    return similarity.PlayerIndex.load(Path(index_dir))


@st.cache_data(show_spinner="Loading metrics...")
def _cached_metrics(metrics_dir: str) -> pd.DataFrame:
    return corpus.load_metrics(Path(metrics_dir))


def index_missing_message(error: Exception | None = None) -> str:
    """What to tell a user who has no index, instead of a traceback."""
    detail = f"\n\n{error}" if error else ""
    return (
        "No player index found. The card needs one before it can show anything."
        "\n\n"
        "- `scout demo` uses the bundled offline index and is the fastest way to "
        "see the card.\n"
        "- `scout embed` builds one from your own fetched corpus.\n"
        f"{detail}"
    )


# --- percentiles and the radar -------------------------------------------


def player_profiles(
    frame: pd.DataFrame, columns: tuple[str, ...] | list[str] = RADAR_COLUMNS
) -> pd.DataFrame:
    """One row per player: minutes-weighted per-90s across every season they have.

    Weighted by minutes rather than averaged over rows, so a 90-minute cameo in
    a second competition cannot drag a 3,000-minute season halfway toward it.
    """
    present = [c for c in columns if c in frame.columns]
    if frame.empty or "player_id" not in frame.columns:
        return pd.DataFrame(columns=["player_id", "minutes", "gender", *present])

    work = frame[["player_id", "minutes", *present]].copy()
    work["minutes"] = work["minutes"].astype(float).clip(lower=0.0)
    for column in present:
        work[column] = work[column].astype(float).fillna(0.0) * work["minutes"]

    grouped = work.groupby("player_id", as_index=False).sum()
    minutes = grouped["minutes"].replace(0.0, np.nan)
    for column in present:
        grouped[column] = grouped[column] / minutes

    if "gender" in frame.columns:
        # The gender of the season they played most, which is the one the rest
        # of the card is describing.
        ordered = frame.sort_values("minutes")
        genders = ordered.groupby("player_id")["gender"].last()
        grouped["gender"] = grouped["player_id"].map(genders)
    else:
        grouped["gender"] = None
    return grouped


def _percentile(values: np.ndarray, value: float) -> float:
    """Share of the peer pool strictly below `value`, as 0-100.

    Same convention as `similarity.percentiles` so the two cannot disagree about
    what a "70th percentile" means on the same card.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not np.isfinite(value):
        return float("nan")
    return float((finite < value).mean() * 100.0)


def radar_frame(
    index: similarity.PlayerIndex,
    frame: pd.DataFrame,
    player_id: int,
    columns: tuple[str, ...] | list[str] = RADAR_COLUMNS,
    *,
    min_minutes: float = PEER_MIN_MINUTES,
    same_gender: bool = True,
) -> pd.DataFrame:
    """Tidy percentile frame for the radar: one row per axis, bounded 0-100.

    Percentiles rather than raw per-90s, because raw rates are not comparable
    across competitions and a radar of them would mostly plot league scoring
    rates. Peers are restricted to the same gender by default: comparing a
    women's-league per-90 against a men's-league distribution measures the
    competition, not the player.
    """
    columns = list(columns)
    profiles = player_profiles(frame, columns)
    row = profiles[profiles["player_id"] == int(player_id)] if not profiles.empty else profiles

    if row.empty:
        return _radar_from_index(index, player_id, columns)

    pool = profiles[profiles["minutes"] >= min_minutes]
    if same_gender and "gender" in profiles.columns:
        gender = row.iloc[0]["gender"]
        if gender is not None and not pd.isna(gender):
            pool = pool[pool["gender"] == gender]
    restricted = True
    if pool.empty:
        # Better a percentile against everyone than no card at all -- but the
        # fallback drops the minutes floor *and* the same-gender restriction, so
        # it must not be reported under a caption claiming either was applied.
        # `restricted` travels with the frame so the caller states what actually
        # happened rather than what was requested.
        pool = profiles
        restricted = False

    records = []
    for column in columns:
        if column not in profiles.columns:
            continue
        value = float(row.iloc[0][column])
        records.append(
            {
                "metric": column,
                "label": label_of(column),
                "value": value,
                "percentile": _percentile(pool[column].to_numpy(dtype=float), value),
                "peers": int(len(pool)),
                "restricted": restricted,
            }
        )
    return pd.DataFrame(
        records, columns=["metric", "label", "value", "percentile", "peers", "restricted"]
    )


def _radar_from_index(
    index: similarity.PlayerIndex, player_id: int, columns: list[str]
) -> pd.DataFrame:
    """Fallback when the metrics frame has nothing for this player.

    The index carries a few of the same columns, so the card can still show
    something; `similarity.percentiles` silently drops the ones it lacks, which
    is the right behaviour here -- a missing axis is better than a fabricated one.
    """
    scores = similarity.percentiles(index, player_id, columns)
    records = [
        {
            "metric": column,
            "label": label_of(column),
            "value": float(
                index.players.loc[index.players["player_id"] == player_id, column].iloc[0]
            )
            if column in index.players
            else float("nan"),
            "percentile": score,
            "peers": len(index),
        }
        for column, score in scores.items()
    ]
    return pd.DataFrame(records, columns=["metric", "label", "value", "percentile", "peers"])


# --- comparables ----------------------------------------------------------

COMPS_COLUMNS = ("player_id", "name", "similarity", "competition", "season", "minutes", "position")


def comps_frame(
    index: similarity.PlayerIndex,
    player_id: int,
    k: int = 10,
    *,
    min_minutes: float = 0.0,
    position: str | None = None,
) -> pd.DataFrame:
    """The comparables table: nearest neighbours, best first, self excluded."""
    matches = index.similar(int(player_id), k=k, min_minutes=min_minutes, position=position or None)
    return pd.DataFrame(
        [
            {
                "player_id": match.player_id,
                "name": match.name,
                "similarity": match.score,
                "competition": match.competition,
                "season": match.season,
                "minutes": match.minutes,
                "position": match.position,
            }
            for match in matches
        ],
        columns=list(COMPS_COLUMNS),
    )


# --- projection -----------------------------------------------------------


def projection_note() -> str:
    """The standing caveat. Shown beside every projected number, not appended.

    The numbers in here are measured, not rhetorical: `scout.population` counted
    them and `scout.projection` scored the predictors that follow from them.
    """
    return (
        "**This projection is the identity: it is the player's own output, not a "
        "translation.** The open corpus cannot support a learned cross-competition "
        "projection. Only 275 men's and 211 women's players have a genuine "
        "club-to-international move with adequate minutes on both sides, and no "
        "predictor -- a fitted per-league offset or a low-rank map from the "
        "embedding -- beats the identity outside its bootstrap confidence interval. "
        "Age, which a real projection would need, does not exist in open data at "
        'all. Read the number as "what they already do", never as "what they '
        'would do there".'
    )


PROJECTION_COLUMNS = (
    "metric",
    "label",
    "observed",
    "projected",
    "basis",
    "source_league_mean",
    "target_league_mean",
)


def projection_frame(
    frame: pd.DataFrame,
    player_id: int,
    target_competition: str,
    *,
    targets: tuple[str, ...] = projection.TARGETS,
) -> pd.DataFrame:
    """Projected output in `target_competition`, plus the league context.

    `projected` equals `observed` by construction. The two league means are
    shown alongside deliberately *unapplied*: the fitted version of that shift is
    predictor B1, which did not beat the identity, so applying it here would be
    presenting an unresolved difference as a correction. They are context for a
    human, not an adjustment.
    """
    profiles = player_profiles(frame, list(targets))
    row = profiles[profiles["player_id"] == int(player_id)] if not profiles.empty else profiles
    if row.empty:
        return pd.DataFrame(columns=list(PROJECTION_COLUMNS))

    source = source_competition(frame, player_id)
    records = []
    for target in targets:
        if target not in profiles.columns:
            continue
        observed = float(row.iloc[0][target])
        means = projection.league_coefficients(frame, target)
        records.append(
            {
                "metric": target,
                "label": label_of(target),
                "observed": observed,
                "projected": observed,
                "basis": "identity (B0)",
                "source_league_mean": float(means.get(source, float("nan"))),
                "target_league_mean": float(means.get(target_competition, float("nan"))),
            }
        )
    return pd.DataFrame(records, columns=list(PROJECTION_COLUMNS))


def source_competition(frame: pd.DataFrame, player_id: int) -> str:
    """The competition the player has most minutes in -- what a card describes."""
    rows = frame[frame["player_id"] == int(player_id)] if not frame.empty else frame
    if rows.empty:
        return ""
    return str(rows.sort_values("minutes").iloc[-1]["competition"])


# --- value ----------------------------------------------------------------


def value_note(target: str = valuation.TARGET) -> str:
    """The standing caveat for the value gauge. Shown beside the gauge itself."""
    return (
        f"**`{target}` is a scarcity proxy, not money.** It is the minutes a player "
        "went on to play the following season at the same or a higher competition "
        "tier, computed from StatsBomb lineups alone. It is not a fee, a wage or a "
        "market value. Transfermarkt's terms prohibit both automated access and "
        "training on their content, and all of their competitions are men's "
        "football while much of this corpus is not -- modelling price would have "
        "silently dropped every woman in the index. Age and contract length, which "
        "any real valuation needs, are absent from open data. This gauge is "
        "observed history, not a forecast."
    )


@dataclass
class ValueReading:
    player_id: int
    season: str
    minutes_at_tier: float
    percentile: float
    peers: int


VALUE_COLUMNS = ("player_id", "season", "minutes_at_tier")


def value_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Observed next-season-minutes-at-tier for every player-season with a label.

    The gauge shows what actually happened rather than what a model predicts.
    That is partly honesty and partly architecture: the fitted valuation model
    needs sklearn, and importing sklearn would put it on the query path.

    The target definition -- next season's minutes, zeroed if the player dropped
    a tier, and no label at all for a player's final season -- is
    `valuation.build_dataset`'s, and this reproduces it with a shift instead of
    a Python loop over players. That loop costs ~10s on a 6,300-player corpus
    because it materialises a dict per row, which alone would double the app's
    latency budget; here it is ~30ms. `tier_of` and `season_sort_key` are still
    imported rather than restated, so the rule has one home.
    """
    needed = ["player_id", "competition", "season", "minutes"]
    if frame.empty or not set(needed) <= set(frame.columns):
        return pd.DataFrame(columns=list(VALUE_COLUMNS))

    work = frame[needed].copy()
    work["tier"] = work["competition"].map(valuation.tier_of)
    # A stable sort so two competitions inside one season keep a deterministic
    # order; which of them counts as "next" is arbitrary either way, but it must
    # not change between reruns of the same page.
    work = work.sort_values(
        ["player_id", "season"],
        key=lambda column: (
            column.map(valuation.season_sort_key) if column.name == "season" else column
        ),
        kind="stable",
    )

    following = work.groupby("player_id")[["tier", "minutes"]].shift(-1)
    # A player's last season in the corpus has no successor and so no label.
    # Dropping it is deliberate: "we do not know" is not "they played nowhere".
    labelled = following["minutes"].notna()
    dropped_tier = following["tier"] > work["tier"]
    minutes = following["minutes"].where(~dropped_tier, 0.0)

    return pd.DataFrame(
        {
            "player_id": work.loc[labelled, "player_id"].astype(int).to_numpy(),
            "season": work.loc[labelled, "season"].astype(str).to_numpy(),
            "minutes_at_tier": minutes[labelled].astype(float).to_numpy(),
        },
        columns=list(VALUE_COLUMNS),
    )


def value_reading(table: pd.DataFrame, player_id: int) -> ValueReading | None:
    """The player's most recent labelled season, ranked against every other.

    Returns None when the player has no following season in the corpus. That is
    "we do not know", and rendering it as a zeroed gauge would say "they played
    nowhere", which is a different and usually false claim.
    """
    if table.empty:
        return None
    rows = table[table["player_id"] == int(player_id)]
    if rows.empty:
        return None
    latest = rows.sort_values("season", key=lambda s: s.map(valuation.season_sort_key)).iloc[-1]
    values = table["minutes_at_tier"].to_numpy(dtype=float)
    return ValueReading(
        player_id=int(player_id),
        season=str(latest["season"]),
        minutes_at_tier=float(latest["minutes_at_tier"]),
        percentile=_percentile(values, float(latest["minutes_at_tier"])),
        peers=int(len(table)),
    )


# --- name resolution ------------------------------------------------------


def resolve_choices(
    index: similarity.PlayerIndex, query: str, limit: int = 8
) -> list[tuple[int, str, float]]:
    return index.resolve(query.strip(), limit=limit) if query and query.strip() else []


def is_unambiguous(
    choices: list[tuple[int, str, float]],
    *,
    score: float = CONFIDENT_SCORE,
    margin: float = CONFIDENT_MARGIN,
) -> bool:
    """Is the top hit safe to take without asking?

    Two players sharing a surname is the common case, not the edge case, so a
    near-tie is always sent to the pick-list.
    """
    if not choices:
        return False
    if choices[0][2] < score:
        return False
    return len(choices) == 1 or (choices[0][2] - choices[1][2]) >= margin


# --- charts ---------------------------------------------------------------

RADAR_DOMAIN = (0.0, 100.0)


def radar_chart(frame: pd.DataFrame) -> alt.Chart:
    """A radial bar chart of percentiles.

    A radial bar rather than a joined polygon: Vega-Lite has no polar line mark,
    and a polygon over eight arbitrarily-ordered axes encodes an area that means
    nothing anyway -- reorder the spokes and the "shape" of the player changes.
    Bars only claim what is true, which is the length of each spoke.
    """
    return (
        alt.Chart(frame)
        .mark_arc(stroke="#ffffff", strokeWidth=1)
        .encode(
            theta=alt.Theta("label:N").sort(None),
            radius=alt.Radius("percentile:Q").scale(type="linear", domain=list(RADAR_DOMAIN)),
            color=alt.Color("percentile:Q")
            .scale(scheme="blues", domain=list(RADAR_DOMAIN))
            .legend(None),
            tooltip=[
                alt.Tooltip("label:N", title="metric"),
                alt.Tooltip("value:Q", title="per 90", format=".2f"),
                alt.Tooltip("percentile:Q", title="percentile", format=".0f"),
                alt.Tooltip("peers:Q", title="peers"),
            ],
        )
        .properties(height=320)
    )


def gauge_chart(percentile: float, label: str = "percentile") -> alt.Chart:
    """A half-dial for a single 0-100 reading.

    Angles are computed here rather than left to a scale so the empty remainder
    is drawn explicitly: a lone arc with no track reads as "full" at a glance.
    """
    value = 0.0 if not np.isfinite(percentile) else float(min(max(percentile, 0.0), 100.0))
    edge = -math.pi / 2 + math.pi * value / 100.0
    bands = pd.DataFrame(
        [
            {"band": label, "start": -math.pi / 2, "end": edge, "value": value},
            {"band": "remainder", "start": edge, "end": math.pi / 2, "value": 100.0 - value},
        ]
    )
    return (
        alt.Chart(bands)
        .mark_arc(innerRadius=60, outerRadius=100)
        .encode(
            theta=alt.Theta("start:Q").scale(None),
            theta2=alt.Theta2("end:Q"),
            color=alt.Color("band:N")
            .scale(domain=[label, "remainder"], range=["#3b7dd8", "#e6e6e6"])
            .legend(None),
            tooltip=[alt.Tooltip("value:Q", format=".0f")],
        )
        .properties(height=200)
    )


# --- the Streamlit layer --------------------------------------------------


def _player_row(index: similarity.PlayerIndex, player_id: int) -> pd.Series:
    return index.players[index.players["player_id"] == int(player_id)].iloc[0]


def _pick_player(index: similarity.PlayerIndex, query: str) -> int | None:
    """Search box plus pick-list. Returns None until a player is settled on."""
    choices = resolve_choices(index, query)
    if not choices:
        if query.strip():
            st.warning(f"No player matches {query!r}.")
        return None
    if is_unambiguous(choices):
        return choices[0][0]

    st.caption("More than one player matches. Pick one:")
    # Keyed by position, not by label. Two distinct players can share a name,
    # competition and season -- which is precisely the case a pick-list exists
    # to disambiguate -- and a dict keyed on the label would collapse them, show
    # one option, and hand back the other player's card with no indication that
    # a substitution happened.
    options = [
        (
            pid,
            f"{name}  ({_player_row(index, pid)['competition']}, "
            f"{_player_row(index, pid)['season']}, id {pid})",
        )
        for pid, name, _ in choices
    ]
    chosen = st.radio(
        "Matches",
        range(len(options)),
        format_func=lambda i: options[i][1],
        label_visibility="collapsed",
    )
    return options[chosen][0]


def render_card(
    index: similarity.PlayerIndex,
    frame: pd.DataFrame,
    player_id: int,
    *,
    k: int = 10,
    min_minutes: float = 0.0,
    position: str | None = None,
    target_competition: str | None = None,
) -> None:
    """Draw one scout card. Photo omitted -- no open source licenses player images."""
    player = _player_row(index, player_id)
    st.subheader(str(player["name"]))
    st.caption(
        f"{player.get('competition', '')} {player.get('season', '')} - "
        f"{player.get('position') or 'position unknown'} - "
        f"{float(player.get('minutes', 0.0)):,.0f} minutes"
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Percentile profile")
        radar = radar_frame(index, frame, player_id)
        if radar.empty:
            st.info("No metrics for this player, so no percentile profile.")
        else:
            # No `use_container_width`/`width` argument anywhere in this module:
            # Streamlit removed the former and only added the latter in 1.49,
            # and the declared floor is 1.38. Defaults render on both.
            st.altair_chart(radar_chart(radar))
            peers = int(radar["peers"].iloc[0])
            restricted = bool(radar["restricted"].iloc[0]) if "restricted" in radar else True
            if restricted:
                pool_text = (
                    f"against {peers:,} same-gender players with at least "
                    f"{PEER_MIN_MINUTES:,.0f} minutes"
                )
            else:
                pool_text = (
                    f"against all {peers:,} players -- no same-gender or minutes "
                    "filter applied, because no peer cleared them"
                )
            st.caption(
                f"Percentile {pool_text}. Rank, not rate: per-90s are not "
                "comparable across competitions."
            )

    with right:
        st.markdown("#### Value")
        reading = value_reading(value_table(frame), player_id)
        if reading is None:
            st.info(
                "No following season in the corpus for this player, so no reading. "
                "Unknown is not zero."
            )
        else:
            st.altair_chart(gauge_chart(reading.percentile))
            st.metric(
                f"{valuation.TARGET} after {reading.season}",
                f"{reading.minutes_at_tier:,.0f} min",
                f"{reading.percentile:.0f}th percentile of {reading.peers:,}",
                delta_color="off",
            )
        # Adjacent to the number, deliberately. This is the caveat that stops the
        # gauge being read as a price.
        st.warning(value_note())

    st.markdown("#### Comparable players")
    comps = comps_frame(index, player_id, k=k, min_minutes=min_minutes, position=position)
    if comps.empty:
        st.info("No comparable players survive those filters. Loosen them.")
    else:
        st.dataframe(
            comps.style.format({"similarity": "{:.3f}", "minutes": "{:,.0f}"}),
            hide_index=True,
        )

    st.markdown("#### Projection")
    # The caveat comes first here: the table below it is meaningless without it.
    st.warning(projection_note())
    target = target_competition or str(player.get("competition", ""))
    projected = projection_frame(frame, player_id, target)
    if projected.empty:
        st.info("No projection targets available for this player.")
    else:
        st.caption(f"Output carried unchanged into **{target}**.")
        st.dataframe(
            projected.drop(columns=["metric"]).style.format(
                {
                    "observed": "{:.3f}",
                    "projected": "{:.3f}",
                    "source_league_mean": "{:.3f}",
                    "target_league_mean": "{:.3f}",
                }
            ),
            hide_index=True,
        )
        st.caption(
            "The two league means are context, not an adjustment. Applying them is "
            "predictor B1, which does not beat the identity at this sample size."
        )


def main() -> None:
    """Streamlit entry point: `streamlit run` this module's file."""
    st.set_page_config(page_title="scout", page_icon="*", layout="wide")
    st.title("scout")

    try:
        index, frame = load_index_and_metrics(cached=True)
    except similarity.IndexMissing as error:
        # An actionable message, not a traceback. A missing index is the normal
        # state of a fresh clone, not an exceptional one.
        st.error(index_missing_message(error))
        st.stop()
        return

    if frame.empty:
        st.warning(
            "No metrics found under `data/derived/metrics`. Percentiles fall back to "
            "what the index itself carries, and the projection and value panels will "
            "be empty. Run `scout build` to produce them."
        )

    with st.sidebar:
        st.header("Search")
        query = st.text_input("Player name", placeholder="e.g. Kerr")
        st.header("Comparables")
        k = st.slider("How many", min_value=3, max_value=25, value=10)
        min_minutes = st.slider("Minimum minutes", 0, 3000, 450, step=90)
        positions = sorted({str(p) for p in index.players["position"].dropna().unique()})
        position = st.selectbox("Position", ["any", *positions])
        st.header("Projection")
        competitions = sorted({str(c) for c in index.players["competition"].dropna().unique()})
        target_competition = st.selectbox("Target competition", competitions or [""])
        st.caption(
            f"{len(index):,} players - vocabulary {index.meta.get('vocab_fingerprint', '?')}"
        )

    player_id = _pick_player(index, query)
    if player_id is None:
        st.info("Search for a player to build their card.")
        return

    render_card(
        index,
        frame,
        player_id,
        k=k,
        min_minutes=float(min_minutes),
        position=None if position == "any" else position,
        target_competition=target_competition or None,
    )


if __name__ == "__main__":
    main()
