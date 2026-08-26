"""How much supervision does the open corpus actually contain for projection?

The spec asks for a projection head: given a player's representation in
competition A, predict their output in competition B, trained on players who
appear in both. Whether that is a model or wishful thinking is an empirical
question about the corpus, and this module answers it before any model code runs.

The distinction that matters is between *pair count* and *useful pair count*:

  * A named substitute who never came on is in the lineup and emits no events.
  * Two editions of the same competition (Euro 2020 to Euro 2024) are a pair,
    but a head fit on them learns "same tournament, four years older" -- an
    aging model, not a domain map.
  * A club-to-international move is the only genuine domain shift in this
    corpus, and it is also the rarest.

So the counts are reported decomposed, not as one headline number.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

from scout.crosswalk import PlayerSeason

# National-team competitions. Everything else -- including the Champions League,
# Europa League and Copa del Rey -- is club football: a player's team, teammates
# and league context carry over, so it is not a domain shift in the sense the
# projection head needs.
INTERNATIONAL = frozenset(
    {
        "African Cup of Nations",
        "Copa America",
        "FIFA U20 World Cup",
        "FIFA World Cup",
        "UEFA Euro",
        "UEFA Women's Euro",
        "Women's World Cup",
    }
)


def is_international(competition: str) -> bool:
    return competition in INTERNATIONAL


@dataclass
class PairCounts:
    total: int = 0
    same_competition: int = 0
    cross_league: int = 0
    club_to_international: int = 0
    cross_gender: int = 0


@dataclass
class Population:
    players: int
    rows: int
    rows_with_minutes: int
    multi: dict[float, int]
    pairs: dict[float, PairCounts]
    league_pair_counts: Counter
    per_gender: dict[str, int]


def analyse(
    rows: list[PlayerSeason], floors: tuple[float, ...] = (0, 90, 270, 450, 900)
) -> Population:
    by_player: dict[int, list[PlayerSeason]] = defaultdict(list)
    for row in rows:
        by_player[row.player_id].append(row)

    multi: dict[float, int] = {}
    pairs: dict[float, PairCounts] = {}
    league_pairs: Counter = Counter()
    per_gender: dict[str, int] = {}

    for floor in floors:
        counts = PairCounts()
        qualifying = 0
        for seasons in by_player.values():
            eligible = [s for s in seasons if s.minutes >= floor]
            if len(eligible) < 2:
                continue
            qualifying += 1
            for a, b in combinations(eligible, 2):
                counts.total += 1
                if a.competition == b.competition:
                    counts.same_competition += 1
                    continue
                a_intl, b_intl = is_international(a.competition), is_international(b.competition)
                if a_intl != b_intl:
                    counts.club_to_international += 1
                elif not a_intl and not b_intl:
                    counts.cross_league += 1
                    if floor == 270:
                        league_pairs[tuple(sorted((a.competition, b.competition)))] += 1
                if (a.gender or "") != (b.gender or ""):
                    counts.cross_gender += 1
        multi[floor] = qualifying
        pairs[floor] = counts

    # Gender split of the club-to-international population at the 270 floor,
    # because a single head across both is unsupported: essentially no player
    # appears in both men's and women's competitions.
    for gender in ("male", "female"):
        n = 0
        for seasons in by_player.values():
            eligible = [s for s in seasons if s.minutes >= 270 and s.gender == gender]
            if len(eligible) < 2:
                continue
            if any(
                is_international(a.competition) != is_international(b.competition)
                for a, b in combinations(eligible, 2)
            ):
                n += 1
        per_gender[gender] = n

    return Population(
        players=len(by_player),
        rows=len(rows),
        rows_with_minutes=sum(1 for r in rows if r.seconds > 0),
        multi=multi,
        pairs=pairs,
        league_pair_counts=league_pairs,
        per_gender=per_gender,
    )


def verdict(population: Population, threshold: int = 300) -> tuple[bool, str]:
    """Is there enough supervision to fit a cross-domain projection head?

    Assessed *per gender*, not on the pooled total. Essentially no player appears
    in both men's and women's competitions, so a single head across both is
    unsupported -- which means the pooled count is not a sample size any model
    ever gets to use. Summing the two and comparing that to the threshold would
    be the same mistake as counting named substitutes: a bigger number that no
    estimator can spend.

    `threshold` is the count below which a 256-to-256 map -- 65k parameters --
    or anything much beyond a handful of free parameters is not identifiable.
    """
    usable = {gender: count for gender, count in population.per_gender.items() if count}
    passing = {g: n for g, n in usable.items() if n >= threshold}
    detail = ", ".join(f"{g} {n}" for g, n in sorted(usable.items()))

    if not passing:
        return False, (
            f"no gender stratum reaches the {threshold} players needed to identify even a "
            f"heavily constrained cross-domain map ({detail}). A learned projection is not "
            "supported by this data. Ship identity as the baseline and say so in the README "
            "and next to every projected number in the UI."
        )
    if len(passing) < len(usable):
        return False, (
            f"only {', '.join(sorted(passing))} reaches the {threshold}-player bar ({detail}), "
            "so a projection would ship for one gender and not the other. Treat the module as "
            "a reported null result rather than shipping an asymmetric model."
        )
    return True, (
        f"every gender stratum clears {threshold} players with a genuine club-to-international "
        f"move at 270+ minutes per side ({detail}). Proceed, but constrain the map to a few "
        "dozen free parameters and report bootstrap CIs against identity."
    )


def report(population: Population) -> str:
    ok, message = verdict(population)
    lines = [
        "# Projection power",
        "",
        "Can a cross-competition projection head be trained on the open corpus?",
        "Measured over every lineup in StatsBomb open data, not sampled.",
        "",
        f"- player-season rows: {population.rows:,}",
        f"- rows with more than zero minutes: {population.rows_with_minutes:,} "
        f"({100 * population.rows_with_minutes / max(population.rows, 1):.1f}%)",
        f"- distinct players: {population.players:,}",
        "",
        "## Players appearing in two or more competition-seasons",
        "",
        "| minutes floor, each side | players |",
        "| --- | ---: |",
    ]
    for floor, count in population.multi.items():
        label = "any (squad membership)" if floor == 0 else f">= {floor:.0f} min"
        lines.append(f"| {label} | {count:,} |")

    lines += [
        "",
        "## What those pairs actually are",
        "",
        "A pair is only useful supervision if it spans a genuine domain shift.",
        "",
        "| minutes floor | all pairs | same competition | cross-league | club <-> international |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for floor, counts in population.pairs.items():
        label = "any" if floor == 0 else f">= {floor:.0f}"
        lines.append(
            f"| {label} | {counts.total:,} | {counts.same_competition:,} | "
            f"{counts.cross_league:,} | {counts.club_to_international:,} |"
        )

    if population.league_pair_counts:
        lines += [
            "",
            "## Cross-league pairs at the 270-minute floor",
            "",
            "Every league-to-league pairing with any players at all:",
            "",
            "| competition A | competition B | players |",
            "| --- | --- | ---: |",
        ]
        for (a, b), count in population.league_pair_counts.most_common(15):
            lines.append(f"| {a} | {b} | {count} |")

    lines += [
        "",
        "## Gender",
        "",
        "Fit per gender: essentially no player appears in both men's and women's",
        "competitions, so one head across both is unsupported.",
        "",
        "| gender | club <-> international players at >= 270 min |",
        "| --- | ---: |",
    ]
    for gender, count in population.per_gender.items():
        lines.append(f"| {gender} | {count} |")

    lines += [
        "",
        "## Verdict",
        "",
        f"**{'PROCEED' if ok else 'INSUFFICIENT DATA'}** -- {message}",
        "",
        "Note that this says nothing about similarity, which is the rest of scout.",
        "Contrastive positives for similarity come from splitting one player's own",
        "season by match parity, which yields thousands of pairs. The contrast",
        "between that number and the one above is the central fact about what open",
        "football data can and cannot support.",
    ]
    return "\n".join(lines)
