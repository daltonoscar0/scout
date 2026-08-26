# Projection power

Can a cross-competition projection head be trained on the open corpus?
Measured over every lineup in StatsBomb open data, not sampled.

- player-season rows: 21,405
- rows with more than zero minutes: 18,113 (84.6%)
- distinct players: 11,794

## Players appearing in two or more competition-seasons

| minutes floor, each side | players |
| --- | ---: |
| any (squad membership) | 4,079 |
| >= 90 min | 2,591 |
| >= 270 min | 935 |
| >= 450 min | 501 |
| >= 900 min | 211 |

## What those pairs actually are

A pair is only useful supervision if it spans a genuine domain shift.

| minutes floor | all pairs | same competition | cross-league | club <-> international |
| --- | ---: | ---: | ---: | ---: |
| any | 25,445 | 15,386 | 3,199 | 5,455 |
| >= 90 | 14,742 | 8,687 | 1,750 | 3,453 |
| >= 270 | 3,697 | 1,792 | 265 | 1,259 |
| >= 450 | 2,126 | 1,466 | 196 | 402 |
| >= 900 | 1,097 | 988 | 109 | 0 |

## Cross-league pairs at the 270-minute floor

Every league-to-league pairing with any players at all:

| competition A | competition B | players |
| --- | --- | ---: |
| La Liga | Ligue 1 | 65 |
| FA Women's Super League | NWSL | 43 |
| La Liga | Major League Soccer | 38 |
| La Liga | Premier League | 31 |
| La Liga | Serie A | 23 |
| FA Women's Super League | Liga F | 11 |
| Ligue 1 | Serie A | 9 |
| FA Women's Super League | Frauen Bundesliga | 9 |
| Ligue 1 | Premier League | 8 |
| FA Women's Super League | Serie A Women | 8 |
| Liga F | NWSL | 5 |
| Premier League | Serie A | 3 |
| Indian Super league | La Liga | 3 |
| Ligue 1 | Major League Soccer | 2 |
| NWSL | Serie A Women | 2 |

## Gender

Fit per gender: essentially no player appears in both men's and women's
competitions, so one head across both is unsupported.

| gender | club <-> international players at >= 270 min |
| --- | ---: |
| male | 211 |
| female | 275 |

## Verdict

**INSUFFICIENT DATA** -- no gender stratum reaches the 300 players needed to identify even a heavily constrained cross-domain map (female 275, male 211). A learned projection is not supported by this data. Ship identity as the baseline and say so in the README and next to every projected number in the UI.

Note that this says nothing about similarity, which is the rest of scout.
Contrastive positives for similarity come from splitting one player's own
season by match parity, which yields thousands of pairs. The contrast
between that number and the one above is the central fact about what open
football data can and cannot support.