# scout

Player similarity, projection and valuation from open football event data.

Type a player's name, get their nearest comparables, a projection of their output
in a target competition, and a model-implied value against a market baseline.
Ships as a CLI, a Streamlit scout card, and an Excel export, with a prebuilt
index so queries run without retraining.

**Status: in development.** The vocabulary and the sportio adapter are in place.
The tokenizer, LM, embeddings, projection and valuation are not yet built.

## How it works

Possessions from StatsBomb open data are tokenized into sequences, a small causal
transformer is trained to predict the next event, and a player's embedding is the
pooled hidden state over the positions where they were the actor. Similar players
are near neighbours in that space.

The token is factorized into five fields rather than fused into one symbol, and
that choice is load-bearing. A fused (action, zone, delta, flags, dt) space is on
the order of 10^8 symbols; the softmax alone would be tens of billions of
parameters. Five small embedding tables summed before the first block cost about
71k parameters at `d_model=256`. Run `python -m scout.vocab` to print the table.

## Install

```sh
uv sync                      # query path: CLI, app, export
uv sync --extra train        # adds torch, for `scout train` / `scout embed`
uv sync --extra models       # adds lightgbm, for `scout fit-value`
```

`--extra models` additionally needs `brew install libomp`.

The dependency split is deliberate. The query path never imports torch: importing
it costs 2-4 seconds cold, which would consume most of the sub-5-second latency
budget on its own. Training and scoring run offline as separate processes and
export plain arrays the query path memory-maps.

## What this data can and cannot support

Two findings shape what ships, and both are stated up front rather than buried:

**Similarity works.** Contrastive positives come from splitting a single player's
own season by match parity, which yields thousands of training pairs across the
roughly 3,000 players clearing the minutes floor.

**Cross-competition projection does not.** The spec asks to project a player
from competition A into competition B, trained on players appearing in both.
Measured over **every lineup in the corpus** -- 3,961 matches, 11,794 players,
21,405 player-seasons, not a sample:

| minutes floor, each side | players in 2+ competition-seasons |
| --- | ---: |
| any (squad membership) | 4,079 |
| >= 270 min | 935 |
| >= 900 min | 211 |

Squad membership collapses by 77% once real minutes are required on both sides.
And most of what survives is not a domain shift: at the 270-minute floor, 1,792
of 3,697 pairs are two editions of the *same* competition, which teaches "four
years older", not "how this player translates".

What is left is **486 players with a genuine club-to-international move** -- and
those cannot be pooled, because essentially no player appears in both men's and
women's competitions. Per gender the usable counts are **275 female and 211
male**. A 256-to-256 linear map has 65,000 parameters. Cross-*league* projection
is worse still: the largest single league pairing in the entire open corpus is
La Liga to Ligue 1 at 65 players, and most pairings are in single digits.

So the projection module ships as a **reported null result**: identity as the
baseline, bootstrap confidence intervals, and the verdict stated in
`reports/projection_power.md`, in this README, and next to every projected number
in the app. It is not hidden behind a plausible-looking number.

That contrast -- thousands of training pairs for similarity, 211 for projection,
from the same players in the same data -- is the most important thing to
understand about open football data.

**Valuation does not predict money by default.** See `TODO.md`; Transfermarkt's
terms prohibit exactly this use, so the default target is next-season minutes at
tier, computed from StatsBomb alone.

## Sources and licences

| Source | Used for | Licence |
| --- | --- | --- |
| StatsBomb open data (via sportio) | events, lineups, 360 frames | StatsBomb open data terms, non-commercial, attribution required |
| Transfermarkt mirror | optional valuation target | **not redistributed**; opt-in, fetched at build time, see `TODO.md` |

Player similarity and projection here are derived from StatsBomb open data.
