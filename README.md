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

**Cross-competition projection almost certainly does not.** The spec asks to
project a player from competition A into competition B using players who appear
in both. Measured across 32% of the corpus, only **33 players** inside the
360-era universe have a genuine club-to-international move with at least 270
minutes on each side, and cross-league pairs run 7 to 22 players. A 256-to-256
linear map has 65,000 parameters; there is nothing like enough supervision to fit
one. The projection module therefore ships as three baselines with bootstrap
confidence intervals and an honest verdict in `reports/projection_power.md`. If
the learned map does not beat identity outside its confidence interval, this
README will say so, and so will the app, next to every projected number.

That contrast -- thousands of pairs for similarity, 33 for projection -- is the
most important thing to understand about open football data.

**Valuation does not predict money by default.** See `TODO.md`; Transfermarkt's
terms prohibit exactly this use, so the default target is next-season minutes at
tier, computed from StatsBomb alone.

## Sources and licences

| Source | Used for | Licence |
| --- | --- | --- |
| StatsBomb open data (via sportio) | events, lineups, 360 frames | StatsBomb open data terms, non-commercial, attribution required |
| Transfermarkt mirror | optional valuation target | **not redistributed**; opt-in, fetched at build time, see `TODO.md` |

Player similarity and projection here are derived from StatsBomb open data.
