# scout

Player similarity, projection and valuation from open football event data.

Type a player's name, get their nearest comparables, a projection of their output
in a target competition, and a model-implied value against a market baseline.
Ships as a CLI, a Streamlit scout card, and an Excel export, with a prebuilt
index so queries run without retraining.

## How it works

Possessions from StatsBomb open data are tokenized into sequences, a small causal
transformer is trained to predict the next event, and a player's embedding is the
pooled hidden state over the positions where they were the actor. Similar players
are near neighbours in that space.

The token is factorized into five fields rather than fused into one symbol, and
that choice is load-bearing. A fused (action, zone, delta, flags, dt) space is on
the order of 10^8 symbols; the softmax alone would be tens of billions of
parameters. Five small embedding tables summed before the first block total 347
rows and about 89k parameters at `d_model=256`. Run `python -m scout.vocab` to
print the table.

## Quickstart

```sh
uv sync                             # query path only: CLI, app, export
uv run scout demo                   # runs offline from the bundled index

uv sync --extra train --extra models   # everything, for a rebuild
uv run scout build-index            # fetch, tokenize, train, embed; resumable

uv run scout similar "Bernardo Silva" --k 10
uv run scout project "Bernardo Silva" --to "FIFA World Cup"
uv run scout value --league "La Liga" --top 25
uv run scout export shortlist.yaml --xlsx out.xlsx
uv run scout app                    # the Streamlit scout card
uv run scout info                   # versions, vocabulary, what is cached
```

`scout build-index` is the single documented rebuild. Every stage skips work
already on disk, so if it is interrupted you rerun the same command and it
resumes rather than starting over.

### The dependency split is deliberate

| install | contains | used by |
| --- | --- | --- |
| `uv sync` | numpy, pandas, streamlit, openpyxl | everything on the query path |
| `--extra train` | torch | `scout train`, `scout embed` |
| `--extra models` | scikit-learn | `scout fit-value` |

The query path never imports torch. Importing it costs 2-4 seconds cold, which
would consume most of the spec's sub-5-second budget before any work happened,
and Streamlit reruns its script on every interaction. Training and model fitting
run offline in separate processes and export plain arrays the query path
memory-maps. A test asserts `torch` is absent from `sys.modules` after importing
the query path, so this cannot regress by accident.

There is no non-pip prerequisite. `scikit-learn` is used instead of LightGBM
specifically because LightGBM's dylib resolves `libomp` through Homebrew and
fails at import on a clean machine.

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

**Valuation does not predict money.** The spec asks to regress Transfermarkt
market value. Their terms (clause 11.1) prohibit automated access *and* prohibit
using the content to train machine learning models, by name; the spec also asks
to commit the snapshot into `data/`, which would make this repo a redistribution
point for scraped commercial data covering ~50,000 identifiable people. On top of
that, all 65 of Transfermarkt's competitions are men's football while much of
this corpus is not, so those players would silently lose their target variable.

The shipped default target is **next-season minutes at the same or a better
competition tier**, computed from StatsBomb lineups alone. It is legally clean,
covers everyone, and has thousands of labels. It is a scarcity proxy, **not a
price**, and every surface that shows it says so. Market value remains available
behind `scout value --enable-transfermarkt`, fetched at build time and never
committed.

Three features the spec asks for cannot be built from open data at all, and are
reported missing rather than quietly dropped: **age** and **contract length**
(no open source carries either) and **market value** (above).

`reports/valuation.md` answers the spec's out-of-sample question -- do residuals
predict next-window value changes? -- with whatever the data says, including no.

## Sources and licences

| Source | Used for | Licence |
| --- | --- | --- |
| StatsBomb open data (via sportio) | events, lineups | StatsBomb open data terms: free for public research and analysis, attribution required, non-commercial |
| Transfermarkt mirror | optional valuation target | **not redistributed.** Opt-in, fetched at build time, gitignored. Their terms prohibit automated access and ML training; see `TODO.md` |

Everything shipped here is derived from StatsBomb open data. Nothing under
`data/` is committed.

## Reports

| file | what it answers |
| --- | --- |
| `reports/projection_power.md` | can the corpus support a projection head at all? |
| `reports/projection_baselines.md` | the three-way comparison, scored out of sample |
| `reports/valuation.md` | the fit, and whether residuals persist between windows |
| `models/MODEL_CARD.md` | the LM's architecture, training and honest result |
