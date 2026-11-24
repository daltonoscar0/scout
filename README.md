# scout

[![tests](https://github.com/daltonoscar0/scout/actions/workflows/tests.yml/badge.svg)](https://github.com/daltonoscar0/scout/actions/workflows/tests.yml)

Player similarity, projection and valuation from open football event data.

Type a player's name, get their nearest comparables, a projection of their output
in a target competition, and a model-implied value. Ships as a CLI, a Streamlit
scout card, and an Excel export, with a prebuilt index so queries run without
retraining.

Two of those three come with a measured caveat attached, stated below and
repeated wherever the number appears: the open corpus cannot support a learned
cross-competition projection, and "value" here is not money.

## How it works

Possessions from StatsBomb open data are tokenized into sequences, a small causal
transformer is trained to predict the next event, and a player's embedding is the
pooled hidden state over the positions where they were the actor. Similar players
are near neighbours in that space.

The token is factorized into five fields rather than fused into one symbol, and
that choice is load-bearing. A fused (action, zone, delta, flags, dt) space is on
the order of 10^8 symbols; the softmax alone would be tens of billions of
parameters. Five small embedding tables summed before the first block total 354
rows and about 91k parameters at `d_model=256`. Run `python -m scout.vocab` to
print the table and its fingerprint.

The vocabulary is a contract: every checkpoint, embedding and token shard records
the fingerprint it was built under, and loading one against a different
vocabulary is an error rather than a warning. Embeddings from two vocabularies
are not comparable, and the symptom would be plausible-looking wrong neighbours
rather than a crash.

## Quickstart

```sh
uv sync                             # query path only: CLI, app, export
uv run scout demo                   # runs offline from the bundled index

uv sync --extra train --extra models   # everything, for a rebuild
uv pip install -e ../sportio           # plus sportio, for `scout build`
uv run scout build-index            # fetch, tokenize, train, embed; resumable

uv run scout similar "Bernardo Silva" --k 10
uv run scout project "Bernardo Silva" --to "FIFA World Cup"
uv run scout value --league "La Liga" --top 25
uv run scout export shortlist.example.yaml --xlsx out.xlsx
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

`sportio` is not declared anywhere in `pyproject.toml`. It is an editable
checkout of a sibling repo, and uv resolves a path source even when the package
sits behind an extra -- so merely declaring it makes `uv sync` fail in any clone
that does not happen to have `../sportio` beside it. That is exactly what broke
`scout demo` the first time it was tried against a genuine fresh clone rather
than a simulated one. Install it explicitly when rebuilding the corpus:

```sh
uv pip install -e ../sportio    # only for `scout fetch` and `scout build`
```

The query path never imports torch. Importing it costs 2-4 seconds cold, which
would consume most of the spec's sub-5-second budget before any work happened,
and Streamlit reruns its script on every interaction. Training and model fitting
run offline in separate processes and export plain arrays the query path
memory-maps. A test asserts `torch` is absent from `sys.modules` after importing
the query path, so this cannot regress by accident.

There is no non-pip prerequisite. `scikit-learn` is used instead of LightGBM
specifically because LightGBM's dylib resolves `libomp` through Homebrew and
fails at import on a clean machine.

## Results

Measured, on the corpus described below. One of the three deliverables works
well; the other two are null results, and they are reported as such.

### Similarity: works

Held-out same-player retrieval. Each player's matches are split by parity into
two disjoint halves; the query is one half and the target is the same player's
other half among 1,500 others. Chance is 0.067%.

| | recall@1 | recall@10 | mean rank |
| --- | ---: | ---: | ---: |
| pooled, before contrastive training | 8.1% | 29.3% | 122.2 |
| **after contrastive training** | **33.3%** | **73.1%** | **18.2** |

500x chance. The model is never shown a player's identity or position, yet the
neighbours are positionally coherent: Busquets returns six defensive
midfielders, Neuer six goalkeepers, van Dijk six centre-backs, and Messi returns
creative attackers including Aitana Bonmatí from the women's game.

### Projection: the honest three-way comparison

Every fitted predictor is scored **out of sample**, by 5-fold cross-validation
grouped on player, with a 1,000-sample bootstrap interval on mean absolute
error. Lower MAE is better.

| predictor | uses player information? |
| --- | --- |
| B0 identity | no -- predict the source value unchanged |
| B1 league offset | no -- plus a fitted per-competition-pair shift |
| B1b shrunk identity | no -- ridge on the source value alone |
| B2 embedding ridge | **yes** -- low-rank projection of the player embedding |

**xG+xA per 90**

| predictor | male MAE (n=980) | female MAE (n=925) |
| --- | ---: | ---: |
| B0 identity | 0.146 `[0.136, 0.155]` | 0.155 `[0.145, 0.166]` |
| B1 league offset | 0.157 `[0.148, 0.166]` | 0.160 `[0.150, 0.171]` |
| B1b shrunk identity | 0.153 `[0.144, 0.161]` | 0.164 `[0.155, 0.174]` |
| B2 embedding ridge | 0.146 `[0.138, 0.155]` | **0.135** `[0.127, 0.145]` |

**Progressive actions per 90**

| predictor | male MAE (n=980) | female MAE (n=925) |
| --- | ---: | ---: |
| B0 identity | 2.147 `[2.045, 2.254]` | 2.389 `[2.281, 2.510]` |
| B1 league offset | 2.083 `[1.978, 2.181]` | 2.223 `[2.115, 2.349]` |
| B1b shrunk identity | 1.959 `[1.868, 2.059]` | 2.108 `[2.009, 2.213]` |
| B2 embedding ridge | 1.835 `[1.739, 1.934]` | 1.972 `[1.879, 2.070]` |

**B1b is the control that makes this honest.** A ridge shrinks a noisy target
toward the mean, beating carry-forward while knowing nothing about the player at
all. Without it, B2 appears to beat identity on three of four strata and the
obvious reading is "the embedding knows how a player translates". Against B1b,
**three of those four wins disappear into overlapping intervals.** Only xG+xA for
women survives both controls (0.135 against the control's 0.164).

One stratum in four is not a basis for projecting anyone, so **scout returns the
identity projection everywhere**, and says so next to every projected number.
Regenerate with `scout projection`; full output in
`reports/projection_baselines.md`.

### Valuation: does not beat predicting the mean

MAE 560.6 against a predict-the-training-mean baseline of 526.3, rank
correlation -0.014. The cause is in `reports/valuation.md`: only 70.7% of rows
are labelled by a genuinely consecutive season, because open coverage of any one
competition is intermittent, and for the rest the label means "whenever we next
observed them".

### The language model

6,557,538 parameters (spec cap: 20M). Held-out action perplexity **2.56 against
a bigram baseline of 3.04** -- a 16% improvement, converged by epoch 14. Real,
but not a landslide, and stated that way rather than as a win.

### Latency

`scout similar` returns in **1.35 s** including interpreter startup, against the
spec's 5 s budget. The query path is numpy over a 8,988 x 256 matrix.

## What this data can and cannot support

Two findings shape what ships, and both are stated up front rather than buried:

**Similarity works.** Contrastive positives come from splitting a single player's
own matches by parity, which yields 6,605 training pairs across the players
clearing the touches floor -- and the retrieval numbers above are what that buys.

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
price**, and every surface that shows it says so.

**There is no Transfermarkt path, opt-in or otherwise.** An earlier draft
promised a `--enable-transfermarkt` flag. It was never built, and shipping the
promise without the implementation is worse than not offering it: their terms
prohibit both automated access and training models on the content, their
coverage is entirely men's football, and the valuation model does not beat
predicting the mean even on the clean target. Nothing here would be improved by
adding a legally doubtful data source to it.

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

## License

MIT. See [LICENSE](LICENSE).
