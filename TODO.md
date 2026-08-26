# TODO

Open items, honest constraints, and things that need a decision. Anything here
that says BLOCKED is blocked by the data, not by effort.

## Needs a decision from the user

- **Transfermarkt data must not be committed to this repo.** The spec asks for a
  Kaggle Transfermarkt mirror snapshot committed under `data/`. Transfermarkt's
  terms (clause 11.1, fetched 2026-08-25) prohibit automated access *and*
  prohibit use of the content for training machine learning models, by name.
  The data is additionally protected by the EU sui generis database right, and
  `players.csv` carries date of birth, city of birth and agent name for ~50,000
  identifiable people. The CC0 label on the third-party mirror is a unilateral
  declaration by someone who did not hold the rights to grant it.
  Not legal advice, but this is a conscious decision, not one the build should
  make silently. Current plan: valuation ships with a different default target
  (below); Transfermarkt is opt-in, fetched at build time, gitignored, never
  redistributed.

## Known-degraded against the spec

- **MEASURED -- the projection head is statistically hollow and ships as a
  reported null result.** Full-corpus measurement (all 3,961 lineups, not a
  sample) in `reports/projection_power.md`, reproducible with `scout projection`.
  11,794 players and 21,405 player-seasons yield 4,079 players in 2+
  competition-seasons on squad membership, but only 935 once 270 minutes are
  required on both sides. Of the surviving pairs, 1,792 of 3,697 are two editions
  of the same competition -- an aging model, not a domain map. The genuine
  club-to-international population is **486 players, which cannot be pooled**:
  essentially nobody appears in both men's and women's competitions, so the real
  strata are **275 female and 211 male**, both under the 300 needed to identify
  even a heavily constrained map against a 65,000-parameter target.
  Cross-league is worse: the largest league pairing in the whole corpus is La
  Liga to Ligue 1 at 65 players, most are single digits.
  **This does not affect similarity.** Contrastive positives come from splitting
  one player's own season by match parity, which yields thousands of pairs. The
  gap between "thousands" and "211" -- same players, same data -- is the single
  most important fact about what this corpus can and cannot support.

- **Valuation's default target is not money.** It is `next_season_minutes_at_tier`
  -- minutes played the following season at the same-or-higher competition tier,
  computed from StatsBomb lineups alone. Legally clean, covers women's football
  (which Transfermarkt does not cover at all: all 65 of its competitions are
  men's), thousands of labels. It is a scarcity proxy, not a price, and the docs
  say so. `--enable-transfermarkt` switches to log market value, scoped to men's
  club competitions 2004 onward.

- **FAISS is banned, and the spec's FAISS index is replaced by brute-force
  numpy.** faiss-cpu and torch each vendor their own libomp; importing both in
  one process aborts with rc=134 in either order, and `KMP_DUPLICATE_LIB_OK=TRUE`
  converts the clean abort into a segfault. Import-order workarounds are
  unshippable because ruff's `I` rule re-sorts imports on autofix. We do not need
  it: cosine similarity over ~6,300 players x 256 dims is one 1.6 MFLOP matvec,
  about 2 ms in numpy. `hnswlib` is the escape hatch if the index ever outgrows
  brute force (verified torch-safe -- it links no libomp).

- **The tokenizer cannot use 360 freeze frames for off-ball players.** All 52,558
  freeze-frame entries checked in one match have key set exactly
  `{teammate, actor, keeper, location}`. There is no player id and no name. Only
  the single `actor` entry is identifiable. Any claim of per-player 360 positions
  for all 22 players is factually wrong about this data.

- **Zone grid is 12x8 = 96 cells, not the spec's 18.** 18 cells are 17.5m x 22.7m,
  too coarse to separate a near-post cross from a cut-back -- exactly the
  distinction similarity needs. Costs 20k parameters.

## Must do before training

- **DONE -- vocabulary calibrated against real events.** The first cut keyed on
  raw StatsBomb type names; sportio snake_cases everything, so every lookup would
  have missed and the corpus would have been nothing but fallback tokens. Rebuilt
  against a survey of four Euro 2024 matches: 32 event types, 32 distinct outcome
  strings, `counterpress` inside `extra` rather than a column. Every (type,
  outcome class) pair now gets a row rather than only the pairs judged reachable
  in advance -- 173 action rows, 347 total. Tokenizing 14,588 real events maps
  100% with no unmodelled types and no unmapped outcomes.

- **DONE -- LM timing measured on MPS, comfortably inside an overnight window.**
  Config S is 6,553,947 parameters and config M is 14,549,339, both under the
  spec's 20M cap. At batch 96: 2,734 ms/step, 35 sequences/s. That is 13.3
  min/epoch over the full 7.1M-event corpus, so 30 epochs is about 6.6 hours.

- **DONE -- no silent MPS fallback.** 30 full forward/backward/step iterations
  run with `PYTORCH_ENABLE_MPS_FALLBACK=0` without raising, and every parameter
  stays on the MPS device. A single operator falling back to CPU would be a ~10x
  regression that is invisible by default, so this is worth rechecking after any
  change to the model's operator set.

- **Measure the 2+-competition population exactly.** The figures above come from
  a partial sample. A full pass over all 3,961 lineup files costs ~20 MB gzipped
  and about 33 minutes at 2 req/s. If the corpus-wide club-to-international
  population at >=270 min/side comes in under 300, the projection module is
  declared hollow before any model code is written.

## Environment

- **`brew install libomp` is a hard prerequisite** for the `models` extra.
  lightgbm links `@rpath/libomp.dylib` and works on this machine only because
  Homebrew's libomp happens to be present. On a clean Mac or a CI runner it fails
  at import. Preflight check needed.

- **sportio has no player/minutes crosswalk yet.** It exports `load_events`,
  `corners`, `set_pieces` and friends, but nothing that answers "which players
  played how many minutes in which competition". scout builds its own in
  `scout.crosswalk` from StatsBomb lineups. If sportio grows a `player_index()`,
  prefer it and delete ours.

- **sportio is developed concurrently in another session** and installed here as
  an editable path dependency, so its API can change under us. Every import of it
  is confined to `src/scout/sportio_adapter.py`, which checks on use that the
  symbols scout needs still exist and fails with a message naming the missing
  ones.

- **Disk is the binding constraint on this machine**, not compute. The build
  keeps everything gzipped at rest (measured 10.6x on events, 7.6x on 360) and
  never decompresses in place. The full StatsBomb corpus is 15.4 GiB raw.

## Deferred

- Ad-hoc embedding of a player not in the prebuilt index (needs torch in the
  query path, which would break the latency budget -- would run as a subprocess).
- Promotion of the LM from config S (6.5M params) to config M (14.5M). Both are
  under the spec's 20M cap; S must train end to end first.
