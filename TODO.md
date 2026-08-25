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

- **Projection head is expected to be statistically hollow, and will ship as a
  reported null result.** The spec asks to predict competition-B output from a
  competition-A embedding for players appearing in 2+ open competitions. Measured
  over 32% of the corpus: inside the 360-era universe only **33 players** have a
  genuine club-to-international domain shift with >=270 minutes on each side.
  Cross-league pairs are worse -- 7 to 22 players per league pair. That is not a
  training set. What ships is three baselines (identity, a same-league
  aging/difficulty offset, and a rank-constrained club-to-international map) with
  bootstrap confidence intervals, plus `reports/projection_power.md` stating the
  n and whether anything beats identity outside its CI. If nothing does, the
  README and the app both say so next to every projected number.
  **This does not affect similarity.** Contrastive positives come from splitting
  one player's own season by match parity, which yields thousands of pairs. The
  gap between "thousands" and "33" is the single most important fact about what
  this data can and cannot support.

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

- **Calibrate the action vocabulary against real events.** `vocab.ACTION_TYPES`
  is currently derived a priori from the StatsBomb spec and yields 47 action
  rows; an independent count of realized (type, outcome) combinations suggested
  closer to 105. The collapse of e.g. Duel's four outcomes into success/fail may
  be discarding signal. The tokenizer must count and report every pair it cannot
  map, and a one-competition pass will settle it. Changing the vocabulary after
  training invalidates every checkpoint and embedding, so settle it first.

- **Measure the 2+-competition population exactly.** The 33/194 figures above are
  from a 32% sample. A full pass over all 4,235 lineup files costs ~20 MB gzipped
  and about 105 minutes and converts the extrapolation into a measurement. If the
  corpus-wide club-to-international population at >=270 min/side comes in under
  300, the projection module is declared hollow before any model code is written.

- **Time-box the LM on MPS before committing to an overnight run.** 200 steps of
  the S config on random batches with `PYTORCH_ENABLE_MPS_FALLBACK=0`. If it
  exceeds 12 min/epoch, drop to d=192/6 layers or halve the corpus. A single op
  falling back to CPU is a 10x regression and is silent by default.

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
