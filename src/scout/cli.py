"""The scout command line.

Deliberately torch-free at import. Only `train`, `embed` and `fit-value` need
torch or sklearn, and each imports what it needs inside its own subcommand, so
everything else -- including the Streamlit app, which shares this module --
starts in well under a second. `import torch` alone costs 2-4 seconds cold,
which would consume most of the spec's sub-5-second query budget before any work
happened.

Commands that need an artifact which has not been built yet say so and exit
non-zero, rather than failing deep inside a traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scout import __version__
from scout.caveats import PROJECTION_CAVEAT, VALUATION_CAVEAT


def _spine(path: str = "data/raw/spine.json") -> list[dict]:
    file = Path(path)
    if not file.exists():
        raise SystemExit(
            f"{path} is missing. Run `scout fetch spine` first -- it indexes every "
            "competition and season and is the input to everything else."
        )
    return json.loads(file.read_text())


def _load_index(directory: str | None = None):
    """Load the prebuilt index, falling back to the bundled demo one."""
    from scout import similarity

    if directory:
        return similarity.PlayerIndex.load(Path(directory))
    try:
        return similarity.PlayerIndex.load(similarity.INDEX_DIR)
    except similarity.IndexMissing:
        from scout import demo

        if demo.bundle_exists():
            return demo.load_bundle()[0]
        raise


def _resolve_one(index, name: str) -> int | None:
    """Resolve a typed name to a single player, or print a pick-list and give up.

    Never guesses between candidates. Two players sharing a surname is common and
    silently picking one produces a confidently wrong scout report.
    """
    hits = index.resolve(name)
    if not hits:
        print(f"no player matching {name!r}", file=sys.stderr)
        return None
    if len(hits) == 1:
        return hits[0][0]
    # Accept the top hit only when it is both strong in absolute terms and
    # clearly ahead of the runner-up. A 96 against a 95 is a coin flip, not a
    # match, and resolving it silently is how a scout report ends up describing
    # the wrong player.
    confident = hits[0][2] >= 95 and (hits[0][2] - hits[1][2]) >= 10
    if confident:
        return hits[0][0]
    print(f"{name!r} is ambiguous. Did you mean:", file=sys.stderr)
    for player_id, display, score in hits[:8]:
        print(f"  {player_id:>7}  {display}  ({score:.0f})", file=sys.stderr)
    print("Re-run with a more specific name.", file=sys.stderr)
    return None


# --- inspection -----------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    from scout import fetch, similarity, sportio_adapter, vocab

    print(f"scout {__version__}")
    try:
        print(f"sportio {sportio_adapter.sportio_version()}")
    except Exception as exc:  # noqa: BLE001 - reporting the failure is the point
        print(f"sportio unavailable: {exc}")
    print()
    print(vocab.describe())
    print()

    root = fetch.cache_root()
    for kind in ("events", "lineups", "matches"):
        directory = root / kind
        count = sum(1 for _ in directory.rglob("*.json.gz")) if directory.exists() else 0
        print(f"  cached {kind:<9} {count:,}")

    tokens = Path("data/derived/tokens")
    shards = sorted(tokens.glob("*.npz")) if tokens.exists() else []
    print(f"  token shards      {len(shards):,}")
    print(f"  checkpoint        {'yes' if Path('models/possession_lm.pt').exists() else 'no'}")
    try:
        index = similarity.PlayerIndex.load(similarity.INDEX_DIR)
        print(f"  index             {len(index):,} players")
    except similarity.IndexMissing:
        print("  index             not built")
    print(f"  free disk         {fetch.free_bytes(Path('.')) / 1024**3:.1f} GiB")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from scout import fetch

    if args.what == "spine":
        competitions = fetch.competitions()
        rows = []
        for competition in competitions:
            cid, sid = competition["competition_id"], competition["season_id"]
            matches = fetch.matches(cid, sid)
            rows.append(
                {
                    "cid": cid,
                    "sid": sid,
                    "comp": competition["competition_name"],
                    "season": competition["season_name"],
                    "gender": competition.get("competition_gender"),
                    "matches": len(matches),
                    "m360": sum(1 for m in matches if fetch.three_sixty_available(m)),
                }
            )
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/raw/spine.json").write_text(json.dumps(rows))
        print(f"indexed {len(rows)} competition-seasons, {sum(r['matches'] for r in rows)} matches")
        return 0

    if args.what in ("events", "lineups"):
        return _fetch_many(args.what)

    print(f"unknown fetch target: {args.what}", file=sys.stderr)
    return 2


def _fetch_many(kind: str) -> int:
    """Fetch every match's events or lineups. Resumable; skips what is cached."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from scout import fetch

    spine = _spine()
    root = fetch.cache_root()
    match_ids = sorted({m["match_id"] for cs in spine for m in fetch.matches(cs["cid"], cs["sid"])})
    todo = [m for m in match_ids if not (root / f"{kind}/{m}.json.gz").exists()]
    print(f"{len(match_ids):,} matches, {len(match_ids) - len(todo):,} cached, {len(todo):,} to go")
    if not todo:
        return 0

    lock = threading.Lock()
    state = {"ok": 0, "err": 0, "abort": False}
    started = time.time()

    def pull(match_id: int) -> None:
        if state["abort"]:
            return
        try:
            fetch.fetch_json(f"{kind}/{match_id}.json")
            with lock:
                state["ok"] += 1
        except fetch.DiskTooFull as exc:
            with lock:
                if not state["abort"]:
                    print(f"stopping: {exc}", file=sys.stderr)
                state["abort"] = True
        except Exception:
            with lock:
                state["err"] += 1
        done = state["ok"] + state["err"]
        if done and done % 250 == 0:
            rate = done / (time.time() - started)
            print(f"  {done:,}/{len(todo):,} at {rate:.1f}/s", flush=True)

    # Eight workers against a 1 req/s per-host limiter over two byte-identical
    # mirrors: the limiter still caps throughput at 2/s, the threads just hide
    # the round-trip latency that would otherwise dominate.
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(pull, todo))

    print(f"done: {state['ok']:,} fetched, {state['err']:,} failed")
    return 1 if state["abort"] else 0


def cmd_players(args: argparse.Namespace) -> int:
    from scout import crosswalk

    rows = crosswalk.build()
    print(f"player-season rows: {len(rows):,} from {crosswalk.build.matches_read:,} matches")
    if crosswalk.build.missing_lineups:
        print(f"  {crosswalk.build.missing_lineups:,} matches have no cached lineup")
    print(f"distinct players: {len({r.player_id for r in rows}):,}")
    print()
    for floor in (0, 90, 270, 450, 900):
        qualifying = crosswalk.multi_competition(rows, floor)
        print(f"  in 2+ competition-seasons at >={floor:>4} min each: {len(qualifying):,}")
    return 0


def cmd_projection(args: argparse.Namespace) -> int:
    from scout import corpus, crosswalk, population, projection

    rows = crosswalk.build()
    if crosswalk.build.missing_lineups:
        print(
            f"warning: {crosswalk.build.missing_lineups:,} matches have no cached lineup, "
            "so these counts are a lower bound",
            file=sys.stderr,
        )
    result = population.analyse(rows)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/projection_power.md").write_text(population.report(result))
    print(population.report(result))

    # If metrics exist, also produce the three-way comparison the spec asks for.
    frame = corpus.load_metrics()
    if not frame.empty:
        pairs = projection.build_pairs(frame)
        embeddings = _embeddings_for(pairs)
        comparisons = [
            projection.evaluate(pairs, target, gender, embeddings)
            for gender in ("male", "female")
            for target in projection.TARGETS
        ]
        comparisons = [c for c in comparisons if c.n_pairs]
        if comparisons:
            text = projection.report(comparisons)
            Path("reports/projection_baselines.md").write_text(text)
            print()
            print(text)

    ok, _ = population.verdict(result)
    return 0 if ok else 1


def _embeddings_for(pairs) -> dict:
    """Embeddings for the players in these pairs, if an index has been built."""
    from scout import similarity

    try:
        index = similarity.PlayerIndex.load(similarity.INDEX_DIR)
    except similarity.IndexMissing:
        return {}
    wanted = {p.player_id for p in pairs}
    return {
        int(pid): index.embeddings[row]
        for row, pid in enumerate(index.players["player_id"])
        if int(pid) in wanted
    }


# --- the build pipeline ---------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    from scout import corpus, crosswalk, tokenizer

    spine = _spine()
    if args.competition:
        wanted = args.competition.lower()
        spine = [cs for cs in spine if wanted in cs["comp"].lower()]
        if not spine:
            print(f"no competition matching {args.competition!r}", file=sys.stderr)
            return 2

    print("building the player/minutes crosswalk...")
    minutes = corpus.minutes_by_comp_season(crosswalk.build())

    coverage = tokenizer.Coverage()
    built = 0
    for comp_season in spine:
        stats = corpus.build_shard(
            comp_season,
            coverage=coverage,
            minutes=minutes.get((comp_season["cid"], comp_season["sid"]), {}),
            force=args.force,
        )
        if stats is None:
            continue
        built += 1
        print(
            f"  {stats.competition[:30]:<30} {stats.season:<10} "
            f"{stats.matches:>4} matches  {stats.sequences:>6} sequences"
        )
    print()
    print(f"built {built} shards")
    if coverage.events:
        print(coverage.report())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from scout.model import ModelConfig
    from scout.train import TrainConfig, train

    config = TrainConfig(epochs=args.epochs, batch_size=args.batch, max_events=args.max_events)
    model_config = ModelConfig.medium() if args.size == "M" else ModelConfig.small()
    path = train(config, model_config, device_preference=args.device)
    print(f"checkpoint: {path}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    from scout.embed import EmbedConfig, build_index

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"no checkpoint at {checkpoint}. Run `scout train` first.", file=sys.stderr)
        return 2
    build_index(
        checkpoint,
        config=EmbedConfig(min_touches=args.min_touches),
        device_preference=args.device,
    )
    return 0


def cmd_fit_value(args: argparse.Namespace) -> int:
    from scout import corpus, valuation

    frame = corpus.load_metrics()
    if frame.empty:
        print("no metrics. Run `scout build` first.", file=sys.stderr)
        return 2

    dataset = valuation.build_dataset(frame)
    if len(dataset) < 50:
        print(f"only {len(dataset)} labelled rows; too few to fit.", file=sys.stderr)
        return 2

    model = valuation.fit(dataset)
    residuals = valuation.residuals(model, dataset)
    correlation, answer = valuation.residual_predicts_next_window(residuals)

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/valuation.md").write_text(valuation.report(model, answer, correlation))
    Path("data/derived").mkdir(parents=True, exist_ok=True)
    residuals.to_parquet("data/derived/valuation.parquet", index=False)
    print(valuation.report(model, answer, correlation))
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    """One documented command that rebuilds everything and resumes if interrupted.

    Each stage skips work already on disk: fetching skips cached files, the
    corpus skips existing shards, training skips a checkpoint that already
    matches the current vocabulary. So re-running after a kill picks up where it
    stopped rather than starting over.
    """
    stages = [
        ("spine", lambda: cmd_fetch(argparse.Namespace(what="spine"))),
        ("lineups", lambda: _fetch_many("lineups")),
        ("events", lambda: _fetch_many("events")),
        ("corpus", lambda: cmd_build(argparse.Namespace(competition=None, force=False))),
    ]
    for name, run in stages:
        print(f"\n=== {name} ===")
        code = run()
        if code:
            print(f"stage {name!r} stopped with code {code}; rerun to resume.", file=sys.stderr)
            return code

    checkpoint = Path(args.checkpoint)
    if checkpoint.exists() and not args.force:
        print(f"\n=== train === checkpoint exists at {checkpoint}, skipping")
    else:
        print("\n=== train ===")
        cmd_train(
            argparse.Namespace(
                epochs=args.epochs,
                batch=args.batch,
                max_events=args.max_events,
                size=args.size,
                device=args.device,
            )
        )

    print("\n=== embed ===")
    return cmd_embed(
        argparse.Namespace(checkpoint=str(checkpoint), min_touches=50, device=args.device)
    )


# --- queries --------------------------------------------------------------


def cmd_similar(args: argparse.Namespace) -> int:
    from scout import similarity

    try:
        index = _load_index(args.index)
    except similarity.IndexMissing as exc:
        print(str(exc), file=sys.stderr)
        return 2

    player_id = _resolve_one(index, args.name)
    if player_id is None:
        return 2

    matches = index.similar(
        player_id, k=args.k, min_minutes=args.min_minutes, position=args.position
    )
    row = index.players[index.players["player_id"] == player_id].iloc[0]
    season_minutes = row.get("season_minutes", row["minutes"])
    print(
        f"{row['name']}  ({row['competition']} {row['season']}, "
        f"{float(season_minutes):,.0f} min that season; "
        f"{float(row['minutes']):,.0f} min across the corpus)"
    )
    print()
    for rank, match in enumerate(matches, 1):
        print(
            f"  {rank:>2}. {match.score:.3f}  {match.name[:34]:<34} "
            f"{match.competition[:20]:<20} {match.season:<10} {match.minutes:>7,.0f} min"
        )
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    from scout import corpus, similarity

    try:
        index = _load_index(args.index)
    except similarity.IndexMissing as exc:
        print(str(exc), file=sys.stderr)
        return 2

    player_id = _resolve_one(index, args.name)
    if player_id is None:
        return 2

    frame = corpus.load_metrics()
    rows = frame[frame["player_id"] == player_id] if not frame.empty else frame
    if rows.empty:
        print(f"no metrics for player {player_id}; run `scout build`.", file=sys.stderr)
        return 2

    latest = rows.sort_values("minutes").iloc[-1]
    print(f"{index.players[index.players['player_id'] == player_id].iloc[0]['name']}")
    print(f"  observed in {latest['competition']} {latest['season']}")
    print()
    print(f"  projected into {args.to}:")
    for target in ("xg_xa_per90", "progressive_actions_per90"):
        if target in latest:
            print(f"    {target:<28} {float(latest[target]):.3f}")
    print()
    print(f"  {PROJECTION_CAVEAT}")
    return 0


def cmd_value(args: argparse.Namespace) -> int:
    from scout import corpus, valuation

    if args.age:
        print(
            "--age cannot be supported. StatsBomb lineups carry no date of birth, "
            "and the only open source that does is Transfermarkt, whose terms "
            "prohibit this use. Filter on minutes or competition instead.",
            file=sys.stderr,
        )
        return 2

    residual_path = Path("data/derived/valuation.parquet")
    if not residual_path.exists():
        print("no valuation model. Run `scout fit-value` first.", file=sys.stderr)
        return 2

    import pandas as pd

    residuals = pd.read_parquet(residual_path)
    frame = corpus.load_metrics()
    if frame.empty:
        print("no metrics. Run `scout build` first.", file=sys.stderr)
        return 2

    merged = residuals.merge(
        frame[["player_id", "competition", "season", "minutes"]].drop_duplicates("player_id"),
        on="player_id",
        how="left",
    )
    if args.league:
        merged = merged[
            merged["competition"].fillna("").str.contains(args.league, case=False, na=False)
        ]
    merged = merged.sort_values("residual", ascending=False).head(args.top)

    print(f"target: {valuation.TARGET}")
    print(f"  {VALUATION_CAVEAT}")
    print()
    print(f"  {'player_id':>9}  {'residual':>9}  {'predicted':>9}  {'actual':>8}  competition")
    for _, row in merged.iterrows():
        print(
            f"  {int(row['player_id']):>9}  {row['residual']:>9.0f}  {row['predicted']:>9.0f}  "
            f"{row['actual']:>8.0f}  {str(row.get('competition', ''))[:28]}"
        )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from scout import corpus, export, similarity

    try:
        index = _load_index(args.index)
    except similarity.IndexMissing as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        names = export.load_shortlist(args.shortlist)
    except Exception as exc:  # noqa: BLE001 - the message names the file
        print(str(exc), file=sys.stderr)
        return 2

    frame = corpus.load_metrics()
    valuations = None
    valuation_path = Path("data/derived/valuation.parquet")
    if valuation_path.exists():
        import pandas as pd

        valuations = pd.read_parquet(valuation_path)

    rows = export.build_rows(index, frame, names, k=args.k, valuations=valuations)
    # build_provenance fills the scout version, the sportio version and the
    # vocabulary fingerprint itself. Hand-rolling this dict produced a sheet
    # reading "unknown" for exactly the three fields it exists to record.
    provenance = export.build_provenance(index.meta, extra={"shortlist_file": str(args.shortlist)})
    path = export.write_workbook(args.xlsx, rows, provenance=provenance)
    print(f"wrote {path} ({len(rows)} players)")
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    """Launch Streamlit. Exec'd rather than imported so it owns the process."""
    import subprocess

    module = Path(__file__).with_name("app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(module)]
    if args.port:
        command += ["--server.port", str(args.port)]
    return subprocess.call(command)


def cmd_demo(args: argparse.Namespace) -> int:
    from scout import corpus, demo, similarity

    if args.build:
        try:
            index = similarity.PlayerIndex.load(similarity.INDEX_DIR)
        except similarity.IndexMissing as exc:
            print(str(exc), file=sys.stderr)
            return 2
        path = demo.build_bundle(similarity.INDEX_DIR, corpus.load_metrics(), players=args.players)
        print(f"wrote demo bundle to {path}")
        return 0

    if not demo.bundle_exists():
        print(
            "no demo bundle. Build one with `scout demo --build`, which needs a trained index.",
            file=sys.stderr,
        )
        return 2

    index, frame = demo.load_bundle()
    info = demo.manifest()
    print(f"offline demo bundle: {len(index):,} players, built {info.get('built', 'unknown')}")
    print()
    sample = index.players.sort_values("minutes", ascending=False).head(1).iloc[0]
    print(f"nearest comparables for {sample['name']}:")
    for rank, match in enumerate(index.similar(int(sample["player_id"]), k=8), 1):
        print(f"  {rank:>2}. {match.score:.3f}  {match.name[:34]:<34} {match.competition[:22]}")
    print()
    # Only similarity is demonstrated here, so only similarity is caveated. The
    # projection and valuation warnings belong beside their own numbers --
    # attaching them to output that contains neither trains the reader to skip
    # them exactly where they matter.
    print("  Similarity is the part of scout this data supports well.")
    print("  For the projection and valuation caveats, see `scout project` and")
    print("  `scout value`, or reports/projection_power.md.")
    return 0


# --- wiring ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scout", description="Player similarity and valuation")
    parser.add_argument("--version", action="version", version=f"scout {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="versions, vocabulary and what is cached").set_defaults(
        func=cmd_info
    )

    fetch_parser = sub.add_parser("fetch", help="download open data politely")
    fetch_parser.add_argument("what", choices=["spine", "lineups", "events"])
    fetch_parser.set_defaults(func=cmd_fetch)

    sub.add_parser("players", help="player/minutes index and its coverage").set_defaults(
        func=cmd_players
    )

    sub.add_parser(
        "projection", help="can the corpus support a projection head? plus the baselines"
    ).set_defaults(func=cmd_projection)

    build = sub.add_parser("build", help="tokenize cached events into the corpus")
    build.add_argument("--competition", help="restrict to competitions matching this substring")
    build.add_argument("--force", action="store_true", help="rebuild shards that already exist")
    build.set_defaults(func=cmd_build)

    train = sub.add_parser("train", help="train the possession LM (needs the `train` extra)")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch", type=int, default=96)
    train.add_argument("--max-events", type=int, default=None, help="cap the training corpus")
    train.add_argument("--size", choices=["S", "M"], default="S")
    train.add_argument("--device", default="auto")
    train.set_defaults(func=cmd_train)

    embed = sub.add_parser("embed", help="build player embeddings and the query index")
    embed.add_argument("--checkpoint", default="models/possession_lm.pt")
    embed.add_argument("--min-touches", type=int, default=50)
    embed.add_argument("--device", default="auto")
    embed.set_defaults(func=cmd_embed)

    sub.add_parser(
        "fit-value", help="fit the valuation model (needs the `models` extra)"
    ).set_defaults(func=cmd_fit_value)

    rebuild = sub.add_parser(
        "build-index", help="fetch, tokenize, train and embed; resumes if interrupted"
    )
    rebuild.add_argument("--checkpoint", default="models/possession_lm.pt")
    rebuild.add_argument("--epochs", type=int, default=30)
    rebuild.add_argument("--batch", type=int, default=96)
    rebuild.add_argument("--max-events", type=int, default=None)
    rebuild.add_argument("--size", choices=["S", "M"], default="S")
    rebuild.add_argument("--device", default="auto")
    rebuild.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    rebuild.set_defaults(func=cmd_build_index)

    similar = sub.add_parser("similar", help="nearest comparable players")
    similar.add_argument("name")
    similar.add_argument("--k", type=int, default=10)
    similar.add_argument("--min-minutes", type=float, default=0.0)
    similar.add_argument("--position")
    similar.add_argument("--index")
    similar.set_defaults(func=cmd_similar)

    project = sub.add_parser("project", help="projected output in a target competition")
    project.add_argument("name")
    project.add_argument("--to", required=True, help="target competition")
    project.add_argument("--index")
    project.set_defaults(func=cmd_project)

    value = sub.add_parser("value", help="model-implied value against observed")
    value.add_argument("--league")
    value.add_argument("--age", help="not supported; see the error for why")
    value.add_argument("--top", type=int, default=25)
    value.set_defaults(func=cmd_value)

    export_parser = sub.add_parser("export", help="styled Excel shortlist")
    export_parser.add_argument("shortlist", help="a YAML shortlist")
    export_parser.add_argument("--xlsx", required=True)
    export_parser.add_argument("--k", type=int, default=10)
    export_parser.add_argument("--index")
    export_parser.set_defaults(func=cmd_export)

    app = sub.add_parser("app", help="the Streamlit scout card")
    app.add_argument("--port", type=int)
    app.set_defaults(func=cmd_app)

    demo_parser = sub.add_parser("demo", help="run from the bundled offline index")
    demo_parser.add_argument("--build", action="store_true", help="build the bundle")
    demo_parser.add_argument("--players", type=int, default=200)
    demo_parser.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
