"""The scout command line.

Deliberately torch-free. Only `scout train` and `scout embed` need torch, and
they are imported inside those subcommands so that everything else -- including
the Streamlit app, which shells nothing but still shares this module -- starts
in well under a second. `import torch` alone costs 2-4 seconds cold, which would
consume most of the sub-5-second latency budget before any work happened.

Commands that require an artifact that has not been built yet say so and exit
non-zero, rather than failing deep inside a traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scout import __version__


def _spine(path: str = "data/raw/spine.json") -> list[dict]:
    file = Path(path)
    if not file.exists():
        raise SystemExit(
            f"{path} is missing. Run `scout fetch spine` first -- it indexes every "
            "competition and season and is the input to everything else."
        )
    return json.loads(file.read_text())


def cmd_info(args: argparse.Namespace) -> int:
    from scout import fetch, sportio_adapter, vocab

    print(f"scout {__version__}")
    try:
        print(f"sportio {sportio_adapter.sportio_version()}")
    except Exception as exc:  # noqa: BLE001 - the message is the point
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
    free = fetch.free_bytes(Path("."))
    print(f"  free disk         {free / 1024**3:.1f} GiB")
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

    print(f"unknown fetch target: {args.what}", file=sys.stderr)
    return 2


def cmd_players(args: argparse.Namespace) -> int:
    from scout import crosswalk

    rows = crosswalk.build()
    missing = crosswalk.build.missing_lineups
    print(f"player-season rows: {len(rows):,} from {crosswalk.build.matches_read:,} matches")
    if missing:
        print(f"  {missing:,} matches have no cached lineup -- this index is partial")
    players = {row.player_id for row in rows}
    print(f"distinct players: {len(players):,}")
    print()
    for floor in (0, 90, 270, 450, 900):
        qualifying = crosswalk.multi_competition(rows, floor)
        print(f"  in 2+ competition-seasons at >={floor:>4} min each: {len(qualifying):,}")
    return 0


def cmd_projection(args: argparse.Namespace) -> int:
    from scout import crosswalk, population

    rows = crosswalk.build()
    if crosswalk.build.missing_lineups:
        print(
            f"warning: {crosswalk.build.missing_lineups:,} matches have no cached lineup, "
            "so these counts are a lower bound",
            file=sys.stderr,
        )
    result = population.analyse(rows)
    text = population.report(result)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/projection_power.md").write_text(text)
    print(text)
    ok, _ = population.verdict(result)
    # Non-zero when the data cannot support the module, so this is usable as a
    # gate rather than something a reader has to interpret.
    return 0 if ok else 1


def cmd_build(args: argparse.Namespace) -> int:
    from scout import corpus, crosswalk, tokenizer

    spine = _spine()
    if args.competition:
        wanted = args.competition.lower()
        spine = [cs for cs in spine if wanted in cs["comp"].lower()]
        if not spine:
            print(f"no competition matching {args.competition!r}", file=sys.stderr)
            return 2

    # Read the lineup crosswalk once for the whole build rather than per shard.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scout", description=__doc__)
    parser.add_argument("--version", action="version", version=f"scout {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="versions, vocabulary, and what is cached").set_defaults(
        func=cmd_info
    )

    fetch_parser = sub.add_parser("fetch", help="download open data politely")
    fetch_parser.add_argument("what", choices=["spine"], help="what to fetch")
    fetch_parser.set_defaults(func=cmd_fetch)

    players = sub.add_parser("players", help="player/minutes index and its coverage")
    players.set_defaults(func=cmd_players)

    projection = sub.add_parser(
        "projection", help="measure whether the corpus can support a projection head"
    )
    projection.set_defaults(func=cmd_projection)

    build = sub.add_parser("build", help="tokenize cached events into the corpus")
    build.add_argument("--competition", help="restrict to competitions matching this substring")
    build.add_argument("--force", action="store_true", help="rebuild shards that already exist")
    build.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
