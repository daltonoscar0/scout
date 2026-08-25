"""Polite, resumable, gzip-at-rest fetching of StatsBomb open data.

sportio owns schema normalisation, and scout does not duplicate it. But sportio
does not yet expose lineups or a player/minutes crosswalk, and its cache is not
compressed at rest -- which matters here, because this machine is disk-bound
rather than compute-bound. So scout fetches the raw files it needs itself and
hands the parsed JSON to sportio (or to `scout.crosswalk`) for interpretation.

Three things this module is careful about, each learned from a measurement:

  * Gzip at rest. Events compress about 10.6x and 360 frames about 7.6x. The
    full corpus is 15.4 GiB raw and roughly 1.6 GiB gzipped. We store the gzip
    stream and decompress only in the parser; nothing is ever gunzipped in place.

  * Never HEAD for a size. With gzip negotiated the response is chunked and
    carries no content-length, and `curl -I` on these raw paths returns
    content-length: 0. The size oracle is the git tree manifest instead.

  * Two byte-identical hosts. raw.githubusercontent.com and cdn.jsdelivr.net
    serve the same bytes. Alternating between them keeps each host at 1 req/s
    while halving wall-clock, which turns a ~105 minute lineups pass into ~55.
"""

from __future__ import annotations

import gzip
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import requests

REPO = "statsbomb/open-data"

# Byte-identical mirrors. The repo was renamed statsbomb -> hudl and the GitHub
# API now answers 301 for it, but raw.githubusercontent.com still serves the old
# name transparently. A client that does not follow redirects silently ingests
# the redirect body as data, so allow_redirects stays on everywhere.
HOSTS = (
    f"https://raw.githubusercontent.com/{REPO}/master/data",
    f"https://cdn.jsdelivr.net/gh/{REPO}@master/data",
)

MIN_INTERVAL = 1.0  # seconds between requests to any one host
MAX_RETRIES = 6
# Generous, because the largest files here (a full La Liga season index is
# 700 KB, an events file 3 MB) were the ones that failed at 30s during a burst.
TIMEOUT = 90

# This machine runs several builds concurrently and has sat at 98% full. A fetch
# that fills the disk corrupts in-flight writes in every other repo too, so the
# loop checks free space and stops cleanly rather than pushing through.
MIN_FREE_BYTES = 2 * 1024**3


class DiskTooFull(RuntimeError):
    """Aborted a fetch because free space fell below the safety floor."""


class _HostLimiter:
    """One rate limit per host, so alternating hosts genuinely doubles throughput."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            earliest = self._last.get(host, 0.0) + MIN_INTERVAL
            delay = earliest - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._last[host] = now


_limiter = _HostLimiter()

# Successive fetches alternate hosts. Each host still sees at most 1 req/s, but
# the caller sees 2/s, which is what turns the full lineups pass from ~105
# minutes into ~55. Callers can pin a host explicitly if they need to.
_round_robin = 0
_round_robin_lock = threading.Lock()


def _next_host_index() -> int:
    global _round_robin
    with _round_robin_lock:
        _round_robin += 1
        return _round_robin


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def cache_root(root: Path | str = "data/raw/statsbomb") -> Path:
    return Path(root)


def _cache_path(root: Path, rel: str) -> Path:
    return root / (rel + ".gz")


def read_cached(root: Path, rel: str) -> Any | None:
    """Return parsed JSON from the cache, or None if it is not there.

    A truncated cache file -- the signature of a fetch killed mid-write, or of a
    disk that filled -- is treated as absent and refetched rather than raising.
    """
    path = _cache_path(root, rel)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, EOFError, json.JSONDecodeError, gzip.BadGzipFile):
        path.unlink(missing_ok=True)
        return None


def _write_cached(root: Path, rel: str, payload: bytes) -> None:
    path = _cache_path(root, rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename, so an interrupted fetch never leaves a
    # half-written file that looks like a valid cache entry.
    tmp = path.with_suffix(path.suffix + ".part")
    with gzip.open(tmp, "wb", compresslevel=6) as handle:
        handle.write(payload)
    tmp.replace(path)


def fetch_json(
    rel: str,
    *,
    root: Path | None = None,
    session: requests.Session | None = None,
    host_index: int | None = None,
    force: bool = False,
) -> Any:
    """Fetch one JSON file, using the gzip cache when possible.

    `rel` is a repo-relative path such as "competitions.json" or
    "lineups/3930166.json".
    """
    root = cache_root() if root is None else root
    if not force:
        cached = read_cached(root, rel)
        if cached is not None:
            return cached

    if free_bytes(root if root.exists() else Path(".")) < MIN_FREE_BYTES:
        raise DiskTooFull(
            f"free space below {MIN_FREE_BYTES / 1024**3:.1f} GiB; stopping before "
            "the fetch fills the disk. Cached files are intact -- rerun to resume."
        )

    if host_index is None:
        host_index = _next_host_index()

    owns_session = session is None
    session = requests.Session() if session is None else session
    try:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            base = HOSTS[(host_index + attempt) % len(HOSTS)]
            _limiter.wait(base)
            url = f"{base}/{rel}"
            try:
                response = session.get(
                    url,
                    timeout=TIMEOUT,
                    allow_redirects=True,
                    headers={"Accept-Encoding": "gzip", "User-Agent": "scout/0.1 (+local build)"},
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue

            if response.status_code == 404:
                raise FileNotFoundError(f"{rel} is not in the open-data repo ({url})")
            if response.status_code in (429, 500, 502, 503, 504):
                # Honour Retry-After when the server sends one; it knows better
                # than an exponential guess.
                delay = float(response.headers.get("Retry-After", 2**attempt))
                last_error = RuntimeError(f"HTTP {response.status_code} for {url}")
                time.sleep(min(delay, 60))
                continue
            response.raise_for_status()

            payload = response.content
            parsed = json.loads(payload)
            _write_cached(root, rel, payload)
            return parsed

        raise RuntimeError(
            f"gave up on {rel} after {MAX_RETRIES} attempts; last error: {last_error}"
        ) from last_error
    finally:
        if owns_session:
            session.close()


# --------------------------------------------------------------------------
# The spine: competitions -> matches -> lineups
# --------------------------------------------------------------------------


def competitions(root: Path | None = None) -> list[dict[str, Any]]:
    return fetch_json("competitions.json", root=root)


def matches(competition_id: int, season_id: int, root: Path | None = None) -> list[dict[str, Any]]:
    return fetch_json(f"matches/{competition_id}/{season_id}.json", root=root)


def lineup(match_id: int, root: Path | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    return fetch_json(f"lineups/{match_id}.json", root=root, **kwargs)


def three_sixty_available(match: dict[str, Any]) -> bool:
    """Whether this specific match actually has a 360 file.

    Filter on the per-match `match_status_360`, never on competitions.json's
    `match_available_360`. AFCON 2023 advertises 360 at the competition level but
    only 1 of its 52 matches has a three-sixty file; trusting the competition
    flag issues 51 doomed requests and invents a phantom competition.
    """
    return match.get("match_status_360") == "available"
