"""The only module in scout that is allowed to import sportio.

sportio is being developed concurrently in a separate working tree, installed
here as an editable path dependency. That means its public API can change under
us between one command and the next. Confining every import to this file turns
"scout is broken" into "one adapter function needs updating", and the drift
check below turns a confusing AttributeError deep in the tokenizer into a clear
message naming exactly which symbol disappeared.

Observed sportio state when this adapter was written:
    version 0.1.0
    HEAD    dfa356e ("docs: record the skillcorner event frame and the metrica
            period-clock trap", 2026-08-25 17:58)
    exports load_events, load_tracking, corners, set_pieces, freeze_frames,
            conform_events, validate_events, EVENT_COLUMNS, sources, ...

Two things sportio does NOT yet provide, which scout therefore builds itself:

  * A player/minutes crosswalk (player_id -> {(competition, season): minutes}).
    scout derives this from StatsBomb lineups in `scout.crosswalk`. If sportio
    later grows a `player_index()`, prefer it and delete ours.
  * Any notion of a token or an embedding. Those are scout's own concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

# Symbols scout actually relies on. Kept deliberately short: every name here is
# a hostage to another session's refactor, so we take only what we need.
_REQUIRED = (
    "load_events",
    "conform_events",
    "validate_events",
    "EVENT_COLUMNS",
    "extra_field",
)


class SportioApiDrift(RuntimeError):
    """sportio no longer exposes something scout depends on."""


def _sportio() -> Any:
    try:
        import sportio
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SportioApiDrift(
            "sportio is not importable. It is an editable path dependency on "
            "../sportio; run `uv sync` in the scout repo."
        ) from exc

    missing = [name for name in _REQUIRED if not hasattr(sportio, name)]
    if missing:
        raise SportioApiDrift(
            "sportio is installed but no longer exports: "
            + ", ".join(missing)
            + ". sportio is developed in a separate session and its API has "
            "drifted. Update scout/sportio_adapter.py to match -- no other "
            "scout module imports sportio, so this file is the whole fix."
        )
    return sportio


def sportio_version() -> str:
    """Version string, recorded into every artifact scout writes for provenance."""
    return str(getattr(_sportio(), "__version__", "unknown"))


def event_columns() -> list[str]:
    """The canonical 18-column event schema, as sportio defines it today."""
    return list(_sportio().EVENT_COLUMNS)


def load_events(competition: str, season: str | None = None, **kwargs: Any) -> pd.DataFrame:
    """Canonical events for one competition-season.

    Returns sportio's canonical frame unchanged. scout never reaches past this
    into raw StatsBomb JSON -- that is precisely the boundary sportio exists to
    enforce, and duplicating schema knowledge here would defeat the point.
    """
    sportio = _sportio()
    if season is None:
        return sportio.load_events("statsbomb", competition=competition, **kwargs)
    return sportio.load_events("statsbomb", competition=competition, season=season, **kwargs)


def extra_field(events: pd.DataFrame, key: str) -> pd.Series:
    """Pull one key out of the JSON-encoded `extra` column."""
    return _sportio().extra_field(events, key)


def validate_events(events: pd.DataFrame) -> None:
    """Raise if a frame is not schema-conformant. Cheap insurance at stage boundaries."""
    _sportio().validate_events(events)
