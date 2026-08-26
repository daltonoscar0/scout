"""The standing caveats, defined once.

Every surface that shows a projected or a valuation number has to show the
limitation with it: the CLI, the Streamlit card, the Excel export, the demo. The
first attempt stated each caveat "once" in three different modules, and within a
day they had drifted into two different wordings that could both appear in the
same workbook.

They live here, imported everywhere, so there is exactly one wording and one
place to change it. This module imports nothing but the standard library, so any
part of the codebase can use it without a dependency question.
"""

from __future__ import annotations

from typing import Final

# The measurement behind this is in reports/projection_power.md: over every
# lineup in the open corpus, only 275 female and 211 male players have a genuine
# club-to-international move with adequate minutes on both sides, against a
# 65,000-parameter target.
PROJECTION_CAVEAT: Final = (
    "Identity projection: the player's source-competition output, unchanged. "
    "The open corpus contains only 275 female and 211 male players with a "
    "genuine club-to-international move and adequate minutes on both sides, and "
    "no fitted predictor beat the identity null outside its bootstrap confidence "
    "interval. Read this as 'what they did', not 'what they will do'."
)

VALUATION_TARGET: Final = "next_season_minutes_at_tier"

VALUATION_CAVEAT: Final = (
    "Not money. This is predicted minutes played next season at the same or a "
    "higher competition tier -- a scarcity proxy derived from StatsBomb open "
    "lineups alone."
)

VALUATION_LICENCE: Final = (
    "Derived from StatsBomb open lineups only. Not a transfer fee. Transfermarkt "
    "market value is not used: its terms prohibit automated access and prohibit "
    "training models on the content, and all of its competitions are men's "
    "football while much of this corpus is not."
)

# Used as an Excel column header and a table heading, so the unit travels with
# the number even when a screenshot of the column is all anyone sees.
VALUATION_COLUMN: Final = "Implied value (next-season minutes at tier, not money)"
