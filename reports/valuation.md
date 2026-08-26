# Valuation

Target: **next_season_minutes_at_tier** -- minutes played the following season at the same or a
higher competition tier, computed from StatsBomb lineups alone.

This is not money. The spec asks to model Transfermarkt market value, but
Transfermarkt's terms prohibit automated access and prohibit training models
on their content by name, and all 65 of their competitions are men's football
while a large part of this corpus is not. A minutes-at-tier target is legally
clean, covers everyone, and is a genuine scarcity proxy. Market value is
available behind `--enable-transfermarkt`, fetched at build time and never
committed.

## Features the spec asks for that open data cannot supply

| feature | why not |
| --- | --- |
| age | StatsBomb lineups carry no date of birth. |
| contract_length | No open source covers contracts. |
| market_value | Transfermarkt's terms prohibit this use; opt in with --enable-transfermarkt. |

## Fit

Validated on held-out **seasons**, not random rows: a random split puts a
player's 2015/16 row in train and their 2016/17 row in test.

| | |
| --- | ---: |
| training rows | 4,170 |
| held-out rows | 1,308 |
| MAE (minutes) | 560.6 |
| baseline MAE (predict the mean) | 526.3 |
| rank correlation with actual | -0.014 |

The model DOES NOT beat predicting the training mean.

## Why the fit looks like this

Only 70.7% of rows are labelled by a genuinely consecutive season. For the rest the gap is 2 to 16 years, because open coverage of any one competition is intermittent. Where the gap is large the label means "whenever we next observed them", not "next season", and this season's production should not be expected to predict it.

## Does the residual predict the next window?

The spec requires this answer in the README even if it is no.

**yes, weakly. Spearman +0.180 over 3278 pairs. The residual persists somewhat between windows, but this is a rank correlation on a proxy target, not evidence of a tradable edge.**
