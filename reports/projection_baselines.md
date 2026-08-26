# Projection: the honest three-way comparison

Predicting a player's output in a target competition from their output in
a source competition. Every predictor is scored on the same pairs, with a
1000-sample bootstrap confidence interval on mean absolute error.

| predictor | what it does |
| --- | --- |
| B0 identity | predict the source value unchanged -- the null hypothesis |
| B1 league offset | identity plus a fitted per-competition-pair shift |
| B1b shrunk identity | ridge on the source value alone -- no player information |
| B2 embedding ridge | low-rank embedding projection, plus the source value |

B1 and B2 have fitted parameters and are scored **out of sample**, by
5-fold cross-validation grouped by player so no player is ever trained on
and tested on at once. This matters: scored in sample, B2 beats identity
using embeddings made of pure random noise -- ten free parameters will fit
a few hundred pairs. An in-sample table would be wrong in precisely the
direction that flatters the model.

**Age is absent.** The spec asks for age as a feature. StatsBomb lineups
carry no date of birth and the only open source that does is Transfermarkt,
whose terms prohibit this use. The feature is reported missing, not faked.

## xg_xa_per90, male (980 pairs)

| predictor | MAE | RMSE | 95% CI on MAE | n |
| --- | ---: | ---: | :---: | ---: |
| B0 identity | 0.146 | 0.204 | [0.136, 0.155] | 980 |
| B1 league offset | 0.157 | 0.213 | [0.148, 0.166] | 980 |
| B1b shrunk identity | 0.153 | 0.202 | [0.144, 0.161] | 980 |
| B2 embedding ridge | 0.146 | 0.199 | [0.138, 0.155] | 980 |

**No predictor beats identity outside its confidence interval.** At this sample size the difference is not resolvable, which is the expected result given the population measured in `projection_power.md`. scout returns the identity projection and says so wherever a projected number appears.

## progressive_actions_per90, male (980 pairs)

| predictor | MAE | RMSE | 95% CI on MAE | n |
| --- | ---: | ---: | :---: | ---: |
| B0 identity | 2.147 | 2.737 | [2.045, 2.254] | 980 |
| B1 league offset | 2.083 | 2.689 | [1.978, 2.181] | 980 |
| B1b shrunk identity | 1.959 | 2.523 | [1.868, 2.059] | 980 |
| B2 embedding ridge | 1.835 | 2.417 | [1.739, 1.934] | 980 |

**The embedding beats identity, but not the shrinkage control.** A ridge that knows nothing about the player does about as well, so the gain is regression to the mean on a noisy target rather than evidence that the embedding knows how this player translates. scout returns the identity projection.

## xg_xa_per90, female (925 pairs)

| predictor | MAE | RMSE | 95% CI on MAE | n |
| --- | ---: | ---: | :---: | ---: |
| B0 identity | 0.155 | 0.230 | [0.145, 0.166] | 925 |
| B1 league offset | 0.160 | 0.224 | [0.150, 0.171] | 925 |
| B1b shrunk identity | 0.164 | 0.219 | [0.155, 0.174] | 925 |
| B2 embedding ridge | 0.135 | 0.188 | [0.127, 0.145] | 925 |

**B2 embedding ridge beats both identity and the shrinkage control** outside their confidence intervals. On this stratum the embedding carries cross-competition information that shrinkage alone does not explain.

## progressive_actions_per90, female (925 pairs)

| predictor | MAE | RMSE | 95% CI on MAE | n |
| --- | ---: | ---: | :---: | ---: |
| B0 identity | 2.389 | 3.044 | [2.281, 2.510] | 925 |
| B1 league offset | 2.223 | 2.872 | [2.115, 2.349] | 925 |
| B1b shrunk identity | 2.108 | 2.679 | [2.009, 2.213] | 925 |
| B2 embedding ridge | 1.972 | 2.530 | [1.879, 2.070] | 925 |

**The embedding beats identity, but not the shrinkage control.** A ridge that knows nothing about the player does about as well, so the gain is regression to the mean on a noisy target rather than evidence that the embedding knows how this player translates. scout returns the identity projection.
