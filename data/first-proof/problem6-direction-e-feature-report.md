# Problem 6: Direction E Feature Operationalization Report

- seed: `11`
- nmax: `40`
- eps: `[0.1, 0.12, 0.15, 0.2, 0.25, 0.3]`
- samples (UST): `40`
- c_step: `0.5`
- hard threshold on kappa_row: `1.6`
- Case-2b instances: `153`
- Matched row count (with UST kappa + features): `269`
- Hard rows: `30`

## Top Feature Correlations

| feature | corr(kappa) | mean | p90 | max |
|---|---:|---:|---:|---:|
| `deg_r_max` | 0.973 | 1.539 | 3.000 | 5.000 |
| `deg_r_mean` | 0.831 | 1.298 | 2.000 | 5.000 |
| `m_cross` | 0.775 | 28.833 | 60.200 | 175.000 |
| `s_size` | 0.726 | 1.870 | 3.000 | 5.000 |
| `deg_r_cv` | 0.608 | 0.085 | 0.353 | 0.531 |
| `deg_r_min` | 0.572 | 1.190 | 2.000 | 5.000 |
| `active` | 0.475 | 20.030 | 35.000 | 39.000 |
| `r_size` | 0.475 | 20.030 | 35.000 | 39.000 |
| `gl_gain` | 0.472 | 10.544 | 25.540 | 41.076 |
| `gl_margin` | 0.454 | 10.442 | 25.540 | 41.076 |
| `x_headroom` | 0.431 | 0.055 | 0.312 | 0.556 |
| `reff_diam_i0` | -0.416 | 0.996 | 2.500 | 4.477 |

## Best Single-Feature Hard-Row Classifiers

| feature | rule | precision | recall | f1 | tp/fp/fn |
|---|---|---:|---:|---:|---|
| `deg_r_max` | `deg_r_max >= 3.0000` | 0.833 | 1.000 | 0.909 | 30/6/0 |
| `s_size` | `s_size >= 3.0000` | 0.469 | 1.000 | 0.638 | 30/34/0 |
| `deg_r_mean` | `deg_r_mean >= 1.5000` | 0.491 | 0.900 | 0.635 | 27/28/3 |
| `m_cross` | `m_cross >= 56.0000` | 0.588 | 0.667 | 0.625 | 20/14/10 |
| `lambda2_cross` | `lambda2_cross >= 2.0000` | 0.857 | 0.400 | 0.545 | 12/2/18 |
| `deg_r_cv` | `deg_r_cv >= 0.4126` | 0.714 | 0.333 | 0.455 | 10/4/20 |
| `lambda_max_cross` | `lambda_max_cross >= 23.2474` | 0.333 | 0.600 | 0.429 | 18/36/12 |
| `active` | `active >= 25.0000` | 0.280 | 0.867 | 0.423 | 26/67/4 |
| `r_size` | `r_size >= 25.0000` | 0.280 | 0.867 | 0.423 | 26/67/4 |
| `reff_diam_i0` | `reff_diam_i0 <= 0.2732` | 0.248 | 0.967 | 0.395 | 29/88/1 |
| `gl_gain` | `gl_gain >= 22.1674` | 0.341 | 0.467 | 0.394 | 14/27/16 |
| `gl_margin` | `gl_margin >= 22.1674` | 0.341 | 0.467 | 0.394 | 14/27/16 |

## Interpretation

Use this table to choose a concrete Direction-E parameter `P_t` and threshold for a theorem attempt.
A useful candidate should separate hard rows (high recall) with manageable false positives.
