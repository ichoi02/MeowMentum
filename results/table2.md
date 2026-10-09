# Table II: whole-cat self-righting scores

| Analysis | No tail mean ± SD (n) | With tail mean ± SD (n) | Adjusted tail effect | 95% HC3 CI | HC3 p | Holm-adjusted p | Blocked permutation p |
|---|---|---|---:|---|---:|---:|---:|
| All trials | 0.688 ± 0.155 (37) | 0.798 ± 0.090 (35) | +0.157 | [0.062, 0.252] | 0.0016 | — | 0.0008 |
| Roll sweep | 0.627 ± 0.140 (19) | 0.807 ± 0.086 (17) | +0.185 | [0.075, 0.296] | 0.0019 | 0.0056 | 0.0004 |
| Pitch sweep | 0.715 ± 0.164 (24) | 0.789 ± 0.092 (24) | +0.114 | [-0.057, 0.286] | 0.1841 | 0.3682 | 0.0958 |

- Raw scores: mean ± sample SD (ddof=1), number of trials in parentheses. The adjusted tail effect (with tail minus no tail) is the OLS coefficient after controlling for release condition, collection date, initial whole-cat score and initial angular speed; CI and p use HC3 standard errors with a t reference.
- Holm-adjusted p: Holm over three tests -- the roll sweep, the pitch sweep and the pitch-leveling test in pitch_level_analysis.csv (`p_holm_three_requested` in sweep_results.csv). Holm over the two sweeps alone is `p_holm_two_sweeps`.
- Blocked permutation p: tail labels shuffled within condition x date blocks; only blocks containing both morphologies are used (n_permutation trials).
