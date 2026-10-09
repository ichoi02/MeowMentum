# Hardware drop analysis

Scripts that turn the 73 hardware drop logs in `data/hardware_trials/` into the
paper's hardware results: Fig. 4, the 1 m / 2 m tilt values in Section III-A,
Table II and Fig. 5.

## Reproduce everything

From the repository root:

```bash
python analysis/reproduce.py
```

This runs `whole_cat_stats.py`, `fig4_tilt.py` and `fig5_whole_cat.py`, writes
everything under `results/`, and then checks `results/table2.csv` and
`results/tilt_milestones.csv` against the values printed in the paper
(`analysis/expected_values.json`) at the paper's precision. It exits non-zero if
any value differs. A full run takes about 30 s, almost all of it in the
blocked permutation tests. Use `--results <dir>` to write somewhere other than
`results/`.

Requirements: numpy, scipy >= 1.14 (for `scalar_first` quaternions), pandas and
matplotlib. Tested with numpy 2.2.6, scipy 1.15.3, pandas 2.3.3 and
matplotlib 3.9.4. The scripts are run as files (`python analysis/<name>.py`); they
import `kinematics.py` from their own folder.

## Scripts

| Script | What it does | Paper |
|---|---|---|
| `kinematics.py` | Shared code: trial file discovery and naming, CSV loading, the \|roll\| + \|pitch\| tilt, rear-body reconstruction, free-fall time/distance | – |
| `whole_cat_stats.py` | Whole-cat score per trial, ANCOVA with HC3 errors, Holm adjustment, blocked permutation tests, Table II | Table II, Sec. III-B to III-D |
| `fig4_tilt.py` | Mean body tilt over the fall per release condition; prints and saves the 1 m / 2 m values | Fig. 4, Sec. III-A |
| `fig5_whole_cat.py` | Whole-cat score box plots per release condition, from `results/trial_metrics.csv` | Fig. 5 |
| `plot_drop.py` | One drop: front/rear tilt and the commanded joint targets | – (diagnostic) |
| `reproduce.py` | Runs the three result scripts in order and checks the numbers | – |

Each script takes `--help`. Figures are written as SVG with editable text; an
`--out` name ending in `.pdf` is honoured and any other extension is replaced by
`.svg`.

## Outputs in `results/`

| File | Contents |
|---|---|
| `trial_metrics.csv` | One row per included trial (72): metadata, impact time and estimated fall distance, the whole-cat score (`posture_score`) and its components at the evaluation instant, diagnostics, and the initial-state covariates |
| `excluded_trials.csv` | Trials dropped by the exclusion rule, with the reason (1 trial) |
| `model_results.csv` | All-trials ANCOVA of the tail effect, one row per outcome: the score, its components, and angle/rate diagnostics. `p_holm_all_outcomes` is Holm over these 13 rows |
| `sweep_results.csv` | Roll-sweep and pitch-sweep ANCOVAs of the score. `p_holm_three_requested` is the Holm value printed in Table II; `p_holm_two_sweeps` is Holm over the two sweeps alone |
| `pitch_level_analysis.csv` | The pitch-leveling test (third member of the Holm family, see below) |
| `table2.csv`, `table2.md` | Table II: raw mean ± sample SD (n) per arm, adjusted tail effect, 95% HC3 CI, HC3 p, Holm p, blocked permutation p, plus the permutation subset size and residual df |
| `REPORT.md` | The tables above with the metric and model definitions |
| `tilt_milestones.csv` | Pooled mean body tilt at 1 m and 2 m of fall per morphology, with population and sample SD |
| `figures/fig4_tilt.svg` | Fig. 4 |
| `figures/fig5_whole_cat.svg` | Fig. 5 |

`results/training_logs/` holds the teacher training logs and is not written by
these scripts.

## Definitions

**Time base.** Each log starts at the first control step after free fall was
detected (front |acc| < 3.5 m/s²), so the first row is t0 = release, up to a few
tens of ms of detection latency. The logged `Time` column counts from when the
operator armed the controller (pressed Enter), not from the drop; analyses use
time relative to the first row.

**Rear body.** The rear IMU never reported (`B_Q` is identity and `B_ACC` is 0 in
every log), so the rear-body orientation is reconstructed from the front IMU and
the spine encoders (paper Eq. 2):
`R_rear = R_front · Rx(rot1) · Ry(pitch) · Rx(rot2)`, with rot1 = `F_M1`,
pitch = `F_M2`, rot2 = `B_M2`. The tail (`B_M1`) is a child of the rear body and
does not enter.

**Fig. 4 tilt.** Per body, tilt = |roll| + |pitch| from the extrinsic `xyz` Euler
decomposition of its world orientation (yaw drops out; a 180° roll / 15° pitch
release starts at 195°). The front and rear tilts are averaged per trial and step,
and each condition's line and band are the mean and ± population SD (ddof=0)
across its trials. Trials are stacked by control-step index, not resampled; the
first 33 steps (about 0.64 s, 2 m of fall) are shown, and step k is drawn at the
mean of the trials' timestamps at k. Groups have 6 trials, except 7 for no tail at
90° roll.

**1 m / 2 m tilt (Section III-A).** The same per-trial mean tilt, read from the
sample nearest 0.452 s and 0.639 s (the free-fall times of 1 m and 2 m), pooled over
all trials of a morphology. The paper quotes mean ± population SD (ddof=0):
with tail 42.3 ± 22.5° and 29.6 ± 14.2° (n = 36), no tail 51.2 ± 24.7° and
35.6 ± 18.2° (n = 37). `tilt_milestones.csv` also lists the sample SDs (22.9, 14.4,
25.1, 18.5). Fig. 4 and these values include all 73 trials, also the short drop
that the whole-cat analysis excludes; its impact (step 35, 0.70 s) comes after both
the plotted window and the 2 m sample. Without it the with-tail values are
42.4 ± 22.9° and 29.0 ± 13.9° (n = 35, ddof=0).

**Whole-cat score.** With `up_F` and `up_R` the world-frame +z axes of the two bodies:

```text
S_upright = 1 - acos(z · normalize(up_F + up_R)) / pi
S_align   = 1 - theta / pi,   theta = rotation angle of R_front^-1 R_rear
S_whole   = S_upright * S_align
```

It is evaluated at one instant, t0 + sqrt(2 · 2.5 m / g) = t0 + 0.714 s, the moment
of 2.5 m of ballistic fall. Front and rear rotations at that instant are
SLERP-interpolated between the logged samples up to impact. The score is invariant
to common yaw; relative twist between the bodies lowers `S_align`.

**Impact and exclusion.** Impact is the sample with the largest acceleration
(max of `F_ACC` and `B_ACC`; `B_ACC` is always 0, so this is the front
accelerometer peak). A trial is excluded when its impact comes before the
evaluation instant, i.e. the drop was shorter than 2.5 m. This excludes one trial,
`with_tail/roll/roll_090deg/wt_roll_090deg_0816_rep03.csv` (impact at an estimated
2.42 m), so Table II and Fig. 5 have 35 with-tail trials. The estimated fall
distance (g/2 · (t_impact − t0)²) of the 72 included trials is 2.56 to 3.55 m.

**Model.** OLS of the score on an intercept, a with-tail indicator, release
condition dummies, collection-date dummies (filename date labels 0810 / 0816 /
0817), and two z-scored covariates: the initial whole-cat score (first sample) and
the initial front-body angular speed (norm of the median of the first five
angular-velocity samples, from consecutive front quaternions with a 15 Hz
zero-phase Butterworth low-pass). The adjusted tail effect is the with-tail
coefficient; its standard error is HC3 and the 95% CI and p use a t reference with
the residual df (61 for all trials, 28 for the roll sweep, 39 for the pitch sweep).
The roll sweep is 45/90/180° roll at 0° pitch (36 trials); the pitch sweep is
180° roll at 0/15/30/45° pitch (48 trials), so the 180° roll trials enter both.
Table II's raw values are mean ± sample SD (ddof=1).

**Blocked permutation test.** Tail labels are shuffled within condition × date
blocks, 5000 permutations, seed 20260819, and the same model is refit each time;
p = (1 + #{|β_perm| ≥ |β_obs|}) / 5001. Only blocks that contain both
morphologies can be permuted, so the test, including its observed coefficient,
uses only those trials: 42 of 72 for all trials, 30 of 36 for the roll sweep and
18 of 48 for the pitch sweep (`n_overlap` / `n_permutation`).

**Holm adjustment.** The Holm-adjusted p-values in Table II (0.0056 roll sweep,
0.3682 pitch sweep) come from a family of three tests: the roll sweep, the pitch
sweep, and a pitch-leveling test (tail effect on the final absolute whole-cat pitch
in the pitch sweep, HC3 p = 0.858; `pitch_level_analysis.csv`). The all-trials
analysis is the primary one and has no Holm value. Holm over the two sweeps alone
would give 0.0038 and 0.1841 (`p_holm_two_sweeps`).

**Other columns in `sweep_results.csv`.** `estimate_no_date` / `p_no_date` refit the
model without the date dummies; `overlap_estimate` is the tail coefficient on the
permutable (overlap-block) trials only; `mde_80` is the smallest tail effect the sweep
could detect with 80% power at α = 0.05, given its fitted residual SD.

## Evaluation-distance sensitivity

`whole_cat_stats.py --distance D --output <dir>` scores every trial at D metres of
fall instead (trials whose impact comes earlier are excluded). Write these runs to
a separate folder so `results/` keeps the 2.5 m analysis. Adjusted tail effect
(95% HC3 CI; HC3 p; blocked permutation p):

| Distance | Trials | All trials | Roll sweep | Pitch sweep |
|---:|---:|---|---|---|
| 2.5 m (paper) | 72 | +0.157 [0.062, 0.252]; 0.0016; 0.0008 | +0.185 [0.075, 0.296]; 0.0019; 0.0004 | +0.114 [−0.057, 0.286]; 0.1841; 0.0958 |
| 2.8 m | 69 | +0.151 [0.051, 0.251]; 0.0038; 0.0016 | +0.166 [0.030, 0.302]; 0.0188; 0.0028 | +0.159 [−0.020, 0.337]; 0.0801; 0.0594 |
| 3.0 m | 58 | +0.100 [−0.016, 0.215]; 0.0892; 0.0454 | +0.119 [−0.043, 0.281]; 0.1420; 0.0656 | +0.136 [−0.052, 0.325]; 0.1502; 0.1120 |

## Data notes

`data/README.md` describes the logs, and `data/hardware_trials/MANIFEST.csv` lists
per trial the policy that flew it and its provenance. Three points bear on the
results above: the excluded short drop; 19 no-tail trials flown with the with-tail
policy (all 18 pitched releases and one 180° roll release, i.e. 19 of the 24 no-tail
pitch-sweep trials and 1 of the 19 no-tail roll-sweep trials); and four filename
date labels that git history does not support. Date labels enter the model
as fixed effects and define the permutation blocks.
