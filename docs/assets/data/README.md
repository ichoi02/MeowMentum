# Project page data

`page-data.js` sets `window.PAGE_DATA`, a JSON object, so the page loads it with a
plain `<script>` tag. Opened from `file://`, everything except the 3D replay works (an ES
module cannot load from a file URL); serve `docs/` over http to see the replay. The file is
generated; do not edit it by hand.

## Regenerating

From the repository root:

```bash
python docs/assets/data/build_page_data.py
```

The script reads:

| Input | Used for |
|---|---|
| `data/hardware_trials/**/*.csv` | `trials`, `tilt`, raw columns of `table2` |
| `results/model_results.csv`, `results/sweep_results.csv` (whole-cat analysis output) | adjusted columns of `table2` |
| `policies/student_policy{,_notail}.pth`, `model/cat*.xml`, `cat_env/`, `distillation.py` | `sim`, `meta.model` |
| `results/training_logs/teacher_{tail,notail}/` (TensorBoard) | `training` |

Every path can be overridden: `--root`, `--data`, `--stats`, `--logs`, `--out`.
It needs numpy, scipy, pandas, torch, mujoco, gymnasium, stable-baselines3 (imported by
`distillation.py`) and tensorboard, and takes a few seconds.

## Schema

All floats are rounded to 4 decimals. Joint angles (`j`, joint `range`) are in radians;
release angles (`roll`, `pitch`) and tilts (`tilt`, `mean`, `sd`) are in degrees.
Quaternions are `[w, x, y, z]` and rotate body to world. Joint vectors are always
`[rot1, pitch, rot2, tail]`.

### `trials` (73 hardware drops)

| Field | Meaning |
|---|---|
| `id` | file stem, e.g. `nt_roll_180deg_pitch_+015deg_0817_rep01` |
| `morph` | physical configuration from the folder: `tail` or `notail` |
| `roll`, `pitch` | release condition (deg) |
| `date`, `rep` | session label and repeat number, as written in the file name. Four labels do not match the flight date; see `data/hardware_trials/MANIFEST.csv` (`first_commit_date`, `notes`) |
| `included` | `false` only for `wt_roll_090deg_0816_rep03`, whose impact came before 2.5 m of fall |
| `score`, `s_up`, `s_align` | whole-cat score `S_upright * S_align` and its two factors at 2.5 m of ballistic fall (`t0 + 0.714 s`, both body rotations SLERP-interpolated); `null` when excluded |
| `policy` | policy that flew: `tail` if the tail command `Cmd_B1` is ever nonzero, else `notail` (the no-tail controller pins it to 0). This agrees with an ONNX replay of every log. 19 `notail` drops flew the `tail` policy |
| `impact_t` | s from the first sample to blanket impact (the front accelerometer peak); samples after it are post-landing |
| `t` | seconds since the first sample (free-fall detection) |
| `q` | front-body world quaternion per sample (`F_Q0..F_Q3`, as logged) |
| `j` | joint angles per sample: `F_M1, F_M2, B_M2, B_M1` |
| `tilt` | mean body tilt per sample (deg), defined as for `tilt` below |

The rear IMU never reported, so the rear body is reconstructed as
`R_rear = R_front * Rx(rot1) * Ry(pitch) * Rx(rot2)`. The heading (yaw) in `q` is
whatever the IMU reported, so it differs from drop to drop; tilt and score do not
depend on it.

### `tilt` (Fig. 4, 12 curves)

`{morph, roll, pitch, n, t[33], mean[33], sd[33]}`. A body's tilt is `|roll| + |pitch|`
of its extrinsic xyz Euler angles; each drop's tilt is the mean of the front and rear
bodies. Runs are stacked by control step over the first 33 steps, `t` is the mean of
the runs' timestamps, and `sd` is the population SD. Every trial in the condition
folder is used, including the excluded short drop, as in the paper's Fig. 4.

### `table2` (Table II, 3 rows)

`{analysis: all|roll|pitch, label, notail: {mean, sd, n}, tail: {mean, sd, n}, effect,
ci: [lo, hi], p, p_holm, p_perm}`. `mean` and `sd` (sample SD) are of the included
whole-cat scores. The roll sweep is roll 45/90/180 at pitch 0; the pitch sweep is roll
180 at pitch 0/15/30/45. `effect`, `ci` and `p` are the HC3 ANCOVA tail effect,
`p_holm` is the analysis's `p_holm_three_requested` column (`null` for the primary
all-trials row), and `p_perm` is the blocked permutation p.

### `sim` (12 rollouts)

`{morph, roll, pitch, t[51], q[51], j[51], tilt[51]}`: one rollout of each deployed
student in MuJoCo per release condition. Physics is nominal (the per-episode domain
randomization is overwritten with the `model/*.xml` values and the action delay is 0).
The release has zero angular velocity and zero joint angles, and observations are
noise-free. The student observation and no-tail masking match `evaluate.py`. Sample k
is the state at `t = 0.02 k` that the k-th action is computed from; the last sample is
the end of the 1 s episode. `tilt` uses the simulated front and rear body orientations.

### `training` (2 curves)

`{morph, step[150], up_mean[150], r_pos[150]}` from the teacher TensorBoard logs.
`up_mean` is the per-step uprightness `0.5 (u_f + u_r)` and `r_pos` the uprightness
reward. Each point is the mean of consecutive logged values, placed at the mean
timestep of its block.

### `meta`

- `robot`: mass and length with and without the tail, policy and PD rates, the
  free-fall trigger and the episode length, all from the paper.
- `evaluation`: the score's fall distance and time, and the Fig. 4 window.
- `model`: geometry from `model/cat.xml`, in metres and kilograms, in the front-body
  frame with every joint at 0:
  - box half-sizes and body centres
  - spine centre
  - tail pivot, tail tip and tip mass
  - paw spheres
- `joints`: axis, origin and range of each joint.
- `kinematics`: the formulas that place every part from `q` and `j`.
