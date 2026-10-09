# Hardware drop-test data

`hardware_trials/` holds the 73 drop logs behind the paper's hardware results
(Fig. 4, Table II, Fig. 5). Each CSV is one hand drop of the robot from about
2.8 m (nominal), released at a set roll/pitch attitude that the operator read from
the front IMU. The logs were written by `hardware/controller.py`: logging starts
at the first 50 Hz control step after free fall is detected (front |acc| < 3.5 m/s²)
and runs for 1.0 s of wall clock, which gives 50 rows (49 in two files). The impact
(front accelerometer peak) comes 0.70 to 0.85 s after the first row, so the last
rows of each log are after touchdown.

The files are byte-identical to the logs as first committed from the robot (only
the file names differ); `MANIFEST.csv` gives each file's SHA-256 and original name.

## Layout and naming

```text
hardware_trials/
  MANIFEST.csv
  with_tail/ | no_tail/                  physical configuration of the robot
    roll/roll_045deg/ roll_090deg/ roll_180deg/                    roll sweep, 0° pitch
    roll_pitch/roll_180deg_pitch_+015deg/ _+030deg/ _+045deg/      pitch sweep, 180° roll
      <wt|nt>_roll_<RRR>deg[_pitch_<+PPP>deg]_<MMDD>_rep<NN>.csv
```

`wt` / `nt` repeats the folder (with tail / no tail), `RRR` and `PPP` are the
release roll and pitch in degrees, `MMDD` is the collection-date label (2026), and
`NN` numbers the repetitions within one label. The pitch sweep's 0° point is the
`roll_180deg` folder, so those trials belong to both sweeps.

## Trials per condition

| Release | No tail | With tail |
|---|---|---|
| 45° roll | 6 (0810 ×3, 0816 ×3) | 6 (0810 ×3, 0816 ×3) |
| 90° roll | 7 (0810 ×3, 0816 ×4) | 6 (0810 ×3, 0816 ×3) |
| 180° roll | 6 (0816 ×3, 0817 ×3) | 6 (0810 ×3, 0816 ×3) |
| 180° roll, 15° pitch | 6 (0817) | 6 (0816) |
| 180° roll, 30° pitch | 6 (0817) | 6 (0816) |
| 180° roll, 45° pitch | 6 (0817) | 6 (0817) |
| Total | 37 | 36 |

Date labels are as in the filenames; see Data notes for four of them.

## Columns

One row per control step, in the controller's header order. Each row holds the
sensor values the policy received on that step and the targets it sent.

| Column | Unit | Meaning |
|---|---|---|
| `Time` | s | Wall clock since the controller started (not since the drop). The first row is the first step after free-fall detection; steps are about 20 ms apart. |
| `F_Q0`–`F_Q3` | – | Front-body orientation in the world frame, quaternion (w, x, y, z). Already aligned to the body axes by the controller; this is the quaternion the policy used. |
| `F_M1` | rad | rot1, front spine roll joint (encoder, joint side). |
| `F_M2` | rad | Spine pitch joint. |
| `F_ACC` | m/s² | Front accelerometer magnitude, gravity included. |
| `Cmd_F1` | rad | rot1 target. |
| `Cmd_F2` | rad | Pitch target. |
| `B_Q0`–`B_Q3` | – | Rear IMU quaternion placeholder: always (1, 0, 0, 0). The rear IMU never reported. |
| `B_M1` | rad | Tail joint. |
| `B_M2` | rad | rot2, rear spine roll joint. |
| `B_ACC` | m/s² | Rear accelerometer placeholder: always 0. |
| `Cmd_B1` | rad | Tail target. |
| `Cmd_B2` | rad | rot2 target, always −`Cmd_F1`. |

`Cmd_*` are joint position targets after the controller's action low-pass and
range mapping, not motor commands. Encoder resolution at the joint is 13.5 mrad for
the roll joints and 3.85 mrad for pitch and tail.

Because the rear IMU never reported, the rear-body orientation is reconstructed from
the front IMU and the spine encoders (paper Eq. 2):
`R_rear = R_front · Rx(F_M1) · Ry(F_M2) · Rx(B_M2)`
(`analysis/kinematics.py`).

## MANIFEST.csv

One row per trial file:

| Column | Meaning |
|---|---|
| `file` | Path relative to `hardware_trials/` |
| `morphology` | Physical configuration, from the folder (`with_tail` / `no_tail`) |
| `release_roll_deg`, `release_pitch_deg`, `date_label`, `rep` | From the filename |
| `n_rows` | Number of logged control steps |
| `original_robot_filename` | Name under which the log was first committed from the robot (development repository, branch `final`; not part of this release) |
| `first_commit_date` | Earliest commit, on any branch of the development repository, containing the identical file |
| `sha256` | SHA-256 of the file |
| `policy_flown` | Policy that produced the logged commands, established by offline replay: `tail` (`policies/cat_controller.onnx`) or `notail` (`policies/cat_controller_notail.onnx`, tail channel zeroed) |
| `included_in_whole_cat` | Whether the trial enters the whole-cat score analysis (Table II, Fig. 5) |
| `whole_cat_score` | Whole-cat score at 2.5 m of fall (`results/trial_metrics.csv`); empty if excluded |
| `notes` | The data notes below that apply to the trial |

## Data notes

**Excluded short drop.** `with_tail/roll/roll_090deg/wt_roll_090deg_0816_rep03.csv`
reached impact (front accelerometer peak) after an estimated 2.42 m of fall, before the
2.5 m point at which the whole-cat score is evaluated. It is excluded from the
whole-cat score, Table II and Fig. 5 (35 with-tail trials there) and included in
Fig. 4 and the 1 m / 2 m tilt values (36 with-tail trials). See
`analysis/README.md`.

**No-tail trials flown with the with-tail policy.** Replaying each log's
observations through both deployed ONNX policies reproduces the logged joint targets
to within a few 1e-15 (normalised action; the other policy is off by order 1) with
exactly one of them (`python tools/verify_onnx.py --replay`;
the result is the `policy_flown` column). All 36 with-tail trials and 18 no-tail
trials were flown with their own policy. The other 19 no-tail trials were flown with
the with-tail policy, the controller having been run with `--variant tail` (its
default at the time): all 18 files in `no_tail/roll_pitch/` and
`no_tail/roll/roll_180deg/nt_roll_180deg_0816_rep03.csv`. In these 19 logs the tail
target and tail encoder (`Cmd_B1`, `B_M1`) move, up to 1.55 rad and 1.49 rad; in
the other 18 no-tail logs both are exactly 0. The files are kept under `no_tail/`,
the robot's physical configuration, and the paper's no-tail results include them.

**Date labels.** Four filename date labels are not supported by the git history of
the logs:

| File | First committed | Original name |
|---|---|---|
| `no_tail/roll/roll_180deg/nt_roll_180deg_0817_rep01.csv` | 2026-08-10 | `180deg_NTsuccess_0810_1.csv` |
| `no_tail/roll/roll_180deg/nt_roll_180deg_0817_rep02.csv` | 2026-08-10 | `180deg_NTsuccess_0810_2.csv` |
| `no_tail/roll/roll_180deg/nt_roll_180deg_0817_rep03.csv` | 2026-08-10 | `180deg_NTsuccess_0810_3.csv` |
| `no_tail/roll/roll_180deg/nt_roll_180deg_0816_rep03.csv` | 2026-08-17 | `NT180_0817_3.csv` |

The files keep the names used for the paper's analysis. The date label enters the
whole-cat statistics as a model term and defines the permutation blocks.
