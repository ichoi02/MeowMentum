# MeowMentum

**A multi-strategy aerial self-righting robot for evaluating the contribution of an actuated tail across fall conditions.**

[Project page](https://ichoi02.github.io/MeowMentum/) · [Drop data](results/hardware_trials) · [Results](results)

<p align="center"><img src="docs/assets/img/robot.jpg" width="640" alt="The robot: two black cylindrical body segments joined by an actuated spine, with a tail rod on the rear segment and red feet"></p>

MeowMentum is a 1.86 kg robot that rights itself in free fall. Two body segments are
joined by an actuated two-degree-of-freedom spine (pitch plus a coupled roll pair), and a
one-degree-of-freedom tail is mounted on the rear segment. A policy trained only in
MuJoCo, first as a privileged SAC teacher and then distilled with DAgger into a student
that sees only onboard sensors, runs at 50 Hz on a Raspberry Pi. It was deployed to the
robot without tuning. Dropping the robot with and without its tail from a range of roll
and pitch attitudes shows that the tail improves whole-body righting overall and across
roll releases. Across pitch releases its extra benefit is less clear.

This repository contains the code, the policies that flew, every hardware drop log, and
the analysis that produces the paper's figures and tables.

## Repository layout

| Path | Contents |
|---|---|
| `train/` | Simulation and learning: the Gymnasium/MuJoCo environment (`cat_env/`), SAC teacher training with a temporal smoothness term in the actor loss (`train.py`, `smooth_sac.py`) and DAgger distillation into the 11-dim onboard-observation student (`distillation.py`) |
| `model/` | MuJoCo models for both morphologies (`cat.xml`, `cat_notail.xml`) and the variant table (`variants.py`) |
| `policies/` | The deployed teachers, students and ONNX files ([provenance](#released-policies)) |
| `hardware/` | Raspberry Pi controller, Teensy firmware, flashing tools and the student → ONNX export ([hardware/README.md](hardware/README.md)) |
| `tools/` | Interactive MuJoCo policy viewer, ONNX/telemetry verification and the PD-gain step-response sweep |
| `results/hardware_trials/` | The 73 hardware drop logs at 50 Hz, with a manifest (`MANIFEST.csv`) |
| `results/figures/` | Fig. 4 and Fig. 5 (SVG) |
| `results/training_logs/` | TensorBoard logs of the deployed teachers |
| `results/plot_scripts/` | Whole-cat score, Table II statistics, and the Fig. 4 and Fig. 5 scripts |
| `docs/` | Project page (GitHub Pages) |

## Install

Python 3.10. From the repository root:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The pins are the environment this release was verified in. The robot itself needs only
numpy, scipy, gymnasium, onnxruntime and pyserial (see `hardware/README.md`). Every script
is a module run from the repository root with `python -m`.

## Reproduce the paper's hardware results

```bash
python -m results.plot_scripts.whole_cat_stats   # Table II
python -m results.plot_scripts.fig4_tilt         # Fig. 4 and the 1 m / 2 m tilt values
python -m results.plot_scripts.fig5_whole_cat    # Fig. 5
```

The first command takes about 30 s. It computes the whole-cat score of every drop, the
ANCOVA with HC3 errors, the Holm adjustment and the blocked permutation tests, and prints
Table II; add `--output <dir>` to also save the tables as CSV. The figure scripts write
`results/figures/fig4_tilt.svg` and `results/figures/fig5_whole_cat.svg`.

| | No tail | With tail | Adjusted tail effect | 95% HC3 CI | *p* |
|---|---|---|---|---|---|
| All releases | 0.688 ± 0.155 (37) | 0.798 ± 0.090 (35) | +0.157 | [0.062, 0.252] | 0.0016 |
| Roll sweep | 0.627 ± 0.140 (19) | 0.807 ± 0.086 (17) | +0.185 | [0.075, 0.296] | 0.0019 |
| Pitch sweep | 0.715 ± 0.164 (24) | 0.789 ± 0.092 (24) | +0.114 | [−0.057, 0.286] | 0.1841 |

## Verify the deployed policies

```bash
python -m tools.verify_onnx --replay
```

This checks that the ONNX files are bit-identical to the PyTorch students and agree with
them to float32 rounding. It then replays every hardware log through both policies. The
logged joint targets are reproduced to within a few 1e-15 by exactly one policy, which
identifies the controller that flew each drop.

`python -m hardware.e2e_test --variant tail` (or `notail`) checks, without hardware, that
the controller's observation, filtering and joint mapping match the simulator, and that the
ONNX student still rights the robot in closed loop.

## Train from scratch

```bash
python -m train.train --variant tail               # SAC teacher, 1M steps, 10 envs
python -m train.distillation --variant tail        # DAgger student, 50 x 2,000 steps
python -m hardware.onnx_conversion --variant tail  # policies/cat_controller.onnx
```

Run the same with `--variant notail`. Training and distillation write timestamped
checkpoints to `policies/`. Copy them to the canonical names (`cat_controller<suffix>.zip`,
`student_policy<suffix>.pth`) before the next stage. TensorBoard logs go to `run_logs/`.

The defaults are the configuration of the released teachers:
- **Reward.** `r = 1.5·(½(u_f + u_r) + u_f·u_r) − 0.002·mean(ω²)`, with `u_i = 1 − φ_i/π`, where
  φ_i is the angle between body *i*'s up vector and the world up vector.
  - The torque, joint-velocity and simultaneity penalties have zero weight.
  - Dividing by 3 gives the paper's normalisation, `¼(u_f + u_r) + ½u_f·u_r`, with an
    angular-velocity weight of about 6.7e-4.
- **Smoothness.** The actor-loss smoothness coefficient is 2.0.

Weights can be overridden without code changes through `CAT_W_POS`, `CAT_W_EN`,
`CAT_W_AV`, `CAT_W_JV` and `CAT_W_TIME`. Domain randomisation draws from the global NumPy
RNG in each worker, so training runs are not bit-reproducible.

To watch a policy in simulation:

```bash
mjpython -m tools.view_policy --variant tail --roll 180 --pitch 45 --slow 0.2    # python on Linux
```

## Released policies

`policies/` holds exactly what flew in the Aug 10, 16 and 17, 2026 drop experiments:

| File | What | sha256 (prefix) |
|---|---|---|
| `cat_controller.zip`, `cat_controller_notail.zip` | SAC teachers (2026-08-08, run `av002`) | `a9109065`, `bfcad505` |
| `student_policy.pth`, `student_policy_notail.pth` | DAgger students (MLP 11-256-256-3, tanh) | `ed3ec97b`, `befc4f4d` |
| `cat_controller.onnx`, `cat_controller_notail.onnx` | ONNX exports run on the Raspberry Pi | `c88c79ca`, `8f3e6990` |

With torch 2.10.0, onnx 1.21.0 and onnxscript 0.6.2, `hardware/onnx_conversion.py`
reproduces both ONNX files byte for byte. The no-tail student was distilled with the raw
tail action fed back as its previous action (`prev_action[2]`), whereas the robot feeds
back 0. This is documented in `train/distillation.py` and left unchanged so the code
matches the released policy.

## Hardware

Two Teensy 4.0 boards run 1 kHz PD loops for the four joints. The front board carries a
BNO085 IMU. A Raspberry Pi detects free fall (|a| < 3.5 m/s²), runs the ONNX student at
50 Hz for one second and logs every step. Wiring, flashing, the start-up procedure, the
observation layout and the telemetry format are in [hardware/README.md](hardware/README.md).

## Data notes

The drop logs are kept exactly as recorded. `results/hardware_trials/MANIFEST.csv` lists
every drop with its release attitude, the policy that flew it and its whole-cat score.
Three points about the data:
- **Excluded short drop.** One with-tail drop reached impact before the 2.5 m evaluation
  point. It is excluded from the whole-cat score, so the with-tail n is 35.
- **No-tail drops flown with the with-tail policy.** Replay shows that 19 of the 37
  no-tail drops were flown with the with-tail policy: all 18 pitched no-tail drops and
  one roll-180°, pitch-0° drop, which is 19 of the 24 no-tail drops in the pitch sweep.
  The controller defaulted to `--variant tail` at the time; the flag is now required.
- **Date labels.** Four filename date labels do not match the git history of the logs.

## Project page

The page in `docs/` is static and has no build step. To publish it, open
Settings → Pages, choose *Deploy from a branch*, and select `main` and the `/docs`
folder. To preview it locally, run `python -m http.server --directory docs` and open
http://localhost:8000.

## Citation

```bibtex
@inproceedings{meowmentum,
  title     = {{MeowMentum}: A Multi-Strategy Aerial Self-Righting Robot for Evaluating
               the Contribution of an Actuated Tail Across Fall Conditions},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```
