"""Check the ONNX students against their PyTorch checkpoints and find which flew each drop.

    python -m tools.verify_onnx              # weights + 100k inputs, both variants
    python -m tools.verify_onnx --replay     # also replay every drop in results/hardware_trials/

1. Every ONNX initializer is bit-identical to its state_dict tensor (skipped without `onnx`).
2. torch and onnxruntime agree within 1e-4 on 100k realistic random observations.
3. With --replay, both ONNX files replay each drop log as hardware/controller.py runs them;
   the one that reproduces the logged actions flew the drop, and folder mismatches are flagged.

Prints PASS and exits with status 0 if every check passes, otherwise FAIL and 1.
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

# Drop logs replayed by --replay
TRIALS = "results/hardware_trials"

# Policy-file suffix of each variant, as in model/variants.py
VARIANTS = {"tail": "", "notail": "_notail"}

# Action pipeline of hardware/controller.py: FILTER_ALPHA and the roll, pitch, tail ranges (rad)
ALPHA = np.array([0.1, 0.3, 0.1])
RANGE = np.array([7.28, 1.57, 1.9199])

# Pass tolerances: torch vs onnxruntime output, and replayed vs logged raw action
TOL = 1e-4
REPLAY_TOL = 1e-5


def torch_student(state_dict):
    """Rebuild train.distillation.StudentPolicy without importing the training stack."""
    net = nn.Sequential(nn.Linear(11, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 3), nn.Tanh())
    net.load_state_dict({k.removeprefix("net."): v for k, v in state_dict.items()})
    return net.eval()


def random_obs(n, rng):
    """Draw n realistic student observations as a float32 array of shape (n, 11)."""
    # Unit gravity direction
    gravity = rng.normal(size=(n, 3))
    gravity /= np.linalg.norm(gravity, axis=1, keepdims=True)

    # Joint angles within the joint ranges (rad), previous action, and step / 50
    joints = rng.uniform(-1, 1, (n, 4)) * np.array([7.28, 1.57, 7.28, 1.92])
    prev_action = rng.uniform(-1, 1, (n, 3))
    step = rng.integers(0, 51, (n, 1)) / 50
    return np.hstack([gravity, joints, prev_action, step]).astype(np.float32)


def initializers_match(variant, onnx_file, state_dict):
    """Print and return whether every ONNX initializer equals its state_dict tensor bit for bit.

    Returns True without checking when the onnx package is not installed.
    """
    try:
        import onnx
        from onnx import numpy_helper
        init = {t.name: numpy_helper.to_array(t) for t in onnx.load(onnx_file).graph.initializer}
        same = set(init) == set(state_dict) and all(
            np.array_equal(init[k], t.numpy()) for k, t in state_dict.items())
        print(f"[{variant}] {len(init)} initializers bit-identical to the state_dict: {same}")
        return same
    except ImportError:
        print(f"[{variant}] onnx not installed; skipping the initializer check")
        return True


def check_variant(variant, suffix, obs):
    """Check one variant's ONNX file against its checkpoint on the observations `obs`.

    Returns:
        Whether both checks passed, and the onnxruntime session of the ONNX file.
    """
    # Compare the weights
    onnx_file = f"policies/cat_controller{suffix}.onnx"
    state_dict = torch.load(f"policies/student_policy{suffix}.pth", map_location="cpu",
                            weights_only=True)
    ok = initializers_match(variant, onnx_file, state_dict)

    # Compare torch and onnxruntime outputs
    with torch.no_grad():
        torch_out = torch_student(state_dict)(torch.from_numpy(obs)).numpy()
    session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {session.get_inputs()[0].name: obs})[0]
    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"[{variant}] torch vs onnxruntime on {len(obs)} inputs: max |diff| = {max_diff:.2e}")
    return ok and max_diff < TOL, session


def proj_grav(q):
    """Return world -z in the body frame for quaternions q = [w, x, y, z] of shape (n, 4).

    Same as train.cat_env.env_util.to_projected_gravity.
    """
    w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)


def drop_logs():
    """Return the sorted drop-log CSVs, leaving out MANIFEST.csv at the top of TRIALS."""
    return sorted(path for folder in ("with_tail", "no_tail")
                  for path in glob.glob(os.path.join(TRIALS, folder, "**", "*.csv"),
                                        recursive=True))


def raw_actions(log):
    """Recover the raw policy output of every logged step from the logged joint targets."""
    # Undo the range map back to the filtered action in [-1, 1]
    filtered = np.stack([log["Cmd_F1"], log["Cmd_F2"], log["Cmd_B1"]], 1) / RANGE

    # Undo the low-pass filtered = ALPHA * raw + (1 - ALPHA) * previous filtered, from zero
    previous_filtered = np.vstack([np.zeros(3), filtered[:-1]])
    return (filtered - (1 - ALPHA) * previous_filtered) / ALPHA


def replay_errors(log, sessions):
    """Return, per variant, the max |raw action| error of its ONNX session over a drop log."""
    # Rebuild the inputs shared by both variants: orientation, previous raw action and step
    quaternions = np.stack([log["F_Q0"], log["F_Q1"], log["F_Q2"], log["F_Q3"]], 1)
    raw = raw_actions(log)
    prev_action = np.vstack([np.zeros(3), raw[:-1]])
    step = np.minimum(np.arange(len(log)) / 50, 1.0)[:, None]

    # Score each variant's ONNX session against the recovered raw actions
    errors = {}
    for variant, session in sessions.items():
        # Read the joint angles, with the tail zeroed for the no-tail policy (controller.zero_tail)
        joints = np.stack([log["F_M1"], log["F_M2"], log["B_M2"], log["B_M1"]], 1)
        if variant == "notail":
            joints[:, 3] = 0.0

        # Run the policy; the controller holds the no-tail policy's tail channel at zero
        obs = np.hstack([proj_grav(quaternions), joints, prev_action, step]).astype(np.float32)
        action = session.run(None, {session.get_inputs()[0].name: obs})[0]
        action = np.clip(action.astype(float), -1, 1)
        if variant == "notail":
            action[:, 2] = 0.0
        errors[variant] = np.abs(action - raw).max()
    return errors


def replay(sessions):
    """Print which policy flew each logged drop; return False if any drop is unexplained."""
    # Attribute each drop to the policy that reproduces it and compare with its folder
    files = drop_logs()
    counts, flagged, unexplained = {}, [], []
    for path in files:
        # Pick the policy that best reproduces the log and count it under the drop's folder
        errors = replay_errors(np.genfromtxt(path, delimiter=",", names=True), sessions)
        flown_by = min(errors, key=errors.get)
        folder_variant = "notail" if f"{os.sep}no_tail{os.sep}" in path else "tail"
        counts[(folder_variant, flown_by)] = counts.get((folder_variant, flown_by), 0) + 1

        # Note a drop that neither policy reproduces, or that the other morphology's policy flew
        rel = os.path.relpath(path, TRIALS)
        note = ""
        if errors[flown_by] > REPLAY_TOL:
            unexplained.append(rel)
            note = "   <-- reproduced by neither policy"
        elif flown_by != folder_variant:
            flagged.append(rel)
            note = f"   <-- folder says {folder_variant}"

        # Print the attribution with the other policy's error for comparison
        other = max(e for v, e in errors.items() if v != flown_by)
        print(f"  {rel:86s} {flown_by:6s} policy  max err {errors[flown_by]:.1e} "
              f"(other {other:.1e}){note}")

    # Summarize the attributions per folder
    print(f"\nreplayed {len(files)} drops under results/hardware_trials "
          f"(error = max |raw action| difference)")
    for folder_variant in VARIANTS:
        folder_counts = [(policy, n) for (drop_variant, policy), n in sorted(counts.items())
                         if drop_variant == folder_variant]
        total = sum(n for _, n in folder_counts)
        by_policy = ", ".join(f"{n} by the {policy} policy" for policy, n in folder_counts)
        folder = "with_tail" if folder_variant == "tail" else "no_tail"
        print(f"  {folder:9s} {total:3d} drops: {by_policy}")

    # List the drops flown with the other morphology's policy
    if flagged:
        print(f"{len(flagged)} drops were flown with the other morphology's policy:")
        for rel in flagged:
            print(f"  {rel}")
    return not unexplained


def main():
    """Run the checks for both variants, print PASS or FAIL and exit with its status."""
    # Parse the command line
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", action="store_true",
                        help="also replay every drop in results/hardware_trials/ through both "
                             "ONNX files")
    args = parser.parse_args()

    # Check both variants on the same random observations
    obs = random_obs(100_000, np.random.default_rng(0))
    ok, sessions = True, {}
    for variant, suffix in VARIANTS.items():
        passed, sessions[variant] = check_variant(variant, suffix, obs)
        ok &= passed

    # Replay the hardware drops
    if args.replay:
        ok &= replay(sessions)

    # Report the result
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
