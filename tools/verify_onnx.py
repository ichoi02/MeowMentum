"""Check that the deployed ONNX students equal their PyTorch checkpoints, and which
policy flew each logged hardware drop.

    python tools/verify_onnx.py              # weights + 100k inputs, both variants
    python tools/verify_onnx.py --replay     # also replay every drop in data/hardware_trials/

1. Every ONNX initializer must be bit-identical to the matching state_dict tensor
   (this step needs the `onnx` package and is skipped if it is not installed).
2. On 100k realistic random observations, torch and onnxruntime must agree to
   float32 accumulation error (tolerance 1e-4).
3. With --replay, each drop is rebuilt into the 11-dim student observation exactly
   as hardware/controller.py builds it, from the logged sensor columns. The logged
   joint targets are passed back through the inverse of the action low-pass and the
   range map to recover the raw policy output, and each ONNX file is run as the
   controller runs it with --variant <that file's variant>. The file that reproduces
   the log is the policy that flew; drops whose folder (with_tail / no_tail) names
   the other morphology are flagged.
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIALS = os.path.join(ROOT, "data", "hardware_trials")
VARIANTS = {"tail": "", "notail": "_notail"}       # variants.py suffixes
ALPHA = np.array([0.1, 0.3, 0.1])                   # controller.FILTER_ALPHA
RANGE = np.array([7.28, 1.57, 1.9199])              # roll, pitch, tail joint ranges (rad)
TOL = 1e-4                                          # torch vs onnxruntime
REPLAY_TOL = 1e-5                                   # raw action, replay vs log


def student(sd):
    """distillation.StudentPolicy, without importing the training stack."""
    net = nn.Sequential(nn.Linear(11, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 3), nn.Tanh())
    net.load_state_dict({k.removeprefix("net."): v for k, v in sd.items()})
    return net.eval()


def random_obs(n, rng):
    g = rng.normal(size=(n, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    j = rng.uniform(-1, 1, (n, 4)) * np.array([7.28, 1.57, 7.28, 1.92])
    a = rng.uniform(-1, 1, (n, 3))
    s = rng.integers(0, 51, (n, 1)) / 50
    return np.hstack([g, j, a, s]).astype(np.float32)


def check_variant(v, sfx, X):
    pth = os.path.join(ROOT, "policies", f"student_policy{sfx}.pth")
    onx = os.path.join(ROOT, "policies", f"cat_controller{sfx}.onnx")
    sd = torch.load(pth, map_location="cpu", weights_only=True)
    ok = True
    try:
        import onnx
        from onnx import numpy_helper
        init = {t.name: numpy_helper.to_array(t) for t in onnx.load(onx).graph.initializer}
        same = set(init) == set(sd) and all(np.array_equal(init[k], t.numpy()) for k, t in sd.items())
        print(f"[{v}] {len(init)} initializers bit-identical to the state_dict: {same}")
        ok &= same
    except ImportError:
        print(f"[{v}] onnx not installed; skipping the initializer check")
    with torch.no_grad():
        yt = student(sd)(torch.from_numpy(X)).numpy()
    sess = ort.InferenceSession(onx, providers=["CPUExecutionProvider"])
    yo = sess.run(None, {sess.get_inputs()[0].name: X})[0]
    d = np.abs(yt - yo).max()
    print(f"[{v}] torch vs onnxruntime on {len(X)} inputs: max |diff| = {d:.2e}")
    return ok and d < TOL, sess


def proj_grav(q):
    """World -z in the body frame (cat_env.env_util.to_projected_gravity), q = [w, x, y, z]."""
    w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)


def replay(sessions):
    # Only the drop logs: data/hardware_trials/ also holds MANIFEST.csv.
    files = sorted(f for m in ("with_tail", "no_tail")
                   for f in glob.glob(os.path.join(TRIALS, m, "**", "*.csv"), recursive=True))
    counts, flagged, unexplained = {}, [], []
    for f in files:
        d = np.genfromtxt(f, delimiter=",", names=True)
        q = np.stack([d["F_Q0"], d["F_Q1"], d["F_Q2"], d["F_Q3"]], 1)
        filt = np.stack([d["Cmd_F1"], d["Cmd_F2"], d["Cmd_B1"]], 1) / RANGE
        raw = (filt - (1 - ALPHA) * np.vstack([np.zeros(3), filt[:-1]])) / ALPHA
        prev = np.vstack([np.zeros(3), raw[:-1]])
        step = np.minimum(np.arange(len(d)) / 50, 1.0)[:, None]
        errs = {}
        for v, sess in sessions.items():
            j = np.stack([d["F_M1"], d["F_M2"], d["B_M2"], d["B_M1"]], 1)
            if v == "notail":
                j[:, 3] = 0.0                      # controller.zero_tail
            obs = np.hstack([proj_grav(q), j, prev, step]).astype(np.float32)
            y = np.clip(sess.run(None, {sess.get_inputs()[0].name: obs})[0].astype(float), -1, 1)
            if v == "notail":
                y[:, 2] = 0.0                      # controller holds the tail at zero
            errs[v] = np.abs(y - raw).max()
        flew = min(errs, key=errs.get)
        label = "notail" if f"{os.sep}no_tail{os.sep}" in f else "tail"
        counts[(label, flew)] = counts.get((label, flew), 0) + 1
        rel = os.path.relpath(f, TRIALS)
        note = ""
        if errs[flew] > REPLAY_TOL:
            unexplained.append(rel)
            note = "   <-- reproduced by neither policy"
        elif flew != label:
            flagged.append(rel)
            note = f"   <-- folder says {label}"
        other = max(e for v, e in errs.items() if v != flew)
        print(f"  {rel:86s} {flew:6s} policy  max err {errs[flew]:.1e} (other {other:.1e}){note}")

    print(f"\nreplayed {len(files)} drops under data/hardware_trials "
          f"(error = max |raw action| difference)")
    for label in VARIANTS:
        n = sum(c for (lab, _), c in counts.items() if lab == label)
        by = ", ".join(f"{c} by the {fl} policy" for (lab, fl), c in sorted(counts.items())
                       if lab == label)
        print(f"  {'with_tail' if label == 'tail' else 'no_tail':9s} {n:3d} drops: {by}")
    if flagged:
        print(f"{len(flagged)} drops were flown with the other morphology's policy:")
        for r in flagged:
            print(f"  {r}")
    return not unexplained


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="store_true",
                    help="also replay every drop in data/hardware_trials/ through both ONNX files")
    args = ap.parse_args()
    X = random_obs(100_000, np.random.default_rng(0))
    ok, sessions = True, {}
    for v, sfx in VARIANTS.items():
        good, sessions[v] = check_variant(v, sfx, X)
        ok &= good
    if args.replay:
        ok &= replay(sessions)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
