"""Tune the inner-loop PD gains in MuJoCo: fastest step response with no oscillation.

Each joint is stepped through the same control path as the env (PD, Teensy deadband
and minimum-PWM floor, ctrlrange torque map, 1 ms mj_step) on the free-floating robot
with contact disabled. A (kp, kd) grid is searched per joint group, and the candidates
are re-scored across the domain-randomization range, because the gains are flashed to
hardware. The winner has the fastest settling among the gains with no overshoot and no
limit cycle.

Usage, from the repository root:

    python -m tools.tune_pd_gains                  # tune all four joints
    python -m tools.tune_pd_gains --group pitch    # one group (roll = rot1 + rot2)
    python -m tools.tune_pd_gains --plot           # also write the step-response figure
    python -m tools.tune_pd_gains --plot --out pd.pdf

The result is printed as pd_nominal (train/cat_env/cat_env.py) and as the firmware
gains (x1024) for hardware/PD_control_{front,back}.
"""
import argparse
import os

import numpy as np
import mujoco

import train.cat_env.env_util as util
from train.cat_env.cat_env import CatEnv

# Joints in model order, the order of CatEnv.pd_nominal
JOINTS = ["rot1", "pitch", "rot2", "tail"]

# Joint groups that share one gain pair (the env drives rot2 as -rot1)
GROUPS = {"roll": ("rot1", "rot2"), "pitch": ("pitch",), "tail": ("tail",)}

# Step sizes per joint in rad: one inside the torque limit and one that saturates it
STEPS = {"rot1": (0.3, 1.5), "rot2": (0.3, 1.5), "pitch": (0.3, 1.0), "tail": (0.3, 1.0)}

# (kp, kd) search grids; roll has ~1/10 the torque range of pitch and tail (0.17 vs 1.1 Nm)
ROLL_GRID = ([1, 2, 3, 5, 8, 12, 20, 30, 50, 80],
             [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0])
PITCH_TAIL_GRID = ([5, 10, 20, 40, 70, 120, 200, 350],
                   [0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0])

# Step-test length and acceptance gates
SETTLE_S = 1.0          # simulated seconds per step test
SETTLE_BAND = 0.03      # rad, settling band; fixed, not tied to the controller deadband
OVERSHOOT_MAX = 2.0     # percent, at OVERSHOOT_PCTL of the DR draws
OVERSHOOT_PCTL = 95
CROSSINGS_MAX = 1       # velocity sign changes allowed after first entering the band


def step_response(env, joint, target, kp, kd, seconds=SETTLE_S, group=None):
    """Step one joint from rest to `target` while holding the other joints at 0.

    The candidate gains go to every joint in `group` (only to `joint` when `group` is
    None) and the other joints keep `env.pd_nominal`. The joints are coupled on the
    free-floating base, so a group that must share gains is tested with the partner
    at the candidate gains too.

    Returns:
        (t, q): time in s and the angle of `joint` in rad, one sample per physics step.
    """
    # Reset the robot to its initial pose at rest
    model, data = env.model, env.data
    mujoco.mj_resetData(model, data)
    data.qpos[:] = env.init_qpos
    data.qvel[:] = env.init_qvel
    mujoco.mj_forward(model, data)

    # Build one PD controller per joint with the candidate or nominal gains
    gains = dict(zip(JOINTS, env.pd_nominal))
    for name in (GROUPS[group] if group else (joint,)):
        gains[name] = (kp, kd)
    pds = {name: util.PDController(*gains[name], resolution=util.encoder_resolution(name),
                                   dt=model.opt.timestep) for name in JOINTS}

    # Step only `joint`; unlike in the env, rot2 does not counter-twist rot1
    targets = {name: (target if name == joint else 0.0) for name in JOINTS}

    # Simulate the PD loop through the Teensy deadband and the ctrlrange torque map
    n_steps = int(seconds / model.opt.timestep)
    q = np.zeros(n_steps)
    for k in range(n_steps):
        for name in JOINTS:
            pos = data.qpos[env._joint_qpos_idx[name]]
            torque = pds[name].get_torque(targets[name], pos)
            torque = util.apply_motor_deadband(torque, targets[name] - pds[name].meas_pos,
                                               util.joint_deadband(name))
            lo, hi = model.actuator_ctrlrange[env._actuator_idx[name]]
            data.ctrl[env._actuator_idx[name]] = util.map_value(torque, -1.0, 1.0, lo, hi)
        mujoco.mj_step(model, data)
        q[k] = data.qpos[env._joint_qpos_idx[joint]]
    return np.arange(n_steps) * model.opt.timestep, q


def metrics(t, q, target):
    """Measure the rise time, settling time, overshoot and oscillation of a step response.

    Returns:
        dict with `rise` and `settle` in s (inf when never reached), `overshoot` in
        percent of the step, and `crossings`, the number of velocity sign changes after
        first entering the settling band.
    """
    # Set the fixed settling band and express the response as a positive step
    amplitude = abs(target)
    band = max(0.02 * amplitude, SETTLE_BAND)
    x = q * np.sign(target)

    # Measure the rise time to 90 % of the step, inf if it is never reached
    reached = x >= 0.9 * amplitude
    rise = np.argmax(reached) * (t[1] - t[0]) if np.any(reached) else np.inf

    # Measure the overshoot past the step, in percent of the step
    overshoot = 100.0 * (x.max() - amplitude) / amplitude if x.max() > amplitude else 0.0

    # Find the settling time, the sample after the last one outside the band
    dt = t[1] - t[0]
    inside = np.abs(x - amplitude) <= band
    outside = np.flatnonzero(~inside)
    if not inside.any() or (len(outside) and outside[-1] + 1 >= len(x)):
        settle = np.inf  # never inside the band, or still outside at the end
    elif len(outside) == 0:
        settle = 0.0
    else:
        settle = (outside[-1] + 1) * dt

    # Count velocity sign changes after first entering the band, to catch limit cycles
    velocity = np.diff(x)
    first_inside = np.argmax(inside) if inside.any() else len(velocity)
    moving = velocity[first_inside:]
    moving = moving[np.abs(moving) > 1e-7]
    crossings = int(np.sum(np.diff(np.sign(moving)) != 0)) if len(moving) > 1 else 0

    return dict(rise=rise, settle=settle, overshoot=overshoot, crossings=crossings)


def score(env, group, kp, kd):
    """Score a gain pair by its worst case over the group's joints and step sizes."""
    # Measure every step test of the group
    results = []
    for joint in GROUPS[group]:
        for step in STEPS[joint]:
            t, q = step_response(env, joint, step, kp, kd, group=group)
            results.append(metrics(t, q, step))

    # Keep the worst value of each metric
    return {key: max(result[key] for result in results)
            for key in ("settle", "rise", "overshoot", "crossings")}


def search(env, group, kp_grid, kd_grid, n_dr=24):
    """Search the (kp, kd) grid of one group and print the best candidates.

    A nominal-physics prefilter drops pairs that cannot settle or ring badly. The rest
    are re-scored over `n_dr` domain-randomization draws and gated on overshoot,
    oscillation and failures to settle there, because these gains are flashed to
    hardware.

    Returns:
        The passing candidates, best first, as dicts of the gains and their metrics.
    """
    # Score every grid point at nominal physics
    rows = []
    for kp in kp_grid:
        for kd in kd_grid:
            row = score(env, group, kp, kd)
            row.update(kp=kp, kd=kd)
            rows.append(row)

    # Keep the pairs that settle and do not ring badly at nominal
    candidates = [row for row in rows if np.isfinite(row["settle"])
                  and row["overshoot"] <= 10.0 and row["crossings"] <= 3]

    # Re-score the candidates across domain randomization
    scored = []
    for row in candidates:
        randomized = robustness(env, group, row["kp"], row["kd"], n=n_dr)
        scored.append(dict(kp=row["kp"], kd=row["kd"], nom_settle=row["settle"],
                           settle=randomized["settle"], overshoot=randomized["overshoot"],
                           overshoot_max=randomized["overshoot_max"],
                           unsettled=randomized["unsettled"],
                           crossings=randomized["crossings"], rise=row["rise"]))

    # Gate on the randomized metrics and rank by settling time, then nominal rise time
    ranked = [row for row in scored
              if row["overshoot"] <= OVERSHOOT_MAX and row["crossings"] <= CROSSINGS_MAX
              and row["unsettled"] == 0]
    ranked.sort(key=lambda row: (row["settle"], row["rise"]))

    # Break ties (settling within 5 % of the best) by oscillation, overshoot, then kd
    if ranked:
        tie_limit = ranked[0]["settle"] * 1.05
        ranked.sort(key=lambda row: (row["settle"] > tie_limit, row["crossings"],
                                     row["overshoot"], row["kd"], row["settle"]))

    # Report the ranking
    print_ranking(group, len(candidates), len(rows), ranked)
    return ranked


def print_ranking(group, n_candidates, n_rows, ranked):
    """Print how many pairs passed each stage and the six best candidates."""
    print(f"\n{group}: {n_candidates}/{n_rows} settle at nominal; "
          f"{len(ranked)} of those hold up across domain randomization")
    print(f"  {'kp':>8}{'kd':>8}{'settle90':>10}{'nom':>8}{'ovr95':>7}{'ovrmax':>8}{'osc':>5}")
    for row in ranked[:6]:
        print(f"  {row['kp']:>8.2f}{row['kd']:>8.3f}{row['settle']:>10.3f}"
              f"{row['nom_settle']:>8.3f}{row['overshoot']:>7.2f}"
              f"{row['overshoot_max']:>8.2f}{row['crossings']:>5d}")
    print("  (settle90/ovr95 = 90th/95th percentile over the DR draws; osc = max)")


def randomize_physics(env, group, rng):
    """Draw the group's motor parameters and all body masses and inertias around nominal."""
    # Scale the damping, armature, friction and torque range of each joint in the group
    for joint in GROUPS[group]:
        dof, act = env._joint_qvel_idx[joint], env._actuator_idx[joint]
        env.model.dof_damping[dof] = env.nominal_damping[dof] * rng.uniform(0.7, 1.3)
        env.model.dof_armature[dof] = env.nominal_armature[dof] * rng.uniform(0.4, 2.5)
        env.model.dof_frictionloss[dof] = env.nominal_frictionloss[dof] * rng.uniform(0.7, 1.3)
        env.model.actuator_ctrlrange[act] = env.nominal_ctrlrange[act] * rng.uniform(0.8, 1.2)

    # Scale every body's mass and inertia
    env.model.body_mass[:] = env.nominal_mass * rng.uniform(0.8, 1.2, env.nominal_mass.shape)
    env.model.body_inertia[:] = env.nominal_inertia * rng.uniform(
        0.8, 1.2, env.nominal_inertia.shape)


def restore_nominal_physics(env):
    """Reset the parameters changed by randomize_physics() to their nominal values."""
    env.model.dof_damping[:] = env.nominal_damping
    env.model.dof_armature[:] = env.nominal_armature
    env.model.dof_frictionloss[:] = env.nominal_frictionloss
    env.model.actuator_ctrlrange[:] = env.nominal_ctrlrange
    env.model.body_mass[:] = env.nominal_mass
    env.model.body_inertia[:] = env.nominal_inertia


def robustness(env, group, kp, kd, n=24, seed=0):
    """Score a gain pair over `n` domain-randomization draws.

    Returns:
        dict with the 90th-percentile `settle` in s, the OVERSHOOT_PCTL-percentile and
        maximum overshoot in percent, the maximum `crossings`, the number of step tests
        that never settle (`unsettled`) and the number of step tests (`n`).
    """
    # Run every step test of the group under each randomized draw
    rng = np.random.RandomState(seed)
    results = []
    for _ in range(n):
        randomize_physics(env, group, rng)
        for joint in GROUPS[group]:
            for step in STEPS[joint]:
                t, q = step_response(env, joint, step, kp, kd, group=group)
                results.append(metrics(t, q, step))
    restore_nominal_physics(env)

    # Summarize the distribution of the results
    settle = np.array([result["settle"] for result in results])
    settled = settle[np.isfinite(settle)]
    overshoots = [result["overshoot"] for result in results]
    return dict(
        settle=float(np.percentile(settled, 90)) if len(settled) else np.inf,
        overshoot=float(np.percentile(overshoots, OVERSHOOT_PCTL)),
        overshoot_max=float(max(overshoots)),
        crossings=int(max(result["crossings"] for result in results)),
        unsettled=int(np.sum(~np.isfinite(settle))),
        n=len(results),
    )


def figure_path(path):
    """Return `path` unchanged if it ends in .pdf, else with its extension replaced by .svg."""
    root, ext = os.path.splitext(path)
    return path if ext.lower() == ".pdf" else root + ".svg"


def plot(env, chosen, current, out):
    """Plot the large-step response of each tuned joint with its current and tuned gains.

    `chosen` and `current` map joint names to (kp, kd). The figure goes to
    figure_path(`out`).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Create one panel per tuned joint
    colors = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8880",
                  grid="#e4e3dd", new="#2a78d6", old="#eb6834")
    path = figure_path(out)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joints = [joint for joint in JOINTS if joint in chosen]
    fig, axes = plt.subplots(1, len(joints), figsize=(3.6 * len(joints), 3.6), squeeze=False)
    fig.patch.set_facecolor(colors["surface"])

    # Draw the current and tuned responses to each joint's large step
    for ax, joint in zip(axes[0], joints):
        # Simulate the step with the current and the tuned gains
        kp, kd = chosen[joint]
        step = STEPS[joint][1]
        t, q_old = step_response(env, joint, step, *current[joint])
        t, q_new = step_response(env, joint, step, kp, kd)

        # Plot both responses against the dashed target
        ax.set_facecolor(colors["surface"])
        ax.grid(axis="y", color=colors["grid"], lw=0.7)
        ax.set_axisbelow(True)
        ax.axhline(step, color=colors["ink3"], lw=1, ls=(0, (4, 3)), zorder=2)
        ax.plot(t, q_old, color=colors["old"], lw=1.6,
                label=f"current {current[joint][0]:g}/{current[joint][1]:g}")
        ax.plot(t, q_new, color=colors["new"], lw=1.9, label=f"tuned {kp:g}/{kd:g}")

        # Style the panel
        ax.set_title(joint, color=colors["ink"], fontsize=11, loc="left")
        ax.set_xlabel("time (s)", color=colors["ink2"], fontsize=9)
        ax.tick_params(colors=colors["ink2"], labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(colors["grid"])
        legend = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        for text in legend.get_texts():
            text.set_color(colors["ink2"])

    # Label the figure
    axes[0][0].set_ylabel("joint angle (rad)", color=colors["ink2"], fontsize=9)
    fig.suptitle("PD step response — dashed = commanded target",
                 color=colors["ink"], fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # Save with the text kept editable in the SVG
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig.savefig(path, facecolor=colors["surface"])
    plt.close(fig)
    print(f"\nwrote {path}")


def main():
    """Tune the selected joint groups, print the gains and optionally plot the steps."""
    # Parse the command line
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=list(GROUPS), default=None,
                        help="tune just one group (roll = rot1+rot2 together)")
    parser.add_argument("--plot", action="store_true", help="write the step-response figure")
    parser.add_argument("--out", default="pd_step_response.svg",
                        help="figure path for --plot. Saved as vector SVG with editable "
                             "text; an explicit .pdf is kept, any other extension is "
                             "replaced by .svg (default: pd_step_response.svg)")
    parser.add_argument("--robust", type=int, default=24,
                        help="domain-randomization samples for the robustness check")
    args = parser.parse_args()

    # Load the robot and the gains it currently uses
    env = CatEnv(model_path="model/cat.xml")
    groups = [args.group] if args.group else list(GROUPS)
    current = dict(zip(JOINTS, env.pd_nominal))

    # Tune each group and compare the winner with the current gains
    chosen = {}
    for group in groups:
        # Search the gain grid for this group
        joint = GROUPS[group][0]
        kp_grid, kd_grid = ROLL_GRID if group == "roll" else PITCH_TAIL_GRID
        ranked = search(env, group, kp_grid, kd_grid, n_dr=args.robust)
        if not ranked:
            print(f"  {group}: no gain pair passed the gates")
            continue

        # Score the current gains and the winner under domain randomization
        best = ranked[0]
        base = score(env, group, *current[joint])
        randomized = robustness(env, group, best["kp"], best["kd"], n=args.robust)

        # Print the comparison and keep the winner for every joint in the group
        print(f"  current (kp={current[joint][0]}, kd={current[joint][1]}): "
              f"settle {base['settle']:.3f}s  over {base['overshoot']:.1f}%  "
              f"osc {base['crossings']}")
        print(f"  chosen  (kp={best['kp']}, kd={best['kd']}): "
              f"settle p90 {randomized['settle']:.3f}s  "
              f"overshoot p95 {randomized['overshoot']:.1f}% "
              f"(max {randomized['overshoot_max']:.1f}%)  osc {randomized['crossings']}  "
              f"unsettled {randomized['unsettled']}/{randomized['n']} DR draws")
        for name in GROUPS[group]:
            chosen[name] = (best["kp"], best["kd"])

    # Print the gains for the env and for the firmware, which scales them by 1024
    final = [chosen.get(name, current[name]) for name in JOINTS]
    print("\npd_nominal = [" + ", ".join(f"({kp:g}, {kd:g})" for kp, kd in final) + "]")
    print("firmware (sim x 1024): " + ", ".join(
        f"{name} Kp={chosen[name][0]*1024:.1f} Kd={chosen[name][1]*1024:.1f}"
        for name in JOINTS if name in chosen))

    # Plot the step responses of the tuned joints
    if args.plot and chosen:
        plot(env, chosen, current, args.out)


if __name__ == "__main__":
    main()
