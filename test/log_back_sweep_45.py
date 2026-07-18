#!/usr/bin/env python3
"""Capture back_sweep_45 serial CSV and plot encoder angles.

Firmware waits for RUN / RUN M1 / RUN M2 (does not auto-sweep on boot).

Usage:
  # Flash first, then:
  python test/log_back_sweep_45.py
  python test/log_back_sweep_45.py --motor m1
  python test/log_back_sweep_45.py --motor m2

  # Plot an existing CSV:
  python test/log_back_sweep_45.py --plot-only path/to/back_sweep_45_....csv
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports

SN_BACK = "18452630"
BAUD = 115200
COLS = ("t_ms", "m1_rad", "m2_rad", "tgt1", "tgt2", "active_motor")
READY_TIMEOUT_S = 8.0

MOTOR_CMD = {
    "both": b"RUN BOTH\n",
    "m1": b"RUN M1\n",
    "m2": b"RUN M2\n",
}


def port_by_sn(sn: str) -> str:
    for p in serial.tools.list_ports.comports():
        if p.serial_number == sn:
            return p.device
    raise SystemExit(
        f"No Teensy SN={sn}. Run: python hardware/discover_teensy.py"
    )


def capture(path: str, out_csv: Path, motor: str, seconds: float = 30.0) -> Path:
    cmd = MOTOR_CMD[motor]
    print(f"Opening {path} @ {BAUD} (SN_BACK={SN_BACK})")
    print(f"Will send: {cmd.decode().strip()!r}")
    # USB open may reset Teensy — keep the boot banner; do not flush it away.
    ser = serial.Serial(path, BAUD, timeout=0.1)
    time.sleep(0.2)

    rows: list[str] = []
    events: list[str] = []
    buf = ""
    saw_header = False
    ready = False
    finished = False
    run_sent = False

    def handle_line(line: str) -> None:
        nonlocal saw_header, ready, finished, run_sent
        if not line:
            return
        if line.startswith("#"):
            print(line)
            events.append(line)
            if line.startswith("# ready"):
                ready = True
            if line.startswith("# done"):
                finished = True
            return
        if line.startswith("t_ms,"):
            saw_header = True
            return
        if not saw_header or not run_sent:
            return
        parts = line.split(",")
        if len(parts) == 6:
            rows.append(line)

    try:
        ready_deadline = time.time() + READY_TIMEOUT_S
        while time.time() < ready_deadline and not ready:
            chunk = ser.read(512)
            if chunk:
                buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                handle_line(line.rstrip("\r"))

        if not ready:
            print("No '# ready' yet — sending command anyway…")
        else:
            print(f"Sending {cmd.decode().strip()}…")

        ser.write(cmd)
        ser.flush()
        run_sent = True
        finished = False

        end = time.time() + seconds
        print("Capturing until '# done' or timeout…")
        while time.time() < end and not finished:
            chunk = ser.read(512)
            if chunk:
                buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                handle_line(line.rstrip("\r"))
    finally:
        ser.close()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(COLS) + "\n")
        for r in rows:
            f.write(r + "\n")
        for e in events:
            f.write(f"# {e[2:] if e.startswith('# ') else e[1:]}\n")

    print(f"Wrote {len(rows)} samples -> {out_csv}")
    if len(rows) == 0:
        raise SystemExit("No data rows captured. Reflash back_sweep_45 and retry.")
    return out_csv


def plot_csv(csv_path: Path, out_png: Path | None, show: bool) -> None:
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    data_lines = [ln for ln in lines if ln and not ln.startswith("#")]
    from io import StringIO

    df = pd.read_csv(StringIO("\n".join(data_lines)))
    for c in COLS:
        if c not in df.columns:
            raise SystemExit(f"Missing column {c} in {csv_path}")

    t = df["t_ms"].to_numpy(dtype=float) / 1000.0
    m1 = np.degrees(df["m1_rad"].to_numpy(dtype=float))
    m2 = np.degrees(df["m2_rad"].to_numpy(dtype=float))
    tgt1 = np.degrees(df["tgt1"].to_numpy(dtype=float))
    tgt2 = np.degrees(df["tgt2"].to_numpy(dtype=float))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, m1, label="M1 encoder (tail)", color="#0072b2")
    axes[0].plot(t, tgt1, "--", label="M1 target", color="#0072b2", alpha=0.5)
    axes[0].axhline(45, color="gray", lw=0.8, alpha=0.4)
    axes[0].axhline(-45, color="gray", lw=0.8, alpha=0.4)
    axes[0].set_ylabel("deg")
    axes[0].set_title("Back sweep 45° — M1 tail")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, m2, label="M2 encoder (roll)", color="#d55e00")
    axes[1].plot(t, tgt2, "--", label="M2 target", color="#d55e00", alpha=0.5)
    axes[1].axhline(45, color="gray", lw=0.8, alpha=0.4)
    axes[1].axhline(-45, color="gray", lw=0.8, alpha=0.4)
    axes[1].set_ylabel("deg")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Back sweep 45° — M2 roll")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if out_png is None:
        out_png = csv_path.with_suffix(".png")
    fig.savefig(out_png, dpi=150)
    print(f"Plot -> {out_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log + plot back_sweep_45")
    parser.add_argument(
        "--motor",
        choices=("both", "m1", "m2"),
        default="both",
        help="Which motor(s) to sweep (default: both)",
    )
    parser.add_argument(
        "--plot-only",
        type=Path,
        help="Skip serial capture; plot this CSV",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: test/logs/back_sweep_45_<motor>_<timestamp>.csv)",
    )
    parser.add_argument("--no-show", action="store_true", help="Save plot only")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.plot_only:
        plot_csv(args.plot_only, None, show=not args.no_show)
        return

    out = args.output
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("test/logs") / f"back_sweep_45_{args.motor}_{stamp}.csv"

    path = port_by_sn(SN_BACK)
    csv_path = capture(path, out, motor=args.motor, seconds=args.seconds)
    plot_csv(csv_path, None, show=not args.no_show)


if __name__ == "__main__":
    main()
