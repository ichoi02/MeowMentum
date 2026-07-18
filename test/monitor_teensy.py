#!/usr/bin/env python3
"""Monitor motor_test_* Teensy serial output.

Works with DEBUG_ENCODER_ONLY firmware, which streams:
  m1_ticks,m2_ticks,m1_rad,m2_rad

Usage:
  python test/monitor_teensy.py --front
  python test/monitor_teensy.py --back
  python test/monitor_teensy.py --port /dev/ttyACM0

Ctrl-C to quit.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import serial
import serial.tools.list_ports

# Keep in sync with hardware/controllerV2.py
SN_FRONT = "18451300"
SN_BACK = "18452630"
BAUD = 115200

# ticks1,ticks2,rad1,rad2[,m1A,m1B,m2A,m2B]
CSV_RE = re.compile(
    r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*,\s*([01])\s*,\s*([01])\s*,\s*([01])\s*,\s*([01]))?\s*$"
)


def port_by_sn(serial_number: str) -> str:
    for port in serial.tools.list_ports.comports():
        if port.serial_number == serial_number:
            return port.device
    raise SystemExit(
        f"No Teensy with serial {serial_number!r}.\n"
        "  Check USB, then run: python hardware/discover_teensy.py\n"
        "  (Encoder-debug firmware must already be flashed.)"
    )


def format_encoder_line(line: str) -> str | None:
    m = CSV_RE.match(line)
    if not m:
        return None
    t1, t2, r1, r2, a1, b1, a2, b2 = m.groups()
    out = (
        f"m1_ticks={t1:>8}  m2_ticks={t2:>8}  "
        f"m1_rad={float(r1):+8.4f}  m2_rad={float(r2):+8.4f}"
    )
    if a1 is not None:
        out += f"  pins m1={a1}{b1} m2={a2}{b2}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor motor_test Teensy encoder / sweep serial output"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--front", action="store_true", help=f"Front Teensy SN={SN_FRONT}")
    group.add_argument("--back", action="store_true", help=f"Back Teensy SN={SN_BACK}")
    group.add_argument("--port", type=str, help="Explicit device, e.g. /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw serial lines (default: pretty-print encoder CSV)",
    )
    args = parser.parse_args()

    if args.port:
        path = args.port
        label = path
    elif args.front:
        path = port_by_sn(SN_FRONT)
        label = f"Front SN={SN_FRONT} ({path})"
    else:
        path = port_by_sn(SN_BACK)
        label = f"Back SN={SN_BACK} ({path})"

    print(f"Opening {label} @ {args.baud}", flush=True)
    print("Expecting DEBUG encoder stream: m1_ticks,m2_ticks,m1_rad,m2_rad", flush=True)
    print("Turn shafts by hand. Ctrl-C to quit.\n", flush=True)

    # Opening the port usually resets the Teensy; give it a moment to reboot.
    ser = serial.Serial(path, args.baud, timeout=0.1)
    time.sleep(0.3)
    ser.reset_input_buffer()

    buf = ""
    try:
        while True:
            chunk = ser.read(256)
            if not chunk:
                continue
            try:
                text = chunk.decode("utf-8", errors="replace")
            except Exception:
                continue
            buf += text
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.rstrip("\r")
                if not line:
                    continue
                if args.raw:
                    print(line, flush=True)
                    continue
                pretty = format_encoder_line(line)
                if pretty is not None:
                    print(pretty, flush=True)
                else:
                    # Banners / sweep status / unexpected lines
                    print(line, flush=True)
    except KeyboardInterrupt:
        print("\nClosed.", flush=True)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
