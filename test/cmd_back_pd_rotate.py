#!/usr/bin/env python3
"""Send a relative PD move to back_pd_rotate firmware.

Requires: bash test/flash_back_pd_rotate.sh (SN_BACK).

Usage:
  python test/cmd_back_pd_rotate.py m2 0.2
  python test/cmd_back_pd_rotate.py m1 -0.1
  python test/cmd_back_pd_rotate.py zero
  python test/cmd_back_pd_rotate.py stop
  python test/cmd_back_pd_rotate.py m2 0.2 --listen 3
"""

from __future__ import annotations

import argparse
import time

import serial
import serial.tools.list_ports

SN_BACK = "18452630"
BAUD = 115200
BOOT_WAIT_S = 0.6


def port_by_sn(sn: str) -> str:
    for p in serial.tools.list_ports.comports():
        if p.serial_number == sn:
            return p.device
    raise SystemExit(
        f"No Teensy SN={sn}. Run: python hardware/discover_teensy.py"
    )


def build_command(args: argparse.Namespace) -> bytes:
    if args.cmd == "zero":
        return b"ZERO\n"
    if args.cmd == "stop":
        return b"STOP\n"
    # m1 / m2
    motor = args.cmd.upper()
    return f"{motor} {args.delta}\n".encode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Command back_pd_rotate (relative rad, 200 ms cap)"
    )
    parser.add_argument(
        "cmd",
        choices=("m1", "m2", "zero", "stop"),
        help="m1/m2 = relative move; zero/stop = helpers",
    )
    parser.add_argument(
        "delta",
        nargs="?",
        type=float,
        default=None,
        help="Relative radians (required for m1/m2)",
    )
    parser.add_argument(
        "--listen",
        type=float,
        default=2.0,
        help="Seconds to print serial after send (default: 2)",
    )
    parser.add_argument(
        "--no-reset-wait",
        action="store_true",
        help="Skip wait after open (board may still be booting)",
    )
    args = parser.parse_args()

    if args.cmd in ("m1", "m2") and args.delta is None:
        raise SystemExit(f"Usage: python test/cmd_back_pd_rotate.py {args.cmd} <delta_rad>")

    path = port_by_sn(SN_BACK)
    line = build_command(args)
    print(f"Opening {path} @ {BAUD} (SN_BACK={SN_BACK})")
    print(f"Sending: {line.decode().rstrip()!r}")

    ser = serial.Serial(path, BAUD, timeout=0.1)
    try:
        if not args.no_reset_wait:
            time.sleep(BOOT_WAIT_S)
        ser.reset_input_buffer()
        ser.write(line)
        ser.flush()

        if args.listen <= 0:
            return

        end = time.time() + args.listen
        buf = ""
        while time.time() < end:
            chunk = ser.read(256)
            if not chunk:
                continue
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                row, buf = buf.split("\n", 1)
                row = row.rstrip("\r")
                if row:
                    print(row)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
