#!/usr/bin/env python3
"""Read Front/Back Teensy telemetry (same format as controllerV2) for a few seconds.

Telemetry CSV per line:
  qr,qi,qj,qk,m1_rad,m2_rad,acc_mag

Usage:
  python hardware/parseData.py
  python hardware/parseData.py --seconds 3
  python hardware/parseData.py --front-only
  python hardware/parseData.py --back-only
  python hardware/parseData.py --raw
"""

from __future__ import annotations

import argparse
import time

import serial
import serial.tools.list_ports

# Keep in sync with hardware/controllerV2.py
SN_FRONT = "18451300"
SN_BACK = "18452630"
BAUD_RATE = 115200
SERIAL_DRAIN_MAX_LINES = 32


def port_by_sn(sn: str) -> str:
    for p in serial.tools.list_ports.comports():
        if p.serial_number == sn:
            return p.device
    raise SystemExit(
        f"No Teensy SN={sn}. Run: python hardware/discover_teensy.py"
    )


def open_port(path: str) -> serial.Serial:
    return serial.Serial(path, BAUD_RATE, timeout=0.005, write_timeout=0)


def parse_line(line: str) -> tuple[list[float], float, float, float] | None:
    parts = line.split(",")
    if len(parts) != 7:
        return None
    try:
        quat = [float(x) for x in parts[:4]]
        m1 = float(parts[4])
        m2 = float(parts[5])
        acc = float(parts[6])
        return quat, m1, m2, acc
    except ValueError:
        return None


class StreamReader:
    def __init__(self, name: str, sn: str, path: str):
        self.name = name
        self.sn = sn
        self.path = path
        self.ser = open_port(path)
        self._rx = b""
        self.n_lines = 0
        self.n_parsed = 0

    def close(self) -> None:
        try:
            self.ser.close()
        except (OSError, serial.SerialException):
            pass

    def drain(self, raw: bool) -> None:
        try:
            waiting = self.ser.in_waiting
        except (OSError, serial.SerialException):
            return
        if waiting <= 0:
            return
        try:
            chunk = self.ser.read(min(waiting, 4096))
        except (OSError, serial.SerialException):
            return
        if not chunk:
            return

        self._rx += chunk
        if b"\n" not in self._rx:
            if len(self._rx) > 8192:
                self._rx = self._rx[-4096:]
            return

        parts = self._rx.split(b"\n")
        self._rx = parts[-1]
        complete = parts[:-1][-SERIAL_DRAIN_MAX_LINES:]

        for c in complete:
            s = c.decode("utf-8", errors="ignore").strip()
            if not s:
                continue
            self.n_lines += 1
            if raw:
                print(f"[{self.name}] {s}", flush=True)
                continue
            parsed = parse_line(s)
            if parsed is None:
                print(f"[{self.name}] (unparsed) {s}", flush=True)
                continue
            self.n_parsed += 1
            quat, m1, m2, acc = parsed
            print(
                f"[{self.name} SN={self.sn}] "
                f"q=[{quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f}] "
                f"m1={m1:+.4f} m2={m2:+.4f} acc_mag={acc:.3f}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log Front/Back Teensy telemetry to the terminal"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="How long to stream (default: 3)",
    )
    parser.add_argument("--front-only", action="store_true")
    parser.add_argument("--back-only", action="store_true")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw CSV lines instead of parsed fields",
    )
    args = parser.parse_args()
    if args.front_only and args.back_only:
        raise SystemExit("Use only one of --front-only / --back-only")

    readers: list[StreamReader] = []
    if not args.back_only:
        path = port_by_sn(SN_FRONT)
        readers.append(StreamReader("Front", SN_FRONT, path))
        print(f"Front -> {path} (SN={SN_FRONT})")
    if not args.front_only:
        path = port_by_sn(SN_BACK)
        readers.append(StreamReader("Back", SN_BACK, path))
        print(f"Back  -> {path} (SN={SN_BACK})")

    print(f"Logging for {args.seconds:.1f}s… (close controllerV2 / other serial users first)\n")
    time.sleep(0.3)
    for r in readers:
        try:
            r.ser.reset_input_buffer()
        except (OSError, serial.SerialException):
            pass

    t_end = time.time() + args.seconds
    try:
        while time.time() < t_end:
            for r in readers:
                r.drain(raw=args.raw)
            time.sleep(0.01)
    finally:
        for r in readers:
            print(
                f"[{r.name}] lines={r.n_lines} parsed={r.n_parsed}",
                flush=True,
            )
            r.close()


if __name__ == "__main__":
    main()
