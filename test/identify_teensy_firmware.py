#!/usr/bin/env python3
"""After flashing, confirm which Teensy has which motor_power firmware.

Reads serial banners from SN_FRONT / SN_BACK (from hardware/controllerV2.py).

Usage:
  python test/identify_teensy_firmware.py
"""

from __future__ import annotations

import re
import time

import serial
import serial.tools.list_ports

# Keep in sync with hardware/controllerV2.py
SN_FRONT = "18451300"
SN_BACK = "18452630"
BAUD = 115200


def port_by_sn(sn: str) -> str | None:
    for p in serial.tools.list_ports.comports():
        if p.serial_number == sn:
            return p.device
    return None


def read_banner(path: str, seconds: float = 4.0) -> str:
    # Opening the port usually resets the Teensy; do NOT flush — that deletes the banner.
    ser = serial.Serial(path, BAUD, timeout=0.1)
    try:
        # Pulse DTR to encourage a clean reboot on some CDC stacks.
        ser.dtr = False
        time.sleep(0.05)
        ser.dtr = True
        end = time.time() + seconds
        chunks: list[str] = []
        while time.time() < end:
            data = ser.read(512)
            if data:
                chunks.append(data.decode("utf-8", errors="replace"))
        return "".join(chunks)
    finally:
        ser.close()


def classify(text: str) -> str:
    if "encoder_check_FRONT" in text or "FIRMWARE: encoder_check_FRONT" in text:
        return "encoder_check_FRONT"
    if "encoder_check_BACK" in text or "FIRMWARE: encoder_check_BACK" in text:
        return "encoder_check_BACK"
    if "motor_power_FRONT" in text or "FIRMWARE: motor_power_FRONT" in text:
        return "motor_power_FRONT"
    if "motor_power_BACK" in text or "FIRMWARE: motor_power_BACK" in text:
        return "motor_power_BACK"
    if "motor_test_front" in text or "front DEBUG" in text:
        return "motor_test_front (or debug)"
    if "motor_test_back" in text or "back DEBUG" in text:
        return "motor_test_back (or debug)"
    if "PD_control_front" in text or "FIRMWARE PD_control_front" in text:
        return "PD_control_front"
    if "PD_control_back" in text or "FIRMWARE PD_control_back" in text:
        return "PD_control_back"
    if "PD_control" in text or re.search(r"\d\.\d+,\d\.\d+", text):
        return "likely PD_control / other"
    if not text.strip():
        return "(no serial output — open port may not have reset the board; try again)"
    return "(unknown — first lines below)\n" + "\n".join(text.splitlines()[:8])


def main() -> None:
    print("controllerV2 labels: SN_FRONT={}  SN_BACK={}".format(SN_FRONT, SN_BACK))
    print("teensy_loader_cli cannot select by SN — verify banners after flash.\n")

    for role, sn in (("FRONT label", SN_FRONT), ("BACK label", SN_BACK)):
        path = port_by_sn(sn)
        if not path:
            print(f"{role} SN={sn}: NOT FOUND on USB")
            continue
        print(f"{role} SN={sn} -> {path}")
        try:
            text = read_banner(path)
        except Exception as e:
            print(f"  read error: {e}")
            continue
        kind = classify(text)
        print(f"  firmware seen: {kind}")
        if role.startswith("FRONT") and (
            "motor_power_BACK" in kind or "encoder_check_BACK" in kind
        ):
            print("  *** MISMATCH: SN labeled FRONT is running BACK firmware")
        if role.startswith("BACK") and (
            "motor_power_FRONT" in kind or "encoder_check_FRONT" in kind
        ):
            print("  *** MISMATCH: SN labeled BACK is running FRONT firmware")
        print()


if __name__ == "__main__":
    main()
