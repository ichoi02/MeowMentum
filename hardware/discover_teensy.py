"""List the Teensy serial ports and their USB serial numbers.

    python -m hardware.discover_teensy          # Teensy ports only
    python -m hardware.discover_teensy --all    # every serial port

Copy the two boards' serial numbers into SN_FRONT / SN_BACK in hardware/controller.py.
With both boards plugged in, unplug one to tell which is which. If no Teensy port is
found, the Linux USB bus is scanned for PJRC boards that do not expose a serial port yet.
"""
import argparse
import os
import sys

import serial.tools.list_ports

# PJRC USB vendor ID and lowercase-hex product IDs (https://www.pjrc.com/teensy/usb_ids.html)
PJRC_VID = 0x16C0
PJRC_PID_MEANING = {
    "0478": "Halfkay bootloader — no serial device; upload sketch or exit bootloader",
    "0483": "USB Serial (Teensyduino) — should appear as /dev/ttyACM*",
    "0485": "USB Serial (alternate PID)",
    "0487": "USB Serial (RAW I/O)",
    "0489": "USB Dual Serial",
    "0490": "USB Triple Serial",
}

# Product IDs of the bootloader and of the modes that expose a serial port
PJRC_BOOTLOADER_PID = "0478"
PJRC_SERIAL_PIDS = ("0483", "0485", "0487", "0489", "0490")


def _linux_pjrc_pid_counts() -> dict[str, int]:
    """Count PJRC (16c0) USB devices by PID from sysfs (Linux only, else {})."""
    # Use the Linux USB device tree, if there is one
    base = "/sys/bus/usb/devices"
    if sys.platform != "linux" or not os.path.isdir(base):
        return {}

    # Count the PJRC devices by product ID
    counts: dict[str, int] = {}
    for name in os.listdir(base):
        # Read the device's vendor and product IDs, skipping entries without them
        device_dir = os.path.join(base, name)
        vendor_path = os.path.join(device_dir, "idVendor")
        product_path = os.path.join(device_dir, "idProduct")
        if not (os.path.isfile(vendor_path) and os.path.isfile(product_path)):
            continue
        try:
            with open(vendor_path, encoding="ascii", errors="replace") as f:
                vid = f.read().strip().lower()
            with open(product_path, encoding="ascii", errors="replace") as f:
                pid = f.read().strip().lower()
        except OSError:
            continue

        # Count it if it is a PJRC device
        if vid == "16c0":
            counts[pid] = counts.get(pid, 0) + 1
    return counts


def _is_teensy(port) -> bool:
    """Return True if a serial port looks like a Teensy (by name, maker or PJRC vendor ID).

    Linux often reports "USB Serial" / "Teensyduino" instead of "Teensy" / "PJRC".
    """
    description = port.description or ""
    manufacturer = port.manufacturer or ""
    return ("Teensy" in description or "PJRC" in manufacturer or "Teensyduino" in manufacturer
            or port.vid == PJRC_VID or "16C0" in (port.hwid or "").upper())


def _print_port(port, is_teensy) -> None:
    """Print one serial port's metadata."""
    print(f"Device: {port.device}")
    if not is_teensy:
        print("  (serial port, not classified as Teensy)")
    print(f"  Description: {port.description!r}")
    print(f"  Manufacturer: {port.manufacturer!r}")
    if port.vid is not None and port.pid is not None:
        print(f"  VID:PID: {port.vid:04x}:{port.pid:04x}")
    print(f"  HWID: {port.hwid!r}")
    print(f"  Serial Number: {port.serial_number!r}")
    print("-" * 30)


def _print_missing_teensy_hints(ports, list_all) -> None:
    """Explain why no Teensy port was found, using the Linux USB bus scan if available."""
    # Report the miss
    print("No Teensy serial ports found. Check USB cable and power.")

    # Look for PJRC boards on the USB bus
    pjrc = _linux_pjrc_pid_counts()

    # PJRC boards are plugged in but none has a serial port: say what mode each is in
    if pjrc:
        total = sum(pjrc.values())
        print(
            f"USB scan: {total} PJRC device(s) (VID 16c0) plugged in, "
            "but none expose a serial port yet:"
        )
        for pid in sorted(pjrc.keys()):
            hint = PJRC_PID_MEANING.get(pid, "see PJRC USB ID list")
            print(f"  {pjrc[pid]}× PID {pid} — {hint}")
        if pjrc.get(PJRC_BOOTLOADER_PID) and not any(p in pjrc for p in PJRC_SERIAL_PIDS):
            print(
                "All visible boards are in the bootloader. Program them "
                "(Arduino/Teensyduino or teensy_loader_cli); after your "
                "sketch runs with USB Serial, re-run this tool."
            )

    # Otherwise suggest --all, or report the empty USB scan
    elif not list_all and ports:
        print("Tip: run with --all to list every serial port.")
    elif sys.platform == "linux":
        print("No PJRC USB devices seen under /sys/bus/usb/devices either.")


def discover_teensys(*, list_all: bool = False) -> None:
    """Print the Teensy serial ports (every serial port with list_all), or hints if none."""
    # List the serial ports
    print("--- Teensy Discovery Tool ---")
    ports = list(serial.tools.list_ports.comports())

    # Print each Teensy port, and every other port too with list_all
    found = False
    for port in ports:
        is_teensy = _is_teensy(port)
        found = found or is_teensy
        if is_teensy or list_all:
            _print_port(port, is_teensy)

    # Explain an empty result
    if not found:
        _print_missing_teensy_hints(ports, list_all)


def main():
    """Parse the command line and list the ports."""
    # Parse the command line
    parser = argparse.ArgumentParser(description="Find Teensy serial devices")
    parser.add_argument(
        "--all",
        action="store_true",
        help="List all serial ports with metadata (debugging)",
    )
    args = parser.parse_args()

    # List the ports
    discover_teensys(list_all=args.all)


if __name__ == "__main__":
    main()
