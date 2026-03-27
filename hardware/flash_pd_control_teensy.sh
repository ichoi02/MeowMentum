#!/usr/bin/env bash
# Compile hardware/PD_control/PD_control.ino (Teensy 4.0) and flash with teensy_loader_cli.
# Flash order matches hardware/controller.py: Front (SN_FRONT) first, then Back (SN_BACK).
# teensy_loader_cli cannot select a device by USB serial; for each step press the program
# button on the board whose serial number is shown (only one board in HalfKay at a time).
#
# Prerequisites: conda env cat + arduino-cli, teensy-loader-cli, PJRC udev rules.
#
# Usage:
#   conda activate cat
#   bash hardware/flash_pd_control_teensy.sh
#   FQBN=teensy:avr:teensy41 bash hardware/flash_pd_control_teensy.sh

set -euo pipefail

FQBN="${FQBN:-teensy:avr:teensy40}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROLLER_PY="${SCRIPT_DIR}/controller.py"
SKETCH_DIR="${SCRIPT_DIR}/PD_control"
SKETCH_INO="${SKETCH_DIR}/PD_control.ino"
OUT="${PD_CONTROL_BUILD_DIR:-${TMPDIR:-/tmp}/pd_teensy_build_$$}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "conda activate cat first."
  exit 1
fi

CLI="${CONDA_PREFIX}/bin/arduino-cli"
if [[ ! -x "$CLI" ]]; then
  echo "arduino-cli missing. Run: bash hardware/install_arduino_cli_cat_env.sh"
  exit 1
fi

if ! command -v teensy_loader_cli >/dev/null 2>&1; then
  echo "teensy_loader_cli not found. Install: sudo apt install teensy-loader-cli"
  exit 1
fi

if [[ ! -f "$CONTROLLER_PY" ]]; then
  echo "Missing ${CONTROLLER_PY}"
  exit 1
fi

if [[ ! -f "$SKETCH_INO" ]]; then
  echo "Missing sketch ${SKETCH_INO}"
  exit 1
fi

readarray -t FLASH_ORDER < <(CONTROLLER_PY="$CONTROLLER_PY" python3 <<'PY'
import os
import re

path = os.environ["CONTROLLER_PY"]
text = open(path, encoding="utf-8").read()

def grab(name: str) -> str:
    m = re.search(rf"^{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    if not m:
        raise SystemExit(f"Could not parse {name} from {path}")
    return m.group(1)

print(f"Front\t{grab('SN_FRONT')}")
print(f"Back\t{grab('SN_BACK')}")
PY
)

export ARDUINO_DIRECTORIES_DATA="${CONDA_PREFIX}/arduino/data"
export ARDUINO_DIRECTORIES_USER="${CONDA_PREFIX}/arduino/user"

rm -rf "$OUT"
mkdir -p "$OUT"

echo "Compiling FQBN=$FQBN -> $OUT"
"$CLI" compile --fqbn "$FQBN" --output-dir "$OUT" "$SKETCH_DIR"

HEX="${OUT}/PD_control.ino.hex"
if [[ ! -f "$HEX" ]]; then
  echo "Expected hex not found: $HEX"
  exit 1
fi

MCU=TEENSY40
case "$FQBN" in
  *teensy41*) MCU=TEENSY41 ;;
  *teensy36*) MCU=TEENSY36 ;;
  *teensy32*) MCU=TEENSY32 ;;
  *teensy31*) MCU=TEENSY31 ;;
  *teensy30*) MCU=TEENSY30 ;;
esac

echo "Hex: $HEX ($(wc -c <"$HEX") bytes)"
echo ""

step=1
for line in "${FLASH_ORDER[@]}"; do
  IFS=$'\t' read -r role serial <<<"$line"
  echo ">>> Step $step: ${role} (USB serial ${serial}) <<<"
  echo "    Press the program button on THIS board only (or put only it in bootloader)."
  teensy_loader_cli --mcu="$MCU" -w -v "$HEX"
  echo ""
  ((step++)) || true
done

echo "Done. Front and Back firmware should match ${SKETCH_INO}."
