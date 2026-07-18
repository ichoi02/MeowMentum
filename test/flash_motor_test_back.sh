#!/usr/bin/env bash
# Compile and flash test/motor_test_back to the back Teensy (SN_BACK).
#
# Prerequisites: conda env cat + arduino-cli, teensy-loader-cli, PJRC udev rules.
#
# Usage:
#   conda activate cat
#   bash test/flash_motor_test_back.sh
#   bash test/flash_motor_test_back.sh --compile-only
#   bash test/flash_motor_test_back.sh --flash-only
#   FQBN=teensy:avr:teensy41 bash test/flash_motor_test_back.sh
#
# teensy_loader_cli cannot select by USB serial — press the program button on
# the back board only before flashing.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash test/flash_motor_test_back.sh [options]

  (default)      Compile motor_test_back, then flash to SN_BACK.
  --compile-only Compile only; skip teensy_loader_cli.
  --flash-only   Flash hex already under the build directory.
  --help, -h     This text.

  FQBN=teensy:avr:teensy41   Board variant (default: teensy:avr:teensy40)
  MOTOR_TEST_BUILD_DIR=...   Build output root (default: test/.motor_test_build)
EOF
}

_strip_openblas_preload() {
  local p="${LD_PRELOAD:-}"
  [[ "$p" == *libopenblas* ]] || return 0
  local -a keep=() parts
  local part joined
  if [[ "$p" == *:* ]]; then
    IFS=':' read -ra parts <<< "$p"
    for part in "${parts[@]}"; do
      [[ -n "$part" && "$part" != *libopenblas* ]] && keep+=("$part")
    done
    if ((${#keep[@]})); then
      joined="$(printf '%s:' "${keep[@]}")"
      export LD_PRELOAD="${joined%:}"
    else
      unset LD_PRELOAD
    fi
  else
    for part in $p; do
      [[ -n "$part" && "$part" != *libopenblas* ]] && keep+=("$part")
    done
    if ((${#keep[@]})); then
      export LD_PRELOAD="${keep[*]}"
    else
      unset LD_PRELOAD
    fi
  fi
}
_strip_openblas_preload

FQBN="${FQBN:-teensy:avr:teensy40}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTROLLER_PY="${REPO_ROOT}/hardware/controllerV2.py"
SKETCH="${SCRIPT_DIR}/motor_test_back"
INO="${SKETCH}/motor_test_back.ino"
DEFAULT_BUILD_ROOT="${SCRIPT_DIR}/.motor_test_build"
OUT_DIR="${MOTOR_TEST_BUILD_DIR:-$DEFAULT_BUILD_ROOT}/back"
HEX="${OUT_DIR}/motor_test_back.ino.hex"

OPT_COMPILE_ONLY=0
OPT_FLASH_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compile-only) OPT_COMPILE_ONLY=1 ;;
    --flash-only) OPT_FLASH_ONLY=1 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown option: $1 (try --help)" >&2
      exit 1
      ;;
  esac
  shift
done

if [[ "$OPT_COMPILE_ONLY" -eq 1 && "$OPT_FLASH_ONLY" -eq 1 ]]; then
  echo "Use either --compile-only or --flash-only, not both." >&2
  exit 1
fi

DO_COMPILE=1
DO_FLASH=1
[[ "$OPT_COMPILE_ONLY" -eq 1 ]] && DO_FLASH=0
[[ "$OPT_FLASH_ONLY" -eq 1 ]] && DO_COMPILE=0

if [[ "$DO_COMPILE" -eq 1 ]]; then
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "conda activate cat first."
    exit 1
  fi
  CLI="${CONDA_PREFIX}/bin/arduino-cli"
  if [[ ! -x "$CLI" ]]; then
    echo "arduino-cli missing. Run: bash hardware/install_arduino_cli_cat_env.sh"
    exit 1
  fi
  if [[ ! -f "$INO" ]]; then
    echo "Missing ${INO}"
    exit 1
  fi
fi

if [[ "$DO_FLASH" -eq 1 ]]; then
  if ! command -v teensy_loader_cli >/dev/null 2>&1; then
    echo "teensy_loader_cli not found. Install: sudo apt install teensy-loader-cli"
    exit 1
  fi
fi

if [[ ! -f "$CONTROLLER_PY" ]]; then
  echo "Missing ${CONTROLLER_PY}"
  exit 1
fi

SN_BACK="$(python3 - "$CONTROLLER_PY" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r'^SN_BACK\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
if not m:
    raise SystemExit(f"Could not parse SN_BACK from {sys.argv[1]}")
print(m.group(1))
PY
)"

if [[ "$DO_COMPILE" -eq 1 ]]; then
  export ARDUINO_DIRECTORIES_DATA="${CONDA_PREFIX}/arduino/data"
  export ARDUINO_DIRECTORIES_USER="${CONDA_PREFIX}/arduino/user"
  rm -rf "$OUT_DIR"
  mkdir -p "$OUT_DIR"
  echo "Compiling Back motor test: FQBN=$FQBN -> $OUT_DIR"
  "$CLI" compile --clean --fqbn "$FQBN" --output-dir "$OUT_DIR" "$SKETCH"
fi

if [[ ! -f "$HEX" ]]; then
  echo "Missing $HEX" >&2
  echo "  Run a compile first (this script without --flash-only)." >&2
  exit 1
fi

echo "Back hex: $HEX ($(wc -c <"$HEX") bytes)"

if [[ "$DO_FLASH" -eq 0 ]]; then
  echo "Compile-only finished."
  echo "Flash later with: bash test/flash_motor_test_back.sh --flash-only"
  exit 0
fi

MCU=TEENSY40
case "$FQBN" in
  *teensy41*) MCU=TEENSY41 ;;
  *teensy36*) MCU=TEENSY36 ;;
  *teensy32*) MCU=TEENSY32 ;;
  *teensy31*) MCU=TEENSY31 ;;
  *teensy30*) MCU=TEENSY30 ;;
esac

echo ">>> Back (USB serial ${SN_BACK}) <<<"
echo "    Firmware: $(basename "$HEX")"
echo "    Press the program button on the BACK board only."
if ! teensy_loader_cli --mcu="$MCU" -w -v "$HEX"; then
  echo "USB error, retrying..."
  sleep 1
  teensy_loader_cli --mcu="$MCU" -w -v "$HEX"
fi

echo "Done. Back -> ${INO}"
