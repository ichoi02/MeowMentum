#!/usr/bin/env bash
# Compile hardware/PD_control_front and hardware/PD_control_back, then flash each hex to
# the matching Teensy (SN_FRONT / SN_BACK from hardware/controller.py).
# teensy_loader_cli cannot select by USB serial; on each step put only that board in
# bootloader (program button) — the script prints role + serial to match hardware.
#
# Prerequisites: conda env cat + arduino-cli, teensy-loader-cli, PJRC udev rules.
#
# Usage:
#   conda activate cat
#   bash hardware/flash_pd_control_teensy.sh              # both
#   bash hardware/flash_pd_control_teensy.sh --front      # compile + flash front only
#   bash hardware/flash_pd_control_teensy.sh --back       # back only
#   FQBN=teensy:avr:teensy41 bash hardware/flash_pd_control_teensy.sh --front
#
#   --help    show usage

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash hardware/flash_pd_control_teensy.sh [--front] [--back]

  (no flags)   Build and flash both boards (order: Front, then Back).
  --front      Build PD_control_front only; flash SN_FRONT.
  --back       Build PD_control_back only; flash SN_BACK.
  --front --back   Same as default (both).

  FQBN=teensy:avr:teensy41 ...   Teensy board variant (default: teensy:avr:teensy40)

Requires: conda activate cat, teensy_loader_cli, PJRC udev rules.
EOF
}

# Conda sometimes sets LD_PRELOAD to a missing OpenBLAS DSO → noisy ld.so errors in child processes.
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
CONTROLLER_PY="${SCRIPT_DIR}/controller.py"

SELECT_FRONT=0
SELECT_BACK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --front) SELECT_FRONT=1 ;;
    --back) SELECT_BACK=1 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown option: $1 (try --help)" >&2
      exit 1
      ;;
  esac
  shift
done
if [[ "$SELECT_FRONT" -eq 0 && "$SELECT_BACK" -eq 0 ]]; then
  SELECT_FRONT=1
  SELECT_BACK=1
fi
SKETCH_FRONT="${SCRIPT_DIR}/PD_control_front"
SKETCH_BACK="${SCRIPT_DIR}/PD_control_back"
INO_FRONT="${SKETCH_FRONT}/PD_control_front.ino"
INO_BACK="${SKETCH_BACK}/PD_control_back.ino"
OUT_BASE="${PD_CONTROL_BUILD_DIR:-${TMPDIR:-/tmp}/pd_teensy_build_$$}"
OUT_FRONT="${OUT_BASE}/front"
OUT_BACK="${OUT_BASE}/back"

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

if [[ "$SELECT_FRONT" -eq 1 && ! -f "$INO_FRONT" ]]; then
  echo "Missing ${INO_FRONT}"
  exit 1
fi
if [[ "$SELECT_BACK" -eq 1 && ! -f "$INO_BACK" ]]; then
  echo "Missing ${INO_BACK}"
  exit 1
fi

readarray -t FLASH_ORDER < <(CONTROLLER_PY="$CONTROLLER_PY" SELECT_FRONT="$SELECT_FRONT" SELECT_BACK="$SELECT_BACK" python3 <<'PY'
import os
import re

path = os.environ["CONTROLLER_PY"]
want_front = os.environ.get("SELECT_FRONT") == "1"
want_back = os.environ.get("SELECT_BACK") == "1"
text = open(path, encoding="utf-8").read()

def grab(name: str) -> str:
    m = re.search(rf"^{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    if not m:
        raise SystemExit(f"Could not parse {name} from {path}")
    return m.group(1)

if want_front:
    print(f"Front\t{grab('SN_FRONT')}")
if want_back:
    print(f"Back\t{grab('SN_BACK')}")
PY
)

export ARDUINO_DIRECTORIES_DATA="${CONDA_PREFIX}/arduino/data"
export ARDUINO_DIRECTORIES_USER="${CONDA_PREFIX}/arduino/user"

rm -rf "$OUT_BASE"
mkdir -p "$OUT_FRONT" "$OUT_BACK"

HEX_FRONT=""
HEX_BACK=""

if [[ "$SELECT_FRONT" -eq 1 ]]; then
  echo "Compiling Front: FQBN=$FQBN -> $OUT_FRONT"
  # --clean: avoid linking against zero-byte .o files from a corrupted ~/.cache/arduino/sketches entry.
  "$CLI" compile --clean --fqbn "$FQBN" --output-dir "$OUT_FRONT" "$SKETCH_FRONT"
  HEX_FRONT="${OUT_FRONT}/PD_control_front.ino.hex"
  if [[ ! -f "$HEX_FRONT" ]]; then
    echo "Expected hex not found: $HEX_FRONT"
    exit 1
  fi
fi

if [[ "$SELECT_BACK" -eq 1 ]]; then
  echo "Compiling Back: FQBN=$FQBN -> $OUT_BACK"
  "$CLI" compile --clean --fqbn "$FQBN" --output-dir "$OUT_BACK" "$SKETCH_BACK"
  HEX_BACK="${OUT_BACK}/PD_control_back.ino.hex"
  if [[ ! -f "$HEX_BACK" ]]; then
    echo "Expected hex not found: $HEX_BACK"
    exit 1
  fi
fi

MCU=TEENSY40
case "$FQBN" in
  *teensy41*) MCU=TEENSY41 ;;
  *teensy36*) MCU=TEENSY36 ;;
  *teensy32*) MCU=TEENSY32 ;;
  *teensy31*) MCU=TEENSY31 ;;
  *teensy30*) MCU=TEENSY30 ;;
esac

echo ""
[[ -n "$HEX_FRONT" ]] && echo "Front hex: $HEX_FRONT ($(wc -c <"$HEX_FRONT") bytes)"
[[ -n "$HEX_BACK" ]] && echo "Back hex:  $HEX_BACK ($(wc -c <"$HEX_BACK") bytes)"
echo ""

step=1
for line in "${FLASH_ORDER[@]}"; do
  IFS=$'\t' read -r role serial <<<"$line"
  case "$role" in
    Front) HEX="$HEX_FRONT"
      if [[ -z "$HEX" ]]; then
        echo "Internal error: Front selected but HEX_FRONT empty."
        exit 1
      fi
      ;;
    Back) HEX="$HEX_BACK"
      if [[ -z "$HEX" ]]; then
        echo "Internal error: Back selected but HEX_BACK empty."
        exit 1
      fi
      ;;
    *)
      echo "Unexpected role: $role"
      exit 1
      ;;
  esac
  echo ">>> Step $step: ${role} (USB serial ${serial}) <<<"
  echo "    Firmware: $(basename "$HEX")"
  echo "    Press the program button on THIS board only (or put only it in bootloader)."
  teensy_loader_cli --mcu="$MCU" -w -v "$HEX"
  echo ""
  ((step++)) || true
done

done_msg=()
[[ "$SELECT_FRONT" -eq 1 ]] && done_msg+=("Front -> ${INO_FRONT}")
[[ "$SELECT_BACK" -eq 1 ]] && done_msg+=("Back -> ${INO_BACK}")
printf -v _done_summary '%s; ' "${done_msg[@]}"
echo "Done. ${_done_summary%; }"
