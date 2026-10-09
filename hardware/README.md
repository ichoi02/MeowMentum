# Hardware

The robot is two bodies joined by a spine (front roll `rot1`, spine `pitch`, rear roll
`rot2`) plus a `tail` on the rear body. A Raspberry Pi runs the distilled student policy
(`policies/cat_controller{,_notail}.onnx`) at 50 Hz. Two Teensy 4.0 boards, connected to
the Pi over USB, close a 1 kHz PD loop on two joints each.

| Board | Firmware | Motor 1 | Motor 2 | IMU |
|---|---|---|---|---|
| Front | `PD_control_front/` | `rot1`, 9.68:1 | `pitch`, 34.014:1 | BNO085 over I2C (50 kHz): game rotation vector and accelerometer at 100 Hz |
| Back | `PD_control_back/` | `tail`, 34.014:1 | `rot2`, 9.68:1 | none; the board streams a placeholder |

- **Motor drivers.** Each board drives its two brushed DC gearmotors through a Pololu
  VNH5019 dual driver, with 20 kHz, 10-bit PWM.
- **Encoders.** Quadrature, 48 counts per motor revolution. The joint angle is
  `counts / 48 / gear * 2*pi`, which is 13.5 mrad per count on roll and 3.85 mrad on
  pitch and tail.
- **Joint PD.** The gains are the simulator's `pd_nominal` × 1024:
  - roll: Kp 4096, Kd 409.6
  - pitch and tail: Kp 20480, Kd 1024
- **Control law.** The D term is computed on the measurement, with a one-pole velocity
  low-pass (`VEL_ALPHA` 0.1). Inside a 0.03 rad deadband the motor is stopped. Outside
  it, any nonzero command is raised to at least 100/1023 PWM.
- **Matching the simulator.** These constants are mirrored in `cat_env/env_util.py` and
  `cat_env/cat_env.py` (`pd_nominal`). Change both sides together.
  `tools/tune_pd_gains.py` is the MuJoCo step-response sweep behind these gains. With the
  released model and randomisation ranges it selects the deployed pitch and tail gains
  (Kp 20, Kd 1.0) but picks Kp 3, Kd 0.4 for roll; the flashed roll pair is Kp 4, Kd 0.4.

Gear ratios are listed as configured in the firmware and the simulator.

| File | Purpose |
|---|---|
| `controller.py` | Pi main loop: tare, wait for free fall, run the policy, log telemetry |
| `PD_control_front/`, `PD_control_back/` | Teensy firmware (Arduino sketches) |
| `flash_pd_control_teensy.sh` | Compile both sketches with arduino-cli and flash them with teensy_loader_cli |
| `discover_teensy.py` | List Teensy serial ports and their USB serial numbers |
| `e2e_test.py` | Hardware-free test: the controller code driving the MuJoCo sim |

## Setup

**Pi software.** Install numpy, scipy (1.14 or newer, for `scalar_first`), gymnasium
(imported via `cat_env`), onnxruntime and pyserial. Pinned versions are in
`requirements.txt`.

**Firmware.** You need three things:
- arduino-cli with the Teensy core. This is Teensyduino, FQBN `teensy:avr:teensy40`, and
  it bundles the PJRC `Encoder` and `Wire` libraries.
- The `Adafruit BNO08x` library (front board only).
- teensy_loader_cli. On Linux, also install the PJRC udev rules.

Install the core and library as follows:

```bash
arduino-cli core install teensy:avr --additional-urls https://www.pjrc.com/teensy/package_teensy_index.json
arduino-cli lib install "Adafruit BNO08x"
```

Then compile and flash:

```bash
bash hardware/flash_pd_control_teensy.sh              # compile + flash both boards
bash hardware/flash_pd_control_teensy.sh --front      # one board
bash hardware/flash_pd_control_teensy.sh --compile-only
```

teensy_loader_cli cannot pick a board by serial number. At each step, press the program
button on the board the script names. The arduino-cli binary is taken from
`$ARDUINO_CLI`, then `$CONDA_PREFIX/bin/arduino-cli`, then `PATH`. The Arduino IDE with
Teensyduino also works: open the `.ino`, select board Teensy 4.0 and USB type Serial.

**Serial numbers.** `python hardware/discover_teensy.py` lists each board's USB serial
number. Put the two numbers into `SN_FRONT` and `SN_BACK` at the top of `controller.py`.
The flash script reads the same two lines.

## Running a drop

1. Set the robot upright and level, in its home pose with the spine straight. The
   controller zeroes the encoders and tares the IMU at startup, so this pose becomes the
   zero of every joint and the identity orientation.
2. Start the controller, naming the morphology that is physically on the robot:
   ```bash
   python hardware/controller.py --variant tail      # or --variant notail
   ```
   `--variant` is required (except with `--debug`). It prints the variant and the policy path
   and sha256. Then it:
   1. reboots both boards
   2. sends `STOP` and `RESET` (zero the encoders)
   3. sends `RESET_IMU` (tare)
   4. flushes the serial input
   5. sends `START` (drivers enabled; the joints hold zero)
3. Press Enter. While you hold the robot at the release attitude, the script prints the
   front attitude (Euler xyz, degrees) at 50 Hz.
4. Release the robot. Free fall is detected when the front accelerometer magnitude drops
   below 3.5 m/s^2. The policy then runs at 50 Hz for 1.0 s (50 steps). After that the
   motors stop and the log is written to `telemetry/<variant>_<unix time>.csv` (relative
   to the working directory; use `--log-dir` to change it), with a `.json` sidecar that
   records the variant, the ONNX path and its sha256, and the command line. In the logged drops the
   impact falls around step 40, so the last rows of each log are after landing.

Each run is one drop. Restart the script for the next drop, which re-zeroes and re-tares
the robot.

Other flags:
- `--onnx PATH` flies a different exported student.
- `--motoroff` leaves the drivers disabled.
- `--debug` skips the policy; you type `<motor> <target rad>` to move joints by hand
  (1 front roll, 2 pitch, 3 tail, 4 rear roll).

## Policy interface (`controller.py`)

The observation has 11 dimensions, in float32. It is identical to
`distillation.build_student_obs`.

| Index | Content |
|---|---|
| 0-2 | Projected gravity `R^T [0, 0, -1]` from the front quaternion, after tare and alignment `Rz(90°)^-1 · q · Rz(90°)` |
| 3-6 | Joint angles `[rot1, pitch, rot2, tail]` in rad. For `notail` the tail entry is forced to 0. |
| 7-9 | The previous raw action `[roll, pitch, tail]` after clipping (zeros at release). For `notail` the tail entry is 0. During DAgger, `distillation.py` instead fed back the unzeroed tail output. |
| 10 | `min(step / 50, 1)`, which is 0 on the first step |

**Action.** The ONNX output is 3 values in [-1, 1], clipped. For `notail` the tail
output is set to 0, so the PD holds the tail joint at zero. Each channel then goes
through a low-pass `a_f <- alpha * a + (1 - alpha) * a_f` with `alpha = [0.1, 0.3, 0.1]`
for roll, pitch and tail. The filter state is zero at release. The filtered values map
to joint targets:

- `rot1 = 7.28 * a_f[0]`
- `pitch = 1.57 * a_f[1]`
- `tail = 1.9199 * a_f[2]`
- `rot2 = -rot1`

All targets are in rad, and the ranges come from `model/cat.xml`.

## Serial protocol

The link is USB CDC (nominally 115200 baud). Lines end in a newline.

| Direction | Line | Meaning |
|---|---|---|
| Pi → board | `m1,m2` (4 decimals) | Joint targets in rad: front `rot1,pitch`, back `tail,rot2` |
| Pi → board | `RESET` | Zero both encoders |
| Pi → board | `RESET_IMU` | Front: tare all axes (the current orientation becomes identity). Back: ignored. |
| Pi → board | `RESET_I2C` | Front: re-initialise the I2C bus and the IMU. Back: ignored. |
| Pi → board | `START` / `STOP` | Enable / disable the motor driver |
| Pi → board | `REBOOT` | Software reset |
| board → Pi | `qr,qi,qj,qk,angle1,angle2,acc` every 20 ms | Quaternion (w,x,y,z), motor 1 and motor 2 joint angles in rad, accelerometer magnitude in m/s^2 (includes gravity). The back board always sends `1,0,0,0` and `acc` 0. |

The controller uses only the newest complete line from each board on each tick.

## Telemetry CSV

There is one row per 50 Hz control step, starting at free-fall detection: normally 50
rows; 2 of the 73 logs in `data/hardware_trials/` have 49. Each row holds the sensor
values the policy used on that step and the targets it sent.

| Column | Meaning | Unit |
|---|---|---|
| `Time` | Time since Enter was pressed (not since release), 4 decimals | s |
| `F_Q0..F_Q3` | Front body orientation quaternion w,x,y,z (tared and aligned) | – |
| `F_M1` | `rot1` joint angle (front roll) | rad |
| `F_M2` | `pitch` joint angle | rad |
| `F_ACC` | Front accelerometer magnitude, including gravity | m/s^2 |
| `Cmd_F1` | `rot1` target | rad |
| `Cmd_F2` | `pitch` target | rad |
| `B_Q0..B_Q3` | Placeholder, always 1,0,0,0 (no back IMU) | – |
| `B_M1` | `tail` joint angle | rad |
| `B_M2` | `rot2` joint angle (rear roll) | rad |
| `B_ACC` | Placeholder, always 0 | – |
| `Cmd_B1` | `tail` target | rad |
| `Cmd_B2` | `rot2` target (= -`Cmd_F1`) | rad |

The rear-body orientation is not measured. Reconstruct it from the front orientation and
the encoders: `R_rear = R_front · Rx(F_M1) · Ry(F_M2) · Rx(B_M2)`.

## Checks

- `python hardware/e2e_test.py [--variant tail|notail]` (no hardware needed). It runs
  `controller.py`'s observation, inference, filter and mapping code against the MuJoCo
  sim. It checks:
  - ONNX against PyTorch
  - the controller observation against the sim student observation
  - the filter and mapping against the target the sim executes
  - a closed-loop righting tripwire
- `python tools/verify_onnx.py --replay` performs three checks:
  - the ONNX weights are bit-identical to `policies/student_policy*.pth`
  - torch and onnxruntime agree
  - it replays every drop in `data/hardware_trials/` through both ONNX files and reports
    which policy flew it

  Every logged command is reproduced to about 1e-15. The replay also shows that 19 of
  the 37 `no_tail` drops were flown with the tail policy: the controller then defaulted
  to `--variant tail`. This is why `--variant` is now required and recorded in the
  telemetry filename and its sidecar.
