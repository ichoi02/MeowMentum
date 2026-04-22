# MeowMentum — Mechatronics Reference

MeowMentum is an aerial righting robot: it is dropped upside-down (or at an angle) and must rotate itself to land feet-first within ~0.71 s (2.5 m free-fall). Two independent Teensy 4.0 boards run embedded PD controllers at 1 kHz; a host Python process runs a neural-network policy at 50 Hz and logs telemetry for post-flight analysis.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Host (Python 50 Hz)                │
│  controller.py                                      │
│  ┌──────────┐   ONNX inference   ┌───────────────┐  │
│  │ Sensor   │ ─────────────────► │ Neural Network│  │
│  │ Parsing  │                    │ Policy (22D→3D)│  │
│  └──────────┘                    └───────┬───────┘  │
│       ▲  telemetry CSV / UDP            │ motor cmds│
└───────┼─────────────────────────────────┼───────────┘
        │ Serial USB 115200               │ Serial USB
   ┌────┴────┐                       ┌────┴────┐
   │ Front   │                       │  Back   │
   │Teensy4.0│                       │Teensy4.0│
   │ 1 kHz   │                       │ 1 kHz   │
   └────┬────┘                       └────┬────┘
   IMU  │  Encoders                  IMU  │  Encoders
  (I2C) │  (quad)                   (I2C) │  (quad)
        │                                 │
   Motor1 (rot1)                    Motor1 (rot2)
   Motor2 (pitch)                   Motor2 (tail)
```

**Front board** controls the front body: spine roll (rot1) and spine pitch.  
**Back board** controls the rear body: rear roll (rot2) and tail.

---

## 2. Microcontrollers — Teensy 4.0

| Property | Value |
|---|---|
| MCU | ARM Cortex-M7 @ 600 MHz |
| Firmware | Arduino (.ino) via Teensyduino |
| Loop rate | 1 kHz (1000 µs timer interrupt) |
| Serial | USB CDC at 115200 baud |
| Telemetry rate | 50 Hz (every 20 ms) |
| Watchdog | Motors auto-stop after 500 ms without a command |

**Serial numbers** (used by `controller.py` to identify boards at startup):

| Board | Serial Number |
|---|---|
| Front | `18451300` |
| Back | `18452630` |

**Startup command sequence** sent by the host:

```
REBOOT       → hard-reset the Teensy, wait 2 s for USB re-enumeration
RESET        → zero all encoder counts
RESET_IMU    → tare quaternions to identity
START        → enable motor H-bridge enable pins
```

---

## 3. Sensors

### 3.1 IMU — Adafruit BNO08x

One per board, on the I2C bus.

| Property | Value |
|---|---|
| Chip | BNO085 / BNO08x |
| Interface | I2C at 50 kHz (low speed for long-wire stability) |
| I2C address | 0x4A |
| Pins | SDA = 18, SCL = 19 (both boards) |
| Output rate | 50 Hz via Game Rotation Vector report |

**Data produced per board:**

| Field | Description |
|---|---|
| Quaternion (w, x, y, z) | Gravity-compensated orientation, scalar-first |
| Accelerometer (x, y, z) | Linear acceleration in m/s² |
| `ACC` magnitude | `√(ax²+ay²+az²)` — used for free-fall detection |

**Telemetry columns:** `F_Q0 F_Q1 F_Q2 F_Q3 F_ACC` (front), `B_Q0 B_Q1 B_Q2 B_Q3 B_ACC` (back).

**Alignment:** Raw sensor quaternions are rotated in software to align with the global body frame:

| Board | Alignment rotation |
|---|---|
| Front | +90° about Z |
| Back | +180° about X, then −90° about Z |

**I2C watchdog (front board only):** If no IMU data arrives for 1500 ms, the firmware manually bit-bangs SCL up to 20 times to un-wedge the bus, then re-initialises I2C and re-enables the sensor report.

---

### 3.2 Motor Encoders

Quadrature incremental encoders on each motor shaft.

| Property | Value |
|---|---|
| Type | Quadrature (A/B channels) |
| Ticks per motor revolution | 48 |
| Gear ratio | 9.68 : 1 (all four motors) |
| Angle formula | `θ (rad) = ticks / 48 / 9.68 × 2π` |

**Pin mapping:**

| Board | Motor | Channel A | Channel B |
|---|---|---|---|
| Front | rot1  | 14 | 15 |
| Front | pitch | 17 | 16 |
| Back  | rot2  | 13 | 14 |
| Back  | tail  | 15 | 16 |

**Telemetry columns:** `F_M1 F_M2` (front rot1, pitch), `B_M1 B_M2` (back rot2, tail), in radians.

---

## 4. Actuators — DC Gear Motors

Four brushed DC motors with 9.68:1 gearboxes, driven by H-bridge ICs.

| Property | Value |
|---|---|
| Driver type | DRV8833-compatible H-bridge |
| PWM frequency | 20 kHz |
| PWM resolution | 10-bit (0–1023) |
| Deadband | ±0.03 rad (motor stops if error is within this range) |
| Minimum PWM | 100 (below this the motor stalls) |

**Pin mapping:**

| Board | Motor | INA | INB | PWM | EN |
|---|---|---|---|---|---|
| Front | rot1  | 2 | 4 | 9  | 6  |
| Front | pitch | 7 | 8 | 12 | 10 |
| Back  | rot2  | 2 | 4 | 6  | 9  |
| Back  | tail  | 7 | 8 | 10 | 12 |

**Motor direction reversal** is configured per-motor in firmware (a flag negates the PWM sign before writing to INA/INB/PWM).

---

## 5. Embedded PD Controller (Firmware, 1 kHz)

Each Teensy runs an independent PD controller for its two motors at 1 kHz.

```
error      = target_angle − current_encoder_angle
derivative = (error − prev_error) / dt
output     = Kp × error + Kd × derivative
output     = clip(output, −1023, +1023)

if |error| < 0.03 rad:  stop motor (deadband)
if |output| < 100:      stop motor (minimum PWM)
```

**Gains:**

| Board | Motor | Kp | Kd |
|---|---|---|---|
| Front | rot1  | 2048   | 20.48  |
| Front | pitch | 20480  | 204.8  |
| Back  | rot2  | 1024   | 10.24  |
| Back  | tail  | 1024   | 10.24  |

**Command format** (host → firmware over serial):

```
angle1,angle2\n        # two floats in radians, comma-separated
```

---

## 6. Host Control Loop (`controller.py`, 50 Hz)

### 6.1 Startup Sequence

1. Discover both Teensies by USB serial number.
2. Send `REBOOT` to both; wait 2 s; reconnect.
3. Send `RESET` (zero encoders) and `RESET_IMU` (tare quaternions).
4. Flush stale serial buffers.
5. Load ONNX model.
6. Send `START` to enable motors.
7. Poll accelerometer until `acc_mag < 3.0 m/s²` (free-fall detected → begin control).

### 6.2 Control Loop (each 20 ms tick)

```
1. Read latest telemetry line from each board (non-blocking, last of buffered lines)
2. Parse: quaternions, encoder angles, accelerometer magnitude
3. Apply alignment rotations to quaternions
4. Build 22-D observation:
     front_rotation_matrix (9D, flattened 3×3)
     back_rotation_matrix  (9D, flattened 3×3)
     joint_angles          (4D: rot1, pitch, rot2, tail)
5. ONNX inference → 3D action (tanh output, range [−1, 1])
6. Map actions to target angles:
     rot1  (roll)  : action × 2π
     pitch         : action × π/2
     tail          : action × π/2
     rot2          : −rot1   (symmetric)
7. Send "rot1_target,pitch_target\n" to front board
   Send "rot2_target,tail_target\n"  to back board
8. Log row to CSV telemetry file
9. Broadcast key:value pairs to UDP 127.0.0.1:47269 (Teleplot)
10. Exit after 0.8 s control duration
```

### 6.3 Telemetry CSV columns

```
Time, F_Q0, F_Q1, F_Q2, F_Q3, F_M1, F_M2, F_ACC, Cmd_F1, Cmd_F2,
      B_Q0, B_Q1, B_Q2, B_Q3, B_M1, B_M2, B_ACC, Cmd_B1, Cmd_B2
```

---

## 7. Neural Network Policy

| Property | Value |
|---|---|
| Format | ONNX (converted from PyTorch) |
| Input | 22-D (two 3×3 rotation matrices + 4 joint angles) |
| Output | 3-D (roll, pitch, tail), tanh-bounded to [−1, 1] |
| Training algorithm | PPO (Proximal Policy Optimization) |
| Compression | DAgger distillation from full-state expert to partial-obs student |
| Simulator | MuJoCo 3.5.0 |
| Model files | `cat_controller.onnx` + `cat_controller.onnx.data` |

**Why rotation matrices instead of quaternions?** Rotation matrices avoid the quaternion double-cover ambiguity (q and −q represent the same rotation) and avoid Euler-angle singularities, giving the network a smoother, unambiguous input.

---

## 8. Communication Summary

| Link | Protocol | Rate | Direction |
|---|---|---|---|
| Host ↔ Front Teensy | USB Serial 115200 | 50 Hz telemetry / ad-hoc commands | Bidirectional |
| Host ↔ Back Teensy  | USB Serial 115200 | 50 Hz telemetry / ad-hoc commands | Bidirectional |
| Teensy ↔ IMU        | I2C 50 kHz        | 50 Hz sensor reports | Read only |
| Teensy ↔ Encoders   | Digital GPIO (quadrature) | 1 kHz poll | Read only |
| Teensy ↔ Motors     | PWM + GPIO (H-bridge) | 1 kHz | Write only |
| Host → Teleplot      | UDP 127.0.0.1:47269 | 50 Hz | Write only |

---

## 9. Data Flow (End-to-End)

```
BNO08x IMU (50 Hz)
Quadrature encoders (1 kHz)
        │
        ▼
Teensy firmware (1 kHz loop)
  ├─ PD controller → PWM → motor
  └─ Telemetry buffer → Serial @ 50 Hz
        │
        ▼
controller.py (50 Hz)
  ├─ Parse quaternions + encoder angles
  ├─ Build rotation-matrix observation
  ├─ ONNX inference → action
  ├─ Send motor targets back via serial
  ├─ Write CSV row
  └─ Broadcast to Teleplot (UDP)
        │
        ▼
data_analysis.py (post-flight)
  ├─ Load CSV
  ├─ Detect impact, clip to 2.5 m window
  ├─ Compute FK rear orientation (front IMU × encoder chain)
  ├─ Compute angular velocity from quaternion finite differences
  └─ Save plots + report.csv
```

---

## 10. Key Files

| File | Role |
|---|---|
| `hardware/controller.py` | Host control loop, serial parsing, ONNX inference, telemetry logging |
| `hardware/PD_control_front/PD_control_front.ino` | Front Teensy firmware (rot1 + pitch, front IMU) |
| `hardware/PD_control_back/PD_control_back.ino` | Back Teensy firmware (rot2 + tail, back IMU) |
| `hardware/discover_teensy.py` | Finds Teensy COM/tty ports by USB serial number |
| `model/cat.xml` | MuJoCo robot description (joints, bodies, geometry) |
| `cat_env/cat_env.py` | Simulation environment + reward function |
| `cat_env/env_util.py` | PD controller class, rotation utilities, sensor noise |
| `distillation.py` | DAgger training loop (expert → student compression) |
| `onnx_conversion2.py` | PyTorch → ONNX export |
| `data_analysis/data_analysis.py` | Post-flight telemetry analysis and reporting |
| `data_analysis/mujoco_playback.py` | Replay flight in MuJoCo viewer or save video |
| `data_analysis/plot_performance.py` | Performance scatter plot across all trials |
