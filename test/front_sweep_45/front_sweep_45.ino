/*
 * Front Teensy — ±45° PD sweep with encoder logging.
 * M1 = roll, M2 = pitch (same as hardware/controllerV2.py / PD_control_front).
 *
 * Each selected motor: 45° CW, settle, then CCW back to home.
 * Move aborts if not within deadband in MOVE_TIMEOUT_MS (1 s).
 *
 * Streams CSV:
 *   t_ms,m1_rad,m2_rad,tgt1,tgt2,active_motor
 * Event lines start with '#'.
 *
 * Serial commands (host must send; does not auto-sweep on boot):
 *   RUN       — sweep both motors (M1 then M2)
 *   RUN BOTH  — same
 *   RUN M1    — M1 (roll) only
 *   RUN M2    — M2 (pitch) only
 *
 * Flash: bash test/flash_front_sweep_45.sh
 * Log+plot: python test/log_front_sweep_45.py [--motor both|m1|m2]
 */

#include <Encoder.h>
#include <math.h>
#include <string.h>

const float M1GEAR = 9.68;
const float M2GEAR = 34.0;
const int TICKS_PER_REV = 48;

// Pins match hardware/PD_control_front/PD_control_front.ino
const int M1INA = 2, M1INB = 4, M1PWM = 9, M1EN = 6;
Encoder enc1(14, 15);

const int M2INA = 7, M2INB = 8, M2PWM = 12, M2EN = 10;
Encoder enc2(17, 16);

bool MOTOR1_REVERSED = true;
bool ENCODER1_REVERSED = true;
bool MOTOR2_REVERSED = false;
bool ENCODER2_REVERSED = true;

double Kp1 = 2048.0;
double Kd1 = 204.8;
double Kp2 = 20480.0;
double Kd2 = 204.8;

float deadband = 0.03;
const int minPWM = 100;
const int PWM_MAX = 1023;

const float CONTROL_HZ = 1000.0f;
const uint32_t CONTROL_PERIOD_US = 1000000UL / (uint32_t)CONTROL_HZ;
const uint32_t LOG_PERIOD_MS = 10;  // 100 Hz encoder log

const float SWEEP_RAD = PI / 4.0f;  // 45 degrees
const uint32_t SETTLE_MS = 500;
const uint32_t MOVE_TIMEOUT_MS = 1000;
const uint32_t PAUSE_BETWEEN_MS = 500;

float targetPos1 = 0;
float targetPos2 = 0;
float lastErr1 = 0;
float lastErr2 = 0;
uint32_t lastControlMicros = 0;
uint32_t lastLogMs = 0;
uint32_t t0Ms = 0;
int activeMotor = 0;  // 0=none, 1=M1, 2=M2
int pendingMask = 0;  // bit0=M1, bit1=M2; set by pollRunCommand

long rawEncoder1() {
  long pos = enc1.read();
  return ENCODER1_REVERSED ? -pos : pos;
}

long rawEncoder2() {
  long pos = enc2.read();
  return ENCODER2_REVERSED ? -pos : pos;
}

float readEncoder1() {
  return (float)rawEncoder1() / (TICKS_PER_REV * M1GEAR) * 2.0f * (float)PI;
}

float readEncoder2() {
  return (float)rawEncoder2() / (TICKS_PER_REV * M2GEAR) * 2.0f * (float)PI;
}

void stopMotor1() {
  digitalWrite(M1INA, LOW);
  digitalWrite(M1INB, LOW);
  analogWrite(M1PWM, 0);
}

void stopMotor2() {
  digitalWrite(M2INA, LOW);
  digitalWrite(M2INB, LOW);
  analogWrite(M2PWM, 0);
}

void driveMotor1(double speed) {
  int pwmVal = constrain((int)abs(speed), 0, PWM_MAX);
  if (pwmVal > 0 && pwmVal < minPWM) pwmVal = minPWM;
  if (pwmVal == 0) {
    stopMotor1();
    return;
  }
  if (speed > 0) {
    digitalWrite(M1INA, HIGH);
    digitalWrite(M1INB, LOW);
  } else {
    digitalWrite(M1INA, LOW);
    digitalWrite(M1INB, HIGH);
  }
  analogWrite(M1PWM, pwmVal);
}

void driveMotor2(double speed) {
  int pwmVal = constrain((int)abs(speed), 0, PWM_MAX);
  if (pwmVal > 0 && pwmVal < minPWM) pwmVal = minPWM;
  if (pwmVal == 0) {
    stopMotor2();
    return;
  }
  if (speed > 0) {
    digitalWrite(M2INA, HIGH);
    digitalWrite(M2INB, LOW);
  } else {
    digitalWrite(M2INA, LOW);
    digitalWrite(M2INB, HIGH);
  }
  analogWrite(M2PWM, pwmVal);
}

void logSample() {
  uint32_t now = millis();
  if ((uint32_t)(now - lastLogMs) < LOG_PERIOD_MS) {
    return;
  }
  lastLogMs = now;
  Serial.print(now - t0Ms);
  Serial.print(',');
  Serial.print(readEncoder1(), 5);
  Serial.print(',');
  Serial.print(readEncoder2(), 5);
  Serial.print(',');
  Serial.print(targetPos1, 5);
  Serial.print(',');
  Serial.print(targetPos2, 5);
  Serial.print(',');
  Serial.println(activeMotor);
}

void runController(double dt) {
  if (dt <= 0.0) dt = 0.001;

  float err1 = targetPos1 - readEncoder1();
  if (fabsf(err1) <= deadband) {
    stopMotor1();
  } else {
    double cmd1 = (Kp1 * err1) + (Kd1 * (err1 - lastErr1) / dt);
    if (MOTOR1_REVERSED) cmd1 = -cmd1;
    driveMotor1(cmd1);
  }

  float err2 = targetPos2 - readEncoder2();
  if (fabsf(err2) <= deadband) {
    stopMotor2();
  } else {
    double cmd2 = (Kp2 * err2) + (Kd2 * (err2 - lastErr2) / dt);
    if (MOTOR2_REVERSED) cmd2 = -cmd2;
    driveMotor2(cmd2);
  }

  lastErr1 = err1;
  lastErr2 = err2;
}

void controlAndLogOnce() {
  uint32_t nowMicros = micros();
  if ((uint32_t)(nowMicros - lastControlMicros) >= CONTROL_PERIOD_US) {
    double dt = (nowMicros - lastControlMicros) / 1000000.0;
    lastControlMicros = nowMicros;
    runController(dt);
  }
  logSample();
}

void runForMs(uint32_t durationMs) {
  uint32_t start = millis();
  while ((uint32_t)(millis() - start) < durationMs) {
    controlAndLogOnce();
  }
}

bool moveAndWait(int motor, float target, uint32_t timeoutMs) {
  if (motor == 1) {
    targetPos1 = target;
  } else {
    targetPos2 = target;
  }
  activeMotor = motor;

  uint32_t start = millis();
  while ((uint32_t)(millis() - start) < timeoutMs) {
    controlAndLogOnce();
    float pos = (motor == 1) ? readEncoder1() : readEncoder2();
    if (fabsf(target - pos) <= deadband) {
      return true;
    }
  }
  return false;
}

void sweepMotor(int motor, const char *name) {
  float home1 = readEncoder1();
  float home2 = readEncoder2();
  targetPos1 = home1;
  targetPos2 = home2;
  lastErr1 = 0;
  lastErr2 = 0;
  activeMotor = motor;

  float home = (motor == 1) ? home1 : home2;
  float cw = home - SWEEP_RAD;
  float ccw_home = home;

  Serial.print("# EVENT start_cw motor=");
  Serial.println(name);
  if (!moveAndWait(motor, cw, MOVE_TIMEOUT_MS)) {
    Serial.print("# EVENT timeout_cw motor=");
    Serial.println(name);
  } else {
    Serial.print("# EVENT reached_cw motor=");
    Serial.println(name);
  }
  runForMs(SETTLE_MS);

  Serial.print("# EVENT start_ccw_home motor=");
  Serial.println(name);
  if (!moveAndWait(motor, ccw_home, MOVE_TIMEOUT_MS)) {
    Serial.print("# EVENT timeout_home motor=");
    Serial.println(name);
  } else {
    Serial.print("# EVENT reached_home motor=");
    Serial.println(name);
  }
  runForMs(SETTLE_MS);
}

void runSweepSequence(int mask) {
  enc1.write(0);
  enc2.write(0);
  targetPos1 = 0;
  targetPos2 = 0;
  lastErr1 = 0;
  lastErr2 = 0;
  t0Ms = millis();
  lastLogMs = 0;
  lastControlMicros = micros();
  activeMotor = 0;

  bool doM1 = (mask & 0x1) != 0;
  bool doM2 = (mask & 0x2) != 0;

  Serial.print("# EVENT mask=");
  Serial.println(mask);

  digitalWrite(M1EN, doM1 ? HIGH : LOW);
  digitalWrite(M2EN, doM2 ? HIGH : LOW);
  Serial.println("# EVENT drivers_on");

  if (doM1) {
    Serial.println("# EVENT sweep_m1_roll");
    sweepMotor(1, "M1_roll");
    activeMotor = 0;
    if (doM2) {
      runForMs(PAUSE_BETWEEN_MS);
    }
  }

  if (doM2) {
    Serial.println("# EVENT sweep_m2_pitch");
    sweepMotor(2, "M2_pitch");
  }

  stopMotor1();
  stopMotor2();
  digitalWrite(M1EN, LOW);
  digitalWrite(M2EN, LOW);
  activeMotor = 0;
  Serial.println("# EVENT done");
  Serial.println("# done");
}

// Returns true and sets pendingMask when a RUN line is complete.
bool pollRunCommand() {
  static char lineBuf[40];
  static size_t lineLen = 0;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen == 0) continue;
      lineBuf[lineLen] = '\0';
      lineLen = 0;

      // Accept: RUN | RUN BOTH | RUN M1 | RUN M2
      if (strcmp(lineBuf, "RUN") == 0 || strcmp(lineBuf, "RUN BOTH") == 0) {
        pendingMask = 0x3;
        return true;
      }
      if (strcmp(lineBuf, "RUN M1") == 0) {
        pendingMask = 0x1;
        return true;
      }
      if (strcmp(lineBuf, "RUN M2") == 0) {
        pendingMask = 0x2;
        return true;
      }
      continue;
    }
    if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;
    }
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
  }

  Serial.println("# FIRMWARE front_sweep_45");
  Serial.println("# M1=roll M2=pitch EXPECT_SN=18451300");
  Serial.println("# MOVE_TIMEOUT_MS=1000");
  Serial.println("# Send RUN | RUN M1 | RUN M2 | RUN BOTH");
  Serial.println("t_ms,m1_rad,m2_rad,tgt1,tgt2,active_motor");

  pinMode(M1INA, OUTPUT);
  pinMode(M1INB, OUTPUT);
  pinMode(M1PWM, OUTPUT);
  pinMode(M1EN, OUTPUT);
  pinMode(M2INA, OUTPUT);
  pinMode(M2INB, OUTPUT);
  pinMode(M2PWM, OUTPUT);
  pinMode(M2EN, OUTPUT);

  analogWriteFrequency(M1PWM, 20000);
  analogWriteFrequency(M2PWM, 20000);
  analogWriteRes(10);
  stopMotor1();
  stopMotor2();
  digitalWrite(M1EN, LOW);
  digitalWrite(M2EN, LOW);

  enc1.write(0);
  enc2.write(0);
  targetPos1 = 0;
  targetPos2 = 0;
  lastControlMicros = micros();
  t0Ms = millis();
  lastLogMs = 0;

  Serial.println("# ready");
}

void loop() {
  if (pollRunCommand()) {
    Serial.print("# EVENT run_received mask=");
    Serial.println(pendingMask);
    runSweepSequence(pendingMask);
    Serial.println("# ready");
  }
}
