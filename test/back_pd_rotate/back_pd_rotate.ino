/*
 * Back Teensy — PD nudge either motor; print encoders in rad.
 * M1 = tail, M2 = back roll (hardware/PD_control_back).
 *
 * Each move runs PD for at most MOVE_TIMEOUT_MS (200 ms), then stops.
 *
 * Serial (line-oriented):
 *   M1 <delta_rad>   — relative move on M1 (tail)
 *   M2 <delta_rad>   — relative move on M2 (back roll)
 *   ZERO             — zero both encoders / targets
 *   STOP             — stop motors, disable drivers
 *
 * Streams CSV while idle and during moves:
 *   t_ms,m1_rad,m2_rad,tgt1,tgt2,active_motor
 * Event lines start with '#'.
 *
 * Flash: bash test/flash_back_pd_rotate.sh
 */

#include <Encoder.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

const float M1GEAR = 34.0;
const float M2GEAR = 9.68;
const int TICKS_PER_REV = 48;

const int M1INA = 2, M1INB = 4, M1PWM = 6, M1EN = 9;
Encoder enc1(13, 14);

const int M2INA = 7, M2INB = 8, M2PWM = 10, M2EN = 12;
Encoder enc2(15, 16);

// Match hardware/PD_control_back
bool MOTOR1_REVERSED = false;
bool ENCODER1_REVERSED = false;
bool MOTOR2_REVERSED = true;
bool ENCODER2_REVERSED = true;

double Kp1 = 1024.0;
double Kd1 = 102.4;
double Kp2 = 3072.0;
double Kd2 = 307.2;

float deadband = 0.03;
const int minPWM = 100;
const int PWM_MAX = 1023;

const float CONTROL_HZ = 1000.0f;
const uint32_t CONTROL_PERIOD_US = 1000000UL / (uint32_t)CONTROL_HZ;
const uint32_t LOG_PERIOD_MS = 20;  // 50 Hz
const uint32_t MOVE_TIMEOUT_MS = 1000;

float targetPos1 = 0;
float targetPos2 = 0;
float lastErr1 = 0;
float lastErr2 = 0;
uint32_t lastControlMicros = 0;
uint32_t lastLogMs = 0;
uint32_t t0Ms = 0;
int activeMotor = 0;  // 0=idle, 1=M1, 2=M2
uint32_t moveStartMs = 0;
bool driversOn = false;

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

void setDrivers(bool m1, bool m2) {
  digitalWrite(M1EN, m1 ? HIGH : LOW);
  digitalWrite(M2EN, m2 ? HIGH : LOW);
  driversOn = m1 || m2;
}

void stopAll() {
  stopMotor1();
  stopMotor2();
  setDrivers(false, false);
  activeMotor = 0;
  targetPos1 = readEncoder1();
  targetPos2 = readEncoder2();
  lastErr1 = 0;
  lastErr2 = 0;
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

  if (activeMotor == 1) {
    float err1 = targetPos1 - readEncoder1();
    if (fabsf(err1) <= deadband) {
      stopMotor1();
    } else {
      double cmd1 = (Kp1 * err1) + (Kd1 * (err1 - lastErr1) / dt);
      if (MOTOR1_REVERSED) cmd1 = -cmd1;
      driveMotor1(cmd1);
    }
    lastErr1 = err1;
  } else {
    stopMotor1();
  }

  if (activeMotor == 2) {
    float err2 = targetPos2 - readEncoder2();
    if (fabsf(err2) <= deadband) {
      stopMotor2();
    } else {
      double cmd2 = (Kp2 * err2) + (Kd2 * (err2 - lastErr2) / dt);
      if (MOTOR2_REVERSED) cmd2 = -cmd2;
      driveMotor2(cmd2);
    }
    lastErr2 = err2;
  } else {
    stopMotor2();
  }
}

void finishMove(const char *reason) {
  Serial.print("# EVENT ");
  Serial.print(reason);
  Serial.print(" motor=");
  Serial.print(activeMotor);
  Serial.print(" m1=");
  Serial.print(readEncoder1(), 5);
  Serial.print(" m2=");
  Serial.println(readEncoder2(), 5);
  stopAll();
}

void startRelativeMove(int motor, float deltaRad) {
  stopMotor1();
  stopMotor2();
  lastErr1 = 0;
  lastErr2 = 0;

  float p1 = readEncoder1();
  float p2 = readEncoder2();
  targetPos1 = p1;
  targetPos2 = p2;

  if (motor == 1) {
    targetPos1 = p1 + deltaRad;
    setDrivers(true, false);
  } else {
    targetPos2 = p2 + deltaRad;
    setDrivers(false, true);
  }

  activeMotor = motor;
  moveStartMs = millis();
  lastControlMicros = micros();

  Serial.print("# EVENT start motor=");
  Serial.print(motor);
  Serial.print(" delta=");
  Serial.print(deltaRad, 5);
  Serial.print(" tgt=");
  Serial.println(motor == 1 ? targetPos1 : targetPos2, 5);
}

void handleMoveTimeoutAndGoal() {
  if (activeMotor == 0) {
    return;
  }

  float pos = (activeMotor == 1) ? readEncoder1() : readEncoder2();
  float tgt = (activeMotor == 1) ? targetPos1 : targetPos2;

  if (fabsf(tgt - pos) <= deadband) {
    finishMove("reached");
    return;
  }

  if ((uint32_t)(millis() - moveStartMs) >= MOVE_TIMEOUT_MS) {
    finishMove("timeout");
  }
}

void parseCommand(char *line) {
  // Trim leading spaces
  while (*line == ' ' || *line == '\t') {
    line++;
  }
  if (*line == '\0') {
    return;
  }

  if (strcmp(line, "STOP") == 0) {
    Serial.println("# EVENT stop");
    stopAll();
    return;
  }

  if (strcmp(line, "ZERO") == 0) {
    stopAll();
    enc1.write(0);
    enc2.write(0);
    targetPos1 = 0;
    targetPos2 = 0;
    Serial.println("# EVENT zero");
    return;
  }

  // M1 <float> | M2 <float>
  int motor = 0;
  char *arg = nullptr;
  if (strncmp(line, "M1 ", 3) == 0) {
    motor = 1;
    arg = line + 3;
  } else if (strncmp(line, "M2 ", 3) == 0) {
    motor = 2;
    arg = line + 3;
  } else {
    Serial.print("# ERR unknown cmd: ");
    Serial.println(line);
    return;
  }

  while (*arg == ' ' || *arg == '\t') {
    arg++;
  }
  char *end = nullptr;
  float delta = strtof(arg, &end);
  if (end == arg) {
    Serial.println("# ERR need float delta rad, e.g. M1 0.2");
    return;
  }

  startRelativeMove(motor, delta);
}

void pollSerial() {
  static char lineBuf[48];
  static size_t lineLen = 0;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen == 0) {
        continue;
      }
      lineBuf[lineLen] = '\0';
      lineLen = 0;
      parseCommand(lineBuf);
      continue;
    }
    if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
  }

  Serial.println("# FIRMWARE back_pd_rotate");
  Serial.println("# M1=tail M2=back_roll EXPECT_SN=18452630");
  Serial.println("# MOVE_TIMEOUT_MS=200");
  Serial.println("# cmds: M1 <rad> | M2 <rad> | ZERO | STOP");
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
  stopAll();

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
  pollSerial();

  uint32_t nowMicros = micros();
  if ((uint32_t)(nowMicros - lastControlMicros) >= CONTROL_PERIOD_US) {
    double dt = (nowMicros - lastControlMicros) / 1000000.0;
    lastControlMicros = nowMicros;
    if (activeMotor != 0) {
      runController(dt);
      handleMoveTimeoutAndGoal();
    }
  }

  logSample();
}
