/*
 * Front Teensy motor smoke test.
 *
 * DEBUG_ENCODER_ONLY 1  -> only print encoder readings (motors never enabled)
 * DEBUG_ENCODER_ONLY 0  -> sweep each motor +0.5 rad CCW then back CW
 *
 * Flash to the front board. Watch with: python test/monitor_teensy.py --front
 */

#include <Encoder.h>
#include <math.h>
#include <cstring>

// 1 = encoder readout only; 0 = motor sweep test
#define DEBUG_ENCODER_ONLY 1

const uint32_t ENCODER_PRINT_MS = 50;

const float M1GEAR = 9.68;
const float M2GEAR = 34.0;
const int TICKS_PER_REV = 48;

// Motor 1
const int M1INA = 2;
const int M1INB = 4;
const int M1PWM = 9;
const int M1EN = 6;
Encoder enc1(14, 15);

// Motor 2
const int M2INA = 7;
const int M2INB = 8;
const int M2PWM = 12;
const int M2EN = 10;
Encoder enc2(17, 16);

bool MOTOR1_REVERSED = true;
bool ENCODER1_REVERSED = true;
bool MOTOR2_REVERSED = false;
bool ENCODER2_REVERSED = true;

#if !DEBUG_ENCODER_ONLY
double Kp1 = 2048.0;
double Kd1 = 204.8;
double Kp2 = 20480.0;
double Kd2 = 204.8;

float deadband = 0.03;
const int minPWM = 100;
const int PWM_MAX = 1023;

const float CONTROL_HZ = 1000.0f;
const uint32_t CONTROL_PERIOD_US = 1000000UL / (uint32_t)CONTROL_HZ;

const float SWEEP_RAD = 0.5f;
const uint32_t SETTLE_MS = 400;
const uint32_t MOVE_TIMEOUT_MS = 5000;
const uint32_t PAUSE_BETWEEN_MOTORS_MS = 1000;
const uint32_t LOOP_PAUSE_MS = 3000;

float targetPos1 = 0;
float targetPos2 = 0;
float lastErr1 = 0;
float lastErr2 = 0;
uint32_t lastControlMicros = 0;
#endif

uint32_t lastEncoderPrintMs = 0;

long rawEncoder1() {
  long pos = enc1.read();
  if (ENCODER1_REVERSED) pos = -pos;
  return pos;
}

long rawEncoder2() {
  long pos = enc2.read();
  if (ENCODER2_REVERSED) pos = -pos;
  return pos;
}

float readEncoder1() {
  return (float)rawEncoder1() / (TICKS_PER_REV * M1GEAR) * 2.0f * (float)PI;
}

float readEncoder2() {
  return (float)rawEncoder2() / (TICKS_PER_REV * M2GEAR) * 2.0f * (float)PI;
}

#if !DEBUG_ENCODER_ONLY
void driveMotor1(double speed);
void driveMotor2(double speed);
void stopMotor1();
void stopMotor2();
void runController(double dt);
bool moveAndWait(int motor, float target, uint32_t timeoutMs);
void enableMotors(bool on);
void sweepMotor(int motor);
#endif

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
  }

  // Always set up driver pins. In debug we enable EN but hold motors in coast
  // so encoder supply / signal paths stay live while shafts can be turned by hand.
  pinMode(M1INA, OUTPUT);
  pinMode(M1INB, OUTPUT);
  pinMode(M1PWM, OUTPUT);
  pinMode(M1EN, OUTPUT);
  pinMode(M2INA, OUTPUT);
  pinMode(M2INB, OUTPUT);
  pinMode(M2PWM, OUTPUT);
  pinMode(M2EN, OUTPUT);

  digitalWrite(M1INA, LOW);
  digitalWrite(M1INB, LOW);
  digitalWrite(M2INA, LOW);
  digitalWrite(M2INB, LOW);
  analogWriteFrequency(M1PWM, 20000);
  analogWriteFrequency(M2PWM, 20000);
  analogWriteRes(10);
  analogWrite(M1PWM, 0);
  analogWrite(M2PWM, 0);

  enc1.write(0);
  enc2.write(0);
  lastEncoderPrintMs = millis();

#if DEBUG_ENCODER_ONLY
  digitalWrite(M1EN, HIGH);
  digitalWrite(M2EN, HIGH);
  Serial.println("front DEBUG: encoder readout only (drivers EN=HIGH, motors coast)");
  Serial.println("m1_ticks,m2_ticks,m1_rad,m2_rad,m1A,m1B,m2A,m2B");
  Serial.println("If pins never change when turning by hand: check motor battery power + you are on FRONT joints.");
  Serial.println("Running 400ms strong nudge on M1 then M2 (watch FRONT joints move)...");

  // Brief open-loop nudge — if ticks stay 0, encoder signals are not reaching these pins.
  auto nudge = [](int ina, int inb, int pwm, Encoder &enc, const char *name) {
    long before = enc.read();
    digitalWrite(ina, HIGH);
    digitalWrite(inb, LOW);
    analogWrite(pwm, 600);  // 10-bit scale (0-1023); strong enough to overcome stiction
    delay(400);
    analogWrite(pwm, 0);
    digitalWrite(ina, LOW);
    digitalWrite(inb, LOW);
    delay(50);
    long after = enc.read();
    Serial.print(name);
    Serial.print(" nudge ticks: ");
    Serial.print(before);
    Serial.print(" -> ");
    Serial.println(after);
  };
  nudge(M1INA, M1INB, M1PWM, enc1, "M1");
  delay(300);
  nudge(M2INA, M2INB, M2PWM, enc2, "M2");
  enc1.write(0);
  enc2.write(0);
  Serial.println("Hand-turn test follows (pins should flip):");
#else
  digitalWrite(M1EN, LOW);
  digitalWrite(M2EN, LOW);
  targetPos1 = 0;
  targetPos2 = 0;
  lastControlMicros = micros();

  Serial.println("motor_test_front: starting in 2s...");
  delay(2000);
  digitalWrite(M1EN, HIGH);
  digitalWrite(M2EN, HIGH);
#endif
}

void loop() {
#if DEBUG_ENCODER_ONLY
  uint32_t now = millis();
  if ((uint32_t)(now - lastEncoderPrintMs) < ENCODER_PRINT_MS) {
    return;
  }
  lastEncoderPrintMs = now;

  // ticks + rad + raw channel levels (should flip 0/1 when you turn by hand)
  Serial.print(rawEncoder1());
  Serial.print(',');
  Serial.print(rawEncoder2());
  Serial.print(',');
  Serial.print(readEncoder1(), 4);
  Serial.print(',');
  Serial.print(readEncoder2(), 4);
  Serial.print(',');
  Serial.print(digitalRead(14));
  Serial.print(',');
  Serial.print(digitalRead(15));
  Serial.print(',');
  Serial.print(digitalRead(17));
  Serial.print(',');
  Serial.println(digitalRead(16));
#else
  Serial.println("--- Front motor sweep ---");

  Serial.println("Motor 1: +0.5 rad CCW, then 0.5 rad CW");
  sweepMotor(1);
  delay(PAUSE_BETWEEN_MOTORS_MS);

  Serial.println("Motor 2: +0.5 rad CCW, then 0.5 rad CW");
  sweepMotor(2);

  Serial.println("--- Sweep done ---");
  delay(LOOP_PAUSE_MS);
#endif
}

#if !DEBUG_ENCODER_ONLY
void sweepMotor(int motor) {
  float home1 = readEncoder1();
  float home2 = readEncoder2();
  targetPos1 = home1;
  targetPos2 = home2;
  lastErr1 = 0;
  lastErr2 = 0;

  float home = (motor == 1) ? home1 : home2;
  float ccw = home + SWEEP_RAD;
  float cw = home;

  if (!moveAndWait(motor, ccw, MOVE_TIMEOUT_MS)) {
    Serial.print("Motor ");
    Serial.print(motor);
    Serial.println(": timeout reaching CCW target");
  } else {
    Serial.print("Motor ");
    Serial.print(motor);
    Serial.println(": at +0.5 rad");
  }
  delay(SETTLE_MS);

  if (!moveAndWait(motor, cw, MOVE_TIMEOUT_MS)) {
    Serial.print("Motor ");
    Serial.print(motor);
    Serial.println(": timeout reaching CW (home) target");
  } else {
    Serial.print("Motor ");
    Serial.print(motor);
    Serial.println(": back at home");
  }
  delay(SETTLE_MS);
}

bool moveAndWait(int motor, float target, uint32_t timeoutMs) {
  if (motor == 1) {
    targetPos1 = target;
  } else {
    targetPos2 = target;
  }

  uint32_t start = millis();
  while ((uint32_t)(millis() - start) < timeoutMs) {
    uint32_t nowMicros = micros();
    if ((uint32_t)(nowMicros - lastControlMicros) >= CONTROL_PERIOD_US) {
      double dt = (nowMicros - lastControlMicros) / 1000000.0;
      lastControlMicros = nowMicros;
      runController(dt);
    }

    float pos = (motor == 1) ? readEncoder1() : readEncoder2();
    if (fabsf(target - pos) <= deadband) {
      return true;
    }
  }
  return false;
}

void runController(double dt) {
  if (dt <= 0.0) {
    dt = 0.001;
  }

  float currentPos1 = readEncoder1();
  float currentPos2 = readEncoder2();

  float err1 = targetPos1 - currentPos1;
  if (fabsf(err1) <= deadband) {
    stopMotor1();
  } else {
    double deriv1 = (err1 - lastErr1) / dt;
    double cmd1 = (Kp1 * err1) + (Kd1 * deriv1);
    if (MOTOR1_REVERSED) cmd1 = -cmd1;
    driveMotor1(cmd1);
  }

  float err2 = targetPos2 - currentPos2;
  if (fabsf(err2) <= deadband) {
    stopMotor2();
  } else {
    double deriv2 = (err2 - lastErr2) / dt;
    double cmd2 = (Kp2 * err2) + (Kd2 * deriv2);
    if (MOTOR2_REVERSED) cmd2 = -cmd2;
    driveMotor2(cmd2);
  }

  lastErr1 = err1;
  lastErr2 = err2;
}

void enableMotors(bool on) {
  digitalWrite(M1EN, on ? HIGH : LOW);
  digitalWrite(M2EN, on ? HIGH : LOW);
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

void stopMotor1() {
  digitalWrite(M1INA, LOW);
  digitalWrite(M1INB, LOW);
  analogWrite(M1PWM, 0);
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

void stopMotor2() {
  digitalWrite(M2INA, LOW);
  digitalWrite(M2INB, LOW);
  analogWrite(M2PWM, 0);
}
#endif
