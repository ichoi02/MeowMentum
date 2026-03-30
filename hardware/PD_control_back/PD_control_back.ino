#include <Encoder.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>

const float M1GEAR = 9.68;
const float M2GEAR = 9.68;
const int TICKS_PER_REV = 48;

// Motor 1
const int M1INA = 2;
const int M1INB = 4;
const int M1PWM = 6;
const int M1EN = 9;
Encoder enc1(13, 14);

// Motor 2
const int M2INA = 7;
const int M2INB = 8;
const int M2PWM = 10;
const int M2EN = 12;
Encoder enc2(15, 16);

Adafruit_BNO08x bno08x;
sh2_SensorValue_t sensorValue;

// ==========================================
// 2. HARDWARE CALIBRATION
// ==========================================
// Flip these if needed after testing.
//
// MOTORx_REVERSED flips the commanded motor direction.
// ENCx_REVERSED flips the encoder sign.
//
// If a motor runs away even though the encoder count changes,
// one of these is likely wrong.

bool MOTOR1_REVERSED = true;
bool ENCODER1_REVERSED = true;

bool MOTOR2_REVERSED = true;
bool ENCODER2_REVERSED = false;

// ==========================================
// 3. CONTROL SETTINGS
// ==========================================

// PD gains
double Kp = 1000.0;
double Kd = 10.0;

// If the error is within this many ticks, motor stops
float deadband = 0.03;

// Minimum PWM to overcome static friction
const int minPWM = 100;

// PWM range for 10-bit resolution
const int PWM_MAX = 1023;

// Fixed control frequency
const float CONTROL_HZ = 1000.0f;               // 1 kHz
const uint32_t CONTROL_PERIOD_US = 1000000UL / CONTROL_HZ;

// Telemetry frequency
const uint32_t PRINT_PERIOD_MS = 50;

// ==========================================
// 4. STATE VARIABLES
// ==========================================

float targetPos1 = 0;
float targetPos2 = 0;
float lastErr1 = 0;
float lastErr2 = 0;

uint32_t lastControlMicros = 0;
uint32_t lastPrintMillis = 0;

// ==========================================
// 5. SETUP
// ==========================================

void setup() {
  Serial.begin(115200);

  // Motor 1 pins
  pinMode(M1INA, OUTPUT);
  pinMode(M1INB, OUTPUT);
  pinMode(M1PWM, OUTPUT);
  pinMode(M1EN, OUTPUT);

  // Motor 2 pins
  pinMode(M2INA, OUTPUT);
  pinMode(M2INB, OUTPUT);
  pinMode(M2PWM, OUTPUT);
  pinMode(M2EN, OUTPUT);

  // Enable drivers
  digitalWrite(M1EN, HIGH);
  digitalWrite(M2EN, HIGH);

  // Teensy 4.0 PWM setup
  analogWriteFrequency(M1PWM, 20000);
  analogWriteFrequency(M2PWM, 20000);
  analogWriteRes(10);

  // Start with motors off
  stopMotor1();
  stopMotor2();

  // Initialize target to current Motor 1 position
  // so the system does not jump on startup.
  targetPos1 = readEncoder1();
  targetPos2 = readEncoder2();

  lastControlMicros = micros();
  lastPrintMillis = millis();

  // IMU Setup
  Wire.begin();
  Wire.setClock(400000);
  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    while (1) { delay(100); }
  }
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 10000)) {
    while (1) { delay(100); }
  }
}

// ==========================================
// 6. MAIN LOOP
// ==========================================

void loop() {
  handleSerialInput();

  uint32_t nowMicros = micros();

  // Fixed-rate control loop
  if ((uint32_t)(nowMicros - lastControlMicros) >= CONTROL_PERIOD_US) {
    double dt = (nowMicros - lastControlMicros) / 1000000.0;
    lastControlMicros = nowMicros;

    runController(dt);
  }

  // Fixed-rate telemetry
  uint32_t nowMillis = millis();
  if ((uint32_t)(nowMillis - lastPrintMillis) >= PRINT_PERIOD_MS) {
    lastPrintMillis = nowMillis;
    
    if (bno08x.getSensorEvent(&sensorValue)) {
      if (sensorValue.sensorId == SH2_ROTATION_VECTOR) {
        float qr = sensorValue.un.rotationVector.real;
        float qi = sensorValue.un.rotationVector.i;
        float qj = sensorValue.un.rotationVector.j;
        float qk = sensorValue.un.rotationVector.k;
        //float acc = sensorValue.un.rotationVector.accuracy;

        float angle1 = readEncoder1();
        float angle2 = readEncoder2();

        Serial.print(qr, 6); Serial.print(",");
        Serial.print(qi, 6); Serial.print(",");
        Serial.print(qj, 6); Serial.print(",");
        Serial.print(qk, 6); Serial.print(",");
        //Serial.print(acc, 6); Serial.print(",");
        Serial.print(angle1, 4); Serial.print(",");
        Serial.println(angle2, 4);
      }
    }
  }
}

// ==========================================
// 7. CONTROL LOOP
// ==========================================

void runController(double dt) {
  if (dt <= 0.0) {
    dt = 0.001; // fallback
  }

  float currentPos1 = readEncoder1();
  float currentPos2 = readEncoder2();

  // ----- Motor 1 -----
  float err1 = targetPos1 - currentPos1;

  if (abs(err1) <= deadband) {
    stopMotor1();
  } else {
    double deriv1 = (err1 - lastErr1) / dt;
    double cmd1 = (Kp * err1) + (Kd * deriv1);

    if (MOTOR1_REVERSED) {
      cmd1 = -cmd1;
    }
    
    driveMotor1(cmd1);
  }

  // ----- Motor 2 -----
  float err2 = targetPos2 - currentPos2;

  if (abs(err2) <= deadband) {
    stopMotor2();
  } else {
    double deriv2 = (err2 - lastErr2) / dt;
    double cmd2 = (Kp * err2) + (Kd * deriv2);

    if (MOTOR2_REVERSED) {
      cmd2 = -cmd2;
    }

    driveMotor2(cmd2);
  }

  lastErr1 = err1;
  lastErr2 = err2;
}

// ==========================================
// 8. SERIAL INPUT
// ==========================================

void handleSerialInput() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input == "RESET") {
      enc1.write(0);
      enc2.write(0);
    } 
    if (input == "START") { // ONLY USE AFTER RESET
      digitalWrite(M1EN, HIGH);
      digitalWrite(M2EN, HIGH);
    }
    if (input == "STOP") {
      digitalWrite(M1EN, LOW);
      digitalWrite(M2EN, LOW);
    }
    else {
      int commaIndex = input.indexOf(',');
      if (commaIndex > 0) {
        targetPos1 = input.substring(0, commaIndex).toFloat();
        targetPos2 = input.substring(commaIndex + 1).toFloat();
      }
    }
  }
}

// ==========================================
// 9. TELEMETRY
// ==========================================


float readEncoder1() {
  long pos = enc1.read();
  if (ENCODER1_REVERSED) {
    pos = -pos;
  }
  float angle = (float)pos/TICKS_PER_REV/M1GEAR*2*PI;
  return angle;
}

float readEncoder2() {
  long pos = enc2.read();
  if (ENCODER2_REVERSED) {
    pos = -pos;
  }
  float angle = (float)pos/TICKS_PER_REV/M1GEAR*2*PI;
  return angle;
}

// ==========================================
// 11. MOTOR DRIVE FUNCTIONS
// ==========================================

void driveMotor1(double speed) {
  int pwmVal = constrain((int)abs(speed), 0, PWM_MAX);

  if (pwmVal > 0 && pwmVal < minPWM) {
    pwmVal = minPWM;
  }

  if (speed > 0) {
    digitalWrite(M1INA, HIGH);
    digitalWrite(M1INB, LOW);
  } else {
    digitalWrite(M1INA, LOW);
    digitalWrite(M1INB, HIGH);
  }
  // Serial.println(pwmVal);
  analogWrite(M1PWM, pwmVal);
}

void stopMotor1() {
  digitalWrite(M1INA, LOW);
  digitalWrite(M1INB, LOW);
  analogWrite(M1PWM, 0);
}

void driveMotor2(double speed) {
  int pwmVal = constrain((int)abs(speed), 0, PWM_MAX);

  if (pwmVal > 0 && pwmVal < minPWM) {
    pwmVal = minPWM;
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