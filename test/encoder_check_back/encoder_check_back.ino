/*
 * Back Teensy — encoder readout check (motors coast, not driven).
 *
 * Streams: m1_ticks,m2_ticks,m1_rad,m2_rad,m1A,m1B,m2A,m2B
 * Pinout / reverse flags match hardware/PD_control_back.
 * M2GEAR = 34. Expected USB SN_BACK = 18452630.
 *
 * Flash: bash test/flash_encoder_check_back.sh
 * Watch: python test/monitor_teensy.py --back
 * Turn BACK joints by hand; ticks and pin bits should change.
 */

#include <Encoder.h>
#include <math.h>

const float M1GEAR = 9.68;
const float M2GEAR = 34.0;
const int TICKS_PER_REV = 48;
const uint32_t PRINT_MS = 50;

const int M1INA = 2, M1INB = 4, M1PWM = 6, M1EN = 9;
const int M2INA = 7, M2INB = 8, M2PWM = 10, M2EN = 12;
const int ENC1_A = 13, ENC1_B = 14;
const int ENC2_A = 15, ENC2_B = 16;

Encoder enc1(ENC1_A, ENC1_B);
Encoder enc2(ENC2_A, ENC2_B);

bool ENCODER1_REVERSED = false;
bool ENCODER2_REVERSED = true;

uint32_t lastPrintMs = 0;

long raw1() {
  long p = enc1.read();
  return ENCODER1_REVERSED ? -p : p;
}

long raw2() {
  long p = enc2.read();
  return ENCODER2_REVERSED ? -p : p;
}

float rad1() {
  return (float)raw1() / (TICKS_PER_REV * M1GEAR) * 2.0f * (float)PI;
}

float rad2() {
  return (float)raw2() / (TICKS_PER_REV * M2GEAR) * 2.0f * (float)PI;
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
  }

  Serial.println("========================================");
  Serial.println("FIRMWARE: encoder_check_BACK");
  Serial.println("PINOUT:   PD_control_back enc1(13,14) enc2(15,16)");
  Serial.println("EXPECT_SN:18452630");
  Serial.println("NOTE:     m1 = tail, m2 = back roll (controllerV2 mapping)");
  Serial.println("========================================");
  Serial.println("m1_ticks,m2_ticks,m1_rad,m2_rad,m1A,m1B,m2A,m2B");

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
  digitalWrite(M1EN, HIGH);
  digitalWrite(M2EN, HIGH);

  enc1.write(0);
  enc2.write(0);
  lastPrintMs = millis();
}

void loop() {
  uint32_t now = millis();
  if ((uint32_t)(now - lastPrintMs) < PRINT_MS) {
    return;
  }
  lastPrintMs = now;

  Serial.print(raw1());
  Serial.print(',');
  Serial.print(raw2());
  Serial.print(',');
  Serial.print(rad1(), 4);
  Serial.print(',');
  Serial.print(rad2(), 4);
  Serial.print(',');
  Serial.print(digitalRead(ENC1_A));
  Serial.print(',');
  Serial.print(digitalRead(ENC1_B));
  Serial.print(',');
  Serial.print(digitalRead(ENC2_A));
  Serial.print(',');
  Serial.println(digitalRead(ENC2_B));
}
