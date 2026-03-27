#include <Wire.h>
#include <Adafruit_BNO08x.h>

// -----------------------------
// BNO08x setup
// -----------------------------
Adafruit_BNO08x bno08x;

sh2_SensorValue_t sensorValue;

// -----------------------------
// Timing
// -----------------------------
const unsigned long PRINT_INTERVAL_MS = 100;
unsigned long lastPrint = 0;

// -----------------------------
// Setup
// -----------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}

  Serial.println("====================================");
  Serial.println("Teensy 4.0 + BNO08x Quaternion Test");
  Serial.println("====================================");

  Wire.begin();
  Wire.setClock(100000);

  Serial.println("Initializing BNO08x...");

  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("ERROR: Could not find BNO08x.");
    while (1) delay(100);
  }

  Serial.println("BNO08x initialized.");

  if (!bno08x.enableReport(SH2_ROTATION_VECTOR)) {
    Serial.println("Could not enable rotation vector.");
    while (1) delay(100);
  }

  Serial.println("qr,qi,qj,qk,accuracy");
}

// -----------------------------
// Loop
// -----------------------------
void loop() {
  if (bno08x.getSensorEvent(&sensorValue)) {
    if (sensorValue.sensorId == SH2_ROTATION_VECTOR) {

      float qr = sensorValue.un.rotationVector.real;
      float qi = sensorValue.un.rotationVector.i;
      float qj = sensorValue.un.rotationVector.j;
      float qk = sensorValue.un.rotationVector.k;
      float acc = sensorValue.un.rotationVector.accuracy;

      Serial.print(qr, 6); Serial.print(",");
      Serial.print(qi, 6); Serial.print(",");
      Serial.print(qj, 6); Serial.print(",");
      Serial.print(qk, 6); Serial.print(",");
      Serial.println(acc, 6);
    }
  }
}