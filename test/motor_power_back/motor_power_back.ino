/*
 * Back Teensy — one-shot motor power check.
 * M1 full forward 1s, stop, then M2 full forward 1s, stop. Then idle.
 *
 * Pinout = hardware/PD_control_back (NOT front — PWM/EN pins differ).
 * Expected USB serial (controllerV2): SN_BACK = 18452630
 *
 * Watch BACK joints. Requires motor battery ON.
 * Flash: bash test/flash_motor_power_back.sh
 *   -> press program on the board whose USB serial is 18452630
 */

const int M1INA = 2;
const int M1INB = 4;
const int M1PWM = 6;   // front uses 9 — if FRONT joints move, wrong board was flashed
const int M1EN = 9;    // front uses 6

const int M2INA = 7;
const int M2INB = 8;
const int M2PWM = 10;  // front uses 12
const int M2EN = 12;   // front uses 10

const int PWM_MAX = 1023;
const uint32_t RUN_MS = 100;
const uint32_t GAP_MS = 500;

void stopAll() {
  digitalWrite(M1INA, LOW);
  digitalWrite(M1INB, LOW);
  digitalWrite(M2INA, LOW);
  digitalWrite(M2INB, LOW);
  analogWrite(M1PWM, 0);
  analogWrite(M2PWM, 0);
}

void driveForward(int ina, int inb, int pwm) {
  digitalWrite(ina, HIGH);
  digitalWrite(inb, LOW);
  analogWrite(pwm, PWM_MAX * 0.5);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
  }

  // Identity: if you see this banner on SN_FRONT (18451300), the wrong Teensy was flashed.
  Serial.println("========================================");
  Serial.println("FIRMWARE: motor_power_BACK");
  Serial.println("PINOUT:   PD_control_back (M1PWM=6 M1EN=9)");
  Serial.println("EXPECT_SN:18452630");
  Serial.println("========================================");

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

  digitalWrite(M1EN, HIGH);
  digitalWrite(M2EN, HIGH);

  Serial.println("back motor_power: starting in 2s (M1 then M2, 1s each @ full PWM)");
  delay(2000);

  Serial.println("M1 ON 1s");
  driveForward(M1INA, M1INB, M1PWM);
  delay(RUN_MS);
  stopAll();
  Serial.println("M1 OFF");
  delay(GAP_MS);

  Serial.println("M2 ON 1s");
  driveForward(M2INA, M2INB, M2PWM);
  delay(RUN_MS);
  stopAll();
  Serial.println("M2 OFF");

  digitalWrite(M1EN, LOW);
  digitalWrite(M2EN, LOW);
  Serial.println("done (drivers disabled). Identity heartbeat follows; reset to re-run motors.");
}

void loop() {
  static uint32_t last = 0;
  if (millis() - last >= 1000) {
    last = millis();
    Serial.println("FIRMWARE: motor_power_BACK idle EXPECT_SN:18452630");
  }
}
