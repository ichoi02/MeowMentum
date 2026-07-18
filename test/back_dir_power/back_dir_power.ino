/*
 * Back Teensy — open-loop FWD/REV power check (both directions).
 * Pinout = hardware/PD_control_back. EXPECT_SN=18452630
 *
 * For each motor: forward (INA high / INB low), stop, reverse (INA low / INB high).
 * 50% PWM, same as motor_power_back. Watch joint + meter on INA/INB/PWM.
 *
 * Flash: bash test/flash_back_dir_power.sh
 */

const int M1INA = 2;
const int M1INB = 4;
const int M1PWM = 6;
const int M1EN = 9;

const int M2INA = 7;
const int M2INB = 8;
const int M2PWM = 10;
const int M2EN = 12;

const int PWM_MAX = 1023;
const int PWM_DUTY = (int)(PWM_MAX * 0.5);  // ~50%
const uint32_t RUN_MS = 400;
const uint32_t GAP_MS = 600;

void stopAll() {
  digitalWrite(M1INA, LOW);
  digitalWrite(M1INB, LOW);
  digitalWrite(M2INA, LOW);
  digitalWrite(M2INB, LOW);
  analogWrite(M1PWM, 0);
  analogWrite(M2PWM, 0);
}

void drive(int ina, int inb, int pwmPin, bool forward) {
  if (forward) {
    digitalWrite(ina, HIGH);
    digitalWrite(inb, LOW);
  } else {
    digitalWrite(ina, LOW);
    digitalWrite(inb, HIGH);
  }
  analogWrite(pwmPin, PWM_DUTY);
}

void runPulse(const char *label, int en, int ina, int inb, int pwmPin, bool forward) {
  Serial.print("# EVENT ");
  Serial.print(label);
  Serial.println(forward ? " FWD (INA=1 INB=0)" : " REV (INA=0 INB=1)");

  digitalWrite(en, HIGH);
  drive(ina, inb, pwmPin, forward);
  delay(RUN_MS);
  stopAll();
  digitalWrite(en, LOW);

  Serial.print("# EVENT ");
  Serial.print(label);
  Serial.println(" OFF");
  delay(GAP_MS);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
  }

  Serial.println("========================================");
  Serial.println("FIRMWARE: back_dir_power");
  Serial.println("PINOUT:   PD_control_back");
  Serial.println("EXPECT_SN:18452630");
  Serial.println("PWM_DUTY: ~50% both FWD and REV");
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
  digitalWrite(M1EN, LOW);
  digitalWrite(M2EN, LOW);

  Serial.println("# starting in 2s — clear of joints");
  delay(2000);

  runPulse("M1", M1EN, M1INA, M1INB, M1PWM, true);
  runPulse("M1", M1EN, M1INA, M1INB, M1PWM, false);
  runPulse("M2", M2EN, M2INA, M2INB, M2PWM, true);
  runPulse("M2", M2EN, M2INA, M2INB, M2PWM, false);

  Serial.println("# done — both motors FWD+REV pulsed");
  Serial.println("# If PD negative deltas fail but REV pulse here also fails → hardware.");
  Serial.println("# If REV pulse moves but PD reverse does not → sign/firmware.");
}

void loop() {
  static uint32_t last = 0;
  if (millis() - last >= 1000) {
    last = millis();
    Serial.println("FIRMWARE: back_dir_power idle EXPECT_SN:18452630");
  }
}
