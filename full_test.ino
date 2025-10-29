// dual_therm.ino
// ESP32 sketch: relay state machine and temperature streaming
//
// Sequence: STOP -> LEFT -> STOP -> RIGHT -> STOP -> LEFT ...
// Frequency (Hz) means full cycles per second (LEFT->RIGHT->LEFT), each cycle has 4 states.
// state_duration_ms = 1000 / (freq * 4)
// Serial commands:
//   freq:X          -> set oscillation frequency in Hz (float). freq <= 0 pauses oscillation.
//   start_stream    -> begin streaming temperatures (lines)
//   stop_stream     -> stop streaming
//   ivan:0/1        -> enable/disable optional third thermistor
//   interval:ms     -> set temperature streaming interval in ms
//
// Streaming format (CSV per line):
//   prox,dist[,ivan]\n

#include <Arduino.h>

// --- Pins ---
#define RELAY_DIR    32  // controls left/right (LOW = LEFT, HIGH = RIGHT)
#define RELAY_MOTION 33  // controls on/off (LOW = STOP, HIGH = MOTION)

#define THERM_PROX   25
#define THERM_DIST   26
#define THERM_IVAN   27  // optional

// --- Globals ---
volatile float oscillationHz = 1.0;
volatile bool streamTemps = false;
volatile bool useIvan = false;
volatile unsigned long tempIntervalMs = 10;

unsigned long lastStateChangeMs = 0;
uint8_t stateIndex = 0; // 0=STOP,1=LEFT,2=STOP,3=RIGHT

// ADC config
const float ADC_MAX = 4095.0; // 12-bit ADC resolution default on ESP32
const float VREF = 3.3;       // reference voltage; approximate

// --- Helper functions ---
float readTempC(int pin) {
  // Basic TMP36-like conversion used previously:
  // voltage -> (voltage - 0.5) * 100 = degC
  // Make analogRead resolution explicit
  int raw = analogRead(pin);
  float voltage = (raw / ADC_MAX) * VREF;
  float tempC = (voltage - 0.5) * 100.0;
  return tempC;
}

void applyState(uint8_t idx) {
  // idx: 0 STOP, 1 LEFT, 2 STOP, 3 RIGHT
  bool motion = false;
  bool dir = LOW; // false = LEFT, true = RIGHT

  switch (idx) {
    case 0: // STOP
      motion = false;
      break;
    case 1: // LEFT
      motion = true;
      dir = LOW;
      break;
    case 2: // STOP
      motion = false;
      break;
    case 3: // RIGHT
      motion = true;
      dir = HIGH;
      break;
    default:
      motion = false;
      break;
  }

  digitalWrite(RELAY_MOTION, motion ? HIGH : LOW);
  digitalWrite(RELAY_DIR, dir ? HIGH : LOW);
}

unsigned long stateIntervalMs() {
  float hz = oscillationHz;
  if (hz <= 0.0) return ULONG_MAX; // effectively disabled
  float interval = 1000.0f / (hz * 4.0f); // 4 states per full cycle
  if (interval < 1.0f) interval = 1.0f;   // cap to avoid zero
  return (unsigned long)(interval);
}

void handleSerialInput(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd.startsWith("freq:")) {
    String s = cmd.substring(5);
    float v = s.toFloat();
    if (isnan(v)) return;
    oscillationHz = v;
    Serial.print("ACK:freq:");
    Serial.println(oscillationHz, 4);
  } else if (cmd == "start_stream") {
    streamTemps = true;
    Serial.println("ACK:start_stream");
  } else if (cmd == "stop_stream") {
    streamTemps = false;
    Serial.println("ACK:stop_stream");
  } else if (cmd.startsWith("ivan:")) {
    String s = cmd.substring(5);
    useIvan = (s.toInt() != 0);
    Serial.print("ACK:ivan:");
    Serial.println(useIvan ? "1" : "0");
  } else if (cmd.startsWith("interval:")) {
    String s = cmd.substring(9);
    unsigned long v = (unsigned long)s.toInt();
    if (v > 0) tempIntervalMs = v;
    Serial.print("ACK:interval:");
    Serial.println(tempIntervalMs);
  } else {
    Serial.print("UNK_CMD:");
    Serial.println(cmd);
  }
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12); // 12-bit ADC
  pinMode(RELAY_DIR, OUTPUT);
  pinMode(RELAY_MOTION, OUTPUT);
  pinMode(THERM_PROX, INPUT);
  pinMode(THERM_DIST, INPUT);
  pinMode(THERM_IVAN, INPUT);

  digitalWrite(RELAY_DIR, LOW);
  digitalWrite(RELAY_MOTION, LOW);

  lastStateChangeMs = millis();
  applyState(stateIndex);

  Serial.println("Ready. Use commands: freq:X, start_stream, stop_stream, ivan:0/1, interval:ms");
}

void loop() {
  // Non-blocking serial command handling
  static String incoming = "";
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (incoming.length() > 0) {
        handleSerialInput(incoming);
        incoming = "";
      }
    } else {
      incoming += c;
      // prevent runaway string
      if (incoming.length() > 200) incoming = incoming.substring(0, 200);
    }
  }

  // Handle relay state machine timing
  unsigned long now = millis();
  unsigned long interval = stateIntervalMs();
  if (interval != ULONG_MAX) {
    if (now - lastStateChangeMs >= interval) {
      lastStateChangeMs = now;
      stateIndex = (stateIndex + 1) % 4; // 0..3
      applyState(stateIndex);
      // Optional: emit state for debugging
      // Serial.print("STATE:");
      // Serial.println(stateIndex);
    }
  }

  // Handle temperature streaming
  static unsigned long lastTempMs = 0;
  if (streamTemps) {
    if (now - lastTempMs >= tempIntervalMs) {
      lastTempMs = now;
      float tP = readTempC(THERM_PROX);
      float tD = readTempC(THERM_DIST);
      // Send with high precision, CSV line
      if (useIvan) {
        float tI = readTempC(THERM_IVAN);
        Serial.print(tP, 3); Serial.print(','); Serial.print(tD, 3); Serial.print(','); Serial.println(tI, 3);
      } else {
        Serial.print(tP, 3); Serial.print(','); Serial.println(tD, 3);
      }
    }
  }

  // Small yield
  delay(0);
}
