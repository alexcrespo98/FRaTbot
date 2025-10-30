// dual_therm_fixed_waveform.ino
// ESP32 sketch: relay state machine and temperature streaming
//
// Sequence: LEFT -> STOP -> RIGHT -> STOP -> repeat
// Frequency (Hz) means full cycles per second (LEFT->RIGHT->LEFT), each cycle has 4 states.
// state_duration_ms = 1000 / (freq * 4)
// waveform: fraction of cycle that is "motion" (default 0.5)
// Serial commands:
//   freq:X          -> set oscillation frequency in Hz (float). freq <= 0 pauses oscillation.
//   waveform:X      -> set relative on/off fraction (0-1)
//   start_stream    -> begin streaming temperatures (lines)
//   stop_stream     -> stop streaming
//   ivan:0/1        -> enable/disable optional third thermistor
//   interval:ms     -> set temperature streaming interval in ms
//
// Streaming format (CSV per line):
//   prox,dist[,ivan]\n

#include <Arduino.h>

// --- Pins ---
#define RELAY_1     32  // Relay 1
#define RELAY_2     33  // Relay 2

#define THERM_PROX  25
#define THERM_DIST  26
#define THERM_IVAN  27  // optional

// --- Globals ---
volatile float oscillationHz = 0.1; // default 0.1 Hz
volatile float waveform = 0.5;      // default 50% motion
volatile bool streamTemps = false;
volatile bool useIvan = false;
volatile unsigned long tempIntervalMs = 10;

unsigned long lastStateChangeMs = 0;
uint8_t stateIndex = 0; // 0=LEFT,1=STOP,2=RIGHT,3=STOP

const float ADC_MAX = 4095.0; // 12-bit ADC
const float VREF = 3.3;

// --- Helper functions ---
float readTempC(int pin) {
  int raw = analogRead(pin);
  float voltage = (raw / ADC_MAX) * VREF;
  float tempC = (voltage - 0.5) * 100.0;
  return tempC;
}

void applyState(uint8_t idx) {
  bool r1 = false;
  bool r2 = false;

  switch (idx) {
    case 0: // LEFT
      r1 = true;
      r2 = false;
      break;
    case 1: // STOP after LEFT
      r1 = false;
      r2 = true;
      break;
    case 2: // RIGHT
      r1 = false;
      r2 = false;
      break;
    case 3: // STOP after RIGHT
      r1 = true;
      r2 = true;
      break;
    default:
      r1 = false;
      r2 = false;
      break;
  }

  digitalWrite(RELAY_1, r1 ? HIGH : LOW);
  digitalWrite(RELAY_2, r2 ? HIGH : LOW);

  Serial.print("STATE ");
  switch(idx) {
    case 0: Serial.println("LEFT"); break;
    case 1: Serial.println("STOP"); break;
    case 2: Serial.println("RIGHT"); break;
    case 3: Serial.println("STOP"); break;
  }
}

unsigned long stateIntervalMs() {
  float hz = oscillationHz;
  if (hz <= 0.0) return ULONG_MAX;
  float fullCycleMs = 1000.0f / hz; // full cycle period
  // Each state duration scaled by waveform fraction
  float motionFraction = waveform;
  float stopFraction = 1.0f - waveform;
  float interval = fullCycleMs * ((stateIndex % 2 == 0) ? motionFraction/2.0f : stopFraction/2.0f);
  if (interval < 1.0f) interval = 1.0f;
  return (unsigned long)interval;
}

void handleSerialInput(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd.startsWith("freq:")) {
    float v = cmd.substring(5).toFloat();
    if (!isnan(v)) oscillationHz = v;
    Serial.print("ACK:freq:"); Serial.println(oscillationHz,4);
  } else if (cmd.startsWith("waveform:")) {
    float v = cmd.substring(9).toFloat();
    if (v >= 0.0 && v <= 1.0) waveform = v;
    Serial.print("ACK:waveform:"); Serial.println(waveform,2);
  } else if (cmd == "start_stream") {
    streamTemps = true; Serial.println("ACK:start_stream");
  } else if (cmd == "stop_stream") {
    streamTemps = false; Serial.println("ACK:stop_stream");
  } else if (cmd.startsWith("ivan:")) {
    useIvan = (cmd.substring(5).toInt() != 0);
    Serial.print("ACK:ivan:"); Serial.println(useIvan ? "1" : "0");
  } else if (cmd.startsWith("interval:")) {
    unsigned long v = (unsigned long)cmd.substring(9).toInt();
    if (v>0) tempIntervalMs=v;
    Serial.print("ACK:interval:"); Serial.println(tempIntervalMs);
  } else {
    Serial.print("UNK_CMD:"); Serial.println(cmd);
  }
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);

  pinMode(RELAY_1, OUTPUT);
  pinMode(RELAY_2, OUTPUT);
  pinMode(THERM_PROX, INPUT);
  pinMode(THERM_DIST, INPUT);
  pinMode(THERM_IVAN, INPUT);

  digitalWrite(RELAY_1, LOW);
  digitalWrite(RELAY_2, LOW);

  lastStateChangeMs = millis();
  applyState(stateIndex);

  Serial.println("Ready. Commands: freq:X, waveform:X, start_stream, stop_stream, ivan:0/1, interval:ms");
}

void loop() {
  static String incoming = "";
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c=='\n'||c=='\r') {
      if (incoming.length()>0) handleSerialInput(incoming);
      incoming = "";
    } else {
      incoming += c;
      if (incoming.length()>200) incoming = incoming.substring(0,200);
    }
  }

  unsigned long now = millis();
  unsigned long interval = stateIntervalMs();
  if (interval != ULONG_MAX) {
    if (now - lastStateChangeMs >= interval) {
      lastStateChangeMs = now;
      stateIndex = (stateIndex + 1) % 4;
      applyState(stateIndex);
    }
  }

  static unsigned long lastTempMs = 0;
  if (streamTemps && now - lastTempMs >= tempIntervalMs) {
    lastTempMs = now;
    float tP = readTempC(THERM_PROX);
    float tD = readTempC(THERM_DIST);
    if (useIvan) {
      float tI = readTempC(THERM_IVAN);
      Serial.print(tP,3); Serial.print(','); Serial.print(tD,3); Serial.print(','); Serial.println(tI,3);
    } else {
      Serial.print(tP,3); Serial.print(','); Serial.println(tD,3);
    }
  }

  delay(0);
}
