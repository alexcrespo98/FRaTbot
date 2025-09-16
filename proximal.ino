/*
 * ESP32-C6 BLE Messaging - Proximal (Client/Central)
 * Uses DS18B20 OneWire thermometer for local temperature
 * Requests remote temperature data
 * Serial Plotter friendly output
 * Sends temps out TX (D6/GPIO16) to Teensy via Serial1
 * --- Added: Independent relay toggling every 1 second ---
 */

#include <BLEDevice.h>
#include <BLEClient.h>
#include <BLEUtils.h>
#include <BLEScan.h>

#include <OneWire.h>
#include <DallasTemperature.h>

#define SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "87654321-4321-4321-4321-cba987654321"
#define ONE_WIRE_BUS        D0        // DS18B20 data pin
#define RELAY_PIN           D1        // Relay IN pin
#define TX_PIN              16        // D6 on XIAO ESP32C6

BLEClient* pClient;
BLERemoteCharacteristic* pRemoteCharacteristic;
BLEAdvertisedDevice* myDevice = nullptr;
bool doConnect = false;
bool connected = false;

// --- DallasTemperature Setup ---
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

float localTemp = 0.0;
float remoteTemp = 0.0;

// --- Relay control variables ---
unsigned long lastRelayToggle = 0;
bool relayState = false;
const unsigned long relayInterval = 1000; // 1 second toggle

class MyAdvertisedDeviceCallbacks : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice advertisedDevice) override {
    if (advertisedDevice.haveServiceUUID() &&
        advertisedDevice.isAdvertisingService(BLEUUID(SERVICE_UUID))) {
      BLEDevice::getScan()->stop();
      myDevice = new BLEAdvertisedDevice(advertisedDevice);
      doConnect = true;
      Serial.println("[CLIENT] Found server, attempting to connect...");
    }
  }
};

bool connectToServer() {
  pClient = BLEDevice::createClient();
  if (!pClient->connect(myDevice)) {
    Serial.println("[CLIENT] Failed to connect.");
    return false;
  }
  Serial.println("[CLIENT] Connected to server.");

  BLERemoteService* pService = pClient->getService(SERVICE_UUID);
  if (!pService) {
    Serial.println("[CLIENT] Service not found.");
    pClient->disconnect();
    return false;
  }

  pRemoteCharacteristic = pService->getCharacteristic(CHARACTERISTIC_UUID);
  if (!pRemoteCharacteristic) {
    Serial.println("[CLIENT] Characteristic not found.");
    pClient->disconnect();
    return false;
  }

  connected = true;
  return true;
}

// --- DS18B20 read ---
float readDS18B20() {
  sensors.requestTemperatures();
  float tempC = sensors.getTempCByIndex(0);
  if (tempC == DEVICE_DISCONNECTED_C) {
    Serial.println("[CLIENT] Sensor error");
    return NAN;
  }
  return tempC;
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200, SERIAL_8N1, -1, TX_PIN); // Only TX used for Teensy

  sensors.begin(); // start DS18B20

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW); // start relay off

  BLEDevice::init("ProximalBLE");
  BLEScan* scan = BLEDevice::getScan();
  scan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
  scan->setActiveScan(true);
  scan->start(0, false);

  Serial.println("[CLIENT] Scanning for server...");
}

void loop() {
  // --- Independent Relay Toggling ---
  unsigned long now = millis();
  if (now - lastRelayToggle >= relayInterval) {
    relayState = !relayState;
    digitalWrite(RELAY_PIN, relayState ? HIGH : LOW);
    lastRelayToggle = now;
  }

  // --- BLE connection handling ---
  if (doConnect && !connected) {
    if (connectToServer()) {
      Serial.println("[CLIENT] Connection established.");
    }
    doConnect = false;
  }

  // --- Local temperature read ---
  localTemp = readDS18B20();

  // --- Remote temperature request ---
  if (connected) {
    String request = "TEMP_REQUEST";
    uint8_t buf[request.length()];
    memcpy(buf, request.c_str(), request.length());
    pRemoteCharacteristic->writeValue(buf, request.length(), false);

    delay(10); // allow response

    String response = pRemoteCharacteristic->readValue().c_str();
    if (response.length() > 0 && response != "TEMP_REQUEST") {
      remoteTemp = response.toFloat();
    }

    // Serial Plotter output
    Serial.print(localTemp, 2);
    Serial.print("\t");
    Serial.println(remoteTemp, 2);

    // TX to Teensy
    Serial1.print(localTemp, 2);
    Serial1.print(",");
    Serial1.println(remoteTemp, 2);
  }

  // --- Reconnect if disconnected ---
  if (connected && pClient && !pClient->isConnected()) {
    Serial.println("[CLIENT] Lost connection. Restarting scan...");
    connected = false;
    doConnect = false;
    delete myDevice;
    myDevice = nullptr;
    BLEDevice::getScan()->start(0, false);
  }

  delay(50); // 20 Hz update rate
}
