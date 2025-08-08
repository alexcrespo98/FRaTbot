/*
 * ESP32 BLE Thermistor Distal Sensor (Peripheral/Server)
 * 
 * This sketch implements a battery-powered BLE server that:
 * - Reads a thermistor connected between ADC pin and GND with 4.7kΩ pull-up to 3.3V
 * - Sends thermistor voltage readings via BLE notifications
 * - Acts as a BLE peripheral for the proximal sensor to connect to
 * 
 * Hardware Setup:
 * - 4.7kΩ thermistor: one end to ADC pin (GPIO36), other end to GND
 * - 4.7kΩ pull-up resistor: from ADC pin to 3.3V
 * 
 * BLE Service UUID: 12345678-1234-1234-1234-123456789abc
 * BLE Characteristic UUID: 87654321-4321-4321-4321-cba987654321
 */

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// BLE UUIDs - must match proximal sketch
#define SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "87654321-4321-4321-4321-cba987654321"

// Thermistor configuration
#define THERMISTOR_PIN 36  // ADC1_CH0 (GPIO36)
#define READING_INTERVAL 1000  // Send reading every 1000ms

// BLE objects
BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

// Server callback class
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("Device connected");
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("Device disconnected");
    }
};

void setup() {
  Serial.begin(115200);
  Serial.println("Starting ESP32 BLE Distal Thermistor Sensor...");

  // Initialize BLE
  BLEDevice::init("DistalThermistor");
  
  // Create BLE Server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create BLE Service
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Create BLE Characteristic
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );

  // Add descriptor for notifications
  pCharacteristic->addDescriptor(new BLE2902());

  // Start the service
  pService->start();

  // Start advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x0);  // Use 0x0 for optimal compatibility
  BLEDevice::startAdvertising();
  
  Serial.println("BLE advertising started. Waiting for connections...");
  Serial.println("Characteristic UUID: " + String(CHARACTERISTIC_UUID));
}

void loop() {
  // Handle connection state changes
  if (!deviceConnected && oldDeviceConnected) {
    delay(500); // Give the bluetooth stack time to get ready
    pServer->startAdvertising(); // Restart advertising
    Serial.println("Restarting advertising");
    oldDeviceConnected = deviceConnected;
  }
  
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }

  // Read thermistor and send data if connected
  if (deviceConnected) {
    float voltage = readThermistorVoltage();
    
    // Convert voltage to string for BLE transmission
    String voltageStr = String(voltage, 3);
    
    // Send notification
    pCharacteristic->setValue(voltageStr.c_str());
    pCharacteristic->notify();
    
    Serial.println("Sent voltage: " + voltageStr + "V");
  }
  
  delay(READING_INTERVAL);
}

/**
 * Read thermistor voltage (placeholder implementation)
 * 
 * Hardware setup:
 * - 4.7kΩ thermistor between ADC pin and GND
 * - 4.7kΩ pull-up resistor from ADC pin to 3.3V
 * 
 * This creates a voltage divider where:
 * V_out = 3.3V * (R_thermistor / (R_pullup + R_thermistor))
 * 
 * @return Voltage reading in volts
 */
float readThermistorVoltage() {
  // TODO: Implement actual thermistor reading
  // For now, return a simulated voltage that varies slightly
  static float baseVoltage = 1.65; // Mid-range voltage for testing
  float noise = (random(-50, 50) / 1000.0); // ±0.05V noise
  
  // Simulate temperature variation (could be replaced with actual ADC reading)
  float simulatedVoltage = baseVoltage + noise;
  
  // Actual implementation would be:
  // int adcReading = analogRead(THERMISTOR_PIN);
  // float voltage = (adcReading / 4095.0) * 3.3;
  // return voltage;
  
  return simulatedVoltage;
}