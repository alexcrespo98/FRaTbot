/*
 * ESP32 BLE Thermistor Proximal Sensor (Central/Client)
 * 
 * This sketch implements a computer-attached BLE client that:
 * - Reads a local thermistor connected between ADC pin and GND with 4.7kΩ pull-up to 3.3V
 * - Connects to the distal BLE thermistor sensor
 * - Receives remote thermistor readings via BLE notifications
 * - Computes and logs delta-t (proximal - distal) measurements
 * 
 * Hardware Setup:
 * - 4.7kΩ thermistor: one end to ADC pin (GPIO39), other end to GND
 * - 4.7kΩ pull-up resistor: from ADC pin to 3.3V
 * 
 * Future enhancements (TODO):
 * - OLED display for real-time data visualization
 * - Rotary encoder for distance input configuration
 * 
 * BLE Service UUID: 12345678-1234-1234-1234-123456789abc
 * BLE Characteristic UUID: 87654321-4321-4321-4321-cba987654321
 */

#include <BLEDevice.h>
#include <BLEClient.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// BLE UUIDs - must match distal sketch
#define SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "87654321-4321-4321-4321-cba987654321"

// Thermistor configuration
#define THERMISTOR_PIN 39  // ADC1_CH3 (GPIO39)
#define READING_INTERVAL 1000  // Read local thermistor every 1000ms

// BLE objects
BLEClient* pClient = NULL;
BLERemoteCharacteristic* pRemoteCharacteristic = NULL;
BLEAdvertisedDevice* myDevice = NULL;

// Connection state
bool doConnect = false;
bool connected = false;
bool doScan = false;

// Data storage
float proximalVoltage = 0.0;
float distalVoltage = 0.0;
unsigned long lastReading = 0;

// BLE scan callback class
class MyAdvertisedDeviceCallbacks: public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice advertisedDevice) {
    // Check if this is our distal thermistor device
    if (advertisedDevice.haveServiceUUID() && advertisedDevice.isAdvertisingService(BLEUUID(SERVICE_UUID))) {
      BLEDevice::getScan()->stop();
      myDevice = new BLEAdvertisedDevice(advertisedDevice);
      doConnect = true;
      doScan = true;
      Serial.println("Found distal thermistor device!");
    }
  }
};

// BLE client callback class
class MyClientCallback : public BLEClientCallbacks {
  void onConnect(BLEClient* pclient) {
    Serial.println("Connected to distal thermistor");
  }

  void onDisconnect(BLEClient* pclient) {
    connected = false;
    Serial.println("Disconnected from distal thermistor");
  }
};

// Notification callback for receiving distal thermistor data
static void notifyCallback(BLERemoteCharacteristic* pBLERemoteCharacteristic, 
                          uint8_t* pData, 
                          size_t length, 
                          bool isNotify) {
  String receivedData = "";
  for (int i = 0; i < length; i++) {
    receivedData += (char)pData[i];
  }
  
  distalVoltage = receivedData.toFloat();
  Serial.println("Received distal voltage: " + String(distalVoltage, 3) + "V");
  
  // Calculate and display delta-t
  calculateAndDisplayDeltaT();
}

bool connectToServer() {
  Serial.println("Forming a connection to distal thermistor...");
  
  pClient = BLEDevice::createClient();
  pClient->setClientCallbacks(new MyClientCallback());

  // Connect to the remote BLE Server
  pClient->connect(myDevice);
  Serial.println("Connected to server");

  // Obtain a reference to the service
  BLERemoteService* pRemoteService = pClient->getService(SERVICE_UUID);
  if (pRemoteService == nullptr) {
    Serial.println("Failed to find our service UUID");
    pClient->disconnect();
    return false;
  }
  Serial.println("Found our service");

  // Obtain a reference to the characteristic
  pRemoteCharacteristic = pRemoteService->getCharacteristic(CHARACTERISTIC_UUID);
  if (pRemoteCharacteristic == nullptr) {
    Serial.println("Failed to find our characteristic UUID");
    pClient->disconnect();
    return false;
  }
  Serial.println("Found our characteristic");

  // Register for notifications
  if(pRemoteCharacteristic->canNotify()) {
    pRemoteCharacteristic->registerForNotify(notifyCallback);
    Serial.println("Registered for notifications");
  }

  connected = true;
  return true;
}

void setup() {
  Serial.begin(115200);
  Serial.println("Starting ESP32 BLE Proximal Thermistor Sensor...");

  // Initialize BLE
  BLEDevice::init("ProximalThermistor");

  // Create BLE Scanner
  BLEScan* pBLEScan = BLEDevice::getScan();
  pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
  pBLEScan->setInterval(1349);
  pBLEScan->setWindow(449);
  pBLEScan->setActiveScan(true);
  pBLEScan->start(5, false);

  Serial.println("Scanning for distal thermistor device...");
  Serial.println("Expected service UUID: " + String(SERVICE_UUID));
  
  // TODO: Initialize OLED display
  // TODO: Initialize rotary encoder for distance input
}

void loop() {
  // Handle BLE connection
  if (doConnect == true) {
    if (connectToServer()) {
      Serial.println("Successfully connected to distal thermistor server");
    } else {
      Serial.println("Failed to connect to distal thermistor server");
    }
    doConnect = false;
  }

  // Handle disconnection and reconnection
  if (!connected && doScan) {
    BLEDevice::getScan()->start(0);  // Restart scanning
  }

  // Read local thermistor at regular intervals
  if (millis() - lastReading > READING_INTERVAL) {
    proximalVoltage = readThermistorVoltage();
    Serial.println("Proximal voltage: " + String(proximalVoltage, 3) + "V");
    
    // Calculate delta-t if we have both readings
    if (connected && distalVoltage > 0) {
      calculateAndDisplayDeltaT();
    }
    
    lastReading = millis();
  }

  // TODO: Update OLED display with current readings and delta-t
  // TODO: Check rotary encoder for distance input changes
  
  delay(100); // Small delay to prevent excessive polling
}

/**
 * Read local thermistor voltage (placeholder implementation)
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
  static float baseVoltage = 1.70; // Slightly different from distal for testing
  float noise = (random(-30, 30) / 1000.0); // ±0.03V noise
  
  // Simulate temperature variation (could be replaced with actual ADC reading)
  float simulatedVoltage = baseVoltage + noise;
  
  // Actual implementation would be:
  // int adcReading = analogRead(THERMISTOR_PIN);
  // float voltage = (adcReading / 4095.0) * 3.3;
  // return voltage;
  
  return simulatedVoltage;
}

/**
 * Calculate and display delta-t measurement
 * 
 * Delta-t = Proximal temperature - Distal temperature
 * For voltage readings, higher voltage typically indicates lower temperature
 * for a negative temperature coefficient (NTC) thermistor
 */
void calculateAndDisplayDeltaT() {
  if (proximalVoltage > 0 && distalVoltage > 0) {
    float deltaVoltage = proximalVoltage - distalVoltage;
    
    // TODO: Convert voltage readings to actual temperatures using thermistor equation
    // For now, display voltage difference as proxy for temperature difference
    Serial.println("=== DELTA MEASUREMENT ===");
    Serial.println("Proximal: " + String(proximalVoltage, 3) + "V");
    Serial.println("Distal:   " + String(distalVoltage, 3) + "V");
    Serial.println("Delta-V:  " + String(deltaVoltage, 3) + "V");
    Serial.println("========================");
    
    // TODO: Convert to actual temperature delta-t
    // Using Steinhart-Hart equation or thermistor lookup table
    // float proximalTemp = voltageToTemperature(proximalVoltage);
    // float distalTemp = voltageToTemperature(distalVoltage);
    // float deltaT = proximalTemp - distalTemp;
    // Serial.println("Delta-T:  " + String(deltaT, 2) + "°C");
    
    // TODO: Log data to SD card or send to computer via Serial/WiFi
    // TODO: Update OLED display with delta-t value
  }
}

// TODO: Future function to convert voltage to temperature
/*
float voltageToTemperature(float voltage) {
  // Calculate resistance from voltage divider
  float resistance = (4700.0 * voltage) / (3.3 - voltage);
  
  // Use Steinhart-Hart equation or lookup table
  // This is a placeholder - actual coefficients depend on specific thermistor
  float tempK = 1.0 / (0.001129148 + (0.000234125 * log(resistance)) + 
                      (0.0000000876741 * log(resistance) * log(resistance) * log(resistance)));
  float tempC = tempK - 273.15;
  
  return tempC;
}
*/

// TODO: Future functions for OLED display
/*
void initializeOLED() {
  // Initialize OLED display (e.g., SSD1306)
}

void updateOLEDDisplay() {
  // Update display with:
  // - Current proximal and distal readings
  // - Delta-t value
  // - Connection status
  // - Distance setting from rotary encoder
}
*/

// TODO: Future functions for rotary encoder
/*
void initializeRotaryEncoder() {
  // Initialize rotary encoder pins and interrupts
}

void handleRotaryEncoder() {
  // Read encoder rotation and button press
  // Update distance setting for delta-t calculations
}
*/