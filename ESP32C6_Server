/*
 * ESP32C6 BLE Messaging - Distal (Server/Peripheral)
 * Responds to temperature requests with local thermistor reading
 */

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "87654321-4321-4321-4321-cba987654321"
#define THERMISTOR_PIN      D0

BLEServer* pServer;
BLECharacteristic* pCharacteristic;
bool deviceConnected = false;

// Thermistor calculations
const float BETA = 3950.0;
const float R0 = 10000.0;
const float T0 = 298.15;

class MyServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("Client connected!");
    }
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("Client disconnected!");
      BLEDevice::getAdvertising()->start();
    }
};

float readThermistor() {
  int adcValue = analogRead(THERMISTOR_PIN);
  float voltage = (adcValue / 4095.0) * 3.3;
  float resistance = (3.3 - voltage) * R0 / voltage;
  float tempK = 1.0 / (1.0/T0 + (1.0/BETA) * log(resistance/R0));
  return tempK - 273.15; // Convert to Celsius
}

class MyCallbacks : public BLECharacteristicCallbacks {
  private:
    String lastMessage = "";
    unsigned long lastMessageTimestamp = 0;
    
  public:
    void onWrite(BLECharacteristic* pCharacteristic) override {
      uint8_t* pData = pCharacteristic->getData();
      int length = pCharacteristic->getValue().length();
      if (length > 0) {
        String rxValue = "";
        for (int i = 0; i < length; i++) {
          rxValue += (char)pData[i];
        }
        
        // Only filter if exact same message AND timestamp are within 5ms (true duplicate)
        unsigned long currentTimestamp = millis();
        if (rxValue != lastMessage || (currentTimestamp - lastMessageTimestamp) > 5) {
          
          // Check if this is a temperature request
          if (rxValue == "TEMP_REQUEST") {
            float temp = readThermistor();
            String tempStr = String(temp, 2);
            pCharacteristic->setValue(tempStr.c_str());
            Serial.print("Sent temperature: ");
            Serial.println(temp, 2);
          } else {
            Serial.print("Received from client: ");
            Serial.println(rxValue);
          }
          
          lastMessage = rxValue;
          lastMessageTimestamp = currentTimestamp;
        }
      }
    }
};

void setup() {
  Serial.begin(115200);
  pinMode(THERMISTOR_PIN, INPUT);
  
  BLEDevice::init("DistalBLE");

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
      CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_READ
  );

  pCharacteristic->setCallbacks(new MyCallbacks());

  pService->start();

  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->start();

  Serial.println("Server started, advertising...");
}

void loop() {
  delay(10);
}
